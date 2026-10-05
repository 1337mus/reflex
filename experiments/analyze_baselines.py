"""Validate and analyze the saved development baseline comparison artifact."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import cast

from reflex_decisions.baseline_data import verify_prepared_data
from reflex_decisions.data import DecisionRecord, SplitManifest
from reflex_decisions.evaluation import PredictionRecord, evaluate
from reflex_decisions.permutation import compare_permutations
from reflex_decisions.scoring import score_candidates
from reflex_decisions.smoke import reserve_output, write_json_artifact

MODELS = ("qwen", "intern", "kev")
MODEL_PINS = {
    "qwen": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    "intern": ("internlm/Intern-Decision-0.8B", "85a0cc5a99d67ea8d56dfe98115689212867171d"),
    "kev": ("jaredpalmer/kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e"),
}
CALIBRATION_ID = "model-native-temp"
MAX_INPUT_TOKENS = 2048
_HEX = set("0123456789abcdef")


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _wilson_interval(correct: int, count: int) -> dict[str, float | int]:
    if count < 1 or not 0 <= correct <= count:
        raise ValueError("Wilson interval counts are invalid")
    z = 1.959963984540054
    z2 = z * z
    observed = correct / count
    denominator = 1.0 + z2 / count
    center = (observed + z2 / (2.0 * count)) / denominator
    margin = z * math.sqrt(observed * (1.0 - observed) / count + z2 / (4.0 * count * count))
    margin /= denominator
    return {
        "correct": correct,
        "record_count": count,
        "lower_95": max(0.0, center - margin),
        "upper_95": min(1.0, center + margin),
    }


def analyze_model(
    model_name: str,
    receipt: Mapping[str, object],
    records: Sequence[DecisionRecord],
    manifest: SplitManifest,
    *,
    records_sha256: str,
    manifest_sha256: str,
) -> dict[str, object]:
    """Analyze a complete model receipt; kept testable on small synthetic fixtures."""

    if receipt.get("status") != "passed":
        raise ValueError(f"{model_name} model receipt did not pass")
    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{model_name} receipt has no provenance object")
    model_id, revision = provenance.get("model_id"), provenance.get("model_revision")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"{model_name} provenance has no model ID")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"{model_name} provenance has no model revision")
    temperature = _finite_number(provenance.get("temperature"), "model temperature")
    if temperature <= 0:
        raise ValueError("model temperature must be positive")
    raw_rows = receipt.get("presentations")
    if not isinstance(raw_rows, list) or len(raw_rows) != 2 * len(records):
        raise ValueError("each model must have exactly two presentations per source record")
    by_id = {record.record_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("analysis records contain duplicate record IDs")
    pairs: dict[str, dict[int, list[float]]] = {record_id: {} for record_id in by_id}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("presentation rows must be JSON objects")
        record_id, index = row.get("record_id"), row.get("permutation_index")
        if not isinstance(record_id, str) or record_id not in by_id:
            raise ValueError("presentation references an unexpected record ID")
        if isinstance(index, bool) or not isinstance(index, int) or index not in (0, 1):
            raise ValueError("presentation permutation_index must be 0 or 1")
        if index in pairs[record_id]:
            raise ValueError("duplicate presentation for record and permutation")
        record = by_id[record_id]
        option_ids = [option.id for option in record.request.options]
        if index:
            option_ids.reverse()
        if row.get("request_hash") != record.request.request_hash:
            raise ValueError("presentation request_hash does not match its source record")
        if row.get("option_ids") != option_ids:
            raise ValueError("presentation option IDs do not match the approved permutation")
        raw_logits = row.get("raw_logits")
        if not isinstance(raw_logits, list) or len(raw_logits) != len(option_ids):
            raise ValueError("presentation logits do not match its option count")
        logits = [_finite_number(value, "raw logit") for value in raw_logits]
        tokens, prompt = row.get("input_tokens"), row.get("prompt_sha256")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or not 1 <= tokens <= MAX_INPUT_TOKENS
        ):
            raise ValueError("presentation input token count is outside the approved limit")
        if not isinstance(prompt, str) or len(prompt) != 64 or any(c not in _HEX for c in prompt):
            raise ValueError("presentation prompt hash must be a lowercase SHA-256 digest")
        pairs[record_id][index] = logits
    if any(set(pair) != {0, 1} for pair in pairs.values()):
        raise ValueError("every source record must have one original and one reversed presentation")
    predictions = tuple(
        PredictionRecord(
            record_id=record.record_id,
            request_hash=record.request.request_hash,
            logits=dict(
                zip((o.id for o in record.request.options), pairs[record.record_id][0], strict=True)
            ),
            model_revision=revision,
        )
        for record in records
    )
    prediction_sha256 = _digest(
        json.dumps(
            [p.model_dump(mode="json") for p in predictions],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    raw_report = evaluate(
        records,
        predictions,
        manifest,
        records_sha256=records_sha256,
        predictions_sha256=prediction_sha256,
        manifest_sha256=manifest_sha256,
        temperature=1.0,
        calibration_id=None,
    )
    shipped_report = evaluate(
        records,
        predictions,
        manifest,
        records_sha256=records_sha256,
        predictions_sha256=prediction_sha256,
        manifest_sha256=manifest_sha256,
        temperature=temperature,
        calibration_id=None if model_name == "qwen" else CALIBRATION_ID,
    )
    correct_by_record, divergences = [], []
    original_correct = reversed_correct = flips = 0
    for record in records:
        original_logits, reversed_logits = pairs[record.record_id][0], pairs[record.record_id][1]
        original = score_candidates(
            record.request,
            original_logits,
            model_revision=revision,
            temperature=temperature,
            calibration_id=CALIBRATION_ID,
        )
        reversed_request = record.request.model_copy(
            update={"options": tuple(reversed(record.request.options))}
        )
        reversed_result = score_candidates(
            reversed_request,
            reversed_logits,
            model_revision=revision,
            temperature=temperature,
            calibration_id=CALIBRATION_ID,
        )
        is_original_correct = original.top_option_id == record.answer_id
        correct_by_record.append(is_original_correct)
        original_correct += is_original_correct
        reversed_correct += reversed_result.top_option_id == record.answer_id
        comparison = compare_permutations(original, reversed_result)
        divergences.append(comparison.jensen_shannon_divergence)
        flips += comparison.forced_choice_flipped
    mean_js = math.fsum(divergences) / len(divergences)
    return {
        "model_id": model_id,
        "model_revision": revision,
        "temperature": temperature,
        "provenance": provenance,
        "prediction_sha256": prediction_sha256,
        "raw_report": raw_report.model_dump(mode="json"),
        "shipped_report": shipped_report.model_dump(mode="json"),
        "accuracy_interval": _wilson_interval(original_correct, len(records)),
        "order_sensitivity": {
            "pair_count": len(records),
            "forced_choice_flip_count": flips,
            "forced_choice_flip_fraction": flips / len(records),
            "mean_js_divergence_nats": mean_js,
            "original_order_accuracy": original_correct / len(records),
            "reversed_order_accuracy": reversed_correct / len(records),
        },
        "_correct": correct_by_record,
    }


def analyze_run(
    run_path: str | Path, records_path: str | Path, manifest_path: str | Path
) -> dict[str, object]:
    """Require the exact complete pilot, then analyze its three frozen model receipts."""

    run_bytes = Path(run_path).read_bytes()
    try:
        run = json.loads(run_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("runner receipt is not valid UTF-8 JSON") from exc
    if not isinstance(run, dict) or run.get("status") != "passed":
        raise ValueError("runner receipt must have passed status")
    records_sha256 = _digest(Path(records_path).read_bytes())
    manifest_sha256 = _digest(Path(manifest_path).read_bytes())
    if run.get("records_sha256") != records_sha256:
        raise ValueError("runner receipt records SHA-256 does not match the prepared data")
    if run.get("manifest_sha256") != manifest_sha256:
        raise ValueError("runner receipt manifest SHA-256 does not match the prepared data")
    manifest, records = verify_prepared_data(records_path, manifest_path)
    if (
        _digest(Path(records_path).read_bytes()) != records_sha256
        or _digest(Path(manifest_path).read_bytes()) != manifest_sha256
    ):
        raise ValueError("prepared data changed while the analysis was loading it")
    receipts = run.get("models")
    if not isinstance(receipts, dict) or set(receipts) != set(MODELS):
        raise ValueError("runner receipt must contain exactly qwen, intern, and kev")
    per_model, correct_by_model = {}, {}
    for name in MODELS:
        receipt = receipts[name]
        if not isinstance(receipt, dict):
            raise ValueError(f"{name} model receipt must be an object")
        provenance = receipt.get("provenance")
        if (
            not isinstance(provenance, dict)
            or (provenance.get("model_id"), provenance.get("model_revision")) != MODEL_PINS[name]
        ):
            raise ValueError(f"{name} model provenance does not match the frozen pin")
        result = analyze_model(
            name,
            receipt,
            records,
            manifest,
            records_sha256=records_sha256,
            manifest_sha256=manifest_sha256,
        )
        correct_by_model[name] = cast(list[bool], result.pop("_correct"))
        per_model[name] = result
    paired = []
    for left, right in itertools.combinations(MODELS, 2):
        left_correct, right_correct = correct_by_model[left], correct_by_model[right]
        both = sum(a and b for a, b in zip(left_correct, right_correct, strict=True))
        left_only = sum(a and not b for a, b in zip(left_correct, right_correct, strict=True))
        right_only = sum(not a and b for a, b in zip(left_correct, right_correct, strict=True))
        paired.append(
            {
                "left_model": left,
                "right_model": right,
                "record_count": len(records),
                "both_correct": both,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "both_wrong": len(records) - both - left_only - right_only,
                "accuracy_difference_left_minus_right": (left_only - right_only) / len(records),
            }
        )
    return {
        "status": "passed",
        "dataset_id": manifest.datasets[0].dataset_id,
        "run_sha256": _digest(run_bytes),
        "records_sha256": records_sha256,
        "manifest_sha256": manifest_sha256,
        "record_count": len(records),
        "per_model": per_model,
        "paired_comparisons": paired,
        "interpretation_limit": (
            "This is a 32-item development pilot, not an unseen-family or broad-generalization "
            "claim."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        with reserve_output(args.output) as reservation:
            result = analyze_run(args.run, args.records, args.manifest)
            write_json_artifact(reservation, result)
    except (OSError, TypeError, ValueError) as exc:
        print(f"analysis failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "passed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
