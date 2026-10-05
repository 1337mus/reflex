"""Validate and analyze a complete BoolQ/SNLI development comparison receipt."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import cast

from experiments.baseline_runner_core import EXPECTED_AUXILIARY_FORWARD_COUNTS
from experiments.broader_runner_core import (
    DATASET_IDS,
    MODELS,
    RECORDS_PER_DATASET,
    presentations,
    serialize_remote_records,
    validate_presentation_result,
)
from experiments.modal_broader import (
    EXPECTED_PROTOCOL,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_SOURCE_AMENDMENT,
    EXPECTED_SOURCE_AMENDMENT_SHA256,
    _plan,
    _source_fingerprints,
)
from reflex_decisions import broader_data
from reflex_decisions.data import DecisionRecord, SplitManifest, audit_splits
from reflex_decisions.evaluation import PredictionRecord, evaluate
from reflex_decisions.schema import DecisionRequest
from reflex_decisions.scoring import DecisionResult, score_candidates
from reflex_decisions.smoke import reserve_output, write_json_artifact

MODEL_PINS = {
    "qwen": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    "intern": ("internlm/Intern-Decision-0.8B", "85a0cc5a99d67ea8d56dfe98115689212867171d"),
    "kev": ("jaredpalmer/kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e"),
}
CALIBRATION_ID = "model-native-temp"
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


def _validate_record_set(records: Sequence[DecisionRecord], manifest: SplitManifest) -> None:
    if manifest.data_kind != "benchmark" or len(manifest.datasets) != len(DATASET_IDS):
        raise ValueError("prepared broader manifest must contain the two benchmark datasets")
    datasets = {dataset.dataset_id: dataset for dataset in manifest.datasets}
    if set(datasets) != set(DATASET_IDS) or any(
        dataset.split != "development" for dataset in datasets.values()
    ):
        raise ValueError("prepared broader manifest must contain only development data")
    if len(records) != 2 * RECORDS_PER_DATASET:
        raise ValueError("prepared broader data must contain exactly 64 records")
    counts = {dataset_id: 0 for dataset_id in DATASET_IDS}
    groups = {dataset_id: set() for dataset_id in DATASET_IDS}
    ids: set[str] = set()
    for record in records:
        if record.record_id in ids:
            raise ValueError("prepared broader data contains duplicate record IDs")
        ids.add(record.record_id)
        if record.dataset_id not in counts or not record.record_id.startswith(
            f"{record.dataset_id}-"
        ):
            raise ValueError("prepared broader record ID and dataset do not match")
        counts[record.dataset_id] += 1
        groups[record.dataset_id].add(record.source_group_id)
        expected = (
            ("yes", "no")
            if record.dataset_id == DATASET_IDS[0]
            else (
                "entailment",
                "neutral",
                "contradiction",
            )
        )
        if tuple(option.id for option in record.request.options) != expected:
            raise ValueError("prepared broader record has the wrong original option order")
    if any(count != RECORDS_PER_DATASET for count in counts.values()) or any(
        len(group_ids) != RECORDS_PER_DATASET for group_ids in groups.values()
    ):
        raise ValueError("prepared broader data must contain 32 distinct groups per dataset")
    audit_splits(manifest, tuple(records))


def _validate_analysis_records(records: Sequence[DecisionRecord], manifest: SplitManifest) -> None:
    if not records:
        raise ValueError("analysis records must not be empty")
    dataset_specs = {dataset.dataset_id: dataset for dataset in manifest.datasets}
    if not set(dataset_specs).issubset(DATASET_IDS):
        raise ValueError("analysis manifest contains an unexpected dataset")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("analysis records contain duplicate record IDs")
    for record in records:
        if record.dataset_id not in dataset_specs:
            raise ValueError("analysis record references an unexpected dataset")
        expected = (
            ("yes", "no")
            if record.dataset_id == DATASET_IDS[0]
            else (
                "entailment",
                "neutral",
                "contradiction",
            )
        )
        if tuple(option.id for option in record.request.options) != expected:
            raise ValueError("analysis record has the wrong original option order")
    audit_splits(manifest, tuple(records))


def _report_metrics(report: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw_rows = report.get("per_dataset")
    if not isinstance(raw_rows, list):
        raise ValueError("evaluation report has no per-dataset metrics")
    metrics: dict[str, dict[str, object]] = {}
    for row in raw_rows:
        if not isinstance(row, dict) or not isinstance(row.get("dataset_id"), str):
            raise ValueError("evaluation report contains invalid dataset metrics")
        metrics[row["dataset_id"]] = row
    return metrics


def analyze_model(
    model_name: str,
    receipt: Mapping[str, object],
    records: Sequence[DecisionRecord],
    manifest: SplitManifest,
    *,
    records_sha256: str,
    manifest_sha256: str,
) -> dict[str, object]:
    """Analyze original-order metrics and the repeated permutation measurements."""

    _validate_analysis_records(records, manifest)
    if model_name not in MODELS or receipt.get("status") != "passed":
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

    expected_rows = presentations(serialize_remote_records(tuple(records)))
    expected_by_key = {(row["record_id"], row["permutation_index"]): row for row in expected_rows}
    raw_rows = receipt.get("presentations")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(expected_rows):
        raise ValueError("each model must have the exact complete approved presentation set")
    normalized_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            raise ValueError("presentation rows must be JSON objects")
        if set(row) != {
            "record_id",
            "request_hash",
            "permutation_index",
            "option_ids",
            "raw_logits",
            "input_tokens",
            "prompt_sha256",
        }:
            raise ValueError("presentation row does not match the raw-logit receipt contract")
        record_id, index = row.get("record_id"), row.get("permutation_index")
        if not isinstance(record_id, str) or type(index) is not int:
            raise ValueError("presentation identity fields have invalid types")
        key = (record_id, index)
        expected = expected_by_key.get(key)
        if expected is None:
            raise ValueError("presentation references an unexpected record or permutation")
        if key in normalized_by_key:
            raise ValueError("duplicate presentation for record and permutation")
        if (
            row.get("request_hash") != expected["request_hash"]
            or row.get("option_ids") != expected["option_ids"]
        ):
            raise ValueError(
                "presentation request hash or option order does not match the approved permutation"
            )
        normalized_by_key[key] = validate_presentation_result(
            expected,
            {
                "raw_logits": row.get("raw_logits"),
                "input_tokens": row.get("input_tokens"),
                "prompt_sha256": row.get("prompt_sha256"),
            },
        )
    if set(normalized_by_key) != set(expected_by_key):
        raise ValueError("model receipt is missing one or more approved permutations")

    result_by_key: dict[tuple[str, int], DecisionResult] = {}
    logits_by_key: dict[tuple[str, int], list[float]] = {}
    for key, row in normalized_by_key.items():
        request = DecisionRequest.model_validate(expected_by_key[key]["request"])
        logits = cast(list[float], row["raw_logits"])
        logits_by_key[key] = logits
        result_by_key[key] = score_candidates(
            request,
            logits,
            model_revision=revision,
            temperature=temperature,
            calibration_id=CALIBRATION_ID if model_name != "qwen" else None,
        )
    original_predictions = [
        PredictionRecord(
            record_id=record.record_id,
            request_hash=record.request.request_hash,
            logits=dict(
                zip(
                    (option.id for option in record.request.options),
                    cast(list[float], normalized_by_key[(record.record_id, 0)]["raw_logits"]),
                    strict=True,
                )
            ),
            model_revision=revision,
        )
        for record in records
    ]
    prediction_sha256 = _digest(
        json.dumps(
            [prediction.model_dump(mode="json") for prediction in original_predictions],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    raw_report = evaluate(
        records,
        original_predictions,
        manifest,
        records_sha256=records_sha256,
        predictions_sha256=prediction_sha256,
        manifest_sha256=manifest_sha256,
        temperature=1.0,
        calibration_id=None,
    ).model_dump(mode="json")
    shipped_report = evaluate(
        records,
        original_predictions,
        manifest,
        records_sha256=records_sha256,
        predictions_sha256=prediction_sha256,
        manifest_sha256=manifest_sha256,
        temperature=temperature,
        calibration_id=None if model_name == "qwen" else CALIBRATION_ID,
    ).model_dump(mode="json")
    raw_metrics, shipped_metrics = _report_metrics(raw_report), _report_metrics(shipped_report)

    records_by_dataset: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        records_by_dataset[record.dataset_id].append(record)
    per_dataset: dict[str, dict[str, object]] = {}
    permutation_accuracy: dict[str, dict[str, object]] = {}
    original_correct_by_record: dict[str, bool] = {}
    flips_by_record: dict[str, bool] = {}
    for dataset_id in DATASET_IDS:
        dataset_records = records_by_dataset[dataset_id]
        if not dataset_records:
            continue
        correct_count = 0
        ties = 0
        tie_counts_by_permutation = [0] * math.factorial(len(dataset_records[0].request.options))
        positions = len(dataset_records[0].request.options)
        gold_positions = {str(position): 0 for position in range(positions)}
        predicted_positions = {str(position): 0 for position in range(positions)}
        group_flips: dict[str, bool] = {}
        permutation_count = math.factorial(positions)
        correct_by_permutation = [0] * permutation_count
        permutation_orders = list(
            itertools.permutations(option.id for option in dataset_records[0].request.options)
        )
        for record in dataset_records:
            original = result_by_key[(record.record_id, 0)]
            original_correct = original.top_option_id == record.answer_id
            original_correct_by_record[record.record_id] = original_correct
            correct_count += original_correct
            ties += original.abstained and original.reason == "tie"
            group_flips.setdefault(record.source_group_id, False)
            original_option_ids = [option.id for option in record.request.options]
            gold_positions[str(original_option_ids.index(record.answer_id))] += 1
            predicted_positions[str(original_option_ids.index(original.top_option_id))] += 1
            flipped = False
            for permutation_index in range(permutation_count):
                result = result_by_key[(record.record_id, permutation_index)]
                tie_counts_by_permutation[permutation_index] += (
                    result.abstained and result.reason == "tie"
                )
                if result.top_option_id != original.top_option_id:
                    flipped = True
                correct_by_permutation[permutation_index] += (
                    result.top_option_id == record.answer_id
                )
            flips_by_record[record.record_id] = flipped
            group_flips[record.source_group_id] |= flipped
        raw_metrics_row = raw_metrics[dataset_id]
        shipped_metrics_row = shipped_metrics[dataset_id]
        per_dataset[dataset_id] = {
            "record_count": len(dataset_records),
            "independent_group_count": len(group_flips),
            "accuracy_interval": _wilson_interval(correct_count, len(dataset_records)),
            "raw_metrics": {
                key: raw_metrics_row[key] for key in ("accuracy", "macro_f1", "nll", "brier", "ece")
            },
            "shipped_temperature_metrics": {
                key: shipped_metrics_row[key]
                for key in ("accuracy", "macro_f1", "nll", "brier", "ece")
            },
            "gold_predicted_position_histogram": {
                "gold": gold_positions,
                "predicted": predicted_positions,
            },
            "tie_count_original_order": ties,
            "tie_count_all_permutations": sum(tie_counts_by_permutation),
            "any_semantic_flip_record_count": sum(
                flips_by_record[record.record_id] for record in dataset_records
            ),
            "any_semantic_flip_group_count": sum(group_flips.values()),
            "semantic_flip_group_count": len(group_flips),
        }
        permutation_accuracy[dataset_id] = {
            "independent_unit_count": len(dataset_records),
            "permutations": [
                {
                    "permutation_index": index,
                    "option_ids": list(permutation_orders[index]),
                    "correct": correct_by_permutation[index],
                    "record_count": len(dataset_records),
                    "accuracy": correct_by_permutation[index] / len(dataset_records),
                    "tie_count": tie_counts_by_permutation[index],
                }
                for index in range(permutation_count)
            ],
        }

    return {
        "model_id": model_id,
        "model_revision": revision,
        "temperature": temperature,
        "provenance": provenance,
        "prediction_sha256": prediction_sha256,
        "raw_report": raw_report,
        "shipped_report": shipped_report,
        "per_dataset": per_dataset,
        "permutation_accuracy": permutation_accuracy,
        "_original_correct": original_correct_by_record,
    }


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError("runner receipt contains a duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("runner receipt contains a non-finite JSON number")


def _validate_source_fingerprints(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("runner receipt has no source fingerprint map")
    checked: dict[str, str] = {}
    for path, digest in value.items():
        if (
            not isinstance(path, str)
            or path.startswith("/")
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise ValueError("runner receipt contains an invalid source fingerprint")
        checked[path] = digest
    if checked != _source_fingerprints():
        raise ValueError("runner receipt source fingerprints do not match the current code")
    return checked


def _paired_counts(
    left: Mapping[str, bool], right: Mapping[str, bool], records: Sequence[DecisionRecord]
) -> dict[str, int]:
    if set(left) != set(right) or set(left) != {record.record_id for record in records}:
        raise ValueError("paired model correctness does not cover the same original records")
    both = left_only = right_only = neither = 0
    for record in records:
        left_correct, right_correct = left[record.record_id], right[record.record_id]
        both += left_correct and right_correct
        left_only += left_correct and not right_correct
        right_only += right_correct and not left_correct
        neither += not left_correct and not right_correct
    return {
        "record_count": len(records),
        "both_correct": both,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "neither_correct": neither,
    }


def analyze_run(
    run_path: str | Path,
    records_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, object]:
    """Validate input and protocol provenance, then analyze all three receipts."""

    try:
        run_bytes = Path(run_path).read_bytes()
        run = json.loads(
            run_bytes,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as exc:
        raise ValueError("runner receipt is not strict UTF-8 JSON") from exc
    if not isinstance(run, dict) or run.get("status") != "passed":
        raise ValueError("runner receipt must have passed status")
    if run.get("schema_version") != 1:
        raise ValueError("runner receipt schema version is unsupported")
    if run.get("limits") != _plan()["modal"]:
        raise ValueError("runner receipt Modal limits do not match the frozen bounded plan")
    records_sha256 = _digest(Path(records_path).read_bytes())
    manifest_sha256 = _digest(Path(manifest_path).read_bytes())
    if run.get("records_sha256") != records_sha256:
        raise ValueError("runner receipt records SHA-256 does not match the prepared data")
    if run.get("manifest_sha256") != manifest_sha256:
        raise ValueError("runner receipt manifest SHA-256 does not match the prepared data")
    provenance = run.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("runner receipt has no provenance object")
    if (
        provenance.get("modal_sdk_version") != "1.6.1"
        or not isinstance(provenance.get("profile"), str)
        or not provenance["profile"].strip()
        or not isinstance(provenance.get("workspace"), str)
        or not provenance["workspace"].strip()
    ):
        raise ValueError("runner receipt has invalid Modal environment provenance")
    if provenance.get("baseline_protocol") != EXPECTED_PROTOCOL:
        raise ValueError("runner receipt does not identify the frozen broader protocol")
    protocol_path = Path(__file__).resolve().parents[1] / EXPECTED_PROTOCOL
    if (
        _digest(protocol_path.read_bytes()) != EXPECTED_PROTOCOL_SHA256
        or provenance.get("baseline_protocol_sha256") != EXPECTED_PROTOCOL_SHA256
    ):
        raise ValueError("runner receipt protocol SHA-256 does not match the frozen protocol")
    amendment_path = Path(__file__).resolve().parents[1] / EXPECTED_SOURCE_AMENDMENT
    if (
        provenance.get("source_amendment") != EXPECTED_SOURCE_AMENDMENT
        or _digest(amendment_path.read_bytes()) != EXPECTED_SOURCE_AMENDMENT_SHA256
        or provenance.get("source_amendment_sha256") != EXPECTED_SOURCE_AMENDMENT_SHA256
    ):
        raise ValueError("runner receipt source amendment does not match its frozen version")
    source_fingerprints = _validate_source_fingerprints(provenance.get("source_file_sha256"))

    manifest, records = broader_data.verify_prepared_data(records_path, manifest_path)
    _validate_record_set(records, manifest)
    if (
        _digest(Path(records_path).read_bytes()) != records_sha256
        or _digest(Path(manifest_path).read_bytes()) != manifest_sha256
    ):
        raise ValueError("prepared data changed while the analysis was loading it")
    if _source_fingerprints() != source_fingerprints:
        raise ValueError("source files changed while the analysis was running")

    receipts = run.get("models")
    if not isinstance(receipts, dict) or set(receipts) != set(MODELS):
        raise ValueError("runner receipt must contain exactly qwen, intern, and kev")
    per_model: dict[str, dict[str, object]] = {}
    correct_by_model: dict[str, dict[str, bool]] = {}
    expected_presentation_count = len(presentations(serialize_remote_records(records)))
    for name in MODELS:
        receipt = receipts[name]
        auxiliary = EXPECTED_AUXILIARY_FORWARD_COUNTS[name]
        if (
            not isinstance(receipt, dict)
            or receipt.get("model_name") != name
            or receipt.get("status") != "passed"
            or type(receipt.get("scored_presentation_count")) is not int
            or receipt["scored_presentation_count"] != expected_presentation_count
            or type(receipt.get("auxiliary_forward_count")) is not int
            or receipt["auxiliary_forward_count"] != auxiliary
            or type(receipt.get("total_forward_count")) is not int
            or receipt["total_forward_count"] != receipt["scored_presentation_count"] + auxiliary
        ):
            raise ValueError(f"{name} model receipt is incomplete or mismatched")
        model_provenance = receipt.get("provenance")
        if (
            not isinstance(model_provenance, dict)
            or (model_provenance.get("model_id"), model_provenance.get("model_revision"))
            != MODEL_PINS[name]
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
        correct_by_model[name] = cast(dict[str, bool], result.pop("_original_correct"))
        per_model[name] = result

    paired_original: list[dict[str, object]] = []
    records_by_dataset: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        records_by_dataset[record.dataset_id].append(record)
    for left, right in itertools.combinations(MODELS, 2):
        paired_original.append(
            {
                "left_model": left,
                "right_model": right,
                "overall": _paired_counts(correct_by_model[left], correct_by_model[right], records),
                "per_dataset": {
                    dataset_id: _paired_counts(
                        {
                            record.record_id: correct_by_model[left][record.record_id]
                            for record in dataset_records
                        },
                        {
                            record.record_id: correct_by_model[right][record.record_id]
                            for record in dataset_records
                        },
                        dataset_records,
                    )
                    for dataset_id, dataset_records in records_by_dataset.items()
                },
            }
        )
    return {
        "schema_version": 1,
        "status": "analyzed",
        "run_sha256": _digest(run_bytes),
        "records_sha256": records_sha256,
        "manifest_sha256": manifest_sha256,
        "baseline_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_amendment_sha256": EXPECTED_SOURCE_AMENDMENT_SHA256,
        "source_file_sha256": source_fingerprints,
        "record_count": len(records),
        "independent_group_count_by_dataset": {
            dataset_id: RECORDS_PER_DATASET for dataset_id in DATASET_IDS
        },
        "permutation_measurement_note": (
            "Repeated permutations are not additional independent units."
        ),
        "per_model": per_model,
        "paired_original_order": paired_original,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="saved broader Modal runner JSON receipt")
    parser.add_argument("--records", default="data/processed/broader-dev-pilot-v1.jsonl")
    parser.add_argument("--manifest", default="data/baselines/broader-dev-manifest.json")
    parser.add_argument("--output", help="new analysis JSON path; stdout by default")
    args = parser.parse_args(argv)
    try:
        result = analyze_run(args.run, args.records, args.manifest)
        if args.output:
            with reserve_output(args.output) as reservation:
                write_json_artifact(reservation, result)
        else:
            print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"analysis failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
