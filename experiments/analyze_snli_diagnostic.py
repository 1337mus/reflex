"""Independently validate and analyze a saved balanced-SNLI diagnostic receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import cast

from experiments import snli_diagnostic_core as core
from experiments import snli_diagnostic_results as results
from reflex_decisions.broader_data import SNLI_LABELS
from reflex_decisions.data import (
    DecisionRecord,
    SplitManifest,
    audit_splits,
    load_manifest,
    load_records,
)
from reflex_decisions.smoke import reserve_output, write_json_artifact
from reflex_decisions.snli_diagnostic import DATASET_ID

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261006
DEFAULT_RECORDS = Path("data/processed/snli-balanced-v1/records.jsonl")
DEFAULT_MANIFEST = Path("data/processed/snli-balanced-v1/manifest.json")
DEFAULT_RECIPE = Path("data/processed/snli-balanced-v1/recipe.json")
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _ScoredRow:
    group_id: str
    record_id: str
    gold: str
    prediction: str
    logits: tuple[float, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON object contains a duplicate key")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("JSON contains a non-finite number")


def _read_json(path: Path, label: str) -> tuple[bytes, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: {label} is not strict UTF-8 JSON") from exc
    return raw, value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("bootstrap percentile inputs are invalid")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = floor(location)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = location - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _bootstrap_samples(group_count: int) -> list[list[int]]:
    if group_count < 1:
        raise ValueError("bootstrap requires at least one source group")
    generator = random.Random(BOOTSTRAP_SEED)
    return [
        [generator.randrange(group_count) for _ in range(group_count)]
        for _ in range(BOOTSTRAP_REPLICATES)
    ]


def _bootstrap_interval(
    group_correct: Mapping[str, int],
    group_total: Mapping[str, int],
    group_ids: Sequence[str],
    samples: Sequence[Sequence[int]],
) -> dict[str, float]:
    values = [
        sum(group_correct[group_ids[index]] for index in sample)
        / sum(group_total[group_ids[index]] for index in sample)
        for sample in samples
    ]
    return {"lower": _percentile(values, 0.025), "upper": _percentile(values, 0.975)}


def _bootstrap_difference_interval(
    final_correct: Mapping[str, int],
    final_total: Mapping[str, int],
    comparison_correct: Mapping[str, int],
    comparison_total: Mapping[str, int],
    group_ids: Sequence[str],
    samples: Sequence[Sequence[int]],
) -> dict[str, float]:
    values: list[float] = []
    for sample in samples:
        differences = [
            final_correct[group_ids[index]] / final_total[group_ids[index]]
            - comparison_correct[group_ids[index]] / comparison_total[group_ids[index]]
            for index in sample
        ]
        values.append(sum(differences) / len(differences))
    return {"lower": _percentile(values, 0.025), "upper": _percentile(values, 0.975)}


def _summarize_rows(
    rows: Sequence[_ScoredRow],
    group_ids: Sequence[str],
    samples: Sequence[Sequence[int]],
) -> tuple[dict[str, object], dict[str, int], dict[str, int]]:
    if not rows:
        raise ValueError("cannot summarize an empty presentation set")
    labels = SNLI_LABELS
    confusion = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    ties = 0
    unique_rows = 0
    unique_correct = 0
    group_correct: dict[str, int] = {group_id: 0 for group_id in group_ids}
    group_total: dict[str, int] = {group_id: 0 for group_id in group_ids}
    correct = 0
    for row in rows:
        confusion[row.gold][row.prediction] += 1
        is_correct = row.gold == row.prediction
        correct += is_correct
        group_correct[row.group_id] += is_correct
        group_total[row.group_id] += 1
        maximum = max(row.logits)
        maximum_count = sum(logit == maximum for logit in row.logits)
        ties += maximum_count > 1
        if maximum_count == 1:
            unique_rows += 1
            unique_correct += is_correct
    recall = {
        label: {
            "correct": confusion[label][label],
            "support": sum(confusion[label].values()),
            "recall": confusion[label][label] / sum(confusion[label].values()),
        }
        for label in labels
    }
    group_accuracy = sum(
        group_correct[group_id] / group_total[group_id]
        for group_id in group_ids
        if group_total[group_id]
    ) / len(group_ids)
    metric: dict[str, object] = {
        "accuracy": correct / len(rows),
        "correct": correct,
        "presentation_count": len(rows),
        "per_class_recall": recall,
        "confusion": confusion,
        "group_equal_weight_accuracy": group_accuracy,
        "unique_max_coverage": unique_rows / len(rows),
        "unique_max_count": unique_rows,
        "unique_max_correct_count": unique_correct,
        "unique_max_accuracy": unique_correct / unique_rows if unique_rows else None,
        "tie_count": ties,
        "bootstrap_95": _bootstrap_interval(group_correct, group_total, group_ids, samples),
    }
    return metric, group_correct, group_total


def _validated_receipt(
    receipt: object, records: Sequence[DecisionRecord]
) -> tuple[dict[str, object], dict[str, object]]:
    envelope = core._normalize_object(receipt, "receipt")
    saved_payload = core.validate_payload(envelope.get("payload"))
    pins = saved_payload["pins"]
    if not isinstance(pins, dict):
        raise ValueError("receipt payload pins are malformed")
    local_payload = core.build_payload(
        records,
        pins=pins,
        run_id=cast(str, saved_payload["run_id"]),
        nonce=cast(str, saved_payload["nonce"]),
    )
    validated = results.validate_receipt(envelope, local_payload)
    return validated, local_payload


def _validate_local_artifacts(
    *,
    root: Path,
    receipt_payload: Mapping[str, object],
    records: Sequence[DecisionRecord],
    manifest: SplitManifest,
    records_bytes: bytes,
    manifest_bytes: bytes,
    recipe_bytes: bytes,
) -> None:
    pins = receipt_payload.get("pins")
    if not isinstance(pins, dict):
        raise ValueError("receipt payload pins are malformed")
    actual = {
        "records_sha256": _sha256(records_bytes),
        "manifest_sha256": _sha256(manifest_bytes),
        "recipe_sha256": _sha256(recipe_bytes),
    }
    expected = {
        "records_sha256": core.EXPECTED_RECORDS_SHA256,
        "manifest_sha256": core.EXPECTED_MANIFEST_SHA256,
        "recipe_sha256": core.EXPECTED_RECIPE_SHA256,
    }
    if actual != expected or any(pins.get(name) != digest for name, digest in actual.items()):
        raise ValueError("local SNLI artifacts differ from their frozen receipt pins")
    if (
        manifest.data_kind != "benchmark"
        or manifest.held_out_families
        or len(manifest.datasets) != 1
        or manifest.datasets[0].dataset_id != DATASET_ID
        or manifest.datasets[0].split != "development"
    ):
        raise ValueError("local SNLI manifest differs from the frozen development panel")
    audit_splits(manifest, tuple(records))
    core._validate_records(records)
    core.verify_protocol(root / core.PROTOCOL_PATH)
    source_hashes = core.source_fingerprints(root)
    if pins.get("protocol_sha256") != core.EXPECTED_PROTOCOL_SHA256:
        raise ValueError("receipt protocol pin differs from the frozen protocol")
    if pins.get("source_file_sha256") != source_hashes:
        raise ValueError("local source fingerprints differ from the saved receipt pins")


def _scored_rows(
    state: Mapping[str, object], records: Sequence[DecisionRecord]
) -> tuple[list[_ScoredRow], list[_ScoredRow]]:
    record_by_id = {record.record_id: record for record in records}
    all_rows: list[_ScoredRow] = []
    original_rows: list[_ScoredRow] = []
    presentations = state.get("presentations")
    if not isinstance(presentations, list):
        raise ValueError("validated model state has no presentation list")
    for output in presentations:
        if not isinstance(output, dict):
            raise ValueError("validated presentation row is malformed")
        record_id = cast(str, output["record_id"])
        record = record_by_id[record_id]
        row = _ScoredRow(
            group_id=record.source_group_id,
            record_id=record_id,
            gold=record.answer_id,
            prediction=cast(str, output["winner_option_id"]),
            logits=tuple(cast(list[float], output["candidate_logits"])),
        )
        all_rows.append(row)
        if output["order_index"] == 0:
            original_rows.append(row)
    return all_rows, original_rows


def analyze_receipt(receipt: object, records: Sequence[DecisionRecord]) -> dict[str, object]:
    """Validate receipt identity against local records, then report complete-model metrics."""
    validated, payload = _validated_receipt(receipt, records)
    group_ids = sorted({record.source_group_id for record in records})
    samples = _bootstrap_samples(len(group_ids))
    models: dict[str, dict[str, object]] = {}
    model_failures: dict[str, dict[str, object]] = {}
    group_counts: dict[str, tuple[dict[str, int], dict[str, int]]] = {}
    model_states = cast(dict[str, dict[str, object]], validated["models"])
    for name in core.MODEL_NAMES:
        state = model_states[name]
        if state["status"] == "failed":
            failure = state.get("failure")
            counts = state["forward_counts"]
            model_failures[name] = {
                "status": "failed",
                "model_id": state["model_id"],
                "model_revision": state["model_revision"],
                "failure": failure,
                "forward_counts": counts,
                "presentations_completed": len(cast(list[object], state["presentations"])),
            }
            continue
        all_rows, original_rows = _scored_rows(state, records)
        all_metrics, all_correct, all_total = _summarize_rows(all_rows, group_ids, samples)
        original_metrics, _, _ = _summarize_rows(original_rows, group_ids, samples)
        group_counts[name] = (all_correct, all_total)
        record_predictions: dict[str, set[str]] = defaultdict(set)
        record_correct: dict[str, list[bool]] = defaultdict(list)
        for row in all_rows:
            record_predictions[row.record_id].add(row.prediction)
            record_correct[row.record_id].append(row.gold == row.prediction)
        models[name] = {
            "model_id": state["model_id"],
            "model_revision": state["model_revision"],
            "forward_counts": state["forward_counts"],
            "all_order": all_metrics,
            "original_order": original_metrics,
            "record_behavior": {
                "semantic_flip_count": sum(
                    len(predictions) > 1 for predictions in record_predictions.values()
                ),
                "all_six_correct_record_count": sum(
                    len(record_correct[record.record_id]) == core.ORDERS_PER_RECORD
                    and all(record_correct[record.record_id])
                    for record in records
                ),
                "record_count": len(records),
            },
        }

    paired_differences: dict[str, dict[str, object]] = {}
    if "qwen_final" in group_counts:
        final_correct, final_total = group_counts["qwen_final"]
        final_model = models["qwen_final"]
        final_accuracy = cast(float, cast(dict[str, object], final_model["all_order"])["accuracy"])
        for comparison in core.MODEL_NAMES:
            if comparison == "qwen_final" or comparison not in group_counts:
                continue
            other_correct, other_total = group_counts[comparison]
            other_model = models[comparison]
            other_accuracy = cast(
                float, cast(dict[str, object], other_model["all_order"])["accuracy"]
            )
            key = f"qwen_final_minus_{comparison}"
            paired_differences[key] = {
                "estimate": final_accuracy - other_accuracy,
                "ci95": _bootstrap_difference_interval(
                    final_correct,
                    final_total,
                    other_correct,
                    other_total,
                    group_ids,
                    samples,
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
    return {
        "schema_version": core.SCHEMA_VERSION,
        "status": "analyzed",
        "run": {
            "run_id": payload["run_id"],
            "nonce": payload["nonce"],
            "receipt_status": validated["status"],
            "panel_sha256": payload["panel_sha256"],
            "pins": payload["pins"],
            "lifecycle_failure": validated.get("failure"),
        },
        "bootstrap": {
            "unit": "source_group",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "95% percentile",
        },
        "models": models,
        "model_failures": model_failures,
        "paired_all_order_differences": paired_differences,
    }


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True, help="saved Modal receipt JSON")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, help="write analysis JSON to a new path")
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    records_path = _resolve(root, args.records)
    manifest_path = _resolve(root, args.manifest)
    recipe_path = _resolve(root, args.recipe)
    receipt_path = _resolve(root, args.receipt)
    records_bytes = records_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    recipe_bytes = recipe_path.read_bytes()
    _, raw_receipt = _read_json(receipt_path, "receipt")
    records = load_records(records_path)
    manifest = load_manifest(manifest_path)
    envelope = core._normalize_object(raw_receipt, "receipt")
    receipt_payload = core.validate_payload(envelope.get("payload"))
    _validate_local_artifacts(
        root=root,
        receipt_payload=receipt_payload,
        records=records,
        manifest=manifest,
        records_bytes=records_bytes,
        manifest_bytes=manifest_bytes,
        recipe_bytes=recipe_bytes,
    )
    if (
        _sha256(records_path.read_bytes()) != _sha256(records_bytes)
        or _sha256(manifest_path.read_bytes()) != _sha256(manifest_bytes)
        or _sha256(recipe_path.read_bytes()) != _sha256(recipe_bytes)
    ):
        raise ValueError("local SNLI artifacts changed during analysis preflight")
    report = analyze_receipt(raw_receipt, records)
    encoded = json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output is None:
        print(encoded)
        return 0
    output_path = _resolve(root, args.output)
    with reserve_output(output_path) as reservation:
        write_json_artifact(reservation, report)
    print(json.dumps({"status": report["status"], "artifact": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
