"""Strictly validate and summarize a fresh public-source evaluation receipt."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import cast

from experiments import fresh_eval_analysis_validation as validation

_STATES = validation.STATES
_DATASET_COUNTS = validation.DATASET_COUNTS
_RECEIPT_FIELDS = validation.RECEIPT_FIELDS
_winner_maps = validation.winner_maps
_record_sha256 = validation._record_sha256
_local_records = validation._local_records
_validate_host_metadata = validation._validate_host_metadata
_BOOTSTRAP_REPLICATES = 2_000
_BOOTSTRAP_SEED = 20261012
_DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _fraction(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _percentile_type7(values: Sequence[Fraction], probability: Fraction) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = location.numerator // location.denominator
    weight = location - lower
    upper = min(lower + 1, len(ordered) - 1)
    return float(ordered[lower] + weight * (ordered[upper] - ordered[lower]))


def _task_metrics(
    dataset_id: str,
    records: Sequence[Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, Mapping[int, str]]],
    *,
    bootstrap_draws: Sequence[Sequence[str]],
) -> dict[str, object]:
    """Summarize validated question winners with equal source-group weight."""

    if not dataset_id or not records or set(predictions) != set(_STATES):
        raise ValueError("task metrics inputs are incomplete")
    if not bootstrap_draws:
        raise ValueError("bootstrap draws are empty")
    ids: set[str] = set()
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        record_id = record.get("record_id")
        if (
            record.get("dataset_id") != dataset_id
            or not isinstance(record_id, str)
            or not record_id
            or record_id in ids
            or not isinstance(record.get("source_group_id"), str)
            or not record.get("source_group_id")
            or not isinstance(record.get("answer_id"), str)
            or not isinstance(record.get("option_ids"), list)
            or len(record["option_ids"]) < 2
            or len(set(record["option_ids"])) != len(record["option_ids"])
            or record["answer_id"] not in record["option_ids"]
        ):
            raise ValueError("task metric record identity or labels are malformed")
        ids.add(record_id)
        groups[record["source_group_id"]].append(record)
    group_ids = sorted(groups)
    for draw in bootstrap_draws:
        if len(draw) != len(group_ids) or any(group_id not in groups for group_id in draw):
            raise ValueError("bootstrap draw does not resample source groups")

    raw = {state: {"original": 0, "both": 0, "flips": 0} for state in _STATES}
    group_state_scores: dict[str, dict[str, dict[str, Fraction]]] = {}
    for state in _STATES:
        state_predictions = predictions[state]
        if set(state_predictions) != ids:
            raise ValueError(f"{state} predictions are missing or adding records")
        group_sums: dict[str, dict[str, Fraction]] = {
            group_id: {"original": Fraction(0), "both": Fraction(0)} for group_id in group_ids
        }
        for record in records:
            record_id = record["record_id"]
            group_id = record["source_group_id"]
            answers = state_predictions[record_id]
            if set(answers) != {0, 1} or any(
                not isinstance(winner, str) or winner not in record["option_ids"]
                for winner in answers.values()
            ):
                raise ValueError(f"{state} has missing orders or a non-semantic winner")
            original = int(answers[0] == record["answer_id"])
            rotated = int(answers[1] == record["answer_id"])
            raw[state]["original"] += original
            raw[state]["both"] += original + rotated
            raw[state]["flips"] += int(answers[0] != answers[1])
            group_sums[group_id]["original"] += Fraction(original)
            group_sums[group_id]["both"] += Fraction(original + rotated, 2)
        group_state_scores[state] = {
            group_id: {
                metric: score / len(group_records) for metric, score in group_sums[group_id].items()
            }
            for group_id, group_records in groups.items()
        }

    model_reports: dict[str, object] = {}
    for state in _STATES:
        values = raw[state]
        reports: dict[str, object] = {}
        for metric, count, denominator in (
            ("original_order_accuracy", "original", len(records)),
            ("both_orders_accuracy", "both", 2 * len(records)),
        ):
            group_mean = sum(
                (group_state_scores[state][group_id][count] for group_id in group_ids),
                Fraction(0),
            ) / len(group_ids)
            reports[metric] = {
                "correct": values[count],
                "total": denominator,
                "accuracy": values[count] / denominator,
                "source_group_mean_accuracy": _fraction(group_mean),
            }
        reports["semantic_winner_changes"] = {
            "count": values["flips"],
            "total": len(records),
        }
        model_reports[state] = reports

    paired: dict[str, object] = {}
    for metric in ("original", "both"):
        differences = {
            group_id: group_state_scores["adapter"][group_id][metric]
            - group_state_scores["base"][group_id][metric]
            for group_id in group_ids
        }
        estimate = sum(differences.values(), Fraction(0)) / len(group_ids)
        draws = [
            sum((differences[group_id] for group_id in sample), Fraction(0)) / len(sample)
            for sample in bootstrap_draws
        ]
        paired["original_order" if metric == "original" else "both_orders"] = {
            "group_mean_difference": _fraction(estimate),
            "estimate": float(estimate),
            "bootstrap_95": {
                "lower": _percentile_type7(draws, Fraction(1, 40)),
                "upper": _percentile_type7(draws, Fraction(39, 40)),
            },
        }
    return {
        "question_count": len(records),
        "source_group_count": len(group_ids),
        "models": model_reports,
        "paired_adapter_minus_base": paired,
    }


def _bootstrap_group_draws(group_ids: Sequence[str], generator: random.Random) -> list[list[str]]:
    if not group_ids:
        raise ValueError("cannot bootstrap an empty task")
    return [
        [group_ids[generator.randrange(len(group_ids))] for _ in range(len(group_ids))]
        for _ in range(_BOOTSTRAP_REPLICATES)
    ]


def _hans_diagnostics(
    records: Sequence[Mapping[str, object]],
    metadata_by_id: Mapping[str, Mapping[str, object]],
    predictions: Mapping[str, Mapping[str, Mapping[int, str]]],
) -> dict[str, object]:
    if not records or set(metadata_by_id) != {cast(str, row["record_id"]) for row in records}:
        raise ValueError("HANS diagnostic metadata does not cover the local task")
    groupings: dict[str, dict[str, list[Mapping[str, object]]]] = {
        "heuristic": defaultdict(list),
        "subcase": defaultdict(list),
    }
    for record in records:
        record_id = cast(str, record["record_id"])
        row = metadata_by_id[record_id]
        for field in groupings:
            label = row.get(field)
            if not isinstance(label, str) or not label:
                raise ValueError(f"HANS {field} diagnostic label is missing")
            groupings[field][label].append(record)
    if len(groupings["heuristic"]) != 3 or len(groupings["subcase"]) != 30:
        raise ValueError("HANS diagnostics require exactly three heuristics and 30 subcases")
    output: dict[str, object] = {}
    for dimension, groups in groupings.items():
        reports: dict[str, object] = {}
        for label in sorted(groups):
            task_records = groups[label]
            models: dict[str, object] = {}
            for state in _STATES:
                original_correct = 0
                both_correct = 0
                flips = 0
                for record in task_records:
                    record_id = cast(str, record["record_id"])
                    answer = cast(str, record["answer_id"])
                    orders = predictions[state][record_id]
                    original_correct += int(orders[0] == answer)
                    both_correct += int(orders[0] == answer) + int(orders[1] == answer)
                    flips += int(orders[0] != orders[1])
                models[state] = {
                    "original_order_accuracy": {
                        "correct": original_correct,
                        "total": len(task_records),
                        "accuracy": original_correct / len(task_records),
                    },
                    "both_orders_accuracy": {
                        "correct": both_correct,
                        "total": 2 * len(task_records),
                        "accuracy": both_correct / (2 * len(task_records)),
                    },
                    "semantic_winner_changes": {"count": flips, "total": len(task_records)},
                }
            reports[label] = {"question_count": len(task_records), "models": models}
        output[dimension] = reports
    return output


def analyze_receipt(receipt: object, *, root: str | Path = _DEFAULT_ROOT) -> dict[str, object]:
    """Validate a successful receipt against local data, then compute fixed metrics."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("receipt schema is malformed")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("experiment_id") != "fresh-eval-v1"
        or not isinstance(receipt.get("run_id"), str)
        or not receipt["run_id"].strip()
        or receipt.get("status") != "passed"
        or receipt.get("phase") != "complete"
        or receipt.get("failure") is not None
    ):
        raise ValueError("only a complete passed fresh-evaluation receipt can be analyzed")
    execution = receipt.get("execution")
    modal = receipt.get("modal")
    if (
        not isinstance(execution, Mapping)
        or execution.get("unknown_work") is not None
        or not isinstance(execution.get("run_marker"), str)
        or not execution["run_marker"].strip()
        or not isinstance(modal, Mapping)
        or modal.get("profile") != "reflex-personal"
        or modal.get("workspace") != "rajath-61258"
        or modal.get("environment") != "main"
        or modal.get("sdk_version") != "1.6.1"
        or modal.get("volume_read_only") is not True
    ):
        raise ValueError("passed receipt execution identity or isolation evidence is incomplete")
    from experiments import fresh_eval_core as core
    from reflex_decisions.fresh_eval_data import load_panel

    project_root = Path(root).expanduser().resolve(strict=True)
    payload = core.validate_payload(receipt.get("payload"), root=project_root)
    if payload.get("run_id") != receipt["run_id"]:
        raise ValueError("receipt run ID differs from its frozen payload")
    result = core.validate_result(receipt.get("result"), payload, root=project_root)
    raw_worker_result = receipt.get("raw_worker_result")
    validated_raw = core.validate_result(raw_worker_result, payload, root=project_root)
    if (
        result.get("status") != "passed"
        or result.get("run_id") != receipt["run_id"]
        or validated_raw != result
    ):
        raise ValueError("receipt, raw worker result, and validated result disagree")
    records, metadata = load_panel(project_root)
    normalized_records, records_by_id = _local_records(records)
    stored_gpu_pins = payload.get("pins", {}).get("gpu_pins")
    local_gpu_pins = metadata.get("gpu_pins") if isinstance(metadata, Mapping) else None
    if stored_gpu_pins != local_gpu_pins:
        raise ValueError("receipt dataset pins differ from the locally loaded public data")
    local_provenance = _validate_host_metadata(metadata, records_by_id, stored_gpu_pins, core=core)
    winners = _winner_maps(payload, result, records_by_id)
    generator = random.Random(_BOOTSTRAP_SEED)
    tasks: dict[str, dict[str, object]] = {}
    for dataset_id in sorted(_DATASET_COUNTS):
        task_records = [row for row in normalized_records if row["dataset_id"] == dataset_id]
        group_ids = sorted({cast(str, row["source_group_id"]) for row in task_records})
        draws = _bootstrap_group_draws(group_ids, generator)
        task_ids = {cast(str, row["record_id"]) for row in task_records}
        task_winners = {
            state: {record_id: winners[state][record_id] for record_id in task_ids}
            for state in _STATES
        }
        tasks[dataset_id] = _task_metrics(
            dataset_id, task_records, task_winners, bootstrap_draws=draws
        )
    hans_records = [row for row in normalized_records if row["dataset_id"] == "hans-eval-v1"]
    hans_metadata = local_provenance.pop("hans_record_metadata")
    if not isinstance(hans_metadata, Mapping):
        raise ValueError("HANS diagnostic metadata is unavailable")
    hans_diagnostics = _hans_diagnostics(hans_records, hans_metadata, winners)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "analyzed",
        "run": {
            "experiment_id": "fresh-eval-v1",
            "run_id": receipt["run_id"],
            "receipt_status": receipt["status"],
            "result_status": result["status"],
            "selection": payload["selection"],
            "selection_sha256": payload["selection_sha256"],
            "payload_sha256": payload["payload_sha256"],
        },
        "provenance": {
            **local_provenance,
            "gpu_data_pins": stored_gpu_pins,
            "code_source_file_sha256": payload["pins"]["source_file_sha256"],
            "model_id": result["provenance"]["model_id"],
            "model_revision": result["provenance"]["model_revision"],
            "adapter_tensor_sha256": result["evidence"]["adapter_identity"]["tensor_sha256"],
            "reload_max_candidate_logit_delta": result["evidence"]["reload_parity"][
                "max_candidate_logit_delta"
            ],
        },
        "bootstrap": {
            "unit": "source_group_within_task",
            "replicates": _BOOTSTRAP_REPLICATES,
            "seed": _BOOTSTRAP_SEED,
            "interval": "95% paired type-7 percentile",
            "task_and_group_order": "sorted",
        },
        "tasks": tasks,
    }
    report["hans_diagnostics"] = hans_diagnostics
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from experiments.mixture_training_contracts import strict_json_loads
        from reflex_decisions import smoke

        root = args.root.expanduser().resolve(strict=True)
        receipt_path = args.receipt.expanduser()
        output_path = args.output.expanduser()
        if not receipt_path.is_absolute():
            receipt_path = root / receipt_path
        if not output_path.is_absolute():
            output_path = root / output_path
        with smoke.reserve_output(output_path) as reservation:
            report = analyze_receipt(strict_json_loads(receipt_path.read_bytes()), root=root)
            smoke.write_json_artifact(reservation, report)
    except (OSError, ValueError, RecursionError) as exc:
        try:
            from reflex_decisions import smoke

            message = smoke.sanitize_exception_message(exc)
        except Exception:
            message = type(exc).__name__
        print(f"analysis failed: {message}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "artifact": str(reservation.destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
