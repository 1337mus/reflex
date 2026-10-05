"""Strictly validate and summarize a saved adapter-transfer receipt."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import cast

from experiments import adapter_transfer_core as core
from experiments import adapter_transfer_data as data
from experiments.mixture_training_contracts import strict_json_loads
from reflex_decisions import smoke
from reflex_decisions.data import DecisionRecord

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261009
DATASET_IDS = ("boolq-dev-pilot-v1", "copa-dev-pilot-v1")
GROUPS_PER_DATASET = 32
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
_RECEIPT_FIELDS = frozenset(
    "schema_version run_id status phase payload result failure modal".split()
)


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _fraction(numerator: int, denominator: int) -> dict[str, int]:
    value = Fraction(numerator, denominator)
    return {"numerator": value.numerator, "denominator": value.denominator}


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("bootstrap percentile inputs are invalid")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    weight = location - lower
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * weight


def _failure_report(
    receipt: Mapping[str, object], result: Mapping[str, object] | None = None
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "failed",
        "run": {
            "run_id": receipt["run_id"],
            "phase": receipt["phase"],
            "receipt_status": receipt["status"],
            "result_status": result.get("status") if result else None,
            "selection_sha256": result.get("selection_sha256") if result else None,
            "payload_sha256": result.get("payload_sha256") if result else None,
            "failure": receipt.get("failure") or (result.get("failure") if result else None),
            "partial_evidence": result.get("evidence") if result else None,
        },
    }


def _local_payload(
    receipt: Mapping[str, object], root: Path
) -> tuple[dict[str, object], tuple[DecisionRecord, ...]]:
    stored = core.validate_payload(receipt["payload"], root=root)
    if stored.get("run_id") != receipt["run_id"]:
        raise ValueError("receipt run ID differs from the payload")
    records, presentations, selection, pins = data.load_inputs(root)
    expected = core.build_payload(
        cast(str, receipt["run_id"]),
        cast(str, stored["nonce"]),
        presentations,
        selection,
        pins,
        root=root,
    )
    if stored != expected:
        raise ValueError(
            "stored payload differs from the local payload, selection, data, or sources"
        )
    return expected, tuple(records)


def _model_winners(
    outputs: object, name: str, records: Mapping[str, DecisionRecord]
) -> dict[str, dict[int, str]]:
    rows = _object(outputs, "model outputs").get(name)
    if not isinstance(rows, list) or len(rows) != len(records) * 2:
        raise ValueError(f"{name} presentation rows are incomplete")
    winners: dict[str, dict[int, str]] = {record_id: {} for record_id in records}
    for raw in rows:
        row = cast(Mapping[str, object], raw)
        record_id, order = cast(str, row["record_id"]), cast(int, row["order_index"])
        if order in winners[record_id]:
            raise ValueError(f"{name} output contains a duplicate presentation")
        winners[record_id][order] = cast(str, row["winner_option_id"])
    if any(set(orders) != {0, 1} for orders in winners.values()):
        raise ValueError(f"{name} output is missing an order row")
    return winners


def _task_report(
    dataset_id: str,
    records: Sequence[DecisionRecord],
    winners: Mapping[str, Mapping[str, Mapping[int, str]]],
    samples: Sequence[Sequence[int]],
) -> dict[str, object]:
    ordered = sorted(records, key=lambda record: record.source_group_id)
    if (
        len(ordered) != GROUPS_PER_DATASET
        or len({r.source_group_id for r in ordered}) != GROUPS_PER_DATASET
    ):
        raise ValueError(f"{dataset_id} must contain exactly 32 source groups")
    totals = {name: {"original": 0, "both": 0, "flips": 0} for name in ("qwen_base", "adapter")}
    group_means, original_deltas, both_deltas = [], [], []
    for record in ordered:
        means, correct = {}, {}
        for output, model in (("base", "qwen_base"), ("adapter", "adapter")):
            predictions = winners[output][record.record_id]
            original = int(predictions[0] == record.answer_id)
            both = original + int(predictions[1] == record.answer_id)
            flipped = predictions[0] != predictions[1]
            totals[model]["original"] += original
            totals[model]["both"] += both
            totals[model]["flips"] += int(flipped)
            correct[model] = (original, both)
            means[model] = {
                "original_order": _fraction(original, 1),
                "both_orders": _fraction(both, 2),
                "semantic_answer_flipped": flipped,
            }
        base_original, base_both = correct["qwen_base"]
        adapter_original, adapter_both = correct["adapter"]
        original_deltas.append(adapter_original - base_original)
        both_deltas.append(adapter_both - base_both)
        means["paired_adapter_minus_base"] = {
            "original_order": _fraction(adapter_original - base_original, 1),
            "both_orders": _fraction(adapter_both - base_both, 2),
        }
        group_means.append({"source_group_id": record.source_group_id, **means})

    models = {
        name: {
            "original_order": {
                "correct": values["original"],
                "total": GROUPS_PER_DATASET,
                "exact_mean": _fraction(values["original"], GROUPS_PER_DATASET),
                "accuracy": values["original"] / GROUPS_PER_DATASET,
                "accuracy_percent": 100 * values["original"] / GROUPS_PER_DATASET,
            },
            "both_orders": {
                "correct": values["both"],
                "total": GROUPS_PER_DATASET * 2,
                "exact_mean": _fraction(values["both"], GROUPS_PER_DATASET * 2),
                "accuracy": values["both"] / (GROUPS_PER_DATASET * 2),
                "accuracy_percent": 100 * values["both"] / (GROUPS_PER_DATASET * 2),
            },
            "semantic_answer_flips": {"count": values["flips"], "total": GROUPS_PER_DATASET},
        }
        for name, values in totals.items()
    }

    def difference(values: Sequence[int], denominator: int, scale: int) -> dict[str, object]:
        draws = [
            sum(values[index] / scale for index in sample) / GROUPS_PER_DATASET
            for sample in samples
        ]
        delta = sum(values)
        return {
            "correct_delta": delta,
            "total": denominator,
            "exact_delta": _fraction(delta, denominator),
            "estimate": delta / denominator,
            "bootstrap_95": {
                "lower": _percentile(draws, 0.025),
                "upper": _percentile(draws, 0.975),
            },
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }

    return {
        "group_count": GROUPS_PER_DATASET,
        "models": models,
        "paired_adapter_minus_base": {
            "original_order": difference(original_deltas, GROUPS_PER_DATASET, 1),
            "both_orders": difference(both_deltas, GROUPS_PER_DATASET * 2, 2),
        },
        "group_means": group_means,
    }


def analyze_receipt(receipt: object, *, root: str | Path = DEFAULT_ROOT) -> dict[str, object]:
    """Rebuild local identities, strictly validate the result, then score complete runs."""

    envelope = _object(receipt, "receipt")
    run_id = envelope.get("run_id")
    if (
        set(envelope) != _RECEIPT_FIELDS
        or type(envelope.get("schema_version")) is not int
        or envelope["schema_version"] != 1
        or not isinstance(run_id, str)
        or not run_id.strip()
        or envelope.get("status") not in ("passed", "failed")
        or not isinstance(envelope.get("phase"), str)
        or (envelope.get("status") == "passed" and envelope["phase"] != "complete")
        or not isinstance(envelope.get("modal"), Mapping)
    ):
        raise ValueError("receipt schema, identity, or status is malformed")
    project_root = Path(root).expanduser().resolve(strict=True)
    if envelope["payload"] is None:
        if (
            envelope["status"] != "failed"
            or envelope["result"] is not None
            or envelope["failure"] is None
        ):
            raise ValueError("receipt is missing its payload")
        return _failure_report(envelope)
    payload, records = _local_payload(envelope, project_root)
    if envelope["result"] is None:
        if envelope["status"] != "failed" or envelope["failure"] is None:
            raise ValueError("receipt has no result or preserved failure")
        return _failure_report(envelope)
    result = core.validate_result(envelope["result"], payload, root=project_root)
    if (
        result.get("run_id") != envelope["run_id"]
        or (envelope["status"] == "passed" and result["status"] != "passed")
        or (envelope["status"] == "passed" and envelope["failure"] is not None)
    ):
        raise ValueError("receipt and validated result status or identity differ")
    if envelope["status"] != "passed" or result["status"] != "passed":
        return _failure_report(envelope, result)

    evidence = _object(result.get("evidence"), "result evidence")
    by_id = {record.record_id: record for record in records}
    winners = {
        name: _model_winners(evidence.get("outputs"), name, by_id) for name in ("base", "adapter")
    }
    generator = random.Random(BOOTSTRAP_SEED)
    tasks: dict[str, dict[str, object]] = {}
    for dataset_id in DATASET_IDS:
        samples = [
            [generator.randrange(GROUPS_PER_DATASET) for _ in range(GROUPS_PER_DATASET)]
            for _ in range(BOOTSTRAP_REPLICATES)
        ]
        tasks[dataset_id] = _task_report(
            dataset_id, [r for r in records if r.dataset_id == dataset_id], winners, samples
        )
    return {
        "schema_version": 1,
        "status": "analyzed",
        "run": {
            "run_id": envelope["run_id"],
            "receipt_status": envelope["status"],
            "result_status": result["status"],
            "selection": payload["selection"],
            "selection_sha256": payload["selection_sha256"],
            "payload_sha256": payload["payload_sha256"],
        },
        "bootstrap": {
            "unit": "source_group",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "95% type-7 percentile",
        },
        "tasks": tasks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.repo_root.expanduser().resolve(strict=True)
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
        print(f"analysis failed: {smoke.sanitize_exception_message(exc)}", file=sys.stderr)
        return 2
    status = cast(str, report["status"])
    print(json.dumps({"status": status, "artifact": str(reservation.destination)}))
    return 0 if status == "analyzed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
