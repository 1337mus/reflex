"""Validate and analyze one saved real-pilot run using CPU-only metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from numbers import Real
from pathlib import Path

from experiments import real_pilot_core as core
from reflex_decisions import calibration, pilot_data
from reflex_decisions.data import DecisionRecord, SplitManifest
from reflex_decisions.evaluation import EvaluationReport, PredictionRecord, evaluate
from reflex_decisions.schema import DecisionRequest
from reflex_decisions.scoring import score_candidates
from reflex_decisions.smoke import reserve_output, write_json_artifact

DEFAULT_RECORDS = Path("data/processed/real-pilot-v1.jsonl")
DEFAULT_MANIFEST = Path("data/pilots/real-pilot-v1-manifest.json")
DEFAULT_RECIPE = Path("data/pilots/real-pilot-v1-recipe.json")
DEFAULT_OUTPUT = Path("artifacts/real-pilot-analysis-v1.json")
CALIBRATION_ID = "real-pilot-v1-fixed-grid-nll"
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261005
_CALIBRATION_DATASETS = (
    "dbpedia14-pilot-v1-calibration",
    "sms-pilot-v1-calibration",
)
_DEVELOPMENT_DATASETS = (
    "dbpedia14-pilot-v1-development",
    "sms-pilot-v1-development",
    "snli-pilot-v1-development",
)
_METRIC_FIELDS = (
    "accuracy",
    "macro_f1",
    "nll",
    "brier",
    "ece",
    "coverage",
    "accepted_risk",
    "aurc",
)


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError("JSON object contains a duplicate key")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("JSON contains a non-finite number")


def _read_json(path: str | Path, label: str) -> tuple[bytes, object]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as exc:
        raise ValueError(f"{source}: {label} is not strict UTF-8 JSON") from exc
    return raw, value


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("analysis value is not strict JSON") from exc
    return _digest(encoded)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _wilson_interval(successes: int, count: int) -> dict[str, float | int]:
    if type(successes) is not int or type(count) is not int or not 0 <= successes <= count:
        raise ValueError("Wilson interval counts are invalid")
    if count < 1:
        raise ValueError("Wilson interval requires at least one source group")
    z = 1.959963984540054
    z2 = z * z
    observed = successes / count
    denominator = 1.0 + z2 / count
    center = (observed + z2 / (2.0 * count)) / denominator
    margin = z * math.sqrt(observed * (1.0 - observed) / count + z2 / (4.0 * count * count))
    margin /= denominator
    return {
        "successes": successes,
        "source_group_count": count,
        "lower_95": max(0.0, center - margin),
        "upper_95": min(1.0, center + margin),
    }


def _percentile_linear(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile inputs are invalid")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    fraction = location - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _gate_fraction(value: Real | Fraction, label: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(_finite_number(value, label)))


def _minimum_delta_passed(delta: Real | Fraction, *, minimum: Real | Fraction) -> bool:
    """Apply an inclusive predeclared accuracy-delta threshold."""

    return _gate_fraction(delta, "accuracy delta") >= _gate_fraction(minimum, "minimum delta")


def _order_flip_reduction(
    base_fraction: Real | Fraction, final_fraction: Real | Fraction
) -> tuple[float | None, bool]:
    """Apply the relative 25% flip-rate gate, including the zero-base rule."""

    base = _gate_fraction(base_fraction, "base order-flip fraction")
    final = _gate_fraction(final_fraction, "final order-flip fraction")
    if not 0.0 <= base <= 1.0 or not 0.0 <= final <= 1.0:
        raise ValueError("order-flip fractions must be between zero and one")
    if base == 0.0:
        return (0.0, True) if final == 0.0 else (None, False)
    reduction = (base - final) / base
    return float(reduction), reduction >= Fraction(1, 4)


def _group_records(records: Sequence[DecisionRecord]) -> dict[str, tuple[DecisionRecord, ...]]:
    grouped: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_group_id].append(record)
    return {group_id: tuple(rows) for group_id, rows in grouped.items()}


def _require_single_record_per_group(records: Sequence[DecisionRecord]) -> None:
    if len(_group_records(records)) != len(records):
        raise ValueError("each pilot dataset must contain one representative per source group")


def _panel_maps(
    expected_payload: Mapping[str, object],
) -> tuple[dict[str, dict[str, object]], dict[tuple[str, int], dict[str, object]]]:
    raw_rows = expected_payload.get("evaluation_presentations")
    if not isinstance(raw_rows, list):
        raise ValueError("expected payload has no evaluation presentation panel")
    by_id: dict[str, dict[str, object]] = {}
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for value in raw_rows:
        if not isinstance(value, dict):
            raise ValueError("expected presentation rows must be objects")
        presentation_id, record_id, order_index = (
            value.get("presentation_id"),
            value.get("record_id"),
            value.get("order_index"),
        )
        if (
            not isinstance(presentation_id, str)
            or not isinstance(record_id, str)
            or type(order_index) is not int
        ):
            raise ValueError("expected presentation identity is malformed")
        key = (record_id, order_index)
        if presentation_id in by_id or key in by_key:
            raise ValueError("expected presentation panel contains duplicate IDs")
        normalized = value
        by_id[presentation_id] = normalized
        by_key[key] = normalized
    if len(by_id) != core.EVALUATION_PRESENTATION_COUNT:
        raise ValueError("expected presentation panel is incomplete")
    return by_id, by_key


def _output_maps(receipt: Mapping[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("passed receipt has no evidence object")
    outputs: dict[str, dict[str, dict[str, object]]] = {}
    for state, key in (("base", "base_outputs"), ("final", "final_outputs")):
        rows = evidence.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"passed receipt has no {state} outputs")
        by_id: dict[str, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("presentation_id"), str):
                raise ValueError(f"{state} output row is malformed")
            presentation_id = row["presentation_id"]
            if presentation_id in by_id:
                raise ValueError(f"{state} outputs contain a duplicate presentation")
            by_id[presentation_id] = row
        outputs[state] = by_id
    return outputs


def _semantic_logits(
    output: Mapping[str, object], presentation: Mapping[str, object]
) -> dict[str, float]:
    option_ids, raw_logits = presentation.get("order_ids"), output.get("candidate_logits")
    if not isinstance(option_ids, list) or not isinstance(raw_logits, list):
        raise ValueError("presentation option IDs or candidate logits are malformed")
    if len(option_ids) != len(raw_logits) or not option_ids:
        raise ValueError("candidate logits do not align with the presentation option IDs")
    ids: list[str] = []
    logits: list[float] = []
    for index, (option_id, raw_logit) in enumerate(zip(option_ids, raw_logits, strict=True)):
        if not isinstance(option_id, str) or not option_id.strip():
            raise ValueError("presentation option IDs must be nonblank strings")
        ids.append(option_id)
        logits.append(_finite_number(raw_logit, f"candidate_logits[{index}]"))
    if len(ids) != len(set(ids)):
        raise ValueError("presentation option IDs must be unique")
    return dict(zip(ids, logits, strict=True))


def _original_logits(
    record: DecisionRecord,
    output_by_id: Mapping[str, Mapping[str, object]],
    panel_by_key: Mapping[tuple[str, int], Mapping[str, object]],
) -> tuple[float, ...]:
    presentation = panel_by_key.get((record.record_id, 0))
    if presentation is None:
        raise ValueError("evaluation panel is missing a record's original order")
    presentation_id = presentation.get("presentation_id")
    output = output_by_id.get(presentation_id) if isinstance(presentation_id, str) else None
    if output is None:
        raise ValueError("validated output is missing a record's original order")
    logits_by_id = _semantic_logits(output, presentation)
    option_ids = tuple(option.id for option in record.request.options)
    if tuple(presentation.get("order_ids", ())) != option_ids or set(logits_by_id) != set(
        option_ids
    ):
        raise ValueError("original presentation IDs differ from the verified source record")
    return tuple(logits_by_id[option_id] for option_id in option_ids)


def _calibration_fit(
    records: Sequence[DecisionRecord],
    output_by_id: Mapping[str, Mapping[str, object]],
    panel_by_key: Mapping[tuple[str, int], Mapping[str, object]],
) -> dict[str, object]:
    by_dataset = {
        dataset_id: tuple(record for record in records if record.dataset_id == dataset_id)
        for dataset_id in _CALIBRATION_DATASETS
    }
    expected_sizes = dict(zip(_CALIBRATION_DATASETS, (56, 60), strict=True))
    if any(len(by_dataset[dataset_id]) != expected_sizes[dataset_id] for dataset_id in by_dataset):
        raise ValueError("calibration pools do not match the frozen DBpedia/SMS membership counts")

    examples: list[calibration.CalibrationExample] = []
    row_hashes: dict[str, str] = {}
    memberships: dict[str, list[str]] = {}
    for dataset_id in _CALIBRATION_DATASETS:
        dataset_records = by_dataset[dataset_id]
        _require_single_record_per_group(dataset_records)
        row_weight = 1.0 / len(dataset_records)
        memberships[dataset_id] = []
        for record in sorted(dataset_records, key=lambda item: item.record_id):
            logits = _original_logits(record, output_by_id, panel_by_key)
            option_ids = tuple(option.id for option in record.request.options)
            gold_index = option_ids.index(record.answer_id)
            examples.append(calibration.CalibrationExample(logits, gold_index, row_weight))
            membership = {
                "record_id": record.record_id,
                "dataset_id": dataset_id,
                "source_group_id": record.source_group_id,
                "request_hash": record.request.request_hash,
                "gold_option_id": record.answer_id,
                "semantic_logits": dict(zip(option_ids, logits, strict=True)),
            }
            row_hashes[record.record_id] = _json_digest(membership)
            memberships[dataset_id].append(record.record_id)

    fit = calibration.fit_temperature(examples)
    ordered_hashes = [
        row_hashes[record_id]
        for dataset_id in _CALIBRATION_DATASETS
        for record_id in memberships[dataset_id]
    ]
    return {
        "temperature": fit.temperature,
        "raw_nll": fit.raw_nll,
        "fitted_nll": fit.fitted_nll,
        "example_count": fit.example_count,
        "grid_range": list(fit.grid_range),
        "grid_count": fit.grid_count,
        "boundary_hit": fit.boundary_hit,
        "task_weighting": {
            dataset_id: {
                "record_count": len(memberships[dataset_id]),
                "row_weight": 1.0 / len(memberships[dataset_id]),
                "total_weight": 1.0,
            }
            for dataset_id in _CALIBRATION_DATASETS
        },
        "fit_membership": memberships,
        "row_sha256_by_record_id": row_hashes,
        "pool_sha256": _json_digest(
            {
                "calibration_id": CALIBRATION_ID,
                "ordered_row_sha256": ordered_hashes,
                "fit_membership": memberships,
            }
        ),
        "interpretation": "calibration-pool NLL is a fitting diagnostic, not held-out performance",
    }


def _prediction_rows(
    records: Sequence[DecisionRecord],
    output_by_id: Mapping[str, Mapping[str, object]],
    panel_by_key: Mapping[tuple[str, int], Mapping[str, object]],
    model_revision: str,
) -> tuple[PredictionRecord, ...]:
    predictions: list[PredictionRecord] = []
    for record in records:
        logits = _original_logits(record, output_by_id, panel_by_key)
        predictions.append(
            PredictionRecord(
                record_id=record.record_id,
                request_hash=record.request.request_hash,
                logits=dict(
                    zip((option.id for option in record.request.options), logits, strict=True)
                ),
                model_revision=model_revision,
            )
        )
    return tuple(predictions)


def _prediction_digest(predictions: Sequence[PredictionRecord]) -> str:
    return _json_digest([prediction.model_dump(mode="json") for prediction in predictions])


def _evaluate_dataset(
    records: Sequence[DecisionRecord],
    predictions: Sequence[PredictionRecord],
    manifest: SplitManifest,
    *,
    records_sha256: str,
    manifest_sha256: str,
    temperature: float,
    calibration_id: str | None,
) -> dict[str, object]:
    report: EvaluationReport = evaluate(
        records,
        predictions,
        manifest,
        records_sha256=records_sha256,
        predictions_sha256=_prediction_digest(predictions),
        manifest_sha256=manifest_sha256,
        temperature=temperature,
        calibration_id=calibration_id,
    )
    if len(report.per_dataset) != 1:
        raise ValueError("per-task evaluation must return exactly one dataset")
    return report.per_dataset[0].model_dump(mode="json")


def _macro_metrics(by_dataset: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    rows = [by_dataset[dataset_id] for dataset_id in _DEVELOPMENT_DATASETS[:2]]
    result: dict[str, object] = {}
    for field in _METRIC_FIELDS:
        values = [row.get(field) for row in rows]
        finite = [
            float(value)
            for value in values
            if isinstance(value, Real) and not isinstance(value, bool)
        ]
        result[field] = math.fsum(finite) / len(finite) if finite else None
    result["dataset_count"] = len(rows)
    result["record_count"] = sum(int(row["record_count"]) for row in rows)
    result["pooling"] = "unweighted DBpedia/SMS task macro average; SNLI excluded"
    return result


def _semantic_order_analysis(
    dataset_records: Sequence[DecisionRecord],
    output_by_id: Mapping[str, Mapping[str, object]],
    panel_by_id: Mapping[str, Mapping[str, object]],
    model_revision: str,
) -> tuple[dict[str, object], dict[str, bool]]:
    _require_single_record_per_group(dataset_records)
    correct_by_record: dict[str, bool] = {}
    group_flip_count = 0
    original_correct = 0
    all_orders_correct = 0
    tie_count = 0
    tie_count_original = 0
    correct_by_order: dict[int, int] = defaultdict(int)
    tie_by_order: dict[int, int] = defaultdict(int)
    order_count: int | None = None

    rows_by_record: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = (
        defaultdict(list)
    )
    for presentation_id, presentation in panel_by_id.items():
        if presentation.get("dataset_id") != dataset_records[0].dataset_id:
            continue
        output = output_by_id.get(presentation_id)
        if output is None:
            raise ValueError("order analysis output is missing a fixed presentation")
        record_id = presentation.get("record_id")
        if not isinstance(record_id, str):
            raise ValueError("order panel contains an invalid record ID")
        rows_by_record[record_id].append((presentation, output))

    for record in dataset_records:
        presentations = rows_by_record.get(record.record_id)
        if not presentations:
            raise ValueError("order analysis panel is missing a source record")
        presentations.sort(key=lambda pair: int(pair[0]["order_index"]))
        indexes = [int(presentation["order_index"]) for presentation, _ in presentations]
        if indexes != list(range(len(indexes))) or len(indexes) < 2:
            raise ValueError("order analysis panel has missing or duplicate fixed orders")
        if order_count is None:
            order_count = len(indexes)
        elif order_count != len(indexes):
            raise ValueError("records in one dataset must have the same fixed order count")
        winners: set[str] = set()
        row_correct: list[bool] = []
        for (presentation, output), order_index in zip(presentations, indexes, strict=True):
            request = DecisionRequest.model_validate(presentation["request"])
            logits_by_id = _semantic_logits(output, presentation)
            logits = tuple(logits_by_id[option.id] for option in request.options)
            result = score_candidates(request, logits, model_revision=model_revision)
            winners.add(result.top_option_id)
            is_correct = result.top_option_id == record.answer_id
            correct_by_order[order_index] += is_correct
            row_correct.append(is_correct)
            if result.reason == "tie":
                tie_count += 1
                tie_by_order[order_index] += 1
                if order_index == 0:
                    tie_count_original += 1
        is_original_correct = row_correct[0]
        correct_by_record[record.record_id] = is_original_correct
        original_correct += is_original_correct
        all_orders_correct += all(row_correct)
        group_flip_count += len(winners) > 1

    total = len(dataset_records)
    order_total = order_count or 0
    return (
        {
            "record_count": total,
            "source_group_count": total,
            "independent_unit_count": total,
            "order_count": order_total,
            "semantic_top_answer_flip_record_count": group_flip_count,
            "semantic_top_answer_flip_fraction": group_flip_count / total,
            "semantic_top_answer_flip_wilson_95": _wilson_interval(group_flip_count, total),
            "original_order_correct_count": original_correct,
            "original_order_accuracy": original_correct / total,
            "original_order_accuracy_wilson_95": _wilson_interval(original_correct, total),
            "correct_by_order_index": [
                {
                    "order_index": order_index,
                    "correct_count": correct_by_order[order_index],
                    "record_count": total,
                    "accuracy": correct_by_order[order_index] / total,
                    "tie_count": tie_by_order[order_index],
                }
                for order_index in range(order_total)
            ],
            "minimum_order_accuracy": min(
                correct_by_order[index] / total for index in range(order_total)
            ),
            "maximum_order_accuracy": max(
                correct_by_order[index] / total for index in range(order_total)
            ),
            "all_orders_correct_count": all_orders_correct,
            "tie_count_original_order": tie_count_original,
            "tie_count_all_orders": tie_count,
            "sampling_note": (
                "source groups are the units; fixed option orders are repeated measurements"
            ),
        },
        correct_by_record,
    )


def _paired_bootstrap(
    records: Sequence[DecisionRecord],
    base_correct: Mapping[str, bool],
    final_correct: Mapping[str, bool],
    rng: random.Random,
) -> dict[str, object]:
    _require_single_record_per_group(records)
    deltas = [
        int(final_correct[record.record_id]) - int(base_correct[record.record_id])
        for record in records
    ]
    if not deltas:
        raise ValueError("paired bootstrap requires at least one source group")
    replicates: list[float] = []
    count = len(deltas)
    for _ in range(BOOTSTRAP_REPLICATES):
        replicates.append(math.fsum(deltas[rng.randrange(count)] for _ in range(count)) / count)
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "source_group_count": count,
        "seed": BOOTSTRAP_SEED,
        "percentile_95": [
            _percentile_linear(replicates, 0.025),
            _percentile_linear(replicates, 0.975),
        ],
        "percentile_interpolation": "linear at h=(n-1)*p (type-7 convention)",
    }


def _paired_dataset_comparison(
    records: Sequence[DecisionRecord],
    base_correct: Mapping[str, bool],
    final_correct: Mapping[str, bool],
    rng: random.Random,
) -> dict[str, object]:
    if set(base_correct) != set(final_correct) or set(base_correct) != {
        record.record_id for record in records
    }:
        raise ValueError("base and final correctness must cover the same source records")
    both = base_only = final_only = neither = 0
    for record in records:
        left, right = base_correct[record.record_id], final_correct[record.record_id]
        both += left and right
        base_only += left and not right
        final_only += right and not left
        neither += not left and not right
    count = len(records)
    base_accuracy = sum(base_correct.values()) / count
    final_accuracy = sum(final_correct.values()) / count
    return {
        "record_count": count,
        "source_group_count": count,
        "base_accuracy": base_accuracy,
        "final_accuracy": final_accuracy,
        "accuracy_delta_final_minus_base": final_accuracy - base_accuracy,
        "base_correct_final_wrong": base_only,
        "base_wrong_final_correct": final_only,
        "both_correct": both,
        "both_wrong": neither,
        "paired_group_bootstrap_95": _paired_bootstrap(records, base_correct, final_correct, rng),
    }


def _macro_bootstrap_interval(
    records_by_dataset: Mapping[str, Sequence[DecisionRecord]],
    base_correct: Mapping[str, Mapping[str, bool]],
    final_correct: Mapping[str, Mapping[str, bool]],
    rng: random.Random,
) -> dict[str, object]:
    datasets = _DEVELOPMENT_DATASETS[:2]
    grouped_deltas: dict[str, list[int]] = {}
    for dataset_id in datasets:
        records = records_by_dataset[dataset_id]
        _require_single_record_per_group(records)
        grouped_deltas[dataset_id] = [
            int(final_correct[dataset_id][record.record_id])
            - int(base_correct[dataset_id][record.record_id])
            for record in records
        ]
    replicates: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        dataset_deltas = []
        for dataset_id in datasets:
            deltas = grouped_deltas[dataset_id]
            dataset_deltas.append(
                math.fsum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
            )
        replicates.append(math.fsum(dataset_deltas) / len(dataset_deltas))
    point = math.fsum(
        sum(grouped_deltas[dataset_id]) / len(grouped_deltas[dataset_id]) for dataset_id in datasets
    ) / len(datasets)
    return {
        "dataset_ids": list(datasets),
        "accuracy_delta_final_minus_base": point,
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "percentile_95": [
            _percentile_linear(replicates, 0.025),
            _percentile_linear(replicates, 0.975),
        ],
        "percentile_interpolation": "linear at h=(n-1)*p (type-7 convention)",
        "sampling": "independent source-group resampling within each dataset, then unweighted mean",
    }


def _analyze_state(
    state: str,
    receipt: Mapping[str, object],
    output_by_id: Mapping[str, Mapping[str, object]],
    panel_by_id: Mapping[str, Mapping[str, object]],
    panel_by_key: Mapping[tuple[str, int], Mapping[str, object]],
    records: Sequence[DecisionRecord],
    manifest: SplitManifest,
    *,
    records_sha256: str,
    manifest_sha256: str,
) -> tuple[dict[str, object], dict[str, dict[str, bool]]]:
    provenance = receipt.get("provenance")
    evidence = receipt.get("evidence")
    if not isinstance(provenance, dict) or not isinstance(evidence, dict):
        raise ValueError("passed receipt is missing provenance or evidence")
    base_revision = provenance.get("model_revision")
    if not isinstance(base_revision, str) or not base_revision.strip():
        raise ValueError("passed receipt has no pinned model revision")
    final_adapter_hash = _final_adapter_sha256(evidence)
    model_revision = (
        base_revision if state == "base" else f"{base_revision}+adapter-sha256:{final_adapter_hash}"
    )
    calibration_report = _calibration_fit(records, output_by_id, panel_by_key)
    temperature = float(calibration_report["temperature"])
    evaluation_records = tuple(
        record for record in records if record.dataset_id not in core.TRAIN_DATASET_IDS
    )
    predictions = _prediction_rows(evaluation_records, output_by_id, panel_by_key, model_revision)
    by_dataset_records: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        by_dataset_records[record.dataset_id].append(record)

    metric_sets: dict[str, dict[str, dict[str, object]]] = {"temperature_1": {}, "fitted": {}}
    correctness: dict[str, dict[str, bool]] = {}
    order_analysis: dict[str, dict[str, object]] = {}
    for dataset_id in _DEVELOPMENT_DATASETS:
        dataset_records = tuple(by_dataset_records[dataset_id])
        dataset_record_ids = {record.record_id for record in dataset_records}
        dataset_predictions = tuple(
            prediction for prediction in predictions if prediction.record_id in dataset_record_ids
        )
        for metric_name, metric_temperature, calibration_id in (
            ("temperature_1", 1.0, None),
            ("fitted", temperature, CALIBRATION_ID),
        ):
            metric_sets[metric_name][dataset_id] = _evaluate_dataset(
                dataset_records,
                dataset_predictions,
                manifest,
                records_sha256=records_sha256,
                manifest_sha256=manifest_sha256,
                temperature=metric_temperature,
                calibration_id=calibration_id,
            )
        order_result, correct_map = _semantic_order_analysis(
            dataset_records, output_by_id, panel_by_id, model_revision
        )
        order_analysis[dataset_id] = order_result
        correctness[dataset_id] = correct_map

    for metric_name in metric_sets:
        metric_sets[metric_name]["dbpedia-sms-macro"] = _macro_metrics(metric_sets[metric_name])
    return (
        {
            "model_revision": model_revision,
            "temperature": temperature,
            "calibration_fit": calibration_report,
            "metrics": metric_sets,
            "order_analysis": order_analysis,
            "tie_policy": (
                "semantic option ID breaks score ties; tied results automatically abstain"
            ),
            "winner_temperature_invariance": (
                "positive temperature scaling preserves non-tied semantic winners"
            ),
        },
        correctness,
    )


def _final_adapter_sha256(evidence: Mapping[str, object]) -> str:
    snapshots = evidence.get("adapter_paths")
    if not isinstance(snapshots, list):
        raise ValueError("passed receipt has no adapter snapshots")
    final = next(
        (
            row
            for row in snapshots
            if isinstance(row, dict) and row.get("update") == core.MAX_UPDATES
        ),
        None,
    )
    files = final.get("files_sha256") if isinstance(final, dict) else None
    value = files.get("adapter_model.safetensors") if isinstance(files, dict) else None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("passed receipt has no final adapter model SHA-256")
    return value


def _validate_data_pins(
    records_path: str | Path,
    manifest_path: str | Path,
    recipe_path: str | Path,
    recipe: Mapping[str, object],
    *,
    records_sha256: str,
    manifest_sha256: str,
    recipe_sha256: str,
) -> None:
    outputs = recipe.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("verified recipe has no prepared-output pins")
    project_root = Path(__file__).resolve().parents[1]
    if (project_root / pilot_data.DEFAULT_RECIPE_PATH).resolve() != Path(recipe_path).resolve():
        raise ValueError("prepared recipe path differs from the pinned recipe path")
    expected_records_path = outputs.get("records_path")
    expected_manifest_path = outputs.get("manifest_path")
    if (
        not isinstance(expected_records_path, str)
        or not isinstance(expected_manifest_path, str)
        or (project_root / expected_records_path).resolve() != Path(records_path).resolve()
        or (project_root / expected_manifest_path).resolve() != Path(manifest_path).resolve()
    ):
        raise ValueError("prepared data paths differ from the verified recipe")
    expected = {
        "records_sha256": records_sha256,
        "manifest_sha256": manifest_sha256,
    }
    if any(outputs.get(key) != value for key, value in expected.items()):
        raise ValueError("prepared data hashes or paths differ from the verified recipe")
    # Reading the recipe bytes is part of its verifier pin; the run receipt pins this exact file.
    if len(recipe_sha256) != 64:
        raise ValueError("verified recipe SHA-256 is malformed")


def analyze_receipt(
    receipt_value: object,
    expected_payload: dict[str, object],
    records: Sequence[DecisionRecord],
    manifest: SplitManifest,
    *,
    records_sha256: str,
    manifest_sha256: str,
) -> dict[str, object]:
    """Validate one complete runner receipt before joining any local labels."""

    receipt = core.validate_passed_receipt(receipt_value, expected_payload=expected_payload)
    core.validate_pilot_data(manifest, records)
    by_group_dataset: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        if record.dataset_id not in core.TRAIN_DATASET_IDS:
            by_group_dataset[record.dataset_id].append(record)
    for dataset_records in by_group_dataset.values():
        _require_single_record_per_group(dataset_records)

    panel_by_id, panel_by_key = _panel_maps(expected_payload)
    output_maps = _output_maps(receipt)
    base_rows, final_rows = output_maps["base"], output_maps["final"]
    if set(base_rows) != set(panel_by_id) or set(final_rows) != set(panel_by_id):
        raise ValueError("passed receipt does not cover the exact verified evaluation panel")
    for presentation_id in panel_by_id:
        if base_rows[presentation_id].get("prompt_sha256") != final_rows[presentation_id].get(
            "prompt_sha256"
        ):
            raise ValueError("base and final outputs used different prompt hashes")

    base, base_correct = _analyze_state(
        "base",
        receipt,
        base_rows,
        panel_by_id,
        panel_by_key,
        records,
        manifest,
        records_sha256=records_sha256,
        manifest_sha256=manifest_sha256,
    )
    final, final_correct = _analyze_state(
        "final",
        receipt,
        final_rows,
        panel_by_id,
        panel_by_key,
        records,
        manifest,
        records_sha256=records_sha256,
        manifest_sha256=manifest_sha256,
    )

    records_by_dataset: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        if record.dataset_id in _DEVELOPMENT_DATASETS:
            records_by_dataset[record.dataset_id].append(record)
    bootstrap_rng = random.Random(BOOTSTRAP_SEED)
    paired = {
        dataset_id: _paired_dataset_comparison(
            records_by_dataset[dataset_id],
            base_correct[dataset_id],
            final_correct[dataset_id],
            bootstrap_rng,
        )
        for dataset_id in _DEVELOPMENT_DATASETS
    }
    macro_bootstrap = _macro_bootstrap_interval(
        records_by_dataset, base_correct, final_correct, bootstrap_rng
    )
    paired_macro = {
        "dataset_ids": list(_DEVELOPMENT_DATASETS[:2]),
        "record_count": sum(
            int(paired[dataset_id]["record_count"]) for dataset_id in _DEVELOPMENT_DATASETS[:2]
        ),
        "base_accuracy": math.fsum(
            float(paired[item]["base_accuracy"]) for item in _DEVELOPMENT_DATASETS[:2]
        )
        / 2,
        "final_accuracy": math.fsum(
            float(paired[item]["final_accuracy"]) for item in _DEVELOPMENT_DATASETS[:2]
        )
        / 2,
        "accuracy_delta_final_minus_base": macro_bootstrap["accuracy_delta_final_minus_base"],
        "paired_group_bootstrap_95": macro_bootstrap,
    }
    base_macro_accuracy = float(base["metrics"]["temperature_1"]["dbpedia-sms-macro"]["accuracy"])
    final_macro_accuracy = float(final["metrics"]["temperature_1"]["dbpedia-sms-macro"]["accuracy"])
    task_deltas = {
        dataset_id: float(paired[dataset_id]["accuracy_delta_final_minus_base"])
        for dataset_id in _DEVELOPMENT_DATASETS
    }
    task_delta_fractions = {
        dataset_id: Fraction(
            sum(final_correct[dataset_id].values()) - sum(base_correct[dataset_id].values()),
            len(records_by_dataset[dataset_id]),
        )
        for dataset_id in _DEVELOPMENT_DATASETS
    }
    dbpedia_sms = _DEVELOPMENT_DATASETS[:2]
    base_macro_accuracy_fraction = (
        sum(
            (
                Fraction(
                    sum(base_correct[dataset_id].values()), len(records_by_dataset[dataset_id])
                )
                for dataset_id in dbpedia_sms
            ),
            Fraction(0, 1),
        )
        / 2
    )
    final_macro_accuracy_fraction = (
        sum(
            (
                Fraction(
                    sum(final_correct[dataset_id].values()), len(records_by_dataset[dataset_id])
                )
                for dataset_id in dbpedia_sms
            ),
            Fraction(0, 1),
        )
        / 2
    )
    base_flip = {
        dataset_id: float(base["order_analysis"][dataset_id]["semantic_top_answer_flip_fraction"])
        for dataset_id in _DEVELOPMENT_DATASETS
    }
    final_flip = {
        dataset_id: float(final["order_analysis"][dataset_id]["semantic_top_answer_flip_fraction"])
        for dataset_id in _DEVELOPMENT_DATASETS
    }
    base_macro_flip = math.fsum(base_flip[item] for item in _DEVELOPMENT_DATASETS[:2]) / 2
    final_macro_flip = math.fsum(final_flip[item] for item in _DEVELOPMENT_DATASETS[:2]) / 2
    base_macro_flip_fraction = (
        sum(
            (
                Fraction(
                    int(
                        base["order_analysis"][dataset_id]["semantic_top_answer_flip_record_count"]
                    ),
                    len(records_by_dataset[dataset_id]),
                )
                for dataset_id in dbpedia_sms
            ),
            Fraction(0, 1),
        )
        / 2
    )
    final_macro_flip_fraction = (
        sum(
            (
                Fraction(
                    int(
                        final["order_analysis"][dataset_id]["semantic_top_answer_flip_record_count"]
                    ),
                    len(records_by_dataset[dataset_id]),
                )
                for dataset_id in dbpedia_sms
            ),
            Fraction(0, 1),
        )
        / 2
    )
    flip_reduction, flip_gate_passed = _order_flip_reduction(
        base_macro_flip_fraction, final_macro_flip_fraction
    )
    gate_components = {
        "dbpedia_sms_macro_accuracy_gain": {
            "delta": final_macro_accuracy - base_macro_accuracy,
            "paired_bootstrap_95": macro_bootstrap["percentile_95"],
            "required_minimum": 0.05,
            "passed": _minimum_delta_passed(
                final_macro_accuracy_fraction - base_macro_accuracy_fraction, minimum=0.05
            ),
        },
        "dbpedia_accuracy_drop_limit": {
            "delta": task_deltas[_DEVELOPMENT_DATASETS[0]],
            "paired_bootstrap_95": paired[_DEVELOPMENT_DATASETS[0]]["paired_group_bootstrap_95"][
                "percentile_95"
            ],
            "minimum_allowed_delta": -0.05,
            "passed": _minimum_delta_passed(
                task_delta_fractions[_DEVELOPMENT_DATASETS[0]], minimum=-0.05
            ),
        },
        "sms_accuracy_drop_limit": {
            "delta": task_deltas[_DEVELOPMENT_DATASETS[1]],
            "paired_bootstrap_95": paired[_DEVELOPMENT_DATASETS[1]]["paired_group_bootstrap_95"][
                "percentile_95"
            ],
            "minimum_allowed_delta": -0.05,
            "passed": _minimum_delta_passed(
                task_delta_fractions[_DEVELOPMENT_DATASETS[1]], minimum=-0.05
            ),
        },
        "dbpedia_sms_macro_order_flip_reduction": {
            "base_fraction": base_macro_flip,
            "final_fraction": final_macro_flip,
            "relative_reduction": flip_reduction,
            "base_wilson_95": {
                dataset_id: base["order_analysis"][dataset_id]["semantic_top_answer_flip_wilson_95"]
                for dataset_id in _DEVELOPMENT_DATASETS[:2]
            },
            "final_wilson_95": {
                dataset_id: final["order_analysis"][dataset_id][
                    "semantic_top_answer_flip_wilson_95"
                ]
                for dataset_id in _DEVELOPMENT_DATASETS[:2]
            },
            "required_relative_reduction": 0.25,
            "passed": flip_gate_passed,
        },
        "snli_accuracy_drop_limit": {
            "delta": task_deltas[_DEVELOPMENT_DATASETS[2]],
            "paired_bootstrap_95": paired[_DEVELOPMENT_DATASETS[2]]["paired_group_bootstrap_95"][
                "percentile_95"
            ],
            "minimum_allowed_delta": -0.05,
            "passed": _minimum_delta_passed(
                task_delta_fractions[_DEVELOPMENT_DATASETS[2]], minimum=-0.05
            ),
        },
    }
    gate = {
        "components": gate_components,
        "overall_passed": all(bool(component["passed"]) for component in gate_components.values()),
        "interpretation": "predeclared engineering progression gate; not a significance test",
    }

    provenance = receipt["provenance"]
    evidence = receipt["evidence"]
    return {
        "analysis_protocol": "reflex-real-pilot-analysis-v1",
        "run_identity": {
            "run_id": receipt["run_id"],
            "nonce": receipt["nonce"],
            "pins": receipt["pins"],
        },
        "model": {
            "model_id": provenance["model_id"],
            "base_model_revision": provenance["model_revision"],
            "source_file_sha256": provenance["source_file_sha256"],
            "measured_source_file_sha256": provenance.get("measured_source_file_sha256"),
            "adapter_snapshots": evidence["adapter_paths"],
        },
        "states": {"base": base, "final": final},
        "paired_comparison": {
            "per_dataset": paired,
            "dbpedia_sms_macro": paired_macro,
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "sampling_unit": "source group",
                "percentile_interpolation": "linear at h=(n-1)*p (type-7 convention)",
                "permutations_are_independent_units": False,
            },
        },
        "engineering_gate": gate,
        "data_disclosure": (
            "SNLI is development-only; no corpus text or training labels are included"
        ),
    }


def _final_data_recipe(
    records_path: str | Path,
    manifest_path: str | Path,
    recipe_path: str | Path,
) -> tuple[SplitManifest, tuple[DecisionRecord, ...], dict[str, object], dict[str, str]]:
    records_raw, manifest_raw, recipe_raw = (
        Path(records_path).read_bytes(),
        Path(manifest_path).read_bytes(),
        Path(recipe_path).read_bytes(),
    )
    records_sha256, manifest_sha256, recipe_sha256 = map(
        _digest, (records_raw, manifest_raw, recipe_raw)
    )
    manifest, records, recipe_value = pilot_data.verify_prepared_data(
        records_path, manifest_path, recipe_path
    )
    if not isinstance(recipe_value, dict):
        raise ValueError("prepared-data verifier returned an invalid recipe")
    _validate_data_pins(
        records_path,
        manifest_path,
        recipe_path,
        recipe_value,
        records_sha256=records_sha256,
        manifest_sha256=manifest_sha256,
        recipe_sha256=recipe_sha256,
    )
    if (
        _digest(Path(records_path).read_bytes()) != records_sha256
        or _digest(Path(manifest_path).read_bytes()) != manifest_sha256
        or _digest(Path(recipe_path).read_bytes()) != recipe_sha256
    ):
        raise ValueError("prepared files changed while the analysis was loading them")
    return (
        manifest,
        records,
        recipe_value,
        {
            "records_sha256": records_sha256,
            "manifest_sha256": manifest_sha256,
            "recipe_sha256": recipe_sha256,
        },
    )


def analyze_run(
    receipt_path: str | Path,
    records_path: str | Path = DEFAULT_RECORDS,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    recipe_path: str | Path = DEFAULT_RECIPE,
) -> dict[str, object]:
    """Verify local source pins, rebuild the panel, and analyze one passed receipt."""

    _receipt_bytes, receipt_value = _read_json(receipt_path, "runner receipt")
    if not isinstance(receipt_value, dict):
        raise ValueError("runner receipt must be a JSON object")
    manifest, records, recipe, hashes = _final_data_recipe(records_path, manifest_path, recipe_path)
    # Verify the frozen protocol and all code dependencies before reconstructing the worker payload.
    project_root = Path(__file__).resolve().parents[1]
    protocol_path = project_root / core.PROTOCOL_PATH
    protocol_sha256 = core.verify_protocol(protocol_path)
    source_file_sha256 = core.source_fingerprints(project_root)
    if source_file_sha256.get(core.PROTOCOL_PATH) != protocol_sha256:
        raise ValueError("protocol hash differs from the source fingerprint map")
    run_id, nonce = receipt_value.get("run_id"), receipt_value.get("nonce")
    expected_payload = core.build_remote_payload(
        manifest,
        records,
        run_id=run_id,  # type: ignore[arg-type]
        nonce=nonce,  # type: ignore[arg-type]
        records_sha256=hashes["records_sha256"],
        manifest_sha256=hashes["manifest_sha256"],
        recipe_sha256=hashes["recipe_sha256"],
        protocol_sha256=protocol_sha256,
        source_file_sha256=source_file_sha256,
    )
    result = analyze_receipt(
        receipt_value,
        expected_payload,
        records,
        manifest,
        records_sha256=hashes["records_sha256"],
        manifest_sha256=hashes["manifest_sha256"],
    )
    if (
        _digest(Path(records_path).read_bytes()) != hashes["records_sha256"]
        or _digest(Path(manifest_path).read_bytes()) != hashes["manifest_sha256"]
        or _digest(Path(recipe_path).read_bytes()) != hashes["recipe_sha256"]
        or core.source_fingerprints(project_root) != source_file_sha256
    ):
        raise ValueError("source inputs changed while the analysis was running")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = analyze_run(args.receipt, args.records, args.manifest, args.recipe)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with reserve_output(args.output) as reservation:
            write_json_artifact(reservation, result)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved real-pilot analysis to {args.output}")


if __name__ == "__main__":
    main()
