"""Fixed-grid calibration diagnostics for mixture evaluation outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

from experiments.mixture_training_statistics import (
    _CALIBRATION_COUNTS,
    _DEV_DATASET_IDS,
    _Observed,
)
from reflex_decisions import calibration, evaluation
from reflex_decisions.data import DecisionRecord
from reflex_decisions.scoring import DecisionResult, score_candidates


def _semantic_logits(observed: _Observed) -> tuple[float, ...]:
    order_ids = cast(list[str], observed.row["order_ids"])
    logits = cast(list[float], observed.row["candidate_logits"])
    by_option = dict(zip(order_ids, logits, strict=True))
    return tuple(by_option[option.id] for option in observed.record.request.options)


def _calibration_fit(
    outputs: Mapping[str, Mapping[str, object]],
    records: Sequence[DecisionRecord],
) -> tuple[calibration.TemperatureFitResult, dict[str, object]]:
    record_by_id = {record.record_id: record for record in records}
    rows_by_task: dict[str, list[calibration.CalibrationExample]] = defaultdict(list)
    for output in outputs.values():
        record_id = cast(str, output["record_id"])
        record = record_by_id[record_id]
        if record.dataset_id not in _CALIBRATION_COUNTS or output["order_index"] != 0:
            continue
        order_ids = cast(list[str], output["order_ids"])
        logits = cast(list[float], output["candidate_logits"])
        rows_by_task[record.dataset_id].append(
            calibration.CalibrationExample(
                tuple(logits),
                order_ids.index(record.answer_id),
                1.0 / _CALIBRATION_COUNTS[record.dataset_id],
            )
        )
    counts = {name: len(rows_by_task.get(name, ())) for name in _CALIBRATION_COUNTS}
    if counts != _CALIBRATION_COUNTS or sum(counts.values()) != 241:
        raise ValueError("calibration must contain exactly 241 original-order rows from four tasks")
    fit = calibration.fit_temperature(
        example for task in _CALIBRATION_COUNTS for example in rows_by_task[task]
    )
    if fit.grid_count != 82 or fit.example_count != 241:
        raise ValueError("calibration utility returned an unexpected fixed-grid result")
    return fit, {
        "temperature": fit.temperature,
        "raw_nll": fit.raw_nll,
        "fitted_nll": fit.fitted_nll,
        "example_count": fit.example_count,
        "grid_range": list(fit.grid_range),
        "grid_count": fit.grid_count,
        "boundary_hit": fit.boundary_hit,
        "fit_dataset_counts": counts,
        "fit_task_weight": {name: 1.0 for name in _CALIBRATION_COUNTS},
    }


def _confidence_metrics(
    observed: Mapping[str, Sequence[_Observed]], temperature: float, revision: str
) -> dict[str, object]:
    report: dict[str, object] = {}
    for dataset_id in _DEV_DATASET_IDS:
        scored: list[Any] = []
        for row in observed[dataset_id]:
            if row.row["order_index"] != 0:
                continue
            result: DecisionResult = score_candidates(
                row.record.request,
                _semantic_logits(row),
                model_revision=revision,
                temperature=temperature,
            )
            confidence = max(score.probability for score in result.scores)
            scored.append(
                evaluation._ScoredRow(
                    record=row.record,
                    result=result,
                    correct=result.top_option_id == row.record.answer_id,
                    confidence=confidence,
                )
            )
        values = evaluation._metrics(scored)
        fields = evaluation._metric_fields(values)
        fields["risk_coverage_curve"] = [
            point.model_dump(mode="json") for point in values.risk_coverage_curve
        ]
        report[dataset_id] = fields
    return report
