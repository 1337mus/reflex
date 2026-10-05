"""Deterministic, CPU-only descriptive statistics for mixture evaluations."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from reflex_decisions.data import DecisionRecord

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261007

_DEV_DATASET_IDS = (
    "dbpedia14-pilot-v1-development",
    "sms-pilot-v1-development",
    "snli-pilot-v1-development",
    "snli-balanced-v1-development",
    "synthetic-atomic-fact-inference-v1-development",
    "synthetic-numeric-selection-v1-development",
)
_CALIBRATION_COUNTS = {
    "dbpedia14-pilot-v1-calibration": 56,
    "sms-pilot-v1-calibration": 60,
    "synthetic-atomic-fact-inference-v1-calibration": 75,
    "synthetic-numeric-selection-v1-calibration": 50,
}


@dataclass(frozen=True)
class _Observed:
    row: Mapping[str, object]
    record: DecisionRecord
    correct: bool
    tied: bool


def _fraction(count: int, total: int) -> Fraction:
    if total < 1 or not 0 <= count <= total:
        raise ValueError("accuracy count is outside its presentation total")
    return Fraction(count, total)


def paired_group_delta(
    final: Mapping[str, Fraction], control: Mapping[str, Fraction]
) -> dict[str, Fraction]:
    """Return per-source-group final-minus-control accuracy differences."""

    if set(final) != set(control):
        raise ValueError("paired model states must contain the same source groups")
    return {group_id: final[group_id] - control[group_id] for group_id in sorted(final)}


def passes_minimum(value: Fraction, threshold: Fraction) -> bool:
    """Compare exact raw proportions to a predeclared minimum."""

    if not isinstance(value, Fraction) or not isinstance(threshold, Fraction):
        raise TypeError("gate values and thresholds must be exact Fraction values")
    return value >= threshold


def _type7(values: Sequence[Fraction], probability: Fraction) -> Fraction:
    if not values or not isinstance(probability, Fraction) or not 0 <= probability <= 1:
        raise ValueError("percentile input is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _bootstrap_samples(group_count: int, rng: random.Random) -> tuple[tuple[int, ...], ...]:
    if group_count < 1:
        raise ValueError("bootstrap requires at least one source group")
    return tuple(
        tuple(rng.randrange(group_count) for _ in range(group_count))
        for _ in range(BOOTSTRAP_REPLICATES)
    )


def _bootstrap_interval(
    group_values: Mapping[str, Fraction], samples: Sequence[Sequence[int]]
) -> dict[str, Fraction]:
    group_ids = sorted(group_values)
    values = [
        sum((group_values[group_ids[index]] for index in sample), Fraction()) / len(sample)
        for sample in samples
    ]
    lower = _type7(values, Fraction(1, 40))
    upper = _type7(values, Fraction(39, 40))
    return {"lower": lower, "upper": upper}


def _interval_report(interval: Mapping[str, Fraction]) -> dict[str, float]:
    return {name: float(interval[name]) for name in ("lower", "upper")}


def _float(value: Fraction) -> float:
    return float(value.numerator / value.denominator)


def _exact(value: Fraction) -> dict[str, object]:
    return {"fraction": f"{value.numerator}/{value.denominator}", "value": _float(value)}


def _observed_rows(
    outputs: Mapping[str, Mapping[str, object]],
    panel: Sequence[Mapping[str, object]],
    records: Sequence[DecisionRecord],
) -> dict[str, list[_Observed]]:
    record_by_id = {record.record_id: record for record in records}
    rows: dict[str, list[_Observed]] = defaultdict(list)
    for presentation in panel:
        presentation_id = cast(str, presentation["presentation_id"])
        output = outputs[presentation_id]
        record = record_by_id[cast(str, presentation["record_id"])]
        winner = cast(str, output["winner_option_id"])
        logits = cast(list[float], output["candidate_logits"])
        tied = sum(score == max(logits) for score in logits) > 1
        rows[record.dataset_id].append(_Observed(output, record, winner == record.answer_id, tied))
    return rows


def _fraction_by_group(rows: Sequence[_Observed]) -> dict[str, Fraction]:
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for observed in rows:
        group = observed.record.source_group_id
        total[group] += 1
        correct[group] += observed.correct
    if not total:
        raise ValueError("cannot calculate group accuracy for an empty dataset")
    return {group: _fraction(correct[group], total[group]) for group in sorted(total)}


def _accuracy_summary(rows: Sequence[_Observed]) -> dict[str, object]:
    correct = sum(row.correct for row in rows)
    groups = _fraction_by_group(rows)
    by_record: dict[str, list[bool]] = defaultdict(list)
    predictions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_record[row.record.record_id].append(row.correct)
        predictions[row.record.record_id].add(cast(str, row.row["winner_option_id"]))
    record_means = [Fraction(sum(values), len(values)) for values in by_record.values()]
    group_mean = sum(groups.values(), Fraction()) / len(groups)
    return {
        "correct": correct,
        "presentation_count": len(rows),
        "accuracy": _float(_fraction(correct, len(rows))),
        "source_group_mean_accuracy": _float(group_mean),
        "record_equal_weight_accuracy": _float(sum(record_means, Fraction()) / len(record_means)),
        "source_group_count": len(groups),
        "tie_count": sum(row.tied for row in rows),
        "semantic_flip_record_count": sum(len(values) > 1 for values in predictions.values()),
        "all_selected_orders_correct_record_count": sum(
            bool(values) and all(values) for values in by_record.values()
        ),
    }


def _confusion(rows: Sequence[_Observed]) -> dict[str, object]:
    labels = sorted({option.id for row in rows for option in row.record.request.options})
    matrix = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    for row in rows:
        prediction = cast(str, row.row["winner_option_id"])
        matrix[row.record.answer_id][prediction] += 1
    return {"labels": labels, "matrix": matrix}


def _metrics(
    observed: Mapping[str, Sequence[_Observed]],
) -> tuple[dict[str, object], dict[str, dict[str, Fraction]], dict[str, dict[str, Fraction]]]:
    report: dict[str, object] = {}
    selected_groups: dict[str, dict[str, Fraction]] = {}
    original_groups: dict[str, dict[str, Fraction]] = {}
    for dataset_id in _DEV_DATASET_IDS:
        rows = observed.get(dataset_id, ())
        selected = list(rows)
        original = [row for row in selected if row.row["order_index"] == 0]
        if not selected or not original:
            raise ValueError(
                f"development dataset is missing selected or original rows: {dataset_id}"
            )
        report[dataset_id] = {
            "selected_order": _accuracy_summary(selected),
            "original_order": _accuracy_summary(original),
            "original_order_confusion": _confusion(original) if "snli-" in dataset_id else None,
        }
        selected_groups[dataset_id] = _fraction_by_group(selected)
        original_groups[dataset_id] = _fraction_by_group(original)
    return report, selected_groups, original_groups


def _synthetic_metrics(observed: Mapping[str, Sequence[_Observed]]) -> dict[str, object]:
    families: dict[str, object] = {}
    for dataset_id in _DEV_DATASET_IDS:
        if not dataset_id.startswith("synthetic-"):
            continue
        rows = list(observed[dataset_id])
        family = dataset_id.removeprefix("synthetic-").removesuffix("-v1-development")
        by_options: dict[int, list[_Observed]] = defaultdict(list)
        for row in rows:
            by_options[len(cast(list[object], row.row["order_ids"]))].append(row)
        families[family] = {
            "all_selected_orders": _accuracy_summary(rows),
            "by_option_count": {
                str(option_count): {
                    "order_protocol": "all_permutations"
                    if option_count <= 3
                    else "fixed_rotations",
                    **_accuracy_summary(group),
                }
                for option_count, group in sorted(by_options.items())
            },
        }
    return families


def _state_analysis(
    name: str,
    outputs: Mapping[str, Mapping[str, object]],
    panel: Sequence[Mapping[str, object]],
    records: Sequence[DecisionRecord],
    revision: str,
) -> tuple[dict[str, object], dict[str, dict[str, Fraction]], dict[str, dict[str, Fraction]]]:
    from experiments.mixture_training_calibration import _calibration_fit, _confidence_metrics

    observed = _observed_rows(outputs, panel, records)
    per_dataset, selected, original = _metrics(observed)
    fit, fit_report = _calibration_fit(outputs, records)
    calibration_report = {
        "fit": fit_report,
        "development_original_order": {
            "temperature_1": _confidence_metrics(observed, 1.0, revision),
            "fitted": _confidence_metrics(observed, fit.temperature, revision),
        },
    }
    return (
        {
            "per_dataset": per_dataset,
            "synthetic": _synthetic_metrics(observed),
            "calibration": calibration_report,
        },
        selected,
        original,
    )


def _gate(
    *,
    value: Fraction,
    threshold: Fraction,
    comparison: str,
    interval: Mapping[str, Fraction] | None = None,
    require_positive_lower: bool = False,
) -> dict[str, object]:
    passed = passes_minimum(value, threshold)
    if require_positive_lower:
        passed = passed and interval is not None and interval["lower"] > 0.0
    return {
        "comparison": comparison,
        "value": _exact(value),
        "minimum": _exact(threshold),
        "passed": passed,
        "paired_bootstrap_95": _interval_report(interval) if interval is not None else None,
    }


def _gate_report(
    selected: Mapping[str, Mapping[str, Mapping[str, Fraction]]],
    original: Mapping[str, Mapping[str, Mapping[str, Fraction]]],
    historical_reference: Mapping[str, Mapping[str, Fraction]],
) -> tuple[dict[str, object], dict[str, object]]:
    rng = random.Random(BOOTSTRAP_SEED)
    samples: dict[str, tuple[tuple[int, ...], ...]] = {}
    expected_states = {"base", "real_only", "synthetic_mix"}
    for dataset_id in sorted(selected):
        states = selected[dataset_id]
        if set(states) != expected_states:
            raise ValueError(f"paired model states are incomplete for {dataset_id}")
        group_ids = set(states["base"])
        if not group_ids or any(set(values) != group_ids for values in states.values()):
            raise ValueError(
                f"paired model states must contain the same source groups for {dataset_id}"
            )
        samples[dataset_id] = _bootstrap_samples(len(group_ids), rng)

    historical = (
        ("dbpedia_original_anchor", "dbpedia14-pilot-v1-development", Fraction(50, 56)),
        ("sms_original_anchor", "sms-pilot-v1-development", Fraction(58, 60)),
        ("original_snli_anchor", "snli-pilot-v1-development", Fraction(57, 128)),
    )
    for _name, dataset_id, _anchor in historical:
        states = original.get(dataset_id)
        if dataset_id not in samples or not isinstance(states, Mapping):
            raise ValueError(f"historical anchor is missing original-order groups: {dataset_id}")
        if set(states) != expected_states:
            raise ValueError(f"historical anchor model states are incomplete for {dataset_id}")
        selected_groups = set(selected[dataset_id]["base"])
        if any(set(values) != selected_groups for values in states.values()):
            raise ValueError(
                "selected and original-order states must contain the same source groups "
                f"for {dataset_id}"
            )
        reference_groups = historical_reference.get(dataset_id)
        if not isinstance(reference_groups, Mapping) or set(reference_groups) != selected_groups:
            raise ValueError(
                "historical reference and mixture outputs must contain the same source groups "
                f"for {dataset_id}"
            )

    components: dict[str, object] = {}
    intervals: dict[str, dict[str, Fraction]] = {}
    comparisons = (
        (
            "balanced_snli_mix_minus_real_only",
            "snli-balanced-v1-development",
            selected,
            "synthetic_mix",
            "real_only",
        ),
        (
            "balanced_snli_mix_minus_base",
            "snli-balanced-v1-development",
            selected,
            "synthetic_mix",
            "base",
        ),
        (
            "dbpedia_selected_mix_minus_real_only",
            "dbpedia14-pilot-v1-development",
            selected,
            "synthetic_mix",
            "real_only",
        ),
        (
            "sms_selected_mix_minus_real_only",
            "sms-pilot-v1-development",
            selected,
            "synthetic_mix",
            "real_only",
        ),
    )
    for name, dataset_id, states, final_name, comparison_name in comparisons:
        delta = paired_group_delta(
            states[dataset_id][final_name], states[dataset_id][comparison_name]
        )
        interval = _bootstrap_interval(delta, samples[dataset_id])
        intervals[name] = interval
        value = sum(delta.values(), Fraction()) / len(delta)
        threshold = (
            Fraction(1, 20) if name == "balanced_snli_mix_minus_real_only" else Fraction(-1, 20)
        )
        components[name] = _gate(
            value=value,
            threshold=threshold,
            comparison=f"{final_name} - {comparison_name}",
            interval=interval,
            require_positive_lower=name == "balanced_snli_mix_minus_real_only",
        )

    for name, dataset_id, anchor in historical:
        final_groups = original[dataset_id]["synthetic_mix"]
        reference_groups = historical_reference[dataset_id]
        paired_delta = paired_group_delta(final_groups, reference_groups)
        interval = _bootstrap_interval(paired_delta, samples[dataset_id])
        intervals[name] = interval
        value = sum(final_groups.values(), Fraction()) / len(final_groups) - anchor
        components[name] = _gate(
            value=value,
            threshold=Fraction(-1, 20),
            comparison=f"synthetic_mix original-order accuracy - historical anchor {anchor}",
            interval=interval,
        )
    all_passed = all(
        cast(Mapping[str, object], component)["passed"] for component in components.values()
    )
    report = {
        "evaluated": True,
        "components": components,
        "overall_passed": all_passed,
        "interpretation": "predeclared engineering gates; not a significance test",
    }
    return report, {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "sampling_unit": "source_group_id within each dataset",
        "percentile_interpolation": "type-7 linear at h=(n-1)*p",
        "intervals": {name: _interval_report(interval) for name, interval in intervals.items()},
        "conditional_on_training_seed": True,
    }
