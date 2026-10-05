"""Fixed CPU statistics and gates for the runtime-rule study."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

from experiments import runtime_rule_study_data as study
from experiments.mixture_training_outputs import _validate_output_rows
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest

if TYPE_CHECKING:
    from experiments.runtime_rule_study_inputs import StudyInputs

STATE_NAMES = ("unchanged", "continued_practice", "runtime_mix")
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20266010
TASK_ORDER = (
    "routing-data-v1-development",
    "tool-data-v1-development",
    "dbpedia14-pilot-v1-development",
    "sms-pilot-v1-development",
    "snli-balanced-v1-development",
    "synthetic-atomic-fact-inference-v1-development",
    "synthetic-numeric-selection-v1-development",
    "boolq-dev-pilot-v1",
    "copa-dev-pilot-v1",
)
EXPECTED_TASK_COUNTS = {
    "routing-data-v1-development": 82,
    "tool-data-v1-development": 82,
    "dbpedia14-pilot-v1-development": 1_568,
    "sms-pilot-v1-development": 120,
    "snli-balanced-v1-development": 1_152,
    "synthetic-atomic-fact-inference-v1-development": 450,
    "synthetic-numeric-selection-v1-development": 364,
    "boolq-dev-pilot-v1": 64,
    "copa-dev-pilot-v1": 64,
}
RETENTION_TASKS = TASK_ORDER[2:]
NEW_TASKS = TASK_ORDER[:2]
PRESENTATION_FIELDS = study.PRESENTATION_FIELDS
REQUEST_FIELDS = {"context", "question", "options"}


@dataclass(frozen=True, slots=True)
class _TaskSummary:
    raw_correct: int
    raw_presentations: int
    original_correct: int
    original_presentations: int
    record_count: int
    group_means: dict[str, Fraction]

    @property
    def accuracy(self) -> Fraction:
        return sum(self.group_means.values(), Fraction()) / len(self.group_means)


@dataclass(frozen=True, slots=True)
class _BootstrapInterval:
    lower: Fraction
    upper: Fraction

    def report(self) -> dict[str, float]:
        return {"lower": float(self.lower), "upper": float(self.upper)}


def _passes_threshold(value: Fraction, minimum: Fraction) -> bool:
    if not isinstance(value, Fraction) or not isinstance(minimum, Fraction):
        raise TypeError("point values and thresholds must be exact Fraction values")
    return value >= minimum


def _group_means(
    observations: Sequence[tuple[str, str, bool]],
) -> tuple[dict[str, Fraction], int]:
    by_group: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for group_id, record_id, correct in observations:
        if not group_id or not record_id or type(correct) is not bool:
            raise ValueError("group accuracy observations have invalid identities or correctness")
        by_group[group_id][record_id].append(correct)
    if not by_group:
        raise ValueError("group accuracy requires at least one observation")
    means = {
        group_id: sum(
            (Fraction(sum(values), len(values)) for values in record_orders.values()), Fraction()
        )
        / len(record_orders)
        for group_id, record_orders in sorted(by_group.items())
    }
    return means, sum(len(records) for records in by_group.values())


def _type7_fraction(values: Sequence[Fraction], probability: Fraction) -> Fraction:
    if (
        not values
        or not isinstance(probability, Fraction)
        or not 0 <= probability <= 1
        or any(not isinstance(value, Fraction) for value in values)
    ):
        raise ValueError("exact type-7 percentile input is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = position.numerator // position.denominator
    upper = -(-position.numerator // position.denominator)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _bootstrap_samples(
    group_count: int, rng: random.Random, *, draws: int = 2_000
) -> tuple[tuple[int, ...], ...]:
    if type(group_count) is not int or group_count < 1:
        raise ValueError("bootstrap requires a positive source-group count")
    if type(draws) is not int or draws < 1 or not isinstance(rng, random.Random):
        raise ValueError("bootstrap draw count or random generator is invalid")
    return tuple(
        tuple(rng.randrange(group_count) for _ in range(group_count)) for _ in range(draws)
    )


def _fraction_object(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _point_report(value: Fraction) -> dict[str, object]:
    return {"fraction": _fraction_object(value), "value": float(value)}


def _study_panel(
    inputs: StudyInputs,
) -> tuple[list[Mapping[str, object]], dict[str, DecisionRecord]]:
    from experiments.runtime_rule_study_inputs import StudyInputs

    if not isinstance(inputs, StudyInputs):
        raise TypeError("inputs must be a validated StudyInputs value")
    source_records = inputs.evaluation_records
    if not isinstance(source_records, Sequence) or isinstance(source_records, (str, bytes)):
        raise ValueError("evaluation records must be a sequence")
    records_by_id: dict[str, DecisionRecord] = {}
    for record in source_records:
        if not isinstance(record, DecisionRecord):
            raise ValueError("evaluation records must contain DecisionRecord values")
        if record.record_id in records_by_id:
            raise ValueError("evaluation source records contain duplicate record IDs")
        records_by_id[record.record_id] = record

    if len(inputs.development_pools) != 2:
        raise ValueError("StudyInputs must contain the routing and tool development pools")
    routing, tool = inputs.development_pools
    new_rows = study.build_new_evaluation_presentations(routing, tool)
    study.audit_new_evaluation_presentations(inputs.new_presentations, routing, tool)
    retained = inputs.retention_presentations
    if not isinstance(retained, Sequence) or isinstance(retained, (str, bytes)):
        raise ValueError("retention presentations must be a sequence")
    if len(retained) != 3_782:
        raise ValueError("retention panel must contain exactly 3782 presentations")
    panel: list[Mapping[str, object]] = [*retained, *new_rows]
    if len(panel) != 3_946:
        raise ValueError("runtime-rule evaluation panel must contain exactly 3946 rows")

    presentation_ids: set[str] = set()
    record_orders: set[tuple[str, int]] = set()
    panel_record_ids: set[str] = set()
    counts: Counter[str] = Counter()
    for row in panel:
        if not isinstance(row, Mapping) or set(row) != PRESENTATION_FIELDS:
            raise ValueError("evaluation presentation has an unexpected request-only schema")
        if type(row["order_index"]) is not int:
            raise ValueError("evaluation presentation order_index must be an integer")
        string_fields = (
            "presentation_id",
            "record_id",
            "dataset_id",
            "source_group_id",
            "request_hash",
        )
        if any(not isinstance(row[field], str) for field in string_fields):
            raise ValueError("evaluation presentation identity fields must be strings")
        presentation_id = cast(str, row["presentation_id"])
        record_id = cast(str, row["record_id"])
        dataset_id = cast(str, row["dataset_id"])
        order_index = row["order_index"]
        if presentation_id in presentation_ids:
            raise ValueError("evaluation presentations contain duplicate presentation IDs")
        presentation_ids.add(presentation_id)
        if (record_id, order_index) in record_orders:
            raise ValueError("evaluation presentations duplicate a record order index")
        record_orders.add((record_id, order_index))
        panel_record_ids.add(record_id)
        counts[dataset_id] += 1
        panel_record = records_by_id.get(record_id)
        if panel_record is None or dataset_id != panel_record.dataset_id:
            raise ValueError("evaluation panel references a substituted or unmatched record")
        order_ids = row["order_ids"]
        if not isinstance(order_ids, list) or any(
            not isinstance(value, str) for value in order_ids
        ):
            raise ValueError("evaluation presentation order_ids must be a list of strings")
        option_by_id = {option.id: option for option in panel_record.request.options}
        if len(order_ids) != len(option_by_id) or set(order_ids) != set(option_by_id):
            raise ValueError("evaluation presentation order differs from its source record")
        request_value = row["request"]
        if not isinstance(request_value, Mapping) or set(request_value) != REQUEST_FIELDS:
            raise ValueError("evaluation presentation request must contain only request fields")
        request = DecisionRequest.model_validate(request_value)
        expected = panel_record.request.model_copy(
            update={"options": tuple(option_by_id[option_id] for option_id in order_ids)}
        )
        if (
            request != expected
            or row["request_hash"] != request.request_hash
            or order_ids != [option.id for option in request.options]
        ):
            raise ValueError(
                "evaluation presentation request or option order differs from its source"
            )
    if panel_record_ids != set(records_by_id):
        raise ValueError("evaluation records do not match final panel membership")
    if counts != Counter(EXPECTED_TASK_COUNTS):
        raise ValueError("evaluation task counts differ from the fixed nine-task panel")
    return panel, records_by_id


def _validated_state_rows(
    rows: object, presentations: Sequence[Mapping[str, object]], *, state: str
) -> list[dict[str, object]]:
    for presentation in presentations:
        if type(presentation.get("order_index")) is not int:
            raise ValueError("presentation order_index must be an integer")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping) and type(row.get("order_index")) is not int:
                raise ValueError(f"outputs.{state} order_index must be an integer")
    return _validate_output_rows(
        rows, presentations, label=f"outputs.{state}", require_complete=True
    )


def _validated_states(
    outputs_by_state: object, presentations: Sequence[Mapping[str, object]]
) -> dict[str, list[dict[str, object]]]:
    if not isinstance(outputs_by_state, Mapping) or set(outputs_by_state) != set(STATE_NAMES):
        raise ValueError(
            "outputs_by_state must contain exactly unchanged, continued_practice, and runtime_mix"
        )
    return {
        state: _validated_state_rows(outputs_by_state[state], presentations, state=state)
        for state in STATE_NAMES
    }


def _summarize_state(
    outputs: Sequence[Mapping[str, object]],
    panel: Sequence[Mapping[str, object]],
    records_by_id: Mapping[str, DecisionRecord],
) -> dict[str, _TaskSummary]:
    observations: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)
    raw_correct: Counter[str] = Counter()
    raw_total: Counter[str] = Counter()
    original_correct: Counter[str] = Counter()
    original_total: Counter[str] = Counter()
    for presentation, output in zip(panel, outputs, strict=True):
        record = records_by_id[cast(str, presentation["record_id"])]
        task = record.dataset_id
        correct = output["winner_option_id"] == record.answer_id
        observations[task].append(
            (cast(str, presentation["source_group_id"]), record.record_id, correct)
        )
        raw_correct[task] += correct
        raw_total[task] += 1
        if presentation["order_index"] == 0:
            original_correct[task] += correct
            original_total[task] += 1
    if set(observations) != set(TASK_ORDER):
        raise ValueError("scored outputs do not cover all nine fixed tasks")
    summaries = {}
    for task in TASK_ORDER:
        group_means, record_count = _group_means(observations[task])
        if not original_total[task]:
            raise ValueError(f"task has no original-order presentations: {task}")
        summaries[task] = _TaskSummary(
            raw_correct[task],
            raw_total[task],
            original_correct[task],
            original_total[task],
            record_count,
            group_means,
        )
    return summaries


def _task_report(summary: _TaskSummary) -> dict[str, object]:
    original_accuracy = Fraction(summary.original_correct, summary.original_presentations)
    raw_accuracy = Fraction(summary.raw_correct, summary.raw_presentations)
    return {
        "raw_correct": summary.raw_correct,
        "raw_presentations": summary.raw_presentations,
        "raw_accuracy_fraction": _fraction_object(raw_accuracy),
        "raw_accuracy": float(raw_accuracy),
        "accuracy_fraction": _fraction_object(summary.accuracy),
        "accuracy": float(summary.accuracy),
        "record_count": summary.record_count,
        "source_group_count": len(summary.group_means),
        "original_order_correct": summary.original_correct,
        "original_order_presentations": summary.original_presentations,
        "original_order_accuracy_fraction": _fraction_object(original_accuracy),
        "original_order_accuracy": float(original_accuracy),
    }


def _group_delta(left: _TaskSummary, right: _TaskSummary) -> dict[str, Fraction]:
    if set(left.group_means) != set(right.group_means):
        raise ValueError("paired states must contain the same source groups")
    return {
        group_id: left.group_means[group_id] - right.group_means[group_id]
        for group_id in sorted(left.group_means)
    }


def _bootstrap_interval(
    group_deltas: Mapping[str, Fraction], samples: Sequence[Sequence[int]]
) -> _BootstrapInterval:
    group_ids = sorted(group_deltas)
    if not group_ids or not samples:
        raise ValueError("paired bootstrap requires groups and samples")
    draws = [
        sum((group_deltas[group_ids[index]] for index in sample), Fraction()) / len(sample)
        for sample in samples
    ]
    return _BootstrapInterval(
        _type7_fraction(draws, Fraction(1, 40)),
        _type7_fraction(draws, Fraction(39, 40)),
    )


def _bootstrap_macro_interval(
    group_deltas: Sequence[Mapping[str, Fraction]],
    samples_by_task: Sequence[Sequence[Sequence[int]]],
) -> _BootstrapInterval:
    if len(group_deltas) != 2 or len(samples_by_task) != 2:
        raise ValueError("new-family bootstrap requires routing and tool group samples")
    task_values = [[delta[group_id] for group_id in sorted(delta)] for delta in group_deltas]
    draws = []
    for routing_sample, tool_sample in zip(*samples_by_task, strict=True):
        routing = sum((task_values[0][index] for index in routing_sample), Fraction()) / len(
            routing_sample
        )
        tool = sum((task_values[1][index] for index in tool_sample), Fraction()) / len(tool_sample)
        draws.append((routing + tool) / 2)
    return _BootstrapInterval(
        _type7_fraction(draws, Fraction(1, 40)),
        _type7_fraction(draws, Fraction(39, 40)),
    )


def _point_deltas(
    summaries: Mapping[str, Mapping[str, _TaskSummary]],
) -> dict[str, dict[str, Fraction]]:
    comparisons = {
        "runtime_mix_minus_continued_practice": "continued_practice",
        "runtime_mix_minus_unchanged": "unchanged",
    }
    result = {}
    for comparison, baseline in comparisons.items():
        deltas = {
            task: summaries["runtime_mix"][task].accuracy - summaries[baseline][task].accuracy
            for task in TASK_ORDER
        }
        deltas["new_family_macro"] = (deltas[NEW_TASKS[0]] + deltas[NEW_TASKS[1]]) / 2
        result[comparison] = deltas
    return result


def _bank_digest_update(
    digest: Any, task: str, samples: Sequence[Sequence[int]], first: bool
) -> None:
    hasher = digest
    if first:
        hasher.update(b"{")
    else:
        hasher.update(b",")
    hasher.update(json.dumps(task, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    hasher.update(b":[")
    for sample_index, sample in enumerate(samples):
        if sample_index:
            hasher.update(b",")
        hasher.update(json.dumps(list(sample), separators=(",", ":")).encode("ascii"))
    hasher.update(b"]")


def _gate_component(
    value: Fraction,
    minimum: Fraction,
    interval: _BootstrapInterval | None = None,
    *,
    require_positive_lower: bool = False,
) -> dict[str, object]:
    passed = _passes_threshold(value, minimum)
    if require_positive_lower:
        passed = passed and interval is not None and interval.lower > 0
    return {
        "delta": _point_report(value),
        "minimum": _point_report(minimum),
        "paired_bootstrap_95": interval.report() if interval is not None else None,
        "positive_interval_lower_required": require_positive_lower,
        "passed": passed,
    }


def analyze_study_outputs(inputs: StudyInputs, outputs_by_state: object) -> dict[str, object]:
    panel, records_by_id = _study_panel(inputs)
    validated_outputs = _validated_states(outputs_by_state, panel)
    summaries = {
        state: _summarize_state(validated_outputs[state], panel, records_by_id)
        for state in STATE_NAMES
    }
    point_deltas = _point_deltas(summaries)

    rng = random.Random(BOOTSTRAP_SEED)
    bank_hash = hashlib.sha256()
    samples_by_task: dict[str, tuple[tuple[int, ...], ...]] = {}
    intervals: dict[str, dict[str, _BootstrapInterval]] = {
        "runtime_mix_minus_continued_practice": {},
        "runtime_mix_minus_unchanged": {},
    }
    baselines = {
        "runtime_mix_minus_continued_practice": "continued_practice",
        "runtime_mix_minus_unchanged": "unchanged",
    }
    for task_index, task in enumerate(TASK_ORDER):
        group_count = len(summaries["runtime_mix"][task].group_means)
        samples = _bootstrap_samples(group_count, rng, draws=BOOTSTRAP_REPLICATES)
        _bank_digest_update(bank_hash, task, samples, task_index == 0)
        if task in NEW_TASKS:
            samples_by_task[task] = samples
        for comparison, baseline in baselines.items():
            deltas = _group_delta(summaries["runtime_mix"][task], summaries[baseline][task])
            intervals[comparison][task] = _bootstrap_interval(deltas, samples)
    bank_hash.update(b"}")
    for comparison, baseline in baselines.items():
        family_deltas = [
            _group_delta(summaries["runtime_mix"][task], summaries[baseline][task])
            for task in NEW_TASKS
        ]
        intervals[comparison]["new_family_macro"] = _bootstrap_macro_interval(
            family_deltas, [samples_by_task[task] for task in NEW_TASKS]
        )

    gates: dict[str, dict[str, object]] = {}
    control = "runtime_mix_minus_continued_practice"
    unchanged = "runtime_mix_minus_unchanged"
    gates[f"{control}.new_family_macro_minimum"] = _gate_component(
        point_deltas[control]["new_family_macro"],
        Fraction(1, 20),
        intervals[control]["new_family_macro"],
    )
    gates[f"{control}.new_family_macro_positive_interval"] = _gate_component(
        point_deltas[control]["new_family_macro"],
        Fraction(),
        intervals[control]["new_family_macro"],
        require_positive_lower=True,
    )
    for task in NEW_TASKS:
        gates[f"{control}.{task}_nonnegative"] = _gate_component(
            point_deltas[control][task], Fraction(), intervals[control][task]
        )
    gates[f"{unchanged}.new_family_macro_minimum"] = _gate_component(
        point_deltas[unchanged]["new_family_macro"],
        Fraction(1, 20),
        intervals[unchanged]["new_family_macro"],
    )
    for task in NEW_TASKS:
        gates[f"{unchanged}.{task}_nonnegative"] = _gate_component(
            point_deltas[unchanged][task], Fraction(), intervals[unchanged][task]
        )
    for comparison in (control, unchanged):
        for task in RETENTION_TASKS:
            gates[f"{comparison}.{task}_retention_floor"] = _gate_component(
                point_deltas[comparison][task], Fraction(-1, 20), intervals[comparison][task]
            )
    states_report: dict[str, dict[str, object]] = {}
    for state in STATE_NAMES:
        state_tasks = summaries[state]
        macro = sum((state_tasks[task].accuracy for task in NEW_TASKS), Fraction()) / 2
        states_report[state] = {
            "tasks": {task: _task_report(state_tasks[task]) for task in TASK_ORDER},
            "new_family_macro_accuracy_fraction": _fraction_object(macro),
            "new_family_macro_accuracy": float(macro),
        }
    return {
        "task_order": list(TASK_ORDER),
        "states": states_report,
        "point_deltas": {
            comparison: {task: _point_report(value) for task, value in task_values.items()}
            for comparison, task_values in point_deltas.items()
        },
        "paired_bootstrap_95": {
            comparison: {task: interval.report() for task, interval in task_intervals.items()}
            for comparison, task_intervals in intervals.items()
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "sampling_unit": (
                "source_group_id within task; new family macro pairs routing and tool draws"
            ),
            "percentile_interpolation": "type-7 at h=(n-1)*p, p=0.025 and 0.975",
            "resample_bank_sha256": bank_hash.hexdigest(),
        },
        "gates": {
            "components": gates,
            "overall_passed": all(gate["passed"] for gate in gates.values()),
        },
    }
