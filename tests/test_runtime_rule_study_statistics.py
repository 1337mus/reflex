from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from fractions import Fraction
from typing import cast

import pytest

from experiments import runtime_rule_study_data, runtime_rule_study_inputs
from experiments.runtime_rule_study_inputs import StudyInputs
from experiments.runtime_rule_study_statistics import (
    _bootstrap_interval,
    _bootstrap_samples,
    _gate_component,
    _group_means,
    _passes_threshold,
    _type7_fraction,
    _validated_state_rows,
    _validated_states,
    analyze_study_outputs,
)
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def test_exact_gate_threshold_includes_equality_and_rejects_near_miss() -> None:
    assert _passes_threshold(Fraction(1, 20), Fraction(1, 20))
    assert not _passes_threshold(Fraction(49, 1_000), Fraction(1, 20))
    assert _passes_threshold(Fraction(-1, 20), Fraction(-1, 20))
    assert not _passes_threshold(Fraction(-51, 1_000), Fraction(-1, 20))


def test_group_means_weight_records_and_groups_equally() -> None:
    observations = (
        *(("group-a", "record-a", True) for _ in range(4)),
        ("group-a", "record-b", False),
        ("group-b", "record-c", True),
    )

    group_means, record_count = _group_means(observations)

    assert group_means == {"group-a": Fraction(1, 2), "group-b": Fraction(1)}
    assert record_count == 3
    assert sum(group_means.values(), Fraction()) / len(group_means) == Fraction(3, 4)


def test_type7_percentile_linearly_interpolates_adjacent_draws() -> None:
    exact_draws = (Fraction(0), Fraction(1), Fraction(2), Fraction(3))
    assert _type7_fraction(exact_draws, Fraction(1, 4)) == Fraction(3, 4)


def test_zero_bootstrap_lower_bound_stays_zero_for_strict_positive_gate() -> None:
    deltas = {
        "group-a": Fraction(-1),
        "group-b": Fraction(1, 6),
        "group-c": Fraction(5, 6),
    }

    interval = _bootstrap_interval(deltas, ((0, 1, 2),))
    component = _gate_component(Fraction(), Fraction(), interval, require_positive_lower=True)

    assert interval.lower == 0
    assert interval.report()["lower"] == 0.0
    assert component["passed"] is False


def test_bootstrap_samples_consume_one_rng_in_task_order() -> None:
    rng = random.Random(20266010)
    actual_first = _bootstrap_samples(2, rng, draws=3)
    actual_second = _bootstrap_samples(3, rng, draws=3)
    reference = random.Random(20266010)
    expected_first = tuple(tuple(reference.randrange(2) for _ in range(2)) for _ in range(3))
    expected_second = tuple(tuple(reference.randrange(3) for _ in range(3)) for _ in range(3))

    assert actual_first == expected_first
    assert actual_second == expected_second


def _toy_study_inputs() -> StudyInputs:
    from test_runtime_rule_study_data import _development_pools
    from test_runtime_rule_study_inputs import _retention_records, _transfer_records

    real, balanced, synthetic = _retention_records()
    transfer = _transfer_records()
    retention = runtime_rule_study_inputs.build_retention_presentations(
        real, balanced, synthetic, transfer
    )
    development = _development_pools()
    new = runtime_rule_study_data.build_new_evaluation_presentations(*development)
    records_by_id = {
        record.record_id: record
        for record in (*real, *balanced, *synthetic, *transfer, *development[0], *development[1])
    }
    panel_ids = dict.fromkeys(str(row["record_id"]) for row in (*retention, *new))
    return StudyInputs(
        training_pools=(),
        development_pools=development,
        evaluation_records=tuple(records_by_id[record_id] for record_id in panel_ids),
        retention_presentations=retention,
        new_presentations=new,
        selection={},
        file_sha256={},
    )


def _toy_outputs(inputs: StudyInputs) -> dict[str, list[dict[str, object]]]:
    panel = (*inputs.retention_presentations, *inputs.new_presentations)
    records = {record.record_id: record for record in inputs.evaluation_records}
    states = ("unchanged", "continued_practice", "runtime_mix")
    output: dict[str, list[dict[str, object]]] = {}
    for state in states:
        state_rows = []
        for presentation in panel:
            record = records[str(presentation["record_id"])]
            task = str(presentation["dataset_id"])
            index = int(record.record_id.rsplit("-", 1)[-1])
            if task in {"routing-data-v1-development", "tool-data-v1-development"}:
                correct = state == "runtime_mix" or (
                    state == "continued_practice" and index % 2 == 0
                )
                if state == "unchanged":
                    correct = index % 3 == 0
            else:
                correct = True
            order_ids = cast(list[str], presentation["order_ids"])
            winner = (
                record.answer_id
                if correct
                else next(option_id for option_id in order_ids if option_id != record.answer_id)
            )
            request = DecisionRequest.model_validate(presentation["request"])
            state_rows.append(
                {
                    "presentation_id": presentation["presentation_id"],
                    "record_id": presentation["record_id"],
                    "dataset_id": presentation["dataset_id"],
                    "source_group_id": presentation["source_group_id"],
                    "request_hash": presentation["request_hash"],
                    "order_index": presentation["order_index"],
                    "order_ids": list(order_ids),
                    "candidate_logits": [
                        1.0 if option_id == winner else 0.0 for option_id in order_ids
                    ],
                    "winner_option_id": winner,
                    "input_tokens": 1,
                    "prompt_sha256": hashlib.sha256(render_prompt(request).encode()).hexdigest(),
                }
            )
        output[state] = state_rows
    return output


def _single_output() -> tuple[dict[str, object], dict[str, object]]:
    options = (Option(id="z", label="Zulu"), Option(id="a", label="Alpha"))
    request = DecisionRequest(context="fixture", question="Choose", options=options)
    presentation: dict[str, object] = {
        "presentation_id": "presentation-one",
        "record_id": "record-one",
        "dataset_id": "routing-data-v1-development",
        "source_group_id": "group-one",
        "request_hash": request.request_hash,
        "order_index": 1,
        "order_ids": ["z", "a"],
        "request": request.model_dump(mode="json"),
    }
    output = {
        **{
            key: presentation[key]
            for key in (
                "presentation_id",
                "record_id",
                "dataset_id",
                "source_group_id",
                "request_hash",
                "order_index",
                "order_ids",
            )
        },
        "candidate_logits": [0.0, 1.0],
        "winner_option_id": "a",
        "input_tokens": 1,
        "prompt_sha256": hashlib.sha256(render_prompt(request).encode()).hexdigest(),
    }
    return presentation, output


def test_analyze_study_outputs_reports_fixed_toy_gates_and_intervals() -> None:
    inputs = _toy_study_inputs()
    outputs = _toy_outputs(inputs)

    report = analyze_study_outputs(inputs, outputs)

    task_order = (
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
    assert report["task_order"] == list(task_order)
    assert set(report["states"]) == {"unchanged", "continued_practice", "runtime_mix"}
    runtime_routing = report["states"]["runtime_mix"]["tasks"][task_order[0]]
    assert runtime_routing["raw_correct"] == runtime_routing["raw_presentations"] == 82
    assert runtime_routing["original_order_correct"] == runtime_routing["record_count"] == 14
    assert runtime_routing["source_group_count"] == 14
    assert runtime_routing["accuracy_fraction"] == {"numerator": 1, "denominator": 1}
    assert report["point_deltas"]["runtime_mix_minus_continued_practice"]["new_family_macro"][
        "fraction"
    ] == {"numerator": 1, "denominator": 2}
    assert report["point_deltas"]["runtime_mix_minus_unchanged"]["new_family_macro"][
        "fraction"
    ] == {"numerator": 9, "denominator": 14}
    assert len(report["gates"]["components"]) == 21
    assert report["gates"]["overall_passed"] is True
    for contrast in (
        "runtime_mix_minus_continued_practice",
        "runtime_mix_minus_unchanged",
    ):
        assert set(report["paired_bootstrap_95"][contrast]) == set(task_order) | {
            "new_family_macro"
        }
    assert (
        report["paired_bootstrap_95"]["runtime_mix_minus_continued_practice"]["new_family_macro"][
            "lower"
        ]
        > 0
    )
    assert report["bootstrap"]["replicates"] == 2_000
    assert report["bootstrap"]["seed"] == 20_266_010
    assert len(report["bootstrap"]["resample_bank_sha256"]) == 64


def test_analyzer_rejects_duplicate_source_record_ids() -> None:
    inputs = _toy_study_inputs()
    duplicate = replace(
        inputs,
        evaluation_records=(*inputs.evaluation_records, inputs.evaluation_records[0]),
    )

    with pytest.raises(ValueError, match="duplicate record IDs"):
        analyze_study_outputs(duplicate, None)


def test_analyzer_rejects_source_records_outside_panel_membership() -> None:
    inputs = _toy_study_inputs()
    extra = inputs.evaluation_records[0].model_copy(update={"record_id": "unmatched-source-record"})
    unmatched = replace(inputs, evaluation_records=(*inputs.evaluation_records, extra))

    with pytest.raises(ValueError, match="do not match final panel membership"):
        analyze_study_outputs(unmatched, None)


@pytest.mark.parametrize("bad_index", (True, 1.0))
def test_output_validation_rejects_boolean_and_float_order_indexes(bad_index: object) -> None:
    presentation, valid_output = _single_output()
    assert _validated_state_rows([valid_output], [presentation], state="runtime_mix")
    malformed = {**valid_output, "order_index": bad_index}

    with pytest.raises(ValueError, match="order_index must be an integer"):
        _validated_state_rows([malformed], [presentation], state="runtime_mix")


def test_output_validation_requires_complete_ordered_unique_rows() -> None:
    presentation, output = _single_output()
    second_presentation = {**presentation, "presentation_id": "presentation-two", "order_index": 0}
    second_output = {**output, "presentation_id": "presentation-two", "order_index": 0}

    with pytest.raises(ValueError, match="every expected presentation"):
        _validated_state_rows([output], [presentation, second_presentation], state="runtime_mix")
    with pytest.raises(ValueError, match="identities and order"):
        _validated_state_rows(
            [output, output], [presentation, second_presentation], state="runtime_mix"
        )
    assert _validated_state_rows(
        [output, second_output], [presentation, second_presentation], state="runtime_mix"
    ) == [output, second_output]


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ({"candidate_logits": [1.0, float("inf")]}, "strict JSON"),
        ({"candidate_logits": [1.0, 1.0], "winner_option_id": "z"}, "lexicographic tie rule"),
    ),
)
def test_output_validation_rejects_nonfinite_scores_and_wrong_tie_winner(
    change: dict[str, object], reason: str
) -> None:
    presentation, output = _single_output()
    malformed = {**output, **change}

    with pytest.raises(ValueError, match=reason):
        _validated_state_rows([malformed], [presentation], state="runtime_mix")


def test_state_validation_requires_exact_three_complete_states() -> None:
    presentation, output = _single_output()
    valid = {state: [output] for state in ("unchanged", "continued_practice", "runtime_mix")}
    assert set(_validated_states(valid, [presentation])) == set(valid)

    with pytest.raises(ValueError, match="exactly unchanged, continued_practice, and runtime_mix"):
        _validated_states(
            {key: value for key, value in valid.items() if key != "unchanged"}, [presentation]
        )
    with pytest.raises(ValueError, match="exactly unchanged, continued_practice, and runtime_mix"):
        _validated_states({**valid, "base": [output]}, [presentation])
