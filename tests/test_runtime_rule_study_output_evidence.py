from __future__ import annotations

import hashlib
from copy import deepcopy
from functools import lru_cache
from typing import Any

import pytest

from experiments import runtime_rule_study_contracts as study_contracts
from experiments import runtime_rule_study_output_evidence as output_evidence
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def _presentation_and_spec(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    request = DecisionRequest(
        context=f"Self-authored output-evidence fixture {index}.",
        question=f"Which choice applies to item {index}?",
        options=(Option(id="z", label="Zulu"), Option(id="a", label="Alpha")),
    )
    presentation = {
        "presentation_id": f"presentation-{index}",
        "record_id": f"record-{index}",
        "dataset_id": "fixture-v1",
        "source_group_id": f"group-{index}",
        "request_hash": request.request_hash,
        "order_index": 0,
        "order_ids": [option.id for option in request.options],
        "request": request.model_dump(mode="json"),
    }
    spec = {
        "presentation_id": presentation["presentation_id"],
        "record_id": presentation["record_id"],
        "request_hash": presentation["request_hash"],
        "input_tokens": 17,
        "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
    }
    return presentation, spec


def _output(presentation: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    return {
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
        "candidate_logits": [0.5, 0.5],
        "winner_option_id": "a",
        "input_tokens": spec["input_tokens"],
        "prompt_sha256": spec["prompt_sha256"],
    }


@lru_cache(maxsize=1)
def _reload_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    presentations_and_specs = [_presentation_and_spec(index) for index in range(32)]
    presentations = [presentation for presentation, _spec in presentations_and_specs]
    specs = [spec for _presentation, spec in presentations_and_specs]
    counterparts = [_output(presentation, spec) for presentation, spec in presentations_and_specs]
    evaluation_prefix = [_output(*_presentation_and_spec(10_000 + index)) for index in range(7)]
    unselected = [{"presentation_id": f"unselected-{index}"} for index in range(3_907)]
    return presentations, specs, [*evaluation_prefix, *unselected, *counterparts]


def _reload_state(
    outputs: list[dict[str, Any]],
    *,
    tensor_equal: object = None,
    tensor_sha256: object = None,
    winner_match_count: object = None,
    max_difference: object = None,
) -> dict[str, Any]:
    return {
        "outputs": outputs,
        "tensor_equal": tensor_equal,
        "tensor_sha256": tensor_sha256,
        "winner_match_count": winner_match_count,
        "max_candidate_logit_difference": max_difference,
    }


def test_scored_output_success_is_detached_and_uses_lexicographic_tie_winner() -> None:
    presentation, spec = _presentation_and_spec(0)
    output = _output(presentation, spec)

    normalized = output_evidence.validate_scored_outputs(
        [output],
        presentations=[presentation],
        specs=[spec],
        completed_forwards=1,
        passed=True,
    )

    assert normalized == [output]
    normalized[0]["candidate_logits"][0] = 99.0
    assert output["candidate_logits"] == [0.5, 0.5]


def test_failed_scoring_accepts_only_a_valid_completed_prefix() -> None:
    pairs = [_presentation_and_spec(index) for index in range(3)]
    presentations = [presentation for presentation, _spec in pairs]
    specs = [spec for _presentation, spec in pairs]
    outputs = [_output(presentation, spec) for presentation, spec in pairs[:2]]

    assert (
        output_evidence.validate_scored_outputs(
            outputs,
            presentations=presentations,
            specs=specs,
            completed_forwards=3,
            passed=False,
        )
        == outputs
    )

    with pytest.raises(ValueError, match="completed prefix"):
        output_evidence.validate_scored_outputs(
            outputs[:1],
            presentations=presentations,
            specs=specs,
            completed_forwards=3,
            passed=False,
        )


def test_scored_output_rejects_a_gap_or_numeric_identity_alias() -> None:
    pairs = [_presentation_and_spec(index) for index in range(3)]
    presentations = [presentation for presentation, _spec in pairs]
    specs = [spec for _presentation, spec in pairs]
    gap = [_output(*pairs[0]), _output(*pairs[2])]
    with pytest.raises(ValueError, match="identity differs canonically"):
        output_evidence.validate_scored_outputs(
            gap,
            presentations=presentations,
            specs=specs,
            completed_forwards=2,
            passed=False,
        )

    aliased = _output(*pairs[0])
    aliased["order_index"] = False
    with pytest.raises(ValueError, match="identity differs canonically"):
        output_evidence.validate_scored_outputs(
            [aliased],
            presentations=presentations[:1],
            specs=specs[:1],
            completed_forwards=1,
            passed=False,
        )


@pytest.mark.parametrize(
    ("completed_forwards", "passed", "message"),
    [(True, False, "strict integer"), (1, 1, "strict boolean")],
)
def test_scored_output_counts_and_passed_flag_are_strict(
    completed_forwards: object, passed: object, message: str
) -> None:
    presentation, spec = _presentation_and_spec(0)
    with pytest.raises(ValueError, match=message):
        output_evidence.validate_scored_outputs(
            [],
            presentations=[presentation],
            specs=[spec],
            completed_forwards=completed_forwards,
            passed=passed,
        )


def test_scored_output_checks_spec_identity_even_when_no_row_returned() -> None:
    presentation, spec = _presentation_and_spec(0)
    spec["record_id"] = "wrong-record"

    with pytest.raises(ValueError, match="spec 0 identity differs"):
        output_evidence.validate_scored_outputs(
            [],
            presentations=[presentation],
            specs=[spec],
            completed_forwards=0,
            passed=False,
        )


def test_scored_output_checks_compiled_tokens_and_prompt_hash() -> None:
    presentation, spec = _presentation_and_spec(0)
    output = _output(presentation, spec)
    output["input_tokens"] += 1

    with pytest.raises(ValueError, match="input_tokens differ from its compiled spec"):
        output_evidence.validate_scored_outputs(
            [output],
            presentations=[presentation],
            specs=[spec],
            completed_forwards=1,
            passed=False,
        )

    output = _output(presentation, spec)
    output["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="prompt hash differs from the canonical"):
        output_evidence.validate_scored_outputs(
            [output],
            presentations=[presentation],
            specs=[spec],
            completed_forwards=1,
            passed=False,
        )


def test_scored_output_rejects_extra_fields_and_nonfinite_logits() -> None:
    presentation, spec = _presentation_and_spec(0)
    extra = _output(presentation, spec)
    extra["unreviewed"] = "field"
    with pytest.raises(ValueError, match="unexpected schema"):
        output_evidence.validate_scored_outputs(
            [extra],
            presentations=[presentation],
            specs=[spec],
            completed_forwards=1,
            passed=False,
        )

    nonfinite = _output(presentation, spec)
    nonfinite["candidate_logits"][0] = float("inf")
    with pytest.raises(ValueError):
        output_evidence.validate_scored_outputs(
            [nonfinite],
            presentations=[presentation],
            specs=[spec],
            completed_forwards=1,
            passed=False,
        )


def test_initial_reload_evidence_has_the_exact_empty_schema() -> None:
    expected = {
        "outputs": [],
        "tensor_equal": None,
        "tensor_sha256": None,
        "winner_match_count": None,
        "max_candidate_logit_difference": None,
    }

    assert output_evidence.initial_reload_evidence() == expected
    assert output_evidence.initial_reload_evidence() is not expected


@pytest.mark.parametrize("prefix_length", [0, 7])
def test_untouched_reload_accepts_an_empty_or_failed_prefix_of_final_evaluation(
    prefix_length: int,
) -> None:
    presentations, specs, final_outputs = _reload_fixture()
    partial_final_outputs = final_outputs[:prefix_length]
    initial = output_evidence.initial_reload_evidence()

    normalized = output_evidence.validate_reload_evidence(
        initial,
        presentations=presentations,
        specs=specs,
        final_outputs=partial_final_outputs,
        final_tensor_sha256=None,
        completed_forwards=0,
        passed=False,
    )

    assert normalized == initial
    assert normalized is not initial


def test_passed_reload_matches_final_outputs_by_presentation_id() -> None:
    presentations, specs, final_outputs = _reload_fixture()
    reload_outputs = deepcopy(final_outputs[-32:])
    final_digest = "a" * 64
    evidence = {
        "outputs": reload_outputs,
        "tensor_equal": True,
        "tensor_sha256": final_digest,
        "winner_match_count": 32,
        "max_candidate_logit_difference": 0.0,
    }

    normalized = output_evidence.validate_reload_evidence(
        evidence,
        presentations=presentations,
        specs=specs,
        final_outputs=final_outputs,
        final_tensor_sha256=final_digest,
        completed_forwards=32,
        passed=True,
    )

    assert normalized == evidence
    normalized["outputs"][0]["candidate_logits"][0] = 9.0
    assert evidence["outputs"][0]["candidate_logits"] == [0.5, 0.5]


def test_failed_reload_preserves_adverse_results_and_one_terminal_missing_row() -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    reload_outputs = deepcopy(final_outputs[-32:-30])
    reload_outputs[0]["candidate_logits"] = [1.0, 0.0]
    reload_outputs[0]["winner_option_id"] = "z"
    final_digest = "a" * 64
    evidence = _reload_state(
        reload_outputs,
        tensor_equal=False,
        tensor_sha256="b" * 64,
        winner_match_count=1,
        max_difference=0.5,
    )

    normalized = output_evidence.validate_reload_evidence(
        evidence,
        presentations=presentations,
        specs=specs,
        final_outputs=final_outputs,
        final_tensor_sha256=final_digest,
        completed_forwards=3,
        passed=False,
    )

    assert normalized["tensor_equal"] is False
    assert normalized["tensor_sha256"] == "b" * 64
    assert normalized["winner_match_count"] == 1
    assert normalized["max_candidate_logit_difference"] == 0.5
    assert normalized["outputs"][0]["winner_option_id"] == "z"


@pytest.mark.parametrize("replacement", ["missing", "duplicate"])
def test_reload_requires_unique_final_counterpart_for_every_planned_row(
    replacement: str,
) -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    if replacement == "missing":
        final_outputs[-1] = {"presentation_id": "replacement-not-in-panel"}
        expected_error = "missing a planned reload presentation"
    else:
        final_outputs[-1] = deepcopy(final_outputs[-2])
        expected_error = "presentation IDs must be unique"
    evidence = _reload_state([], tensor_equal=False, tensor_sha256="b" * 64)

    with pytest.raises(ValueError, match=expected_error):
        output_evidence.validate_reload_evidence(
            evidence,
            presentations=presentations,
            specs=specs,
            final_outputs=final_outputs,
            final_tensor_sha256="a" * 64,
            completed_forwards=0,
            passed=False,
        )


def test_reload_observation_requires_final_digest_and_complete_evaluation() -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    evidence = _reload_state([], tensor_equal=False, tensor_sha256="b" * 64)
    with pytest.raises(ValueError, match="valid final training tensor digest"):
        output_evidence.validate_reload_evidence(
            evidence,
            presentations=presentations,
            specs=specs,
            final_outputs=final_outputs,
            final_tensor_sha256=None,
            completed_forwards=0,
            passed=False,
        )

    with pytest.raises(ValueError, match="complete final evaluation outputs"):
        output_evidence.validate_reload_evidence(
            _reload_state([deepcopy(final_outputs[-32])]),
            presentations=presentations,
            specs=specs,
            final_outputs=final_outputs[:7],
            final_tensor_sha256="a" * 64,
            completed_forwards=1,
            passed=False,
        )


def test_reload_rejects_gapped_rows_and_boolean_forward_counts() -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    gapped = [deepcopy(final_outputs[-32]), deepcopy(final_outputs[-30])]
    with pytest.raises(ValueError, match="identity differs canonically"):
        output_evidence.validate_reload_evidence(
            _reload_state(gapped),
            presentations=presentations,
            specs=specs,
            final_outputs=final_outputs,
            final_tensor_sha256="a" * 64,
            completed_forwards=2,
            passed=False,
        )

    with pytest.raises(ValueError, match="strict integer"):
        output_evidence.validate_reload_evidence(
            output_evidence.initial_reload_evidence(),
            presentations=presentations,
            specs=specs,
            final_outputs=[],
            final_tensor_sha256=None,
            completed_forwards=True,
            passed=False,
        )

    with pytest.raises(ValueError, match="strict boolean"):
        output_evidence.validate_reload_evidence(
            output_evidence.initial_reload_evidence(),
            presentations=presentations,
            specs=specs,
            final_outputs=[],
            final_tensor_sha256=None,
            completed_forwards=0,
            passed=1,
        )


def test_reload_schema_summaries_and_tensor_fields_are_strict() -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    single_output = deepcopy(final_outputs[-32:-31])
    invalid_states = (
        (_reload_state(single_output, winner_match_count=False), "strict integer"),
        (_reload_state(single_output, max_difference=True), "finite nonnegative number"),
        (_reload_state([], tensor_equal=1, tensor_sha256="b" * 64), "strict boolean"),
        (_reload_state([], tensor_equal=False), "must appear together"),
    )
    for evidence, error in invalid_states:
        with pytest.raises(ValueError, match=error):
            output_evidence.validate_reload_evidence(
                evidence,
                presentations=presentations,
                specs=specs,
                final_outputs=final_outputs,
                final_tensor_sha256="a" * 64,
                completed_forwards=1 if evidence["outputs"] else 0,
                passed=False,
            )


def test_reload_requires_the_exact_32_row_plan_and_valid_supplied_digest() -> None:
    presentations, specs, _final_outputs = deepcopy(_reload_fixture())
    with pytest.raises(ValueError, match="fixed 32-row plan"):
        output_evidence.validate_reload_evidence(
            output_evidence.initial_reload_evidence(),
            presentations=presentations[:-1],
            specs=specs[:-1],
            final_outputs=[],
            final_tensor_sha256=None,
            completed_forwards=0,
            passed=False,
        )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        output_evidence.validate_reload_evidence(
            output_evidence.initial_reload_evidence(),
            presentations=presentations,
            specs=specs,
            final_outputs=[],
            final_tensor_sha256="A" * 64,
            completed_forwards=0,
            passed=False,
        )


def test_reload_rejects_duplicate_planned_presentation_identity_before_any_forward() -> None:
    presentations, specs, _final_outputs = deepcopy(_reload_fixture())
    presentations[1]["presentation_id"] = presentations[0]["presentation_id"]
    specs[1]["presentation_id"] = specs[0]["presentation_id"]

    with pytest.raises(ValueError, match="presentation IDs must be unique"):
        output_evidence.validate_reload_evidence(
            output_evidence.initial_reload_evidence(),
            presentations=presentations,
            specs=specs,
            final_outputs=[],
            final_tensor_sha256=None,
            completed_forwards=0,
            passed=False,
        )


def test_reload_rejects_summaries_without_rows_and_summary_drift() -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    fabricated = _reload_state([], winner_match_count=0, max_difference=0.0)
    with pytest.raises(ValueError, match="cannot be present without scored reload outputs"):
        output_evidence.validate_reload_evidence(
            fabricated,
            presentations=presentations,
            specs=specs,
            final_outputs=[],
            final_tensor_sha256=None,
            completed_forwards=0,
            passed=False,
        )

    one_output = deepcopy(final_outputs[-32:-31])
    for evidence in (
        _reload_state(one_output, winner_match_count=0),
        _reload_state(one_output, max_difference=0.25),
    ):
        with pytest.raises(ValueError, match="summary differs from measured outputs"):
            output_evidence.validate_reload_evidence(
                evidence,
                presentations=presentations,
                specs=specs,
                final_outputs=final_outputs,
                final_tensor_sha256="a" * 64,
                completed_forwards=1,
                passed=False,
            )


def test_reload_rejects_logit_subtraction_overflow() -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    final_outputs[-32]["candidate_logits"] = [-1.7e308, 1.7e308]
    reload_output = deepcopy(final_outputs[-32])
    reload_output["candidate_logits"] = [1.7e308, -1.7e308]
    reload_output["winner_option_id"] = "z"

    with pytest.raises(ValueError, match="difference is nonfinite"):
        output_evidence.validate_reload_evidence(
            _reload_state([reload_output]),
            presentations=presentations,
            specs=specs,
            final_outputs=final_outputs,
            final_tensor_sha256="a" * 64,
            completed_forwards=1,
            passed=False,
        )

    with pytest.raises(ValueError, match="finite nonnegative number"):
        output_evidence.validate_reload_evidence(
            _reload_state([deepcopy(final_outputs[-32])], max_difference=10**1000),
            presentations=presentations,
            specs=specs,
            final_outputs=final_outputs,
            final_tensor_sha256="a" * 64,
            completed_forwards=1,
            passed=False,
        )


@pytest.mark.parametrize("difference", [0.001, 0.0011])
def test_reload_pass_enforces_the_exact_logit_difference_limit(difference: float) -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    final_counterparts = final_outputs[-32:]
    reload_outputs = deepcopy(final_counterparts)
    for final_row, reload_row in zip(final_counterparts, reload_outputs, strict=True):
        final_row["candidate_logits"] = [0.0, 1.0]
        reload_row["candidate_logits"] = [difference, 1.0]
        final_row["winner_option_id"] = reload_row["winner_option_id"] = "a"
    evidence = _reload_state(
        reload_outputs,
        tensor_equal=True,
        tensor_sha256="a" * 64,
        winner_match_count=32,
        max_difference=difference,
    )

    if difference > study_contracts.MAX_CANDIDATE_SCORE_DIFFERENCE:
        with pytest.raises(ValueError, match="exceeds the fixed tolerance"):
            output_evidence.validate_reload_evidence(
                evidence,
                presentations=presentations,
                specs=specs,
                final_outputs=final_outputs,
                final_tensor_sha256="a" * 64,
                completed_forwards=32,
                passed=True,
            )
    else:
        result = output_evidence.validate_reload_evidence(
            evidence,
            presentations=presentations,
            specs=specs,
            final_outputs=final_outputs,
            final_tensor_sha256="a" * 64,
            completed_forwards=32,
            passed=True,
        )
        assert result["max_candidate_logit_difference"] == difference


def test_passed_reload_requires_a_changed_final_tensor_digest() -> None:
    presentations, specs, final_outputs = deepcopy(_reload_fixture())
    evidence = _reload_state(
        deepcopy(final_outputs[-32:]),
        tensor_equal=True,
        tensor_sha256=study_contracts.SELECTED_TENSOR_SHA256,
        winner_match_count=32,
        max_difference=0.0,
    )

    with pytest.raises(ValueError, match="differ from the selected starting tensor"):
        output_evidence.validate_reload_evidence(
            evidence,
            presentations=presentations,
            specs=specs,
            final_outputs=final_outputs,
            final_tensor_sha256=study_contracts.SELECTED_TENSOR_SHA256,
            completed_forwards=32,
            passed=True,
        )
