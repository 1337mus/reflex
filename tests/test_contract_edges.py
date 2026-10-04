"""Negative contract tests for malformed inputs and stale scoring artifacts."""

import math

import pytest
from pydantic import ValidationError

from reflex_decisions import data, evaluation, permutation, rendering, scoring
from reflex_decisions.schema import DecisionRequest, Option


def _request(
    option_ids: tuple[str, ...] = ("a", "b"), *, context: str = "A request"
) -> DecisionRequest:
    return DecisionRequest(
        context=context,
        question="Choose one",
        options=tuple(
            Option(id=option_id, label=f"Option {option_id}") for option_id in option_ids
        ),
    )


def _manifest() -> data.SplitManifest:
    return data.SplitManifest(
        data_kind="fixture",
        datasets=(
            data.DatasetSpec(
                dataset_id="fixture-test",
                source_id="fixture-source",
                task_family="fixture-family",
                split="test",
                source_uri="local://fixture/test",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )


def _record(request: DecisionRequest | None = None) -> data.DecisionRecord:
    active_request = request or _request()
    return data.DecisionRecord(
        record_id="record-1",
        dataset_id="fixture-test",
        source_group_id="group-1",
        request=active_request,
        answer_id=active_request.options[0].id,
    )


def _evaluate(
    record: data.DecisionRecord,
    predictions: tuple[evaluation.PredictionRecord, ...],
) -> evaluation.EvaluationReport:
    return evaluation.evaluate(
        (record,),
        predictions,
        _manifest(),
        records_sha256="0" * 64,
        predictions_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )


@pytest.mark.parametrize("option_count", [2, 16])
def test_request_accepts_option_count_boundaries_and_is_deeply_frozen(option_count: int) -> None:
    request = _request(tuple(f"option-{index}" for index in range(option_count)))

    assert len(request.options) == option_count
    with pytest.raises(ValidationError):
        request.context = "changed"
    with pytest.raises(ValidationError):
        request.options[0].label = "changed"


@pytest.mark.parametrize("option_count", [0, 1, 17])
def test_request_rejects_option_counts_outside_two_through_sixteen(option_count: int) -> None:
    with pytest.raises(ValidationError):
        _request(tuple(f"option-{index}" for index in range(option_count)))


def test_request_rejects_duplicate_ids_and_blank_required_text() -> None:
    with pytest.raises(ValidationError, match="unique"):
        DecisionRequest(
            context="A request",
            question="Choose one",
            options=(Option(id="same", label="First"), Option(id="same", label="Second")),
        )

    with pytest.raises(ValidationError):
        DecisionRequest(
            context="  ",
            question="Choose one",
            options=(Option(id="a", label="A"), Option(id="b", label="B")),
        )


class _BoundaryRetokenizer:
    """Toy tokenizer that merges the answer suffix when A is appended."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        suffix = "Answer:\nA"
        if text.endswith(suffix):
            return [ord(character) for character in text[: -len(suffix)]] + [50_000]
        return [ord(character) for character in text]


class _DuplicateCandidateTokenizer:
    """Toy tokenizer where distinct candidate symbols receive the same token ID."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text.endswith(("A", "B")):
            return [ord(character) for character in text[:-1]] + [50_000]
        return [ord(character) for character in text]


def test_rendering_rejects_candidate_that_retokenizes_prompt_boundary() -> None:
    with pytest.raises(ValueError, match="retokenizes"):
        rendering.compile_request(_request(), _BoundaryRetokenizer())


def test_rendering_rejects_duplicate_candidate_token_ids() -> None:
    # This toy tokenizer exercises the guard; it does not establish Qwen tokenization behavior.
    with pytest.raises(ValueError, match="distinct token IDs"):
        rendering.compile_request(_request(), _DuplicateCandidateTokenizer())


def test_rendering_rejects_prompt_that_exceeds_token_limit() -> None:
    with pytest.raises(ValueError, match="limit is 1"):
        rendering.compile_request(_request(), _DuplicateCandidateTokenizer(), max_tokens=1)


@pytest.mark.parametrize("logits", [[0.0], [0.0, math.nan], [0.0, math.inf], [0.0, True]])
def test_scoring_rejects_wrong_logit_count_and_nonfinite_or_boolean_scores(logits) -> None:
    with pytest.raises(ValueError):
        scoring.score_candidates(_request(), logits, model_revision="fixture-v1")


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.nan, math.inf, True])
def test_scoring_rejects_invalid_temperature(temperature) -> None:
    with pytest.raises(ValueError):
        scoring.score_candidates(
            _request(), [1.0, 0.0], model_revision="fixture-v1", temperature=temperature
        )


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan, math.inf, True])
def test_scoring_rejects_invalid_confidence_threshold(threshold) -> None:
    with pytest.raises(ValueError):
        scoring.score_candidates(
            _request(), [1.0, 0.0], model_revision="fixture-v1", min_confidence=threshold
        )


@pytest.mark.parametrize("logits", [{"a": 1.0}, {"a": 1.0, "b": 0.0, "extra": -1.0}])
def test_evaluation_rejects_prediction_mapping_with_wrong_option_ids(logits) -> None:
    record = _record()
    prediction = evaluation.PredictionRecord(
        record_id=record.record_id,
        request_hash=record.request.request_hash,
        logits=logits,
        model_revision="fixture-v1",
    )

    with pytest.raises(ValueError, match="wrong option IDs"):
        _evaluate(record, (prediction,))


def test_evaluation_rejects_missing_and_unexpected_prediction_joins() -> None:
    record = _record()
    with pytest.raises(ValueError, match="missing predictions"):
        _evaluate(record, ())

    extra = evaluation.PredictionRecord(
        record_id="unmatched-record",
        request_hash=record.request.request_hash,
        logits={"a": 1.0, "b": 0.0},
        model_revision="fixture-v1",
    )
    with pytest.raises(ValueError, match="missing predictions.*unexpected predictions"):
        _evaluate(record, (extra,))


def test_permutation_comparison_rejects_different_semantic_requests() -> None:
    left = scoring.score_candidates(_request(), [1.0, 0.0], model_revision="fixture-v1")
    right = scoring.score_candidates(
        _request(context="A different request"), [1.0, 0.0], model_revision="fixture-v1"
    )

    with pytest.raises(ValueError, match="same semantic request hash"):
        permutation.compare_permutations(left, right)


def test_permutation_comparison_rejects_mismatched_option_set() -> None:
    left = scoring.score_candidates(_request(), [1.0, 0.0], model_revision="fixture-v1")
    tampered_scores = (
        left.scores[0],
        left.scores[1].model_copy(update={"option_id": "stale-id"}),
    )
    right = left.model_copy(update={"scores": tampered_scores})

    with pytest.raises(ValueError, match="same semantic option IDs"):
        permutation.compare_permutations(left, right)


def test_permutation_js_handles_subnormal_probability_without_zero_midpoint_division() -> None:
    request = _request()
    left = scoring.score_candidates(request, [0.0, -745.0], model_revision="fixture-v1")
    right = scoring.score_candidates(request, [0.0, -746.0], model_revision="fixture-v1")

    comparison = permutation.compare_permutations(left, right)

    assert math.isfinite(comparison.jensen_shannon_divergence)
    assert 0.0 <= comparison.jensen_shannon_divergence <= math.log(2.0)
