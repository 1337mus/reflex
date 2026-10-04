import math

from reflex_decisions import scoring
from reflex_decisions.schema import DecisionRequest, Option


def test_candidate_scores_match_hand_computed_softmax() -> None:
    request = DecisionRequest(
        context="A short request",
        question="Choose one",
        options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
    )

    result = scoring.score_candidates(request, [1.0, 0.0], model_revision="fixture-v1")

    assert math.isclose(result.scores[0].probability, 0.7310585786300049)
    assert math.isclose(result.scores[1].probability, 0.2689414213699951)
    assert math.isclose(result.scores[1].log_probability, -1.3132616875182228)


def test_underflowed_probability_keeps_finite_log_probability() -> None:
    request = DecisionRequest(
        context="A short request",
        question="Choose one",
        options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
    )

    result = scoring.score_candidates(request, [0.0, -1000.0], model_revision="fixture-v1")

    assert result.scores[1].probability == 0.0
    assert result.scores[1].log_probability == -1000.0


def test_ties_abstain_with_semantic_lexical_top_and_unknown_can_be_answer() -> None:
    request = DecisionRequest(
        context="A short request",
        question="Choose one",
        options=(Option(id="zeta", label="Zeta"), Option(id="alpha", label="Alpha")),
    )

    tied = scoring.score_candidates(request, [1.0, 1.0], model_revision="fixture-v1")

    assert tied.abstained is True
    assert tied.answer_id is None
    assert tied.reason == "tie"
    assert tied.top_option_id == "alpha"

    unknown_request = DecisionRequest(
        context="A short request",
        question="Choose one",
        options=(Option(id="unknown", label="Unknown"), Option(id="known", label="Known")),
    )
    unknown = scoring.score_candidates(unknown_request, [2.0, 0.0], model_revision="fixture-v1")

    assert unknown.answer_id == "unknown"
    assert unknown.abstained is False


def test_result_records_temperature_without_claiming_calibration() -> None:
    request = DecisionRequest(
        context="A short request",
        question="Choose one",
        options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
    )

    result = scoring.score_candidates(
        request,
        [1.0, 0.0],
        model_revision="fixture-v1",
        temperature=0.5,
    )

    assert result.temperature == 0.5
    assert result.calibration_status == "uncalibrated"
