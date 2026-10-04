import math

import pytest

from reflex_decisions import permutation
from reflex_decisions.schema import DecisionRequest, Option
from reflex_decisions.scoring import score_candidates


def test_semantic_remapping_has_zero_divergence() -> None:
    first = DecisionRequest(
        context="A request",
        question="Choose one",
        options=(Option(id="red", label="Red"), Option(id="blue", label="Blue")),
    )
    reordered = DecisionRequest(
        context="A request",
        question="Choose one",
        options=(Option(id="blue", label="Blue"), Option(id="red", label="Red")),
    )
    first_result = score_candidates(first, [2.0, 0.0], model_revision="fixture-v1")
    reordered_result = score_candidates(reordered, [0.0, 2.0], model_revision="fixture-v1")

    comparison = permutation.compare_permutations(first_result, reordered_result)

    assert comparison.forced_choice_flipped is False
    assert math.isclose(comparison.jensen_shannon_divergence, 0.0, abs_tol=1e-15)


def test_changed_semantic_distribution_has_expected_jensen_shannon_divergence() -> None:
    request = DecisionRequest(
        context="A request",
        question="Choose one",
        options=(Option(id="red", label="Red"), Option(id="blue", label="Blue")),
    )
    first = score_candidates(request, [math.log(0.8), math.log(0.2)], model_revision="fixture-v1")
    second = score_candidates(request, [math.log(0.2), math.log(0.8)], model_revision="fixture-v1")

    comparison = permutation.compare_permutations(first, second)

    assert math.isclose(comparison.jensen_shannon_divergence, 0.192744757022, rel_tol=1e-10)
    assert comparison.forced_choice_flipped is True


@pytest.mark.parametrize("option_count", (11, 12, 16))
def test_sharp_reversed_distributions_remain_symmetric_and_bounded(option_count: int) -> None:
    options = tuple(
        Option(id=f"option-{index}", label=f"Option {index}") for index in range(option_count)
    )
    logits = [0.0] * (option_count // 2) + [-40.0] * (option_count - option_count // 2)
    request = DecisionRequest(context="A request", question="Choose one", options=options)
    reversed_request = request.model_copy(update={"options": tuple(reversed(options))})

    first = score_candidates(request, logits, model_revision="fixture-v1")
    second = score_candidates(reversed_request, logits, model_revision="fixture-v1")

    comparison = permutation.compare_permutations(first, second)
    reversed_comparison = permutation.compare_permutations(second, first)
    divergence = comparison.jensen_shannon_divergence

    assert math.isfinite(divergence)
    assert 0.0 <= divergence <= math.log(2.0)
    assert math.isclose(divergence, math.log(2.0), rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(
        divergence,
        reversed_comparison.jensen_shannon_divergence,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
