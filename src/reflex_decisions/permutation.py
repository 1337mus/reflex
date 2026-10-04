"""Semantic comparison of requests whose option presentation order changed."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from .scoring import DecisionResult


class PermutationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_hash: str
    forced_choice_flipped: bool
    left_top_option_id: str
    right_top_option_id: str
    jensen_shannon_divergence: float = Field(ge=0.0, le=math.log(2.0), allow_inf_nan=False)


def compare_permutations(left: DecisionResult, right: DecisionResult) -> PermutationComparison:
    """Compare semantic choices and score distributions for one request schema."""

    if left.request_hash != right.request_hash:
        raise ValueError("permutation results must have the same semantic request hash")
    left_scores = {score.option_id: score.probability for score in left.scores}
    right_scores = {score.option_id: score.probability for score in right.scores}
    if set(left_scores) != set(right_scores):
        raise ValueError("permutation results must contain the same semantic option IDs")

    contributions: list[float] = []
    for option_id in sorted(left_scores):
        left_probability = left_scores[option_id]
        right_probability = right_scores[option_id]
        probability_sum = left_probability + right_probability
        if probability_sum == 0.0:
            continue
        relative_difference = (left_probability - right_probability) / probability_sum
        magnitude = abs(relative_difference)
        if magnitude == 1.0:
            binary_kl = math.log(2.0)
        elif magnitude < 0.01:
            squared = magnitude * magnitude
            power = squared
            binary_kl = 0.0
            for term_index in range(1, 9):
                binary_kl += power / ((2 * term_index - 1) * (2 * term_index))
                power *= squared
        else:
            upper = 1.0 + magnitude
            lower = 1.0 - magnitude
            lower_term = lower * math.log1p(-magnitude) if lower > 0.0 else 0.0
            binary_kl = 0.5 * (upper * math.log1p(magnitude) + lower_term)
        contributions.append(0.5 * probability_sum * binary_kl)

    divergence = math.fsum(contributions)
    maximum_divergence = math.log(2.0)
    if not math.isfinite(divergence) or divergence < 0.0:
        raise ValueError("Jensen-Shannon divergence is outside its legal range")
    if divergence > maximum_divergence:
        if divergence - maximum_divergence > 1e-14:
            raise ValueError("Jensen-Shannon divergence is outside its legal range")
        divergence = maximum_divergence

    return PermutationComparison(
        request_hash=left.request_hash,
        forced_choice_flipped=left.top_option_id != right.top_option_id,
        left_top_option_id=left.top_option_id,
        right_top_option_id=right.top_option_id,
        jensen_shannon_divergence=divergence,
    )
