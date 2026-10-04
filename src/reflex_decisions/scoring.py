"""Stable candidate scoring and abstention policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema import DecisionRequest


class CandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    option_id: str
    logit: float = Field(allow_inf_nan=False)
    log_probability: float = Field(allow_inf_nan=False)
    probability: float = Field(allow_inf_nan=False, ge=0.0, le=1.0)


class DecisionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: tuple[CandidateScore, ...]
    top_option_id: str
    answer_id: str | None
    abstained: bool
    reason: Literal["tie", "low_confidence"] | None
    request_hash: str
    schema_hash: str
    model_revision: str
    calibration_id: str | None
    calibration_status: Literal["uncalibrated", "profile_id_supplied"]
    temperature: float = Field(gt=0.0, allow_inf_nan=False)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)


def _finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _nonblank(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value


def score_candidates(
    request: DecisionRequest,
    logits: Sequence[float | int],
    *,
    model_revision: str,
    temperature: float | int = 1.0,
    calibration_id: str | None = None,
    min_confidence: float | int | None = None,
) -> DecisionResult:
    """Normalize candidate logits and apply a frozen tie/confidence policy."""

    if len(logits) != len(request.options):
        raise ValueError(f"expected {len(request.options)} logits, received {len(logits)}")
    _nonblank(model_revision, "model_revision")
    if calibration_id is not None:
        _nonblank(calibration_id, "calibration_id")

    temp = _finite_real(temperature, "temperature")
    if temp <= 0.0:
        raise ValueError("temperature must be positive")
    threshold: float | None = None
    if min_confidence is not None:
        threshold = _finite_real(min_confidence, "min_confidence")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

    raw_logits = tuple(_finite_real(value, f"logit[{index}]") for index, value in enumerate(logits))
    scaled: list[float] = []
    for index, value in enumerate(raw_logits):
        try:
            normalized = value / temp
        except OverflowError as exc:
            raise ValueError(f"scaled logit[{index}] is not representable") from exc
        if not math.isfinite(normalized):
            raise ValueError(f"scaled logit[{index}] is not representable")
        scaled.append(normalized)

    maximum = max(scaled)
    differences = [value - maximum for value in scaled]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("scaled logit differences are not representable")
    normalizer = math.fsum(math.exp(value) for value in differences)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("softmax normalizer is not representable")
    log_normalizer = math.log(normalizer)
    log_probabilities = [value - log_normalizer for value in differences]
    probabilities = [math.exp(value) for value in log_probabilities]
    if any(not math.isfinite(value) for value in log_probabilities + probabilities):
        raise ValueError("normalized candidate scores are not representable")

    top_value = maximum
    tied_ids = [
        option.id
        for option, value in zip(request.options, scaled, strict=True)
        if value == top_value
    ]
    top_option_id = min(tied_ids)
    abstained = len(tied_ids) > 1
    reason: Literal["tie", "low_confidence"] | None = "tie" if abstained else None
    answer_id: str | None = None if abstained else top_option_id
    confidence = max(probabilities)
    if not abstained and threshold is not None and confidence < threshold:
        abstained = True
        reason = "low_confidence"
        answer_id = None

    scores = tuple(
        CandidateScore(
            option_id=option.id,
            logit=raw_logit,
            log_probability=log_probability,
            probability=probability,
        )
        for option, raw_logit, log_probability, probability in zip(
            request.options, raw_logits, log_probabilities, probabilities, strict=True
        )
    )
    return DecisionResult(
        scores=scores,
        top_option_id=top_option_id,
        answer_id=answer_id,
        abstained=abstained,
        reason=reason,
        request_hash=request.request_hash,
        schema_hash=request.schema_hash,
        model_revision=model_revision,
        calibration_id=calibration_id,
        calibration_status="uncalibrated" if calibration_id is None else "profile_id_supplied",
        temperature=temp,
        min_confidence=threshold,
    )
