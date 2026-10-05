"""Fixed-grid temperature fitting for held-out calibration examples."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from .scoring import _finite_real

_MIN_TEMPERATURE = 0.05
_MAX_TEMPERATURE = 10.0
_GRID_STEPS = 81
_log_min = math.log(_MIN_TEMPERATURE)
_log_span = math.log(_MAX_TEMPERATURE) - _log_min
_temperatures = tuple(
    _MIN_TEMPERATURE
    if index == 0
    else _MAX_TEMPERATURE
    if index == _GRID_STEPS - 1
    else math.exp(_log_min + _log_span * index / (_GRID_STEPS - 1))
    for index in range(_GRID_STEPS)
)
_TEMPERATURE_GRID = tuple(sorted((*_temperatures, 1.0)))


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    """One row of candidate logits and its zero-based correct candidate index."""

    logits: tuple[float, ...]
    gold_index: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        try:
            frozen_logits = tuple(self.logits)
        except TypeError as exc:
            raise ValueError("logits must be a finite real sequence") from exc
        object.__setattr__(self, "logits", frozen_logits)


@dataclass(frozen=True, slots=True)
class TemperatureFitResult:
    """Immutable result of evaluating the fixed bounded temperature grid."""

    temperature: float
    raw_nll: float
    fitted_nll: float
    example_count: int
    grid_range: tuple[float, float]
    grid_count: int
    boundary_hit: bool


def _row_nll(logits: tuple[float, ...], gold_index: int, temperature: float) -> float:
    maximum = max(logits)
    try:
        scaled_differences = tuple((logit - maximum) / temperature for logit in logits)
    except OverflowError as exc:
        raise ValueError("scaled logit differences are not representable") from exc
    if any(not math.isfinite(value) for value in scaled_differences):
        raise ValueError("scaled logit differences are not representable")
    try:
        normalizer = math.fsum(math.exp(value) for value in scaled_differences)
        nll = math.fsum((math.log(normalizer), -scaled_differences[gold_index]))
    except (OverflowError, ValueError) as exc:
        raise ValueError("candidate NLL is not representable") from exc
    if not math.isfinite(normalizer) or normalizer <= 0.0 or not math.isfinite(nll):
        raise ValueError("candidate NLL is not representable")
    return nll


def _weighted_mean(values: tuple[float, ...], weights: tuple[float, ...]) -> float:
    try:
        result = math.fsum(weight * value for weight, value in zip(weights, values, strict=True))
    except (OverflowError, ValueError) as exc:
        raise ValueError("weighted mean NLL is not representable") from exc
    if not math.isfinite(result):
        raise ValueError("weighted mean NLL is not representable")
    return result


def fit_temperature(examples: Iterable[CalibrationExample]) -> TemperatureFitResult:
    """Fit temperature by weighted NLL on a fixed 82-point bounded grid.

    The grid contains 81 logarithmically spaced values from 0.05 to 10 plus 1.0;
    the result describes that discrete search, not a continuous global optimum.
    """

    try:
        rows = tuple(examples)
    except TypeError as exc:
        raise ValueError("examples must be a nonempty iterable") from exc
    if not rows:
        raise ValueError("examples must not be empty")

    validated: list[tuple[tuple[float, ...], int, float]] = []
    for example in rows:
        if not isinstance(example, CalibrationExample):
            raise ValueError("examples must contain CalibrationExample values")
        if not isinstance(example.logits, tuple) or not 2 <= len(example.logits) <= 16:
            raise ValueError("each example must contain between 2 and 16 logits")
        logits = tuple(_finite_real(value, "logit") for value in example.logits)
        if isinstance(example.gold_index, bool) or not isinstance(example.gold_index, int):
            raise ValueError("gold_index must be a legal integer candidate index")
        if not 0 <= example.gold_index < len(logits):
            raise ValueError("gold_index must be a legal integer candidate index")
        weight = _finite_real(example.weight, "weight")
        if weight <= 0.0:
            raise ValueError("weight must be positive")
        validated.append((logits, example.gold_index, weight))

    largest_weight = max(weight for _, _, weight in validated)
    scaled_weights = tuple(weight / largest_weight for _, _, weight in validated)
    try:
        weight_total = math.fsum(scaled_weights)
    except (OverflowError, ValueError) as exc:
        raise ValueError("normalized weights are not representable") from exc
    if not math.isfinite(weight_total) or weight_total <= 0.0:
        raise ValueError("normalized weights are not representable")
    weights = tuple(weight / weight_total for weight in scaled_weights)

    losses = tuple(
        _weighted_mean(
            tuple(_row_nll(logits, gold_index, temperature) for logits, gold_index, _ in validated),
            weights,
        )
        for temperature in _TEMPERATURE_GRID
    )
    raw_index = _TEMPERATURE_GRID.index(1.0)
    raw_nll = losses[raw_index]
    minimum = min(losses)
    tolerance = max(1e-15, 8.0 * math.ulp(max(1.0, abs(minimum))))
    eligible = (
        index
        for index, loss in enumerate(losses)
        if loss - minimum <= tolerance and loss <= raw_nll
    )
    best_index = min(
        eligible,
        key=lambda index: (abs(math.log(_TEMPERATURE_GRID[index])), _TEMPERATURE_GRID[index]),
    )
    temperature = _TEMPERATURE_GRID[best_index]
    return TemperatureFitResult(
        temperature=temperature,
        raw_nll=raw_nll,
        fitted_nll=losses[best_index],
        example_count=len(validated),
        grid_range=(_MIN_TEMPERATURE, _MAX_TEMPERATURE),
        grid_count=len(_TEMPERATURE_GRID),
        boundary_hit=temperature in (_MIN_TEMPERATURE, _MAX_TEMPERATURE),
    )
