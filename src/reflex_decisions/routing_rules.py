"""Deterministic support-routing decisions from closed rule tables."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod

NO_ROUTE = "no_route"
INSUFFICIENT_INFORMATION = "insufficient_information"
_RESERVED_ANSWERS = frozenset((NO_ROUTE, INSUFFICIENT_INFORMATION))


def _require_nonblank(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be blank")
    return value


@dataclass(frozen=True)
class Feature:
    """A named routing attribute and its closed set of legal values."""

    name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonblank(self.name, "feature name")
        if not isinstance(self.values, tuple):
            raise TypeError("feature values must be a tuple")
        if not 1 <= len(self.values) <= 4:
            raise ValueError("feature values must contain between 1 and 4 values")
        for value in self.values:
            _require_nonblank(value, "feature values")
        if len(self.values) != len(set(self.values)):
            raise ValueError("feature values must be unique")


@dataclass(frozen=True)
class RouteRule:
    """An exact feature-value tuple mapped to one destination."""

    values: tuple[str, ...]
    destination_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise TypeError("rule values must be a tuple")
        for value in self.values:
            _require_nonblank(value, "rule values")
        _require_nonblank(self.destination_id, "rule destination")


@dataclass(frozen=True)
class RoutingScenario:
    """Immutable domains, exact routes, and an optional-value query."""

    features: tuple[Feature, ...]
    destination_ids: tuple[str, ...]
    rules: tuple[RouteRule, ...]
    query: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.features, tuple):
            raise TypeError("features must be a tuple")
        if not 1 <= len(self.features) <= 4:
            raise ValueError("features must contain between 1 and 4 features")
        if any(not isinstance(feature, Feature) for feature in self.features):
            raise TypeError("features must contain Feature values")
        names = tuple(feature.name for feature in self.features)
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        if prod(len(feature.values) for feature in self.features) > 256:
            raise ValueError("feature domains must have at most 256 completions")

        if not isinstance(self.destination_ids, tuple):
            raise TypeError("destination IDs must be a tuple")
        if not 2 <= len(self.destination_ids) <= 6:
            raise ValueError("destination IDs must contain between 2 and 6 destinations")
        for destination_id in self.destination_ids:
            _require_nonblank(destination_id, "destination ID")
        if len(self.destination_ids) != len(set(self.destination_ids)):
            raise ValueError("destination IDs must be unique")
        if _RESERVED_ANSWERS.intersection(self.destination_ids):
            raise ValueError("destination IDs cannot use reserved answer IDs")

        if not isinstance(self.rules, tuple):
            raise TypeError("rules must be a tuple")
        rule_keys: set[tuple[str, ...]] = set()
        for rule in self.rules:
            if not isinstance(rule, RouteRule):
                raise TypeError("rules must contain RouteRule values")
            if len(rule.values) != len(self.features):
                raise ValueError("rule values must match the feature count")
            for rule_value, feature in zip(rule.values, self.features, strict=True):
                if rule_value not in feature.values:
                    raise ValueError("rule values must belong to each feature domain")
            if rule.values in rule_keys:
                raise ValueError("rule keys must be unique")
            rule_keys.add(rule.values)
            if rule.destination_id not in self.destination_ids:
                raise ValueError("rule destination must be a known destination ID")

        if not isinstance(self.query, tuple):
            raise TypeError("query must be a tuple")
        if len(self.query) != len(self.features):
            raise ValueError("query length must match the feature count")
        for query_value, feature in zip(self.query, self.features, strict=True):
            if query_value is not None:
                _require_nonblank(query_value, "query value")
                if query_value not in feature.values:
                    raise ValueError("query values must belong to each feature domain")


def solve(scenario: RoutingScenario) -> str:
    """Return the sole outcome shared by every legal completion of the query."""
    if not isinstance(scenario, RoutingScenario):
        raise TypeError("scenario must be a RoutingScenario")

    rules = {rule.values: rule.destination_id for rule in scenario.rules}
    domains = tuple(
        feature.values if value is None else (value,)
        for feature, value in zip(scenario.features, scenario.query, strict=True)
    )
    outcomes = {rules.get(completion, NO_ROUTE) for completion in product(*domains)}
    if len(outcomes) == 1:
        return outcomes.pop()
    return INSUFFICIENT_INFORMATION


def answer_ids(scenario: RoutingScenario) -> tuple[str, ...]:
    """Return every destination and both reserved answers in a stable order."""
    if not isinstance(scenario, RoutingScenario):
        raise TypeError("scenario must be a RoutingScenario")
    return (*scenario.destination_ids, NO_ROUTE, INSUFFICIENT_INFORMATION)
