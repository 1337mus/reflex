"""Read and verify model-visible support-routing prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .rendering import render_prompt
from .routing_rules import (
    INSUFFICIENT_INFORMATION,
    NO_ROUTE,
    Feature,
    RouteRule,
    RoutingScenario,
    answer_ids,
)
from .routing_text import ROUTING_CONTEXT_PREFIX, ROUTING_QUESTION
from .schema import DecisionRequest

_ANSWER_SUFFIX = "\nAnswer:\n"
_SPECIAL_LABELS = {
    NO_ROUTE: "No route applies",
    INSUFFICIENT_INFORMATION: "Insufficient information",
}


@dataclass(frozen=True)
class ParsedRoutingPrompt:
    scenario: RoutingScenario
    option_ids: tuple[str, ...]


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_keys(value: dict[str, Any], keys: set[str], field: str) -> dict[str, Any]:
    if set(value) != keys:
        raise ValueError(f"{field} has missing or extra keys")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def parse_routing_prompt(prompt: str) -> ParsedRoutingPrompt:
    """Parse the exact model-visible text into its routing facts and menu IDs."""
    if not isinstance(prompt, str) or not prompt.endswith(_ANSWER_SUFFIX):
        raise ValueError("prompt must end with the exact answer suffix")
    try:
        outer = _require_keys(
            _object(
                json.loads(prompt[: -len(_ANSWER_SUFFIX)], object_pairs_hook=_unique_object),
                "prompt",
            ),
            {"context", "question", "options"},
            "prompt",
        )
        context = _string(outer["context"], "context")
        if not context.startswith(ROUTING_CONTEXT_PREFIX):
            raise ValueError("context instructions are unrecognized")
        block = _require_keys(
            _object(
                json.loads(
                    context[len(ROUTING_CONTEXT_PREFIX) :],
                    object_pairs_hook=_unique_object,
                ),
                "routing data",
            ),
            {"features", "destination_ids", "rules", "query"},
            "routing data",
        )

        features_list: list[Feature] = []
        for raw_feature in _list(block["features"], "features"):
            feature = _require_keys(_object(raw_feature, "feature"), {"name", "values"}, "feature")
            values = _list(feature["values"], "feature values")
            features_list.append(
                Feature(
                    _string(feature["name"], "feature name"),
                    tuple(_string(value, "feature value") for value in values),
                )
            )
        features = tuple(features_list)
        destination_rows = _list(block["destination_ids"], "destination_ids")
        destination_ids = tuple(_string(value, "destination ID") for value in destination_rows)
        rules_list: list[RouteRule] = []
        for raw_rule in _list(block["rules"], "rules"):
            rule = _require_keys(_object(raw_rule, "rule"), {"values", "destination_id"}, "rule")
            values = _list(rule["values"], "rule values")
            rules_list.append(
                RouteRule(
                    tuple(_string(value, "rule value") for value in values),
                    _string(rule["destination_id"], "rule destination"),
                )
            )
        rules = tuple(rules_list)
        query_rows = _list(block["query"], "query")
        query = tuple(
            None if value is None else _string(value, "query value") for value in query_rows
        )
        scenario = RoutingScenario(features, destination_ids, rules, query)

        if outer["question"] != ROUTING_QUESTION:
            raise ValueError("question is unrecognized")
        options = _list(outer["options"], "options")
        by_label = {
            "Route to " + json.dumps(destination_id, ensure_ascii=False): destination_id
            for destination_id in scenario.destination_ids
        }
        by_label.update({label: answer_id for answer_id, label in _SPECIAL_LABELS.items()})
        option_ids: list[str] = []
        if len(options) != len(answer_ids(scenario)):
            raise ValueError("options must include every routing choice")
        for index, option in enumerate(options):
            row = _require_keys(
                _object(option, "option"), {"symbol", "label", "description"}, "option"
            )
            expected_symbol = chr(ord("A") + index)
            if row["symbol"] != expected_symbol or row["description"] is not None:
                raise ValueError("options have invalid symbols or descriptions")
            option_id = by_label.get(_string(row["label"], "option label"))
            if option_id is None or option_id in option_ids:
                raise ValueError("options contain an unknown or duplicate choice")
            option_ids.append(option_id)
        expected_ids = answer_ids(scenario)
        if len(option_ids) != len(expected_ids) or set(option_ids) != set(expected_ids):
            raise ValueError("options must contain every routing choice exactly once")
        return ParsedRoutingPrompt(scenario, tuple(option_ids))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid routing prompt: {exc}") from exc


def verify_routing_request(request: DecisionRequest, expected: RoutingScenario) -> None:
    """Confirm the rendered model text exactly preserves facts and option IDs."""
    parsed = parse_routing_prompt(render_prompt(request))
    if parsed.scenario != expected:
        raise ValueError("rendered routing facts do not match the expected scenario")
    request_ids = tuple(option.id for option in request.options)
    if parsed.option_ids != request_ids:
        raise ValueError("rendered routing choices do not match the request option IDs")
