"""Parse and verify the model-visible text for tool-selection requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .rendering import render_prompt
from .schema import DecisionRequest
from .tool_rules import (
    INSUFFICIENT_INFORMATION,
    NO_ELIGIBLE_TOOL,
    ToolScenario,
    ToolSpec,
    answer_ids,
)
from .tool_text import TOOL_TEMPLATES

_ANSWER_SUFFIX = "\nAnswer:\n"
_SPECIAL_LABELS = {
    NO_ELIGIBLE_TOOL: "No eligible tool",
    INSUFFICIENT_INFORMATION: "Insufficient information",
}


@dataclass(frozen=True)
class ParsedToolPrompt:
    scenario: ToolScenario
    option_ids: tuple[str, ...]
    template_id: str = "tool-v1"


def parse_tool_prompt(prompt: str) -> ParsedToolPrompt:
    """Parse exact rendered prompt text without consulting a structured request."""
    if type(prompt) is not str or not prompt.endswith(_ANSWER_SUFFIX):
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
        matches = [
            (template_id, prefix, question)
            for template_id, (prefix, question) in TOOL_TEMPLATES.items()
            if context.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError("context instructions are unrecognized")
        template_id, context_prefix, question = matches[0]
        facts = _require_keys(
            _object(
                json.loads(
                    context[len(context_prefix) :],
                    object_pairs_hook=_unique_object,
                ),
                "tool data",
            ),
            {
                "capabilities",
                "input_types",
                "permissions",
                "tools",
                "required_capabilities",
                "input_type",
                "permission_values",
            },
            "tool data",
        )
        tools: list[ToolSpec] = []
        for raw_tool in _list(facts["tools"], "tools"):
            row = _require_keys(
                _object(raw_tool, "tool"),
                {"id", "capabilities", "input_types", "required_permissions", "cost"},
                "tool",
            )
            tools.append(
                ToolSpec(
                    id=_string(row["id"], "tool ID"),
                    capabilities=_string_tuple(row["capabilities"], "tool capabilities"),
                    input_types=_string_tuple(row["input_types"], "tool input types"),
                    required_permissions=_string_tuple(
                        row["required_permissions"], "tool required permissions"
                    ),
                    cost=_integer(row["cost"], "tool cost"),
                )
            )
        raw_permission_values = _list(facts["permission_values"], "permission_values")
        if any(value is not None and type(value) is not bool for value in raw_permission_values):
            raise ValueError("permission_values must contain only booleans or null")
        scenario = ToolScenario(
            capabilities=_string_tuple(facts["capabilities"], "capabilities"),
            input_types=_string_tuple(facts["input_types"], "input_types"),
            permissions=_string_tuple(facts["permissions"], "permissions"),
            tools=tuple(tools),
            required_capabilities=_string_tuple(
                facts["required_capabilities"], "required_capabilities"
            ),
            input_type=_string(facts["input_type"], "input_type"),
            permission_values=tuple(raw_permission_values),
        )

        if _string(outer["question"], "question") != question:
            raise ValueError("question is unrecognized for tool template")
        options = _list(outer["options"], "options")
        if len(options) != len(answer_ids(scenario)):
            raise ValueError("options must include every tool-selection choice")
        labels = {
            "Use tool " + json.dumps(tool.id, ensure_ascii=False): tool.id
            for tool in scenario.tools
        }
        labels.update({label: answer_id for answer_id, label in _SPECIAL_LABELS.items()})
        option_ids: list[str] = []
        for index, raw_option in enumerate(options):
            row = _require_keys(
                _object(raw_option, "option"),
                {"symbol", "label", "description"},
                "option",
            )
            if _string(row["symbol"], "option symbol") != chr(ord("A") + index):
                raise ValueError("options have invalid symbols")
            if row["description"] is not None:
                raise ValueError("option descriptions must be null")
            label = _string(row["label"], "option label")
            option_id = labels.get(label)
            if option_id is None or option_id in option_ids:
                raise ValueError("options contain an unknown or duplicate choice")
            option_ids.append(option_id)
        expected_ids = answer_ids(scenario)
        if len(option_ids) != len(expected_ids) or set(option_ids) != set(expected_ids):
            raise ValueError("options must contain every tool-selection choice exactly once")
        return ParsedToolPrompt(scenario, tuple(option_ids), template_id)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tool prompt: {exc}") from exc


def verify_tool_request(request: DecisionRequest, scenario: ToolScenario) -> None:
    """Verify that prompt rendering preserves all scenario facts and menu choices."""
    parsed = parse_tool_prompt(render_prompt(request))
    if parsed.scenario != scenario:
        raise ValueError("rendered tool facts do not match the expected scenario")
    request_ids = tuple(option.id for option in request.options)
    if parsed.option_ids != request_ids:
        raise ValueError("rendered tool choices do not match the request option IDs")


def _object(value: Any, field: str) -> dict[str, Any]:
    if type(value) is not dict:
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
    if type(value) is not list:
        raise ValueError(f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field} must be a string")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    return tuple(_string(entry, field) for entry in _list(value, field))


def _integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value
