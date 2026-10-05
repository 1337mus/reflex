"""Deterministic tool selection from declared capabilities and permissions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

NO_ELIGIBLE_TOOL = "no_eligible_tool"
INSUFFICIENT_INFORMATION = "insufficient_information"
_RESERVED_TOOL_IDS = frozenset((NO_ELIGIBLE_TOOL, INSUFFICIENT_INFORMATION))
_MAX_IDENTIFIER_LENGTH = 256


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be blank")
    if len(value) > _MAX_IDENTIFIER_LENGTH:
        raise ValueError(f"{field} must be at most {_MAX_IDENTIFIER_LENGTH} characters")
    return value


def _identifier_tuple(value: object, field: str, *, minimum: int, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be a tuple")
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"{field} must contain between {minimum} and {maximum} values")
    entries = tuple(_identifier(entry, field) for entry in value)
    if len(entries) != len(set(entries)):
        raise ValueError(f"{field} must be unique")
    return entries


@dataclass(frozen=True)
class ToolSpec:
    id: str
    capabilities: tuple[str, ...]
    input_types: tuple[str, ...]
    required_permissions: tuple[str, ...]
    cost: int

    def __post_init__(self) -> None:
        _identifier(self.id, "tool ID")
        if self.id in _RESERVED_TOOL_IDS:
            raise ValueError("tool ID cannot use a reserved answer ID")
        _identifier_tuple(self.capabilities, "capabilities", minimum=1, maximum=4)
        _identifier_tuple(self.input_types, "input types", minimum=1, maximum=3)
        _identifier_tuple(self.required_permissions, "required permissions", minimum=0, maximum=3)
        if type(self.cost) is not int:
            raise TypeError("cost must be an integer")
        if not 0 <= self.cost <= 1_000_000:
            raise ValueError("cost must be between 0 and 1000000")


@dataclass(frozen=True)
class ToolScenario:
    capabilities: tuple[str, ...]
    input_types: tuple[str, ...]
    permissions: tuple[str, ...]
    tools: tuple[ToolSpec, ...]
    required_capabilities: tuple[str, ...]
    input_type: str
    permission_values: tuple[bool | None, ...]

    def __post_init__(self) -> None:
        capabilities = _identifier_tuple(self.capabilities, "capabilities", minimum=1, maximum=4)
        input_types = _identifier_tuple(self.input_types, "input types", minimum=1, maximum=3)
        permissions = _identifier_tuple(self.permissions, "permissions", minimum=0, maximum=3)
        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be a tuple")
        if not 2 <= len(self.tools) <= 6:
            raise ValueError("tools must contain between 2 and 6 values")
        if any(not isinstance(tool, ToolSpec) for tool in self.tools):
            raise TypeError("tools must contain ToolSpec values")
        tool_ids = tuple(tool.id for tool in self.tools)
        if len(tool_ids) != len(set(tool_ids)):
            raise ValueError("tool IDs must be unique")
        required = _identifier_tuple(
            self.required_capabilities, "required capabilities", minimum=1, maximum=4
        )
        if not set(required) <= set(capabilities):
            raise ValueError("required capabilities must belong to the declared domain")
        input_type = _identifier(self.input_type, "input type")
        if input_type not in input_types:
            raise ValueError("input type must belong to the declared domain")
        if not isinstance(self.permission_values, tuple):
            raise TypeError("permission values must be a tuple")
        if len(self.permission_values) != len(permissions):
            raise ValueError("permission values must match the permission count")
        if any(value is not None and type(value) is not bool for value in self.permission_values):
            raise TypeError("permission values must be booleans or None")
        for tool in self.tools:
            if not set(tool.capabilities) <= set(capabilities):
                raise ValueError("tool capabilities must belong to the declared domain")
            if not set(tool.input_types) <= set(input_types):
                raise ValueError("tool input types must belong to the declared domain")
            if not set(tool.required_permissions) <= set(permissions):
                raise ValueError("tool permissions must belong to the declared domain")


def answer_ids(scenario: ToolScenario) -> tuple[str, ...]:
    if not isinstance(scenario, ToolScenario):
        raise TypeError("scenario must be a ToolScenario")
    return (*tuple(tool.id for tool in scenario.tools), NO_ELIGIBLE_TOOL, INSUFFICIENT_INFORMATION)


def solve(scenario: ToolScenario) -> str:
    if not isinstance(scenario, ToolScenario):
        raise TypeError("scenario must be a ToolScenario")

    required = set(scenario.required_capabilities)
    outcomes: set[str] = set()
    permission_domains = tuple(
        (False, True) if value is None else (value,) for value in scenario.permission_values
    )
    for permission_values in product(*permission_domains):
        granted = {
            permission
            for permission, value in zip(scenario.permissions, permission_values, strict=True)
            if value
        }
        eligible = [
            tool
            for tool in scenario.tools
            if required <= set(tool.capabilities)
            and scenario.input_type in tool.input_types
            and set(tool.required_permissions) <= granted
        ]
        if not eligible:
            outcomes.add(NO_ELIGIBLE_TOOL)
            continue
        lowest_cost = min(tool.cost for tool in eligible)
        winners = [tool for tool in eligible if tool.cost == lowest_cost]
        if len(winners) > 1:
            raise ValueError(f"multiple tools tie at lowest cost {lowest_cost}")
        outcomes.add(winners[0].id)

    if len(outcomes) == 1:
        return next(iter(outcomes))
    return INSUFFICIENT_INFORMATION
