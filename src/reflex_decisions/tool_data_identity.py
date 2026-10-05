"""Canonical identities for tool-choice scenarios and catalogs."""

from __future__ import annotations

import hashlib
import itertools
import json
import re

from .data import SplitName
from .tool_rules import ToolScenario


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _named_payload(scenario: ToolScenario, *, include_query: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "capabilities": sorted(scenario.capabilities),
        "input_types": sorted(scenario.input_types),
        "permissions": sorted(scenario.permissions),
        "tools": sorted(
            (
                {
                    "id": tool.id,
                    "capabilities": sorted(tool.capabilities),
                    "input_types": sorted(tool.input_types),
                    "required_permissions": sorted(tool.required_permissions),
                    "cost": tool.cost,
                }
                for tool in scenario.tools
            ),
            key=_canonical_json,
        ),
    }
    if include_query:
        permission_values = dict(zip(scenario.permissions, scenario.permission_values, strict=True))
        payload["query"] = {
            "required_capabilities": sorted(scenario.required_capabilities),
            "input_type": scenario.input_type,
            "permission_values": sorted(permission_values.items()),
        }
    return payload


def catalog_hash(scenario: ToolScenario) -> str:
    if not isinstance(scenario, ToolScenario):
        raise TypeError("scenario must be a ToolScenario")
    return _digest(_named_payload(scenario, include_query=False))


def scenario_hash(scenario: ToolScenario) -> str:
    if not isinstance(scenario, ToolScenario):
        raise TypeError("scenario must be a ToolScenario")
    return _digest(_named_payload(scenario, include_query=True))


def _mask(value: tuple[str, ...], positions: dict[str, int]) -> int:
    return sum(1 << positions[entry] for entry in value)


def _mask_permutations(size: int) -> tuple[tuple[int, ...], ...]:
    maps = []
    for order in itertools.permutations(range(size)):
        maps.append(
            tuple(
                sum(1 << order[index] for index in range(size) if mask & (1 << index))
                for mask in range(1 << size)
            )
        )
    return tuple(maps)


def structural_catalog_hash(scenario: ToolScenario) -> str:
    if not isinstance(scenario, ToolScenario):
        raise TypeError("scenario must be a ToolScenario")

    capability_positions = {value: index for index, value in enumerate(scenario.capabilities)}
    input_positions = {value: index for index, value in enumerate(scenario.input_types)}
    permission_positions = {value: index for index, value in enumerate(scenario.permissions)}
    costs = {
        value: rank for rank, value in enumerate(sorted({tool.cost for tool in scenario.tools}))
    }
    rows = tuple(
        (
            _mask(tool.capabilities, capability_positions),
            _mask(tool.input_types, input_positions),
            _mask(tool.required_permissions, permission_positions),
            costs[tool.cost],
        )
        for tool in scenario.tools
    )
    capability_maps = _mask_permutations(len(scenario.capabilities))
    input_maps = _mask_permutations(len(scenario.input_types))
    permission_maps = _mask_permutations(len(scenario.permissions))

    smallest: tuple[tuple[int, int, int, int], ...] | None = None
    for cap_map, input_map, permission_map in itertools.product(
        capability_maps, input_maps, permission_maps
    ):
        encoded = tuple(
            sorted(
                (cap_map[capability], input_map[input_type], permission_map[permission], rank)
                for capability, input_type, permission, rank in rows
            )
        )
        if smallest is None or encoded < smallest:
            smallest = encoded

    assert smallest is not None
    return _digest(
        {
            "capability_count": len(scenario.capabilities),
            "input_type_count": len(scenario.input_types),
            "permission_count": len(scenario.permissions),
            "tool_count": len(scenario.tools),
            "tools": smallest,
        }
    )


def split_for_structure(digest: str) -> SplitName:
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("structural digest must be 64 lowercase hexadecimal characters")
    slots: tuple[SplitName, ...] = (
        "train",
        "train",
        "train",
        "train",
        "development",
        "calibration",
        "test",
        "test",
    )
    return slots[int(digest, 16) % len(slots)]
