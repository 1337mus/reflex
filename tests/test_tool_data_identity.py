"""Canonical identity contracts for tool-choice data."""

from dataclasses import replace

import pytest

from reflex_decisions.tool_data_identity import (
    catalog_hash,
    scenario_hash,
    split_for_structure,
    structural_catalog_hash,
)
from reflex_decisions.tool_rules import ToolScenario, ToolSpec


def _scenario(
    *,
    capabilities: tuple[str, ...] = ("lookup", "summarize"),
    input_types: tuple[str, ...] = ("memo", "photo"),
    permissions: tuple[str, ...] = ("private", "external"),
    tools: tuple[ToolSpec, ...] | None = None,
    required_capabilities: tuple[str, ...] = ("lookup",),
    input_type: str = "memo",
    permission_values: tuple[bool | None, ...] = (None, False),
) -> ToolScenario:
    selected_tools = tools or (
        ToolSpec("alpha", ("lookup",), ("memo",), ("private",), 3),
        ToolSpec("beta", ("lookup", "summarize"), ("memo", "photo"), (), 10),
    )
    return ToolScenario(
        capabilities,
        input_types,
        permissions,
        selected_tools,
        required_capabilities,
        input_type,
        permission_values,
    )


def _structural_scenario() -> ToolScenario:
    return _scenario(
        capabilities=("lookup", "summarize", "translate", "extract"),
        input_types=("memo", "photo", "clip"),
        permissions=("private", "external"),
        tools=(
            ToolSpec("alpha", ("lookup", "translate"), ("memo", "clip"), ("private",), 1),
            ToolSpec("beta", ("summarize",), ("photo",), (), 4),
            ToolSpec("gamma", ("extract", "lookup"), ("memo", "photo"), ("external",), 9),
        ),
        permission_values=(None, False),
    )


def test_named_catalog_is_order_invariant_and_excludes_query() -> None:
    original = _scenario()
    reordered = _scenario(
        capabilities=("summarize", "lookup"),
        input_types=("photo", "memo"),
        permissions=("external", "private"),
        tools=(
            ToolSpec("beta", ("summarize", "lookup"), ("photo", "memo"), (), 10),
            ToolSpec("alpha", ("lookup",), ("memo",), ("private",), 3),
        ),
        required_capabilities=("lookup",),
        input_type="memo",
        permission_values=(False, None),
    )
    other_query = replace(original, required_capabilities=("lookup", "summarize"))

    assert catalog_hash(original) == catalog_hash(reordered)
    assert catalog_hash(original) == catalog_hash(other_query)


def test_named_hashes_preserve_exact_cost_values() -> None:
    original = _scenario()
    repriced = replace(
        original,
        tools=tuple(replace(tool, cost=tool.cost + 1) for tool in original.tools),
    )

    assert catalog_hash(original) != catalog_hash(repriced)
    assert scenario_hash(original) != scenario_hash(repriced)


def test_named_scenario_preserves_permission_pairing_under_reordering() -> None:
    original = _scenario()
    reordered = _scenario(
        capabilities=("summarize", "lookup"),
        input_types=("photo", "memo"),
        permissions=("external", "private"),
        tools=(
            ToolSpec("beta", ("summarize", "lookup"), ("photo", "memo"), (), 10),
            ToolSpec("alpha", ("lookup",), ("memo",), ("private",), 3),
        ),
        required_capabilities=("lookup",),
        input_type="memo",
        permission_values=(False, None),
    )
    other_query = replace(original, required_capabilities=("lookup", "summarize"))

    assert scenario_hash(original) == scenario_hash(reordered)
    assert scenario_hash(original) != scenario_hash(other_query)


def test_structural_catalog_ignores_domain_names_ids_order_and_cost_scale() -> None:
    original = _structural_scenario()
    renamed = _scenario(
        capabilities=("parse", "convert", "digest", "find"),
        input_types=("movie", "note", "image"),
        permissions=("remote", "locked"),
        tools=(
            ToolSpec("three", ("parse", "find"), ("image", "note"), ("remote",), 87),
            ToolSpec("two", ("digest",), ("image",), (), 37),
            ToolSpec("one", ("convert", "find"), ("movie", "note"), ("locked",), 11),
        ),
        required_capabilities=("parse",),
        input_type="movie",
        permission_values=(True, None),
    )
    renamed_again = _scenario(
        capabilities=("convert", "parse", "find", "digest"),
        input_types=("type-c", "type-a", "type-b"),
        permissions=("permission-y", "permission-x"),
        tools=(
            ToolSpec("green", ("parse", "find"), ("type-a", "type-c"), ("permission-x",), 900),
            ToolSpec("red", ("find", "convert"), ("type-c", "type-b"), ("permission-y",), 100),
            ToolSpec("blue", ("digest",), ("type-a",), (), 400),
        ),
        required_capabilities=("parse",),
        input_type="type-b",
        permission_values=(False, True),
    )

    assert structural_catalog_hash(original) == structural_catalog_hash(renamed)
    assert structural_catalog_hash(original) == structural_catalog_hash(renamed_again)
    assert split_for_structure(structural_catalog_hash(original)) == split_for_structure(
        structural_catalog_hash(renamed_again)
    )


def test_structural_catalog_preserves_cost_order_ties_domains_and_duplicate_rows() -> None:
    original = _structural_scenario()
    rescaled = replace(
        original,
        tools=tuple(replace(tool, cost=tool.cost * 17 + 5) for tool in original.tools),
    )
    swapped = replace(
        original,
        tools=tuple(
            replace(tool, cost=10 if tool.id == "alpha" else (1 if tool.id == "beta" else 4))
            for tool in original.tools
        ),
    )
    tied = replace(
        original,
        tools=tuple(replace(tool, cost=1) for tool in original.tools),
    )
    unused_domain = replace(
        original,
        permissions=(*original.permissions, "unused"),
        permission_values=(*original.permission_values, None),
    )
    duplicate_signature = replace(
        original,
        tools=(
            original.tools[0],
            replace(original.tools[0], id="other-tool", cost=4),
            original.tools[2],
        ),
    )

    original_hash = structural_catalog_hash(original)
    assert structural_catalog_hash(rescaled) == original_hash
    assert structural_catalog_hash(swapped) != original_hash
    assert structural_catalog_hash(tied) != original_hash
    assert structural_catalog_hash(unused_domain) != original_hash
    assert structural_catalog_hash(duplicate_signature) != original_hash


def test_structural_hash_supports_an_empty_permission_domain() -> None:
    scenario = ToolScenario(
        ("lookup",),
        ("memo",),
        (),
        (
            ToolSpec("alpha", ("lookup",), ("memo",), (), 0),
            ToolSpec("beta", ("lookup",), ("memo",), (), 9),
        ),
        ("lookup",),
        "memo",
        (),
    )

    assert len(structural_catalog_hash(scenario)) == 64
    assert structural_catalog_hash(scenario) == structural_catalog_hash(scenario)


def test_split_assignment_uses_the_required_four_one_one_two_buckets() -> None:
    assert split_for_structure("0" * 64) == "train"
    assert split_for_structure("0" * 63 + "4") == "development"
    assert split_for_structure("0" * 63 + "5") == "calibration"
    assert split_for_structure("0" * 63 + "6") == "test"
    assert split_for_structure("0" * 63 + "7") == "test"

    for malformed in ("a" * 63, "A" * 64, "g" * 64, None):
        with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
            split_for_structure(malformed)  # type: ignore[arg-type]
