"""Handwritten tool-choice solver and validation contracts."""

from dataclasses import replace

import pytest

from reflex_decisions.tool_rules import (
    INSUFFICIENT_INFORMATION,
    NO_ELIGIBLE_TOOL,
    ToolScenario,
    ToolSpec,
    answer_ids,
    solve,
)


def _tool(
    tool_id: str,
    *,
    capabilities: tuple[str, ...] = ("search",),
    input_types: tuple[str, ...] = ("text",),
    required_permissions: tuple[str, ...] = (),
    cost: int = 1,
) -> ToolSpec:
    return ToolSpec(tool_id, capabilities, input_types, required_permissions, cost)


def _scenario(
    tools: tuple[ToolSpec, ...],
    *,
    capabilities: tuple[str, ...] = ("search",),
    input_types: tuple[str, ...] = ("text",),
    permissions: tuple[str, ...] = (),
    required_capabilities: tuple[str, ...] = ("search",),
    input_type: str = "text",
    permission_values: tuple[bool | None, ...] = (),
) -> ToolScenario:
    return ToolScenario(
        capabilities,
        input_types,
        permissions,
        tools,
        required_capabilities,
        input_type,
        permission_values,
    )


def test_selects_dearer_tool_when_cheapest_lacks_one_required_capability() -> None:
    scenario = ToolScenario(
        capabilities=("search", "summarize"),
        input_types=("text",),
        permissions=(),
        tools=(
            ToolSpec("cheap-search", ("search",), ("text",), (), 1),
            ToolSpec("full-service", ("search", "summarize"), ("text",), (), 4),
        ),
        required_capabilities=("search", "summarize"),
        input_type="text",
        permission_values=(),
    )

    assert solve(scenario) == "full-service"


def test_answer_ids_keep_tool_order_then_special_answers() -> None:
    scenario = _scenario((_tool("first"), _tool("second")))

    assert answer_ids(scenario) == ("first", "second", NO_ELIGIBLE_TOOL, INSUFFICIENT_INFORMATION)


def test_wrong_input_and_denied_permission_make_tools_ineligible() -> None:
    scenario = _scenario(
        (
            _tool("image-only", input_types=("image",), cost=0),
            _tool("private", required_permissions=("read_private",), cost=2),
        ),
        input_types=("text", "image"),
        permissions=("read_private",),
        permission_values=(False,),
    )

    assert solve(scenario) == NO_ELIGIBLE_TOOL


def test_unknown_permission_that_changes_winner_is_insufficient() -> None:
    scenario = _scenario(
        (
            _tool("public", cost=4),
            _tool("private", required_permissions=("read_private",), cost=1),
        ),
        permissions=("read_private",),
        permission_values=(None,),
    )

    assert solve(scenario) == INSUFFICIENT_INFORMATION


def test_unknown_irrelevant_permission_keeps_the_same_winner() -> None:
    scenario = _scenario(
        (
            _tool("always", cost=1),
            _tool("private", required_permissions=("read_private",), cost=5),
        ),
        permissions=("read_private",),
        permission_values=(None,),
    )

    assert solve(scenario) == "always"


def test_unknown_permission_can_change_between_route_and_no_eligible() -> None:
    scenario = _scenario(
        (
            _tool("private", required_permissions=("read_private",), cost=0),
            _tool("private-alt", required_permissions=("read_private",), cost=1),
        ),
        permissions=("read_private",),
        permission_values=(None,),
    )

    assert solve(scenario) == INSUFFICIENT_INFORMATION


def test_every_permission_completion_without_eligible_tools_returns_none() -> None:
    scenario = _scenario(
        (
            _tool("wrong-capability", capabilities=("summarize",)),
            _tool("wrong-input", input_types=("image",)),
        ),
        capabilities=("search", "summarize"),
        input_types=("text", "image"),
        permissions=("p1", "p2"),
        permission_values=(None, None),
    )

    assert solve(scenario) == NO_ELIGIBLE_TOOL


def test_multiple_unknown_permissions_with_clear_winner_remain_clear() -> None:
    scenario = _scenario(
        (
            _tool("base", cost=1),
            _tool("optional", required_permissions=("p1", "p2"), cost=5),
        ),
        permissions=("p1", "p2"),
        permission_values=(None, None),
    )

    assert solve(scenario) == "base"


def test_multiple_unknown_permissions_with_mixed_outcomes_are_insufficient() -> None:
    scenario = _scenario(
        (
            _tool("first", required_permissions=("p1",), cost=1),
            _tool("second", required_permissions=("p2",), cost=2),
        ),
        permissions=("p1", "p2"),
        permission_values=(None, None),
    )

    assert solve(scenario) == INSUFFICIENT_INFORMATION


def test_higher_cost_ties_are_harmless_when_the_minimum_is_unique() -> None:
    scenario = _scenario(
        (
            _tool("winner", cost=0),
            _tool("tie-a", cost=4),
            _tool("tie-b", cost=4),
        )
    )

    assert solve(scenario) == "winner"


def test_lowest_cost_tie_in_last_unknown_completion_is_rejected() -> None:
    scenario = _scenario(
        (
            _tool("requires-p1", required_permissions=("p1",), cost=1),
            _tool("requires-p2", required_permissions=("p2",), cost=1),
        ),
        permissions=("p1", "p2"),
        permission_values=(None, None),
    )

    with pytest.raises(ValueError, match="multiple tools tie at lowest cost 1"):
        solve(scenario)


def test_reordering_domains_tools_and_permission_axes_preserves_answer() -> None:
    first = _scenario(
        (
            _tool(
                "low", capabilities=("search", "summarize"), required_permissions=("p2",), cost=1
            ),
            _tool("fallback", capabilities=("summarize", "search"), cost=3),
        ),
        capabilities=("search", "summarize"),
        permissions=("p1", "p2"),
        permission_values=(None, True),
    )
    reordered = _scenario(
        (
            _tool("fallback", capabilities=("search", "summarize"), cost=3),
            _tool(
                "low", capabilities=("summarize", "search"), required_permissions=("p2",), cost=1
            ),
        ),
        capabilities=("summarize", "search"),
        permissions=("p2", "p1"),
        permission_values=(True, None),
    )

    assert solve(first) == "low"
    assert solve(reordered) == "low"


def test_solver_does_not_mutate_scenario() -> None:
    scenario = _scenario(
        (_tool("a", cost=0), _tool("b", required_permissions=("p",), cost=1)),
        permissions=("p",),
        permission_values=(None,),
    )
    before = repr(scenario)

    solve(scenario)

    assert repr(scenario) == before


def test_tool_fields_require_tuples_and_unique_nonblank_entries() -> None:
    with pytest.raises(TypeError, match="capabilities must be a tuple"):
        ToolSpec("bad", ["search"], ("text",), (), 1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="capabilities must be unique"):
        ToolSpec("bad", ("search", "search"), ("text",), (), 1)

    with pytest.raises(ValueError, match="must not be blank"):
        ToolSpec("bad", ("   ",), ("text",), (), 1)


def test_tool_cost_requires_an_exact_bounded_integer() -> None:
    for cost in (True, 1.0, "1"):
        with pytest.raises(TypeError, match="cost must be an integer"):
            ToolSpec("bad", ("search",), ("text",), (), cost)  # type: ignore[arg-type]
    for cost in (-1, 1_000_001):
        with pytest.raises(ValueError, match="cost must be between"):
            ToolSpec("bad", ("search",), ("text",), (), cost)


def test_identifier_limit_is_256_and_content_is_preserved() -> None:
    maximum = "x" * 256
    tool_id = 'lookup "quoted"\n雪'
    capability = "café\nsearch"
    accepted = ToolSpec(maximum, (capability,), ("text",), (), 0)
    scenario = _scenario(
        (
            accepted,
            _tool("fallback", capabilities=(capability,), cost=1),
        ),
        capabilities=(capability,),
        required_capabilities=(capability,),
    )
    quoted = _scenario(
        (
            _tool(tool_id, capabilities=(capability,), cost=0),
            _tool("fallback", capabilities=(capability,), cost=1),
        ),
        capabilities=(capability,),
        required_capabilities=(capability,),
    )

    assert answer_ids(scenario)[0] == maximum
    assert solve(quoted) == tool_id
    with pytest.raises(ValueError, match="at most 256 characters"):
        ToolSpec(maximum + "x", ("search",), ("text",), (), 1)


def test_reserved_tool_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="reserved answer ID"):
        _tool(NO_ELIGIBLE_TOOL)
    with pytest.raises(ValueError, match="reserved answer ID"):
        _tool(INSUFFICIENT_INFORMATION)


def test_scenario_requires_bounded_unique_domains_and_a_valid_query() -> None:
    valid = _scenario((_tool("a"), _tool("b")))
    with pytest.raises(TypeError, match="capabilities must be a tuple"):
        replace(valid, capabilities=["search"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capabilities must contain between 1 and 4"):
        replace(valid, capabilities=())
    with pytest.raises(ValueError, match="permissions must be unique"):
        replace(valid, permissions=("p", "p"))
    with pytest.raises(ValueError, match="required capabilities must contain between"):
        replace(valid, required_capabilities=())
    with pytest.raises(ValueError, match="required capabilities must belong"):
        replace(valid, required_capabilities=("missing",))
    with pytest.raises(ValueError, match="input type must belong"):
        replace(valid, input_type="image")
    with pytest.raises(ValueError, match="tool IDs must be unique"):
        replace(valid, tools=(_tool("same"), _tool("same")))
    with pytest.raises(ValueError, match="tools must contain between 2 and 6"):
        replace(valid, tools=(_tool("only"),))
    with pytest.raises(TypeError, match="tools must contain ToolSpec values"):
        replace(valid, tools=(object(), _tool("other")))  # type: ignore[arg-type]


def test_tool_entries_must_fit_declared_capability_input_and_permission_domains() -> None:
    base = _scenario(
        (_tool("a"), _tool("b")),
        permissions=("p",),
        permission_values=(False,),
    )
    invalid_entries = (
        (
            _tool("a", capabilities=("other",)),
            _tool("b"),
        ),
        (
            _tool("a", input_types=("image",)),
            _tool("b"),
        ),
        (
            _tool("a", required_permissions=("other",)),
            _tool("b"),
        ),
    )
    messages = (
        "tool capabilities must belong",
        "tool input types must belong",
        "tool permissions must belong",
    )

    for tools, message in zip(invalid_entries, messages, strict=True):
        with pytest.raises(ValueError, match=message):
            replace(base, tools=tools)


def test_permission_values_require_exact_bool_or_none_and_match_domain() -> None:
    base = _scenario(
        (_tool("a"), _tool("b")),
        permissions=("p",),
        permission_values=(False,),
    )

    with pytest.raises(TypeError, match="permission values must be a tuple"):
        replace(base, permission_values=[True])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="match the permission count"):
        replace(base, permission_values=())
    for invalid in (1, "true", False + 0.5):
        with pytest.raises(TypeError, match="booleans or None"):
            replace(base, permission_values=(invalid,))  # type: ignore[arg-type]


def test_malformed_nested_tool_types_fail_before_solving() -> None:
    with pytest.raises(TypeError, match="input types must be a tuple"):
        ToolSpec("bad", ("search",), ["text"], (), 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="required permissions must be a tuple"):
        ToolSpec("bad", ("search",), ("text",), ["p"], 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="capabilities must be a string"):
        ToolSpec("bad", (1,), ("text",), (), 1)  # type: ignore[arg-type]
