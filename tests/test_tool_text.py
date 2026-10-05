"""Behavior tests for tool-selection prompt rendering and parsing."""

import json
from dataclasses import replace

import pytest

from reflex_decisions.rendering import render_prompt
from reflex_decisions.tool_rules import ToolScenario, ToolSpec, answer_ids
from reflex_decisions.tool_text import TOOL_TEMPLATES, render_tool_request
from reflex_decisions.tool_text_parser import (
    ParsedToolPrompt,
    parse_tool_prompt,
    verify_tool_request,
)

HANDWRITTEN_PROMPT = (
    json.dumps(
        {
            "context": (
                "The declarations below list every capability, input type, and "
                "permission in this task. "
                "The permissions list is complete. Each permission_values entry matches "
                "the permission at the same position. For each permission, true means granted, "
                "false means "
                "denied, and null means unknown. "
                "A tool is eligible only when it has every required capability, "
                "accepts the input type, "
                "and every permission it requires is granted. "
                "Costs are relative abstract units, not dollars. "
                "Choose the eligible tool with the unique lowest cost. "
                "Choose No eligible tool when none qualifies. "
                "If permissions are unknown, evaluate every Boolean completion: "
                "use a tool or No eligible tool only if every completion has the same outcome; "
                "otherwise choose Insufficient information. "
                "A tie for lowest eligible cost in any completion makes the scenario invalid."
                "\n\nTool data:\n"
                '{"capabilities":["lookup","transform"],"input_types":["text","image"],'
                '"permissions":["network"],"tools":[{"id":"search","capabilities":["lookup"],'
                '"input_types":["text"],"required_permissions":[],"cost":0},{"id":"writer",'
                '"capabilities":["transform"],"input_types":["text"],"required_permissions":["network"],'
                '"cost":3}],"required_capabilities":["lookup"],"input_type":"text",'
                '"permission_values":[null]}'
            ),
            "question": (
                "Which tool-selection result follows from every legal permission completion?"
            ),
            "options": [
                {"symbol": "A", "label": 'Use tool "search"', "description": None},
                {"symbol": "B", "label": 'Use tool "writer"', "description": None},
                {"symbol": "C", "label": "No eligible tool", "description": None},
                {"symbol": "D", "label": "Insufficient information", "description": None},
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    + "\nAnswer:\n"
)


def _scenario() -> ToolScenario:
    return ToolScenario(
        capabilities=("lookup", "transform"),
        input_types=("text", "image"),
        permissions=("network",),
        tools=(
            ToolSpec("search", ("lookup",), ("text",), (), 0),
            ToolSpec("writer", ("transform",), ("text",), ("network",), 3),
        ),
        required_capabilities=("lookup",),
        input_type="text",
        permission_values=(None,),
    )


def test_renderer_rejects_unknown_templates_and_invalid_option_orders() -> None:
    scenario = _scenario()
    all_ids = answer_ids(scenario)
    invalid_orders = (
        all_ids[:-1],
        (all_ids[0], *all_ids[1:-1], all_ids[0]),
        (*all_ids[:-1], "unknown"),
        list(all_ids),
    )
    with pytest.raises(ValueError, match="unknown tool template"):
        render_tool_request(scenario, template_id="tool-unknown")
    for order in invalid_orders:
        with pytest.raises(ValueError, match="every answer ID exactly once"):
            render_tool_request(scenario, option_order=order)  # type: ignore[arg-type]


def test_renderer_emits_exact_tool_facts_and_full_menu() -> None:
    scenario = _scenario()

    request = render_tool_request(scenario)

    expected_facts = {
        "capabilities": ["lookup", "transform"],
        "input_types": ["text", "image"],
        "permissions": ["network"],
        "tools": [
            {
                "id": "search",
                "capabilities": ["lookup"],
                "input_types": ["text"],
                "required_permissions": [],
                "cost": 0,
            },
            {
                "id": "writer",
                "capabilities": ["transform"],
                "input_types": ["text"],
                "required_permissions": ["network"],
                "cost": 3,
            },
        ],
        "required_capabilities": ["lookup"],
        "input_type": "text",
        "permission_values": [None],
    }
    assert request.context.endswith(
        "\n\nTool data:\n" + json.dumps(expected_facts, ensure_ascii=False, separators=(",", ":"))
    )
    assert [option.id for option in request.options] == [
        "search",
        "writer",
        "no_eligible_tool",
        "insufficient_information",
    ]
    assert [option.label for option in request.options] == [
        'Use tool "search"',
        'Use tool "writer"',
        "No eligible tool",
        "Insufficient information",
    ]
    assert '"answer"' not in render_prompt(request)


def test_every_registered_template_states_the_complete_answer_semantics() -> None:
    assert set(TOOL_TEMPLATES) == {
        "tool-v1",
        "tool-development-v1",
        "tool-calibration-v1",
        "tool-sealed-v1",
    }
    for prefix, _ in TOOL_TEMPLATES.values():
        assert prefix.endswith("\n\nTool data:\n")
        wording = prefix.lower().replace("-", " ")
        for required_term in (
            "complete",
            "permission",
            "each permission_values entry matches the permission at the same position",
            "true",
            "false",
            "denied",
            "null",
            "unknown",
            "capabilit",
            "input type",
            "unique",
            "lowest cost",
            "abstract",
            "dollars",
            "no eligible tool",
            "insufficient information",
            "every",
            "completion",
            "tie",
            "invalid",
        ):
            assert required_term in wording
        assert any(
            phrase in wording
            for phrase in ("same outcome", "all completions agree", "all completion results match")
        )
        assert any(term in wording for term in ("otherwise", "differ", "disagree"))


def test_renderer_preserves_a_reversed_menu_order() -> None:
    scenario = _scenario()

    request = render_tool_request(scenario, option_order=tuple(reversed(answer_ids(scenario))))

    assert tuple(option.id for option in request.options) == tuple(reversed(answer_ids(scenario)))


def test_parser_reads_handwritten_render_prompt_compatible_fixture() -> None:
    parsed = parse_tool_prompt(HANDWRITTEN_PROMPT)

    assert parsed == ParsedToolPrompt(
        scenario=_scenario(),
        option_ids=(
            "search",
            "writer",
            "no_eligible_tool",
            "insufficient_information",
        ),
        template_id="tool-v1",
    )


def _outer(prompt: str = HANDWRITTEN_PROMPT) -> dict[str, object]:
    return json.loads(prompt[: -len("\nAnswer:\n")])


def _encode_outer(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\nAnswer:\n"


def test_parser_round_trips_every_template_and_reversed_menu() -> None:
    scenario = _scenario()

    for template_id in TOOL_TEMPLATES:
        request = render_tool_request(
            scenario,
            option_order=tuple(reversed(answer_ids(scenario))),
            template_id=template_id,
        )
        parsed = parse_tool_prompt(render_prompt(request))

        assert parsed == ParsedToolPrompt(
            scenario,
            tuple(reversed(answer_ids(scenario))),
            template_id,
        )
        verify_tool_request(request, scenario)


def test_parser_preserves_quoted_newline_and_unicode_tool_ids() -> None:
    unusual_id = 'lookup "quoted"\n🧭'
    scenario = ToolScenario(
        capabilities=("lookup",),
        input_types=("text",),
        permissions=(),
        tools=(
            ToolSpec(unusual_id, ("lookup",), ("text",), (), 0),
            ToolSpec("backup", ("lookup",), ("text",), (), 1),
        ),
        required_capabilities=("lookup",),
        input_type="text",
        permission_values=(),
    )

    request = render_tool_request(scenario)
    parsed = parse_tool_prompt(render_prompt(request))

    assert parsed.scenario == scenario
    assert parsed.option_ids == answer_ids(scenario)
    assert request.options[0].label == "Use tool " + json.dumps(unusual_id, ensure_ascii=False)


def test_renderer_and_parser_preserve_native_boolean_and_integer_json_types() -> None:
    scenario = ToolScenario(
        capabilities=("lookup",),
        input_types=("text",),
        permissions=("network",),
        tools=(
            ToolSpec("search", ("lookup",), ("text",), (), 0),
            ToolSpec("backup", ("lookup",), ("text",), ("network",), 9),
        ),
        required_capabilities=("lookup",),
        input_type="text",
        permission_values=(False,),
    )

    request = render_tool_request(scenario)
    parsed = parse_tool_prompt(render_prompt(request))
    outer = json.loads(render_prompt(request)[: -len("\nAnswer:\n")])
    context = outer["context"]
    assert isinstance(context, str)
    prefix, _ = TOOL_TEMPLATES["tool-v1"]
    facts = json.loads(context[len(prefix) :])

    assert type(facts["tools"][0]["cost"]) is int
    assert type(facts["permission_values"][0]) is bool
    assert facts["permission_values"][0] is False
    assert parsed.scenario == scenario


def test_parser_rejects_missing_and_extra_fact_keys() -> None:
    outer = _outer()
    context = outer["context"]
    assert isinstance(context, str)
    outer["context"] = context.replace('"input_type":"text",', "", 1)
    with pytest.raises(ValueError, match="missing or extra keys"):
        parse_tool_prompt(_encode_outer(outer))

    outer = _outer()
    context = outer["context"]
    assert isinstance(context, str)
    outer["context"] = context.replace(
        '"input_type":"text",', '"input_type":"text","unexpected":1,', 1
    )
    with pytest.raises(ValueError, match="missing or extra keys"):
        parse_tool_prompt(_encode_outer(outer))


def test_parser_rejects_duplicate_json_keys_at_outer_and_nested_levels() -> None:
    outer_prompt = HANDWRITTEN_PROMPT.replace('{"context":', '{"context":"duplicate","context":', 1)
    with pytest.raises(ValueError, match="duplicate JSON key: context"):
        parse_tool_prompt(outer_prompt)

    outer = _outer()
    context = outer["context"]
    assert isinstance(context, str)
    outer["context"] = context.replace('"cost":0', '"cost":0,"cost":1', 1)
    with pytest.raises(ValueError, match="duplicate JSON key: cost"):
        parse_tool_prompt(_encode_outer(outer))


def test_parser_rejects_integer_permissions_and_boolean_costs_without_coercion() -> None:
    outer = _outer()
    context = outer["context"]
    assert isinstance(context, str)
    outer["context"] = context.replace('"permission_values":[null]', '"permission_values":[1]')
    with pytest.raises(ValueError, match="permission_values"):
        parse_tool_prompt(_encode_outer(outer))

    outer = _outer()
    context = outer["context"]
    assert isinstance(context, str)
    outer["context"] = context.replace('"cost":0', '"cost":true', 1)
    with pytest.raises(ValueError, match="tool cost must be an integer"):
        parse_tool_prompt(_encode_outer(outer))


def test_parser_rejects_missing_extra_duplicate_or_mislabeled_options() -> None:
    outer = _outer()
    options = outer["options"]
    assert isinstance(options, list)
    outer["options"] = options[:-1]
    with pytest.raises(ValueError, match="include every"):
        parse_tool_prompt(_encode_outer(outer))

    outer = _outer()
    options = outer["options"]
    assert isinstance(options, list)
    outer["options"] = [*options, options[-1]]
    with pytest.raises(ValueError, match="include every"):
        parse_tool_prompt(_encode_outer(outer))

    outer = _outer()
    options = outer["options"]
    assert isinstance(options, list)
    options[1]["label"] = options[0]["label"]
    with pytest.raises(ValueError, match="unknown or duplicate"):
        parse_tool_prompt(_encode_outer(outer))

    outer = _outer()
    options = outer["options"]
    assert isinstance(options, list)
    options[0]["symbol"] = "B"
    with pytest.raises(ValueError, match="invalid symbols"):
        parse_tool_prompt(_encode_outer(outer))


def test_verifier_rejects_changed_cost_capability_or_permission_facts() -> None:
    scenario = _scenario()
    changed_scenarios = (
        replace(
            scenario,
            tools=(ToolSpec("search", ("lookup",), ("text",), (), 1), scenario.tools[1]),
        ),
        replace(
            scenario,
            tools=(
                ToolSpec("search", ("lookup", "transform"), ("text",), (), 0),
                scenario.tools[1],
            ),
        ),
        replace(
            scenario,
            permissions=("network", "filesystem"),
            permission_values=(None, False),
        ),
    )

    for changed in changed_scenarios:
        with pytest.raises(ValueError, match="facts do not match"):
            verify_tool_request(render_tool_request(changed), scenario)


def test_parser_requires_exact_outer_framing_template_and_question() -> None:
    with pytest.raises(ValueError, match="answer suffix"):
        parse_tool_prompt(HANDWRITTEN_PROMPT.rstrip())

    outer = _outer()
    context = outer["context"]
    assert isinstance(context, str)
    outer["context"] = context.replace("The declarations below", "Other instructions", 1)
    with pytest.raises(ValueError, match="unrecognized"):
        parse_tool_prompt(_encode_outer(outer))

    outer = _outer()
    outer["question"] = "Choose anything."
    with pytest.raises(ValueError, match="question is unrecognized"):
        parse_tool_prompt(_encode_outer(outer))
