"""Visible prompt wording for synthetic tool-selection decisions."""

import json

from .schema import DecisionRequest, Option
from .tool_rules import (
    INSUFFICIENT_INFORMATION,
    NO_ELIGIBLE_TOOL,
    ToolScenario,
    answer_ids,
)

TOOL_TEMPLATES: dict[str, tuple[str, str]] = {
    "tool-v1": (
        "The declarations below list every capability, input type, and permission in this task. "
        "The permissions list is complete. Each permission_values entry matches the permission "
        "at the same position. For each permission, true means granted, false means "
        "denied, and null means unknown. A tool is eligible only when it has every required "
        "capability, accepts the input type, and every permission it requires is granted. Costs "
        "are relative abstract units, not dollars. Choose the eligible tool with the unique "
        "lowest cost. Choose No eligible tool when none qualifies. If permissions are unknown, "
        "evaluate every Boolean completion: use a tool or No eligible tool only if every "
        "completion has the same outcome; otherwise choose Insufficient information. A tie for "
        "lowest eligible cost in any completion makes the scenario invalid.\n\nTool data:\n",
        "Which tool-selection result follows from every legal permission completion?",
    ),
    "tool-development-v1": (
        "Treat the listed capability, input-type, and permission domains as complete, including "
        "the full declared permission list. Each permission_values entry matches the permission "
        "at the same position. A permission value of true is granted, false is "
        "denied, and null is unknown. A tool qualifies exactly when it supplies all required "
        "capabilities, supports the requested input type, and has all required permissions "
        "granted. Lower costs are preferred; costs are abstract relative units, not dollars. "
        "Return the eligible tool with the unique lowest cost, or No eligible tool if "
        "there is none. For unknown permissions, consider every Boolean completion. Return that "
        "completion outcome only when all completions agree; if outcomes differ, return "
        "Insufficient information. A lowest-cost tie in any completion invalidates the task."
        "\n\nTool data:\n",
        "What answer is shared by all legal completions of the permissions?",
    ),
    "tool-calibration-v1": (
        "Use only the complete declarations shown: every capability, input type, and permission "
        "is listed. Each permission_values entry matches the permission at the same position. "
        "true means a permission is granted, false means it is denied, and null "
        "means it is unknown. To "
        "qualify, a tool must provide every required capability, accept the requested input "
        "type, and have every required permission granted. Costs are relative abstract units, "
        "not dollars; among eligible tools, the unique lowest cost wins. When none qualifies, "
        "the result is No eligible tool. Expand each unknown permission over both Boolean "
        "values and check every completion. Select a tool or No eligible tool only if all "
        "completion results match; if they disagree, select Insufficient information. If any "
        "completion has tied lowest eligible costs, the scenario is invalid.\n\nTool data:\n",
        "Which result does the complete tool policy determine?",
    ),
    "tool-sealed-v1": (
        "The data gives the complete sets of capabilities, input types, and declared permissions. "
        "Each permission_values entry matches the permission at the same position. "
        "Permission true means granted, false means denied, and null means unknown; denied is "
        "not unknown. A tool may be chosen only if it covers every required capability, accepts "
        "this input type, and each permission it requires is granted. Compare eligible tools by "
        "their relative abstract costs (not dollars) and select the unique lowest-cost tool. "
        "When no tool qualifies, use No eligible tool. For unknown permissions, inspect every "
        "Boolean completion; return a tool or No eligible tool only when all completions agree, "
        "and otherwise return Insufficient information. Any lowest-cost tie in any completion "
        "makes the scenario invalid.\n\nTool data:\n",
        "What can be concluded from the declared tools and permissions?",
    ),
}

_SPECIAL_LABELS = {
    NO_ELIGIBLE_TOOL: "No eligible tool",
    INSUFFICIENT_INFORMATION: "Insufficient information",
}


def render_tool_request(
    scenario: ToolScenario,
    *,
    option_order: tuple[str, ...] | None = None,
    template_id: str = "tool-v1",
) -> DecisionRequest:
    """Render tool-selection facts and all answer choices as a model request."""
    if type(template_id) is not str or template_id not in TOOL_TEMPLATES:
        raise ValueError(f"unknown tool template: {template_id!r}")
    context_prefix, question = TOOL_TEMPLATES[template_id]
    all_ids = answer_ids(scenario)
    selected_order = all_ids if option_order is None else option_order
    if (
        type(selected_order) is not tuple
        or any(type(option_id) is not str for option_id in selected_order)
        or len(selected_order) != len(all_ids)
        or set(selected_order) != set(all_ids)
    ):
        raise ValueError("option_order must contain every answer ID exactly once")

    facts = {
        "capabilities": list(scenario.capabilities),
        "input_types": list(scenario.input_types),
        "permissions": list(scenario.permissions),
        "tools": [
            {
                "id": tool.id,
                "capabilities": list(tool.capabilities),
                "input_types": list(tool.input_types),
                "required_permissions": list(tool.required_permissions),
                "cost": tool.cost,
            }
            for tool in scenario.tools
        ],
        "required_capabilities": list(scenario.required_capabilities),
        "input_type": scenario.input_type,
        "permission_values": list(scenario.permission_values),
    }
    context = context_prefix + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    labels = {
        tool.id: "Use tool " + json.dumps(tool.id, ensure_ascii=False) for tool in scenario.tools
    }
    labels.update(_SPECIAL_LABELS)
    options = tuple(Option(id=option_id, label=labels[option_id]) for option_id in selected_order)
    return DecisionRequest(context=context, question=question, options=options)
