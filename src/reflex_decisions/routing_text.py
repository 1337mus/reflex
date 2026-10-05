"""Plain-text protocol wording for support-routing requests."""

import json

from .routing_rules import (
    INSUFFICIENT_INFORMATION,
    NO_ROUTE,
    RoutingScenario,
    answer_ids,
)
from .schema import DecisionRequest, Option

ROUTING_CONTEXT_PREFIX = (
    "Apply the complete routing rules below. A fully known key with no matching rule has no route. "
    "For missing features, consider every declared value. "
    "Return one destination or No route applies only when every completion has that same outcome; "
    "otherwise return Insufficient information. "
    "Each feature's listed values are its complete set of legal values. "
    "Rule values and query values "
    "follow the feature order shown. A null query value means that feature is missing."
    "\n\nRouting data:\n"
)
ROUTING_QUESTION = "Which destination or routing outcome applies?"

_SPECIAL_LABELS = {
    NO_ROUTE: "No route applies",
    INSUFFICIENT_INFORMATION: "Insufficient information",
}


def render_routing_request(
    scenario: RoutingScenario,
    *,
    option_order: tuple[str, ...] | None = None,
) -> DecisionRequest:
    """Render scenario facts and the full answer menu as a model request."""
    all_ids = answer_ids(scenario)
    selected_order = all_ids if option_order is None else option_order
    if (
        not isinstance(selected_order, tuple)
        or any(not isinstance(option_id, str) for option_id in selected_order)
        or len(selected_order) != len(all_ids)
        or set(selected_order) != set(all_ids)
    ):
        raise ValueError("option_order must contain every answer ID exactly once")

    facts = {
        "features": [
            {"name": feature.name, "values": list(feature.values)} for feature in scenario.features
        ],
        "destination_ids": list(scenario.destination_ids),
        "rules": [
            {"values": list(rule.values), "destination_id": rule.destination_id}
            for rule in scenario.rules
        ],
        "query": list(scenario.query),
    }
    context = ROUTING_CONTEXT_PREFIX + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    labels = {
        destination_id: "Route to " + json.dumps(destination_id, ensure_ascii=False)
        for destination_id in scenario.destination_ids
    }
    labels.update(_SPECIAL_LABELS)
    options = tuple(Option(id=option_id, label=labels[option_id]) for option_id in selected_order)
    return DecisionRequest(context=context, question=ROUTING_QUESTION, options=options)
