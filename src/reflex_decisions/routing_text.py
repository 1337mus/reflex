"""Plain-text protocol wording for support-routing requests."""

import json

from .routing_rules import (
    INSUFFICIENT_INFORMATION,
    NO_ROUTE,
    RoutingScenario,
    answer_ids,
)
from .schema import DecisionRequest, Option

ROUTING_TEMPLATES: dict[str, tuple[str, str]] = {
    "routing-v1": (
        "Apply the complete routing rules below. A fully known key with no matching "
        "rule has no route. "
        "For missing features, consider every declared value. "
        "Return one destination or No route applies only when every completion has that "
        "same outcome; "
        "otherwise return Insufficient information. "
        "Each feature's listed values are its complete set of legal values. "
        "Rule values and query values "
        "follow the feature order shown. A null query value means that feature is missing."
        "\n\nRouting data:\n",
        "Which destination or routing outcome applies?",
    ),
    "routing-development-v1": (
        "Use only the routing table and legal feature values below. "
        "Each feature lists its complete set of legal values. "
        "Values in each rule and query follow the feature order. null marks a missing query value. "
        "A complete query uses its exact matching rule's destination; without a match, "
        "No route applies. "
        "For missing values, check all allowed completions. "
        "Choose their common destination or No route applies if every completion agrees. "
        "If outcomes differ, choose Insufficient information.\n\nRouting data:\n",
        "Which routing result follows from this table?",
    ),
    "routing-calibration-v1": (
        "The table below is the full routing policy. Each feature lists all of its legal values. "
        "Rule and query entries use the listed feature order; null means a query value is missing. "
        "For each complete query, use its exact matching rule's destination, or No route applies "
        "when none matches. Consider every legal completion of missing values. "
        "Select the shared result if all completions agree; otherwise select "
        "Insufficient information."
        "\n\nRouting data:\n",
        "What result does the full routing policy require?",
    ),
    "routing-sealed-v1": (
        "Decide from the complete policy below. The feature lists contain every permitted value. "
        "Read rule values and query values in feature order. Treat null as missing information. "
        "Return the exact matching rule's destination or No route applies when no rule matches. "
        "If values are missing, consider every permitted completion. "
        "Return the same destination or No route applies only if all completed queries "
        "have that outcome; "
        "return Insufficient information if any outcomes differ.\n\nRouting data:\n",
        "What can this policy determine for the query?",
    ),
}
ROUTING_CONTEXT_PREFIX, ROUTING_QUESTION = ROUTING_TEMPLATES["routing-v1"]

_SPECIAL_LABELS = {
    NO_ROUTE: "No route applies",
    INSUFFICIENT_INFORMATION: "Insufficient information",
}


def render_routing_request(
    scenario: RoutingScenario,
    *,
    option_order: tuple[str, ...] | None = None,
    template_id: str = "routing-v1",
) -> DecisionRequest:
    """Render scenario facts and the full answer menu as a model request."""
    if not isinstance(template_id, str) or template_id not in ROUTING_TEMPLATES:
        raise ValueError(f"unknown routing template: {template_id!r}")
    context_prefix, question = ROUTING_TEMPLATES[template_id]
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
    context = context_prefix + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    labels = {
        destination_id: "Route to " + json.dumps(destination_id, ensure_ascii=False)
        for destination_id in scenario.destination_ids
    }
    labels.update(_SPECIAL_LABELS)
    options = tuple(Option(id=option_id, label=labels[option_id]) for option_id in selected_order)
    return DecisionRequest(context=context, question=question, options=options)
