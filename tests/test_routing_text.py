import json

import pytest

from reflex_decisions import rendering, routing_text, routing_text_parser
from reflex_decisions.routing_rules import Feature, RouteRule, RoutingScenario, answer_ids, solve
from reflex_decisions.schema import DecisionRequest, Option

HANDWRITTEN_PROMPT = (
    r"""{"context":"Apply the complete routing rules below. A fully known key wi"""
    r"""th no matching rule has no route. For missing features, consider every d"""
    r"""eclared value. Return one destination or No route applies only when ever"""
    r"""y completion has that same outcome; otherwise return Insufficient inform"""
    r"""ation. Each feature's listed values are its complete set of legal values"""
    r""". Rule values and query values follow the feature order shown. A null qu"""
    r"""ery value means that feature is missing.\n\nRouting data:\n{\"features"""
    r"""\":[{\"name\":\"tier\",\"values\":[\"standard\",\"urgent\"]},{\"name\":"""
    r"""\"region\",\"values\":[\"west\",\"east\"]}],\"destination_ids\":[\"queue"""
    r"""-a\",\"queue-b\"],\"rules\":[{\"values\":[\"urgent\",\"west\"],\"destina"""
    r"""tion_id\":\"queue-b\"}],\"query\":[\"urgent\",\"west\"]}","question":"Wh"""
    r"""ich destination or routing outcome applies?","options":[{"symbol":"A","l"""
    r"""abel":"Route to \"queue-a\"","description":null},{"symbol":"B","label":"""
    r""""Route to \"queue-b\"","description":null},{"symbol":"C","label":"No rou"""
    r"""te applies","description":null},{"symbol":"D","label":"Insufficient info"""
    r"""rmation","description":null}]}
Answer:
"""
)

_ANSWER_SUFFIX = "\nAnswer:\n"


def _prompt_with_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + _ANSWER_SUFFIX


def _payload_from_prompt(prompt: str) -> dict[str, object]:
    return json.loads(prompt[: -len(_ANSWER_SUFFIX)])


def _payload_with_context(context: str) -> dict[str, object]:
    payload = _payload_from_prompt(HANDWRITTEN_PROMPT)
    payload["context"] = context
    return payload


def _block_from_prompt(prompt: str) -> dict[str, object]:
    payload = _payload_from_prompt(prompt)
    context = payload["context"]
    assert isinstance(context, str)
    return json.loads(context[len(routing_text.ROUTING_CONTEXT_PREFIX) :])


def _prompt_with_block(block: dict[str, object]) -> str:
    context = routing_text.ROUTING_CONTEXT_PREFIX + json.dumps(
        block, ensure_ascii=False, separators=(",", ":")
    )
    return _prompt_with_payload(_payload_with_context(context))


def _handwritten_scenario() -> RoutingScenario:
    return RoutingScenario(
        features=(
            Feature("tier", ("standard", "urgent")),
            Feature("region", ("west", "east")),
        ),
        destination_ids=("queue-a", "queue-b"),
        rules=(RouteRule(("urgent", "west"), "queue-b"),),
        query=("urgent", "west"),
    )


def test_parser_recovers_handwritten_model_prompt_without_renderer() -> None:
    parse_prompt = getattr(routing_text_parser, "parse_routing_prompt", None)
    assert callable(parse_prompt), "routing prompt parser API must exist"

    parsed = parse_prompt(HANDWRITTEN_PROMPT)

    assert parsed.scenario == RoutingScenario(
        features=(
            Feature("tier", ("standard", "urgent")),
            Feature("region", ("west", "east")),
        ),
        destination_ids=("queue-a", "queue-b"),
        rules=(RouteRule(("urgent", "west"), "queue-b"),),
        query=("urgent", "west"),
    )
    assert parsed.option_ids == ("queue-a", "queue-b", "no_route", "insufficient_information")


def test_default_template_keeps_handwritten_prompt_bytes() -> None:
    request = routing_text.render_routing_request(_handwritten_scenario())

    assert rendering.render_prompt(request) == HANDWRITTEN_PROMPT


def test_registered_templates_roundtrip_same_facts_and_reversed_menu() -> None:
    fixture = routing_text_parser.parse_routing_prompt(HANDWRITTEN_PROMPT)
    template_ids = (
        "routing-v1",
        "routing-development-v1",
        "routing-calibration-v1",
        "routing-sealed-v1",
    )
    requested_order = tuple(reversed(answer_ids(fixture.scenario)))

    for template_id in template_ids:
        request = routing_text.render_routing_request(
            fixture.scenario,
            option_order=requested_order,
            template_id=template_id,
        )
        parsed = routing_text_parser.parse_routing_prompt(rendering.render_prompt(request))

        assert parsed.scenario == fixture.scenario
        assert parsed.option_ids == requested_order
        assert parsed.template_id == template_id


def test_templates_are_distinct_and_default_aliases_remain_compatible() -> None:
    expected_ids = {
        "routing-v1",
        "routing-development-v1",
        "routing-calibration-v1",
        "routing-sealed-v1",
    }
    templates = routing_text.ROUTING_TEMPLATES
    assert set(templates) == expected_ids
    assert "Return the exact matching rule's destination" in templates["routing-sealed-v1"][0]
    assert templates["routing-v1"] == (
        routing_text.ROUTING_CONTEXT_PREFIX,
        routing_text.ROUTING_QUESTION,
    )
    prefixes = [prefix for prefix, _ in templates.values()]
    questions = [question for _, question in templates.values()]
    assert len(set(questions)) == len(questions)
    assert all(
        not first.startswith(second)
        for index, first in enumerate(prefixes)
        for other_index, second in enumerate(prefixes)
        if index != other_index
    )

    fixture = routing_text_parser.parse_routing_prompt(HANDWRITTEN_PROMPT)
    assert fixture.template_id == "routing-v1"
    assert (
        routing_text_parser.ParsedRoutingPrompt(fixture.scenario, fixture.option_ids).template_id
        == "routing-v1"
    )


def test_renderer_rejects_unknown_template_id() -> None:
    with pytest.raises(ValueError, match="unknown routing template"):
        routing_text.render_routing_request(_handwritten_scenario(), template_id="unknown-v1")


def test_parser_rejects_template_prefix_with_another_templates_question() -> None:
    request = routing_text.render_routing_request(
        _handwritten_scenario(), template_id="routing-development-v1"
    )
    payload = json.loads(rendering.render_prompt(request)[: -len(_ANSWER_SUFFIX)])
    payload["question"] = routing_text.ROUTING_QUESTION

    with pytest.raises(ValueError, match="question is unrecognized for routing template"):
        routing_text_parser.parse_routing_prompt(_prompt_with_payload(payload))


def test_renderer_keeps_complete_menu_when_order_is_overridden() -> None:
    render_request = getattr(routing_text, "render_routing_request", None)
    assert callable(render_request), "routing request renderer API must exist"
    scenario = RoutingScenario(
        features=(Feature("tier", ("standard", "urgent")),),
        destination_ids=("queue-a", "queue-b"),
        rules=(RouteRule(("urgent",), "queue-b"),),
        query=("urgent",),
    )
    requested_order = ("no_route", "queue-b", "insufficient_information", "queue-a")

    request = render_request(scenario, option_order=requested_order)
    visible_prompt = json.loads(rendering.render_prompt(request)[: -len("\nAnswer:\n")])

    assert isinstance(request, DecisionRequest)
    assert set(visible_prompt) == {"context", "question", "options"}
    assert tuple(option.id for option in request.options) == requested_order
    assert tuple(option.description for option in request.options) == (None,) * 4
    assert {option.id for option in request.options} == {
        "queue-a",
        "queue-b",
        "no_route",
        "insufficient_information",
    }
    assert visible_prompt["context"].startswith(routing_text.ROUTING_CONTEXT_PREFIX)
    block = json.loads(visible_prompt["context"][len(routing_text.ROUTING_CONTEXT_PREFIX) :])
    assert set(block) == {"features", "destination_ids", "rules", "query"}


def test_parser_preserves_reversed_option_order_independently_of_answer() -> None:
    scenario = RoutingScenario(
        features=(Feature("tier", ("standard", "urgent")),),
        destination_ids=("queue-a", "queue-b"),
        rules=(RouteRule(("urgent",), "queue-b"),),
        query=("urgent",),
    )
    requested_order = tuple(reversed(answer_ids(scenario)))
    request = routing_text.render_routing_request(scenario, option_order=requested_order)

    parsed = routing_text_parser.parse_routing_prompt(rendering.render_prompt(request))

    assert solve(scenario) == "queue-b"
    assert parsed.scenario == scenario
    assert parsed.option_ids == requested_order


def test_routing_prompts_roundtrip_every_outcome_with_the_full_menu() -> None:
    cases = (
        (
            RoutingScenario(
                (Feature("tier", ("standard", "urgent")),),
                ("queue-a", "queue-b"),
                (RouteRule(("urgent",), "queue-b"),),
                ("urgent",),
            ),
            "queue-b",
        ),
        (
            RoutingScenario(
                (Feature("tier", ("standard", "urgent")),),
                ("queue-a", "queue-b"),
                (RouteRule(("urgent",), "queue-b"),),
                ("standard",),
            ),
            "no_route",
        ),
        (
            RoutingScenario(
                (Feature("tier", ("standard", "urgent")),),
                ("queue-a", "queue-b"),
                (
                    RouteRule(("standard",), "queue-a"),
                    RouteRule(("urgent",), "queue-a"),
                ),
                (None,),
            ),
            "queue-a",
        ),
        (
            RoutingScenario(
                (Feature("tier", ("standard", "urgent")),),
                ("queue-a", "queue-b"),
                (
                    RouteRule(("standard",), "queue-a"),
                    RouteRule(("urgent",), "queue-b"),
                ),
                (None,),
            ),
            "insufficient_information",
        ),
    )

    for scenario, expected_outcome in cases:
        requested_order = tuple(reversed(answer_ids(scenario)))
        request = routing_text.render_routing_request(scenario, option_order=requested_order)
        parsed = routing_text_parser.parse_routing_prompt(rendering.render_prompt(request))

        assert solve(scenario) == expected_outcome
        assert parsed.scenario == scenario
        assert set(parsed.option_ids) == set(answer_ids(scenario))
        assert parsed.option_ids == requested_order


def test_json_escaping_and_special_like_destination_labels_roundtrip() -> None:
    scenario = RoutingScenario(
        features=(Feature('tier "Ω\n', ("No route applies", 'line\n"雪')),),
        destination_ids=("No route applies", "Insufficient information"),
        rules=(RouteRule(("No route applies",), "No route applies"),),
        query=("No route applies",),
    )

    request = routing_text.render_routing_request(scenario)
    prompt = rendering.render_prompt(request)
    outer = json.loads(prompt[: -len("\nAnswer:\n")])
    parsed = routing_text_parser.parse_routing_prompt(prompt)

    assert parsed.scenario == scenario
    assert {option.label for option in request.options} == {
        'Route to "No route applies"',
        'Route to "Insufficient information"',
        "No route applies",
        "Insufficient information",
    }
    block = json.loads(outer["context"][len(routing_text.ROUTING_CONTEXT_PREFIX) :])
    assert block["features"][0]["name"] == 'tier "Ω\n'


@pytest.mark.parametrize("change", ("rule", "query", "domain"))
def test_verifier_rejects_changed_facts_even_when_outcome_stays_the_same(change: str) -> None:
    base = RoutingScenario(
        features=(
            Feature("tier", ("standard", "urgent")),
            Feature("region", ("west", "east")),
        ),
        destination_ids=("queue-a", "queue-b"),
        rules=(
            RouteRule(("urgent", "west"), "queue-b"),
            RouteRule(("urgent", "east"), "queue-b"),
            RouteRule(("standard", "west"), "queue-a"),
        ),
        query=("urgent", "west"),
    )
    if change == "rule":
        changed = RoutingScenario(
            base.features,
            base.destination_ids,
            (*base.rules[:-1], RouteRule(("standard", "west"), "queue-b")),
            base.query,
        )
    elif change == "query":
        changed = RoutingScenario(
            base.features, base.destination_ids, base.rules, ("urgent", "east")
        )
    else:
        changed = RoutingScenario(
            base.features,
            base.destination_ids,
            base.rules,
            base.query,
        )
        changed = RoutingScenario(
            (base.features[0], Feature("region", ("west", "east", "north"))),
            changed.destination_ids,
            changed.rules,
            changed.query,
        )
    request = routing_text.render_routing_request(changed)

    assert solve(changed) == solve(base) == "queue-b"
    parsed = routing_text_parser.parse_routing_prompt(rendering.render_prompt(request))
    assert parsed.scenario == changed
    with pytest.raises(ValueError, match="rendered routing facts do not match"):
        routing_text_parser.verify_routing_request(request, base)


def test_verifier_accepts_exact_request_scenario_and_option_order() -> None:
    scenario = RoutingScenario(
        features=(Feature("tier", ("standard", "urgent")),),
        destination_ids=("queue-a", "queue-b"),
        rules=(RouteRule(("urgent",), "queue-b"),),
        query=(None,),
    )
    order = tuple(reversed(answer_ids(scenario)))
    request = routing_text.render_routing_request(scenario, option_order=order)

    assert routing_text_parser.verify_routing_request(request, scenario) is None


def test_verifier_rejects_request_ids_that_disagree_with_visible_labels() -> None:
    scenario = RoutingScenario(
        features=(Feature("tier", ("standard", "urgent")),),
        destination_ids=("queue-a", "queue-b"),
        rules=(RouteRule(("urgent",), "queue-b"),),
        query=("urgent",),
    )
    request = routing_text.render_routing_request(scenario)
    mismatched_options = tuple(
        Option(
            id=request.options[(index + 1) % len(request.options)].id,
            label=option.label,
        )
        for index, option in enumerate(request.options)
    )
    mismatched = DecisionRequest(
        context=request.context,
        question=request.question,
        options=mismatched_options,
    )

    with pytest.raises(
        ValueError, match="rendered routing choices do not match the request option IDs"
    ):
        routing_text_parser.verify_routing_request(mismatched, scenario)


def test_parser_rejects_duplicate_outer_and_routing_data_json_keys() -> None:
    duplicate_outer = HANDWRITTEN_PROMPT.replace(
        '{"context":', '{"context":"duplicate","context":', 1
    )
    payload = _payload_from_prompt(HANDWRITTEN_PROMPT)
    context = payload["context"]
    assert isinstance(context, str)
    data_start = len(routing_text.ROUTING_CONTEXT_PREFIX)
    duplicated_data = context[data_start:].replace('{"features":', '{"features":[],"features":', 1)
    duplicate_inner = _prompt_with_payload(
        _payload_with_context(routing_text.ROUTING_CONTEXT_PREFIX + duplicated_data)
    )

    prompts = (
        (duplicate_outer, "duplicate JSON key: context"),
        (duplicate_inner, "duplicate JSON key: features"),
    )
    for prompt, reason in prompts:
        with pytest.raises(ValueError, match=reason):
            routing_text_parser.parse_routing_prompt(prompt)


def test_parser_rejects_missing_or_extra_outer_and_inner_keys() -> None:
    outer_missing = _payload_from_prompt(HANDWRITTEN_PROMPT)
    del outer_missing["question"]
    outer_extra = _payload_from_prompt(HANDWRITTEN_PROMPT)
    outer_extra["label"] = "untrusted metadata"

    block_missing = _block_from_prompt(HANDWRITTEN_PROMPT)
    del block_missing["query"]
    block_extra = _block_from_prompt(HANDWRITTEN_PROMPT)
    block_extra["answer"] = "queue-b"
    feature_extra = _block_from_prompt(HANDWRITTEN_PROMPT)
    feature_extra["features"][0]["extra"] = "untrusted"
    rule_missing = _block_from_prompt(HANDWRITTEN_PROMPT)
    del rule_missing["rules"][0]["destination_id"]

    prompts = (
        (_prompt_with_payload(outer_missing), "prompt has missing or extra keys"),
        (_prompt_with_payload(outer_extra), "prompt has missing or extra keys"),
        (_prompt_with_block(block_missing), "routing data has missing or extra keys"),
        (_prompt_with_block(block_extra), "routing data has missing or extra keys"),
        (_prompt_with_block(feature_extra), "feature has missing or extra keys"),
        (_prompt_with_block(rule_missing), "rule has missing or extra keys"),
    )
    for prompt, reason in prompts:
        with pytest.raises(ValueError, match=reason):
            routing_text_parser.parse_routing_prompt(prompt)


def test_parser_rejects_malformed_menu_choices_symbols_and_descriptions() -> None:
    cases = (
        ("duplicate", "options contain an unknown or duplicate choice"),
        ("missing-special", "options contain an unknown or duplicate choice"),
        ("unknown", "options contain an unknown or duplicate choice"),
        ("missing-option", "options must include every routing choice"),
        ("extra-option", "options must include every routing choice"),
        ("duplicate-symbol", "options have invalid symbols or descriptions"),
        ("invalid-symbol", "options have invalid symbols or descriptions"),
        ("out-of-order-symbol", "options have invalid symbols or descriptions"),
        ("description", "options have invalid symbols or descriptions"),
        ("extra-option-field", "option has missing or extra keys"),
    )
    for mutation, reason in cases:
        payload = _payload_from_prompt(HANDWRITTEN_PROMPT)
        options = payload["options"]
        assert isinstance(options, list)
        if mutation == "duplicate":
            options[0]["label"] = options[1]["label"]
        elif mutation == "missing-special":
            options[3]["label"] = options[2]["label"]
        elif mutation == "unknown":
            options[0]["label"] = "Route somewhere else"
        elif mutation == "missing-option":
            options.pop()
        elif mutation == "extra-option":
            options.append(dict(options[0]))
            options[-1]["symbol"] = "E"
        elif mutation == "duplicate-symbol":
            options[1]["symbol"] = "A"
        elif mutation == "invalid-symbol":
            options[0]["symbol"] = "Q"
        elif mutation == "out-of-order-symbol":
            options[1]["symbol"], options[2]["symbol"] = "C", "B"
        elif mutation == "description":
            options[0]["description"] = "extra clue"
        else:
            options[0]["extra"] = "untrusted"
        with pytest.raises(ValueError, match=reason):
            routing_text_parser.parse_routing_prompt(_prompt_with_payload(payload))


def test_parser_rejects_unrecognized_wording_and_trailing_text() -> None:
    wrong_question = _payload_from_prompt(HANDWRITTEN_PROMPT)
    wrong_question["question"] = "Which queue is best?"
    wrong_instructions = _payload_from_prompt(HANDWRITTEN_PROMPT)
    context = wrong_instructions["context"]
    assert isinstance(context, str)
    wrong_instructions["context"] = context.replace("Apply the complete", "Use these", 1)
    trailing_context = _payload_from_prompt(HANDWRITTEN_PROMPT)
    context = trailing_context["context"]
    assert isinstance(context, str)
    trailing_context["context"] = context + " unrecognized text"

    prompts = (
        (_prompt_with_payload(wrong_question), "question is unrecognized"),
        (_prompt_with_payload(wrong_instructions), "context instructions are unrecognized"),
        (_prompt_with_payload(trailing_context), "invalid routing prompt"),
        (
            HANDWRITTEN_PROMPT[: -len(_ANSWER_SUFFIX)],
            "prompt must end with the exact answer suffix",
        ),
        (HANDWRITTEN_PROMPT + "unrecognized text", "prompt must end with the exact answer suffix"),
        (
            HANDWRITTEN_PROMPT.replace("\nAnswer:\n", "\nResponse:\n", 1),
            "prompt must end with the exact answer suffix",
        ),
    )
    for prompt, reason in prompts:
        with pytest.raises(ValueError, match=reason):
            routing_text_parser.parse_routing_prompt(prompt)


def test_parser_rejects_invalid_domains_rules_and_queries_from_text() -> None:
    cases: list[tuple[dict[str, object], str]] = []
    duplicate_domain = _block_from_prompt(HANDWRITTEN_PROMPT)
    duplicate_domain["features"][0]["values"] = ["standard", "standard"]
    cases.append((duplicate_domain, "feature values must be unique"))
    invalid_rule = _block_from_prompt(HANDWRITTEN_PROMPT)
    invalid_rule["rules"][0]["values"][1] = "north"
    cases.append((invalid_rule, "rule values must belong to each feature domain"))
    invalid_query = _block_from_prompt(HANDWRITTEN_PROMPT)
    invalid_query["query"][1] = "north"
    cases.append((invalid_query, "query values must belong to each feature domain"))
    wrong_domain_shape = _block_from_prompt(HANDWRITTEN_PROMPT)
    wrong_domain_shape["features"][0]["values"] = "standard"
    cases.append((wrong_domain_shape, "feature values must be a list"))

    for block, reason in cases:
        with pytest.raises(ValueError, match=reason):
            routing_text_parser.parse_routing_prompt(_prompt_with_block(block))
