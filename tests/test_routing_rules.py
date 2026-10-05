from dataclasses import FrozenInstanceError

import pytest

from reflex_decisions.routing_rules import Feature, RouteRule, RoutingScenario, answer_ids, solve


def make_scenario(
    *,
    features: tuple[Feature, ...] = (
        Feature("tier", ("standard", "urgent")),
        Feature("region", ("west", "east")),
    ),
    destination_ids: tuple[str, ...] = ("queue-a", "queue-b"),
    rules: tuple[RouteRule, ...] = (RouteRule(("urgent", "west"), "queue-b"),),
    query: tuple[str | None, ...] = ("urgent", "west"),
) -> RoutingScenario:
    return RoutingScenario(features, destination_ids, rules, query)


def test_complete_query_uses_its_matching_route_rule() -> None:
    assert solve(make_scenario()) == "queue-b"


def test_complete_unmapped_query_has_no_route() -> None:
    scenario = make_scenario(query=("standard", "west"))

    assert solve(scenario) == "no_route"


def test_missing_feature_returns_route_when_every_completion_agrees() -> None:
    scenario = make_scenario(
        rules=(
            RouteRule(("standard", "west"), "queue-a"),
            RouteRule(("urgent", "west"), "queue-a"),
        ),
        query=(None, "west"),
    )

    assert solve(scenario) == "queue-a"


def test_missing_feature_returns_no_route_when_every_completion_is_unmapped() -> None:
    scenario = make_scenario(rules=(), query=(None, "west"))

    assert solve(scenario) == "no_route"


def test_missing_feature_with_different_routes_is_insufficient() -> None:
    scenario = make_scenario(
        rules=(
            RouteRule(("standard", "west"), "queue-a"),
            RouteRule(("urgent", "west"), "queue-b"),
        ),
        query=(None, "west"),
    )

    assert solve(scenario) == "insufficient_information"


def test_missing_feature_with_route_and_unmapped_completion_is_insufficient() -> None:
    scenario = make_scenario(query=(None, "west"))

    assert solve(scenario) == "insufficient_information"


def test_multiple_missing_features_enumerate_all_four_completions() -> None:
    scenario = make_scenario(
        rules=tuple(
            RouteRule((tier, region), "queue-a")
            for tier in ("standard", "urgent")
            for region in ("west", "east")
        ),
        query=(None, None),
    )

    assert solve(scenario) == "queue-a"


def test_all_256_legal_completions_are_supported() -> None:
    features = tuple(
        Feature(f"feature-{index}", tuple(f"value-{value}" for value in range(4)))
        for index in range(4)
    )
    scenario = make_scenario(features=features, rules=(), query=(None, None, None, None))

    assert solve(scenario) == "no_route"

    final_completion_route = make_scenario(
        features=features,
        rules=(RouteRule(("value-3",) * 4, "queue-a"),),
        query=(None, None, None, None),
    )
    assert solve(final_completion_route) == "insufficient_information"


def test_answer_menu_does_not_depend_on_query_result() -> None:
    rules = (RouteRule(("urgent", "west"), "queue-a"),)
    known = make_scenario(rules=rules)
    unmapped = make_scenario(rules=rules, query=("standard", "west"))
    ambiguous = make_scenario(rules=rules, query=(None, "west"))

    assert answer_ids(known) == ("queue-a", "queue-b", "no_route", "insufficient_information")
    assert answer_ids(unmapped) == answer_ids(known)
    assert answer_ids(ambiguous) == answer_ids(known)


def test_input_dataclasses_are_immutable() -> None:
    feature = Feature("tier", ("standard", "urgent"))

    with pytest.raises(FrozenInstanceError):
        feature.name = "other"


@pytest.mark.parametrize("name", ("", "   ", None, 3))
def test_feature_rejects_blank_or_non_string_names(name: object) -> None:
    with pytest.raises((TypeError, ValueError), match="name"):
        Feature(name, ("standard",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values", ([], (), ("x", "x"), (" ",), ("x", 2), ("a", "b", "c", "d", "e"))
)
def test_feature_rejects_malformed_value_domains(values: object) -> None:
    with pytest.raises((TypeError, ValueError), match="values"):
        Feature("tier", values)  # type: ignore[arg-type]


@pytest.mark.parametrize("features", ([], (), ("not-a-feature",)))
def test_scenario_rejects_malformed_feature_sequences(features: object) -> None:
    with pytest.raises((TypeError, ValueError), match="features"):
        make_scenario(features=features)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "features",
    (
        (Feature("same", ("x",)), Feature("same", ("y",))),
        tuple(Feature(f"f-{index}", ("x",)) for index in range(5)),
    ),
)
def test_scenario_rejects_duplicate_names_and_too_many_features(
    features: tuple[Feature, ...],
) -> None:
    with pytest.raises(ValueError, match="feature"):
        make_scenario(features=features)


@pytest.mark.parametrize(
    "destination_ids",
    (
        ("queue-a",),
        tuple(f"queue-{index}" for index in range(7)),
        ("queue-a", "queue-a"),
        ("queue-a", " "),
        ("queue-a", "no_route"),
        ("queue-a", "insufficient_information"),
        ("queue-a", 7),
    ),
)
def test_scenario_rejects_invalid_destination_ids(destination_ids: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError), match="destination"):
        make_scenario(destination_ids=destination_ids)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "rules",
    (
        (RouteRule(("urgent", "west"), "queue-a"), RouteRule(("urgent", "west"), "queue-b")),
        (RouteRule(("urgent",), "queue-a"),),
        (RouteRule(("urgent", "north"), "queue-a"),),
        (RouteRule(("urgent", "west"), "queue-c"),),
        (RouteRule(("urgent", "west"), "no_route"),),
        ("not-a-rule",),
    ),
)
def test_scenario_rejects_duplicate_or_invalid_rules(rules: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError), match="rule|destination"):
        make_scenario(rules=rules)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "query",
    (
        ("urgent",),
        ("urgent", "north"),
        ("urgent", 7),
        ("urgent", "west", "extra"),
        ["urgent", "west"],
    ),
)
def test_scenario_rejects_malformed_queries(query: object) -> None:
    with pytest.raises((TypeError, ValueError), match="query"):
        make_scenario(query=query)  # type: ignore[arg-type]


def test_route_rule_rejects_non_tuple_or_non_string_keys() -> None:
    with pytest.raises(TypeError, match="values"):
        RouteRule(["urgent", "west"], "queue-a")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="values"):
        RouteRule(("urgent", 3), "queue-a")  # type: ignore[arg-type]


@pytest.mark.parametrize("destination_id", ("", "  ", 5))
def test_route_rule_rejects_blank_or_non_string_destination(destination_id: object) -> None:
    with pytest.raises((TypeError, ValueError), match="destination"):
        RouteRule(("urgent", "west"), destination_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("function", (solve, answer_ids))
def test_solvers_reject_non_scenario_runtime_values(function: object) -> None:
    with pytest.raises(TypeError, match="RoutingScenario"):
        function(None)  # type: ignore[operator]
