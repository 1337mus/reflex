"""Small hand-authored fixtures for the support-routing data contracts."""

from dataclasses import replace
from itertools import product

import pytest

from reflex_decisions.data import DatasetSpec, DecisionRecord, SplitManifest
from reflex_decisions.routing_data_audit import (
    audit_routing_data,
    classify_scenario,
    rule_set_hash,
    scenario_hash,
    split_for_structure,
    structural_rule_hash,
)
from reflex_decisions.routing_data_spec import (
    CASE_KINDS,
    FAMILY,
    SOURCE_ID,
    SPLITS,
    TEMPLATE_IDS,
    VOCABULARIES,
    RoutingDatum,
)
from reflex_decisions.routing_rules import (
    Feature,
    RouteRule,
    RoutingScenario,
    answer_ids,
    solve,
)
from reflex_decisions.routing_text import render_routing_request


def test_rule_hash_ignores_feature_value_destination_and_rule_order() -> None:
    original = RoutingScenario(
        features=(
            Feature("priority", ("routine", "urgent")),
            Feature("channel", ("email", "phone")),
        ),
        destination_ids=("cedar", "birch", "maple"),
        rules=(RouteRule(("routine", "email"), "cedar"), RouteRule(("urgent", "phone"), "birch")),
        query=("routine", "email"),
    )
    reordered = RoutingScenario(
        features=(
            Feature("channel", ("phone", "email")),
            Feature("priority", ("urgent", "routine")),
        ),
        destination_ids=("maple", "birch", "cedar"),
        rules=(RouteRule(("phone", "urgent"), "birch"), RouteRule(("email", "routine"), "cedar")),
        query=("email", "routine"),
    )

    assert rule_set_hash(original) == rule_set_hash(reordered)


def _structural_toy() -> RoutingScenario:
    return RoutingScenario(
        features=(
            Feature("channel", ("mail", "call")),
            Feature("urgency", ("low", "high")),
            Feature("region", ("north", "south")),
        ),
        destination_ids=("red", "blue", "unused"),
        rules=(
            RouteRule(("mail", "low", "north"), "red"),
            RouteRule(("call", "high", "south"), "blue"),
        ),
        query=("mail", "low", "north"),
    )


def _manifest(split: str, dataset_id: str = "toy-dataset") -> SplitManifest:
    return SplitManifest(
        data_kind="fixture",
        datasets=(
            DatasetSpec(
                dataset_id=dataset_id,
                source_id=SOURCE_ID,
                task_family=FAMILY,
                split=split,  # type: ignore[arg-type]
                source_uri="synthetic://toy",
                source_revision="toy-v1",
                license="self-authored synthetic",
            ),
        ),
        group_partitioned_sources=(SOURCE_ID,),
    )


def _manifest_for_splits(splits: tuple[tuple[str, str], ...]) -> SplitManifest:
    return SplitManifest(
        data_kind="fixture",
        datasets=tuple(
            DatasetSpec(
                dataset_id=dataset_id,
                source_id=SOURCE_ID,
                task_family=FAMILY,
                split=split,  # type: ignore[arg-type]
                source_uri="synthetic://toy",
                source_revision="toy-v1",
                license="self-authored synthetic",
            )
            for split, dataset_id in splits
        ),
        group_partitioned_sources=(SOURCE_ID,),
    )


def _scenario_for_split(
    split: str, query_bits: tuple[int | None, ...] = (0, 0, 0, 0)
) -> RoutingScenario:
    vocabulary = VOCABULARIES[split]  # type: ignore[index]
    values = tuple(feature.values for feature in vocabulary.features)
    destinations = vocabulary.destination_ids
    return RoutingScenario(
        features=vocabulary.features,
        destination_ids=destinations,
        rules=(
            RouteRule(tuple(domain[0] for domain in values), destinations[0]),
            RouteRule(tuple(domain[1] for domain in values), destinations[1]),
        ),
        query=tuple(
            None if bit is None else values[index][bit] for index, bit in enumerate(query_bits)
        ),
    )


def _row_for_scenario(
    scenario: RoutingScenario,
    split: str,
    *,
    record_id: str = "toy-record",
    dataset_id: str = "toy-dataset",
    template_id: str | None = None,
) -> RoutingDatum:
    selected_template = TEMPLATE_IDS[split] if template_id is None else template_id  # type: ignore[index]
    order = tuple(reversed(answer_ids(scenario)))
    request = render_routing_request(
        scenario,
        option_order=order,
        template_id=selected_template,
    )
    structure = structural_rule_hash(scenario)
    record = DecisionRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        source_group_id="routing-group-" + structure,
        request=request,
        answer_id=solve(scenario),
    )
    return RoutingDatum(
        split=split,  # type: ignore[arg-type]
        scenario=scenario,
        record=record,
        template_id=selected_template,
        case_kind=classify_scenario(scenario),
    )


def _valid_row() -> tuple[RoutingDatum, SplitManifest]:
    initial_features = tuple(Feature(f"axis-{index}", ("zero", "one")) for index in range(4))
    initial = RoutingScenario(
        features=initial_features,
        destination_ids=("red", "blue", "u1", "u2", "u3", "u4"),
        rules=(
            RouteRule(("zero", "zero", "zero", "zero"), "red"),
            RouteRule(("one", "one", "one", "one"), "blue"),
        ),
        query=("zero", "zero", "zero", "zero"),
    )
    split = split_for_structure(structural_rule_hash(initial))
    scenario = _scenario_for_split(split)
    row = _row_for_scenario(scenario, split)
    return row, _manifest(split)


def _replace_record(row: RoutingDatum, **changes: object) -> RoutingDatum:
    return replace(row, record=row.record.model_copy(update=changes))


def test_scenario_hash_adds_query_while_rule_hash_excludes_it() -> None:
    first = _structural_toy()
    second = replace(first, query=("call", "high", "south"))

    assert rule_set_hash(first) == rule_set_hash(second)
    assert scenario_hash(first) != scenario_hash(second)
    assert len(scenario_hash(first)) == 64


def test_structural_hash_ignores_names_and_destination_numbering() -> None:
    renamed = RoutingScenario(
        features=(
            Feature("zone", ("S", "N")),
            Feature("channel", ("voice", "post")),
            Feature("priority", ("H", "L")),
        ),
        destination_ids=("unused9", "route_two", "route_one"),
        rules=(
            RouteRule(("S", "voice", "H"), "route_two"),
            RouteRule(("N", "post", "L"), "route_one"),
        ),
        query=("N", "post", "L"),
    )

    assert structural_rule_hash(_structural_toy()) == structural_rule_hash(renamed)


def test_structural_hash_changes_for_table_or_unused_menu_size() -> None:
    original = _structural_toy()
    changed_table = replace(
        original,
        rules=(original.rules[0], RouteRule(original.rules[1].values, "red")),
    )
    larger_unused_menu = replace(
        original,
        destination_ids=(*original.destination_ids, "also-unused"),
    )

    assert structural_rule_hash(original) != structural_rule_hash(changed_table)
    assert structural_rule_hash(original) != structural_rule_hash(larger_unused_menu)


def test_structural_hash_rejects_nonbinary_scenarios() -> None:
    nonbinary = RoutingScenario(
        features=(Feature("x", ("zero", "one", "two")),),
        destination_ids=("red", "blue"),
        rules=(),
        query=("zero",),
    )

    with pytest.raises(ValueError, match="one to four binary features"):
        structural_rule_hash(nonbinary)


def test_four_feature_structure_is_invariant_under_non_symmetric_renaming() -> None:
    original_features = tuple(Feature(f"axis-{index}", ("zero", "one")) for index in range(4))
    original = RoutingScenario(
        features=original_features,
        destination_ids=("red", "blue", "green", "unused"),
        rules=(
            RouteRule(("zero", "zero", "zero", "zero"), "red"),
            RouteRule(("zero", "zero", "one", "one"), "blue"),
            RouteRule(("one", "zero", "one", "zero"), "green"),
        ),
        query=("zero", None, "one", None),
    )
    order = (2, 0, 3, 1)
    new_names = tuple(f"renamed-{index}" for index in range(4))
    value_maps = tuple({"zero": f"v{index}-zero", "one": f"v{index}-one"} for index in range(4))
    renamed = RoutingScenario(
        features=tuple(
            Feature(
                new_names[index],
                (value_maps[index]["one"], value_maps[index]["zero"]),
            )
            for index in order
        ),
        destination_ids=("extra", "gamma", "alpha", "beta"),
        rules=tuple(
            RouteRule(
                tuple(value_maps[index][rule.values[index]] for index in order),
                {"red": "alpha", "blue": "beta", "green": "gamma"}[rule.destination_id],
            )
            for rule in reversed(original.rules)
        ),
        query=tuple(
            None if original.query[index] is None else value_maps[index][original.query[index]]
            for index in order
        ),
    )

    assert structural_rule_hash(original) == structural_rule_hash(renamed)
    all_zero_query = tuple(feature.values[-1] for feature in renamed.features)
    assert structural_rule_hash(renamed) == structural_rule_hash(
        replace(renamed, query=all_zero_query)
    )


def test_structural_split_is_stable_and_rejects_noncanonical_hashes() -> None:
    assert split_for_structure("0" * 64) == "train"
    assert split_for_structure(f"{'0' * 63}4") == "development"
    assert split_for_structure(f"{'0' * 63}5") == "calibration"
    assert split_for_structure(f"{'0' * 63}6") == "test"
    for bad_digest in ("A" * 64, "0" * 63, "g" * 64):
        with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
            split_for_structure(bad_digest)


@pytest.mark.parametrize(
    ("query", "rules", "expected"),
    (
        ((0, 0, 0), ((0, 0, 0, "red"),), "complete_route"),
        ((1, 1, 1), (), "complete_no_route"),
        ((None, 0, 0), ((0, 0, 0, "red"), (1, 0, 0, "red")), "missing_route_agree"),
        ((None, 0, 0), (), "missing_no_route_agree"),
        ((None, 0, 0), ((0, 0, 0, "red"), (1, 0, 0, "blue")), "missing_route_conflict"),
        ((None, 0, 0), ((0, 0, 0, "red"),), "missing_route_vs_none"),
        ((None, None, 0), ((0, 0, 0, "red"),), "multiple_missing"),
        (
            (None, None, 0),
            (
                (0, 0, 0, "red"),
                (1, 0, 0, "red"),
                (0, 1, 0, "red"),
                (1, 1, 0, "red"),
            ),
            "multiple_missing",
        ),
    ),
)
def test_classify_scenario_uses_completed_route_outcomes(
    query: tuple[int | None, ...],
    rules: tuple[tuple[int, int, int, str], ...],
    expected: str,
) -> None:
    features = (
        Feature("x", ("zero", "one")),
        Feature("y", ("zero", "one")),
        Feature("z", ("zero", "one")),
    )
    scenario = RoutingScenario(
        features=features,
        destination_ids=("red", "blue"),
        rules=tuple(
            RouteRule(tuple(features[i].values[bit] for i, bit in enumerate(row[:3])), row[3])
            for row in rules
        ),
        query=tuple(
            None if bit is None else features[index].values[bit] for index, bit in enumerate(query)
        ),
    )

    assert classify_scenario(scenario) == expected


def test_all_four_missing_features_classify_unanimous_and_mixed_outcomes() -> None:
    features = tuple(Feature(f"f{index}", ("zero", "one")) for index in range(4))
    all_routes = tuple(
        RouteRule(tuple(features[index].values[bit] for index, bit in enumerate(bits)), "red")
        for bits in product((0, 1), repeat=4)
    )
    unanimous = RoutingScenario(
        features=features,
        destination_ids=("red", "blue"),
        rules=all_routes,
        query=(None, None, None, None),
    )
    mixed = replace(unanimous, rules=(all_routes[0],))

    assert classify_scenario(unanimous) == "multiple_missing"
    assert classify_scenario(mixed) == "multiple_missing"


def test_valid_audit_reports_aggregate_counts_only() -> None:
    row, manifest = _valid_row()

    result = audit_routing_data((row,), manifest)

    assert result["rows"] == 1
    assert result["groups"] == 1
    assert result["split_counts"] == {split: int(split == row.split) for split in SPLITS}
    assert "request" not in result and "answer" not in result


def test_valid_audit_ignores_protected_feature_and_menu_order() -> None:
    row, manifest = _valid_row()
    scenario = row.scenario
    order = (2, 0, 3, 1)
    reordered = RoutingScenario(
        features=tuple(
            Feature(scenario.features[index].name, tuple(reversed(scenario.features[index].values)))
            for index in order
        ),
        destination_ids=tuple(reversed(scenario.destination_ids)),
        rules=tuple(
            RouteRule(tuple(rule.values[index] for index in order), rule.destination_id)
            for rule in reversed(scenario.rules)
        ),
        query=tuple(scenario.query[index] for index in order),
    )

    reordered_row = _row_for_scenario(reordered, row.split, record_id="toy-reordered")
    assert audit_routing_data((reordered_row,), manifest)["rows"] == 1


def test_unanimous_multi_missing_case_is_a_valid_audited_row() -> None:
    initial_features = (
        Feature("x", ("x0", "x1")),
        Feature("y", ("y0", "y1")),
        Feature("z", ("north", "south")),
        Feature("w", ("left", "right")),
    )
    initial = RoutingScenario(
        features=initial_features,
        destination_ids=("red", "blue", "u1", "u2", "u3", "u4"),
        rules=tuple(
            RouteRule(
                (initial_features[0].values[x], initial_features[1].values[y], "north", "left"),
                "red",
            )
            for x, y in product((0, 1), repeat=2)
        ),
        query=(None, None, "north", "left"),
    )
    split = split_for_structure(structural_rule_hash(initial))
    vocabulary = VOCABULARIES[split]
    scenario = RoutingScenario(
        features=vocabulary.features,
        destination_ids=vocabulary.destination_ids,
        rules=tuple(
            RouteRule(
                (
                    vocabulary.features[0].values[x],
                    vocabulary.features[1].values[y],
                    vocabulary.features[2].values[0],
                    vocabulary.features[3].values[0],
                ),
                vocabulary.destination_ids[0],
            )
            for x, y in product((0, 1), repeat=2)
        ),
        query=(None, None, vocabulary.features[2].values[0], vocabulary.features[3].values[0]),
    )
    row = _row_for_scenario(scenario, split)

    assert row.case_kind == "multiple_missing"
    assert row.record.answer_id == scenario.destination_ids[0]
    assert audit_routing_data((row,), _manifest(split))["rows"] == 1


def test_case_kind_order_matches_generator_contract() -> None:
    assert CASE_KINDS == (
        "complete_route",
        "complete_no_route",
        "missing_route_agree",
        "missing_no_route_agree",
        "missing_route_conflict",
        "missing_route_vs_none",
        "multiple_missing",
    )


def test_four_feature_banks_preserve_and_extend_each_split_vocabulary() -> None:
    expected = {
        "train": (
            (
                ("priority", ("routine", "urgent")),
                ("region", ("east", "west")),
                ("channel", ("email", "phone")),
                ("account", ("individual", "business")),
            ),
            ("cedar", "birch", "maple", "oak", "pine", "elm"),
        ),
        "development": (
            (
                ("plan", ("starter", "premium")),
                ("device", ("desktop", "mobile")),
                ("language", ("english", "spanish")),
                ("connection", ("wired", "wireless")),
            ),
            ("amber", "cobalt", "jade", "ruby", "pearl", "onyx"),
        ),
        "calibration": (
            (
                ("status", ("new", "renewal")),
                ("category", ("billing", "technical")),
                ("period", ("day", "night")),
                ("age", ("recent", "established")),
            ),
            ("falcon", "heron", "ibis", "kestrel", "lark", "osprey"),
        ),
        "test": (
            (
                ("delivery", ("parcel", "freight")),
                ("material", ("paper", "metal")),
                ("service", ("onsite", "remote")),
                ("container", ("crate", "envelope")),
            ),
            ("atlas", "boreal", "comet", "delta", "equinox", "finch"),
        ),
    }

    for split, vocabulary in VOCABULARIES.items():
        features = tuple((feature.name, feature.values) for feature in vocabulary.features)
        assert (features, vocabulary.destination_ids) == expected[split]


def test_audit_rejects_answer_that_differs_from_solver() -> None:
    row, manifest = _valid_row()
    wrong_answer = next(
        option.id for option in row.record.request.options if option.id != row.record.answer_id
    )

    with pytest.raises(ValueError, match="record answer does not match the routing solver"):
        audit_routing_data((_replace_record(row, answer_id=wrong_answer),), manifest)


def test_audit_rejects_template_id_that_does_not_match_split() -> None:
    row, manifest = _valid_row()
    wrong_template = next(value for value in TEMPLATE_IDS.values() if value != row.template_id)

    with pytest.raises(ValueError, match="template ID does not match the declared split"):
        audit_routing_data((replace(row, template_id=wrong_template),), manifest)


def test_audit_rejects_prompt_template_that_disagrees_with_metadata() -> None:
    row, manifest = _valid_row()
    other_template = next(value for value in TEMPLATE_IDS.values() if value != row.template_id)
    request = render_routing_request(
        row.scenario,
        option_order=tuple(option.id for option in row.record.request.options),
        template_id=other_template,
    )

    with pytest.raises(ValueError, match="prompt template does not match the row template ID"):
        audit_routing_data((_replace_record(row, request=request),), manifest)


def test_audit_rejects_protected_vocabulary_or_case_kind_mutations() -> None:
    row, manifest = _valid_row()
    changed_features = (
        Feature("counterfeit", row.scenario.features[0].values),
        *row.scenario.features[1:],
    )
    changed_scenario = replace(row.scenario, features=changed_features)
    with pytest.raises(ValueError, match="protected feature names or domains"):
        audit_routing_data((replace(row, scenario=changed_scenario),), manifest)

    wrong_case = next(kind for kind in CASE_KINDS if kind != row.case_kind)
    with pytest.raises(ValueError, match="case kind does not match"):
        audit_routing_data((replace(row, case_kind=wrong_case),), manifest)


def test_audit_rejects_unapproved_source_family_and_menu_size() -> None:
    row, manifest = _valid_row()
    wrong_family = manifest.model_copy(
        update={
            "datasets": (manifest.datasets[0].model_copy(update={"task_family": "other-family"}),)
        }
    )
    with pytest.raises(ValueError, match="unapproved source or task family"):
        audit_routing_data((row,), wrong_family)

    three_destinations = replace(
        row.scenario,
        destination_ids=row.scenario.destination_ids[:3],
    )
    three_destination_row = _row_for_scenario(
        three_destinations,
        row.split,
        record_id="toy-three-destinations",
    )
    with pytest.raises(ValueError, match="destination menu does not match the split bank"):
        audit_routing_data((three_destination_row,), manifest)


def test_audit_rejects_wrong_record_id_grouping_and_duplicate_ids() -> None:
    row, manifest = _valid_row()
    wrong_group = _replace_record(row, source_group_id="routing-group-" + "0" * 64)
    with pytest.raises(ValueError, match="source group ID does not match"):
        audit_routing_data((wrong_group,), manifest)

    duplicate = DecisionRecord(
        record_id=row.record.record_id,
        dataset_id=row.record.dataset_id,
        source_group_id=row.record.source_group_id,
        request=row.record.request,
        answer_id=row.record.answer_id,
    )
    with pytest.raises(ValueError, match="duplicate record_id"):
        audit_routing_data((row, replace(row, record=duplicate)), manifest)


def test_audit_rejects_structure_routed_to_a_different_split() -> None:
    row, _ = _valid_row()
    wrong_split = next(split for split in SPLITS if split != row.split)
    scenario = _scenario_for_split(wrong_split)
    wrong_row = _row_for_scenario(
        scenario,
        wrong_split,
        dataset_id="wrong-dataset",
        template_id=TEMPLATE_IDS[wrong_split],
    )
    wrong_manifest = _manifest(wrong_split, "wrong-dataset")

    with pytest.raises(ValueError, match="structural rules are assigned to a different split"):
        audit_routing_data((wrong_row,), wrong_manifest)


def test_audit_rejects_counterfeit_metadata_for_changed_prompt_facts() -> None:
    row, manifest = _valid_row()
    changed = replace(
        row.scenario,
        rules=(
            RouteRule(row.scenario.rules[0].values, row.scenario.destination_ids[1]),
            row.scenario.rules[1],
        ),
    )
    changed_request = render_routing_request(
        changed,
        option_order=tuple(option.id for option in row.record.request.options),
        template_id=row.template_id,
    )
    counterfeit = _replace_record(row, request=changed_request)

    with pytest.raises(
        ValueError, match="rendered routing facts do not match the expected scenario"
    ):
        audit_routing_data((counterfeit,), manifest)


def test_related_queries_share_one_group_and_cannot_change_split() -> None:
    first, manifest = _valid_row()
    second_scenario = replace(
        first.scenario,
        query=tuple(feature.values[-1] for feature in first.scenario.features),
    )
    second = _row_for_scenario(second_scenario, first.split, record_id="toy-related")

    result = audit_routing_data((first, second), manifest)
    assert result["rows"] == 2 and result["groups"] == 1

    wrong_split = next(split for split in SPLITS if split != first.split)
    reassigned_scenario = _scenario_for_split(wrong_split)
    reassigned = _row_for_scenario(
        reassigned_scenario,
        wrong_split,
        record_id="toy-reassigned",
        dataset_id="wrong-dataset",
        template_id=TEMPLATE_IDS[wrong_split],
    )
    with pytest.raises(ValueError, match="a source group crosses dataset splits"):
        audit_routing_data(
            (first, reassigned),
            _manifest_for_splits(
                (
                    (first.split, first.record.dataset_id),
                    (wrong_split, reassigned.record.dataset_id),
                )
            ),
        )


def test_audit_rejects_duplicate_scenarios_and_empty_rows() -> None:
    row, manifest = _valid_row()
    with pytest.raises(ValueError, match="duplicate routing scenario hash"):
        audit_routing_data((_replace_record(row, record_id="toy-copy"), row), manifest)
    with pytest.raises(ValueError, match="must not be empty"):
        audit_routing_data((), manifest)
