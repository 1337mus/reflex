"""Canonical hashes and integrity checks for routing-data rows."""

from __future__ import annotations

import hashlib
import json
import re
from itertools import permutations, product

from .data import SplitManifest, SplitName, audit_splits
from .rendering import render_prompt
from .routing_data_spec import (
    CASE_KINDS,
    FAMILY,
    SOURCE_ID,
    SPLITS,
    TEMPLATE_IDS,
    VOCABULARIES,
    RoutingDatum,
)
from .routing_rules import NO_ROUTE, RoutingScenario, solve
from .routing_text_parser import parse_routing_prompt, verify_routing_request


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _named_payload(scenario: RoutingScenario, *, include_query: bool) -> dict[str, object]:
    features = sorted(scenario.features, key=lambda feature: feature.name)
    payload: dict[str, object] = {
        "features": [
            {"name": feature.name, "values": sorted(feature.values)} for feature in features
        ],
        "destinations": sorted(scenario.destination_ids),
        "rules": sorted(
            (
                {
                    "when": sorted(
                        (
                            (feature.name, value)
                            for feature, value in zip(scenario.features, rule.values, strict=True)
                        ),
                        key=lambda pair: pair[0],
                    ),
                    "destination": rule.destination_id,
                }
                for rule in scenario.rules
            ),
            key=_canonical_json,
        ),
    }
    if include_query:
        payload["query"] = sorted(
            (
                (feature.name, value)
                for feature, value in zip(scenario.features, scenario.query, strict=True)
            ),
            key=lambda pair: pair[0],
        )
    return payload


def rule_set_hash(scenario: RoutingScenario) -> str:
    return _digest(_named_payload(scenario, include_query=False))


def scenario_hash(scenario: RoutingScenario) -> str:
    return _digest(_named_payload(scenario, include_query=True))


def structural_rule_hash(scenario: RoutingScenario) -> str:
    feature_count = len(scenario.features)
    if not 1 <= feature_count <= 4 or any(
        len(feature.values) != 2 for feature in scenario.features
    ):
        raise ValueError("structural hashes support one to four binary features only")

    table = {rule.values: rule.destination_id for rule in scenario.rules}
    coordinates = tuple(product((0, 1), repeat=feature_count))
    smallest: tuple[int, ...] | None = None
    for order in permutations(range(feature_count)):
        for flips in product((0, 1), repeat=feature_count):
            labels: dict[str, int] = {}
            codes: list[int] = []
            for coordinate in coordinates:
                original_values = [""] * feature_count
                for index, original_index in enumerate(order):
                    original_values[original_index] = scenario.features[original_index].values[
                        coordinate[index] ^ flips[index]
                    ]
                values = tuple(original_values)
                destination = table.get(values)
                if destination is None:
                    codes.append(0)
                else:
                    if destination not in labels:
                        labels[destination] = len(labels) + 1
                    codes.append(labels[destination])
            code = tuple(codes)
            if smallest is None or code < smallest:
                smallest = code
    assert smallest is not None
    return _digest(
        {
            "feature_count": feature_count,
            "destination_count": len(scenario.destination_ids),
            "table": smallest,
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
    return slots[int(digest, 16) % 8]


def classify_scenario(scenario: RoutingScenario) -> str:
    table = {rule.values: rule.destination_id for rule in scenario.rules}
    missing = sum(value is None for value in scenario.query)
    domains = tuple(
        feature.values if value is None else (value,)
        for feature, value in zip(scenario.features, scenario.query, strict=True)
    )
    outcomes = {table.get(completion, NO_ROUTE) for completion in product(*domains)}
    if missing == 0:
        return "complete_no_route" if outcomes == {NO_ROUTE} else "complete_route"
    if missing == 1:
        if len(outcomes) == 1:
            return "missing_no_route_agree" if outcomes == {NO_ROUTE} else "missing_route_agree"
        if len(outcomes) != 2:
            raise ValueError("one-missing cases support at most two distinct outcomes")
        if NO_ROUTE in outcomes:
            return "missing_route_vs_none"
        return "missing_route_conflict"
    if missing >= 2:
        return "multiple_missing"
    raise ValueError("unsupported missing-value count")


def audit_routing_data(
    rows: tuple[RoutingDatum, ...], manifest: SplitManifest
) -> dict[str, object]:
    if not rows:
        raise ValueError("routing data rows must not be empty")

    vocabulary_terms = [
        {
            term
            for feature in VOCABULARIES[split].features
            for term in (feature.name, *feature.values)
        }
        | set(VOCABULARIES[split].destination_ids)
        for split in SPLITS
    ]
    if any(
        left & right
        for index, left in enumerate(vocabulary_terms)
        for right in vocabulary_terms[index + 1 :]
    ):
        raise ValueError("protected vocabularies must be disjoint across splits")

    records = tuple(row.record for row in rows)
    audit_splits(manifest, records)
    datasets = {dataset.dataset_id: dataset for dataset in manifest.datasets}
    seen_scenarios: set[str] = set()
    rule_groups: dict[str, tuple[SplitName, str]] = {}
    structure_groups: dict[str, tuple[SplitName, str]] = {}
    source_groups: set[str] = set()
    split_counts = {split: 0 for split in SPLITS}
    case_counts = {split: {kind: 0 for kind in CASE_KINDS} for split in SPLITS}
    menu_counts: dict[SplitName, dict[int, int]] = {split: {} for split in SPLITS}

    for row in rows:
        record = row.record
        dataset = datasets[record.dataset_id]
        if dataset.source_id != SOURCE_ID or dataset.task_family != FAMILY:
            raise ValueError("routing record has an unapproved source or task family")
        if row.split != dataset.split:
            raise ValueError("row split does not match its manifest dataset")
        if row.template_id != TEMPLATE_IDS[row.split]:
            raise ValueError("template ID does not match the declared split")

        scenario = row.scenario
        vocabulary = VOCABULARIES[row.split]
        expected_features = {
            feature.name: frozenset(feature.values) for feature in vocabulary.features
        }
        actual_features = {feature.name: frozenset(feature.values) for feature in scenario.features}
        if actual_features != expected_features:
            raise ValueError("protected feature names or domains do not match the split bank")
        if (
            len(scenario.destination_ids) not in (2, 4, 6)
            or len(set(scenario.destination_ids)) != len(scenario.destination_ids)
            or not set(scenario.destination_ids) <= set(vocabulary.destination_ids)
        ):
            raise ValueError("destination menu does not match the split bank")

        parsed = parse_routing_prompt(render_prompt(record.request))
        verify_routing_request(record.request, scenario)
        if parsed.scenario != scenario:
            raise ValueError("rendered routing facts do not match the row scenario")
        if parsed.option_ids != tuple(option.id for option in record.request.options):
            raise ValueError("rendered option order does not match the request")
        if parsed.template_id != row.template_id:
            raise ValueError("prompt template does not match the row template ID")
        if record.answer_id != solve(parsed.scenario):
            raise ValueError("record answer does not match the routing solver")

        expected_case = classify_scenario(scenario)
        if row.case_kind != expected_case:
            raise ValueError("case kind does not match the scenario outcomes")
        scenario_digest = scenario_hash(scenario)
        if scenario_digest in seen_scenarios:
            raise ValueError("duplicate routing scenario hash")
        seen_scenarios.add(scenario_digest)

        rule_digest = rule_set_hash(scenario)
        structure_digest = structural_rule_hash(scenario)
        expected_group = "routing-group-" + structure_digest
        if record.source_group_id != expected_group:
            raise ValueError("source group ID does not match the structural rule hash")
        if split_for_structure(structure_digest) != row.split:
            raise ValueError("structural rules are assigned to a different split")
        group_identity = (row.split, record.source_group_id)
        for digest, groups, label in (
            (rule_digest, rule_groups, "rule-set"),
            (structure_digest, structure_groups, "structural"),
        ):
            prior = groups.setdefault(digest, group_identity)
            if prior != group_identity:
                raise ValueError(f"{label} hash crosses splits or source groups")

        source_groups.add(record.source_group_id)
        split_counts[row.split] += 1
        case_counts[row.split][row.case_kind] += 1
        menu_size = len(record.request.options)
        menu_counts[row.split][menu_size] = menu_counts[row.split].get(menu_size, 0) + 1

    return {
        "rows": len(rows),
        "groups": len(source_groups),
        "split_counts": split_counts,
        "case_kind_counts": case_counts,
        "menu_size_counts": menu_counts,
    }
