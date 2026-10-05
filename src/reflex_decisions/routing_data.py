"""Deterministic CPU-only support-routing fixture generation."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from itertools import product

from .data import DatasetSpec, DecisionRecord, SplitManifest, SplitName
from .rendering import render_prompt
from .routing_data_audit import (
    audit_routing_data,
    classify_scenario,
    split_for_structure,
    structural_rule_hash,
)
from .routing_data_spec import (
    CASE_KINDS,
    FAMILY,
    SOURCE_ID,
    SPLITS,
    TEMPLATE_IDS,
    VERSION,
    VOCABULARIES,
    RoutingDatum,
    RoutingVocabulary,
)
from .routing_rules import NO_ROUTE, RouteRule, RoutingScenario, answer_ids, solve
from .routing_text import render_routing_request
from .routing_text_parser import parse_routing_prompt, verify_routing_request

_MAX_ATTEMPTS = 2048


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _rng(seed: int, split: SplitName, index: int, attempt: int) -> random.Random:
    return random.Random(int(_digest([VERSION, seed, split, index, attempt]), 16))


def _scenario(
    vocabulary: RoutingVocabulary,
    rng: random.Random,
    case_kind: str,
    multiple_case: int,
    destination_count: int,
) -> RoutingScenario:
    features = vocabulary.features
    selected_destinations = set(rng.sample(vocabulary.destination_ids, destination_count))
    destinations = tuple(
        destination
        for destination in vocabulary.destination_ids
        if destination in selected_destinations
    )
    completions = tuple(product(*(feature.values for feature in features)))
    table = {values: rng.choice((*destinations, NO_ROUTE)) for values in completions}
    missing_count = (
        0 if case_kind.startswith("complete_") else (2 if case_kind == "multiple_missing" else 1)
    )
    missing = set(rng.sample(range(len(features)), missing_count))
    query = tuple(
        None if index in missing else rng.choice(feature.values)
        for index, feature in enumerate(features)
    )
    matching = tuple(
        values
        for values in completions
        if all(
            expected is None or expected == actual
            for expected, actual in zip(query, values, strict=True)
        )
    )
    if case_kind == "complete_route":
        table[matching[0]] = rng.choice(destinations)
    elif case_kind == "complete_no_route":
        table.pop(matching[0])
    elif case_kind == "missing_route_agree":
        destination = rng.choice(destinations)
        for key in matching:
            table[key] = destination
    elif case_kind == "missing_no_route_agree":
        for key in matching:
            table.pop(key)
    elif case_kind == "missing_route_conflict":
        for key, destination in zip(matching, rng.sample(destinations, 2), strict=True):
            table[key] = destination
    elif case_kind == "missing_route_vs_none":
        route_key = rng.choice(matching)
        table[route_key] = rng.choice(destinations)
        table.pop(next(key for key in matching if key != route_key))
    elif case_kind == "multiple_missing":
        if multiple_case == 0:
            destination = rng.choice(destinations)
            for key in matching:
                table[key] = destination
        elif multiple_case == 1:
            for key in matching:
                table.pop(key)
        else:
            table[matching[0]] = rng.choice(destinations)
            table.pop(matching[1])
    else:
        raise ValueError("unknown routing case kind")
    rules = tuple(
        RouteRule(values, destination)
        for values, destination in table.items()
        if destination != NO_ROUTE
    )
    return RoutingScenario(features, destinations, rules, query)


@dataclass(frozen=True, slots=True)
class RoutingGenerationConfig:
    """Immutable seed and per-split sizes for a routing fixture candidate."""

    seed: int
    counts: tuple[int, int, int, int] = (56, 14, 14, 28)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer between 0 and 2**64 - 1")
        if (
            not isinstance(self.counts, tuple)
            or len(self.counts) != 4
            or any(type(count) is not int or not 1 <= count <= 112 for count in self.counts)
        ):
            raise ValueError("counts must be a four-item tuple of integers from 1 to 112")


@dataclass(frozen=True, slots=True)
class RoutingCandidate:
    config: RoutingGenerationConfig
    manifest: SplitManifest
    rows: tuple[RoutingDatum, ...]


def generate_routing_data(config: RoutingGenerationConfig) -> RoutingCandidate:
    if not isinstance(config, RoutingGenerationConfig):
        raise TypeError("config must be a RoutingGenerationConfig")
    datasets = tuple(
        DatasetSpec(
            dataset_id=f"{VERSION}-{split}",
            source_id=SOURCE_ID,
            task_family=FAMILY,
            split=split,
            source_uri="synthetic://reflex/support-routing/v1",
            source_revision=VERSION,
            license="self-authored synthetic",
        )
        for split in SPLITS
    )
    manifest = SplitManifest(
        data_kind="fixture",
        datasets=datasets,
        group_partitioned_sources=(SOURCE_ID,),
    )
    rows: list[RoutingDatum] = []
    used_structures: set[str] = set()
    for split, count in zip(SPLITS, config.counts, strict=True):
        vocabulary = VOCABULARIES[split]
        for index in range(count):
            case_kind = CASE_KINDS[index % len(CASE_KINDS)]
            multiple_case = (index // len(CASE_KINDS)) % 3
            destination_count = (2, 4, 6)[(index + index // 21) % 3]
            for attempt in range(_MAX_ATTEMPTS):
                scenario = _scenario(
                    vocabulary,
                    _rng(config.seed, split, index, attempt),
                    case_kind,
                    multiple_case,
                    destination_count,
                )
                if classify_scenario(scenario) != case_kind:
                    raise AssertionError("generated scenario does not match its requested case")
                structure = structural_rule_hash(scenario)
                if split_for_structure(structure) != split or structure in used_structures:
                    continue
                used_structures.add(structure)
                template_id = TEMPLATE_IDS[split]
                option_order = tuple(
                    sorted(
                        answer_ids(scenario),
                        key=lambda answer: _digest(
                            [VERSION, config.seed, split, index, "menu-order", answer]
                        ),
                    )
                )
                request = render_routing_request(
                    scenario, option_order=option_order, template_id=template_id
                )
                verify_routing_request(request, scenario)
                parsed = parse_routing_prompt(render_prompt(request))
                if parsed.template_id != template_id:
                    raise ValueError("rendered routing prompt template does not match its split")
                record = DecisionRecord(
                    record_id="routing-record-" + _digest([VERSION, config.seed, split, index]),
                    dataset_id=f"{VERSION}-{split}",
                    source_group_id="routing-group-" + structure,
                    request=request,
                    answer_id=solve(scenario),
                )
                rows.append(RoutingDatum(split, scenario, record, template_id, case_kind))
                break
            else:
                raise ValueError(
                    f"could not generate a unique {split} structure for item {index} "
                    f"within {_MAX_ATTEMPTS} attempts"
                )
    row_tuple = tuple(rows)
    audit_routing_data(row_tuple, manifest)
    return RoutingCandidate(config, manifest, row_tuple)
