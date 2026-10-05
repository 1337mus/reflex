"""Tests for the deterministic support-routing fixture generator."""

import importlib
from importlib.util import find_spec

import pytest

from reflex_decisions.rendering import render_prompt
from reflex_decisions.routing_data_audit import (
    audit_routing_data,
    classify_scenario,
    split_for_structure,
    structural_rule_hash,
)
from reflex_decisions.routing_data_spec import CASE_KINDS, SPLITS, TEMPLATE_IDS
from reflex_decisions.routing_rules import INSUFFICIENT_INFORMATION, NO_ROUTE, solve
from reflex_decisions.routing_text_parser import verify_routing_request


def test_routing_generator_module_is_available() -> None:
    """Expose a stable package module for callers preparing routing fixtures."""
    assert find_spec("reflex_decisions.routing_data") is not None


def test_routing_generator_exposes_its_config_and_entry_point() -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    assert hasattr(module, "RoutingGenerationConfig")
    assert hasattr(module, "generate_routing_data")


def test_generator_builds_solver_labelled_rows_with_verified_templates() -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    candidate = module.generate_routing_data(module.RoutingGenerationConfig(412, (2, 2, 2, 2)))

    assert len(candidate.rows) == 8
    assert {split: sum(row.split == split for row in candidate.rows) for split in SPLITS} == {
        split: 2 for split in SPLITS
    }
    for row in candidate.rows:
        assert row.record.answer_id == solve(row.scenario)
        assert row.template_id == TEMPLATE_IDS[row.split]
        verify_routing_request(row.record.request, row.scenario)
        assert row.case_kind == classify_scenario(row.scenario)
        prompt = render_prompt(row.record.request)
        assert "record_id" not in prompt and "dataset_id" not in prompt
        assert "source_group_id" not in prompt and "answer_id" not in prompt
    assert audit_routing_data(candidate.rows, candidate.manifest)["rows"] == 8


def test_generator_covers_case_kinds_and_three_menu_sizes() -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    candidate = module.generate_routing_data(module.RoutingGenerationConfig(912, (7, 7, 7, 7)))

    for split in SPLITS:
        rows = tuple(row for row in candidate.rows if row.split == split)
        assert {row.case_kind for row in rows} == set(CASE_KINDS)
        assert {len(row.record.request.options) for row in rows} == {4, 6, 8}


def test_structure_hashes_own_disjoint_splits_and_group_ids() -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    candidate = module.generate_routing_data(module.RoutingGenerationConfig(97, (5, 5, 5, 5)))
    hashes = [structural_rule_hash(row.scenario) for row in candidate.rows]

    assert len(hashes) == len(set(hashes))
    for row, digest in zip(candidate.rows, hashes, strict=True):
        assert split_for_structure(digest) == row.split
        assert row.record.source_group_id == "routing-group-" + digest


def test_generation_is_deterministic_and_prefix_stable_when_counts_grow() -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    small = module.generate_routing_data(module.RoutingGenerationConfig(38, (2, 2, 2, 2)))
    repeated = module.generate_routing_data(module.RoutingGenerationConfig(38, (2, 2, 2, 2)))
    expanded = module.generate_routing_data(module.RoutingGenerationConfig(38, (7, 7, 7, 7)))

    assert small == repeated
    for split in SPLITS:
        small_rows = tuple(row for row in small.rows if row.split == split)
        expanded_rows = tuple(row for row in expanded.rows if row.split == split)
        assert small_rows == expanded_rows[:2]
    assert len({row.record.source_group_id for row in expanded.rows}) == len(expanded.rows)


def test_multiple_missing_cases_cover_common_none_and_mixed_outcomes() -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    candidate = module.generate_routing_data(module.RoutingGenerationConfig(143, (21, 1, 1, 1)))
    multiple = tuple(
        row
        for row in candidate.rows
        if row.split == "train" and row.case_kind == "multiple_missing"
    )

    assert len(multiple) == 3
    outcomes = {solve(row.scenario) for row in multiple}
    has_common_route = any(
        outcome not in {NO_ROUTE, INSUFFICIENT_INFORMATION} for outcome in outcomes
    )
    has_no_route = NO_ROUTE in outcomes
    has_mixed_outcome = INSUFFICIENT_INFORMATION in outcomes
    assert has_common_route and has_no_route and has_mixed_outcome


def _answer_class(outcome: str) -> str:
    if outcome == NO_ROUTE:
        return "no_route"
    if outcome == INSUFFICIENT_INFORMATION:
        return "insufficient"
    return "route"


def test_train_multiple_missing_covers_all_menu_size_and_answer_classes() -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    candidate = module.generate_routing_data(module.RoutingGenerationConfig(65, (63, 1, 1, 1)))
    multiple = tuple(
        row
        for row in candidate.rows
        if row.split == "train" and row.case_kind == "multiple_missing"
    )
    observed = {
        (len(row.record.request.options), _answer_class(solve(row.scenario))) for row in multiple
    }
    expected = {
        (menu_size, answer_class)
        for menu_size in (4, 6, 8)
        for answer_class in ("route", "no_route", "insufficient")
    }

    assert observed == expected


@pytest.mark.parametrize("seed", [4, 7, 2**64 - 1])
def test_previously_scarce_default_seeds_generate_unique_audited_candidates(seed: int) -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    candidate = module.generate_routing_data(module.RoutingGenerationConfig(seed))

    assert len(candidate.rows) == 112
    assert len({row.record.source_group_id for row in candidate.rows}) == len(candidate.rows)
    assert audit_routing_data(candidate.rows, candidate.manifest)["rows"] == 112
    multi_missing = tuple(
        row
        for row in candidate.rows
        if row.split == "train" and row.case_kind == "multiple_missing"
    )
    classes_by_menu = {
        menu_size: {
            _answer_class(solve(row.scenario))
            for row in multi_missing
            if len(row.record.request.options) == menu_size
        }
        for menu_size in (4, 6, 8)
    }
    assert all(len(classes) > 1 for classes in classes_by_menu.values())


def test_maximum_counts_are_supported_for_every_split() -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    candidate = module.generate_routing_data(
        module.RoutingGenerationConfig(7, (112, 112, 112, 112))
    )

    assert len(candidate.rows) == 448
    assert {split: sum(row.split == split for row in candidate.rows) for split in SPLITS} == {
        split: 112 for split in SPLITS
    }
    assert len({row.record.source_group_id for row in candidate.rows}) == len(candidate.rows)


@pytest.mark.parametrize(
    ("seed", "counts", "message"),
    [
        (True, (1, 1, 1, 1), "seed"),
        (-1, (1, 1, 1, 1), "seed"),
        (2**64, (1, 1, 1, 1), "seed"),
        (1, [1, 1, 1, 1], "counts"),
        (1, (1, 1, 1), "counts"),
        (1, (1, 1, True, 1), "counts"),
        (1, (1, 1, 1, 113), "counts"),
    ],
)
def test_config_rejects_invalid_seed_or_counts(seed: object, counts: object, message: str) -> None:
    module = importlib.import_module("reflex_decisions.routing_data")
    with pytest.raises(ValueError, match=message):
        module.RoutingGenerationConfig(seed, counts)
