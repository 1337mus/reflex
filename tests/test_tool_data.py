"""Behavior tests for deterministic synthetic tool-choice data generation."""

from collections import defaultdict

import pytest

from reflex_decisions.rendering import render_prompt
from reflex_decisions.tool_data import ToolGenerationConfig, generate_tool_data
from reflex_decisions.tool_data_identity import (
    catalog_hash,
    split_for_structure,
    structural_catalog_hash,
)
from reflex_decisions.tool_data_spec import CASE_KINDS, MAX_COUNTS, SPLITS, VOCABULARIES
from reflex_decisions.tool_rules import INSUFFICIENT_INFORMATION, NO_ELIGIBLE_TOOL, solve
from reflex_decisions.tool_text_parser import parse_tool_prompt


def test_generation_config_rejects_boolean_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        ToolGenerationConfig(True)  # type: ignore[arg-type]


def test_generation_config_defaults_to_the_bounded_split_maxima() -> None:
    assert ToolGenerationConfig(0).counts == MAX_COUNTS


@pytest.mark.parametrize(
    ("seed", "counts", "message"),
    [
        (-1, (1, 1, 1, 1), "seed"),
        (2**64, (1, 1, 1, 1), "seed"),
        (1, [1, 1, 1, 1], "counts"),
        (1, (1, 1, 1), "counts"),
        (1, (1, 1, True, 1), "counts"),
        (1, (1, 1, 1, 29), "counts"),
    ],
)
def test_generation_config_rejects_out_of_range_seed_or_counts(
    seed: object, counts: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ToolGenerationConfig(seed, counts)  # type: ignore[arg-type]


def test_generation_config_accepts_last_seed_and_maximum_counts() -> None:
    config = ToolGenerationConfig(2**64 - 1, MAX_COUNTS)

    assert config.seed == 2**64 - 1
    assert config.counts == MAX_COUNTS


def test_generator_covers_all_cases_and_tool_counts_with_toy_rows() -> None:
    candidate = generate_tool_data(ToolGenerationConfig(65, (7, 7, 7, 7)))

    assert len(candidate.rows) == 28
    for split in SPLITS:
        rows = tuple(row for row in candidate.rows if row.split == split)
        assert {row.case_kind for row in rows} == set(CASE_KINDS)
        assert {len(row.scenario.tools) for row in rows} == {2, 4, 6}


def test_generator_emits_split_local_facts_with_recomputed_answers() -> None:
    candidate = generate_tool_data(ToolGenerationConfig(17, (7, 7, 7, 7)))

    for datum in candidate.rows:
        parsed = parse_tool_prompt(render_prompt(datum.record.request))
        assert parsed.scenario == datum.scenario
        assert datum.record.answer_id == solve(datum.scenario)
        vocabulary = VOCABULARIES[datum.split]
        assert set(datum.scenario.capabilities) <= set(vocabulary.capabilities)
        assert set(datum.scenario.input_types) <= set(vocabulary.input_types)
        assert set(datum.scenario.permissions) <= set(vocabulary.permissions)
        assert {tool.id for tool in datum.scenario.tools} <= set(vocabulary.tool_ids)


def test_generator_assigns_unique_catalogs_and_structural_source_groups() -> None:
    candidate = generate_tool_data(ToolGenerationConfig(25, (7, 7, 7, 7)))

    assert len({catalog_hash(row.scenario) for row in candidate.rows}) == len(candidate.rows)
    structures = [structural_catalog_hash(row.scenario) for row in candidate.rows]
    assert len(set(structures)) == len(structures)
    for datum, digest in zip(candidate.rows, structures, strict=True):
        assert datum.record.source_group_id == "tool-group-" + digest
        assert split_for_structure(digest) == datum.split


def test_generator_preserves_each_split_prefix_when_counts_expand() -> None:
    small = generate_tool_data(ToolGenerationConfig(31, (7, 7, 7, 7)))
    expanded = generate_tool_data(ToolGenerationConfig(31))

    for split in SPLITS:
        small_records = [row.record.model_dump_json() for row in small.rows if row.split == split]
        expanded_records = [
            row.record.model_dump_json() for row in expanded.rows if row.split == split
        ]
        assert expanded_records[: len(small_records)] == small_records


def test_generator_training_multiple_missing_outcomes_vary_at_each_menu_size() -> None:
    candidate = generate_tool_data(ToolGenerationConfig(41))
    outcomes: dict[int, set[str]] = defaultdict(set)
    query_widths: set[int] = set()
    distractors: set[str] = set()

    for datum in candidate.rows:
        scenario = datum.scenario
        query_widths.add(len(scenario.required_capabilities))
        if any(scenario.input_type not in tool.input_types for tool in scenario.tools):
            distractors.add("wrong_input")
        if any(
            not set(scenario.required_capabilities) <= set(tool.capabilities)
            for tool in scenario.tools
        ):
            distractors.add("missing_capability")
        if any(
            permission in tool.required_permissions
            and scenario.permission_values[scenario.permissions.index(permission)] is not True
            for tool in scenario.tools
            for permission in scenario.permissions
        ):
            distractors.add("permission")
        if datum.split == "train" and datum.case_kind == "multiple_missing":
            answer = datum.record.answer_id
            outcome_class = (
                "tool" if answer not in {NO_ELIGIBLE_TOOL, INSUFFICIENT_INFORMATION} else answer
            )
            outcomes[len(scenario.tools)].add(outcome_class)

    assert set(outcomes) == {2, 4, 6}
    assert all(len(values) >= 2 for values in outcomes.values())
    assert any(NO_ELIGIBLE_TOOL in values for values in outcomes.values())
    assert any(INSUFFICIENT_INFORMATION in values for values in outcomes.values())
    assert min(query_widths) == 1 and max(query_widths) > 1
    assert distractors == {"wrong_input", "missing_capability", "permission"}


@pytest.mark.parametrize("seed", [0, 1, 4, 7, 2**64 - 1])
def test_generator_seed_matrix_completes_at_supported_default_quotas(seed: int) -> None:
    candidate = generate_tool_data(ToolGenerationConfig(seed))

    assert len(candidate.rows) == sum(MAX_COUNTS)
    assert {row.case_kind for row in candidate.rows} == set(CASE_KINDS)


def test_generator_is_repeatable_for_the_same_seed_and_counts() -> None:
    config = ToolGenerationConfig(73, (7, 7, 7, 7))

    first = generate_tool_data(config)
    second = generate_tool_data(config)

    assert first == second
