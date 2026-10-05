"""Public tool-data constants, vocabularies, and immutable row contracts."""

from dataclasses import FrozenInstanceError

import pytest

from reflex_decisions.tool_data_spec import (
    CASE_KINDS,
    FAMILY,
    MAX_COUNTS,
    SOURCE_ID,
    SPLITS,
    TEMPLATE_IDS,
    VERSION,
    VOCABULARIES,
    ToolVocabulary,
)


def test_tool_data_contract_constants_and_templates_are_fixed() -> None:
    assert VERSION == "tool-data-v1"
    assert SOURCE_ID == "synthetic-tool-choice-v1"
    assert FAMILY == "tool-choice"
    assert SPLITS == ("train", "development", "calibration", "test")
    assert MAX_COUNTS == (56, 14, 14, 28)
    assert tuple(TEMPLATE_IDS[split] for split in SPLITS) == (
        "tool-v1",
        "tool-development-v1",
        "tool-calibration-v1",
        "tool-sealed-v1",
    )
    assert CASE_KINDS == (
        "complete_tool",
        "complete_no_eligible",
        "missing_tool_agree",
        "missing_none_agree",
        "missing_tool_conflict",
        "missing_tool_vs_none",
        "multiple_missing",
    )


def test_split_vocabularies_have_exact_sizes_and_are_token_disjoint() -> None:
    assert tuple(VOCABULARIES) == SPLITS
    token_sets = []
    for split in SPLITS:
        vocabulary = VOCABULARIES[split]
        assert len(vocabulary.capabilities) == 4
        assert len(vocabulary.input_types) == 3
        assert len(vocabulary.permissions) == 2
        assert len(vocabulary.tool_ids) == 6
        tokens = (
            *vocabulary.capabilities,
            *vocabulary.input_types,
            *vocabulary.permissions,
            *vocabulary.tool_ids,
        )
        assert len(tokens) == len(set(tokens))
        token_sets.append(set(tokens))

    for index, tokens in enumerate(token_sets):
        assert all(not tokens & other for other in token_sets[index + 1 :])


def test_tool_vocabulary_is_frozen() -> None:
    vocabulary = VOCABULARIES["train"]

    with pytest.raises(FrozenInstanceError):
        vocabulary.tool_ids = ("other",)  # type: ignore[misc]


def test_vocab_mapping_rejects_item_assignment() -> None:
    with pytest.raises(TypeError):
        VOCABULARIES["train"] = ToolVocabulary((), (), (), ())  # type: ignore[index]
