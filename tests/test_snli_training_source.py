import json

from reflex_decisions import snli_training_source


def test_training_parser_excludes_conflicting_pairs_but_keeps_grouping_anchors() -> None:
    rows = [
        {
            "captionID": "caption-a",
            "pairID": "conflict-e",
            "sentence1": "Shared premise.",
            "sentence2": "Conflicting hypothesis.",
            "gold_label": "entailment",
        },
        {
            "captionID": "caption-bridge",
            "pairID": "conflict-c",
            "sentence1": " Shared   premise. ",
            "sentence2": "Conflicting hypothesis.",
            "gold_label": "contradiction",
        },
        {
            "captionID": "caption-bridge",
            "pairID": "candidate-n",
            "sentence1": "Another premise.",
            "sentence2": "A neutral hypothesis.",
            "gold_label": "neutral",
        },
    ]

    parsed = snli_training_source._parse_snli_jsonl(
        (json.dumps(row) for row in rows), include_unlabeled_anchors=True, conflict_policy="exclude"
    )

    assert [(item.source_item_id, item.label) for item in parsed.items] == [
        ("candidate-n", "neutral")
    ]
    conflict_key = ("shared premise.", "conflicting hypothesis.")
    assert parsed.pair_group_ids[conflict_key] == parsed.items[0].source_group_id
    assert parsed.stats.conflicting_pair_count == 1
    assert parsed.stats.conflicting_labeled_row_count == 2


def test_unlabeled_row_bridges_components_without_becoming_a_candidate() -> None:
    rows = [
        {
            "captionID": "caption-a",
            "pairID": "labeled-a",
            "sentence1": "Premise A.",
            "sentence2": "Hypothesis A.",
            "gold_label": "entailment",
        },
        {
            "captionID": "caption-bridge",
            "pairID": "unlabeled-bridge",
            "sentence1": "Premise A.",
            "sentence2": "Unlabeled hypothesis.",
            "gold_label": "-",
        },
        {
            "captionID": "caption-bridge",
            "pairID": "labeled-b",
            "sentence1": "Premise B.",
            "sentence2": "Hypothesis B.",
            "gold_label": "neutral",
        },
    ]

    parsed = snli_training_source._parse_snli_jsonl(
        (json.dumps(row) for row in rows), include_unlabeled_anchors=True, conflict_policy="exclude"
    )

    assert {item.source_item_id for item in parsed.items} == {"labeled-a", "labeled-b"}
    assert len({item.source_group_id for item in parsed.items}) == 1
    assert parsed.stats.unlabeled_row_count == 1


def test_training_parser_rejects_malformed_unlabeled_rows() -> None:
    import pytest

    malformed = json.dumps(
        {
            "captionID": "caption",
            "pairID": "unlabeled-without-premise",
            "sentence1": "   ",
            "sentence2": "A hypothesis.",
            "gold_label": "-",
        }
    )

    with pytest.raises(ValueError, match="blank premise"):
        snli_training_source._parse_snli_jsonl(
            [malformed], include_unlabeled_anchors=True, conflict_policy="exclude"
        )
