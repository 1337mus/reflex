from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reflex_decisions import fresh_eval_data
from reflex_decisions.schema import DecisionRequest, Option

_HANS_FIELDS = (
    "gold_label",
    "sentence1_binary_parse",
    "sentence2_binary_parse",
    "sentence1_parse",
    "sentence2_parse",
    "sentence1",
    "sentence2",
    "pairID",
    "heuristic",
    "subcase",
    "template",
)


def _hans_row(
    *, pair_id: str, sentence1: str, sentence2: str, gold_label: str = "entailment"
) -> tuple[str, ...]:
    return (
        gold_label,
        "(S A)",
        "(S B)",
        "(S A)",
        "(S B)",
        sentence1,
        sentence2,
        pair_id,
        "lexical_overlap",
        "fake-subcase",
        "fake-template",
    )


def _hans_tsv(*rows: tuple[str, ...]) -> bytes:
    return (
        "\t".join(_HANS_FIELDS) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n"
    ).encode()


def test_selection_rank_uses_the_frozen_seed_and_raw_id_only() -> None:
    assert (
        fresh_eval_data._selection_rank_sha256("hans-eval-v1", "pair-17")
        == "bf8b7f97f2c5f9b02d72fa170efe1f2d336bea09955758b438f726dcdf790221"
    )


def test_hans_parser_maps_a_row_without_putting_gold_in_the_request() -> None:
    candidates = fresh_eval_data.parse_hans_tsv(
        _hans_tsv(
            _hans_row(
                pair_id="pair-17",
                sentence1="A red ball rolls.",
                sentence2="A ball moves.",
            )
        )
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.raw_source_id == "pair-17"
    assert candidate.answer_id == "entailment"
    assert candidate.request.context == "A red ball rolls."
    assert candidate.request.question == (
        "Does the following statement necessarily follow from the context? A ball moves."
    )
    assert [option.label for option in candidate.request.options] == [
        "Follows from the sentence",
        "Does not necessarily follow",
    ]
    assert candidate.source_group_id == (
        "hans-subcase:62a87860050907614ebdb23fb28235e03d768c55c6d011bc978d11bd29e134df"
    )
    assert candidate.source_metadata == {
        "heuristic": "lexical_overlap",
        "subcase": "fake-subcase",
        "template": "fake-template",
    }
    assert "answer_id" not in candidate.request.model_dump()


def test_hans_parser_rejects_a_conflicting_duplicate_source_id() -> None:
    with pytest.raises(ValueError, match="duplicate HANS source ID"):
        fresh_eval_data.parse_hans_tsv(
            _hans_tsv(
                _hans_row(
                    pair_id="pair-duplicate",
                    sentence1="A red ball rolls.",
                    sentence2="A ball moves.",
                ),
                _hans_row(
                    pair_id="pair-duplicate",
                    sentence1="A blue cube falls.",
                    sentence2="An object moves.",
                ),
            )
        )


def test_winogrande_parser_aligns_the_dev_label_to_an_unchanged_option() -> None:
    candidates = fresh_eval_data.parse_winogrande_dev(
        b'{"qID":"q-1","sentence":"A _ event.","option1":"rain","option2":"wind"}\n',
        b"2\n",
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.dataset_id == "winogrande-dev-v1"
    assert candidate.raw_source_id == "q-1"
    assert candidate.answer_id == "option2"
    assert candidate.request.context == "A _ event."
    assert candidate.request.question == "Which option fills the blank?"
    assert [(option.id, option.label) for option in candidate.request.options] == [
        ("option1", "rain"),
        ("option2", "wind"),
    ]
    assert set(candidate.request.model_dump()) == {"context", "question", "options"}


def test_arc_parser_preserves_choice_labels_and_gold_key_outside_the_request() -> None:
    source = {
        "id": "science-1",
        "question": {
            "stem": "Which material conducts heat?",
            "choices": [
                {"label": "A", "text": "wood"},
                {"label": "B", "text": "metal"},
                {"label": "C", "text": "cotton"},
            ],
        },
        "answerKey": "B",
    }
    candidates = fresh_eval_data.parse_arc_dev_jsonl(
        (json.dumps(source, separators=(",", ":")) + "\n").encode()
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.dataset_id == "arc-challenge-dev-v1"
    assert candidate.raw_source_id == "science-1"
    assert candidate.answer_id == "B"
    assert candidate.request.context == "Which material conducts heat?"
    assert candidate.request.question == "Choose the correct answer."
    assert [
        (option.id, option.label, option.description) for option in candidate.request.options
    ] == [
        ("A", "wood", None),
        ("B", "metal", None),
        ("C", "cotton", None),
    ]
    from reflex_decisions.rendering import render_prompt

    rendered = render_prompt(candidate.request)
    assert '"label":"wood"' in rendered
    assert '"label":"metal"' in rendered
    assert '"label":"A"' not in rendered
    assert set(candidate.request.model_dump()) == {"context", "question", "options"}


def test_exact_semantic_requests_with_conflicting_gold_labels_fail() -> None:
    def source_row(question_id: str, answer_key: str) -> dict[str, object]:
        return {
            "id": question_id,
            "question": {
                "stem": "A self-authored science question?",
                "choices": [
                    {"label": "A", "text": "first fixture answer"},
                    {"label": "B", "text": "second fixture answer"},
                ],
            },
            "answerKey": answer_key,
        }

    candidates = fresh_eval_data.parse_arc_dev_jsonl(
        (
            "\n".join(
                json.dumps(source_row(question_id, answer_key), separators=(",", ":"))
                for question_id, answer_key in (("fixture-a", "A"), ("fixture-b", "B"))
            )
            + "\n"
        ).encode()
    )

    with pytest.raises(ValueError, match="semantic request has conflicting answer labels"):
        fresh_eval_data._dedupe_candidates(candidates)


def test_selection_freezes_700_rows_and_ten_per_hans_subcase() -> None:
    candidates: list[fresh_eval_data._Candidate] = []
    for group_index in range(30):
        subcase = f"fixture-subcase-{group_index}"
        source_group_id = "hans-subcase:" + hashlib.sha256(subcase.encode()).hexdigest()
        for row_index in range(11):
            raw_id = f"hans-{group_index}-{row_index}"
            candidates.append(
                fresh_eval_data._Candidate(
                    dataset_id="hans-eval-v1",
                    raw_source_id=raw_id,
                    source_group_id=source_group_id,
                    request=DecisionRequest(
                        context=f"self-authored context {raw_id}",
                        question="Choose the fixture option.",
                        options=(Option(id="A", label="A"), Option(id="B", label="B")),
                    ),
                    answer_id="A",
                    source_metadata={
                        "heuristic": "fixture",
                        "subcase": subcase,
                        "template": "fixture-template",
                    },
                )
            )
    for dataset_id, prefix, count in (
        ("winogrande-dev-v1", "wino", 201),
        ("arc-challenge-dev-v1", "arc", 201),
    ):
        for row_index in range(count):
            raw_id = f"{prefix}-{row_index}"
            candidates.append(
                fresh_eval_data._Candidate(
                    dataset_id=dataset_id,
                    raw_source_id=raw_id,
                    source_group_id=raw_id,
                    request=DecisionRequest(
                        context=f"self-authored context {raw_id}",
                        question="Choose the fixture option.",
                        options=(Option(id="A", label="A"), Option(id="B", label="B")),
                    ),
                    answer_id="B",
                    source_metadata={"split": "fixture"},
                )
            )

    records, panel_metadata = fresh_eval_data._select_panel(
        candidates,
        preview_questions=set(),
        preview_question_sha256s={hashlib.sha256(b"self-authored context arc-0").hexdigest()},
    )

    assert len(records) == 700
    assert panel_metadata["selection"]["counts_by_dataset"] == {
        "hans-eval-v1": 300,
        "winogrande-dev-v1": 200,
        "arc-challenge-dev-v1": 200,
    }
    hans_counts: dict[str, int] = {}
    for record in records:
        if record.dataset_id == "hans-eval-v1":
            hans_counts[record.source_group_id] = hans_counts.get(record.source_group_id, 0) + 1
    assert len(hans_counts) == 30
    assert set(hans_counts.values()) == {10}
    assert panel_metadata["selection"]["known_arc_preview_excluded"] == 1
    record_metadata = panel_metadata["record_metadata_by_id"]
    assert "arc-0" not in {
        row["raw_source_id"]
        for row in record_metadata.values()
        if row["dataset_id"] == "arc-challenge-dev-v1"
    }
    assert all("self-authored context" not in record.record_id for record in records)

    presentations = fresh_eval_data.build_presentations(records)
    assert len(presentations) == 1400
    assert len({row["presentation_id"] for row in presentations}) == 1400
    for original, rotated in zip(presentations[::2], presentations[1::2], strict=True):
        assert original["order_index"] == 0
        assert rotated["order_index"] == 1
        assert original["record_id"] == rotated["record_id"]
        original_request = DecisionRequest.model_validate(original["request"])
        rotated_request = DecisionRequest.model_validate(rotated["request"])
        assert rotated_request.options == (
            original_request.options[1:] + original_request.options[:1]
        )
        assert original["request_hash"] == rotated["request_hash"]
        assert set(original) == {
            "presentation_id",
            "record_id",
            "dataset_id",
            "source_group_id",
            "request_hash",
            "order_index",
            "order_ids",
            "request",
        }

    from experiments import fresh_eval_core

    assert fresh_eval_core.validate_presentations(presentations) == presentations
    assert len(fresh_eval_core._parity_presentations(presentations)) == 24


def test_bundle_manifest_pins_and_frozen_file_writer(tmp_path: Path) -> None:
    from reflex_decisions import fresh_eval_bundle

    assert hashlib.sha256(fresh_eval_bundle._manifest_bytes()).hexdigest() == (
        fresh_eval_bundle._EXPECTED_MANIFEST_SHA256
    )
    assert hashlib.sha256(fresh_eval_bundle._recipe_bytes()).hexdigest() == (
        fresh_eval_bundle._EXPECTED_RECIPE_SHA256
    )

    path = tmp_path / "frozen.json"
    fresh_eval_bundle._write_frozen_file(path, b'{"version":1}\n')
    fresh_eval_bundle._write_frozen_file(path, b'{"version":1}\n')
    assert path.read_bytes() == b'{"version":1}\n'
    with pytest.raises(ValueError, match="refusing to replace"):
        fresh_eval_bundle._write_frozen_file(path, b'{"version":2}\n')

    target = tmp_path / "target.json"
    target.write_bytes(b"outside")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        fresh_eval_bundle._write_frozen_file(link, b"outside")
