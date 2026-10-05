import hashlib
import io
import json
from dataclasses import replace
from zipfile import ZipFile

import pytest

from reflex_decisions import broader_data


def test_boolq_selection_uses_connected_passage_title_groups_once() -> None:
    rows = [
        {
            "question": "Can it fly?",
            "answer": True,
            "passage": "Passage A",
            "title": "Shared page",
        },
        {
            "question": "Does it swim?",
            "answer": False,
            "passage": "Passage B",
            "title": " Shared   page ",
        },
        {
            "question": " Can it fly? ",
            "answer": True,
            "passage": " Passage A ",
            "title": "Shared page",
        },
        {
            "question": "Is the third page relevant?",
            "answer": True,
            "passage": "Passage C",
            "title": "Separate page",
        },
    ]
    source = "\n".join(json.dumps(row) for row in rows).encode()

    items = broader_data.parse_boolq_dev_jsonl(source)
    selected = broader_data.select_boolq_pilot_items(items, sample_size=2)

    assert len(items) == 3
    assert len(selected) == 2
    assert len({item.source_group_id for item in selected}) == 2


def test_boolq_duplicate_rows_keep_all_title_links_before_deduplication() -> None:
    rows = [
        {"question": "Q1", "answer": True, "passage": "P1", "title": "Title A"},
        {"question": " Q1 ", "answer": True, "passage": " P1 ", "title": "Title B"},
        {"question": "Q2", "answer": False, "passage": "P2", "title": "Title B"},
        {"question": "Q3", "answer": True, "passage": "P3", "title": "Title C"},
    ]
    source = "\n".join(json.dumps(row) for row in rows).encode()

    items = broader_data.parse_boolq_dev_jsonl(source)

    assert len(items) == 3
    group_by_question = {item.question.strip(): item.source_group_id for item in items}
    assert group_by_question["Q1"] == group_by_question["Q2"]
    assert group_by_question["Q1"] != group_by_question["Q3"]


def test_boolq_selection_is_stable_under_row_order_and_label_changes() -> None:
    rows = [
        {
            "question": f"Question {index}?",
            "answer": index % 2 == 0,
            "passage": f"Passage {index}.",
            "title": f"Title {index}",
        }
        for index in range(5)
    ]
    items = broader_data.parse_boolq_dev_jsonl("\n".join(json.dumps(row) for row in rows).encode())
    flipped = tuple(replace(item, answer=not item.answer) for item in items)

    baseline = broader_data.select_boolq_pilot_items(items, sample_size=3)
    reordered = broader_data.select_boolq_pilot_items(tuple(reversed(items)), sample_size=3)
    relabeled = broader_data.select_boolq_pilot_items(flipped, sample_size=3)

    assert [item.source_item_id for item in baseline] == [item.source_item_id for item in reordered]
    assert [item.source_item_id for item in baseline] == [item.source_item_id for item in relabeled]


def test_jsonl_parser_does_not_treat_unicode_line_separator_as_a_record_boundary() -> None:
    row = {
        "question": "Q\u2028continued",
        "answer": True,
        "passage": "P\u2028continued",
        "title": "A page",
    }
    source = (json.dumps(row, ensure_ascii=False) + "\n").encode()

    items = broader_data.parse_boolq_dev_jsonl(source)

    assert len(items) == 1


def test_snli_dev_groups_caption_and_premise_links_and_skips_unlabeled_rows() -> None:
    rows = [
        {
            "captionID": "caption-a",
            "pairID": "pair-1",
            "sentence1": "A person is outside.",
            "sentence2": "Someone is outdoors.",
            "gold_label": "entailment",
        },
        {
            "captionID": "caption-a",
            "pairID": "pair-2",
            "sentence1": "A person is outside.",
            "sentence2": "Someone is waiting.",
            "gold_label": "neutral",
        },
        {
            "captionID": "caption-b",
            "pairID": "pair-3",
            "sentence1": " A person is outside. ",
            "sentence2": "Nobody is indoors.",
            "gold_label": "contradiction",
        },
        {
            "captionID": "caption-c",
            "pairID": "pair-4",
            "sentence1": "A dog is sleeping.",
            "sentence2": "An animal is resting.",
            "gold_label": "entailment",
        },
        {
            "captionID": "caption-d",
            "pairID": "pair-5",
            "sentence1": "A dog is sleeping.",
            "sentence2": "A dog is awake.",
            "gold_label": "-",
        },
    ]
    source = "\n".join(json.dumps(row) for row in rows).encode()

    items = broader_data.parse_snli_dev_jsonl(source)
    selected = broader_data.select_snli_pilot_items(items, sample_size=2)

    assert len(items) == 4
    assert len(selected) == 2
    assert len({item.source_group_id for item in selected}) == 2
    assert {item.label for item in items} == {"entailment", "neutral", "contradiction"}


def test_snli_duplicate_pairs_preserve_all_caption_links_before_deduplication() -> None:
    rows = [
        {
            "captionID": "caption-a",
            "pairID": "pair-1",
            "sentence1": "Premise one.",
            "sentence2": "Hypothesis one.",
            "gold_label": "entailment",
        },
        {
            "captionID": "caption-b",
            "pairID": "pair-1-duplicate",
            "sentence1": " Premise one. ",
            "sentence2": " Hypothesis one. ",
            "gold_label": "entailment",
        },
        {
            "captionID": "caption-b",
            "pairID": "pair-2",
            "sentence1": "Premise two.",
            "sentence2": "Hypothesis two.",
            "gold_label": "neutral",
        },
        {
            "captionID": "caption-c",
            "pairID": "pair-3",
            "sentence1": "Premise three.",
            "sentence2": "Hypothesis three.",
            "gold_label": "contradiction",
        },
    ]
    source = "\n".join(json.dumps(row) for row in rows).encode()

    items = broader_data.parse_snli_dev_jsonl(source)

    assert len(items) == 3
    group_by_pair = {item.source_item_id: item.source_group_id for item in items}
    assert group_by_pair["pair-1"] == group_by_pair["pair-2"]
    assert group_by_pair["pair-1"] != group_by_pair["pair-3"]


def test_snli_selection_is_stable_under_row_order_and_label_changes() -> None:
    labels = ("entailment", "neutral", "contradiction")
    rows = [
        {
            "captionID": f"caption-{index}",
            "pairID": f"pair-{index}",
            "sentence1": f"Premise {index}.",
            "sentence2": f"Hypothesis {index}.",
            "gold_label": labels[index % len(labels)],
        }
        for index in range(5)
    ]
    items = broader_data.parse_snli_dev_jsonl("\n".join(json.dumps(row) for row in rows).encode())
    relabeled = tuple(
        replace(item, label=labels[(labels.index(item.label) + 1) % len(labels)]) for item in items
    )

    baseline = broader_data.select_snli_pilot_items(items, sample_size=3)
    reordered = broader_data.select_snli_pilot_items(tuple(reversed(items)), sample_size=3)
    relabeled_selection = broader_data.select_snli_pilot_items(relabeled, sample_size=3)

    assert [item.source_item_id for item in baseline] == [item.source_item_id for item in reordered]
    assert [item.source_item_id for item in baseline] == [
        item.source_item_id for item in relabeled_selection
    ]


def test_records_preserve_native_label_ids_and_describe_snli_relations() -> None:
    boolq_item = broader_data.BoolQItem(
        source_item_id="boolq-source-id",
        source_group_id="boolq-group-id",
        question="Is the claim true?",
        passage="The passage.",
        title="A page",
        answer=False,
    )
    snli_item = broader_data.SnliItem(
        source_item_id="snli-pair-id",
        source_group_id="snli-group-id",
        premise="The premise.",
        hypothesis="The hypothesis.",
        label="neutral",
    )

    boolq_record = broader_data.record_from_boolq_item(boolq_item)
    snli_record = broader_data.record_from_snli_item(snli_item)

    assert [option.id for option in boolq_record.request.options] == ["yes", "no"]
    assert boolq_record.answer_id == "no"
    assert boolq_record.request.context == "The passage."
    assert boolq_record.request.question == "Is the claim true?"
    assert [option.id for option in snli_record.request.options] == [
        "entailment",
        "neutral",
        "contradiction",
    ]
    assert [option.description for option in snli_record.request.options] == [
        "The premise guarantees the hypothesis.",
        "The premise neither guarantees nor contradicts the hypothesis.",
        "The premise contradicts the hypothesis.",
    ]
    assert snli_record.answer_id == "neutral"
    assert snli_record.request.question == broader_data.SNLI_QUESTION


def test_snli_record_id_hashes_label_correlated_source_pair_id() -> None:
    item = broader_data.SnliItem(
        source_item_id="source-pair-r1c",
        source_group_id="group-id",
        premise="The premise.",
        hypothesis="The hypothesis.",
        label="contradiction",
    )

    record = broader_data.record_from_snli_item(item)

    expected_hash = hashlib.sha256(item.source_item_id.encode("utf-8")).hexdigest()
    assert record.record_id == f"{broader_data.SNLI_DATASET_ID}-{expected_hash}"
    assert item.source_item_id not in record.record_id


def test_pilot_builder_selects_32_original_groups_for_each_development_dataset() -> None:
    boolq_source = "\n".join(
        json.dumps(
            {
                "question": f"Question {index}?",
                "answer": index % 2 == 0,
                "passage": f"Passage {index}.",
                "title": f"Title {index}",
            }
        )
        for index in range(32)
    ).encode()
    snli_source = "\n".join(
        json.dumps(
            {
                "captionID": f"caption-{index}",
                "pairID": f"pair-{index}",
                "sentence1": f"Premise {index}.",
                "sentence2": f"Hypothesis {index}.",
                "gold_label": ("entailment", "neutral", "contradiction")[index % 3],
            }
        )
        for index in range(32)
    ).encode()

    records = broader_data.build_pilot_records(boolq_source, snli_source)

    boolq_records = [
        record for record in records if record.dataset_id == broader_data.BOOLQ_DATASET_ID
    ]
    snli_records = [
        record for record in records if record.dataset_id == broader_data.SNLI_DATASET_ID
    ]
    assert len(records) == 64
    assert len(boolq_records) == len(snli_records) == 32
    assert len({record.source_group_id for record in boolq_records}) == 32
    assert len({record.source_group_id for record in snli_records}) == 32


def _write_verifier_fixture(tmp_path, monkeypatch):
    boolq_records = tuple(
        broader_data.record_from_boolq_item(
            broader_data.BoolQItem(
                source_item_id=f"boolq-source-{index}",
                source_group_id=f"boolq-group-{index}",
                question=f"Question {index}?",
                passage=f"Passage {index}.",
                title="",
                answer=index % 2 == 0,
            )
        )
        for index in range(32)
    )
    snli_items = tuple(
        broader_data.SnliItem(
            source_item_id=f"snli-pair-{index}",
            source_group_id=f"snli-group-{index}",
            premise=f"Premise {index}.",
            hypothesis=f"Hypothesis {index}.",
            label=("entailment", "neutral", "contradiction")[index % 3],
        )
        for index in range(32)
    )
    snli_records = tuple(broader_data.record_from_snli_item(item) for item in snli_items)
    records = boolq_records + snli_records
    records_bytes = broader_data.serialize_records(records)
    manifest_bytes = broader_data.serialize_manifest()
    recipe_bytes = broader_data.serialize_recipe(
        broader_data.build_recipe(
            records,
            records_bytes,
            manifest_bytes,
            snli_source_item_ids=[item.source_item_id for item in snli_items],
        )
    )
    monkeypatch.setattr(
        broader_data, "EXPECTED_RECORDS_SHA256", hashlib.sha256(records_bytes).hexdigest()
    )
    monkeypatch.setattr(
        broader_data, "EXPECTED_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest()
    )
    monkeypatch.setattr(
        broader_data, "EXPECTED_RECIPE_SHA256", hashlib.sha256(recipe_bytes).hexdigest()
    )
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    recipe_path = tmp_path / "recipe.json"
    records_path.write_bytes(records_bytes)
    manifest_path.write_bytes(manifest_bytes)
    recipe_path.write_bytes(recipe_bytes)
    return records_path, manifest_path, recipe_path


def test_prepared_data_verifier_checks_32_groups_per_dataset_and_manifest(
    tmp_path, monkeypatch
) -> None:
    records_path, manifest_path, recipe_path = _write_verifier_fixture(tmp_path, monkeypatch)

    manifest, records = broader_data.verify_prepared_data(records_path, manifest_path, recipe_path)

    assert len(records) == 64
    assert [dataset.dataset_id for dataset in manifest.datasets] == [
        broader_data.BOOLQ_DATASET_ID,
        broader_data.SNLI_DATASET_ID,
    ]
    assert manifest.held_out_families == ()


def test_prepared_data_verifier_rejects_record_bytes_outside_the_pinned_receipt(
    tmp_path, monkeypatch
) -> None:
    records_path, manifest_path, recipe_path = _write_verifier_fixture(tmp_path, monkeypatch)
    records_path.write_bytes(records_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="records SHA-256 mismatch"):
        broader_data.verify_prepared_data(records_path, manifest_path, recipe_path)


def test_source_pin_validation_reads_snli_development_and_readme_members_only(monkeypatch) -> None:
    boolq_source = b"pinned BoolQ parquet"
    boolq_dev = b'{"question":"Q?","answer":true,"passage":"P"}\n'
    snli_dev = b'{"pairID":"p1","gold_label":"entailment"}\n'
    archive_buffer = io.BytesIO()
    with ZipFile(archive_buffer, "w") as archive:
        archive.writestr(broader_data.SNLI_DEV_MEMBER, snli_dev)
        archive.writestr(broader_data.SNLI_README_MEMBER, b"SNLI attribution README")
        archive.writestr("snli_1.0/snli_1.0_train.jsonl", b"do not read")
        archive.writestr("snli_1.0/snli_1.0_test.jsonl", b"do not read")
    snli_archive = archive_buffer.getvalue()
    monkeypatch.setattr(
        broader_data, "BOOLQ_SOURCE_SHA256", hashlib.sha256(boolq_source).hexdigest()
    )
    monkeypatch.setattr(broader_data, "BOOLQ_DEV_SHA256", hashlib.sha256(boolq_dev).hexdigest())
    monkeypatch.setattr(
        broader_data,
        "SNLI_SOURCE_ARCHIVE_SHA256",
        hashlib.sha256(snli_archive).hexdigest(),
    )
    monkeypatch.setattr(broader_data, "SNLI_DEV_SHA256", hashlib.sha256(snli_dev).hexdigest())
    read_members = []
    original_read = ZipFile.read

    def record_member_read(archive, member, *args, **kwargs):
        read_members.append(member)
        return original_read(archive, member, *args, **kwargs)

    monkeypatch.setattr(broader_data.ZipFile, "read", record_member_read)

    broader_data.validate_source_pins(boolq_source, boolq_dev, snli_archive, snli_dev)
    assert read_members == [broader_data.SNLI_README_MEMBER, broader_data.SNLI_DEV_MEMBER]

    with pytest.raises(ValueError, match="SNLI extracted development"):
        broader_data.validate_source_pins(
            boolq_source, boolq_dev, snli_archive, b"different dev file"
        )
