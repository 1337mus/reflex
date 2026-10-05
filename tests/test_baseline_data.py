import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

import reflex_decisions.baseline_data as baseline_data
from reflex_decisions.baseline_data import CopaItem, parse_copa_dev_xml, record_from_item


def test_copa_dev_maps_relation_and_label_to_stable_semantic_ids() -> None:
    source = b"""<copa-corpus version=\"1.0\">
      <item id=\"42\" asks-for=\"cause\" most-plausible-alternative=\"2\">
        <p>The vase broke.</p><a1>The table shook.</a1><a2>It fell.</a2>
      </item>
    </copa-corpus>"""

    item = parse_copa_dev_xml(source, expected_item_count=1)[0]
    record = record_from_item(item)

    assert record.source_group_id == "42"
    assert record.request.context == "The vase broke."
    assert record.request.question == "Which alternative is the more plausible cause?"
    assert [option.id for option in record.request.options] == ["choice1", "choice2"]
    assert [option.label for option in record.request.options] == [
        "The table shook.",
        "It fell.",
    ]
    assert record.answer_id == "choice2"


def test_effect_relation_is_preserved_as_effect_question() -> None:
    source = b"""<copa-corpus version=\"1.0\">
      <item id=\"9\" asks-for=\"effect\" most-plausible-alternative=\"1\">
        <p>The driver slammed the brakes.</p>
        <a1>The car slowed to a stop.</a1><a2>The driver got a speeding ticket.</a2>
      </item>
    </copa-corpus>"""

    item = parse_copa_dev_xml(source, expected_item_count=1)[0]
    record = record_from_item(item)

    assert record.request.question == "Which alternative is the more plausible effect?"


def test_pilot_selection_is_hash_ordered_and_independent_of_labels() -> None:
    items = tuple(
        CopaItem(
            item_id=str(index),
            relation="cause",
            premise=f"premise {index}",
            choice1=f"choice one {index}",
            choice2=f"choice two {index}",
            answer=index % 2 + 1,
        )
        for index in range(12)
    )

    selected = baseline_data.select_pilot_items(items, sample_size=4)
    changed_labels = tuple(replace(item, answer=3 - item.answer) for item in items)
    selected_with_changed_labels = baseline_data.select_pilot_items(changed_labels, sample_size=4)
    expected = sorted(
        (item.item_id for item in items),
        key=lambda item_id: hashlib.sha256(f"reflex-copa-pilot-v1:{item_id}".encode()).digest(),
    )[:4]

    assert [item.item_id for item in selected] == expected
    assert [item.item_id for item in selected_with_changed_labels] == expected


def test_parser_rejects_normalized_exact_duplicates_even_when_choices_are_reordered() -> None:
    source = b"""<copa-corpus version=\"1.0\">
      <item id=\"1\" asks-for=\"effect\" most-plausible-alternative=\"1\">
        <p>The bell rang.</p><a1>People left.</a1><a2>The bell was struck.</a2>
      </item>
      <item id=\"2\" asks-for=\"effect\" most-plausible-alternative=\"2\">
        <p>The bell rang.</p><a1>The bell was struck.</a1><a2>People left.</a2>
      </item>
    </copa-corpus>"""

    with pytest.raises(ValueError, match="normalized exact duplicate"):
        parse_copa_dev_xml(source, expected_item_count=2)


def test_parser_rejects_unrecognized_copa_xml_version() -> None:
    source = b"""<copa-corpus version=\"2.0\">
      <item id=\"1\" asks-for=\"cause\" most-plausible-alternative=\"1\">
        <p>Premise.</p><a1>First.</a1><a2>Second.</a2>
      </item>
    </copa-corpus>"""

    with pytest.raises(ValueError, match="unsupported COPA XML root"):
        parse_copa_dev_xml(source, expected_item_count=1)


def test_parser_rejects_nested_markup_in_source_text() -> None:
    source = b"""<copa-corpus version=\"1.0\">
      <item id=\"1\" asks-for=\"cause\" most-plausible-alternative=\"1\">
        <p>Premise with <em>markup</em>.</p><a1>First.</a1><a2>Second.</a2>
      </item>
    </copa-corpus>"""

    with pytest.raises(ValueError, match="nested COPA text markup"):
        parse_copa_dev_xml(source, expected_item_count=1)


def test_source_pin_validation_rejects_archive_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        baseline_data.validate_source_pins(
            b"wrong archive",
            b"wrong development XML",
            source_revision=baseline_data.SOURCE_REVISION,
        )


def test_pilot_builder_emits_32_original_records_in_hash_order() -> None:
    elements = [
        (
            f'<item id="{index}" asks-for="cause" '
            f'most-plausible-alternative="{index % 2 + 1}">'
            f"<p>Premise {index}.</p><a1>First {index}.</a1><a2>Second {index}.</a2>"
            "</item>"
        )
        for index in range(1, 501)
    ]
    source = f'<copa-corpus version="1.0">{"".join(elements)}</copa-corpus>'.encode()

    records = baseline_data.build_pilot_records(source)
    expected_ids = sorted(
        (str(index) for index in range(1, 501)),
        key=lambda item_id: hashlib.sha256(f"reflex-copa-pilot-v1:{item_id}".encode()).digest(),
    )[:32]

    assert len(records) == 32
    assert [record.source_group_id for record in records] == expected_ids
    assert all(len(record.request.options) == 2 for record in records)
    assert {record.dataset_id for record in records} == {baseline_data.DATASET_ID}


def test_record_serializer_emits_canonical_jsonl() -> None:
    record = baseline_data.record_from_item(
        CopaItem(
            item_id="7",
            relation="effect",
            premise="Premise.",
            choice1="First.",
            choice2="Second.",
            answer=1,
        )
    )

    payload = baseline_data.serialize_records((record,))

    assert payload == (
        b'{"answer_id":"choice1","dataset_id":"copa-dev-pilot-v1",'
        b'"record_id":"copa-dev-pilot-v1-7","request":{"context":"Premise.",'
        b'"options":[{"id":"choice1","label":"First."},'
        b'{"id":"choice2","label":"Second."}],'
        b'"question":"Which alternative is the more plausible effect?"},'
        b'"source_group_id":"7"}\n'
    )


def test_prepared_data_verifier_binds_and_audits_the_committed_recipe(
    tmp_path, monkeypatch
) -> None:
    items = tuple(
        CopaItem(
            item_id=str(index),
            relation="cause" if index % 2 else "effect",
            premise=f"Premise {index}.",
            choice1=f"First {index}.",
            choice2=f"Second {index}.",
            answer=index % 2 + 1,
        )
        for index in range(1, 33)
    )
    records = tuple(
        record_from_item(item) for item in baseline_data.select_pilot_items(items, sample_size=32)
    )
    records_bytes = baseline_data.serialize_records(records)
    manifest_bytes = (
        Path(__file__).parents[1] / "data/baselines/copa-dev-manifest.json"
    ).read_bytes()
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records_path.write_bytes(records_bytes)
    manifest_path.write_bytes(manifest_bytes)
    records_digest = hashlib.sha256(records_bytes).hexdigest()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    monkeypatch.setattr(baseline_data, "EXPECTED_RECORDS_SHA256", records_digest, raising=False)
    monkeypatch.setattr(baseline_data, "EXPECTED_MANIFEST_SHA256", manifest_digest, raising=False)

    manifest, loaded_records = baseline_data.verify_prepared_data(records_path, manifest_path)

    assert manifest.datasets[0].dataset_id == baseline_data.DATASET_ID
    assert len(loaded_records) == 32


def test_record_writer_never_overwrites_an_existing_output(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b"existing bytes")

    with pytest.raises(FileExistsError):
        baseline_data.write_records_exclusive(path, b"replacement bytes")

    assert path.read_bytes() == b"existing bytes"
