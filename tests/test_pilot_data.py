import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pytest

from reflex_decisions import pilot_data
from reflex_decisions.pilot_data import (
    DATASET_RECORD_COUNTS,
    DBPEDIA_DATASET_IDS,
    DBPEDIA_LABELS,
    SMS_DATASET_IDS,
    DbpediaRow,
    GroupedItem,
    SelectedSnliItem,
    SmsRow,
    build_manifest,
    build_recipe,
    group_dbpedia_rows,
    group_sms_rows,
    record_from_dbpedia_item,
    record_from_sms_item,
    record_from_snli_item,
    select_snli_items,
    serialize_manifest,
    serialize_recipe,
    serialize_records,
    stratified_allocate,
)
from reflex_decisions.pilot_data_sources import parse_sms_collection, parse_snli_native_row_ids


def test_dbpedia_groups_normalized_shared_anchors_and_discards_label_conflicts() -> None:
    rows = (
        DbpediaRow(0, "Alpha", "Common article", 0),
        DbpediaRow(1, "Beta", " common   ARTICLE ", 1),
        DbpediaRow(2, "Gamma", "Repeated text", 1),
        DbpediaRow(3, "Other title", " repeated   TEXT ", 1),
        DbpediaRow(4, "Unique", "Separate article", 2),
    )

    result = group_dbpedia_rows(rows)

    assert result.conflicting_group_count == 1
    assert [item.source_row_id for item in result.items] == [2, 4]
    assert [item.label for item in result.items] == ["1", "2"]
    assert result.items[0].group_id != result.items[1].group_id


def test_stratified_allocation_uses_source_id_rank_and_fixed_split_order() -> None:
    source_id = "source:v1"
    items = tuple(
        GroupedItem(
            source_row_id=index,
            item_id=f"item-{index}",
            group_id=f"group-{label}-{index}",
            label=label,
        )
        for label in ("ham", "spam")
        for index in range(5)
    )

    allocation = stratified_allocate(
        items,
        {"ham": (1, 1, 1), "spam": (1, 1, 1)},
        source_id=source_id,
    )

    ranked_by_label = {
        label: sorted(
            (item for item in items if item.label == label),
            key=lambda item: (
                sha256(
                    f"reflex-real-pilot-v1:20261005:{source_id}:{item.group_id}".encode()
                ).hexdigest(),
                item.group_id,
            ),
        )
        for label in ("ham", "spam")
    }
    assert allocation.train == tuple(ranked_by_label[label][0] for label in ("ham", "spam"))
    assert allocation.development == tuple(ranked_by_label[label][1] for label in ("ham", "spam"))
    assert allocation.calibration == tuple(ranked_by_label[label][2] for label in ("ham", "spam"))


def test_sms_groups_nfkc_casefold_and_whitespace_and_drops_conflicts() -> None:
    rows = (
        SmsRow(0, "spam", "ＦＲＥＥ  offer"),
        SmsRow(1, "ham", "free offer"),
        SmsRow(2, "ham", "Meeting at noon"),
        SmsRow(3, "ham", " meeting   AT NOON "),
        SmsRow(4, "spam", "Different message"),
    )

    result = group_sms_rows(rows)

    assert result.conflicting_group_count == 1
    assert [item.source_row_id for item in result.items] == [2, 4]
    assert [item.label for item in result.items] == ["ham", "spam"]


@dataclass(frozen=True)
class _SnliCandidate:
    source_item_id: str
    source_group_id: str
    premise: str
    hypothesis: str
    label: str


def test_snli_selection_excludes_prior_groups_and_uses_minimum_pair_id() -> None:
    items = (
        _SnliCandidate("z-pair", "group-a", "premise a", "hyp z", "entailment"),
        _SnliCandidate("a-pair", "group-a", "premise a", "hyp a", "neutral"),
        _SnliCandidate("m-pair", "group-b", "premise b", "hyp b", "contradiction"),
        _SnliCandidate("excluded-pair", "prior-group", "premise x", "hyp x", "neutral"),
        _SnliCandidate("b-pair", "group-c", "premise c", "hyp c", "entailment"),
    )
    row_ids = {"z-pair": 4, "a-pair": 8, "m-pair": 10, "excluded-pair": 12, "b-pair": 14}

    selected = select_snli_items(
        items,
        row_ids,
        {"prior-group"},
        sample_size=2,
    )

    expected_groups = sorted(
        {"group-a", "group-b", "group-c"},
        key=lambda group_id: (
            sha256(
                f"reflex-real-pilot-v1:20261005:stanford-snli-1.0:{group_id}".encode()
            ).hexdigest(),
            group_id,
        ),
    )[:2]
    by_group = {item.group_id: item for item in selected}
    assert set(by_group) == set(expected_groups)
    for group_id in expected_groups:
        representative = min(
            (item for item in items if item.source_group_id == group_id),
            key=lambda item: item.source_item_id,
        )
        selected_item = by_group[group_id]
        assert (
            selected_item.item_id
            == sha256(
                f"reflex-real-pilot-item-v1:stanford-snli-1.0:{representative.source_item_id}".encode()
            ).hexdigest()
        )
        assert selected_item.source_row_id == row_ids[representative.source_item_id]


def test_pilot_manifest_uses_canonical_sources_and_only_groups_custom_splits() -> None:
    manifest = build_manifest()

    specs = {spec.dataset_id: spec for spec in manifest.datasets}
    assert set(specs) == {
        "dbpedia14-pilot-v1-train",
        "dbpedia14-pilot-v1-development",
        "dbpedia14-pilot-v1-calibration",
        "sms-pilot-v1-train",
        "sms-pilot-v1-development",
        "sms-pilot-v1-calibration",
        "snli-pilot-v1-development",
    }
    assert {
        specs[dataset_id].source_id for dataset_id in specs if dataset_id.startswith("dbpedia14-")
    } == {"fancyzhx/dbpedia_14"}
    assert {
        specs[dataset_id].source_id for dataset_id in specs if dataset_id.startswith("sms-")
    } == {"uci-sms-spam-collection-228"}
    assert specs["snli-pilot-v1-development"].source_id == "stanford-snli-1.0"
    assert specs["snli-pilot-v1-development"].task_family == "natural-language-inference-three-way"
    assert manifest.group_partitioned_sources == (
        "fancyzhx/dbpedia_14",
        "uci-sms-spam-collection-228",
    )
    assert manifest.held_out_families == ()


def test_record_builders_keep_opaque_ids_independent_of_gold_labels() -> None:
    dbpedia_item = GroupedItem(3, "item-hash", "group-hash", "1")
    dbpedia_row = DbpediaRow(3, "Example title", "Example content", 1)

    dbpedia_record = record_from_dbpedia_item(
        dbpedia_item,
        dbpedia_row,
        dataset_id="dbpedia14-pilot-v1-train",
    )
    relabeled_record = record_from_dbpedia_item(
        GroupedItem(3, "item-hash", "group-hash", "2"),
        DbpediaRow(3, "Example title", "Example content", 2),
        dataset_id="dbpedia14-pilot-v1-train",
    )
    sms_record = record_from_sms_item(
        GroupedItem(4, "sms-item-hash", "sms-group-hash", "spam"),
        SmsRow(4, "spam", "A sample message"),
        dataset_id="sms-pilot-v1-train",
    )
    snli_record = record_from_snli_item(
        SelectedSnliItem(
            source_row_id=5,
            item_id="snli-item-hash",
            group_id="snli-group-hash",
            label="neutral",
            premise="A person is outside.",
            hypothesis="The person is reading.",
        )
    )

    assert dbpedia_record.record_id == "dbpedia14-pilot-v1-train-item-hash"
    assert relabeled_record.record_id == dbpedia_record.record_id
    assert dbpedia_record.answer_id == "1"
    assert len(dbpedia_record.request.options) == 14
    assert dbpedia_record.request.context == "Title: Example title\n\nExample content"
    assert sms_record.record_id == "sms-pilot-v1-train-sms-item-hash"
    assert sms_record.answer_id == "spam"
    assert tuple(option.id for option in sms_record.request.options) == ("ham", "spam")
    assert snli_record.record_id == "snli-pilot-v1-development-snli-item-hash"
    assert snli_record.answer_id == "neutral"
    assert len(snli_record.request.options) == 3


def test_recipe_records_source_pins_output_hashes_and_no_source_text() -> None:
    dataset_ids = (
        "dbpedia14-pilot-v1-train",
        "dbpedia14-pilot-v1-development",
        "dbpedia14-pilot-v1-calibration",
        "sms-pilot-v1-train",
        "sms-pilot-v1-development",
        "sms-pilot-v1-calibration",
        "snli-pilot-v1-development",
    )
    selected = {
        dataset_id: (GroupedItem(index, f"item-{index}", f"group-{index}", "0"),)
        for index, dataset_id in enumerate(dataset_ids)
    }
    max_context = {dataset_id: 10 + index for index, dataset_id in enumerate(dataset_ids)}

    recipe = build_recipe(
        selected,
        b"records-bytes",
        b"manifest-bytes",
        prior_snli_group_ids=("prior-group",),
        dbpedia_conflicting_group_count=12,
        sms_conflicting_group_count=0,
        snli_unlabeled_row_count=100,
        max_request_context_characters=max_context,
    )

    serialized = serialize_recipe(recipe)
    decoded = json.loads(serialized)
    assert set(decoded) == {
        "recipe_id",
        "sources",
        "selection",
        "selected",
        "exclusions",
        "outputs",
    }
    assert decoded["recipe_id"] == "reflex-real-pilot-v1"
    assert decoded["outputs"]["records_sha256"] == sha256(b"records-bytes").hexdigest()
    assert decoded["outputs"]["manifest_sha256"] == sha256(b"manifest-bytes").hexdigest()
    assert decoded["outputs"]["max_request_context_characters"] == max_context
    assert decoded["selected"]["dbpedia14-pilot-v1-train"] == {
        "source_row_ids": [0],
        "item_ids": ["item-0"],
        "group_ids": ["group-0"],
    }
    assert "protocol" not in decoded
    assert b"raw message" not in serialized


def test_verifier_rejects_records_before_loading_unpinned_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    recipe_path = tmp_path / "recipe.json"
    records_path.write_bytes(b"tampered records")
    manifest_path.write_bytes(b"not reached")
    recipe_path.write_bytes(b"not reached")
    monkeypatch.setattr(
        pilot_data,
        "EXPECTED_RECORDS_SHA256",
        sha256(b"approved records").hexdigest(),
    )
    monkeypatch.setattr(pilot_data, "EXPECTED_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setattr(pilot_data, "EXPECTED_RECIPE_SHA256", "b" * 64)

    with pytest.raises(ValueError, match="records SHA-256 mismatch"):
        pilot_data.verify_prepared_data(records_path, manifest_path, recipe_path)


def test_sms_source_parser_preserves_stable_rows_and_embedded_tabs() -> None:
    rows = parse_sms_collection(b"ham\tA message\twith a tab\nspam\tAnother message\n")

    assert rows == (
        SmsRow(0, "ham", "A message\twith a tab"),
        SmsRow(1, "spam", "Another message"),
    )


def test_snli_native_row_map_keeps_first_pair_id_occurrence_and_unlabeled_count() -> None:
    source = b"\n".join(
        (
            b'{"pairID":"pair-a","gold_label":"entailment"}',
            b'{"pairID":"unlabeled","gold_label":"-"}',
            b'{"pairID":"pair-a","gold_label":"entailment"}',
            b'{"pairID":"pair-b","gold_label":"neutral"}',
        )
    )

    row_ids, unlabeled_count, row_count = parse_snli_native_row_ids(source)

    assert row_ids == {"pair-a": 0, "pair-b": 3}
    assert unlabeled_count == 1
    assert row_count == 4


def test_verifier_accepts_complete_pinned_synthetic_data_and_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = {}
    records = []
    next_row_id = {"dbpedia": 0, "sms": 0, "snli": 0}
    for dataset_id in DATASET_RECORD_COUNTS:
        current = []
        if dataset_id in DBPEDIA_DATASET_IDS:
            per_label = 18 if dataset_id.endswith("-train") else 4
            for label_index, _label_name in enumerate(DBPEDIA_LABELS):
                for _repeat in range(per_label):
                    row_id = next_row_id["dbpedia"]
                    next_row_id["dbpedia"] += 1
                    item_id = sha256(f"item:dbpedia:{row_id}".encode()).hexdigest()
                    group_id = sha256(f"group:dbpedia:{row_id}".encode()).hexdigest()
                    item = GroupedItem(row_id, item_id, group_id, str(label_index))
                    row = DbpediaRow(row_id, f"Title {row_id}", f"Content {row_id}", label_index)
                    current.append(item)
                    records.append(record_from_dbpedia_item(item, row, dataset_id=dataset_id))
        elif dataset_id in SMS_DATASET_IDS:
            per_label = 126 if dataset_id.endswith("-train") else 30
            for label in ("ham", "spam"):
                for _repeat in range(per_label):
                    row_id = next_row_id["sms"]
                    next_row_id["sms"] += 1
                    item_id = sha256(f"item:sms:{row_id}".encode()).hexdigest()
                    group_id = sha256(f"group:sms:{row_id}".encode()).hexdigest()
                    item = GroupedItem(row_id, item_id, group_id, label)
                    row = SmsRow(row_id, label, f"Message {row_id}")
                    current.append(item)
                    records.append(record_from_sms_item(item, row, dataset_id=dataset_id))
        else:
            for index in range(DATASET_RECORD_COUNTS[dataset_id]):
                row_id = next_row_id["snli"]
                next_row_id["snli"] += 1
                item = SelectedSnliItem(
                    row_id,
                    sha256(f"item:snli:{row_id}".encode()).hexdigest(),
                    sha256(f"group:snli:{row_id}".encode()).hexdigest(),
                    ("entailment", "neutral", "contradiction")[index % 3],
                    f"Premise {row_id}",
                    f"Hypothesis {row_id}",
                )
                current.append(item)
                records.append(record_from_snli_item(item))
        selected[dataset_id] = tuple(current)

    records_bytes = serialize_records(records)
    manifest_bytes = serialize_manifest(build_manifest())
    max_context = {
        dataset_id: max(
            len(record.request.context) for record in records if record.dataset_id == dataset_id
        )
        for dataset_id in DATASET_RECORD_COUNTS
    }
    recipe = build_recipe(
        selected,
        records_bytes,
        manifest_bytes,
        prior_snli_group_ids=tuple(
            sha256(f"prior:{index}".encode()).hexdigest() for index in range(32)
        ),
        dbpedia_conflicting_group_count=12,
        sms_conflicting_group_count=0,
        snli_unlabeled_row_count=0,
        max_request_context_characters=max_context,
    )
    recipe_bytes = serialize_recipe(recipe)
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    recipe_path = tmp_path / "recipe.json"
    records_path.write_bytes(records_bytes)
    manifest_path.write_bytes(manifest_bytes)
    recipe_path.write_bytes(recipe_bytes)
    monkeypatch.setattr(pilot_data, "EXPECTED_RECORDS_SHA256", sha256(records_bytes).hexdigest())
    monkeypatch.setattr(pilot_data, "EXPECTED_MANIFEST_SHA256", sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(pilot_data, "EXPECTED_RECIPE_SHA256", sha256(recipe_bytes).hexdigest())

    manifest, verified_records, verified_recipe = pilot_data.verify_prepared_data(
        records_path,
        manifest_path,
        recipe_path,
    )

    assert manifest == build_manifest()
    assert len(verified_records) == 864
    assert verified_recipe == recipe
