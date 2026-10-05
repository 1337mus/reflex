"""Build the pinned, deterministic local real-pilot dataset artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from reflex_decisions import broader_data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.pilot_data import (
    DATASET_RECORD_COUNTS,
    DBPEDIA_DATASET_IDS,
    DBPEDIA_SOURCE_ID,
    DBPEDIA_SOURCE_SHA256,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RECIPE_PATH,
    DEFAULT_RECORDS_PATH,
    SMS_ARCHIVE_SHA256,
    SMS_DATASET_IDS,
    SMS_SOURCE_ID,
    SMS_TEXT_SHA256,
    SNLI_DATASET_ID,
    SNLI_DEV_SHA256,
    SNLI_SOURCE_ARCHIVE_SHA256,
    DbpediaRow,
    GroupedItem,
    SelectedSnliItem,
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
from reflex_decisions.pilot_data_sources import (
    parse_snli_items,
    parse_snli_native_row_ids,
    read_dbpedia_rows,
    read_sms_rows,
    verify_source_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DBPEDIA_PATH = PROJECT_ROOT / "data/raw/dbpedia14_train.parquet"
SMS_TEXT_PATH = PROJECT_ROOT / "data/raw/uci_sms_spam_collection.txt"
SMS_ARCHIVE_PATH = PROJECT_ROOT / "data/raw/uci_sms_spam_collection.zip"
SNLI_DEV_PATH = PROJECT_ROOT / "data/raw/snli_1.0_dev.jsonl"
SNLI_ARCHIVE_PATH = PROJECT_ROOT / "data/raw/snli_1.0.zip"
PRIOR_RECIPE_PATH = PROJECT_ROOT / "data/baselines/broader-dev-recipe.json"


def prepare_real_pilot() -> dict[str, object]:
    """Verify pinned sources, select groups, and create raw-text-free receipts."""

    dbpedia_grouping = group_dbpedia_rows(read_dbpedia_rows(DBPEDIA_PATH, DBPEDIA_SOURCE_SHA256))
    dbpedia_targets = {str(label): (18, 4, 4) for label in range(14)}
    dbpedia_allocation = stratified_allocate(
        dbpedia_grouping.items,
        dbpedia_targets,
        source_id=DBPEDIA_SOURCE_ID,
    )
    dbpedia_selected = {
        DBPEDIA_DATASET_IDS[0]: dbpedia_allocation.train,
        DBPEDIA_DATASET_IDS[1]: dbpedia_allocation.development,
        DBPEDIA_DATASET_IDS[2]: dbpedia_allocation.calibration,
    }
    dbpedia_rows = _read_dbpedia_representatives(
        {item.source_row_id for items in dbpedia_selected.values() for item in items}
    )

    sms_rows = read_sms_rows(
        SMS_TEXT_PATH,
        SMS_ARCHIVE_PATH,
        text_sha256=SMS_TEXT_SHA256,
        archive_sha256=SMS_ARCHIVE_SHA256,
    )
    sms_grouping = group_sms_rows(sms_rows)
    sms_allocation = stratified_allocate(
        sms_grouping.items,
        {"ham": (126, 30, 30), "spam": (126, 30, 30)},
        source_id=SMS_SOURCE_ID,
    )
    sms_selected = {
        SMS_DATASET_IDS[0]: sms_allocation.train,
        SMS_DATASET_IDS[1]: sms_allocation.development,
        SMS_DATASET_IDS[2]: sms_allocation.calibration,
    }
    sms_by_row = {row.source_row_id: row for row in sms_rows}

    verify_source_file(SNLI_ARCHIVE_PATH, SNLI_SOURCE_ARCHIVE_SHA256, "SNLI archive")
    verify_source_file(SNLI_DEV_PATH, SNLI_DEV_SHA256, "SNLI development")
    snli_payload = SNLI_DEV_PATH.read_bytes()
    snli_items = parse_snli_items(snli_payload)
    snli_row_ids, snli_unlabeled_rows, _snli_row_count = parse_snli_native_row_ids(
        snli_payload,
        expected_rows=10000,
    )
    prior_snli_group_ids = _read_prior_snli_group_ids(PRIOR_RECIPE_PATH)
    snli_selected = select_snli_items(
        snli_items,
        snli_row_ids,
        set(prior_snli_group_ids),
        sample_size=DATASET_RECORD_COUNTS[SNLI_DATASET_ID],
    )

    selected: dict[str, Iterable[GroupedItem | SelectedSnliItem]] = {
        **dbpedia_selected,
        **sms_selected,
        SNLI_DATASET_ID: snli_selected,
    }
    records: list[DecisionRecord] = []
    for dataset_id, items in selected.items():
        if dataset_id in DBPEDIA_DATASET_IDS:
            records.extend(
                record_from_dbpedia_item(
                    item,
                    dbpedia_rows[item.source_row_id],
                    dataset_id=dataset_id,
                )
                for item in items
                if isinstance(item, GroupedItem)
            )
        elif dataset_id in SMS_DATASET_IDS:
            records.extend(
                record_from_sms_item(item, sms_by_row[item.source_row_id], dataset_id=dataset_id)
                for item in items
                if isinstance(item, GroupedItem)
            )
        else:
            records.extend(
                record_from_snli_item(item) for item in items if isinstance(item, SelectedSnliItem)
            )

    record_counts = Counter(record.dataset_id for record in records)
    if record_counts != Counter(DATASET_RECORD_COUNTS):
        raise ValueError("prepared records do not match the frozen dataset allocation")
    records_bytes = serialize_records(records)
    manifest_bytes = serialize_manifest(build_manifest())
    max_context_lengths = {
        dataset_id: max(
            len(record.request.context) for record in records if record.dataset_id == dataset_id
        )
        for dataset_id in DATASET_RECORD_COUNTS
    }
    recipe = build_recipe(
        selected,
        records_bytes,
        manifest_bytes,
        prior_snli_group_ids=prior_snli_group_ids,
        dbpedia_conflicting_group_count=dbpedia_grouping.conflicting_group_count,
        sms_conflicting_group_count=sms_grouping.conflicting_group_count,
        snli_unlabeled_row_count=snli_unlabeled_rows,
        max_request_context_characters=max_context_lengths,
    )
    recipe_bytes = serialize_recipe(recipe)
    _write_unchanged_or_new(PROJECT_ROOT / DEFAULT_RECORDS_PATH, records_bytes)
    _write_unchanged_or_new(PROJECT_ROOT / DEFAULT_MANIFEST_PATH, manifest_bytes)
    _write_unchanged_or_new(PROJECT_ROOT / DEFAULT_RECIPE_PATH, recipe_bytes)

    return {
        "record_count": len(records),
        "dataset_counts": dict(record_counts),
        "dbpedia_conflicting_group_count": dbpedia_grouping.conflicting_group_count,
        "sms_conflicting_group_count": sms_grouping.conflicting_group_count,
        "snli_unlabeled_row_count": snli_unlabeled_rows,
        "max_request_context_characters": max_context_lengths,
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "outputs": [DEFAULT_RECORDS_PATH, DEFAULT_MANIFEST_PATH, DEFAULT_RECIPE_PATH],
    }


def _read_dbpedia_representatives(row_ids: set[int]) -> dict[int, DbpediaRow]:
    selected = {}
    for row in read_dbpedia_rows(DBPEDIA_PATH, DBPEDIA_SOURCE_SHA256):
        if row.source_row_id in row_ids:
            selected[row.source_row_id] = row
    if set(selected) != row_ids:
        raise ValueError("a selected DBpedia native row is missing from its pinned source")
    return selected


def _read_prior_snli_group_ids(path: Path) -> tuple[str, ...]:
    verify_source_file(path, broader_data.EXPECTED_RECIPE_SHA256, "prior broader-data recipe")
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
        groups = recipe["selected"]["snli"]["source_group_ids"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("prior SNLI selection receipt is unavailable or malformed") from exc
    if not isinstance(groups, list) or len(groups) != 32:
        raise ValueError("prior SNLI recipe must identify exactly 32 source groups")
    if any(not isinstance(group_id, str) or len(group_id) != 64 for group_id in groups):
        raise ValueError("prior SNLI recipe contains an invalid group ID")
    return tuple(groups)


def _write_unchanged_or_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to replace a different prepared artifact: {path}")
        return
    with path.open("xb") as output:
        output.write(payload)


def main() -> None:
    summary = prepare_real_pilot()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
