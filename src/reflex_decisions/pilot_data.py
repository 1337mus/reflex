"""Deterministic source-grouped preparation and verification for the real pilot."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .data import (
    DatasetSpec,
    DecisionRecord,
    SplitManifest,
    SplitName,
    audit_splits,
    load_manifest,
    load_records,
)
from .schema import DecisionRequest, Option

DBPEDIA_SOURCE_ID = "fancyzhx/dbpedia_14"
SMS_SOURCE_ID = "uci-sms-spam-collection-228"
SNLI_SOURCE_ID = "stanford-snli-1.0"
DBPEDIA_DATASET_IDS = (
    "dbpedia14-pilot-v1-train",
    "dbpedia14-pilot-v1-development",
    "dbpedia14-pilot-v1-calibration",
)
SMS_DATASET_IDS = (
    "sms-pilot-v1-train",
    "sms-pilot-v1-development",
    "sms-pilot-v1-calibration",
)
SNLI_DATASET_ID = "snli-pilot-v1-development"
DBPEDIA_SOURCE_REVISION = "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
DBPEDIA_SOURCE_SHA256 = "0640e4664a99cc94c47db1d7b2e01c14455d5bbecb8183ad1f93bde59f3f28ee"
SMS_ARCHIVE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
SMS_TEXT_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
SNLI_DEV_SHA256 = "9c03faff70182ef086ebfeed2cffbabb5fcc6a84a8b3314decbbb5b01f07f4bf"
SNLI_SOURCE_ARCHIVE_SHA256 = "afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e"
DBPEDIA_SOURCE_URI = (
    f"https://huggingface.co/datasets/fancyzhx/dbpedia_14/tree/{DBPEDIA_SOURCE_REVISION}"
)
SMS_SOURCE_URI = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
SNLI_SOURCE_URI = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip#snli_1.0/snli_1.0_dev.jsonl"
DBPEDIA_LICENSE = "CC BY-SA 3.0 / GNU Free Documentation License (version unspecified)"
SMS_LICENSE = "CC BY 4.0 (UCI dataset page; legacy source notice retained)"
SNLI_LICENSE = "CC BY-SA 4.0"
DBPEDIA_TASK_FAMILY = "topic-classification"
SMS_TASK_FAMILY = "spam-detection"
DATASET_SPECS: dict[str, tuple[SplitName, str]] = {
    DBPEDIA_DATASET_IDS[0]: ("train", DBPEDIA_TASK_FAMILY),
    DBPEDIA_DATASET_IDS[1]: ("development", DBPEDIA_TASK_FAMILY),
    DBPEDIA_DATASET_IDS[2]: ("calibration", DBPEDIA_TASK_FAMILY),
    SMS_DATASET_IDS[0]: ("train", SMS_TASK_FAMILY),
    SMS_DATASET_IDS[1]: ("development", SMS_TASK_FAMILY),
    SMS_DATASET_IDS[2]: ("calibration", SMS_TASK_FAMILY),
    SNLI_DATASET_ID: ("development", "natural-language-inference-three-way"),
}
DEFAULT_RECORDS_PATH = "data/processed/real-pilot-v1.jsonl"
DEFAULT_MANIFEST_PATH = "data/pilots/real-pilot-v1-manifest.json"
DEFAULT_RECIPE_PATH = "data/pilots/real-pilot-v1-recipe.json"
EXPECTED_RECORDS_SHA256: str | None = (
    "8d803048df38b1120d2a1e91f404a480fba6097f6f300c0df9b1ec2512cff7ba"
)
EXPECTED_MANIFEST_SHA256: str | None = (
    "baf4c270f5eb6005fbb8ce8c08daa7e01a7762a128d0d2384532f8e3eb9a2cef"
)
EXPECTED_RECIPE_SHA256: str | None = (
    "2315f8c39efa95f2cc018e8d5a7281ba37f304f66161aa737a6ab9389bdded4f"
)
DBPEDIA_LABELS = (
    "Company",
    "Educational Institution",
    "Artist",
    "Athlete",
    "Office Holder",
    "Mean of Transportation",
    "Building",
    "Natural Place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written Work",
)
DBPEDIA_QUESTION = "Which DBpedia category best describes this entity?"
SMS_QUESTION = "Is this message ham or spam?"
SNLI_QUESTION = "Which relation best describes the hypothesis given the premise?"
SNLI_LABELS = ("entailment", "neutral", "contradiction")
DATASET_RECORD_COUNTS = {
    DBPEDIA_DATASET_IDS[0]: 252,
    DBPEDIA_DATASET_IDS[1]: 56,
    DBPEDIA_DATASET_IDS[2]: 56,
    SMS_DATASET_IDS[0]: 252,
    SMS_DATASET_IDS[1]: 60,
    SMS_DATASET_IDS[2]: 60,
    SNLI_DATASET_ID: 128,
}
SOURCE_RECEIPTS = {
    "dbpedia14": {
        "source_id": DBPEDIA_SOURCE_ID,
        "source_uri": DBPEDIA_SOURCE_URI,
        "source_revision": DBPEDIA_SOURCE_REVISION,
        "source_file": "data/raw/dbpedia14_train.parquet",
        "source_sha256": DBPEDIA_SOURCE_SHA256,
        "official_split": "train",
        "rows": 560000,
        "available_fields": ["title", "content", "label"],
        "license": DBPEDIA_LICENSE,
    },
    "sms": {
        "source_id": SMS_SOURCE_ID,
        "source_uri": SMS_SOURCE_URI,
        "source_revision": "UCI dataset 228 / DOI 10.24432/C5CC84; reviewed 2026-10-04",
        "source_archive": "data/raw/uci_sms_spam_collection.zip",
        "source_archive_sha256": SMS_ARCHIVE_SHA256,
        "source_file": "data/raw/uci_sms_spam_collection.txt",
        "source_sha256": SMS_TEXT_SHA256,
        "rows": 5574,
        "label_counts": {"ham": 4827, "spam": 747},
        "license": SMS_LICENSE,
    },
    "snli": {
        "source_id": SNLI_SOURCE_ID,
        "source_uri": SNLI_SOURCE_URI,
        "source_revision": "SNLI 1.0",
        "source_archive": "data/raw/snli_1.0.zip",
        "source_archive_sha256": SNLI_SOURCE_ARCHIVE_SHA256,
        "source_file": "data/raw/snli_1.0_dev.jsonl",
        "source_sha256": SNLI_DEV_SHA256,
        "official_split": "development",
        "rows": 10000,
        "license": SNLI_LICENSE,
    },
}
SELECTION_RECIPE = {
    "seed": 20261005,
    "group_rank": (
        "sha256('reflex-real-pilot-v1:20261005:{source_id}:{group_id}') ascending; "
        "group_id tie-break"
    ),
    "normalization": {
        "dbpedia14_sms": "Unicode NFKC, casefold, and collapsed whitespace",
        "snli": "existing parser normalization: casefold and collapsed whitespace without NFKC",
    },
    "grouping": {
        "dbpedia14": (
            "transitive connected components by normalized nonblank title OR content; "
            "exclude mixed-label components"
        ),
        "sms": "identical normalized full message; exclude mixed-label groups",
        "snli": (
            "existing parser components linked by captionID OR normalized premise; "
            "deduplicate normalized premise-hypothesis pairs and exclude conflicting labels"
        ),
    },
    "representative_order": {
        "dbpedia14_sms": "lowest native source row index",
        "snli": "lexicographically lowest native pairID within each selected group",
    },
    "allocation": {
        "dbpedia14": {
            "train_per_class": 18,
            "development_per_class": 4,
            "calibration_per_class": 4,
        },
        "sms": {
            "train_per_label": 126,
            "development_per_label": 30,
            "calibration_per_label": 30,
        },
        "snli": {"development_groups": 128, "label_balancing": False},
    },
    "record_id": "dataset ID plus opaque, label-independent item ID",
}


@dataclass(frozen=True, slots=True)
class DbpediaRow:
    source_row_id: int
    title: str
    content: str
    label: int


@dataclass(frozen=True, slots=True)
class SmsRow:
    source_row_id: int
    label: str
    message: str


@dataclass(frozen=True, slots=True)
class SnliCandidate(Protocol):
    source_item_id: str
    source_group_id: str
    premise: str
    hypothesis: str
    label: str


@dataclass(frozen=True, slots=True)
class GroupedItem:
    source_row_id: int
    item_id: str
    group_id: str
    label: str


@dataclass(frozen=True, slots=True)
class GroupingResult:
    items: tuple[GroupedItem, ...]
    conflicting_group_count: int


@dataclass(frozen=True, slots=True)
class PilotAllocation:
    train: tuple[GroupedItem, ...]
    development: tuple[GroupedItem, ...]
    calibration: tuple[GroupedItem, ...]


@dataclass(frozen=True, slots=True)
class SelectedSnliItem:
    source_row_id: int
    item_id: str
    group_id: str
    label: str
    premise: str
    hypothesis: str


def group_dbpedia_rows(rows: Iterable[DbpediaRow]) -> GroupingResult:
    """Group DBpedia rows on normalized title/content and drop mixed-label groups."""

    parents: list[int] = []
    component_labels: list[int] = []
    component_conflicts = bytearray()
    component_representatives: list[int] = []
    source_row_ids: list[int] = []
    seen_row_ids: set[int] = set()
    by_anchor: dict[tuple[str, str], int] = {}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            winner, child = min(root_left, root_right), max(root_left, root_right)
            if (
                component_labels[winner] != component_labels[child]
                or component_conflicts[winner]
                or component_conflicts[child]
            ):
                component_conflicts[winner] = 1
            if (
                source_row_ids[component_representatives[child]]
                < source_row_ids[component_representatives[winner]]
            ):
                component_representatives[winner] = component_representatives[child]
            parents[child] = winner

    for row in rows:
        row_id = row.source_row_id
        if type(row_id) is not int or row_id < 0:
            raise ValueError("DBpedia native row IDs must be nonnegative integers")
        if row_id in seen_row_ids:
            raise ValueError("DBpedia native row IDs must be unique")
        seen_row_ids.add(row_id)
        if type(row.label) is not int or row.label not in range(14):
            raise ValueError("DBpedia labels must be integers from 0 through 13")
        index = len(parents)
        parents.append(index)
        component_labels.append(row.label)
        component_conflicts.append(0)
        component_representatives.append(index)
        source_row_ids.append(row_id)
        title = _normalize_nfkc(row.title)
        content = _normalize_nfkc(row.content)
        row_anchors = {("content", content)} if content else set()
        if title:
            row_anchors.add(("title", title))
        if not row_anchors:
            raise ValueError("DBpedia row has no nonblank title or content anchor")
        for anchor in row_anchors:
            prior = by_anchor.get(anchor)
            if prior is None:
                by_anchor[anchor] = index
            else:
                union(index, prior)

    component_anchors: dict[int, list[tuple[str, str]]] = {}
    for anchor, index in by_anchor.items():
        component_anchors.setdefault(find(index), []).append(anchor)

    items: list[GroupedItem] = []
    conflicting_group_count = 0
    for root, parent in enumerate(parents):
        if root != parent:
            continue
        if component_conflicts[root]:
            conflicting_group_count += 1
            continue
        group_anchors = sorted(component_anchors.get(root, []))
        if not group_anchors:
            raise ValueError("DBpedia group has no nonblank title or content anchor")
        group_id = hashlib.sha256(
            json.dumps(group_anchors, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        representative = component_representatives[root]
        source_row_id = source_row_ids[representative]
        item_id = hashlib.sha256(
            f"reflex-real-pilot-item-v1:{DBPEDIA_SOURCE_ID}:{source_row_id}".encode()
        ).hexdigest()
        items.append(
            GroupedItem(
                source_row_id=source_row_id,
                item_id=item_id,
                group_id=group_id,
                label=str(component_labels[root]),
            )
        )
    return GroupingResult(
        items=tuple(sorted(items, key=lambda item: item.source_row_id)),
        conflicting_group_count=conflicting_group_count,
    )


def _normalize_nfkc(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("source text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def group_sms_rows(rows: Iterable[SmsRow]) -> GroupingResult:
    """Group identical normalized SMS messages and drop any mixed-label group."""

    source_rows = tuple(rows)
    row_ids = [row.source_row_id for row in source_rows]
    if any(type(row_id) is not int or row_id < 0 for row_id in row_ids):
        raise ValueError("SMS native row IDs must be nonnegative integers")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("SMS native row IDs must be unique")
    if any(row.label not in {"ham", "spam"} for row in source_rows):
        raise ValueError("SMS labels must be ham or spam")

    grouped: dict[str, list[SmsRow]] = {}
    for row in source_rows:
        normalized = _normalize_nfkc(row.message)
        if not normalized:
            raise ValueError("SMS messages must not be blank")
        grouped.setdefault(normalized, []).append(row)

    items: list[GroupedItem] = []
    conflicting_group_count = 0
    for normalized, members in grouped.items():
        labels = {row.label for row in members}
        if len(labels) != 1:
            conflicting_group_count += 1
            continue
        group_id = hashlib.sha256(
            json.dumps([("message", normalized)], ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        representative = min(members, key=lambda row: row.source_row_id)
        item_id = hashlib.sha256(
            f"reflex-real-pilot-item-v1:{SMS_SOURCE_ID}:{representative.source_row_id}".encode()
        ).hexdigest()
        items.append(
            GroupedItem(
                source_row_id=representative.source_row_id,
                item_id=item_id,
                group_id=group_id,
                label=representative.label,
            )
        )
    return GroupingResult(
        items=tuple(sorted(items, key=lambda item: item.source_row_id)),
        conflicting_group_count=conflicting_group_count,
    )


def stratified_allocate(
    items: Iterable[GroupedItem],
    targets: dict[str, tuple[int, int, int]],
    *,
    source_id: str,
) -> PilotAllocation:
    """Allocate class-specific train/dev/calibration quotas by the frozen rank key."""

    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source ID must be nonblank")
    if not targets:
        raise ValueError("allocation targets must contain at least one label")
    for label, quotas in targets.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("allocation labels must be nonblank strings")
        if (
            not isinstance(quotas, tuple)
            or len(quotas) != 3
            or any(type(count) is not int or count < 0 for count in quotas)
        ):
            raise ValueError("each label target must be three nonnegative integer quotas")

    source_items = tuple(items)
    group_ids = [item.group_id for item in source_items]
    if any(not isinstance(group_id, str) or not group_id for group_id in group_ids):
        raise ValueError("allocation group IDs must be nonblank strings")
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("allocation requires one representative per source group")
    unknown_labels = {item.label for item in source_items} - set(targets)
    if unknown_labels:
        raise ValueError("allocation items contain labels outside the target map")

    def label_order(label: str) -> tuple[int, int | str]:
        return (0, int(label)) if label.isdecimal() else (1, label)

    train: list[GroupedItem] = []
    development: list[GroupedItem] = []
    calibration: list[GroupedItem] = []
    destinations = (train, development, calibration)
    for label in sorted(targets, key=label_order):
        ranked = sorted(
            (item for item in source_items if item.label == label),
            key=lambda item: (
                hashlib.sha256(
                    f"reflex-real-pilot-v1:20261005:{source_id}:{item.group_id}".encode()
                ).hexdigest(),
                item.group_id,
            ),
        )
        quotas = targets[label]
        required = sum(quotas)
        if len(ranked) < required:
            raise ValueError(f"label {label!r} has fewer eligible groups than its allocation")
        offset = 0
        for destination, count in zip(destinations, quotas, strict=True):
            destination.extend(ranked[offset : offset + count])
            offset += count
    return PilotAllocation(tuple(train), tuple(development), tuple(calibration))


def select_snli_items(
    items: Iterable[SnliCandidate],
    source_row_ids_by_item: dict[str, int],
    excluded_group_ids: set[str] | frozenset[str],
    *,
    sample_size: int = 128,
) -> tuple[SelectedSnliItem, ...]:
    """Select SNLI groups by the frozen rank and minimum native pairID representative."""

    if type(sample_size) is not int or sample_size < 1:
        raise ValueError("SNLI sample size must be a positive integer")
    if any(not isinstance(group_id, str) or not group_id for group_id in excluded_group_ids):
        raise ValueError("excluded SNLI group IDs must be nonblank strings")
    by_group: dict[str, list[SnliCandidate]] = {}
    seen_items: set[str] = set()
    for item in items:
        if not isinstance(item.source_item_id, str) or not item.source_item_id.strip():
            raise ValueError("SNLI source item IDs must be nonblank strings")
        if item.source_item_id in seen_items:
            raise ValueError("SNLI parser output must have unique source item IDs")
        seen_items.add(item.source_item_id)
        if not isinstance(item.source_group_id, str) or not item.source_group_id:
            raise ValueError("SNLI parser output has a blank source group ID")
        by_group.setdefault(item.source_group_id, []).append(item)

    ranked_groups = sorted(
        set(by_group) - set(excluded_group_ids),
        key=lambda group_id: (
            hashlib.sha256(
                f"reflex-real-pilot-v1:20261005:{SNLI_SOURCE_ID}:{group_id}".encode()
            ).hexdigest(),
            group_id,
        ),
    )
    if len(ranked_groups) < sample_size:
        raise ValueError("SNLI has fewer eligible groups than the requested sample")

    selected: list[SelectedSnliItem] = []
    for group_id in ranked_groups[:sample_size]:
        representative = min(by_group[group_id], key=lambda item: item.source_item_id)
        try:
            source_row_id = source_row_ids_by_item[representative.source_item_id]
        except KeyError as exc:
            raise ValueError("SNLI representative has no native row ID") from exc
        if type(source_row_id) is not int or source_row_id < 0:
            raise ValueError("SNLI native row IDs must be nonnegative integers")
        item_id = hashlib.sha256(
            f"reflex-real-pilot-item-v1:{SNLI_SOURCE_ID}:{representative.source_item_id}".encode()
        ).hexdigest()
        selected.append(
            SelectedSnliItem(
                source_row_id=source_row_id,
                item_id=item_id,
                group_id=group_id,
                label=representative.label,
                premise=representative.premise,
                hypothesis=representative.hypothesis,
            )
        )
    return tuple(selected)


def build_manifest() -> SplitManifest:
    """Describe all seven pilot datasets with their canonical source identities."""

    datasets = []
    for dataset_id, source_id, source_uri, source_revision, license_name in (
        *(
            (
                dataset_id,
                DBPEDIA_SOURCE_ID,
                DBPEDIA_SOURCE_URI,
                DBPEDIA_SOURCE_REVISION,
                DBPEDIA_LICENSE,
            )
            for dataset_id in DBPEDIA_DATASET_IDS
        ),
        *(
            (
                dataset_id,
                SMS_SOURCE_ID,
                SMS_SOURCE_URI,
                "UCI dataset 228 / DOI 10.24432/C5CC84; reviewed 2026-10-04",
                SMS_LICENSE,
            )
            for dataset_id in SMS_DATASET_IDS
        ),
        (
            SNLI_DATASET_ID,
            SNLI_SOURCE_ID,
            SNLI_SOURCE_URI,
            "SNLI 1.0",
            SNLI_LICENSE,
        ),
    ):
        split, task_family = DATASET_SPECS[dataset_id]
        datasets.append(
            DatasetSpec(
                dataset_id=dataset_id,
                source_id=source_id,
                task_family=task_family,
                split=split,
                source_uri=source_uri,
                source_revision=source_revision,
                license=license_name,
            )
        )
    return SplitManifest(
        data_kind="benchmark",
        datasets=tuple(datasets),
        group_partitioned_sources=(DBPEDIA_SOURCE_ID, SMS_SOURCE_ID),
        held_out_families=(),
    )


def record_from_dbpedia_item(
    item: GroupedItem, row: DbpediaRow, *, dataset_id: str
) -> DecisionRecord:
    """Build a label-independent opaque record identity for one DBpedia representative."""

    if dataset_id not in DBPEDIA_DATASET_IDS:
        raise ValueError("DBpedia record dataset ID is outside the approved pilot")
    if item.source_row_id != row.source_row_id or item.label != str(row.label):
        raise ValueError("DBpedia item and representative row do not match")
    if not row.content.strip():
        raise ValueError("DBpedia content must not be blank")
    return DecisionRecord(
        record_id=f"{dataset_id}-{item.item_id}",
        dataset_id=dataset_id,
        source_group_id=item.group_id,
        request=DecisionRequest(
            context=f"Title: {row.title.strip()}\n\n{row.content.strip()}",
            question=DBPEDIA_QUESTION,
            options=tuple(
                Option(id=str(index), label=label) for index, label in enumerate(DBPEDIA_LABELS)
            ),
        ),
        answer_id=item.label,
    )


def record_from_sms_item(item: GroupedItem, row: SmsRow, *, dataset_id: str) -> DecisionRecord:
    """Build a label-independent opaque record identity for one SMS representative."""

    if dataset_id not in SMS_DATASET_IDS:
        raise ValueError("SMS record dataset ID is outside the approved pilot")
    if item.source_row_id != row.source_row_id or item.label != row.label:
        raise ValueError("SMS item and representative row do not match")
    if row.label not in {"ham", "spam"} or not row.message.strip():
        raise ValueError("SMS representative has an invalid label or blank message")
    return DecisionRecord(
        record_id=f"{dataset_id}-{item.item_id}",
        dataset_id=dataset_id,
        source_group_id=item.group_id,
        request=DecisionRequest(
            context=row.message.strip(),
            question=SMS_QUESTION,
            options=(Option(id="ham", label="Ham"), Option(id="spam", label="Spam")),
        ),
        answer_id=row.label,
    )


def record_from_snli_item(item: SelectedSnliItem) -> DecisionRecord:
    """Build a label-independent opaque record identity for one SNLI representative."""

    descriptions = {
        "entailment": "The premise guarantees the hypothesis.",
        "neutral": "The premise neither guarantees nor contradicts the hypothesis.",
        "contradiction": "The premise contradicts the hypothesis.",
    }
    if item.label not in SNLI_LABELS:
        raise ValueError("SNLI representative label is invalid")
    return DecisionRecord(
        record_id=f"{SNLI_DATASET_ID}-{item.item_id}",
        dataset_id=SNLI_DATASET_ID,
        source_group_id=item.group_id,
        request=DecisionRequest(
            context=f"Premise: {item.premise}\nHypothesis: {item.hypothesis}",
            question=SNLI_QUESTION,
            options=tuple(
                Option(id=label, label=label.capitalize(), description=descriptions[label])
                for label in SNLI_LABELS
            ),
        ),
        answer_id=item.label,
    )


def serialize_records(records: Iterable[DecisionRecord]) -> bytes:
    """Serialize records as deterministic UTF-8 JSONL with LF separators."""

    selected = tuple(records)
    if not selected:
        raise ValueError("prepared records must not be empty")
    if len({record.record_id for record in selected}) != len(selected):
        raise ValueError("prepared record IDs must be unique")
    lines = [
        json.dumps(
            record.model_dump(mode="json", exclude_none=True),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in selected
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def serialize_manifest(manifest: SplitManifest) -> bytes:
    """Serialize a manifest as deterministic, human-readable UTF-8 JSON."""

    payload = json.dumps(
        manifest.model_dump(mode="json", exclude_none=True),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
    )
    return f"{payload}\n".encode()


def build_recipe(
    selected: dict[str, Iterable[GroupedItem | SelectedSnliItem]],
    records_bytes: bytes,
    manifest_bytes: bytes,
    *,
    prior_snli_group_ids: Iterable[str],
    dbpedia_conflicting_group_count: int,
    sms_conflicting_group_count: int,
    snli_unlabeled_row_count: int,
    max_request_context_characters: dict[str, int],
) -> dict[str, object]:
    """Build the raw-text-free source and selection receipt."""

    dataset_ids = set(DATASET_SPECS)
    if set(selected) != dataset_ids or set(max_request_context_characters) != dataset_ids:
        raise ValueError("recipe selection and context-length maps must match the pilot datasets")
    selected_recipe: dict[str, dict[str, list[object]]] = {}
    for dataset_id in DATASET_SPECS:
        rows = tuple(selected[dataset_id])
        selected_recipe[dataset_id] = {
            "source_row_ids": [item.source_row_id for item in rows],
            "item_ids": [item.item_id for item in rows],
            "group_ids": [item.group_id for item in rows],
        }
    contexts = dict(max_request_context_characters)
    if any(type(length) is not int or length < 1 for length in contexts.values()):
        raise ValueError("maximum request context lengths must be positive integers")
    prior_groups = list(prior_snli_group_ids)
    if any(not isinstance(group_id, str) or not group_id for group_id in prior_groups):
        raise ValueError("prior SNLI group IDs must be nonblank strings")
    if len(prior_groups) != len(set(prior_groups)):
        raise ValueError("prior SNLI group IDs must be unique")
    for count in (
        dbpedia_conflicting_group_count,
        sms_conflicting_group_count,
        snli_unlabeled_row_count,
    ):
        if type(count) is not int or count < 0:
            raise ValueError("source exclusion counts must be nonnegative integers")

    return {
        "recipe_id": "reflex-real-pilot-v1",
        "sources": SOURCE_RECEIPTS,
        "selection": SELECTION_RECIPE,
        "selected": selected_recipe,
        "exclusions": {
            "dbpedia14_mixed_label_groups": dbpedia_conflicting_group_count,
            "sms_mixed_label_groups": sms_conflicting_group_count,
            "snli_unlabeled_rows": snli_unlabeled_row_count,
            "snli_prior_pilot_group_ids": prior_groups,
            "rejected_sources": {"financialphrasebank": "Rejected; do not include in this pilot."},
            "limitations": [
                "DBpedia custom splits are carved from the official train source split.",
                "SNLI is development evidence only and is not held out from all prior work.",
                "Exact duplicate grouping does not measure near-paraphrase or model-pretraining "
                "overlap.",
                "SMS upstream component lineage and message privacy have not been reviewed.",
                "Keep source-derived records and artifacts private pending separate rights review.",
            ],
        },
        "outputs": {
            "records_path": DEFAULT_RECORDS_PATH,
            "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
            "manifest_path": DEFAULT_MANIFEST_PATH,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "max_request_context_characters": contexts,
        },
    }


def serialize_recipe(recipe: dict[str, object]) -> bytes:
    """Serialize the recipe as deterministic, human-readable UTF-8 JSON."""

    payload = json.dumps(
        recipe,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    return f"{payload}\n".encode()


def verify_prepared_data(
    records_path: str | Path,
    manifest_path: str | Path,
    recipe_path: str | Path,
) -> tuple[SplitManifest, tuple[DecisionRecord, ...], dict[str, object]]:
    """Verify pinned records, manifest, recipe, and split lineage without source readers."""

    records_file = Path(records_path)
    manifest_file = Path(manifest_path)
    recipe_file = Path(recipe_path)
    records_bytes = _read_pinned_file(records_file, "records", EXPECTED_RECORDS_SHA256)
    manifest_bytes = _read_pinned_file(manifest_file, "manifest", EXPECTED_MANIFEST_SHA256)
    recipe_bytes = _read_pinned_file(recipe_file, "recipe", EXPECTED_RECIPE_SHA256)
    try:
        recipe_value = _strict_json_object(recipe_bytes, "recipe")
    except ValueError as exc:
        raise ValueError("prepared recipe is not strict JSON") from exc
    expected_top_keys = {"recipe_id", "sources", "selection", "selected", "exclusions", "outputs"}
    if set(recipe_value) != expected_top_keys:
        raise ValueError("prepared recipe fields do not match the frozen schema")
    if recipe_value["recipe_id"] != "reflex-real-pilot-v1":
        raise ValueError("prepared recipe ID does not match the frozen contract")
    if recipe_value["sources"] != SOURCE_RECEIPTS:
        raise ValueError("prepared recipe source pins do not match the frozen receipts")
    if recipe_value["selection"] != SELECTION_RECIPE:
        raise ValueError("prepared recipe selection does not match the frozen recipe")

    selected = _validate_recipe_selections(recipe_value["selected"])
    exclusions = _validate_recipe_exclusions(recipe_value["exclusions"], selected)
    outputs = recipe_value["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {
        "records_path",
        "records_sha256",
        "manifest_path",
        "manifest_sha256",
        "max_request_context_characters",
    }:
        raise ValueError("prepared recipe output fields do not match the frozen schema")
    if outputs["records_path"] != DEFAULT_RECORDS_PATH:
        raise ValueError("prepared recipe records path is not canonical")
    if outputs["manifest_path"] != DEFAULT_MANIFEST_PATH:
        raise ValueError("prepared recipe manifest path is not canonical")
    if outputs["records_sha256"] != hashlib.sha256(records_bytes).hexdigest():
        raise ValueError("prepared recipe records hash does not match the record file")
    if outputs["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("prepared recipe manifest hash does not match the manifest file")

    try:
        manifest = load_manifest(manifest_file)
        records = load_records(records_file)
    except ValueError as exc:
        raise ValueError("prepared data files do not satisfy the records/manifest schema") from exc
    if manifest != build_manifest():
        raise ValueError("prepared manifest does not match the frozen source contract")
    if Counter(record.dataset_id for record in records) != Counter(DATASET_RECORD_COUNTS):
        raise ValueError("prepared records do not match the exact dataset counts")
    if len(records) != sum(DATASET_RECORD_COUNTS.values()):
        raise ValueError("prepared records do not contain exactly 864 records")

    _validate_records_against_recipe(records, selected)
    _validate_label_balance(records)
    actual_context_maxima = {
        dataset_id: max(
            len(record.request.context) for record in records if record.dataset_id == dataset_id
        )
        for dataset_id in DATASET_RECORD_COUNTS
    }
    if outputs["max_request_context_characters"] != actual_context_maxima:
        raise ValueError("prepared recipe context lengths do not match the selected requests")
    audit = audit_splits(manifest, list(records))
    if audit.held_out_families or dict(audit.split_counts) != {
        "train": 504,
        "development": 244,
        "calibration": 116,
        "test": 0,
    }:
        raise ValueError("prepared records do not satisfy the frozen split audit")
    # Keep the validated exclusions referenced until all structural checks are complete.
    if not exclusions:
        raise ValueError("prepared recipe exclusions are incomplete")
    return manifest, records, recipe_value


def _read_pinned_file(path: Path, label: str, expected_sha256: str | None) -> bytes:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"expected {label} SHA-256 pin has not been frozen")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"prepared {label} file is unavailable") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"prepared {label} SHA-256 mismatch")
    return payload


def _strict_json_object(payload: bytes, label: str) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    def invalid_constant(_value: str) -> None:
        raise ValueError(f"{label} contains a non-finite JSON value")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _validate_recipe_selections(value: object) -> dict[str, dict[str, list[object]]]:
    if not isinstance(value, dict) or set(value) != set(DATASET_RECORD_COUNTS):
        raise ValueError("prepared recipe selections do not match the dataset allowlist")
    selected: dict[str, dict[str, list[object]]] = {}
    seen_items: set[str] = set()
    for dataset_id, count in DATASET_RECORD_COUNTS.items():
        entry = value[dataset_id]
        if not isinstance(entry, dict) or set(entry) != {"source_row_ids", "item_ids", "group_ids"}:
            raise ValueError("prepared recipe selection fields do not match the frozen schema")
        row_ids, item_ids, group_ids = (
            entry["source_row_ids"],
            entry["item_ids"],
            entry["group_ids"],
        )
        if not all(
            isinstance(values, list) and len(values) == count
            for values in (row_ids, item_ids, group_ids)
        ):
            raise ValueError("prepared recipe selection count does not match the frozen allocation")
        if any(type(row_id) is not int or row_id < 0 for row_id in row_ids):
            raise ValueError("prepared recipe source row IDs must be nonnegative integers")
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("prepared recipe source row IDs must be unique per dataset")
        for digest in (*item_ids, *group_ids):
            if not _is_sha256(digest):
                raise ValueError("prepared recipe item and group IDs must be SHA-256 digests")
        if len(item_ids) != len(set(item_ids)) or len(group_ids) != len(set(group_ids)):
            raise ValueError("prepared recipe item and group IDs must be unique per dataset")
        if seen_items.intersection(item_ids):
            raise ValueError("prepared recipe item IDs must be unique across datasets")
        seen_items.update(item_ids)
        selected[dataset_id] = {
            "source_row_ids": row_ids,
            "item_ids": item_ids,
            "group_ids": group_ids,
        }
    for dataset_ids in (DBPEDIA_DATASET_IDS, SMS_DATASET_IDS):
        row_ids = [
            row_id
            for dataset_id in dataset_ids
            for row_id in selected[dataset_id]["source_row_ids"]
        ]
        group_ids = [
            group_id for dataset_id in dataset_ids for group_id in selected[dataset_id]["group_ids"]
        ]
        if len(row_ids) != len(set(row_ids)) or len(group_ids) != len(set(group_ids)):
            raise ValueError("custom source rows and groups must be disjoint across splits")
    return selected


def _validate_recipe_exclusions(
    value: object, selected: dict[str, dict[str, list[object]]]
) -> bool:
    expected_keys = {
        "dbpedia14_mixed_label_groups",
        "sms_mixed_label_groups",
        "snli_unlabeled_rows",
        "snli_prior_pilot_group_ids",
        "rejected_sources",
        "limitations",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("prepared recipe exclusions do not match the frozen schema")
    if value["dbpedia14_mixed_label_groups"] != 12 or value["sms_mixed_label_groups"] != 0:
        raise ValueError(
            "prepared recipe conflicting-group exclusions do not match the source audit"
        )
    if type(value["snli_unlabeled_rows"]) is not int or value["snli_unlabeled_rows"] < 0:
        raise ValueError("prepared recipe SNLI unlabeled-row count is invalid")
    prior_groups = value["snli_prior_pilot_group_ids"]
    if not isinstance(prior_groups, list) or len(prior_groups) != 32:
        raise ValueError("prepared recipe must record all 32 prior SNLI groups")
    if any(not _is_sha256(group_id) for group_id in prior_groups) or len(set(prior_groups)) != 32:
        raise ValueError("prepared recipe prior SNLI group IDs are invalid")
    if set(prior_groups).intersection(selected[SNLI_DATASET_ID]["group_ids"]):
        raise ValueError("prepared SNLI groups overlap the previous pilot sample")
    if value["rejected_sources"] != {
        "financialphrasebank": "Rejected; do not include in this pilot."
    }:
        raise ValueError("prepared recipe must preserve the rejected-source decision")
    expected_limitations = [
        "DBpedia custom splits are carved from the official train source split.",
        "SNLI is development evidence only and is not held out from all prior work.",
        "Exact duplicate grouping does not measure near-paraphrase or model-pretraining overlap.",
        "SMS upstream component lineage and message privacy have not been reviewed.",
        "Keep source-derived records and artifacts private pending separate rights review.",
    ]
    if value["limitations"] != expected_limitations:
        raise ValueError("prepared recipe limitations do not match the frozen disclosure")
    return True


def _validate_records_against_recipe(
    records: tuple[DecisionRecord, ...], selected: dict[str, dict[str, list[object]]]
) -> None:
    position = 0
    for dataset_id, count in DATASET_RECORD_COUNTS.items():
        selection = selected[dataset_id]
        for index in range(count):
            record = records[position]
            position += 1
            item_id = selection["item_ids"][index]
            group_id = selection["group_ids"][index]
            if (
                record.dataset_id != dataset_id
                or record.record_id != f"{dataset_id}-{item_id}"
                or record.source_group_id != group_id
            ):
                raise ValueError("prepared record identity does not match its source selection")
            if dataset_id in DBPEDIA_DATASET_IDS:
                options = tuple(
                    Option(id=str(index), label=label) for index, label in enumerate(DBPEDIA_LABELS)
                )
                if record.request.question != DBPEDIA_QUESTION or record.request.options != options:
                    raise ValueError("DBpedia request schema does not match the frozen task")
            elif dataset_id in SMS_DATASET_IDS:
                options = (Option(id="ham", label="Ham"), Option(id="spam", label="Spam"))
                if record.request.question != SMS_QUESTION or record.request.options != options:
                    raise ValueError("SMS request schema does not match the frozen task")
            else:
                descriptions = {
                    "entailment": "The premise guarantees the hypothesis.",
                    "neutral": "The premise neither guarantees nor contradicts the hypothesis.",
                    "contradiction": "The premise contradicts the hypothesis.",
                }
                options = tuple(
                    Option(id=label, label=label.capitalize(), description=descriptions[label])
                    for label in SNLI_LABELS
                )
                if record.request.question != SNLI_QUESTION or record.request.options != options:
                    raise ValueError("SNLI request schema does not match the frozen task")


def _validate_label_balance(records: tuple[DecisionRecord, ...]) -> None:
    by_dataset: dict[str, Counter[str]] = {}
    for record in records:
        by_dataset.setdefault(record.dataset_id, Counter())[record.answer_id] += 1
    for dataset_id in DBPEDIA_DATASET_IDS:
        split = DATASET_SPECS[dataset_id][0]
        per_class = 18 if split == "train" else 4
        if by_dataset[dataset_id] != Counter({str(label): per_class for label in range(14)}):
            raise ValueError("DBpedia labels do not match the frozen per-class allocation")
    for dataset_id in SMS_DATASET_IDS:
        split = DATASET_SPECS[dataset_id][0]
        per_label = 126 if split == "train" else 30
        if by_dataset[dataset_id] != Counter({"ham": per_label, "spam": per_label}):
            raise ValueError("SMS labels do not match the frozen per-label allocation")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
