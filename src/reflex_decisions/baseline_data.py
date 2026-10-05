"""COPA development-split parsing and Reflex record construction."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .data import DecisionRecord, SplitManifest, audit_splits, load_manifest, load_records
from .schema import DecisionRequest, Option

DATASET_ID = "copa-dev-pilot-v1"
SOURCE_REVISION = "14496936a3bed7b386aa8f517656d6f91e3fa77a"
SOURCE_ARCHIVE_SHA256 = "5145348834d2081ad90da0397d1db3d70fa044e506bd8ce224194d24b04cdbbe"
SOURCE_DEV_XML_SHA256 = "6251d7d99de4cbc6a2c3323923213e5e73071d286039b97a0908736e2be093bc"
SOURCE_ID = "asgordon-copa-2011"
TASK_FAMILY = "commonsense-causal-choice"
SOURCE_ARCHIVE_URI = (
    "https://raw.githubusercontent.com/asgordon/asgordon.github.io/"
    f"{SOURCE_REVISION}/downloads/COPA-resources.tgz#COPA-resources/datasets/copa-dev.xml"
)
SOURCE_LICENSE = "BSD-2-Clause"
EXPECTED_RECORDS_SHA256 = "a8daefa5a7300cc6243f3af1208bbd2887d41c306416bb37bf95263748657c5b"
EXPECTED_MANIFEST_SHA256 = "4e60e82ea1622d4e069b4244aa596db64cafa03bdb9ec85d3c4d06d2c40e7635"


@dataclass(frozen=True)
class CopaItem:
    item_id: str
    relation: str
    premise: str
    choice1: str
    choice2: str
    answer: int


def validate_source_pins(archive_bytes: bytes, xml_bytes: bytes, *, source_revision: str) -> None:
    """Reject source files that differ from the approved immutable COPA revision."""

    if source_revision != SOURCE_REVISION:
        raise ValueError("source revision does not match the approved COPA pin")
    if hashlib.sha256(archive_bytes).hexdigest() != SOURCE_ARCHIVE_SHA256:
        raise ValueError("COPA archive SHA-256 mismatch")
    if hashlib.sha256(xml_bytes).hexdigest() != SOURCE_DEV_XML_SHA256:
        raise ValueError("COPA development XML SHA-256 mismatch")


def parse_copa_dev_xml(
    source: bytes, *, expected_item_count: int | None = 500
) -> tuple[CopaItem, ...]:
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise ValueError("malformed COPA development XML") from exc
    if root.tag != "copa-corpus" or root.attrib != {"version": "1.0"}:
        raise ValueError("unsupported COPA XML root or version")
    items: list[CopaItem] = []
    seen_ids: set[str] = set()
    seen_semantics: set[tuple[str, str, tuple[str, str]]] = set()
    for element in root:
        if element.tag != "item" or set(element.attrib) != {
            "id",
            "asks-for",
            "most-plausible-alternative",
        }:
            raise ValueError("invalid COPA item structure")
        item_id = element.attrib["id"]
        relation = element.attrib["asks-for"]
        label_text = element.attrib["most-plausible-alternative"]
        if not item_id or item_id in seen_ids:
            raise ValueError("COPA item IDs must be unique and nonblank")
        if relation not in {"cause", "effect"}:
            raise ValueError(f"unknown COPA relation: {relation}")
        if label_text not in {"1", "2"}:
            raise ValueError("COPA label must be 1 or 2")
        if len(element) != 3 or [child.tag for child in element] != ["p", "a1", "a2"]:
            raise ValueError("COPA item must contain premise, choice 1, and choice 2")
        if any(child.attrib for child in element):
            raise ValueError("COPA item text elements must not have attributes")
        if any(len(child) for child in element):
            raise ValueError("nested COPA text markup is not supported")
        children = {child.tag: (child.text or "").strip() for child in element}
        if any(not value for value in children.values()):
            raise ValueError("COPA item text must not be blank")
        normalized_choices = sorted(
            (" ".join(children["a1"].split()), " ".join(children["a2"].split()))
        )
        semantic_key = (
            relation,
            " ".join(children["p"].split()),
            (normalized_choices[0], normalized_choices[1]),
        )
        if semantic_key in seen_semantics:
            raise ValueError("normalized exact duplicate in COPA development split")
        seen_ids.add(item_id)
        seen_semantics.add(semantic_key)
        items.append(
            CopaItem(
                item_id=item_id,
                relation=relation,
                premise=children["p"],
                choice1=children["a1"],
                choice2=children["a2"],
                answer=int(label_text),
            )
        )
    if expected_item_count is not None and len(items) != expected_item_count:
        raise ValueError(f"expected {expected_item_count} COPA items, got {len(items)}")
    return tuple(items)


def record_from_item(item: CopaItem) -> DecisionRecord:
    questions = {
        "cause": "Which alternative is the more plausible cause?",
        "effect": "Which alternative is the more plausible effect?",
    }
    try:
        question = questions[item.relation]
    except KeyError as exc:
        raise ValueError(f"unknown COPA relation: {item.relation}") from exc
    return DecisionRecord(
        record_id=f"{DATASET_ID}-{item.item_id}",
        dataset_id=DATASET_ID,
        source_group_id=item.item_id,
        request=DecisionRequest(
            context=item.premise,
            question=question,
            options=(
                Option(id="choice1", label=item.choice1),
                Option(id="choice2", label=item.choice2),
            ),
        ),
        answer_id=f"choice{item.answer}",
    )


def select_pilot_items(
    items: tuple[CopaItem, ...] | list[CopaItem], *, sample_size: int = 32
) -> tuple[CopaItem, ...]:
    """Choose items by a label-independent, stable hash of their source IDs."""

    if sample_size < 1 or sample_size > len(items):
        raise ValueError("sample_size must be between 1 and the number of source items")
    ordered = sorted(
        items,
        key=lambda item: hashlib.sha256(f"reflex-copa-pilot-v1:{item.item_id}".encode()).digest(),
    )
    return tuple(ordered[:sample_size])


def build_pilot_records(source: bytes) -> tuple[DecisionRecord, ...]:
    """Build the 32 approved source records, with no generated permutations."""

    items = parse_copa_dev_xml(source)
    return tuple(record_from_item(item) for item in select_pilot_items(items))


def serialize_records(records: tuple[DecisionRecord, ...] | list[DecisionRecord]) -> bytes:
    """Serialize records as stable, UTF-8 JSONL for recipe hashing and loading."""

    lines = [
        json.dumps(
            record.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def verify_prepared_data(
    records_path: str | Path, manifest_path: str | Path
) -> tuple[SplitManifest, tuple[DecisionRecord, ...]]:
    """Verify the exact approved local recipe before a baseline run starts."""

    record_bytes = Path(records_path).read_bytes()
    manifest_bytes = Path(manifest_path).read_bytes()
    if hashlib.sha256(record_bytes).hexdigest() != EXPECTED_RECORDS_SHA256:
        raise ValueError("prepared COPA records SHA-256 mismatch")
    if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("COPA manifest SHA-256 mismatch")

    manifest = load_manifest(manifest_path)
    if manifest.data_kind != "benchmark" or len(manifest.datasets) != 1:
        raise ValueError("COPA manifest must declare exactly one benchmark dataset")
    dataset = manifest.datasets[0]
    if (
        dataset.dataset_id,
        dataset.source_id,
        dataset.task_family,
        dataset.split,
        dataset.source_uri,
        dataset.source_revision,
        dataset.license,
    ) != (
        DATASET_ID,
        SOURCE_ID,
        TASK_FAMILY,
        "development",
        SOURCE_ARCHIVE_URI,
        SOURCE_REVISION,
        SOURCE_LICENSE,
    ):
        raise ValueError("COPA manifest does not match the approved source recipe")
    if manifest.held_out_families:
        raise ValueError("COPA development data must not declare held-out families")

    records = load_records(records_path)
    if len(records) != 32 or len({record.source_group_id for record in records}) != 32:
        raise ValueError("prepared COPA data must contain 32 unique original items")
    for record in records:
        if record.dataset_id != DATASET_ID:
            raise ValueError("prepared COPA record references an unexpected dataset")
        if record.record_id != f"{DATASET_ID}-{record.source_group_id}":
            raise ValueError("prepared COPA record ID must preserve the source item ID")
        if [option.id for option in record.request.options] != ["choice1", "choice2"]:
            raise ValueError("prepared COPA record must preserve the original option order")
        if record.request.question not in {
            "Which alternative is the more plausible cause?",
            "Which alternative is the more plausible effect?",
        }:
            raise ValueError("prepared COPA record has an unexpected relation prompt")
    audit_splits(manifest, records)
    return manifest, records


def write_records_exclusive(path: str | Path, payload: bytes) -> None:
    """Create a processed data file without replacing an existing receipt."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(payload)
