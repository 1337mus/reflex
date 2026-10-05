"""Bounded, source-grouped development pilots for BoolQ and SNLI."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from .data import (
    DatasetSpec,
    DecisionRecord,
    SplitManifest,
    audit_splits,
    load_manifest,
    load_records,
)
from .schema import DecisionRequest, Option

BOOLQ_DATASET_ID = "boolq-dev-pilot-v1"
SNLI_DATASET_ID = "snli-dev-pilot-v1"
SNLI_QUESTION = "Which relation best describes the hypothesis given the premise?"
SNLI_LABELS = ("entailment", "neutral", "contradiction")
BOOLQ_SOURCE_ID = "google-boolq"
BOOLQ_SOURCE_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
BOOLQ_SOURCE_SHA256 = "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8"
BOOLQ_DEV_SHA256 = "c81b6520f1a51a8b96de45420b3d1f78184aee00373c77852132b297f2e567ef"
BOOLQ_README_SHA256 = "07447668f46c7c1e62e24b650ea2a73f1ab34f2b3759b1587c7b7a81b4bf1cf4"
BOOLQ_SOURCE_URI = (
    "https://huggingface.co/datasets/google/boolq/resolve/"
    f"{BOOLQ_SOURCE_REVISION}/data/validation-00000-of-00001.parquet"
)
SNLI_SOURCE_ID = "stanford-snli-1.0"
SNLI_SOURCE_REVISION = "SNLI 1.0"
SNLI_SOURCE_ARCHIVE_SHA256 = "afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e"
SNLI_DEV_SHA256 = "9c03faff70182ef086ebfeed2cffbabb5fcc6a84a8b3314decbbb5b01f07f4bf"
SNLI_README_SHA256 = "a27e78bf4ba7bc26033fac15a6596dd41c7ff47b4909e739199c59f91cf1bd4e"
SNLI_LICENSE_PAGE_SHA256 = "9b19a0ac70d0362d90cf59742d803cb93f3c8149d480540e4d211c91b8fdea58"
SNLI_SOURCE_URI = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip#snli_1.0/snli_1.0_dev.jsonl"
SNLI_DEV_MEMBER = "snli_1.0/snli_1.0_dev.jsonl"
SNLI_README_MEMBER = "snli_1.0/README.txt"
BOOLQ_LICENSE = "CC BY-SA 3.0"
SNLI_LICENSE = "CC BY-SA 4.0"
BOOLQ_TASK_FAMILY = "yes-no-reading-comprehension"
SNLI_TASK_FAMILY = "natural-language-inference-three-way"
PROTOCOL_PATH = Path("docs/broader-baseline-protocol.md")
PROTOCOL_SHA256 = "f6043407231506a32b490321d5e64f65562604809eade71c8393a1d3380cbf86"
SOURCE_AMENDMENT_PATH = Path("docs/broader-baseline-source-amendment.md")
SOURCE_AMENDMENT_SHA256 = "74c38813b882b23b31c214bdc9e4f660dea56c54f8a2c445dd9ba2da97e0da59"
DEFAULT_RECIPE_PATH = Path("data/baselines/broader-dev-recipe.json")
EXPECTED_RECORDS_SHA256 = "5d09ac1c1df2b7c1849e675179cb57306040f473c0ebad60527f54d186ac6845"
EXPECTED_MANIFEST_SHA256 = "8dbddaec2ceac622ed998cd0c6051472c11bf38d776a895c08ff94ad5968b085"
EXPECTED_RECIPE_SHA256 = "6450653a445cc12b21304c267cdf0c8976009a01c6982fb07751b349cd81f68b"


@dataclass(frozen=True)
class BoolQItem:
    """One deduplicated BoolQ development question and its source group."""

    source_item_id: str
    source_group_id: str
    question: str
    passage: str
    title: str
    answer: bool


@dataclass(frozen=True)
class SnliItem:
    """One deduplicated SNLI development pair and its source group."""

    source_item_id: str
    source_group_id: str
    premise: str
    hypothesis: str
    label: str


def record_from_boolq_item(item: BoolQItem) -> DecisionRecord:
    """Map native BoolQ truth labels onto stable yes/no option IDs."""

    return DecisionRecord(
        record_id=f"{BOOLQ_DATASET_ID}-{item.source_item_id}",
        dataset_id=BOOLQ_DATASET_ID,
        source_group_id=item.source_group_id,
        request=DecisionRequest(
            context=item.passage,
            question=item.question,
            options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
        ),
        answer_id="yes" if item.answer else "no",
    )


def record_from_snli_item(item: SnliItem) -> DecisionRecord:
    """Recast SNLI labels as a fixed three-option relation decision."""

    descriptions = {
        "entailment": "The premise guarantees the hypothesis.",
        "neutral": "The premise neither guarantees nor contradicts the hypothesis.",
        "contradiction": "The premise contradicts the hypothesis.",
    }
    if item.label not in SNLI_LABELS:
        raise ValueError(f"unknown SNLI label: {item.label}")
    return DecisionRecord(
        record_id=f"{SNLI_DATASET_ID}-{_digest(item.source_item_id)}",
        dataset_id=SNLI_DATASET_ID,
        source_group_id=item.source_group_id,
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


def build_pilot_records(
    boolq_dev_jsonl: bytes, snli_dev_jsonl: bytes
) -> tuple[DecisionRecord, ...]:
    """Build 32 label-independent development records from each pinned split."""

    boolq_items = select_boolq_pilot_items(parse_boolq_dev_jsonl(boolq_dev_jsonl))
    snli_items = select_snli_pilot_items(parse_snli_dev_jsonl(snli_dev_jsonl))
    return tuple(
        [record_from_boolq_item(item) for item in boolq_items]
        + [record_from_snli_item(item) for item in snli_items]
    )


def build_manifest() -> SplitManifest:
    """Describe the two development-only sources without held-out claims."""

    return SplitManifest(
        data_kind="benchmark",
        datasets=(
            DatasetSpec(
                dataset_id=BOOLQ_DATASET_ID,
                source_id=BOOLQ_SOURCE_ID,
                task_family=BOOLQ_TASK_FAMILY,
                split="development",
                source_uri=BOOLQ_SOURCE_URI,
                source_revision=BOOLQ_SOURCE_REVISION,
                license=BOOLQ_LICENSE,
            ),
            DatasetSpec(
                dataset_id=SNLI_DATASET_ID,
                source_id=SNLI_SOURCE_ID,
                task_family=SNLI_TASK_FAMILY,
                split="development",
                source_uri=SNLI_SOURCE_URI,
                source_revision=SNLI_SOURCE_REVISION,
                license=SNLI_LICENSE,
            ),
        ),
        held_out_families=(),
    )


def serialize_records(records: tuple[DecisionRecord, ...] | list[DecisionRecord]) -> bytes:
    """Serialize records as deterministic UTF-8 JSONL."""

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


def serialize_manifest(manifest: SplitManifest | None = None) -> bytes:
    """Serialize the frozen manifest in a stable, reviewable JSON form."""

    selected = manifest or build_manifest()
    payload = json.dumps(
        selected.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        indent=2,
    )
    return f"{payload}\n".encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_snli_dev_jsonl(archive_bytes: bytes) -> bytes:
    """Read only the official development JSONL and README from the SNLI zip."""

    try:
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            names = [member.filename for member in archive.infolist()]
            if names.count(SNLI_DEV_MEMBER) != 1 or names.count(SNLI_README_MEMBER) != 1:
                raise ValueError("SNLI archive must contain one dev JSONL and one README")
            readme_bytes = archive.read(SNLI_README_MEMBER)
            dev_bytes = archive.read(SNLI_DEV_MEMBER)
    except BadZipFile as exc:
        raise ValueError("malformed SNLI source archive") from exc
    if not readme_bytes.strip():
        raise ValueError("SNLI source README is empty")
    return dev_bytes


def validate_source_pins(
    boolq_source_bytes: bytes,
    boolq_dev_jsonl: bytes,
    snli_archive_bytes: bytes,
    snli_dev_jsonl: bytes,
) -> None:
    """Validate downloaded source bytes and the exact SNLI development member."""

    if _sha256(boolq_source_bytes) != BOOLQ_SOURCE_SHA256:
        raise ValueError("BoolQ mirror source SHA-256 mismatch")
    if _sha256(boolq_dev_jsonl) != BOOLQ_DEV_SHA256:
        raise ValueError("BoolQ extracted development SHA-256 mismatch")
    if _sha256(snli_archive_bytes) != SNLI_SOURCE_ARCHIVE_SHA256:
        raise ValueError("SNLI archive SHA-256 mismatch")
    extracted_snli_dev = extract_snli_dev_jsonl(snli_archive_bytes)
    if _sha256(extracted_snli_dev) != SNLI_DEV_SHA256:
        raise ValueError("SNLI extracted development SHA-256 mismatch")
    if extracted_snli_dev != snli_dev_jsonl:
        raise ValueError("SNLI extracted development file differs from archive member")


def build_recipe(
    records: tuple[DecisionRecord, ...] | list[DecisionRecord],
    records_bytes: bytes,
    manifest_bytes: bytes,
    *,
    snli_source_item_ids: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the exact source receipt and sampling recipe for prepared records."""

    boolq_records = [record for record in records if record.dataset_id == BOOLQ_DATASET_ID]
    snli_records = [record for record in records if record.dataset_id == SNLI_DATASET_ID]
    if snli_source_item_ids is None:
        raise ValueError("the local receipt must preserve original SNLI pairIDs")
    if len(snli_source_item_ids) != len(snli_records):
        raise ValueError("SNLI pairID count must match selected SNLI records")
    return {
        "recipe_id": "reflex-broader-dev-pilot-v1",
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": PROTOCOL_SHA256},
        "source_amendment": {
            "path": str(SOURCE_AMENDMENT_PATH),
            "sha256": SOURCE_AMENDMENT_SHA256,
        },
        "sources": {
            "boolq": {
                "dataset_id": BOOLQ_DATASET_ID,
                "source_uri": BOOLQ_SOURCE_URI,
                "source_revision": BOOLQ_SOURCE_REVISION,
                "source_file": "data/raw/boolq-validation.parquet",
                "source_sha256": BOOLQ_SOURCE_SHA256,
                "source_size_bytes": 1257630,
                "development_split": "validation",
                "development_examples": 3270,
                "development_file": "data/raw/boolq-dev.jsonl",
                "development_sha256": BOOLQ_DEV_SHA256,
                "dataset_card_sha256": BOOLQ_README_SHA256,
                "available_fields": ["question", "answer", "passage"],
                "title_field_available": False,
                "export": {
                    "reader": "PyArrow",
                    "version": "25.0.1",
                    "encoding": "UTF-8",
                    "jsonl": "one row per LF; compact JSON; sorted keys; ensure_ascii=false",
                },
                "license": BOOLQ_LICENSE,
            },
            "snli": {
                "dataset_id": SNLI_DATASET_ID,
                "source_uri": SNLI_SOURCE_URI,
                "source_revision": SNLI_SOURCE_REVISION,
                "source_archive": "data/raw/snli_1.0.zip",
                "source_archive_sha256": SNLI_SOURCE_ARCHIVE_SHA256,
                "source_archive_size_bytes": 94550081,
                "development_member": SNLI_DEV_MEMBER,
                "development_file": "data/raw/snli_1.0_dev.jsonl",
                "development_sha256": SNLI_DEV_SHA256,
                "development_examples": 10000,
                "license": SNLI_LICENSE,
                "license_source": "https://nlp.stanford.edu/projects/snli/",
                "license_source_sha256": SNLI_LICENSE_PAGE_SHA256,
                "readme_member": SNLI_README_MEMBER,
                "readme_sha256": SNLI_README_SHA256,
            },
        },
        "selection": {
            "sample_size_per_dataset": 32,
            "normalization": "whitespace collapse after Unicode casefold",
            "jsonl_record_separator": "LF only",
            "deduplication": {
                "boolq": "normalized question and normalized passage",
                "snli": "normalized premise and normalized hypothesis",
                "duplicate_source_links": (
                    "all raw rows contribute grouping anchors before deduplication"
                ),
                "conflicting_duplicate_labels": "reject",
            },
            "grouping": {
                "boolq": "connected components by normalized passage OR nonblank normalized title",
                "snli": "connected components by captionID OR normalized premise",
                "group_id": "sha256 of compact JSON for sorted normalized component anchors",
            },
            "group_order": "sha256('reflex-broader-v1:{dataset}:{group_id}') ascending",
            "representative_order": (
                "sha256('reflex-broader-v1:{dataset}:{group_id}:representative:{source_item_id}') "
                "ascending"
            ),
            "boolq_source_item_id": (
                "sha256 of compact JSON [normalized question, normalized passage]"
            ),
            "snli_source_item_id": "original pairID",
            "answer_balancing": False,
            "snli_unlabeled_rows": "exclude gold_label '-'",
        },
        "selected": {
            "boolq": {
                "dataset_id": BOOLQ_DATASET_ID,
                "source_group_ids": [record.source_group_id for record in boolq_records],
                "source_item_ids": [
                    record.record_id.removeprefix(f"{BOOLQ_DATASET_ID}-")
                    for record in boolq_records
                ],
            },
            "snli": {
                "dataset_id": SNLI_DATASET_ID,
                "source_group_ids": [record.source_group_id for record in snli_records],
                "source_item_ids": list(snli_source_item_ids),
            },
        },
        "outputs": {
            "records_sha256": _sha256(records_bytes),
            "records_path": "data/processed/broader-dev-pilot-v1.jsonl",
            "manifest_sha256": _sha256(manifest_bytes),
            "manifest_path": "data/baselines/broader-dev-manifest.json",
        },
    }


def serialize_recipe(recipe: dict[str, Any]) -> bytes:
    payload = json.dumps(recipe, ensure_ascii=False, sort_keys=True, indent=2)
    return f"{payload}\n".encode()


def _read_pinned_bytes(path: str | Path, label: str) -> bytes:
    source = Path(path)
    try:
        return source.read_bytes()
    except OSError as exc:
        raise ValueError(f"{source}: could not read {label}") from exc


def _validate_protocol_documents() -> None:
    for path, expected, label in (
        (PROTOCOL_PATH, PROTOCOL_SHA256, "development protocol"),
        (SOURCE_AMENDMENT_PATH, SOURCE_AMENDMENT_SHA256, "BoolQ source amendment"),
    ):
        if _sha256(_read_pinned_bytes(path, label)) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")


def verify_prepared_data(
    records_path: str | Path,
    manifest_path: str | Path,
    recipe_path: str | Path = DEFAULT_RECIPE_PATH,
) -> tuple[SplitManifest, tuple[DecisionRecord, ...]]:
    """Verify pinned development records, source receipt, and manifest."""

    records_bytes = _read_pinned_bytes(records_path, "prepared records")
    manifest_bytes = _read_pinned_bytes(manifest_path, "manifest")
    recipe_bytes = _read_pinned_bytes(recipe_path, "recipe")
    if _sha256(records_bytes) != EXPECTED_RECORDS_SHA256:
        raise ValueError("prepared broader-data records SHA-256 mismatch")
    if _sha256(manifest_bytes) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("broader-data manifest SHA-256 mismatch")
    if _sha256(recipe_bytes) != EXPECTED_RECIPE_SHA256:
        raise ValueError("broader-data recipe SHA-256 mismatch")
    _validate_protocol_documents()

    manifest = load_manifest(manifest_path)
    if manifest != build_manifest():
        raise ValueError("broader-data manifest does not match the frozen development sources")
    if manifest.held_out_families:
        raise ValueError("development-only broader data must not claim held-out families")

    try:
        recipe = json.loads(recipe_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("broader-data recipe is malformed JSON") from exc
    if not isinstance(recipe, dict) or recipe.get("recipe_id") != "reflex-broader-dev-pilot-v1":
        raise ValueError("broader-data recipe has an unexpected identifier")
    if recipe.get("protocol") != {"path": str(PROTOCOL_PATH), "sha256": PROTOCOL_SHA256}:
        raise ValueError("broader-data recipe does not pin the frozen protocol")
    if recipe.get("source_amendment") != {
        "path": str(SOURCE_AMENDMENT_PATH),
        "sha256": SOURCE_AMENDMENT_SHA256,
    }:
        raise ValueError("broader-data recipe does not pin the BoolQ source amendment")
    outputs = recipe.get("outputs")
    if not isinstance(outputs, dict) or outputs != {
        "records_sha256": EXPECTED_RECORDS_SHA256,
        "records_path": "data/processed/broader-dev-pilot-v1.jsonl",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_path": "data/baselines/broader-dev-manifest.json",
    }:
        raise ValueError("broader-data recipe output hashes or paths mismatch")
    sources = recipe.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("broader-data recipe is missing source pins")
    boolq_source = sources.get("boolq")
    snli_source = sources.get("snli")
    if (
        not isinstance(boolq_source, dict)
        or boolq_source.get("source_sha256") != BOOLQ_SOURCE_SHA256
        or boolq_source.get("development_sha256") != BOOLQ_DEV_SHA256
        or boolq_source.get("source_revision") != BOOLQ_SOURCE_REVISION
        or boolq_source.get("development_examples") != 3270
        or boolq_source.get("title_field_available") is not False
    ):
        raise ValueError("broader-data recipe BoolQ source pin mismatch")
    if (
        not isinstance(snli_source, dict)
        or snli_source.get("source_archive_sha256") != SNLI_SOURCE_ARCHIVE_SHA256
        or snli_source.get("development_sha256") != SNLI_DEV_SHA256
        or snli_source.get("development_member") != SNLI_DEV_MEMBER
        or snli_source.get("development_examples") != 10000
    ):
        raise ValueError("broader-data recipe SNLI source pin mismatch")

    records = load_records(records_path)
    if len(records) != 64:
        raise ValueError("prepared broader data must contain exactly 64 records")
    selected = recipe.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("broader-data recipe is missing selected source IDs")
    all_expected: list[DecisionRecord] = []
    for key, dataset_id in (("boolq", BOOLQ_DATASET_ID), ("snli", SNLI_DATASET_ID)):
        dataset_selection = selected.get(key)
        if (
            not isinstance(dataset_selection, dict)
            or dataset_selection.get("dataset_id") != dataset_id
        ):
            raise ValueError(f"broader-data recipe {key} selection is malformed")
        group_ids = dataset_selection.get("source_group_ids")
        item_ids = dataset_selection.get("source_item_ids")
        if (
            not isinstance(group_ids, list)
            or not isinstance(item_ids, list)
            or len(group_ids) != 32
            or len(item_ids) != 32
            or len(set(group_ids)) != 32
            or len(set(item_ids)) != 32
        ):
            raise ValueError(f"prepared {key} data must name 32 distinct groups and source items")
        dataset_records = [record for record in records if record.dataset_id == dataset_id]
        if len(dataset_records) != 32:
            raise ValueError(f"prepared {key} data must contain 32 records")
        for index, (record, group_id, item_id) in enumerate(
            zip(dataset_records, group_ids, item_ids, strict=True)
        ):
            if (
                record.source_group_id != group_id
                or record.record_id
                != f"{dataset_id}-{item_id if key == 'boolq' else _digest(item_id)}"
            ):
                raise ValueError(f"prepared {key} record {index} lost its original source identity")
            if key == "boolq":
                if [option.id for option in record.request.options] != ["yes", "no"]:
                    raise ValueError("BoolQ must preserve yes/no option IDs and order")
                if [option.label for option in record.request.options] != ["Yes", "No"]:
                    raise ValueError("BoolQ option labels differ from the frozen schema")
                if [option.description for option in record.request.options] != [None, None]:
                    raise ValueError("BoolQ option descriptions differ from the frozen schema")
                if record.answer_id not in {"yes", "no"}:
                    raise ValueError("BoolQ answer must use the original yes/no IDs")
            else:
                if record.request.question != SNLI_QUESTION:
                    raise ValueError("SNLI request question differs from the frozen prompt")
                if [option.id for option in record.request.options] != list(SNLI_LABELS):
                    raise ValueError(
                        "SNLI must preserve entailment/neutral/contradiction option order"
                    )
                if [option.label for option in record.request.options] != [
                    "Entailment",
                    "Neutral",
                    "Contradiction",
                ]:
                    raise ValueError("SNLI labels differ from the frozen schema")
                expected_descriptions = [
                    "The premise guarantees the hypothesis.",
                    "The premise neither guarantees nor contradicts the hypothesis.",
                    "The premise contradicts the hypothesis.",
                ]
                if [
                    option.description for option in record.request.options
                ] != expected_descriptions:
                    raise ValueError("SNLI option descriptions differ from the frozen schema")
                if record.answer_id not in SNLI_LABELS:
                    raise ValueError("SNLI answer must use an original label ID")
                if (
                    not record.request.context.startswith("Premise: ")
                    or "\nHypothesis: " not in record.request.context
                ):
                    raise ValueError("SNLI context must preserve the premise and hypothesis")
            all_expected.append(record)
    if tuple(all_expected) != records:
        raise ValueError("prepared broader records must keep BoolQ before SNLI in recipe order")
    audit_splits(manifest, records)
    return manifest, records


def write_records_exclusive(path: str | Path, payload: bytes) -> None:
    """Create a processed data file without replacing an existing receipt."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(payload)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_boolq_dev_jsonl(source: bytes) -> tuple[BoolQItem, ...]:
    """Parse and deduplicate BoolQ dev, grouping linked passages and titles."""

    parsed: dict[tuple[str, str], tuple[str, str, str, bool]] = {}
    row_anchors: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for line_number, raw_line in enumerate(source.decode("utf-8").split("\n"), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"BoolQ dev line {line_number} is malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"BoolQ dev line {line_number} must be an object")
        question, passage = row.get("question"), row.get("passage")
        answer, title = row.get("answer"), row.get("title", "")
        if not isinstance(question, str) or not _normalize_text(question):
            raise ValueError(f"BoolQ dev line {line_number} has a blank question")
        if not isinstance(passage, str) or not _normalize_text(passage):
            raise ValueError(f"BoolQ dev line {line_number} has a blank passage")
        if not isinstance(answer, bool):
            raise ValueError(f"BoolQ dev line {line_number} answer must be boolean")
        if not isinstance(title, str):
            raise ValueError(f"BoolQ dev line {line_number} title must be text")
        question_norm = _normalize_text(question)
        passage_norm = _normalize_text(passage)
        semantic_key = (question_norm, passage_norm)
        title_norm = _normalize_text(title)
        row_anchors.setdefault(semantic_key, set()).add(("passage", passage_norm))
        if title_norm:
            row_anchors[semantic_key].add(("title", title_norm))
        previous = parsed.get(semantic_key)
        if previous is not None and previous[3] != answer:
            raise ValueError("normalized duplicate BoolQ question/passage has conflicting labels")
        candidate = (question.strip(), passage.strip(), title.strip(), answer)
        if previous is None or candidate[:3] < previous[:3]:
            parsed[semantic_key] = candidate

    keys = sorted(parsed)
    parents = list(range(len(keys)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[max(root_left, root_right)] = min(root_left, root_right)

    by_source: dict[tuple[str, str], int] = {}
    for index, (question_norm, passage_norm) in enumerate(keys):
        for anchor in row_anchors[(question_norm, passage_norm)]:
            previous_index = by_source.setdefault(anchor, index)
            union(index, previous_index)

    component_anchors: dict[int, set[tuple[str, str]]] = {}
    for index, (question_norm, passage_norm) in enumerate(keys):
        anchors = component_anchors.setdefault(find(index), set())
        anchors.update(row_anchors[(question_norm, passage_norm)])

    group_ids = {
        root: _digest(json.dumps(sorted(anchors), ensure_ascii=False, separators=(",", ":")))
        for root, anchors in component_anchors.items()
    }
    items: list[BoolQItem] = []
    for index, semantic_key in enumerate(keys):
        question_norm, passage_norm = semantic_key
        question, passage, title, answer = parsed[semantic_key]
        source_item_id = _digest(
            json.dumps(semantic_key, ensure_ascii=False, separators=(",", ":"))
        )
        items.append(
            BoolQItem(
                source_item_id=source_item_id,
                source_group_id=group_ids[find(index)],
                question=question,
                passage=passage,
                title=title,
                answer=answer,
            )
        )
    return tuple(items)


def select_boolq_pilot_items(
    items: tuple[BoolQItem, ...] | list[BoolQItem], *, sample_size: int = 32
) -> tuple[BoolQItem, ...]:
    """Choose one stable, label-independent representative per source group."""

    by_group: dict[str, list[BoolQItem]] = {}
    for item in items:
        by_group.setdefault(item.source_group_id, []).append(item)
    if sample_size < 1 or sample_size > len(by_group):
        raise ValueError("sample_size must be between 1 and the number of source groups")
    ordered_groups = sorted(
        by_group,
        key=lambda group_id: _digest(f"reflex-broader-v1:boolq:{group_id}"),
    )[:sample_size]
    return tuple(
        min(
            by_group[group_id],
            key=lambda item: _digest(
                f"reflex-broader-v1:boolq:{group_id}:representative:{item.source_item_id}"
            ),
        )
        for group_id in ordered_groups
    )


def parse_snli_dev_jsonl(source: bytes) -> tuple[SnliItem, ...]:
    """Parse labeled SNLI dev pairs and group shared caption/premise sources."""

    parsed: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    row_anchors: dict[tuple[str, str], set[tuple[str, str]]] = {}
    allowed_labels = {"entailment", "neutral", "contradiction"}
    for line_number, raw_line in enumerate(source.decode("utf-8").split("\n"), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SNLI dev line {line_number} is malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"SNLI dev line {line_number} must be an object")
        label = row.get("gold_label")
        if label == "-":
            continue
        if label not in allowed_labels:
            raise ValueError(f"SNLI dev line {line_number} has an unknown gold label")
        pair_id = row.get("pairID")
        premise, hypothesis = row.get("sentence1"), row.get("sentence2")
        caption_id = row.get("captionID", "")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"SNLI dev line {line_number} has a blank pairID")
        if not isinstance(premise, str) or not _normalize_text(premise):
            raise ValueError(f"SNLI dev line {line_number} has a blank premise")
        if not isinstance(hypothesis, str) or not _normalize_text(hypothesis):
            raise ValueError(f"SNLI dev line {line_number} has a blank hypothesis")
        if not isinstance(caption_id, str):
            raise ValueError(f"SNLI dev line {line_number} captionID must be text")
        premise_norm, hypothesis_norm = _normalize_text(premise), _normalize_text(hypothesis)
        semantic_key = (premise_norm, hypothesis_norm)
        caption_norm = _normalize_text(caption_id)
        row_anchors.setdefault(semantic_key, set()).add(("premise", premise_norm))
        if caption_norm:
            row_anchors[semantic_key].add(("caption", caption_norm))
        previous = parsed.get(semantic_key)
        if previous is not None and previous[4] != label:
            raise ValueError("normalized duplicate SNLI premise/hypothesis has conflicting labels")
        candidate = (
            pair_id.strip(),
            premise.strip(),
            hypothesis.strip(),
            caption_id.strip(),
            label,
        )
        if previous is None or candidate[:4] < previous[:4]:
            parsed[semantic_key] = candidate

    keys = sorted(parsed)
    parents = list(range(len(keys)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[max(root_left, root_right)] = min(root_left, root_right)

    by_source: dict[tuple[str, str], int] = {}
    for index, _ in enumerate(keys):
        for anchor in row_anchors[keys[index]]:
            previous_index = by_source.setdefault(anchor, index)
            union(index, previous_index)

    component_anchors: dict[int, set[tuple[str, str]]] = {}
    for index, _ in enumerate(keys):
        anchors = component_anchors.setdefault(find(index), set())
        anchors.update(row_anchors[keys[index]])

    group_ids = {
        root: _digest(json.dumps(sorted(anchors), ensure_ascii=False, separators=(",", ":")))
        for root, anchors in component_anchors.items()
    }
    items: list[SnliItem] = []
    for index, semantic_key in enumerate(keys):
        pair_id, premise, hypothesis, _, label = parsed[semantic_key]
        items.append(
            SnliItem(
                source_item_id=pair_id,
                source_group_id=group_ids[find(index)],
                premise=premise,
                hypothesis=hypothesis,
                label=label,
            )
        )
    return tuple(items)


def select_snli_pilot_items(
    items: tuple[SnliItem, ...] | list[SnliItem], *, sample_size: int = 32
) -> tuple[SnliItem, ...]:
    """Choose one stable, label-independent representative per source group."""

    by_group: dict[str, list[SnliItem]] = {}
    for item in items:
        by_group.setdefault(item.source_group_id, []).append(item)
    if sample_size < 1 or sample_size > len(by_group):
        raise ValueError("sample_size must be between 1 and the number of source groups")
    ordered_groups = sorted(
        by_group,
        key=lambda group_id: _digest(f"reflex-broader-v1:snli:{group_id}"),
    )[:sample_size]
    return tuple(
        min(
            by_group[group_id],
            key=lambda item: _digest(
                f"reflex-broader-v1:snli:{group_id}:representative:{item.source_item_id}"
            ),
        )
        for group_id in ordered_groups
    )
