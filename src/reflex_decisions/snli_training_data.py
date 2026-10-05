"""Private deterministic preparation of the SNLI training candidate."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from . import broader_data, snli_training_source
from .broader_data import SNLI_LABELS, SnliItem
from .data import DatasetSpec, DecisionRecord, SplitManifest, audit_splits
from .snli_training_source import ParsedSnliSource

TRAIN_DATASET_ID = "snli-training-v1"
TRAIN_SELECTION_SEED = "20261008"
TRAIN_RECORD_COUNT = 500
TRAIN_LABEL_QUOTAS = {"entailment": 167, "neutral": 167, "contradiction": 166}
TRAIN_MEMBER = "snli_1.0/snli_1.0_train.jsonl"
DEV_MEMBER = broader_data.SNLI_DEV_MEMBER
TEST_MEMBER = "snli_1.0/snli_1.0_test.jsonl"
SNLI_ARCHIVE_SHA256 = broader_data.SNLI_SOURCE_ARCHIVE_SHA256
SNLI_TRAIN_MEMBER_SHA256 = "ee95dfbc57800f7b1f62b7602ad2b176c2b983210435a49238e660324a01e963"
SNLI_DEV_MEMBER_SHA256 = broader_data.SNLI_DEV_SHA256
TRAIN_SOURCE_URI = (
    "https://nlp.stanford.edu/projects/snli/snli_1.0.zip#snli_1.0/snli_1.0_train.jsonl"
)
DEFAULT_ARCHIVE_PATH = Path("data/raw/snli_1.0.zip")
DEFAULT_OUTPUT_DIR = Path("data/processed/snli-training-v1")


@dataclass(frozen=True)
class CrossSplitExclusion:
    """Train candidates left after whole-component source-overlap exclusion."""

    remaining_items: tuple[SnliItem, ...]
    excluded_group_ids: tuple[str, ...]
    shared_premise_anchor_count: int
    shared_caption_anchor_count: int
    exact_normalized_pair_overlap_count: int
    excluded_candidate_pair_count: int


def _rank(seed: str, purpose: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{identity}".encode()).hexdigest()


def select_snli_training_items(
    items: Iterable[SnliItem],
) -> tuple[SnliItem, ...]:
    """Select the frozen class-balanced, one-record-per-group train candidate."""

    by_group: dict[str, list[SnliItem]] = defaultdict(list)
    for item in items:
        by_group[item.source_group_id].append(item)
    all_labels = set(SNLI_LABELS)
    eligible = [
        group_id
        for group_id, group_items in by_group.items()
        if {item.label for item in group_items} == all_labels
    ]
    if len(eligible) < TRAIN_RECORD_COUNT:
        raise ValueError("fewer than 500 eligible SNLI source groups")

    group_order = sorted(
        eligible,
        key=lambda group_id: (_rank(TRAIN_SELECTION_SEED, "group", group_id), group_id),
    )[:TRAIN_RECORD_COUNT]
    label_order = sorted(
        group_order,
        key=lambda group_id: (_rank(TRAIN_SELECTION_SEED, "label", group_id), group_id),
    )
    group_labels: dict[str, str] = {}
    offset = 0
    for label in SNLI_LABELS:
        quota = TRAIN_LABEL_QUOTAS[label]
        group_labels.update((group_id, label) for group_id in label_order[offset : offset + quota])
        offset += quota

    selected: list[SnliItem] = []
    for group_id in group_order:
        label = group_labels[group_id]
        selected.append(
            min(
                (item for item in by_group[group_id] if item.label == label),
                key=lambda item: (
                    _rank(TRAIN_SELECTION_SEED, "pair", item.source_item_id),
                    item.source_item_id,
                ),
            )
        )
    return tuple(selected)


def exclude_train_dev_overlaps(
    train: ParsedSnliSource, dev: ParsedSnliSource
) -> CrossSplitExclusion:
    """Exclude complete train components touched by any development source anchor."""

    dev_anchors = set().union(*dev.group_anchors.values()) if dev.group_anchors else set()
    train_anchors = set().union(*train.group_anchors.values()) if train.group_anchors else set()
    shared_anchors = train_anchors & dev_anchors
    excluded = tuple(
        sorted(
            group_id for group_id, anchors in train.group_anchors.items() if anchors & dev_anchors
        )
    )
    excluded_set = set(excluded)
    remaining = tuple(item for item in train.items if item.source_group_id not in excluded_set)
    exact_overlaps = train.pair_keys & dev.pair_keys
    if any(train.pair_group_ids[key] not in excluded_set for key in exact_overlaps):
        raise ValueError("exact SNLI pair overlap escaped source-anchor exclusion")
    return CrossSplitExclusion(
        remaining_items=remaining,
        excluded_group_ids=excluded,
        shared_premise_anchor_count=sum(kind == "premise" for kind, _ in shared_anchors),
        shared_caption_anchor_count=sum(kind == "caption" for kind, _ in shared_anchors),
        exact_normalized_pair_overlap_count=len(exact_overlaps),
        excluded_candidate_pair_count=sum(
            item.source_group_id in excluded_set for item in train.items
        ),
    )


def _sha256_file(path: Path) -> str:
    try:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as exc:
        raise ValueError(f"{path}: could not read SNLI archive") from exc


def _hashed_lines(source: Iterable[bytes], digest: Any, *, source_name: str) -> Iterable[str]:
    for line_number, raw_line in enumerate(source, start=1):
        digest.update(raw_line)
        try:
            yield raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{source_name} line {line_number} is not UTF-8") from exc


def read_snli_sources(
    archive_path: str | Path = DEFAULT_ARCHIVE_PATH,
) -> tuple[ParsedSnliSource, ParsedSnliSource, dict[str, str]]:
    """Verify and stream only the pinned SNLI train and development members."""

    archive_source = Path(archive_path)
    archive_sha256 = _sha256_file(archive_source)
    if archive_sha256 != SNLI_ARCHIVE_SHA256:
        raise ValueError("SNLI archive SHA-256 mismatch")
    train_digest, dev_digest = hashlib.sha256(), hashlib.sha256()
    try:
        with ZipFile(archive_source) as archive:
            names = [member.filename for member in archive.infolist()]
            if names.count(TRAIN_MEMBER) != 1 or names.count(DEV_MEMBER) != 1:
                raise ValueError("SNLI archive must contain exactly one train and one dev member")
            with archive.open(DEV_MEMBER) as dev_file:
                dev = snli_training_source._parse_snli_jsonl(
                    _hashed_lines(dev_file, dev_digest, source_name="SNLI dev"),
                    include_unlabeled_anchors=True,
                    conflict_policy="exclude",
                    source_name="SNLI dev",
                )
            with archive.open(TRAIN_MEMBER) as train_file:
                train = snli_training_source._parse_snli_jsonl(
                    _hashed_lines(train_file, train_digest, source_name="SNLI train"),
                    include_unlabeled_anchors=True,
                    conflict_policy="exclude",
                    source_name="SNLI train",
                )
    except BadZipFile as exc:
        raise ValueError("malformed SNLI archive") from exc
    except OSError as exc:
        raise ValueError("could not read SNLI archive members") from exc

    train_sha256, dev_sha256 = train_digest.hexdigest(), dev_digest.hexdigest()
    if dev_sha256 != SNLI_DEV_MEMBER_SHA256:
        raise ValueError("SNLI dev member SHA-256 mismatch")
    if train_sha256 != SNLI_TRAIN_MEMBER_SHA256:
        raise ValueError("SNLI train member SHA-256 mismatch")
    return (
        train,
        dev,
        {
            "archive_sha256": archive_sha256,
            "train_member_sha256": train_sha256,
            "development_member_sha256": dev_sha256,
        },
    )


@dataclass(frozen=True)
class SnliTrainingCandidate:
    """A private train-only candidate and its source-only reproducibility recipe."""

    selected_items: tuple[SnliItem, ...]
    records: tuple[DecisionRecord, ...]
    manifest: SplitManifest
    recipe: dict[str, Any]
    records_bytes: bytes
    manifest_bytes: bytes
    recipe_bytes: bytes


def _build_train_manifest() -> SplitManifest:
    return SplitManifest(
        data_kind="benchmark",
        datasets=(
            DatasetSpec(
                dataset_id=TRAIN_DATASET_ID,
                source_id=broader_data.SNLI_SOURCE_ID,
                task_family=broader_data.SNLI_TASK_FAMILY,
                split="train",
                source_uri=TRAIN_SOURCE_URI,
                source_revision=broader_data.SNLI_SOURCE_REVISION,
                license=broader_data.SNLI_LICENSE,
            ),
        ),
        held_out_families=(),
    )


def _serialize_recipe(recipe: dict[str, Any]) -> bytes:
    import json

    return (json.dumps(recipe, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def record_from_snli_training_item(item: SnliItem) -> DecisionRecord:
    """Recast the stable SNLI decision record onto the private train dataset ID."""

    dev_record = broader_data.record_from_snli_item(item)
    dev_record_prefix = f"{broader_data.SNLI_DATASET_ID}-"
    if not dev_record.record_id.startswith(dev_record_prefix):
        raise ValueError("SNLI decision record has an unexpected record ID prefix")
    record_data = dev_record.model_dump()
    record_data["record_id"] = (
        f"{TRAIN_DATASET_ID}-{dev_record.record_id[len(dev_record_prefix) :]}"
    )
    record_data["dataset_id"] = TRAIN_DATASET_ID
    return DecisionRecord.model_validate(record_data)


def build_snli_training_candidate_from_parsed(
    train: ParsedSnliSource,
    dev: ParsedSnliSource,
    source_hashes: Mapping[str, str],
) -> SnliTrainingCandidate:
    """Build the deterministic train records and source-only recipe from parsed inputs."""

    required_hashes = {
        "archive_sha256",
        "train_member_sha256",
        "development_member_sha256",
    }
    if set(source_hashes) != required_hashes or any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in source_hashes.values()
    ):
        raise ValueError(
            "SNLI source hashes must include the pinned archive, train, and dev SHA-256s"
        )

    exclusions = exclude_train_dev_overlaps(train, dev)
    by_group: dict[str, set[str]] = defaultdict(set)
    for item in exclusions.remaining_items:
        by_group[item.source_group_id].add(item.label)
    eligible_group_count = sum(set(SNLI_LABELS) == labels for labels in by_group.values())
    if eligible_group_count < TRAIN_RECORD_COUNT:
        raise ValueError("fewer than 500 eligible SNLI source groups after dev exclusions")

    selected_items = select_snli_training_items(exclusions.remaining_items)
    if len({item.source_group_id for item in selected_items}) != TRAIN_RECORD_COUNT:
        raise ValueError("SNLI selection did not produce one item per source group")
    selected_pair_keys = {
        (broader_data._normalize_text(item.premise), broader_data._normalize_text(item.hypothesis))
        for item in selected_items
    }
    if selected_pair_keys & dev.pair_keys:
        raise ValueError("selected SNLI candidate contains an exact development pair")

    records = tuple(record_from_snli_training_item(item) for item in selected_items)
    manifest = _build_train_manifest()
    audit = audit_splits(manifest, records)
    if audit.record_count != TRAIN_RECORD_COUNT or audit.held_out_families:
        raise ValueError("SNLI training records do not match the train-only manifest")
    records_bytes = broader_data.serialize_records(records)
    manifest_bytes = broader_data.serialize_manifest(manifest)
    selected_label_counts = {
        label: sum(item.label == label for item in selected_items) for label in SNLI_LABELS
    }
    group_label_counts = {
        label: sum(label in labels for labels in by_group.values()) for label in SNLI_LABELS
    }
    recipe = {
        "recipe_id": "reflex-snli-training-v1",
        "data_disclosure": (
            "SNLI training data makes future SNLI measurements within-source development; "
            "base-model pretraining overlap is unknown."
        ),
        "sources": {
            "dataset_id": TRAIN_DATASET_ID,
            "source_id": broader_data.SNLI_SOURCE_ID,
            "source_revision": broader_data.SNLI_SOURCE_REVISION,
            "archive_sha256": source_hashes["archive_sha256"],
            "train_member": TRAIN_MEMBER,
            "train_member_sha256": source_hashes["train_member_sha256"],
            "development_member": DEV_MEMBER,
            "development_member_sha256": source_hashes["development_member_sha256"],
            "license": broader_data.SNLI_LICENSE,
            "attribution": (
                "Bowman et al., SNLI (EMNLP 2015); premises derive mostly from Flickr30k "
                "captions with a smaller VisualGenome source."
            ),
        },
        "counts": {
            "train_source": asdict(train.stats),
            "development_source": asdict(dev.stats),
            "cross_split": {
                "shared_premise_anchor_count": exclusions.shared_premise_anchor_count,
                "shared_caption_anchor_count": exclusions.shared_caption_anchor_count,
                "excluded_group_count": len(exclusions.excluded_group_ids),
                "excluded_group_ids": list(exclusions.excluded_group_ids),
                "excluded_candidate_pair_count": exclusions.excluded_candidate_pair_count,
                "exact_normalized_pair_overlap_count": (
                    exclusions.exact_normalized_pair_overlap_count
                ),
            },
            "after_overlap": {
                "candidate_pair_count": len(exclusions.remaining_items),
                "source_group_count": len(train.group_anchors) - len(exclusions.excluded_group_ids),
                "candidate_group_count": len(by_group),
                "group_label_counts": group_label_counts,
                "eligible_group_count": eligible_group_count,
            },
            "selected": {
                "record_count": len(selected_items),
                "source_group_count": len({item.source_group_id for item in selected_items}),
                "label_counts": selected_label_counts,
            },
        },
        "selection": {
            "seed": TRAIN_SELECTION_SEED,
            "group_order": "sha256(seed + ':group:' + group_id), then group_id",
            "label_order": "sha256(seed + ':label:' + group_id), then group_id",
            "label_quotas": TRAIN_LABEL_QUOTAS,
            "pair_order": "sha256(seed + ':pair:' + pairID), then pairID",
            "emit_order": "selected groups in original group-rank order",
            "selected": [
                {
                    "source_group_id": item.source_group_id,
                    "pair_id": item.source_item_id,
                    "label": item.label,
                }
                for item in selected_items
            ],
        },
        "outputs": {
            "records_path": "records.jsonl",
            "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
            "manifest_path": "manifest.json",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
    }
    recipe_bytes = _serialize_recipe(recipe)
    return SnliTrainingCandidate(
        selected_items=selected_items,
        records=records,
        manifest=manifest,
        recipe=recipe,
        records_bytes=records_bytes,
        manifest_bytes=manifest_bytes,
        recipe_bytes=recipe_bytes,
    )


def build_snli_training_candidate(
    archive_path: str | Path = DEFAULT_ARCHIVE_PATH,
) -> SnliTrainingCandidate:
    """Read the pinned archive and create the private deterministic candidate."""

    train, dev, source_hashes = read_snli_sources(archive_path)
    return build_snli_training_candidate_from_parsed(train, dev, source_hashes)


def write_snli_training_candidate(
    candidate: SnliTrainingCandidate,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Write all candidate files into a new, exclusive output directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in (
        ("records.jsonl", candidate.records_bytes),
        ("manifest.json", candidate.manifest_bytes),
        ("recipe.json", candidate.recipe_bytes),
    ):
        with (destination / name).open("xb") as output:
            output.write(payload)
    return destination
