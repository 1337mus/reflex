"""Deterministic, source-group-balanced SNLI development diagnostic data."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .broader_data import (
    SNLI_DEV_MEMBER,
    SNLI_DEV_SHA256,
    SNLI_LABELS,
    SNLI_LICENSE,
    SNLI_LICENSE_PAGE_SHA256,
    SNLI_SOURCE_ARCHIVE_SHA256,
    SNLI_SOURCE_ID,
    SNLI_SOURCE_REVISION,
    SNLI_SOURCE_URI,
    SnliItem,
    record_from_snli_item,
    serialize_manifest,
    serialize_records,
)
from .data import DatasetSpec, DecisionRecord, SplitManifest, audit_splits
from .pilot_data_sources import (
    SNLI_ROWS,
    parse_snli_items,
    parse_snli_native_row_ids,
)

SELECTION_SEED = "reflex-snli-balanced-v1:20261006"
BROADER_RECIPE_SHA256 = "6450653a445cc12b21304c267cdf0c8976009a01c6982fb07751b349cd81f68b"
REAL_PILOT_RECIPE_SHA256 = "2315f8c39efa95f2cc018e8d5a7281ba37f304f66161aa737a6ab9389bdded4f"


@dataclass(frozen=True)
class PriorRecipeSpec:
    path: str
    sha256: str
    selection_key: str
    group_key: str
    group_count: int


@dataclass(frozen=True)
class PriorRecipeReceipt:
    path: str
    sha256: str
    group_ids: tuple[str, ...]


PRIOR_RECIPE_SPECS = (
    PriorRecipeSpec(
        "data/baselines/broader-dev-recipe.json",
        BROADER_RECIPE_SHA256,
        "snli",
        "source_group_ids",
        32,
    ),
    PriorRecipeSpec(
        "data/pilots/real-pilot-v1-recipe.json",
        REAL_PILOT_RECIPE_SHA256,
        "snli-pilot-v1-development",
        "group_ids",
        128,
    ),
)
DATASET_ID = "snli-balanced-v1-development"
TASK_FAMILY = "natural-language-inference-three-way"
DEFAULT_OUTPUT_DIR = Path("data/processed/snli-balanced-v1")
EXPECTED_SOURCE_GROUPS = 3319
EXPECTED_ELIGIBLE_GROUPS = 2488
SELECTED_GROUP_COUNT = 64


@dataclass(frozen=True)
class BalancedDataset:
    manifest: SplitManifest
    records: tuple[DecisionRecord, ...]
    items: tuple[SnliItem, ...]


@dataclass(frozen=True)
class CandidateArtifacts:
    bundle: BalancedDataset
    records_bytes: bytes
    manifest_bytes: bytes
    recipe_bytes: bytes


def _rank(kind: str, value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{kind}:{value}".encode()).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_prior_exclusions(
    repo_root: str | Path = ".",
) -> tuple[tuple[PriorRecipeReceipt, ...], frozenset[str]]:
    """Read and validate the exact prior panels whose source groups must be excluded."""

    root = Path(repo_root)
    receipts: list[PriorRecipeReceipt] = []
    occupied: set[str] = set()
    for spec in PRIOR_RECIPE_SPECS:
        path = root / spec.path
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"prior recipe is unavailable: {spec.path}") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != spec.sha256:
            raise ValueError(f"prior recipe SHA-256 mismatch: {spec.path}")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"prior recipe is malformed: {spec.path}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"prior recipe has an invalid shape: {spec.path}")
        selected = document.get("selected")
        selection = selected.get(spec.selection_key) if isinstance(selected, dict) else None
        raw_groups = selection.get(spec.group_key) if isinstance(selection, dict) else None
        if not isinstance(raw_groups, list) or len(raw_groups) != spec.group_count:
            raise ValueError(f"prior recipe group count mismatch: {spec.path}")
        if any(not _is_sha256(group_id) for group_id in raw_groups):
            raise ValueError(f"prior recipe contains an invalid source group hash: {spec.path}")
        if len(set(raw_groups)) != len(raw_groups):
            raise ValueError(f"prior recipe contains duplicate source groups: {spec.path}")
        if occupied.intersection(raw_groups):
            raise ValueError("prior development panels are not source-group disjoint")
        occupied.update(raw_groups)
        receipts.append(PriorRecipeReceipt(spec.path, digest, tuple(raw_groups)))
    if len(occupied) != 160:
        raise ValueError("the union of prior panels must contain exactly 160 source groups")
    return tuple(receipts), frozenset(occupied)


def select_balanced_items(
    items: Iterable[SnliItem],
    *,
    excluded_group_ids: frozenset[str],
    group_count: int = 64,
) -> tuple[SnliItem, ...]:
    """Select one hash-ranked pair per label from each complete source group."""

    if isinstance(group_count, bool) or not isinstance(group_count, int) or group_count < 1:
        raise ValueError("group_count must be a positive integer")

    groups: dict[str, list[SnliItem]] = defaultdict(list)
    seen_item_ids: set[str] = set()
    for item in items:
        if item.source_item_id in seen_item_ids:
            raise ValueError("SNLI pairIDs must be unique")
        seen_item_ids.add(item.source_item_id)
        groups[item.source_group_id].append(item)

    eligible = [
        group_id
        for group_id, members in groups.items()
        if group_id not in excluded_group_ids
        and {member.label for member in members} >= set(SNLI_LABELS)
    ]
    ranked_groups = sorted(eligible, key=lambda group_id: (_rank("group", group_id), group_id))
    if len(ranked_groups) < group_count:
        raise ValueError(f"only {len(ranked_groups)} eligible SNLI groups; {group_count} required")

    selected: list[SnliItem] = []
    for group_id in ranked_groups[:group_count]:
        members = groups[group_id]
        for label in SNLI_LABELS:
            candidates = [item for item in members if item.label == label]
            selected.append(
                min(
                    candidates,
                    key=lambda item: (
                        _rank("pair", item.source_item_id),
                        item.source_item_id,
                    ),
                )
            )
    return tuple(selected)


def build_balanced_dataset(
    items: Iterable[SnliItem],
    *,
    excluded_group_ids: frozenset[str],
    group_count: int = 64,
) -> BalancedDataset:
    """Build validated development records from the deterministic source-group sample."""

    selected_items = select_balanced_items(
        items, excluded_group_ids=excluded_group_ids, group_count=group_count
    )
    by_group: dict[str, list[str]] = defaultdict(list)
    for item in selected_items:
        by_group[item.source_group_id].append(item.label)
    if len(by_group) != group_count or any(
        Counter(labels) != Counter({label: 1 for label in SNLI_LABELS})
        for labels in by_group.values()
    ):
        raise ValueError("selected SNLI panel must contain one pair per label in every group")
    if Counter(item.label for item in selected_items) != Counter(
        {label: group_count for label in SNLI_LABELS}
    ):
        raise ValueError("selected SNLI panel must have equal class counts")

    manifest = SplitManifest(
        data_kind="benchmark",
        datasets=(
            DatasetSpec(
                dataset_id=DATASET_ID,
                source_id=SNLI_SOURCE_ID,
                task_family=TASK_FAMILY,
                split="development",
                source_uri=SNLI_SOURCE_URI,
                source_revision=SNLI_SOURCE_REVISION,
                license=SNLI_LICENSE,
            ),
        ),
        held_out_families=(),
    )
    records: list[DecisionRecord] = []
    for item in selected_items:
        original = record_from_snli_item(item)
        raw_record = original.model_dump(mode="json")
        digest = hashlib.sha256(item.source_item_id.encode()).hexdigest()
        raw_record["record_id"] = f"{DATASET_ID}-{digest}"
        raw_record["dataset_id"] = DATASET_ID
        records.append(DecisionRecord.model_validate(raw_record))
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("balanced SNLI record IDs must be unique")
    if len({record.request.request_hash for record in records}) != len(records):
        raise ValueError("balanced SNLI request hashes must be unique")
    audit_splits(manifest, records)
    return BalancedDataset(manifest, tuple(records), selected_items)


def parse_and_build_balanced_dataset(
    payload: bytes,
    *,
    excluded_group_ids: frozenset[str],
    group_count: int = 64,
) -> BalancedDataset:
    return build_balanced_dataset(
        parse_snli_items(payload),
        excluded_group_ids=excluded_group_ids,
        group_count=group_count,
    )


def read_verified_snli_development(repo_root: str | Path) -> tuple[bytes, str, str]:
    """Verify the pinned archive and dev member without reading train/test members."""

    root = Path(repo_root)
    archive_path = root / "data/raw/snli_1.0.zip"
    dev_path = root / "data/raw/snli_1.0_dev.jsonl"
    try:
        archive_bytes = archive_path.read_bytes()
        dev_bytes = dev_path.read_bytes()
    except OSError as exc:
        raise ValueError("pinned SNLI archive or development file is unavailable") from exc
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    dev_sha256 = hashlib.sha256(dev_bytes).hexdigest()
    if archive_sha256 != SNLI_SOURCE_ARCHIVE_SHA256:
        raise ValueError("SNLI source archive SHA-256 mismatch")
    if dev_sha256 != SNLI_DEV_SHA256:
        raise ValueError("SNLI development file SHA-256 mismatch")
    try:
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            if archive.namelist().count(SNLI_DEV_MEMBER) != 1:
                raise ValueError("SNLI archive must contain exactly one development member")
            archived_dev = archive.read(SNLI_DEV_MEMBER)
    except BadZipFile as exc:
        raise ValueError("pinned SNLI source archive is malformed") from exc
    if archived_dev != dev_bytes:
        raise ValueError("SNLI development file differs from its pinned archive member")
    return dev_bytes, archive_sha256, dev_sha256


def build_candidate(repo_root: str | Path = ".") -> CandidateArtifacts:
    """Prepare stable records, manifest, and a source-only recipe from pinned SNLI dev."""

    root = Path(repo_root)
    dev_bytes, archive_sha256, dev_sha256 = read_verified_snli_development(root)
    items = parse_snli_items(dev_bytes)
    source_groups = {item.source_group_id for item in items}
    if len(source_groups) != EXPECTED_SOURCE_GROUPS:
        raise ValueError("parsed SNLI source-group count differs from the pinned audit")
    prior_recipes, excluded_groups = load_prior_exclusions(root)
    missing_groups = excluded_groups - source_groups
    if missing_groups:
        raise ValueError("a pinned prior-panel source group is absent from SNLI development data")
    grouped: dict[str, list[SnliItem]] = defaultdict(list)
    for item in items:
        grouped[item.source_group_id].append(item)
    eligible_groups = {
        group_id
        for group_id, members in grouped.items()
        if group_id not in excluded_groups
        and {member.label for member in members} >= set(SNLI_LABELS)
    }
    if len(eligible_groups) != EXPECTED_ELIGIBLE_GROUPS:
        raise ValueError("eligible SNLI source-group count differs from the pinned audit")
    row_ids, unlabeled_count, native_row_count = parse_snli_native_row_ids(
        dev_bytes, expected_rows=SNLI_ROWS
    )
    bundle = build_balanced_dataset(
        items,
        excluded_group_ids=excluded_groups,
        group_count=SELECTED_GROUP_COUNT,
    )
    if any(item.source_item_id not in row_ids for item in bundle.items):
        raise ValueError("selected SNLI pairID has no native development row reference")

    records_bytes = serialize_records(bundle.records)
    manifest_bytes = serialize_manifest(bundle.manifest)
    recipe = build_recipe(
        bundle,
        records_bytes=records_bytes,
        manifest_bytes=manifest_bytes,
        archive_sha256=archive_sha256,
        dev_sha256=dev_sha256,
        prior_recipes=prior_recipes,
        excluded_group_ids=excluded_groups,
        source_group_count=len(source_groups),
        eligible_group_count=len(eligible_groups),
        source_row_ids=row_ids,
        unlabeled_row_count=unlabeled_count,
        native_row_count=native_row_count,
    )
    recipe_bytes = serialize_recipe(recipe)
    return CandidateArtifacts(bundle, records_bytes, manifest_bytes, recipe_bytes)


def build_recipe(
    bundle: BalancedDataset,
    *,
    records_bytes: bytes,
    manifest_bytes: bytes,
    archive_sha256: str,
    dev_sha256: str,
    prior_recipes: tuple[PriorRecipeReceipt, ...],
    excluded_group_ids: frozenset[str],
    source_group_count: int,
    eligible_group_count: int,
    source_row_ids: dict[str, int],
    unlabeled_row_count: int,
    native_row_count: int,
) -> dict[str, object]:
    """Build a source-only, deterministic receipt for the balanced candidate."""

    selected_groups: list[dict[str, object]] = []
    for group_id in dict.fromkeys(item.source_group_id for item in bundle.items):
        members = [item for item in bundle.items if item.source_group_id == group_id]
        selected_groups.append(
            {
                "source_group_id": group_id,
                "members": [
                    {
                        "pair_id": item.source_item_id,
                        "label": item.label,
                        "source_row_id": source_row_ids[item.source_item_id],
                    }
                    for label in SNLI_LABELS
                    for item in members
                    if item.label == label
                ],
            }
        )
    return {
        "recipe_id": "reflex-snli-balanced-v1",
        "data_kind": "benchmark",
        "dataset_id": DATASET_ID,
        "source": {
            "source_id": SNLI_SOURCE_ID,
            "source_uri": SNLI_SOURCE_URI,
            "source_revision": SNLI_SOURCE_REVISION,
            "archive_path": "data/raw/snli_1.0.zip",
            "archive_sha256": archive_sha256,
            "development_member": SNLI_DEV_MEMBER,
            "development_path": "data/raw/snli_1.0_dev.jsonl",
            "development_sha256": dev_sha256,
            "development_rows": native_row_count,
            "unlabeled_rows": unlabeled_row_count,
            "license": SNLI_LICENSE,
            "license_notice_uri": "https://nlp.stanford.edu/projects/snli/",
            "license_page_sha256": SNLI_LICENSE_PAGE_SHA256,
        },
        "prior_recipes": [
            {
                "path": receipt.path,
                "sha256": receipt.sha256,
                "excluded_group_count": len(receipt.group_ids),
            }
            for receipt in prior_recipes
        ],
        "selection": {
            "seed": SELECTION_SEED,
            "group_ranking": "ascending SHA256(seed + ':group:' + group ID), then group ID",
            "pair_ranking": "ascending SHA256(seed + ':pair:' + pairID), then pairID",
            "labels": list(SNLI_LABELS),
            "per_label_per_group": 1,
            "source_group_count": source_group_count,
            "excluded_group_count": len(excluded_group_ids),
            "excluded_group_ids": sorted(excluded_group_ids),
            "eligible_group_count": eligible_group_count,
            "selected_group_count": len(selected_groups),
            "selected_group_ids": [group["source_group_id"] for group in selected_groups],
            "selected_groups": selected_groups,
        },
        "outputs": {
            "manifest_file": "manifest.json",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "records_file": "records.jsonl",
            "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        },
    }


def serialize_recipe(recipe: dict[str, object]) -> bytes:
    return (json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_candidate_files(
    candidate: CandidateArtifacts,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Create a fresh output directory and exclusively write all three candidate files."""

    target = Path(output_dir)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output directory must not already exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    for name, payload in (
        ("records.jsonl", candidate.records_bytes),
        ("manifest.json", candidate.manifest_bytes),
        ("recipe.json", candidate.recipe_bytes),
    ):
        with (target / name).open("xb") as output:
            output.write(payload)
    return target
