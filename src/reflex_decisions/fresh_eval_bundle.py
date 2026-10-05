"""Pinned source bundle generation and read-only panel verification."""

from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from .data import DecisionRecord
from .fresh_eval_build import _select_panel
from .fresh_eval_data import (
    ARC_DATASET_ID,
    DATASET_COUNTS,
    HANS_DATASET_ID,
    HANS_PER_SUBCASE,
    HANS_SUBCASE_COUNT,
    SELECTION_SEED,
    WINOGRANDE_DATASET_ID,
    _Candidate,
    _canonical_json,
    _sha256,
    parse_arc_dev_jsonl,
    parse_hans_tsv,
    parse_winogrande_dev,
)

_RAW_DIRECTORY = Path("data/raw/fresh-eval-v1")
_PROCESSED_DIRECTORY = Path("data/processed/fresh-eval-v1")
_HANS_PATH = _RAW_DIRECTORY / "hans-heuristics_evaluation_set.txt"
_WINO_ARCHIVE_PATH = _RAW_DIRECTORY / "winogrande_1.1.zip"
_ARC_ARCHIVE_PATH = _RAW_DIRECTORY / "ARC-V1-Feb2018.zip"
_WINO_DATA_MEMBER = "winogrande_1.1/dev.jsonl"
_WINO_LABEL_MEMBER = "winogrande_1.1/dev-labels.lst"
_ARC_DEV_MEMBER = "ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl"
_SOURCE_FILE_SHA256 = {
    _HANS_PATH.as_posix(): "c55b62feef9913070e88f38938dc2492018c945ac81f70139346472494124e79",
    _WINO_ARCHIVE_PATH.as_posix(): (
        "3619ab104d8be2977b25c90ff420cb42d491707dcc75362a1e5d22bc082b7318"
    ),
    _ARC_ARCHIVE_PATH.as_posix(): (
        "6d2d5ab50b2ceec6ba5f79c921be77cf2de712ea25a2b3f4fff3acc101cecfa0"
    ),
}
_SOURCE_MEMBER_SHA256 = {
    "hans-eval-v1": _SOURCE_FILE_SHA256[_HANS_PATH.as_posix()],
    "winogrande-dev-jsonl": "1aeac79cf46a3dcbe59a71ca7e5372a1490cde2b4ae2387dc56f00378104d2e8",
    "winogrande-dev-labels": "5ae1e30f52a50b880515685dcfae7d57649cce256d375947cdd02f687de7e6ba",
    "arc-challenge-dev-jsonl": "55224364abd2477f66a9a57c5cbef109fb71d83019aa5e37aec0e7ab757746ca",
}
_KNOWN_ARC_PREVIEW_STEM_SHA256 = "b35be3ef6081eb2879a2e27cb5648bc88cd5ebd63f67ce575169eae33aaff524"
_ARC_CARD_SNAPSHOT_SHA256 = "4a293e352253d771604725c978adcea71ee6504e89ed40e1cb0024ecafa9b37e"
_WINO_CARD_SNAPSHOT_SHA256 = "fd653689cc77706672b40cd040156ad57f4969656d25a14be645cffa8ef55176"
_EXPECTED_MANIFEST_SHA256 = "3937d9ebbb49a05caa0022ffbf08db4ddbf04e2aaf3cce95cf33ad3c2aa612ea"
_EXPECTED_RECIPE_SHA256 = "384ca0db4450014c8a93b8ef611c2c2f19cc453100cbd6ac009af7199da9b30c"


def _record_sha256(record: DecisionRecord) -> str:
    return _sha256(_canonical_json(record.model_dump(mode="json")))


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "fresh-eval-v1",
        "sources": [
            {
                "dataset_id": HANS_DATASET_ID,
                "revision": "tommccoy1/hans@7299f6f657089ce06a0f98e7e81f8d0f5b7741ce",
                "url": "https://github.com/tommccoy1/hans/blob/7299f6f657089ce06a0f98e7e81f8d0f5b7741ce/heuristics_evaluation_set.txt",
                "file": _HANS_PATH.as_posix(),
                "file_sha256": _SOURCE_FILE_SHA256[_HANS_PATH.as_posix()],
                "member": "heuristics_evaluation_set.txt",
                "member_sha256": _SOURCE_MEMBER_SHA256[HANS_DATASET_ID],
                "license": "Repository MIT notice; no separate dataset terms identified.",
            },
            {
                "dataset_id": WINOGRANDE_DATASET_ID,
                "revision": (
                    "allenai/winogrande downloader@727e837f77521ef38bcc56df3b275c8da43f45af"
                ),
                "url": "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip",
                "file": _WINO_ARCHIVE_PATH.as_posix(),
                "file_sha256": _SOURCE_FILE_SHA256[_WINO_ARCHIVE_PATH.as_posix()],
                "members": [
                    {
                        "path": _WINO_DATA_MEMBER,
                        "sha256": _SOURCE_MEMBER_SHA256["winogrande-dev-jsonl"],
                    },
                    {
                        "path": _WINO_LABEL_MEMBER,
                        "sha256": _SOURCE_MEMBER_SHA256["winogrande-dev-labels"],
                    },
                ],
                "license": "Dataset CC-BY, version unspecified; codebase separately Apache-2.0.",
            },
            {
                "dataset_id": ARC_DATASET_ID,
                "revision": "allenai/ai2_arc card snapshot sha256 " + _ARC_CARD_SNAPSHOT_SHA256,
                "url": "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip",
                "file": _ARC_ARCHIVE_PATH.as_posix(),
                "file_sha256": _SOURCE_FILE_SHA256[_ARC_ARCHIVE_PATH.as_posix()],
                "member": _ARC_DEV_MEMBER,
                "member_sha256": _SOURCE_MEMBER_SHA256["arc-challenge-dev-jsonl"],
                "license": "Dataset card CC-BY-SA-4.0.",
            },
        ],
        "source_file_sha256": dict(sorted(_SOURCE_FILE_SHA256.items())),
        "preview_evidence": {
            "arc_card_snapshot_sha256": _ARC_CARD_SNAPSHOT_SHA256,
            "arc_known_preview_stem_sha256": _KNOWN_ARC_PREVIEW_STEM_SHA256,
            "arc_known_preview_excluded": 1,
            "winogrande_card_snapshot_sha256": _WINO_CARD_SNAPSHOT_SHA256,
            "winogrande_preview_records_found": 0,
            "limitation": (
                "Other or future public previews may overlap; absence is not established."
            ),
        },
    }


def _recipe_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": "fresh-eval-v1",
        "parser": "strict-public-json-tsv-v1",
        "selection_seed": SELECTION_SEED,
        "rank": "sha256(utf8(fresh-eval-v1:{seed}:{dataset_id}:{raw_source_id}))",
        "deduplication": [
            "reject duplicate IDs that do not map to identical candidates",
            "collapse exact semantic requests with the same answer",
            "reject exact semantic requests with conflicting answer IDs",
        ],
        "quotas": {
            "hans-eval-v1": {"subcases": HANS_SUBCASE_COUNT, "per_subcase": HANS_PER_SUBCASE},
            "winogrande-dev-v1": DATASET_COUNTS[WINOGRANDE_DATASET_ID],
            "arc-challenge-dev-v1": DATASET_COUNTS[ARC_DATASET_ID],
        },
        "preview_exclusion": {
            "dataset_id": ARC_DATASET_ID,
            "stem_sha256": _KNOWN_ARC_PREVIEW_STEM_SHA256,
        },
        "arc_option_mapping": {
            "option_id": "source answer-choice label",
            "display_label": "source answer-choice text",
            "description": None,
        },
        "presentation": "original option order and one left cyclic rotation; preserve option IDs",
        "sort_order": ["dataset_id", "record_id", "order_index"],
    }


def _manifest_bytes() -> bytes:
    return _canonical_json(_manifest_payload()) + b"\n"


def _recipe_bytes() -> bytes:
    return _canonical_json(_recipe_payload()) + b"\n"


def _records_bytes(records: Sequence[DecisionRecord]) -> bytes:
    return b"".join(_canonical_json(record.model_dump(mode="json")) + b"\n" for record in records)


def _record_metadata_bytes(metadata_by_id: Mapping[str, object]) -> bytes:
    return _canonical_json(metadata_by_id) + b"\n"


def _safe_source_path(root: Path, relative_path: Path) -> Path:
    try:
        path = (root / relative_path).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("a pinned fresh evaluation source is unavailable") from exc
    if not path.is_file():
        raise ValueError("a pinned fresh evaluation source is not a file")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("could not hash a pinned fresh evaluation source") from exc
    return digest.hexdigest()


def _read_zip_member(archive: Path, member: str) -> bytes:
    try:
        with zipfile.ZipFile(archive) as source:
            info = source.getinfo(member)
            if info.file_size > 5_000_000 or info.is_dir():
                raise ValueError("a pinned fresh evaluation archive member has an invalid size")
            return source.read(info)
    except (OSError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("could not read an allowlisted fresh evaluation archive member") from exc


def _load_source_candidates(root: Path) -> tuple[list[_Candidate], dict[str, str]]:
    hans_path = _safe_source_path(root, _HANS_PATH)
    wino_archive = _safe_source_path(root, _WINO_ARCHIVE_PATH)
    arc_archive = _safe_source_path(root, _ARC_ARCHIVE_PATH)
    source_paths = {
        _HANS_PATH.as_posix(): hans_path,
        _WINO_ARCHIVE_PATH.as_posix(): wino_archive,
        _ARC_ARCHIVE_PATH.as_posix(): arc_archive,
    }
    source_hashes = {path: _file_sha256(file_path) for path, file_path in source_paths.items()}
    if source_hashes != _SOURCE_FILE_SHA256:
        raise ValueError("fresh evaluation source archive hashes differ from pinned revisions")
    try:
        hans_data = hans_path.read_bytes()
    except OSError as exc:
        raise ValueError("could not read pinned HANS source") from exc
    wino_data = _read_zip_member(wino_archive, _WINO_DATA_MEMBER)
    wino_labels = _read_zip_member(wino_archive, _WINO_LABEL_MEMBER)
    arc_data = _read_zip_member(arc_archive, _ARC_DEV_MEMBER)
    member_hashes = {
        HANS_DATASET_ID: _sha256(hans_data),
        "winogrande-dev-jsonl": _sha256(wino_data),
        "winogrande-dev-labels": _sha256(wino_labels),
        "arc-challenge-dev-jsonl": _sha256(arc_data),
    }
    if member_hashes != _SOURCE_MEMBER_SHA256:
        raise ValueError("fresh evaluation allowlisted source member hashes differ from pins")
    candidates = [
        *parse_hans_tsv(hans_data),
        *parse_winogrande_dev(wino_data, wino_labels),
        *parse_arc_dev_jsonl(arc_data),
    ]
    return candidates, dict(sorted(source_hashes.items()))


def _project_root(root: str | Path) -> Path:
    try:
        return Path(root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("fresh evaluation project root is unavailable") from exc


def _generated_panel(root: Path) -> tuple[list[DecisionRecord], dict[str, object]]:
    candidates, source_hashes = _load_source_candidates(root)
    records, selected_metadata = _select_panel(
        candidates,
        preview_questions=set(),
        preview_question_sha256s={_KNOWN_ARC_PREVIEW_STEM_SHA256},
    )
    manifest = _manifest_bytes()
    recipe = _recipe_bytes()
    records_payload = _records_bytes(records)
    record_metadata = _record_metadata_bytes(
        selected_metadata["record_metadata_by_id"]  # type: ignore[arg-type]
    )
    manifest_sha = _sha256(manifest)
    recipe_sha = _sha256(recipe)
    if manifest_sha != _EXPECTED_MANIFEST_SHA256 or recipe_sha != _EXPECTED_RECIPE_SHA256:
        raise ValueError("fresh evaluation manifest or recipe differs from its frozen pin")
    records_sha = _sha256(records_payload)
    metadata_sha = _sha256(record_metadata)
    metadata: dict[str, object] = {
        "gpu_pins": {
            "data_file_sha256": source_hashes,
            "dataset_counts": dict(DATASET_COUNTS),
            "panel_sha256": records_sha,
            "source_manifest_sha256": manifest_sha,
        },
        "source_file_sha256": source_hashes,
        "manifest_sha256": manifest_sha,
        "recipe_sha256": recipe_sha,
        "records_sha256": records_sha,
        "record_metadata_sha256": metadata_sha,
        "counts": {
            "total": len(records),
            "by_dataset": dict(Counter(record.dataset_id for record in records)),
            "presentations": 2 * len(records),
            "parity_presentations": 24,
        },
        "selection": selected_metadata["selection"],
        "record_metadata_by_id": selected_metadata["record_metadata_by_id"],
    }
    files = {
        "manifest.json": manifest,
        "recipe.json": recipe,
        "records.jsonl": records_payload,
        "record-metadata.json": record_metadata,
    }
    return records, {**metadata, "_generated_files": files}


def _write_frozen_file(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ValueError("refusing to follow a fresh evaluation bundle symlink")
    if path.exists():
        try:
            if path.read_bytes() == payload:
                return
        except OSError as exc:
            raise ValueError("could not verify an existing fresh evaluation bundle file") from exc
        raise ValueError("refusing to replace a different frozen fresh evaluation bundle file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".fresh-eval-", delete=False
        ) as output:
            temporary_path = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise ValueError(
                    "refusing to replace a different frozen fresh evaluation bundle file"
                ) from None
    except OSError as exc:
        raise ValueError("could not write the fresh evaluation bundle") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_panel(root: str | Path) -> tuple[list[DecisionRecord], dict[str, object]]:
    """Create the ignored pinned bundle once, refusing to overwrite differences."""

    project_root = _project_root(root)
    records, generated = _generated_panel(project_root)
    files = generated.pop("_generated_files")
    assert isinstance(files, dict)
    output_directory = project_root / _PROCESSED_DIRECTORY
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        output_directory = output_directory.resolve(strict=True)
        output_directory.relative_to(project_root)
    except (OSError, ValueError) as exc:
        raise ValueError("fresh evaluation output directory escapes the project root") from exc
    for name, payload in files.items():
        if not isinstance(name, str) or not isinstance(payload, bytes):
            raise ValueError("generated fresh evaluation bundle is malformed")
        _write_frozen_file(output_directory / name, payload)
    return records, generated


def load_panel(root: str | Path) -> tuple[list[DecisionRecord], dict[str, object]]:
    """Verify all pinned raw and processed files without modifying the bundle."""

    project_root = _project_root(root)
    records, expected_metadata = _generated_panel(project_root)
    expected_files = expected_metadata.pop("_generated_files")
    assert isinstance(expected_files, dict)
    for name, expected in expected_files.items():
        if not isinstance(name, str) or not isinstance(expected, bytes):
            raise ValueError("generated fresh evaluation bundle is malformed")
        path = _safe_source_path(project_root, _PROCESSED_DIRECTORY / name)
        try:
            actual = path.read_bytes()
        except OSError as exc:
            raise ValueError("could not read a frozen fresh evaluation bundle file") from exc
        if actual != expected:
            raise ValueError("processed fresh evaluation bundle differs from its pinned sources")
    return records, expected_metadata
