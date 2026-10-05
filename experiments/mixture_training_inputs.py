"""Pinned CPU data loading and candidate rebuild checks for mixture experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiments import mixture_training_data as mixture_data
from experiments import real_pilot_core
from experiments.mixture_training_contracts import (
    DATA_FILE_SHA256,
    PROTOCOL_PATH,
    make_pins,
    verify_protocol,
)
from reflex_decisions import pilot_data, snli_diagnostic, synthetic_data
from reflex_decisions.data import DecisionRecord


def _canonical_data_file_bytes(root: Path, relative: str) -> bytes:
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        return resolved.read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError(f"pinned data file is unavailable: {relative}") from exc


def _verify_data_file_pins(root: Path) -> None:
    for relative, expected in DATA_FILE_SHA256.items():
        actual = hashlib.sha256(_canonical_data_file_bytes(root, relative)).hexdigest()
        if actual != expected:
            raise ValueError(f"data file SHA-256 mismatch: {relative}")


def _verify_balanced_candidate(root: Path) -> tuple[DecisionRecord, ...]:
    candidate = snli_diagnostic.build_candidate(root)
    candidate_bytes = {
        "records.jsonl": candidate.records_bytes,
        "manifest.json": candidate.manifest_bytes,
        "recipe.json": candidate.recipe_bytes,
    }
    expected_paths = {
        "records.jsonl": "data/processed/snli-balanced-v1/records.jsonl",
        "manifest.json": "data/processed/snli-balanced-v1/manifest.json",
        "recipe.json": "data/processed/snli-balanced-v1/recipe.json",
    }
    for filename, content in candidate_bytes.items():
        relative = expected_paths[filename]
        if content != _canonical_data_file_bytes(root, relative):
            raise ValueError(f"balanced SNLI candidate differs from frozen bytes: {filename}")
        if hashlib.sha256(content).hexdigest() != DATA_FILE_SHA256[relative]:
            raise ValueError(f"balanced SNLI candidate pin mismatch: {filename}")
    records = candidate.bundle.records
    if len(records) != 192 or any(row.dataset_id != snli_diagnostic.DATASET_ID for row in records):
        raise ValueError("balanced SNLI candidate has an invalid record panel")
    return records


def _verify_synthetic_candidate(root: Path) -> tuple[DecisionRecord, ...]:
    candidate = synthetic_data.build_candidate()
    files = {
        "records.jsonl": synthetic_data._jsonl(
            row.model_dump(mode="json") for row in candidate.records
        ),
        "manifest.json": synthetic_data._json(
            candidate.manifest.model_dump(mode="json"), pretty=True
        ),
        "recipe.json": synthetic_data._json(candidate.recipe, pretty=True),
        "provenance.jsonl": synthetic_data._jsonl(candidate.provenance),
        "audit.json": synthetic_data._json(candidate.audit, pretty=True),
    }
    base = "data/processed/synthetic-seed-v1-r2/"
    for filename, content in files.items():
        relative = base + filename
        if content != _canonical_data_file_bytes(root, relative):
            raise ValueError(f"synthetic candidate differs from frozen bytes: {filename}")
        if hashlib.sha256(content).hexdigest() != DATA_FILE_SHA256[relative]:
            raise ValueError(f"synthetic candidate pin mismatch: {filename}")
    if len(candidate.records) != sum(mixture_data.SYNTHETIC_DATASET_COUNTS.values()):
        raise ValueError("rebuilt synthetic candidate has an invalid record count")
    return candidate.records


def load_local_data(
    root: str | Path,
) -> tuple[
    tuple[DecisionRecord, ...],
    tuple[DecisionRecord, ...],
    tuple[DecisionRecord, ...],
    dict[str, object],
]:
    """Verify all pinned data and return real, balanced-SNLI, synthetic bundles and pins."""

    project_root = Path(root).resolve(strict=True)
    verify_protocol(project_root / PROTOCOL_PATH)
    _verify_data_file_pins(project_root)
    real_manifest, real_records, _recipe = pilot_data.verify_prepared_data(
        project_root / pilot_data.DEFAULT_RECORDS_PATH,
        project_root / pilot_data.DEFAULT_MANIFEST_PATH,
        project_root / pilot_data.DEFAULT_RECIPE_PATH,
    )
    real_pilot_core.validate_pilot_data(real_manifest, real_records)
    balanced_records = _verify_balanced_candidate(project_root)
    synthetic_records = _verify_synthetic_candidate(project_root)

    real_train = tuple(
        row for row in real_records if row.dataset_id in mixture_data.REAL_TRAIN_DATASET_COUNTS
    )
    synthetic_train = tuple(
        row
        for row in synthetic_records
        if row.dataset_id in mixture_data.SYNTHETIC_TRAIN_DATASET_COUNTS
    )
    mixture_data.build_training_schedule_audit(real_train, synthetic_train)
    presentations = mixture_data.build_evaluation_presentations(
        real_records, balanced_records, synthetic_records
    )
    mixture_data.build_evaluation_panel_audit(
        presentations, (*real_records, *balanced_records, *synthetic_records)
    )
    return real_records, balanced_records, synthetic_records, make_pins(project_root)


__all__ = ["load_local_data"]
