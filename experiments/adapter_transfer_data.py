"""Pinned COPA/BoolQ data panel construction and training-overlap checks."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments import adapter_transfer_contracts as contracts
from experiments import adapter_transfer_core as core
from experiments import mixture_training_data, mixture_training_inputs
from experiments.mixture_training_contracts import validate_pins as validate_training_pins
from reflex_decisions import baseline_data, broader_data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest

_DATASETS = ("boolq-dev-pilot-v1", "copa-dev-pilot-v1")
_RECORDS_PER_DATASET = 32
_PRESENTATION_FIELDS = {
    "presentation_id",
    "record_id",
    "dataset_id",
    "source_group_id",
    "request_hash",
    "order_index",
    "order_ids",
    "request",
}


def _record_group_alias(dataset_id: str, source_group_id: str) -> str:
    digest = hashlib.sha256(
        f"adapter-transfer-v1:{dataset_id}:{source_group_id}".encode()
    ).hexdigest()
    return f"adapter-transfer-v1:{dataset_id}:{digest}"


def build_presentations(records: object) -> list[dict[str, object]]:
    """Build exact sorted original/reversed requests without returning labels."""

    if not isinstance(records, Sequence) or len(records) != 2 * _RECORDS_PER_DATASET:
        raise ValueError("adapter-transfer panel must contain exactly 64 records")
    typed_records = tuple(records)
    if any(not isinstance(record, DecisionRecord) for record in typed_records):
        raise ValueError("adapter-transfer panel contains a malformed decision record")
    counts = Counter(record.dataset_id for record in typed_records)
    if counts != Counter({dataset_id: _RECORDS_PER_DATASET for dataset_id in _DATASETS}):
        raise ValueError("adapter-transfer panel must contain exactly 32 COPA and 32 BoolQ records")
    if len({record.record_id for record in typed_records}) != len(typed_records):
        raise ValueError("adapter-transfer panel contains duplicate record IDs")
    if len({(record.dataset_id, record.source_group_id) for record in typed_records}) != len(
        typed_records
    ):
        raise ValueError("adapter-transfer panel contains duplicate source groups")

    output: list[dict[str, object]] = []
    for dataset_id in _DATASETS:
        dataset_records = sorted(
            (record for record in typed_records if record.dataset_id == dataset_id),
            key=lambda record: record.record_id,
        )
        for record in dataset_records:
            group_alias = _record_group_alias(dataset_id, record.source_group_id)
            for order_index, options in enumerate(
                (record.request.options, tuple(reversed(record.request.options)))
            ):
                request = record.request.model_copy(update={"options": options})
                output.append(
                    {
                        "presentation_id": (
                            f"adapter-transfer-v1:{record.record_id}:order-{order_index}"
                        ),
                        "record_id": record.record_id,
                        "dataset_id": dataset_id,
                        "source_group_id": group_alias,
                        "request_hash": request.request_hash,
                        "order_index": order_index,
                        "order_ids": [option.id for option in request.options],
                        "request": request.model_dump(mode="json"),
                    }
                )
    return output


def _validate_presentation_panel(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 2 * _RECORDS_PER_DATASET * 2:
        raise ValueError("adapter-transfer presentations must contain exactly 128 rows")
    rows: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != _PRESENTATION_FIELDS:
            raise ValueError("adapter-transfer presentation has an unexpected schema")
        if (
            not isinstance(row["record_id"], str)
            or not isinstance(row["dataset_id"], str)
            or row["dataset_id"] not in _DATASETS
            or not isinstance(row["source_group_id"], str)
            or not row["source_group_id"].startswith(f"adapter-transfer-v1:{row['dataset_id']}:")
            or type(row["order_index"]) is not int
            or row["order_index"] not in (0, 1)
        ):
            raise ValueError("adapter-transfer presentation identity is malformed")
        request = DecisionRequest.model_validate(row["request"])
        if (
            row["request_hash"] != request.request_hash
            or row["order_ids"] != [option.id for option in request.options]
            or row["presentation_id"]
            != f"adapter-transfer-v1:{row['record_id']}:order-{row['order_index']}"
        ):
            raise ValueError("adapter-transfer presentation request identity is inconsistent")
        rows.append(row)
    pairs = list(zip(rows[::2], rows[1::2], strict=True))
    record_ids: dict[str, list[str]] = {dataset_id: [] for dataset_id in _DATASETS}
    groups: dict[str, set[str]] = {dataset_id: set() for dataset_id in _DATASETS}
    for original, reversed_order in pairs:
        if (
            original["order_index"] != 0
            or reversed_order["order_index"] != 1
            or original["record_id"] != reversed_order["record_id"]
            or original["dataset_id"] != reversed_order["dataset_id"]
            or original["source_group_id"] != reversed_order["source_group_id"]
        ):
            raise ValueError("adapter-transfer presentations are not original/reversed pairs")
        dataset_id = str(original["dataset_id"])
        record_ids[dataset_id].append(str(original["record_id"]))
        groups[dataset_id].add(str(original["source_group_id"]))
        request = DecisionRequest.model_validate(original["request"])
        reverse_request = DecisionRequest.model_validate(reversed_order["request"])
        if (
            reverse_request.options != tuple(reversed(request.options))
            or original["request_hash"] != reversed_order["request_hash"]
        ):
            raise ValueError("adapter-transfer reversed order does not match its original request")
    if any(record_ids[dataset_id] != sorted(record_ids[dataset_id]) for dataset_id in _DATASETS):
        raise ValueError("adapter-transfer presentation records are not sorted")
    if [row["dataset_id"] for row in rows[::2]] != [
        dataset_id for dataset_id in _DATASETS for _ in range(_RECORDS_PER_DATASET)
    ]:
        raise ValueError("adapter-transfer presentations have an unexpected task order")
    if any(len(set(record_ids[dataset_id])) != _RECORDS_PER_DATASET for dataset_id in _DATASETS):
        raise ValueError("adapter-transfer presentations contain duplicate records")
    if any(len(groups[dataset_id]) != _RECORDS_PER_DATASET for dataset_id in _DATASETS):
        raise ValueError("adapter-transfer presentations contain duplicate source groups")
    return rows


def parity_presentations(presentations: object) -> list[dict[str, object]]:
    """Select both orders for the first four sorted records in each task."""

    rows = _validate_presentation_panel(presentations)
    selected: set[str] = set()
    for dataset_id in _DATASETS:
        record_ids = sorted(
            {
                str(row["record_id"])
                for row in rows
                if row["dataset_id"] == dataset_id and row["order_index"] == 0
            }
        )
        if len(record_ids) != _RECORDS_PER_DATASET:
            raise ValueError("adapter-transfer task does not contain exactly 32 source groups")
        selected.update(record_ids[:4])
    parity = [row for row in rows if row["record_id"] in selected]
    if len(parity) != 16:
        raise ValueError("adapter-transfer reload parity panel is malformed")
    return parity


def validate_training_disjointness(records: object, training_records: object) -> None:
    """Reject record, source-group, or semantic-request overlap with adapter training."""

    if not isinstance(records, Sequence) or not isinstance(training_records, Sequence):
        raise ValueError("adapter-transfer and training pools must be record sequences")
    if any(not isinstance(record, DecisionRecord) for record in (*records, *training_records)):
        raise ValueError("adapter-transfer overlap audit received a malformed record")
    train_ids = {record.record_id for record in training_records}
    train_groups = {record.source_group_id for record in training_records}
    train_requests = {record.request.request_hash for record in training_records}
    if train_ids & {record.record_id for record in records}:
        raise ValueError("adapter-transfer record IDs overlap an approved training pool")
    if train_groups & {record.source_group_id for record in records}:
        raise ValueError("adapter-transfer source groups overlap an approved training pool")
    if train_requests & {record.request.request_hash for record in records}:
        raise ValueError("adapter-transfer semantic requests overlap an approved training pool")


def load_inputs(
    root: str | Path,
) -> tuple[
    tuple[DecisionRecord, ...], list[dict[str, object]], dict[str, object], dict[str, object]
]:
    """Verify selection, source bundles, and approved training-pool separation."""

    project_root = Path(root).resolve(strict=True)
    selection, selection_file_sha256 = contracts.load_selection(project_root)
    panel_hashes = _verify_panel_files(project_root)
    _verify_protocol(project_root)
    _copa_manifest, copa_records = baseline_data.verify_prepared_data(
        project_root / contracts.COPA_RECORDS_PATH,
        project_root / contracts.COPA_MANIFEST_PATH,
    )
    _broader_manifest, broader_records = broader_data.verify_prepared_data(
        project_root / contracts.BROADER_RECORDS_PATH,
        project_root / contracts.BROADER_MANIFEST_PATH,
        project_root / contracts.BROADER_RECIPE_PATH,
    )
    if {row.dataset_id for row in broader_records} != {
        broader_data.BOOLQ_DATASET_ID,
        broader_data.SNLI_DATASET_ID,
    }:
        raise ValueError("verified broader bundle contains an unexpected dataset")
    boolq_records = tuple(
        row for row in broader_records if row.dataset_id == broader_data.BOOLQ_DATASET_ID
    )
    if len(boolq_records) != _RECORDS_PER_DATASET:
        raise ValueError("verified broader bundle does not contain exactly 32 BoolQ records")
    boolq_digest = hashlib.sha256(broader_data.serialize_records(boolq_records)).hexdigest()
    if boolq_digest != contracts.BOOLQ_PANEL_SHA256:
        raise ValueError("filtered BoolQ panel SHA-256 mismatch")

    records = tuple(
        sorted(
            (*copa_records, *boolq_records),
            key=lambda row: (_DATASETS.index(row.dataset_id), row.record_id),
        )
    )
    presentations = build_presentations(records)
    real_records, _balanced_records, synthetic_records, snli_train_records, training_pins_raw = (
        mixture_training_inputs.load_natural_reasoning_data(project_root)
    )
    training_pins = validate_training_pins(training_pins_raw)
    training_records = (
        tuple(
            record
            for record in real_records
            if record.dataset_id in mixture_training_data.REAL_TRAIN_DATASET_COUNTS
        )
        + tuple(
            record
            for record in synthetic_records
            if record.dataset_id in mixture_training_data.SYNTHETIC_TRAIN_DATASET_COUNTS
        )
        + tuple(snli_train_records)
    )
    validate_training_disjointness(records, training_records)
    source_hashes = core.source_fingerprints(project_root)
    pins: dict[str, Any] = {
        "data_file_sha256": panel_hashes,
        "training_data_file_sha256": dict(training_pins["data_file_sha256"]),
        "panel_sha256": {
            "copa": panel_hashes[contracts.COPA_RECORDS_PATH],
            "boolq": boolq_digest,
        },
        "selection_file_sha256": selection_file_sha256,
        "protocol_sha256": contracts.EXPECTED_PROTOCOL_SHA256,
        "training_protocol_sha256": training_pins["protocol_sha256"],
        "source_file_sha256": source_hashes,
    }
    return records, presentations, selection, pins


def _verified_file_bytes(root: Path, relative: str) -> bytes:
    canonical_root = root.resolve(strict=True)
    try:
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(canonical_root)
        if not resolved.is_file():
            raise ValueError("pinned input path is not a file")
        return resolved.read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError(f"pinned adapter-transfer input is unavailable: {relative}") from exc


def _verify_panel_files(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, expected in contracts.PANEL_FILE_SHA256.items():
        digest = hashlib.sha256(_verified_file_bytes(root, relative)).hexdigest()
        if digest != expected:
            raise ValueError(f"adapter-transfer data SHA-256 mismatch: {relative}")
        hashes[relative] = digest
    return hashes


def _verify_protocol(root: Path) -> None:
    digest = hashlib.sha256(_verified_file_bytes(root, contracts.PROTOCOL_PATH)).hexdigest()
    if digest != contracts.EXPECTED_PROTOCOL_SHA256:
        raise ValueError("adapter-transfer protocol SHA-256 mismatch")
