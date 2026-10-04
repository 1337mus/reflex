"""Dataset manifests, decision records, and split-lineage audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .schema import DecisionRequest

SplitName = Literal["train", "development", "calibration", "test"]
DataKind = Literal["fixture", "benchmark"]


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class DatasetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    source_id: str
    task_family: str
    split: SplitName
    source_uri: str
    source_revision: str
    license: str

    @field_validator(
        "dataset_id", "source_id", "task_family", "source_uri", "source_revision", "license"
    )
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        return _nonblank(value)


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_kind: DataKind
    datasets: tuple[DatasetSpec, ...]
    held_out_families: tuple[str, ...] = ()

    @field_validator("held_out_families")
    @classmethod
    def validate_held_out_families(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _nonblank(value)
        if len(values) != len(set(values)):
            raise ValueError("held_out_families must be unique")
        return values

    @model_validator(mode="after")
    def require_unique_dataset_ids(self) -> SplitManifest:
        ids = [dataset.dataset_id for dataset in self.datasets]
        if not ids:
            raise ValueError("datasets must not be empty")
        if len(ids) != len(set(ids)):
            raise ValueError("dataset IDs must be unique")
        return self


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    dataset_id: str
    source_group_id: str
    request: DecisionRequest
    answer_id: str

    @field_validator("record_id", "dataset_id", "source_group_id", "answer_id")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def answer_is_legal_option(self) -> DecisionRecord:
        if self.answer_id not in {option.id for option in self.request.options}:
            raise ValueError("answer_id must match one of the request option IDs")
        return self


class AuditSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_kind: DataKind
    dataset_count: int
    record_count: int
    split_counts: tuple[tuple[SplitName, int], ...]
    held_out_families: tuple[str, ...]


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError("duplicate object key")
        result[key] = value
    return result


def _load_json(text: str, *, path: Path, line: int) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except _DuplicateKeyError as exc:
        raise ValueError(f"{path}:{line}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line}: malformed JSON") from exc


def _validation_message(path: Path, line: int, label: str, error: Exception) -> ValueError:
    if hasattr(error, "errors"):
        details = error.errors(include_input=False, include_context=False)
        fields = [".".join(str(part) for part in item.get("loc", ())) for item in details]
        where = ", ".join(field or "document" for field in fields)
        return ValueError(f"{path}:{line}: invalid {label} fields: {where}")
    return ValueError(f"{path}:{line}: invalid {label}")


def load_manifest(path: str | Path) -> SplitManifest:
    """Load a strict JSON manifest without echoing its payload in errors."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{source}: could not read manifest") from exc
    if not text.strip():
        raise ValueError(f"{source}:1: empty manifest")
    raw = _load_json(text, path=source, line=1)
    try:
        return SplitManifest.model_validate(raw)
    except Exception as exc:
        raise _validation_message(source, 1, "manifest", exc) from exc


def load_records(path: str | Path) -> tuple[DecisionRecord, ...]:
    """Load a JSON array or JSONL records, rejecting duplicate keys and IDs."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{source}: could not read records") from exc
    suffix = source.suffix.lower()
    entries: list[tuple[int, Any]] = []
    if suffix == ".jsonl":
        for line_number, line_text in enumerate(text.split("\n"), start=1):
            if line_text.endswith("\r"):
                line_text = line_text[:-1]
            if line_text.strip():
                entries.append((line_number, _load_json(line_text, path=source, line=line_number)))
    elif suffix == ".json":
        if not text.strip():
            raise ValueError(f"{source}:1: empty records file")
        raw = _load_json(text, path=source, line=1)
        if not isinstance(raw, list):
            raise ValueError(f"{source}:1: JSON records must be an array")
        entries = [(index + 1, value) for index, value in enumerate(raw)]
    else:
        raise ValueError(f"{source}: unsupported records file extension; use .json or .jsonl")

    if not entries:
        raise ValueError(f"{source}:1: records must not be empty")
    records: list[DecisionRecord] = []
    seen_ids: set[str] = set()
    for line, raw_record in entries:
        try:
            record = DecisionRecord.model_validate(raw_record)
        except Exception as exc:
            raise _validation_message(source, line, "record", exc) from exc
        if record.record_id in seen_ids:
            raise ValueError(f"{source}:{line}: duplicate record_id")
        seen_ids.add(record.record_id)
        records.append(record)
    return tuple(records)


def audit_splits(
    manifest: SplitManifest,
    records: tuple[DecisionRecord, ...] | list[DecisionRecord],
) -> AuditSummary:
    """Check supplied dataset, source-group, and exact-request split lineage."""

    datasets = {dataset.dataset_id: dataset for dataset in manifest.datasets}
    source_splits: dict[str, SplitName] = {}
    family_splits: dict[str, set[SplitName]] = {}
    for dataset_spec in manifest.datasets:
        prior_split = source_splits.setdefault(dataset_spec.source_id, dataset_spec.split)
        if prior_split != dataset_spec.split:
            raise ValueError(f"source_id {dataset_spec.source_id!r} crosses dataset splits")
        family_splits.setdefault(dataset_spec.task_family, set()).add(dataset_spec.split)

    for family in manifest.held_out_families:
        splits = family_splits.get(family, set())
        if not splits or splits != {"test"}:
            raise ValueError(f"held-out task family {family!r} must occur only in test")

    seen_record_ids: set[str] = set()
    group_splits: dict[tuple[str, str], SplitName] = {}
    request_splits: dict[str, SplitName] = {}
    counts: dict[SplitName, int] = {
        "train": 0,
        "development": 0,
        "calibration": 0,
        "test": 0,
    }
    for record in records:
        if record.record_id in seen_record_ids:
            raise ValueError(f"duplicate record_id {record.record_id!r}")
        seen_record_ids.add(record.record_id)
        record_dataset = datasets.get(record.dataset_id)
        if record_dataset is None:
            raise ValueError(f"record {record.record_id!r} references an unknown dataset")
        split = record_dataset.split
        counts[split] += 1

        group_key = (record_dataset.source_id, record.source_group_id)
        previous_group_split = group_splits.setdefault(group_key, split)
        if previous_group_split != split:
            raise ValueError("a source group crosses dataset splits")

        previous_request_split = request_splits.setdefault(record.request.request_hash, split)
        if previous_request_split != split:
            raise ValueError("an exact semantic request appears across dataset splits")

    split_order: tuple[SplitName, ...] = ("train", "development", "calibration", "test")
    return AuditSummary(
        data_kind=manifest.data_kind,
        dataset_count=len(manifest.datasets),
        record_count=len(records),
        split_counts=tuple((split, counts[split]) for split in split_order),
        held_out_families=manifest.held_out_families,
    )
