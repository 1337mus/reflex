"""Pinned CPU inputs for the runtime-rule study."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments import (
    adapter_transfer_contracts,
    adapter_transfer_data,
    mixture_training_contracts,
    mixture_training_data,
    mixture_training_inputs,
    runtime_rule_study_data,
)
from reflex_decisions.data import DecisionRecord

_STUDY_PROTOCOL_PATH = "docs/runtime-rule-study-protocol.md"
_STUDY_PROTOCOL_SHA256 = "11d58dbe075e08effd6b02614b64ac64715fe7a0db5330c0a45d6d0dc0a6ea23"
_TRANSFER_SUMMARY_PATH = "docs/verification/adapter-transfer-summary.json"
_TRANSFER_SUMMARY_SHA256 = "9594d10665e18c725f2312f636198f69277b7ac158bbbe47c1d6ca5be27e9441"
_NEW_FILE_SHA256 = {
    "data/processed/routing-data-v1/train.jsonl": (
        "7a4ef948025a8ab98c113d1917bae92cd0ed74389b87b4641c162e4fe53b6b67"
    ),
    "data/processed/routing-data-v1/development.jsonl": (
        "becef704b8fb342f8c1b75159af751212fcd9089fafb773420a416180e726366"
    ),
    "data/processed/tool-data-v1/train.jsonl": (
        "b00d021616d2f36dce5dc6f76e03a3aa884973e9af480b3e1c29fc8238cb9dca"
    ),
    "data/processed/tool-data-v1/development.jsonl": (
        "a9fb7eeefb67f97d16825f835860e6526ee034d1be2502c54eca3539bbd261ba"
    ),
}
_RETENTION_COUNTS = {
    "dbpedia14-pilot-v1-development": 1568,
    "sms-pilot-v1-development": 120,
    "snli-balanced-v1-development": 1152,
    "synthetic-atomic-fact-inference-v1-development": 450,
    "synthetic-numeric-selection-v1-development": 364,
    "boolq-dev-pilot-v1": 64,
    "copa-dev-pilot-v1": 64,
}
_OLD_RETENTION_DATASETS = frozenset(tuple(_RETENTION_COUNTS)[:5])
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class StudyInputs:
    training_pools: tuple[tuple[DecisionRecord, ...], ...]
    development_pools: tuple[tuple[DecisionRecord, ...], tuple[DecisionRecord, ...]]
    evaluation_records: tuple[DecisionRecord, ...]
    retention_presentations: tuple[dict[str, object], ...]
    new_presentations: tuple[dict[str, object], ...]
    selection: dict[str, object]
    file_sha256: dict[str, str]


def build_retention_presentations(
    real: Sequence[DecisionRecord],
    balanced: Sequence[DecisionRecord],
    synthetic: Sequence[DecisionRecord],
    transfer: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    """Build and validate the exact seven-task retention panel."""
    sources = (real, balanced, synthetic, transfer)
    if any(not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) for rows in sources):
        raise TypeError("retention sources must be sequences of DecisionRecord values")
    if any(not isinstance(record, DecisionRecord) for rows in sources for record in rows):
        raise TypeError("retention sources must contain DecisionRecord values")

    old_panel = mixture_training_data.build_evaluation_presentations(real, balanced, synthetic)
    mixture_training_data.build_evaluation_panel_audit(old_panel, (*real, *balanced, *synthetic))
    retained_old = tuple(
        dict(row) for row in old_panel if row["dataset_id"] in _OLD_RETENTION_DATASETS
    )
    transfer_panel = tuple(adapter_transfer_data.build_presentations(transfer))
    presentations = (*retained_old, *transfer_panel)

    if len(presentations) != 3782:
        raise ValueError("retention panel must contain exactly 3,782 presentations")
    if Counter(row["dataset_id"] for row in presentations) != Counter(_RETENTION_COUNTS):
        raise ValueError("retention panel dataset IDs or counts do not match the exact allowlist")
    if len({row["presentation_id"] for row in presentations}) != len(presentations):
        raise ValueError("retention presentation IDs must be unique")
    if any(set(row) != runtime_rule_study_data.PRESENTATION_FIELDS for row in presentations):
        raise ValueError("retention presentation has an unexpected or label-bearing field")
    if any(_has_answer_id(row) for row in presentations):
        raise ValueError("retention request payload must be label-free")

    expected_old_records = {
        record.record_id
        for record in (*real, *balanced, *synthetic)
        if record.dataset_id in _OLD_RETENTION_DATASETS
    }
    expected_transfer_records = {record.record_id for record in transfer}
    observed_old_records = {str(row["record_id"]) for row in retained_old}
    observed_transfer_records = {str(row["record_id"]) for row in transfer_panel}
    if (
        observed_old_records != expected_old_records
        or observed_transfer_records != expected_transfer_records
    ):
        raise ValueError("retention panel does not contain the exact approved record membership")
    return presentations


def _verified_bytes(root: Path, relative: str, expected_sha256: str) -> tuple[bytes, str]:
    """Read one pinned regular file after proving it resolves inside root."""

    try:
        resolved = (root / relative).resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_file():
            raise ValueError("pinned path is not a regular file")
        raw = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"pinned input is unavailable inside the project root: {relative}"
        ) from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"pinned input SHA-256 mismatch: {relative}")
    return raw, digest


def _has_answer_id(row: Mapping[str, object]) -> bool:
    request = row.get("request")
    return isinstance(request, dict) and "answer_id" in request


def _parse_jsonl(raw: bytes, relative: str) -> tuple[DecisionRecord, ...]:
    rows: list[DecisionRecord] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(
                f"pinned data file contains a blank record line: {relative}:{line_number}"
            )
        try:
            value = mixture_training_contracts.strict_json_loads(line)
            record = DecisionRecord.model_validate(value)
        except Exception as exc:
            raise ValueError(
                f"pinned data file has a malformed record: {relative}:{line_number}"
            ) from exc
        rows.append(record)
    if not rows:
        raise ValueError(f"pinned data file contains no records: {relative}")
    return tuple(rows)


def _merge_hashes(destination: dict[str, str], source: Mapping[str, object]) -> None:
    for relative, digest in source.items():
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not _SHA256_PATTERN.fullmatch(digest)
        ):
            raise ValueError(
                "verified file fingerprints must map paths to lowercase SHA-256 values"
            )
        existing = destination.get(relative)
        if existing is not None and existing != digest:
            raise ValueError(f"conflicting verified file fingerprints: {relative}")
        destination[relative] = digest


def _historical_source_hashes(root: Path, file_hashes: dict[str, str]) -> dict[str, str]:
    summary_raw, summary_hash = _verified_bytes(
        root, _TRANSFER_SUMMARY_PATH, _TRANSFER_SUMMARY_SHA256
    )
    summary = mixture_training_contracts.strict_json_loads(summary_raw)
    if not isinstance(summary, dict):
        raise ValueError("pinned adapter-transfer summary must be a JSON object")
    pins = summary.get("pins")
    verification = summary.get("verification")
    source_hashes = pins.get("candidate_source_file_sha256") if isinstance(pins, dict) else None
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != set(adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS)
        or len(source_hashes) != 60
        or not isinstance(verification, dict)
        or verification.get("candidate_source_file_count") != 60
    ):
        raise ValueError(
            "pinned adapter-transfer summary does not contain the exact 60-file source map"
        )

    verified: dict[str, str] = {}
    for relative in adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS:
        expected = source_hashes[relative]
        if not isinstance(expected, str) or not _SHA256_PATTERN.fullmatch(expected):
            raise ValueError(f"pinned historical source hash is malformed: {relative}")
        _raw, digest = _verified_bytes(root, relative, expected)
        verified[relative] = digest
    _merge_hashes(file_hashes, {_TRANSFER_SUMMARY_PATH: summary_hash, **verified})
    return verified


def _validate_cross_bundle_disjointness(
    training: Sequence[DecisionRecord],
    evaluation: Sequence[DecisionRecord],
    transfer_presentations: Sequence[Mapping[str, object]],
) -> None:
    mixture_training_data._validate_disjoint_pools((training, evaluation))
    training_ids = {record.record_id for record in training}
    training_groups = {record.source_group_id for record in training}
    training_requests = {record.request.request_hash for record in training}
    evaluation_ids = {record.record_id for record in evaluation}
    evaluation_groups = {record.source_group_id for record in evaluation}
    evaluation_requests = {record.request.request_hash for record in evaluation}
    alias_groups = {
        str(row["source_group_id"])
        for row in transfer_presentations
        if row["dataset_id"] in {"boolq-dev-pilot-v1", "copa-dev-pilot-v1"}
    }
    if training_ids & evaluation_ids:
        raise ValueError("training and evaluation record IDs overlap")
    if training_groups & evaluation_groups:
        raise ValueError("training and evaluation raw source groups overlap")
    if training_groups & alias_groups or evaluation_groups & alias_groups:
        raise ValueError("aliased transfer source groups overlap another study pool")
    if training_requests & evaluation_requests:
        raise ValueError("training and evaluation semantic requests overlap")


def _unique_evaluation_records(
    presentations: Sequence[Mapping[str, object]],
    source_records: Sequence[DecisionRecord],
) -> tuple[DecisionRecord, ...]:
    by_id: dict[str, DecisionRecord] = {}
    for record in source_records:
        if record.record_id in by_id:
            raise ValueError("evaluation source records contain duplicate record IDs")
        by_id[record.record_id] = record
    result: list[DecisionRecord] = []
    seen: set[str] = set()
    for row in presentations:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in by_id:
            raise ValueError("evaluation panel references a record outside the approved inputs")
        if record_id not in seen:
            result.append(by_id[record_id])
            seen.add(record_id)
    if seen != set(by_id):
        raise ValueError("evaluation records do not match final panel membership")
    return tuple(result)


def load_study_inputs(root: str | Path) -> StudyInputs:
    """Load and verify all fixed CPU inputs without opening sealed new splits."""
    try:
        project_root = Path(root).resolve(strict=True)
    except OSError as exc:
        raise ValueError("study project root is unavailable") from exc
    if not project_root.is_dir():
        raise ValueError("study project root is not a directory")

    file_hashes: dict[str, str] = {}
    study_protocol_raw, study_protocol_hash = _verified_bytes(
        project_root, _STUDY_PROTOCOL_PATH, _STUDY_PROTOCOL_SHA256
    )
    del study_protocol_raw
    file_hashes[_STUDY_PROTOCOL_PATH] = study_protocol_hash
    historical_sources = _historical_source_hashes(project_root, file_hashes)

    new_file_bytes: dict[str, bytes] = {}
    for relative, expected in _NEW_FILE_SHA256.items():
        raw, digest = _verified_bytes(project_root, relative, expected)
        new_file_bytes[relative] = raw
        file_hashes[relative] = digest
    new_records = {
        relative: _parse_jsonl(raw, relative) for relative, raw in new_file_bytes.items()
    }
    routing_train = new_records["data/processed/routing-data-v1/train.jsonl"]
    routing_development = new_records["data/processed/routing-data-v1/development.jsonl"]
    tool_train = new_records["data/processed/tool-data-v1/train.jsonl"]
    tool_development = new_records["data/processed/tool-data-v1/development.jsonl"]

    real, balanced, synthetic, snli_train, old_pins_raw = (
        mixture_training_inputs.load_natural_reasoning_data(project_root)
    )
    old_pins = mixture_training_contracts.validate_pins(old_pins_raw)
    old_data_hashes = old_pins["data_file_sha256"]
    old_source_hashes = old_pins["source_file_sha256"]
    if not isinstance(old_data_hashes, dict) or not isinstance(old_source_hashes, dict):
        raise ValueError("verified natural-reasoning fingerprints have an unexpected schema")
    if any(historical_sources.get(path) != digest for path, digest in old_source_hashes.items()):
        raise ValueError(
            "natural-reasoning source fingerprints differ from the historical source map"
        )
    _merge_hashes(file_hashes, old_data_hashes)
    _merge_hashes(file_hashes, old_source_hashes)
    _merge_hashes(
        file_hashes,
        {mixture_training_contracts.PROTOCOL_PATH: old_pins["protocol_sha256"]},
    )

    transfer_records, transfer_presentations, selection, transfer_pins = (
        adapter_transfer_data.load_inputs(project_root)
    )
    transfer_data_hashes = transfer_pins.get("data_file_sha256")
    transfer_training_hashes = transfer_pins.get("training_data_file_sha256")
    transfer_source_hashes = transfer_pins.get("source_file_sha256")
    selection_hash = transfer_pins.get("selection_file_sha256")
    transfer_protocol_hash = transfer_pins.get("protocol_sha256")
    if (
        not isinstance(transfer_data_hashes, dict)
        or not isinstance(transfer_training_hashes, dict)
        or not isinstance(transfer_source_hashes, dict)
        or not isinstance(selection_hash, str)
        or not isinstance(transfer_protocol_hash, str)
    ):
        raise ValueError("verified adapter-transfer fingerprints have an unexpected schema")
    if transfer_source_hashes != historical_sources:
        raise ValueError(
            "adapter-transfer source fingerprints differ from the pinned historical map"
        )
    if transfer_training_hashes != old_data_hashes:
        raise ValueError("adapter-transfer natural-reasoning pins differ from verified old inputs")
    _merge_hashes(file_hashes, transfer_data_hashes)
    _merge_hashes(
        file_hashes,
        {
            adapter_transfer_contracts.SELECTION_PATH: selection_hash,
            adapter_transfer_contracts.PROTOCOL_PATH: transfer_protocol_hash,
            mixture_training_contracts.PROTOCOL_PATH: transfer_pins.get("training_protocol_sha256"),
        },
    )

    real_train = tuple(
        record
        for record in real
        if record.dataset_id in mixture_training_data.REAL_TRAIN_DATASET_COUNTS
    )
    synthetic_train = tuple(
        record
        for record in synthetic
        if record.dataset_id in mixture_training_data.SYNTHETIC_TRAIN_DATASET_COUNTS
    )
    training_pools = (real_train, synthetic_train, snli_train, routing_train, tool_train)
    schedules = runtime_rule_study_data.build_training_schedules(*training_pools)
    runtime_rule_study_data.audit_training_schedules(schedules, *training_pools)

    new_presentations = runtime_rule_study_data.build_new_evaluation_presentations(
        routing_development, tool_development
    )
    runtime_rule_study_data.audit_new_evaluation_presentations(
        new_presentations, routing_development, tool_development
    )
    retention_presentations = build_retention_presentations(
        real, balanced, synthetic, transfer_records
    )
    transfer_rows = tuple(
        row
        for row in retention_presentations
        if row["dataset_id"] in {"boolq-dev-pilot-v1", "copa-dev-pilot-v1"}
    )

    evaluation_sources = tuple(
        record
        for record in (
            *real,
            *balanced,
            *synthetic,
            *transfer_records,
            *routing_development,
            *tool_development,
        )
        if record.dataset_id in _OLD_RETENTION_DATASETS
        or record.dataset_id in {"boolq-dev-pilot-v1", "copa-dev-pilot-v1"}
        or record.dataset_id in runtime_rule_study_data.NEW_DEVELOPMENT_DATASET_IDS.values()
    )
    final_presentations = (*retention_presentations, *new_presentations)
    evaluation_records = _unique_evaluation_records(final_presentations, evaluation_sources)
    training_records = tuple(record for pool in training_pools for record in pool)
    _validate_cross_bundle_disjointness(training_records, evaluation_records, transfer_rows)
    _merge_hashes(file_hashes, transfer_source_hashes)
    return StudyInputs(
        training_pools=training_pools,
        development_pools=(routing_development, tool_development),
        evaluation_records=evaluation_records,
        retention_presentations=retention_presentations,
        new_presentations=new_presentations,
        selection=selection,
        file_sha256=file_hashes,
    )
