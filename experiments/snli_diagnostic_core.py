"""CPU-side identity, payload, worker-result, and receipt contracts for balanced SNLI."""

from __future__ import annotations

import itertools
import json
import math
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from experiments import (
    baseline_intern,
    baseline_kev,
    baseline_qwen,
    real_pilot_baseline_core,
)
from reflex_decisions import snli_diagnostic
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest

SCHEMA_VERSION = 1
PROTOCOL_PATH = "docs/snli-diagnostic-protocol.md"
EXPECTED_PROTOCOL_SHA256 = "e41334b9d2f6b47d057907782c889251f8e952fae9e77d173e1d7bd69062a4ae"
EXPECTED_RECORDS_SHA256 = "a1fa41d19b381e227ebca258b1561e7e39184f6f58a325aaa5a2b1ff61ddd98c"
EXPECTED_MANIFEST_SHA256 = "4ecab09c6f013c274b38a81c6ef226226fee6520745e01457f461c5e5dc628bd"
EXPECTED_RECIPE_SHA256 = "e11fc94b99376f4860caf3b9afbb22026e6015ee7b4bfeb42344a7a0cc0bbd58"
RECORD_COUNT = 192
GROUP_COUNT = 64
ORDERS_PER_RECORD = math.factorial(3)
PRESENTATION_COUNT = RECORD_COUNT * ORDERS_PER_RECORD
MAX_INPUT_TOKENS = 2048
MODEL_NAMES = ("qwen_base", "qwen_final", "intern", "kev")
WORKERS = ("qwen", "intern", "kev")
WORKER_MODELS = {"qwen": ("qwen_base", "qwen_final"), "intern": ("intern",), "kev": ("kev",)}
MODEL_PINS = {
    "qwen_base": {
        "model_id": baseline_qwen.MODEL_ID,
        "model_revision": baseline_qwen.MODEL_REVISION,
    },
    "qwen_final": {
        "model_id": baseline_qwen.MODEL_ID,
        "model_revision": baseline_qwen.MODEL_REVISION,
    },
    "intern": {"model_id": baseline_intern.MODEL_ID, "model_revision": baseline_intern.REVISION},
    "kev": {"model_id": baseline_kev.MODEL_ID, "model_revision": baseline_kev.REVISION},
}
AUXILIARY_FORWARD_COUNTS = {"qwen_base": 0, "qwen_final": 0, "intern": 1, "kev": 2}
RUNTIME_VERSION_PINS = {
    "torch": "2.14.1+cu130",
    "torchvision": "0.29.1+cu130",
    "transformers": "5.18.0",
    "peft": "0.21.0",
    "Pillow": "12.0.0",
    "pydantic": "2.13.5",
    "huggingface-hub": "1.33.0",
    "tokenizers": "0.23.2",
    "safetensors": "0.8.0",
}
QWEN_ADAPTER_FILE_SHA256 = {
    "adapter_config.json": "fa6fdf55295985c39b5cfd6f6dfb66ab3c941756429cd402f77e1ac455fdea8d",
    "adapter_model.safetensors": "315c23b1517c9590386d22afad5fac2927f97c31f42cdfa73b2c921494266fd8",
}
MODAL_LIMITS: dict[str, object] = {
    "gpu": "A10",
    "cpu_physical_cores": [2.0, 2.0],
    "memory_mib": [16384, 16384],
    "max_containers": 3,
    "min_containers": 0,
    "buffer_containers": 0,
    "scaledown_window_seconds": 2,
    "retries": 0,
    "single_use_containers": True,
    "startup_timeout_seconds": 300,
    "timeout_seconds": 1800,
    "max_scored_forwards_per_model": PRESENTATION_COUNT,
    "auxiliary_forwards": dict(AUXILIARY_FORWARD_COUNTS),
    "max_total_forwards": 4611,
    "volume": "reflex-rehearsal-artifacts",
    "volume_read_only": True,
    "endpoint": None,
    "schedule": None,
    "warm_pool": None,
}
SOURCE_FINGERPRINT_PATHS = tuple(
    dict.fromkeys(
        (
            *real_pilot_baseline_core.SOURCE_FINGERPRINT_PATHS,
            "experiments/snli_diagnostic_core.py",
            "experiments/modal_snli_diagnostic.py",
            "experiments/snli_diagnostic_runtime.py",
            "experiments/snli_diagnostic_results.py",
            "src/reflex_decisions/snli_diagnostic.py",
            "src/reflex_decisions/pilot_data_sources.py",
            PROTOCOL_PATH,
        )
    )
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PIN_FIELDS = {
    "records_sha256",
    "manifest_sha256",
    "recipe_sha256",
    "protocol_sha256",
    "source_file_sha256",
}
_PAYLOAD_FIELDS = {"schema_version", "run_id", "nonce", "pins", "presentations", "panel_sha256"}
_PRESENTATION_FIELDS = {
    "presentation_id",
    "record_id",
    "source_group_id",
    "request_hash",
    "order_index",
    "order_ids",
    "request",
}


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("value is not strict JSON") from exc


def _json_sha256(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _normalize_object(value: object, label: str) -> dict[str, object]:
    try:
        normalized = json.loads(_canonical_json(value))
    except ValueError as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be an object")
    return normalized


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUID")
    return value


def _validate_pins(value: object) -> dict[str, object]:
    pins = _normalize_object(value, "payload pins")
    if set(pins) != _PIN_FIELDS:
        raise ValueError("payload pins have an unexpected schema")
    expected = {
        "records_sha256": EXPECTED_RECORDS_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "recipe_sha256": EXPECTED_RECIPE_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }
    for name, digest in expected.items():
        if _digest(pins[name], name) != digest:
            raise ValueError(f"{name} differs from its frozen pin")
    sources = pins["source_file_sha256"]
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_FINGERPRINT_PATHS):
        raise ValueError("source fingerprints do not match the explicit allowlist")
    for path, digest in sources.items():
        _digest(digest, f"source fingerprint for {path}")
    if sources.get(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("protocol pin and source fingerprint do not match")
    return pins


def verify_protocol(path: str | Path = PROTOCOL_PATH) -> str:
    """Verify and return the frozen protocol document digest."""

    try:
        digest = sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("balanced SNLI protocol is unavailable") from exc
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("balanced SNLI protocol SHA-256 mismatch")
    return digest


def source_fingerprints(root: str | Path | None = None) -> dict[str, str]:
    """Hash every allowlisted local/remote Python dependency and the protocol."""

    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    paths = {relative: project_root / relative for relative in SOURCE_FINGERPRINT_PATHS}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("a required source fingerprint path is missing")
    return {relative: sha256(path.read_bytes()).hexdigest() for relative, path in paths.items()}


def _validate_records(records: Sequence[DecisionRecord]) -> tuple[DecisionRecord, ...]:
    if len(records) != RECORD_COUNT or any(not isinstance(row, DecisionRecord) for row in records):
        raise ValueError("balanced SNLI data must contain exactly 192 decision records")
    labels = set(snli_diagnostic.SNLI_LABELS)
    record_ids: set[str] = set()
    request_hashes: set[str] = set()
    groups: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        if (
            record.dataset_id != snli_diagnostic.DATASET_ID
            or not record.record_id.startswith(f"{snli_diagnostic.DATASET_ID}-")
            or _HEX.fullmatch(record.source_group_id) is None
            or tuple(option.id for option in record.request.options) != snli_diagnostic.SNLI_LABELS
            or record.answer_id not in labels
        ):
            raise ValueError("record identity or approved SNLI schema is invalid")
        if record.record_id in record_ids or record.request.request_hash in request_hashes:
            raise ValueError("balanced SNLI record IDs and requests must be unique")
        record_ids.add(record.record_id)
        request_hashes.add(record.request.request_hash)
        groups[record.source_group_id].append(record)
    if len(groups) != GROUP_COUNT or any(
        len(rows) != 3
        or Counter(row.answer_id for row in rows) != Counter(snli_diagnostic.SNLI_LABELS)
        for rows in groups.values()
    ):
        raise ValueError("balanced SNLI data must have one record per label in 64 groups")
    return tuple(records)


def build_payload(
    records: Sequence[DecisionRecord], *, pins: Mapping[str, object], run_id: str, nonce: str
) -> dict[str, object]:
    """Build the exact request-only six-order panel from the frozen local records."""

    rows = _validate_records(records)
    run_id, nonce = _uuid(run_id, "run_id"), _uuid(nonce, "nonce")
    if run_id == nonce:
        raise ValueError("run_id and nonce must be independently generated")
    normalized_pins = _validate_pins(pins)
    presentations: list[dict[str, object]] = []
    for record in rows:
        for order_index, options in enumerate(itertools.permutations(record.request.options)):
            request = record.request.model_copy(update={"options": options})
            presentations.append(
                {
                    "presentation_id": f"{record.record_id}-order-{order_index}",
                    "record_id": record.record_id,
                    "source_group_id": record.source_group_id,
                    "request_hash": request.request_hash,
                    "order_index": order_index,
                    "order_ids": [option.id for option in options],
                    "request": request.model_dump(mode="json"),
                }
            )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "nonce": nonce,
        "pins": normalized_pins,
        "presentations": presentations,
        "panel_sha256": _json_sha256(presentations),
    }
    return validate_payload(payload)


def validate_payload(value: object) -> dict[str, object]:
    """Normalize and validate exact record membership and all six semantic orders."""

    payload = _normalize_object(value, "payload")
    if set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("payload has an unexpected schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("payload schema version is unsupported")
    run_id, nonce = _uuid(payload["run_id"], "run_id"), _uuid(payload["nonce"], "nonce")
    if run_id == nonce:
        raise ValueError("run_id and nonce must be independently generated")
    pins = _validate_pins(payload["pins"])
    presentations = payload["presentations"]
    if not isinstance(presentations, list) or len(presentations) != PRESENTATION_COUNT:
        raise ValueError("payload must contain exactly 1,152 presentations")
    label_ids = set(snli_diagnostic.SNLI_LABELS)
    by_record: dict[str, list[dict[str, object]]] = defaultdict(list)
    group_by_record: dict[str, str] = {}
    group_records: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(presentations):
        if not isinstance(row, dict) or set(row) != _PRESENTATION_FIELDS:
            raise ValueError("payload presentation has an unexpected schema")
        record_id, group_id = row["record_id"], row["source_group_id"]
        order = row["order_index"]
        if (
            not isinstance(record_id, str)
            or not record_id.startswith(f"{snli_diagnostic.DATASET_ID}-")
            or not isinstance(group_id, str)
            or _HEX.fullmatch(group_id) is None
            or type(order) is not int
            or order != index % ORDERS_PER_RECORD
            or index // ORDERS_PER_RECORD >= RECORD_COUNT
        ):
            raise ValueError("payload presentation membership or order is invalid")
        first_row = presentations[index - (index % ORDERS_PER_RECORD)]
        if index % ORDERS_PER_RECORD and row["record_id"] != first_row["record_id"]:
            raise ValueError("payload presentations must stay grouped by record")
        if index % ORDERS_PER_RECORD == 0 and row["presentation_id"] != f"{record_id}-order-0":
            raise ValueError("payload presentation IDs do not follow record order")
        if index % ORDERS_PER_RECORD and row["presentation_id"] != f"{record_id}-order-{order}":
            raise ValueError("payload presentation IDs do not match their order index")
        ids = row["order_ids"]
        if (
            not isinstance(ids, list)
            or len(ids) != 3
            or set(ids) != label_ids
            or len(set(ids)) != 3
        ):
            raise ValueError("presentation must contain the three approved option IDs")
        request = DecisionRequest.model_validate(row["request"])
        if (
            [option.id for option in request.options] != ids
            or request.request_hash != row["request_hash"]
            or not isinstance(row["request_hash"], str)
            or _HEX.fullmatch(row["request_hash"]) is None
        ):
            raise ValueError("presentation request, order, and semantic hash disagree")
        if record_id in group_by_record and group_by_record[record_id] != group_id:
            raise ValueError("one record appears under multiple source groups")
        group_by_record[record_id] = group_id
        group_records[group_id].add(record_id)
        by_record[record_id].append(row)
    if (
        len(by_record) != RECORD_COUNT
        or len(group_records) != GROUP_COUNT
        or any(len(records) != 3 for records in group_records.values())
    ):
        raise ValueError("payload record and source-group counts differ from the frozen panel")
    for record_rows in by_record.values():
        if len(record_rows) != ORDERS_PER_RECORD:
            raise ValueError("payload record is missing one or more option orders")
        requests = [DecisionRequest.model_validate(row["request"]) for row in record_rows]
        base = requests[0]
        if [tuple(option.id for option in request.options) for request in requests] != list(
            itertools.permutations(tuple(option.id for option in base.options))
        ) or any(
            request.request_hash != base.request_hash
            or request.context != base.context
            or request.question != base.question
            or {option.id: option.model_dump(mode="json") for option in request.options}
            != {option.id: option.model_dump(mode="json") for option in base.options}
            for request in requests
        ):
            raise ValueError("payload orders do not preserve the original request and options")
    if payload["panel_sha256"] != _json_sha256(presentations):
        raise ValueError("payload panel digest does not match its presentations")
    payload["pins"] = pins
    return payload


def validate_model_result(result: object, payload: object, model_name: str) -> dict[str, object]:
    """Validate one result through the separate result/provenance contract module."""

    from experiments.snli_diagnostic_results import validate_model_result as validate

    return validate(result, payload, model_name)


def validate_receipt(receipt: object, payload: object) -> dict[str, object]:
    """Validate a saved envelope while retaining successful sibling model states."""

    from experiments.snli_diagnostic_results import validate_receipt as validate

    return validate(receipt, payload)
