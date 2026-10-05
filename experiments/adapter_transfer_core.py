"""Strict CPU payload validation for the adapter-transfer run."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments import adapter_transfer_contracts as contracts
from experiments.mixture_training_contracts import (
    json_sha256,
    normalize_json_object,
    validate_safe_run_id,
    validate_sha256,
    validate_uuid,
)

_PAYLOAD_FIELDS = {
    "schema_version",
    "experiment_id",
    "run_id",
    "nonce",
    "selection_sha256",
    "selection",
    "pins",
    "presentations",
    "parity_presentations",
    "payload_sha256",
}
_PIN_FIELDS = {
    "data_file_sha256",
    "training_data_file_sha256",
    "panel_sha256",
    "selection_file_sha256",
    "protocol_sha256",
    "training_protocol_sha256",
    "source_file_sha256",
}


def source_fingerprints(root: str | Path | None = None) -> dict[str, str]:
    """Hash only the explicit transfer source allowlist and frozen protocol."""

    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        canonical_root = project_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("adapter-transfer project root is unavailable") from exc
    hashes: dict[str, str] = {}
    for relative in contracts.SOURCE_FINGERPRINT_PATHS:
        try:
            path = (project_root / relative).resolve(strict=True)
            path.relative_to(canonical_root)
            if not path.is_file():
                raise ValueError("source path is not a file")
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise ValueError(f"adapter-transfer source is unavailable: {relative}") from exc
    if hashes.get(contracts.PROTOCOL_PATH) != contracts.EXPECTED_PROTOCOL_SHA256:
        raise ValueError("adapter-transfer protocol source SHA-256 mismatch")
    return hashes


def _selection(value: object) -> dict[str, object]:
    contracts.require_selection_pin()
    return contracts.validate_selection(value)


def _validate_pins(value: object, *, root: str | Path | None) -> dict[str, object]:
    from experiments.mixture_training_contracts import (
        DATA_FILE_SHA256 as TRAINING_DATA_FILE_SHA256,
    )
    from experiments.mixture_training_contracts import (
        EXPECTED_PROTOCOL_SHA256 as TRAINING_PROTOCOL_SHA256,
    )

    pins = normalize_json_object(value, "pins")
    if set(pins) != _PIN_FIELDS:
        raise ValueError("adapter-transfer pins have an unexpected schema")
    data_hashes = pins["data_file_sha256"]
    if not isinstance(data_hashes, Mapping) or dict(data_hashes) != contracts.PANEL_FILE_SHA256:
        raise ValueError("adapter-transfer data file pins differ from the exact panel allowlist")
    training_hashes = pins["training_data_file_sha256"]
    if (
        not isinstance(training_hashes, Mapping)
        or dict(training_hashes) != TRAINING_DATA_FILE_SHA256
    ):
        raise ValueError("adapter-transfer training data pins differ from verified pools")
    expected_panel = {
        "copa": contracts.PANEL_FILE_SHA256[contracts.COPA_RECORDS_PATH],
        "boolq": contracts.BOOLQ_PANEL_SHA256,
    }
    if pins["panel_sha256"] != expected_panel:
        raise ValueError("adapter-transfer panel SHA-256 pins differ from the frozen panels")
    if pins["selection_file_sha256"] != contracts.require_selection_pin():
        raise ValueError("selection file SHA-256 differs from its reviewed pin")
    if pins["protocol_sha256"] != contracts.EXPECTED_PROTOCOL_SHA256:
        raise ValueError("adapter-transfer protocol SHA-256 differs from its frozen pin")
    if pins["training_protocol_sha256"] != TRAINING_PROTOCOL_SHA256:
        raise ValueError("adapter training protocol SHA-256 differs from its frozen pin")
    source_hashes = pins["source_file_sha256"]
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != set(
        contracts.SOURCE_FINGERPRINT_PATHS
    ):
        raise ValueError("adapter-transfer source fingerprints differ from the exact allowlist")
    for path, digest in source_hashes.items():
        validate_sha256(digest, f"source fingerprint for {path}")
    if source_hashes.get(contracts.PROTOCOL_PATH) != contracts.EXPECTED_PROTOCOL_SHA256:
        raise ValueError("adapter-transfer protocol and source hashes do not match")
    actual_sources = source_fingerprints(root)
    if dict(source_hashes) != actual_sources:
        raise ValueError("adapter-transfer source fingerprints differ from local source files")
    return {
        "data_file_sha256": dict(data_hashes),
        "training_data_file_sha256": dict(training_hashes),
        "panel_sha256": expected_panel,
        "selection_file_sha256": contracts.require_selection_pin(),
        "protocol_sha256": contracts.EXPECTED_PROTOCOL_SHA256,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA256,
        "source_file_sha256": dict(source_hashes),
    }


def build_payload(
    run_id: str,
    nonce: str,
    presentations: object,
    selection: object,
    pins: object,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the label-free, immutable request payload for the remote worker."""

    from experiments.adapter_transfer_data import _validate_presentation_panel, parity_presentations

    normalized_selection = _selection(selection)
    normalized_pins = _validate_pins(pins, root=root)
    selection_sha256 = json_sha256(normalized_selection)
    validate_safe_run_id(run_id)
    validate_uuid(nonce, "nonce")
    normalized_presentations = _validate_presentation_panel(
        presentations if isinstance(presentations, list) else list(presentations)
    )
    parity = parity_presentations(normalized_presentations)
    unsigned: dict[str, object] = {
        "schema_version": contracts.SCHEMA_VERSION,
        "experiment_id": contracts.EXPERIMENT_ID,
        "run_id": run_id,
        "nonce": nonce,
        "selection_sha256": selection_sha256,
        "selection": normalized_selection,
        "pins": normalized_pins,
        "presentations": normalized_presentations,
        "parity_presentations": parity,
    }
    payload = {**unsigned, "payload_sha256": json_sha256(unsigned)}
    return validate_payload(payload, root=root)


def validate_payload(value: object, *, root: str | Path | None = None) -> dict[str, Any]:
    """Strictly validate payload schema, hashes, sources, panels, and bounds."""

    from experiments.adapter_transfer_data import _validate_presentation_panel, parity_presentations

    contracts.require_selection_pin()
    payload = normalize_json_object(value, "payload")
    if set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("payload has an unexpected top-level schema")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != contracts.SCHEMA_VERSION
    ):
        raise ValueError("payload schema version is unsupported")
    if payload["experiment_id"] != contracts.EXPERIMENT_ID:
        raise ValueError("payload experiment_id is unsupported")
    validate_safe_run_id(payload["run_id"])
    validate_uuid(payload["nonce"], "payload nonce")
    selection = _selection(payload["selection"])
    selection_sha = validate_sha256(payload["selection_sha256"], "payload selection_sha256")
    if selection_sha != json_sha256(selection):
        raise ValueError("payload selection SHA-256 does not match canonical selection")
    pins = _validate_pins(payload["pins"], root=root)
    presentations = _validate_presentation_panel(payload["presentations"])
    parity = parity_presentations(presentations)
    if payload["parity_presentations"] != parity:
        raise ValueError("payload parity panel differs from the declared first-four selection")
    digest = validate_sha256(payload["payload_sha256"], "payload_sha256")
    unsigned = {key: row for key, row in payload.items() if key != "payload_sha256"}
    if json_sha256(unsigned) != digest:
        raise ValueError("payload SHA-256 does not match its canonical fields")
    payload["selection"] = selection
    payload["selection_sha256"] = selection_sha
    payload["pins"] = pins
    payload["presentations"] = presentations
    payload["parity_presentations"] = parity
    return payload


def validate_result(
    value: object, payload: object, *, root: str | Path | None = None
) -> dict[str, Any]:
    """Validate a complete result or a bounded failure with honest partial evidence."""

    from experiments.adapter_transfer_results import validate_result as validate

    return validate(value, payload, root=root)


def build_failed_result(
    payload: object,
    *,
    stage: str,
    error: BaseException,
    sanitize: Any,
    provenance: object = None,
    evidence: object = None,
) -> dict[str, Any]:
    """Create a failed receipt while preserving bounded known partial evidence."""

    from experiments.adapter_transfer_results import build_failed_result as build

    return build(
        payload,
        stage=stage,
        error=error,
        sanitize=sanitize,
        provenance=provenance,
        evidence=evidence,
    )
