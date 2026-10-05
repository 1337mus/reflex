"""CPU contracts for paired Intern/Kev references to the real pilot."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from experiments import baseline_intern, baseline_kev, baseline_runner_core, real_pilot_core
from reflex_decisions import pilot_data
from reflex_decisions.data import DecisionRecord, SplitManifest

SCHEMA_VERSION = 1
REFERENCE_PROTOCOL_PATH = "docs/real-pilot-baselines-protocol.md"
EXPECTED_REFERENCE_PROTOCOL_SHA256 = (
    "84501b1cc9d13feea0ed0002a42605873147a50fa9fd737e25fb3f76d7cf4b21"
)
EXPECTED_PROTOCOL_SHA256 = real_pilot_core.EXPECTED_PROTOCOL_SHA256
EVALUATION_PRESENTATION_COUNT = real_pilot_core.EVALUATION_PRESENTATION_COUNT
MAX_INPUT_TOKENS = real_pilot_core.MAX_INPUT_TOKENS
MODEL_PINS: dict[str, dict[str, str]] = {
    "intern": {"model_id": baseline_intern.MODEL_ID, "model_revision": baseline_intern.REVISION},
    "kev": {"model_id": baseline_kev.MODEL_ID, "model_revision": baseline_kev.REVISION},
}
AUXILIARY_FORWARD_COUNTS = {
    "intern": baseline_intern.AUXILIARY_FORWARD_COUNT,
    "kev": baseline_kev.AUXILIARY_FORWARD_COUNT,
}
PROMPT_HASH_KINDS = {
    "intern": "rendered_chat_utf8_sha256",
    "kev": "canonical_token_id_array_sha256",
}
MODAL_LIMITS: dict[str, object] = {
    "gpu": "A10",
    "cpu_physical_cores": [2.0, 2.0],
    "memory_mib": [16384, 16384],
    "max_containers": 2,
    "min_containers": 0,
    "buffer_containers": 0,
    "retries": 0,
    "single_use_containers": True,
    "startup_timeout_seconds": 300,
    "timeout_seconds": 3600,
    "max_scored_forwards_per_model": EVALUATION_PRESENTATION_COUNT,
    "auxiliary_forwards": dict(AUXILIARY_FORWARD_COUNTS),
    "max_total_forwards": 5147,
    "volume": None,
    "endpoint": None,
    "schedule": None,
    "warm_pool": None,
}

SOURCE_FINGERPRINT_MODULES = tuple(
    dict.fromkeys(
        (
            *real_pilot_core.SOURCE_FINGERPRINT_MODULES,
            ("experiments.baseline_runner_core", "experiments/baseline_runner_core.py"),
            ("experiments.baseline_intern", "experiments/baseline_intern.py"),
            ("experiments.baseline_kev", "experiments/baseline_kev.py"),
            ("experiments.modal_train_rehearsal", "experiments/modal_train_rehearsal.py"),
            ("experiments.real_pilot_baseline_core", "experiments/real_pilot_baseline_core.py"),
            ("experiments.modal_real_pilot_baselines", "experiments/modal_real_pilot_baselines.py"),
        )
    )
)
SOURCE_FINGERPRINT_PATHS = tuple(
    dict.fromkeys(
        (
            *(path for _module, path in SOURCE_FINGERPRINT_MODULES),
            real_pilot_core.PROTOCOL_PATH,
            REFERENCE_PROTOCOL_PATH,
        )
    )
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_FIELDS = {
    "schema_version",
    "run_id",
    "nonce",
    "pins",
    "evaluation_presentations",
    "evaluation_payload_sha256",
}
_PIN_FIELDS = {
    "records_sha256",
    "manifest_sha256",
    "recipe_sha256",
    "protocol_sha256",
    "reference_protocol_sha256",
    "source_file_sha256",
}
_MODEL_RECEIPT_FIELDS = {
    "schema_version",
    "run_id",
    "nonce",
    "pins",
    "model_name",
    "model_id",
    "model_revision",
    "status",
    "forward_counts",
    "prompt_hash_kind",
    "provenance",
    "presentations",
}
_FAILED_MODEL_RECEIPT_FIELDS = _MODEL_RECEIPT_FIELDS | {"failure"}
_OUTPUT_FIELDS = {
    "presentation_id",
    "record_id",
    "dataset_id",
    "request_hash",
    "order_index",
    "order_ids",
    "candidate_logits",
    "winner_option_id",
    "input_tokens",
    "prompt_sha256",
}
_PINNED_RUNTIME_VERSIONS = {
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


def source_fingerprints(root: str | Path | None = None) -> dict[str, str]:
    """Hash the complete original pilot source set and the new reference sources."""

    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    paths = [project_root / relative for relative in SOURCE_FINGERPRINT_PATHS]
    if any(not path.is_file() for path in paths):
        raise ValueError("a required reference source file or protocol is missing")
    return {
        str(path.relative_to(project_root)): sha256(path.read_bytes()).hexdigest() for path in paths
    }


def verify_reference_protocol(path: str | Path = REFERENCE_PROTOCOL_PATH) -> str:
    """Return the frozen reference protocol digest or reject a changed document."""

    try:
        digest = sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("reference baseline protocol is unavailable") from exc
    if digest != EXPECTED_REFERENCE_PROTOCOL_SHA256:
        raise ValueError("reference baseline protocol SHA-256 mismatch")
    return digest


def _canonical_json(value: object) -> bytes:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("value is not strict JSON") from exc
    return serialized.encode("utf-8")


def _json_sha256(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_run_id(value: object) -> str:
    """Require a canonical UUID for each independently reserved run."""

    if not isinstance(value, str):
        raise ValueError("run_id must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("run_id must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError("run_id must be a canonical UUID")
    return value


def validate_nonce(value: object) -> str:
    """Require a canonical, separately generated nonce."""

    return real_pilot_core.validate_nonce(value)


def _validate_pins(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _PIN_FIELDS:
        raise ValueError("reference payload pins have an unexpected schema")
    for name in ("records_sha256", "manifest_sha256", "recipe_sha256"):
        _validate_sha256(value[name], name)
    if value["records_sha256"] != pilot_data.EXPECTED_RECORDS_SHA256:
        raise ValueError("reference records pin differs from the original pilot")
    if value["manifest_sha256"] != pilot_data.EXPECTED_MANIFEST_SHA256:
        raise ValueError("reference manifest pin differs from the original pilot")
    if value["recipe_sha256"] != pilot_data.EXPECTED_RECIPE_SHA256:
        raise ValueError("reference recipe pin differs from the original pilot")
    if value["protocol_sha256"] != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("reference pilot protocol pin differs from the original pilot")
    if value["reference_protocol_sha256"] != EXPECTED_REFERENCE_PROTOCOL_SHA256:
        raise ValueError("reference protocol pin differs from the frozen document")
    source_hashes = value["source_file_sha256"]
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != set(SOURCE_FINGERPRINT_PATHS)
        or any(
            not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None
            for digest in source_hashes.values()
        )
    ):
        raise ValueError("reference source file fingerprints are malformed")
    if (
        source_hashes.get(real_pilot_core.PROTOCOL_PATH) != value["protocol_sha256"]
        or source_hashes.get(REFERENCE_PROTOCOL_PATH) != value["reference_protocol_sha256"]
    ):
        raise ValueError("protocol pins and source fingerprints do not match")
    return dict(value)


def _normalize_payload(value: object) -> dict[str, object]:
    try:
        normalized = json.loads(_canonical_json(value))
    except ValueError as exc:
        raise ValueError("reference payload is not strict JSON") from exc
    if not isinstance(normalized, dict) or set(normalized) != _PAYLOAD_FIELDS:
        raise ValueError("reference payload has an unexpected schema")
    return normalized


def validate_remote_payload(
    value: object, *, expected_identity: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Validate exact request-only evaluation rows before any model is loaded."""

    payload = _normalize_payload(value)
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("reference payload schema version is unsupported")
    run_id = validate_run_id(payload["run_id"])
    nonce = validate_nonce(payload["nonce"])
    if run_id == nonce:
        raise ValueError("run_id and nonce must be independently generated")
    pins = _validate_pins(payload["pins"])
    rows = payload["evaluation_presentations"]
    if not isinstance(rows, list):
        raise ValueError("reference payload presentations must be an array")
    from experiments import real_pilot_contracts

    normalized_rows = real_pilot_contracts._validate_evaluation_payload(rows)
    expected_digest = _json_sha256(normalized_rows)
    if payload["evaluation_payload_sha256"] != expected_digest:
        raise ValueError("reference evaluation panel digest does not match its rows")
    if expected_identity is not None:
        expected_run_id = validate_run_id(expected_identity.get("run_id"))
        expected_nonce = validate_nonce(expected_identity.get("nonce"))
        expected_pins = _validate_pins(expected_identity.get("pins"))
        if run_id != expected_run_id or nonce != expected_nonce or pins != expected_pins:
            raise ValueError("reference payload identity or pins differ from the local launch")
    payload["pins"] = pins
    payload["evaluation_presentations"] = normalized_rows
    return payload


def build_evaluation_payload(
    manifest: SplitManifest,
    records: Sequence[DecisionRecord],
    *,
    run_id: str,
    nonce: str,
    records_sha256: str,
    manifest_sha256: str,
    recipe_sha256: str,
    protocol_sha256: str,
    reference_protocol_sha256: str,
    source_file_sha256: dict[str, str],
) -> dict[str, object]:
    """Build an exact, request-only panel after validating the frozen pilot allocation."""

    real_pilot_core.validate_pilot_data(manifest, records)
    run_id = validate_run_id(run_id)
    nonce = validate_nonce(nonce)
    if run_id == nonce:
        raise ValueError("run_id and nonce must be independently generated")
    pins = _validate_pins(
        {
            "records_sha256": records_sha256,
            "manifest_sha256": manifest_sha256,
            "recipe_sha256": recipe_sha256,
            "protocol_sha256": protocol_sha256,
            "reference_protocol_sha256": reference_protocol_sha256,
            "source_file_sha256": source_file_sha256,
        }
    )
    rows = list(real_pilot_core.build_evaluation_presentations(records))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "nonce": nonce,
        "pins": pins,
        "evaluation_presentations": rows,
        "evaluation_payload_sha256": _json_sha256(rows),
    }
    identity = {"run_id": run_id, "nonce": nonce, "pins": pins}
    return validate_remote_payload(payload, expected_identity=identity)


def _normalize_object(value: object, label: str) -> dict[str, object]:
    try:
        normalized = json.loads(_canonical_json(value))
    except ValueError as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _expected_identity(payload: Mapping[str, object]) -> tuple[str, str, dict[str, object]]:
    run_id = validate_run_id(payload.get("run_id"))
    nonce = validate_nonce(payload.get("nonce"))
    pins = _validate_pins(payload.get("pins"))
    if run_id == nonce:
        raise ValueError("run_id and nonce must be independently generated")
    return run_id, nonce, pins


def _validate_runtime_versions(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "Pillow",
        "pydantic",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
    }:
        raise ValueError("reference runtime versions have an unexpected schema")
    normalized: dict[str, str] = {}
    for package, version in value.items():
        if not isinstance(version, str) or not version.strip():
            raise ValueError("reference runtime versions must be nonblank strings")
        normalized[package] = version
    if any(normalized[name] != version for name, version in _PINNED_RUNTIME_VERSIONS.items()):
        raise ValueError("reference runtime versions differ from the frozen Modal image")
    return normalized


def _validate_scorer_identity(model_name: str, value: object) -> dict[str, object]:
    if model_name not in MODEL_PINS:
        raise ValueError("model name is outside the reference allowlist")
    adapter = baseline_intern if model_name == "intern" else baseline_kev
    provenance = _normalize_object(value, "adapter provenance")
    baseline_runner_core.validate_adapter_provenance(model_name, adapter, provenance)
    if model_name == "intern":
        if (
            provenance.get("source_commit") != baseline_intern.SOURCE_COMMIT
            or provenance.get("module_sha256") != baseline_intern.MODULE_SHA256
        ):
            raise ValueError("Intern scorer provenance does not match its pinned source")
    else:
        if (
            provenance.get("base_model_id") != baseline_kev.BASE_MODEL_ID
            or provenance.get("base_revision") != baseline_kev.BASE_REVISION
            or provenance.get("source_commit") != baseline_kev.SOURCE_COMMIT
            or provenance.get("source_sha256") != baseline_kev.SOURCE_SHA256
            or provenance.get("safe_checkpoint_module_sha256")
            != baseline_kev.SAFE_CHECKPOINT_SHA256
        ):
            raise ValueError("Kev scorer provenance does not match its pinned source")
    return provenance


def _validate_scorer_provenance(model_name: str, value: object) -> dict[str, object]:
    provenance = _validate_scorer_identity(model_name, value)
    if provenance.get("auxiliary_forward_count") != AUXILIARY_FORWARD_COUNTS[model_name]:
        raise ValueError("adapter auxiliary forwards differ from the frozen scorer path")
    if model_name == "intern":
        if provenance.get("calibration_parity_passed") is not True or not isinstance(
            provenance.get("versions"), dict
        ):
            raise ValueError("Intern scorer provenance does not match its pinned source")
    else:
        parity = provenance.get("temperature_parity_max_abs")
        if (
            isinstance(parity, bool)
            or not isinstance(parity, (int, float))
            or not 0.0 <= float(parity) <= 1e-5
        ):
            raise ValueError("Kev raw-logit temperature parity evidence is invalid")
    return provenance


def _validate_partial_scorer_provenance(model_name: str, value: object) -> dict[str, object]:
    provenance = _validate_scorer_identity(model_name, value)
    if model_name != "intern":
        return _validate_scorer_provenance(model_name, provenance)
    auxiliary = provenance.get("auxiliary_forward_count")
    parity_passed = provenance.get("calibration_parity_passed")
    versions = provenance.get("versions")
    if (
        not isinstance(versions, dict)
        or any(not isinstance(version, str) or not version.strip() for version in versions.values())
        or (parity_passed is False and auxiliary != 0)
        or (parity_passed is True and auxiliary != AUXILIARY_FORWARD_COUNTS[model_name])
        or not isinstance(parity_passed, bool)
    ):
        raise ValueError("partial Intern scorer provenance is inconsistent with observed parity")
    return provenance


def _validate_model_presentations(
    expected_payload: Mapping[str, object], value: object, *, complete: bool
) -> list[dict[str, object]]:
    expected_rows = expected_payload.get("evaluation_presentations")
    if not isinstance(expected_rows, list):
        raise ValueError("expected reference payload has no evaluation panel")
    expected_by_id = {row["presentation_id"]: row for row in expected_rows}
    if not isinstance(value, list) or (complete and len(value) != len(expected_by_id)):
        raise ValueError("reference output does not contain the complete evaluation panel")
    seen: set[str] = set()
    normalized: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != _OUTPUT_FIELDS:
            raise ValueError("reference output row has an unexpected schema")
        presentation_id = row["presentation_id"]
        expected = expected_by_id.get(presentation_id) if isinstance(presentation_id, str) else None
        if expected is None or presentation_id in seen:
            raise ValueError("reference output presentation is unknown or duplicated")
        if type(row["order_index"]) is not int or any(
            row.get(field) != expected.get(field)
            for field in ("record_id", "dataset_id", "request_hash", "order_index", "order_ids")
        ):
            raise ValueError("reference output identity differs from its request-only input")
        order_ids = expected["order_ids"]
        result = baseline_runner_core.validate_presentation_result(
            {
                "record_id": expected["record_id"],
                "request_hash": expected["request_hash"],
                "permutation_index": expected["order_index"],
                "option_ids": order_ids,
            },
            {
                "raw_logits": row["candidate_logits"],
                "input_tokens": row["input_tokens"],
                "prompt_sha256": row["prompt_sha256"],
            },
        )
        logits = result["raw_logits"]
        winner = min(
            option_id
            for option_id, logit in zip(order_ids, logits, strict=True)
            if logit == max(logits)
        )
        if row["winner_option_id"] != winner:
            raise ValueError("reference winner does not match semantic lexical tie-breaking")
        seen.add(presentation_id)
        normalized_row = dict(row)
        normalized_row["candidate_logits"] = logits
        normalized.append(normalized_row)
    if complete and seen != set(expected_by_id):
        raise ValueError("reference outputs omitted an evaluation presentation")
    return normalized


def _validate_model_state(
    value: object,
    *,
    expected_payload: Mapping[str, object],
    model_name: str,
    expected_status: str,
) -> dict[str, object]:
    if model_name not in MODEL_PINS or expected_status not in {"passed", "failed"}:
        raise ValueError("model name or status is outside the reference allowlist")
    state = _normalize_object(value, "reference model receipt")
    expected_fields = (
        _FAILED_MODEL_RECEIPT_FIELDS if expected_status == "failed" else _MODEL_RECEIPT_FIELDS
    )
    if set(state) != expected_fields:
        raise ValueError("reference model receipt has an unexpected schema")
    run_id, nonce, pins = _expected_identity(expected_payload)
    if (
        type(state["schema_version"]) is not int
        or state["schema_version"] != SCHEMA_VERSION
        or state["run_id"] != run_id
        or state["nonce"] != nonce
        or _validate_pins(state["pins"]) != pins
    ):
        raise ValueError("reference model receipt identity or pins differ from launch")
    model = MODEL_PINS[model_name]
    if (
        state["model_name"] != model_name
        or state["model_id"] != model["model_id"]
        or state["model_revision"] != model["model_revision"]
        or state["status"] != expected_status
        or state["prompt_hash_kind"] != PROMPT_HASH_KINDS[model_name]
    ):
        raise ValueError("reference model identity or status differs from its frozen pin")
    outputs = _validate_model_presentations(
        expected_payload, state["presentations"], complete=expected_status == "passed"
    )
    counts = state["forward_counts"]
    if not isinstance(counts, dict) or set(counts) != {"scored", "auxiliary", "total"}:
        raise ValueError("reference model forward counts have an unexpected schema")
    scored, auxiliary, total = counts["scored"], counts["auxiliary"], counts["total"]
    if expected_status == "passed":
        expected_counts = {
            "scored": EVALUATION_PRESENTATION_COUNT,
            "auxiliary": AUXILIARY_FORWARD_COUNTS[model_name],
            "total": EVALUATION_PRESENTATION_COUNT + AUXILIARY_FORWARD_COUNTS[model_name],
        }
        if (
            any(type(counts[key]) is not int for key in expected_counts)
            or counts != expected_counts
        ):
            raise ValueError("reference model forward counts differ from the fixed compute limit")
    elif (
        type(scored) is not int
        or scored != len(outputs)
        or scored > EVALUATION_PRESENTATION_COUNT
        or (auxiliary is None and (total is not None or scored != 0))
        or (
            auxiliary is not None
            and (
                type(auxiliary) is not int
                or auxiliary not in {0, AUXILIARY_FORWARD_COUNTS[model_name]}
                or (auxiliary == 0 and (model_name != "intern" or scored != 0))
                or type(total) is not int
                or total != scored + auxiliary
            )
        )
    ):
        raise ValueError("failed reference forward counts do not match the observed work")

    provenance = _normalize_object(state["provenance"], "reference provenance")
    provenance_fields = {
        "measured_source_file_sha256",
        "scorer",
        "runtime_versions",
        "use_hub_kernels",
        "hf_hub_disable_implicit_token",
    }
    if set(provenance) != provenance_fields:
        raise ValueError("reference provenance has an unexpected schema")
    measured = provenance["measured_source_file_sha256"]
    if expected_status == "passed":
        if measured != pins["source_file_sha256"]:
            raise ValueError("remote measured source files differ from the pinned local sources")
        scorer = _validate_scorer_provenance(model_name, provenance["scorer"])
        versions = _validate_runtime_versions(provenance["runtime_versions"])
        if (
            provenance["use_hub_kernels"] != "NO"
            or provenance["hf_hub_disable_implicit_token"] is not True
        ):
            raise ValueError("reference runtime enabled kernels or implicit Hub credentials")
    else:
        if not isinstance(measured, dict) or any(
            path not in SOURCE_FINGERPRINT_PATHS
            or not isinstance(digest, str)
            or _HEX_SHA256.fullmatch(digest) is None
            for path, digest in measured.items()
        ):
            raise ValueError("failed reference source measurements are malformed")
        scorer_value = _normalize_object(provenance["scorer"], "partial adapter provenance")
        scorer = (
            _validate_partial_scorer_provenance(model_name, scorer_value) if scorer_value else {}
        )
        if (scorer and scorer.get("auxiliary_forward_count") != auxiliary) or (
            not scorer and auxiliary is not None
        ):
            raise ValueError(
                "failed reference forward counts differ from observed scorer provenance"
            )
        versions_value = provenance["runtime_versions"]
        if not isinstance(versions_value, dict) or any(
            not isinstance(version, str) or not version.strip()
            for version in versions_value.values()
        ):
            raise ValueError("partial reference runtime versions are malformed")
        versions = dict(versions_value)

    normalized: dict[str, object] = {
        "model_name": model_name,
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "status": expected_status,
        "forward_counts": {"scored": scored, "auxiliary": auxiliary, "total": total},
        "prompt_hash_kind": PROMPT_HASH_KINDS[model_name],
        "provenance": {
            "measured_source_file_sha256": measured,
            "scorer": scorer,
            "runtime_versions": versions,
            "use_hub_kernels": provenance["use_hub_kernels"],
            "hf_hub_disable_implicit_token": provenance["hf_hub_disable_implicit_token"],
        },
        "presentations": outputs,
    }
    if expected_status == "failed":
        failure = state["failure"]
        if (
            not isinstance(failure, dict)
            or set(failure) != {"stage", "type", "message"}
            or any(
                not isinstance(failure[key], str) or not failure[key].strip()
                for key in ("stage", "type", "message")
            )
        ):
            raise ValueError("failed reference details are malformed")
        normalized["failure"] = dict(failure)
    return normalized


def validate_model_receipt(
    value: object, *, expected_payload: Mapping[str, object], model_name: str
) -> dict[str, object]:
    """Validate one complete worker result and return its analysis-facing model state."""

    return _validate_model_state(
        value, expected_payload=expected_payload, model_name=model_name, expected_status="passed"
    )


def validate_failed_model_receipt(
    value: object, *, expected_payload: Mapping[str, object], model_name: str
) -> dict[str, object]:
    """Retain a matching failed worker state and every valid partial output row."""

    return _validate_model_state(
        value, expected_payload=expected_payload, model_name=model_name, expected_status="failed"
    )


def validate_persisted_model_state(
    value: object, *, expected_payload: Mapping[str, object], model_name: str
) -> dict[str, object]:
    """Revalidate an analysis-facing state bound to the receipt's shared identity pins."""

    state = _normalize_object(value, "persisted reference model state")
    run_id, nonce, pins = _expected_identity(expected_payload)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "nonce": nonce,
        "pins": pins,
        **state,
    }
    if state.get("status") == "passed":
        return validate_model_receipt(
            envelope, expected_payload=expected_payload, model_name=model_name
        )
    if state.get("status") == "failed":
        return validate_failed_model_receipt(
            envelope, expected_payload=expected_payload, model_name=model_name
        )
    raise ValueError("persisted reference model state must be passed or failed")
