"""Strict remote payload and receipt contracts for the bounded real-data pilot."""

from __future__ import annotations

import json
import math
import re
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from hashlib import sha256
from itertools import permutations

from experiments import real_pilot_core as core
from reflex_decisions.data import DecisionRecord, SplitManifest
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest
from reflex_decisions.smoke import sanitize_exception_message

DATASET_RECORD_COUNTS = core.DATASET_RECORD_COUNTS
EVALUATION_PRESENTATION_COUNT = core.EVALUATION_PRESENTATION_COUNT
FINAL_EVALUATION_COUNT = core.FINAL_EVALUATION_COUNT
MAX_FORWARD_COUNT = core.MAX_FORWARD_COUNT
MAX_INPUT_TOKENS = core.MAX_INPUT_TOKENS
MAX_UPDATES = core.MAX_UPDATES
MODEL_ID = core.MODEL_ID
MODEL_REVISION = core.MODEL_REVISION
PROTOCOL_PATH = core.PROTOCOL_PATH
RELOAD_PARITY_COUNT = core.RELOAD_PARITY_COUNT
SOURCE_FINGERPRINT_PATHS = core.SOURCE_FINGERPRINT_PATHS
TRAIN_DATASET_IDS = core.TRAIN_DATASET_IDS
TRAIN_FORWARD_COUNT = core.TRAIN_FORWARD_COUNT
TRAIN_RECORD_COUNT = core.TRAIN_RECORD_COUNT
BASE_EVALUATION_COUNT = core.BASE_EVALUATION_COUNT
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
build_training_schedule = core.build_training_schedule
build_evaluation_presentations = core.build_evaluation_presentations
validate_pilot_data = core.validate_pilot_data
training_rehearsal_core = core.training_rehearsal_core


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
        raise ValueError("pilot payload is not strict JSON") from exc


def _json_sha256(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def validate_run_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_RUN_ID.fullmatch(value) is None:
        raise ValueError(
            "run ID must be 1–64 safe ASCII letters, digits, dots, underscores, or hyphens"
        )
    return value


def validate_nonce(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("run nonce must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("run nonce must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError("run nonce must be a canonical UUID")
    return value


def build_training_payload(records: Sequence[DecisionRecord]) -> tuple[dict[str, object], ...]:
    """Serialize only train records, with their gold answer for supervised updates."""

    selected = [record for record in records if record.dataset_id in TRAIN_DATASET_IDS]
    build_training_schedule(selected)
    return tuple(
        {
            "record_id": record.record_id,
            "dataset_id": record.dataset_id,
            "request": record.request.model_dump(mode="json"),
            "answer_id": record.answer_id,
        }
        for record in selected
    )


def _expected_order_ids(dataset_id: str, base_ids: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    if dataset_id.startswith("dbpedia14-"):
        if dataset_id.endswith("-calibration"):
            return (base_ids,)
        reverse = tuple(reversed(base_ids))
        return tuple(
            dict.fromkeys(
                (
                    *(base_ids[index:] + base_ids[:index] for index in range(len(base_ids))),
                    *(reverse[index:] + reverse[:index] for index in range(len(reverse))),
                )
            )
        )
    if dataset_id.startswith("sms-"):
        return (base_ids,) if dataset_id.endswith("-calibration") else tuple(permutations(base_ids))
    if dataset_id == "snli-pilot-v1-development":
        return tuple(permutations(base_ids))
    raise ValueError("evaluation dataset is outside the pilot allowlist")


def _presentation_id(record_id: str, order_ids: Sequence[str]) -> str:
    identity = "\0".join(("reflex-real-pilot-presentation-v1", record_id, *order_ids))
    return sha256(identity.encode("utf-8")).hexdigest()


def _validate_evaluation_payload(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list) or len(rows) != EVALUATION_PRESENTATION_COUNT:
        raise ValueError("evaluation payload must contain exactly 2,572 presentations")
    expected_fields = {
        "presentation_id",
        "record_id",
        "dataset_id",
        "request_hash",
        "order_index",
        "order_ids",
        "request",
    }
    seen_ids: set[str] = set()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    dataset_presentations: Counter[str] = Counter()
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("evaluation payload may contain only request-only presentation fields")
        presentation_id = _sha256(row["presentation_id"], field="presentation_id")
        record_id, dataset_id = row["record_id"], row["dataset_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("evaluation record ID must be nonblank")
        if dataset_id not in DATASET_RECORD_COUNTS or dataset_id in TRAIN_DATASET_IDS:
            raise ValueError("evaluation dataset is outside the approved evaluation pool")
        if presentation_id in seen_ids:
            raise ValueError("evaluation presentation IDs must be unique")
        seen_ids.add(presentation_id)
        if type(row["order_index"]) is not int or row["order_index"] < 0:
            raise ValueError("evaluation order index must be a nonnegative integer")
        if not isinstance(row["order_ids"], list) or any(
            not isinstance(option_id, str) or not option_id for option_id in row["order_ids"]
        ):
            raise ValueError("evaluation order IDs must be nonblank strings")
        request = DecisionRequest.model_validate(row["request"])
        order_ids = tuple(option.id for option in request.options)
        if order_ids != tuple(row["order_ids"]):
            raise ValueError("evaluation request option order does not match order_ids")
        if row["request_hash"] != request.request_hash:
            raise ValueError("evaluation request hash does not match its request")
        if presentation_id != _presentation_id(record_id, order_ids):
            raise ValueError("evaluation presentation ID is not the stable opaque order hash")
        normalized = dict(row)
        grouped[record_id].append(normalized)
        dataset_presentations[dataset_id] += 1
        normalized_rows.append(normalized)

    expected_presentations = {
        "dbpedia14-pilot-v1-development": 56 * 28,
        "dbpedia14-pilot-v1-calibration": 56,
        "sms-pilot-v1-development": 60 * 2,
        "sms-pilot-v1-calibration": 60,
        "snli-pilot-v1-development": 128 * 6,
    }
    if dataset_presentations != Counter(expected_presentations):
        raise ValueError("evaluation dataset presentation counts do not match the frozen panel")
    expected_record_counts = {
        "dbpedia14-pilot-v1-development": 56,
        "dbpedia14-pilot-v1-calibration": 56,
        "sms-pilot-v1-development": 60,
        "sms-pilot-v1-calibration": 60,
        "snli-pilot-v1-development": 128,
    }
    seen_dataset_by_record: dict[str, str] = {}
    actual_record_counts: Counter[str] = Counter()
    for record_id, record_rows in grouped.items():
        dataset_ids = {str(row["dataset_id"]) for row in record_rows}
        if len(dataset_ids) != 1:
            raise ValueError("one evaluation record ID cannot cross datasets")
        dataset_id = next(iter(dataset_ids))
        seen_dataset_by_record[record_id] = dataset_id
        actual_record_counts[dataset_id] += 1
        order_indices = [int(row["order_index"]) for row in record_rows]
        first_index = order_indices.index(0) if 0 in order_indices else -1
        if first_index < 0:
            raise ValueError("each evaluation record needs its original option order")
        canonical = record_rows[first_index]
        base_request = DecisionRequest.model_validate(canonical["request"])
        base_ids = tuple(option.id for option in base_request.options)
        expected_orders = _expected_order_ids(dataset_id, base_ids)
        if sorted(order_indices) != list(range(len(expected_orders))):
            raise ValueError("evaluation record is missing or duplicating an order index")
        actual_by_index = {int(row["order_index"]): row for row in record_rows}
        for index, option_ids in enumerate(expected_orders):
            current = actual_by_index[index]
            request = DecisionRequest.model_validate(current["request"])
            expected_request = base_request.model_copy(
                update={
                    "options": tuple(
                        {option.id: option for option in base_request.options}[option_id]
                        for option_id in option_ids
                    )
                }
            )
            if request != expected_request or current["order_ids"] != list(option_ids):
                raise ValueError("evaluation option orders do not match the frozen balanced panel")
        if len({tuple(row["order_ids"]) for row in record_rows}) != len(expected_orders):
            raise ValueError("evaluation panel contains duplicate option orders")
    if actual_record_counts != Counter(expected_record_counts):
        raise ValueError("evaluation payload does not contain the exact source-record counts")
    return normalized_rows


def _validate_training_payload(rows: object) -> list[dict[str, object]]:
    if not isinstance(rows, list) or len(rows) != TRAIN_RECORD_COUNT:
        raise ValueError("training payload must contain exactly 504 records")
    expected_fields = {"record_id", "dataset_id", "request", "answer_id"}
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("training rows must contain only train request and gold fields")
        record_id, dataset_id, answer_id = row["record_id"], row["dataset_id"], row["answer_id"]
        if not isinstance(record_id, str) or not record_id.strip() or record_id in seen_ids:
            raise ValueError("training record IDs must be unique and nonblank")
        if dataset_id not in TRAIN_DATASET_IDS:
            raise ValueError("training payload contains a non-train dataset")
        if not isinstance(answer_id, str) or not answer_id.strip():
            raise ValueError("training gold answer must be a nonblank option ID")
        request = DecisionRequest.model_validate(row["request"])
        expected_options = 14 if dataset_id == TRAIN_DATASET_IDS[0] else 2
        if len(request.options) != expected_options:
            raise ValueError("training request option count does not match its task")
        if answer_id not in {option.id for option in request.options}:
            raise ValueError("training gold answer is not one of the request options")
        seen_ids.add(record_id)
        counts[dataset_id] += 1
        normalized.append(dict(row))
    if counts != Counter({TRAIN_DATASET_IDS[0]: 252, TRAIN_DATASET_IDS[1]: 252}):
        raise ValueError("training payload must contain 252 rows from each approved task")
    return normalized


def _validate_pins(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "records_sha256",
        "manifest_sha256",
        "recipe_sha256",
        "protocol_sha256",
        "source_file_sha256",
    }:
        raise ValueError("pilot pins have an unexpected schema")
    for name in ("records_sha256", "manifest_sha256", "recipe_sha256", "protocol_sha256"):
        _sha256(value[name], field=name)
    source_hashes = value["source_file_sha256"]
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != set(SOURCE_FINGERPRINT_PATHS)
        or any(
            not isinstance(digest, str) or _HEX_SHA256.fullmatch(digest) is None
            for digest in source_hashes.values()
        )
    ):
        raise ValueError("pilot source file fingerprints are malformed")
    if source_hashes.get(PROTOCOL_PATH) != value["protocol_sha256"]:
        raise ValueError("protocol pin and uploaded source fingerprint differ")
    return value


def validate_remote_payload(
    payload: object, *, expected_identity: dict[str, object] | None = None
) -> dict[str, object]:
    """Check remote input identity, source pins, exact budgets, and label isolation."""

    normalized = json.loads(_canonical_json(payload))
    if not isinstance(normalized, dict) or set(normalized) != {
        "schema_version",
        "run_id",
        "nonce",
        "pins",
        "train_records",
        "evaluation_presentations",
        "train_payload_sha256",
        "evaluation_payload_sha256",
    }:
        raise ValueError("remote training payload has an unexpected schema")
    if type(normalized["schema_version"]) is not int or normalized["schema_version"] != 1:
        raise ValueError("remote payload schema version is unsupported")
    validate_run_id(normalized["run_id"])
    validate_nonce(normalized["nonce"])
    pins = _validate_pins(normalized["pins"])
    train_records = _validate_training_payload(normalized["train_records"])
    evaluation = _validate_evaluation_payload(normalized["evaluation_presentations"])
    if {row["record_id"] for row in train_records} & {row["record_id"] for row in evaluation}:
        raise ValueError("training and evaluation record IDs must be disjoint")
    if normalized["train_payload_sha256"] != _json_sha256(train_records):
        raise ValueError("remote train payload hash does not match its rows")
    if normalized["evaluation_payload_sha256"] != _json_sha256(evaluation):
        raise ValueError("remote evaluation payload hash does not match its rows")
    if expected_identity is not None:
        expected_pins = expected_identity.get("pins")
        if (
            normalized["run_id"] != expected_identity.get("run_id")
            or normalized["nonce"] != expected_identity.get("nonce")
            or pins != expected_pins
        ):
            raise ValueError("remote payload identity or pins differ from the local launch")
    return normalized


def build_remote_payload(
    manifest: SplitManifest,
    records: Sequence[DecisionRecord],
    *,
    run_id: str,
    nonce: str,
    records_sha256: str,
    manifest_sha256: str,
    recipe_sha256: str,
    protocol_sha256: str,
    source_file_sha256: dict[str, str],
) -> dict[str, object]:
    """Build the label-separated worker input only after the local split audit."""

    validate_pilot_data(manifest, records)
    train_records = list(build_training_payload(records))
    evaluation = list(build_evaluation_presentations(records))
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": validate_run_id(run_id),
        "nonce": validate_nonce(nonce),
        "pins": {
            "records_sha256": records_sha256,
            "manifest_sha256": manifest_sha256,
            "recipe_sha256": recipe_sha256,
            "protocol_sha256": protocol_sha256,
            "source_file_sha256": source_file_sha256,
        },
        "train_records": train_records,
        "evaluation_presentations": evaluation,
        "train_payload_sha256": _json_sha256(train_records),
        "evaluation_payload_sha256": _json_sha256(evaluation),
    }
    identity = {"run_id": run_id, "nonce": nonce, "pins": payload["pins"]}
    return validate_remote_payload(payload, expected_identity=identity)


def _normalize_json_object(value: object, *, label: str) -> dict[str, object]:
    try:
        normalized = json.loads(_canonical_json(value))
    except ValueError as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _validate_receipt_identity(
    receipt: dict[str, object], expected_identity: dict[str, object]
) -> dict[str, object]:
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("run_id") != expected_identity.get("run_id")
        or receipt.get("nonce") != expected_identity.get("nonce")
        or receipt.get("pins") != expected_identity.get("pins")
    ):
        raise ValueError("remote receipt identity or pins do not match this launch")
    validate_run_id(receipt.get("run_id"))
    validate_nonce(receipt.get("nonce"))
    return _validate_pins(receipt.get("pins"))


def validate_recovery_receipt(
    value: object, *, expected_identity: dict[str, object]
) -> dict[str, object]:
    """Accept only this run's strict-JSON progress and always downgrade it to failed."""

    receipt = _normalize_json_object(value, label="recovered progress receipt")
    pins = _validate_receipt_identity(receipt, expected_identity)
    provenance = receipt.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("measured_source_file_sha256") != pins["source_file_sha256"]
    ):
        raise ValueError("recovered progress source measurements do not match this launch")
    receipt["status"] = "failed"
    receipt["phase"] = "recovered_after_remote_exception"
    receipt["failure"] = {
        "stage": "remote_exception",
        "type": "RemoteCallException",
        "message": "the remote call ended before the client received a completion receipt",
    }
    return receipt


def validate_failed_receipt(
    value: object,
    *,
    expected_payload: dict[str, object],
) -> dict[str, object]:
    """Preserve only this launch's strict-JSON remote failure and partial evidence."""

    receipt = _normalize_json_object(value, label="remote failure receipt")
    expected_fields = {
        "schema_version",
        "status",
        "run_id",
        "nonce",
        "pins",
        "phase",
        "failure",
        "provenance",
        "evidence",
        "limits",
    }
    if set(receipt) != expected_fields:
        raise ValueError("remote failure receipt has an unexpected schema")
    identity = {
        "run_id": expected_payload.get("run_id"),
        "nonce": expected_payload.get("nonce"),
        "pins": expected_payload.get("pins"),
    }
    pins = _validate_receipt_identity(receipt, identity)
    if receipt.get("status") != "failed":
        raise ValueError("remote receipt did not report a failed run")
    if not isinstance(receipt.get("phase"), str) or not receipt["phase"].strip():
        raise ValueError("remote failure phase is missing")
    if receipt.get("limits") != core.MODAL_LIMITS:
        raise ValueError("remote failure limits do not match the bounded run contract")
    provenance = receipt.get("provenance")
    expected_sources = pins["source_file_sha256"]
    if not isinstance(provenance, dict) or (
        provenance.get("model_id") != MODEL_ID
        or provenance.get("model_revision") != MODEL_REVISION
        or provenance.get("source_file_sha256") != expected_sources
        or provenance.get("measured_source_file_sha256") != expected_sources
    ):
        raise ValueError("remote failure model or source provenance is incomplete")
    if not isinstance(receipt.get("evidence"), dict):
        raise ValueError("remote failure evidence must be a JSON object")

    failure = receipt.get("failure")
    if not isinstance(failure, dict) or set(failure) != {"stage", "type", "message"}:
        raise ValueError("remote failure details have an unexpected schema")
    stage, error_type, message = failure["stage"], failure["type"], failure["message"]
    if (
        not isinstance(stage, str)
        or not stage.strip()
        or not isinstance(error_type, str)
        or not error_type.strip()
        or not isinstance(message, str)
        or not message.strip()
    ):
        raise ValueError("remote failure details must contain nonblank strings")
    sanitized_message = sanitize_exception_message(ValueError(message))
    if not sanitized_message.strip():
        raise ValueError("remote failure message is empty after sanitization")
    receipt["failure"] = {
        "stage": stage,
        "type": error_type,
        "message": sanitized_message,
    }
    receipt["status"] = "failed"
    return receipt


def _validate_scored_rows(
    expected_rows: list[dict[str, object]], value: object
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != EVALUATION_PRESENTATION_COUNT:
        raise ValueError("scored output does not contain every evaluation presentation")
    expected_by_id = {row["presentation_id"]: row for row in expected_rows}
    actual_ids: set[str] = set()
    expected_fields = {
        "presentation_id",
        "record_id",
        "dataset_id",
        "request_hash",
        "order_ids",
        "candidate_logits",
        "winner_option_id",
        "input_tokens",
        "prompt_sha256",
    }
    normalized: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("scored output row has an unexpected schema")
        presentation_id = row["presentation_id"]
        expected = expected_by_id.get(presentation_id)
        if expected is None or presentation_id in actual_ids:
            raise ValueError("scored output presentation is unknown or duplicated")
        if any(
            row.get(key) != expected.get(key)
            for key in ("record_id", "dataset_id", "request_hash", "order_ids")
        ):
            raise ValueError("scored output metadata differs from its request-only input")
        logits = row["candidate_logits"]
        order_ids = expected["order_ids"]
        if not isinstance(logits, list) or len(logits) != len(order_ids):
            raise ValueError("scored output candidate logits do not match the option count")
        if any(
            isinstance(logit, bool)
            or not isinstance(logit, (int, float))
            or not math.isfinite(float(logit))
            for logit in logits
        ):
            raise ValueError("scored output candidate logits must be finite numbers")
        maximum = max(float(logit) for logit in logits)
        winner = min(
            option_id
            for option_id, logit in zip(order_ids, logits, strict=True)
            if float(logit) == maximum
        )
        if row["winner_option_id"] != winner:
            raise ValueError("scored output winner does not match the candidate logits")
        if type(row["input_tokens"]) is not int or not 1 <= row["input_tokens"] <= MAX_INPUT_TOKENS:
            raise ValueError("scored output token count is outside the supported boundary")
        expected_prompt_hash = sha256(
            render_prompt(DecisionRequest.model_validate(expected["request"])).encode("utf-8")
        ).hexdigest()
        if _sha256(row["prompt_sha256"], field="prompt_sha256") != expected_prompt_hash:
            raise ValueError("scored output prompt hash does not match its request")
        actual_ids.add(presentation_id)
        normalized.append(dict(row))
    if actual_ids != set(expected_by_id):
        raise ValueError("scored output omitted an evaluation presentation")
    return normalized


def validate_passed_receipt(
    value: object,
    *,
    expected_payload: dict[str, object],
) -> dict[str, object]:
    """Reject incomplete success claims before local metrics or artifact adoption."""

    receipt = _normalize_json_object(value, label="remote receipt")
    identity = {
        "run_id": expected_payload.get("run_id"),
        "nonce": expected_payload.get("nonce"),
        "pins": expected_payload.get("pins"),
    }
    _validate_receipt_identity(receipt, identity)
    if receipt.get("status") != "passed":
        raise ValueError("remote receipt did not pass")
    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("model_id") != MODEL_ID
        or provenance.get("model_revision") != MODEL_REVISION
        or provenance.get("source_file_sha256") != identity["pins"]["source_file_sha256"]
        or provenance.get("measured_source_file_sha256") != identity["pins"]["source_file_sha256"]
    ):
        raise ValueError("remote receipt model or source provenance is incomplete")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("remote receipt evidence is missing")
    if (
        evidence.get("optimizer_updates_completed") != MAX_UPDATES
        or evidence.get("base_gradients_none") is not True
        or evidence.get("forward_counts")
        != {
            "base_evaluation": BASE_EVALUATION_COUNT,
            "training": TRAIN_FORWARD_COUNT,
            "final_evaluation": FINAL_EVALUATION_COUNT,
            "reload_parity": RELOAD_PARITY_COUNT,
            "total": MAX_FORWARD_COUNT,
        }
    ):
        raise ValueError("remote receipt fixed training or forward-count evidence is incomplete")
    inventory = evidence.get("adapter_inventory")
    if not isinstance(inventory, dict) or (
        inventory.get("module_count") != training_rehearsal_core.EXPECTED_LORA_MODULES
        or inventory.get("adapter_tensor_count") != training_rehearsal_core.EXPECTED_ADAPTER_TENSORS
        or inventory.get("trainable_parameter_count")
        != training_rehearsal_core.EXPECTED_TRAINABLE_PARAMETERS
        or inventory.get("base_parameters_frozen_bf16") is not True
    ):
        raise ValueError("remote receipt adapter inventory does not match the pinned model")
    losses = evidence.get("training_step_losses")
    if not isinstance(losses, list) or len(losses) != MAX_UPDATES:
        raise ValueError("remote receipt must contain one loss and gradient row per update")
    gradient_total = 0.0
    for expected_update, row in enumerate(losses, start=1):
        if not isinstance(row, dict) or set(row) != {
            "update",
            "mean_loss",
            "lo_ra_b_gradient_l1",
        }:
            raise ValueError("remote receipt training evidence row is malformed")
        if row["update"] != expected_update:
            raise ValueError("remote receipt training updates are missing or reordered")
        loss, gradient = row["mean_loss"], row["lo_ra_b_gradient_l1"]
        if (
            isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or isinstance(gradient, bool)
            or not isinstance(gradient, (int, float))
            or not math.isfinite(float(gradient))
            or float(gradient) < 0
        ):
            raise ValueError("remote receipt training loss or gradient evidence is non-finite")
        gradient_total += float(gradient)
    if gradient_total <= 0:
        raise ValueError("remote receipt did not prove nonzero adapter gradients")
    changed = evidence.get("adapter_update")
    if not isinstance(changed, dict) or (
        type(changed.get("changed_tensor_count")) is not int
        or changed["changed_tensor_count"] < 1
        or changed["changed_tensor_count"] > training_rehearsal_core.EXPECTED_ADAPTER_TENSORS
        or not isinstance(changed.get("changed_tensor_names"), list)
        or len(changed["changed_tensor_names"]) != changed["changed_tensor_count"]
    ):
        raise ValueError("remote receipt does not prove that training changed adapter tensors")
    snapshots = evidence.get("adapter_paths")
    if not isinstance(snapshots, list) or [
        row.get("update") for row in snapshots if isinstance(row, dict)
    ] != [126, 252]:
        raise ValueError("remote receipt must preserve midpoint and final adapter snapshots")
    expected_run_dir = f"/artifacts/runs/{expected_payload['run_id']}"
    for update, snapshot in zip((126, 252), snapshots, strict=True):
        if not isinstance(snapshot, dict) or set(snapshot) != {"update", "path", "files_sha256"}:
            raise ValueError("remote adapter snapshot metadata is malformed")
        files = snapshot["files_sha256"]
        if (
            snapshot["update"] != update
            or snapshot["path"] != f"{expected_run_dir}/adapter-update-{update:03d}"
            or not isinstance(files, dict)
            or not {"adapter_model.safetensors", "adapter_config.json"}.issubset(files)
            or any(
                not isinstance(name, str)
                or not isinstance(digest, str)
                or _HEX_SHA256.fullmatch(digest) is None
                for name, digest in files.items()
            )
        ):
            raise ValueError("remote adapter snapshot path or hashes are invalid")
    expected_presentations = _validate_evaluation_payload(
        expected_payload["evaluation_presentations"]
    )
    _validate_scored_rows(expected_presentations, evidence.get("base_outputs"))
    final_outputs = _validate_scored_rows(expected_presentations, evidence.get("final_outputs"))
    artifacts = evidence.get("output_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"base_outputs", "final_outputs"}:
        raise ValueError("remote output artifact references are incomplete")
    for name, entry in artifacts.items():
        expected_name = f"{expected_run_dir}/{name}.json"
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or entry["path"] != expected_name
            or _HEX_SHA256.fullmatch(entry["sha256"]) is None
        ):
            raise ValueError("remote output artifact path or hash is malformed")
    parity = evidence.get("reload_parity")
    expected_parity_ids = sorted(row["presentation_id"] for row in expected_presentations)[
        :RELOAD_PARITY_COUNT
    ]
    if not isinstance(parity, dict) or set(parity) != {
        "presentation_ids",
        "reloaded_logits",
        "adapter_tensor_keys_shapes_values_match",
        "winner_match",
        "max_abs_logit_diff",
    }:
        raise ValueError("remote reload parity evidence is incomplete")
    final_by_id = {row["presentation_id"]: row for row in final_outputs}
    if (
        parity.get("presentation_ids") != expected_parity_ids
        or parity.get("adapter_tensor_keys_shapes_values_match") is not True
        or parity.get("winner_match") is not True
    ):
        raise ValueError("remote fresh-reload parity identity or adapter evidence failed")
    reloaded_logits = parity.get("reloaded_logits")
    if not isinstance(reloaded_logits, list) or len(reloaded_logits) != RELOAD_PARITY_COUNT:
        raise ValueError("remote fresh-reload logits are incomplete")
    differences: list[float] = []
    for presentation_id, logits in zip(expected_parity_ids, reloaded_logits, strict=True):
        final = final_by_id[presentation_id]
        final_logits = final["candidate_logits"]
        if (
            not isinstance(logits, list)
            or len(logits) != len(final_logits)
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in logits
            )
        ):
            raise ValueError("remote fresh-reload logits are malformed")
        option_ids = final_by_id[presentation_id]["order_ids"]
        final_maximum = max(float(value) for value in final_logits)
        reloaded_maximum = max(float(value) for value in logits)
        final_winner = min(
            option_id
            for option_id, logit in zip(option_ids, final_logits, strict=True)
            if float(logit) == final_maximum
        )
        reloaded_winner = min(
            option_id
            for option_id, logit in zip(option_ids, logits, strict=True)
            if float(logit) == reloaded_maximum
        )
        if final_winner != reloaded_winner:
            raise ValueError("remote fresh-reload winner differed from the saved adapter")
        differences.extend(
            abs(float(left) - float(right))
            for left, right in zip(logits, final_logits, strict=True)
        )
    maximum_difference = max(differences, default=math.inf)
    reported_difference = parity.get("max_abs_logit_diff")
    if (
        isinstance(reported_difference, bool)
        or not isinstance(reported_difference, (int, float))
        or not math.isfinite(float(reported_difference))
        or abs(float(reported_difference) - maximum_difference) > 1e-9
        or maximum_difference > 1e-3
    ):
        raise ValueError("remote fresh-reload logit difference exceeds the pinned tolerance")
    return receipt
