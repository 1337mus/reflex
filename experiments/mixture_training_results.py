"""Strict CPU validation for initialization and paired-training receipts."""

from __future__ import annotations

from experiments.mixture_training_contracts import (
    SCHEMA_VERSION,
    normalize_json_object,
    validate_safe_run_id,
    validate_uuid,
)
from experiments.mixture_training_evidence import (
    _COMMON_EVIDENCE_FIELDS,
    _TRAIN_EVIDENCE_FIELDS,
    _validate_artifact,
    _validate_counts,
    _validate_output_rows,
    _validate_provenance,
    _validate_training_evidence,
)

_RESULT_FIELDS = {
    "schema_version",
    "experiment_id",
    "run_id",
    "nonce",
    "phase",
    "arm",
    "payload_sha256",
    "status",
    "provenance",
    "evidence",
    "failure",
}


_FAILURE_FIELDS = {"stage", "type", "message"}


def validate_result(value: object, payload: object) -> dict[str, object]:
    """Validate a passed or failed receipt without promoting partial evidence to success."""

    from experiments.mixture_training_core import validate_payload

    normalized_payload = validate_payload(payload)
    result = normalize_json_object(value, "result")
    if set(result) != _RESULT_FIELDS:
        raise ValueError("result has an unexpected top-level schema")
    if type(result["schema_version"]) is not int or result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("result schema version is unsupported")
    for name in ("experiment_id", "run_id", "nonce"):
        if result[name] != normalized_payload[name]:
            raise ValueError(f"result {name} does not match its payload")
    validate_safe_run_id(result["run_id"])
    validate_uuid(result["nonce"], "result nonce")
    if result["arm"] != normalized_payload["arm"]:
        raise ValueError("result arm does not match its payload")
    if result["payload_sha256"] != normalized_payload["payload_sha256"]:
        raise ValueError("result payload_sha256 does not match its payload")
    if not isinstance(result["phase"], str) or not result["phase"] or len(result["phase"]) > 64:
        raise ValueError("result phase must identify its last lifecycle phase")
    status = result["status"]
    if status not in {"passed", "failed"}:
        raise ValueError("result status must be passed or failed")
    passed = status == "passed"
    if passed and result["phase"] != normalized_payload["phase"]:
        raise ValueError("passed result phase differs from its payload lifecycle")
    failure = result["failure"]
    if passed:
        if failure is not None:
            raise ValueError("passed result cannot contain a failure")
    else:
        failure = normalize_json_object(failure, "failure")
        if set(failure) != _FAILURE_FIELDS or any(
            not isinstance(failure[name], str) or not failure[name].strip()
            for name in _FAILURE_FIELDS
        ):
            raise ValueError("failed result requires stage, type, and message")
        result["failure"] = failure

    payload_phase = str(normalized_payload["phase"])
    train = payload_phase == "train"
    provenance = _validate_provenance(
        result["provenance"], normalized_payload, passed=passed, train=train
    )
    evidence = normalize_json_object(result["evidence"], "evidence")
    if passed:
        required = _COMMON_EVIDENCE_FIELDS | (_TRAIN_EVIDENCE_FIELDS if train else set())
        if not required.issubset(evidence):
            raise ValueError("passed result is missing required evidence")

    counts: dict[str, int | None] | None = None
    token_counts: dict[str, int | None] | None = None
    if "forward_counts" in evidence or "input_token_counts" in evidence:
        if not {"forward_counts", "input_token_counts"}.issubset(evidence):
            raise ValueError("forward and input token count evidence must appear together")
        counts, token_counts = _validate_counts(
            evidence["forward_counts"],
            evidence["input_token_counts"],
            passed_phase=payload_phase if passed else None,
        )
        evidence["forward_counts"] = counts
        evidence["input_token_counts"] = token_counts

    outputs: list[dict[str, object]] = []
    if "outputs" in evidence and evidence["outputs"] is not None:
        outputs = _validate_output_rows(
            evidence["outputs"],
            normalized_payload["evaluation_presentations"],
            label="outputs",
            require_complete=passed,
        )
        evidence["outputs"] = outputs
    elif passed:
        raise ValueError("passed result must retain all scored outputs")
    artifact = None
    if "output_artifact" in evidence:
        artifact = _validate_artifact(
            evidence["output_artifact"],
            outputs,
            str(normalized_payload["run_id"]),
            required=passed,
        )
        evidence["output_artifact"] = artifact
    elif passed:
        raise ValueError("passed result must retain an output artifact digest")

    initialization = evidence.get("initialization")
    if initialization is not None:
        from experiments.mixture_training_core import _validate_initialization

        normalized_initialization = _validate_initialization(
            initialization, str(normalized_payload["experiment_id"])
        )
        if train and normalized_initialization != normalized_payload["initialization"]:
            raise ValueError("training result initialization descriptor differs from payload")
        evidence["initialization"] = normalized_initialization
    elif passed:
        raise ValueError("passed result must retain initialization evidence")
    if passed and train:
        _validate_training_evidence(evidence, normalized_payload, outputs)
        if token_counts is None or counts is None:
            raise ValueError("passed training result must retain count evidence")
        if token_counts["final_evaluation"] != sum(row["input_tokens"] for row in outputs):
            raise ValueError("final evaluation token sum differs from scored outputs")
        if token_counts["reload_parity"] != sum(
            row["input_tokens"] for row in evidence["reload_parity"]["outputs"]
        ):
            raise ValueError("reload parity token sum differs from parity outputs")
    if passed and not train:
        if counts is None or token_counts is None:
            raise ValueError("passed initialization must retain count evidence")
        if token_counts["base_evaluation"] != sum(row["input_tokens"] for row in outputs):
            raise ValueError("initial evaluation token sum differs from scored outputs")
    result["provenance"] = provenance
    result["evidence"] = evidence
    return result


__all__ = ["validate_result"]
