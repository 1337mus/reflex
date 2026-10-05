"""Authenticate and validate returned runtime-rule study result envelopes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_output_evidence as output_api
from experiments import runtime_rule_study_payloads as payload_api
from experiments import runtime_rule_study_progress as progress_api
from experiments import runtime_rule_study_provenance as provenance_api
from experiments import runtime_rule_study_training_evidence as training_api

_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "study_id",
        "run_id",
        "nonce",
        "role",
        "payload_sha256",
        "bundle_sha256",
        "source_commit",
        "status",
        "phase",
        "failure",
        "provenance",
        "evidence",
    }
)
_IDENTITY_FIELDS = (
    "schema_version",
    "experiment_id",
    "study_id",
    "run_id",
    "nonce",
    "role",
    "payload_sha256",
    "bundle_sha256",
    "source_commit",
)
_EVIDENCE_FIELDS = frozenset(
    {
        "completed_forward_counts",
        "completed_input_token_counts",
        "pending_forward",
        "outputs",
        "selected_adapter_identity",
        "training",
        "reload",
    }
)
_PHASES = frozenset(
    {"preflight", "training", "final_evaluation", "reload", "finalize", "completed"}
)


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not _has_only_string_keys(value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    normalized = contracts.normalize_json_object(value, label)
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], normalized)


def _has_only_string_keys(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _has_only_string_keys(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_has_only_string_keys(item) for item in value)
    return True


def _nonblank(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be a nonblank string of at most {limit} characters")
    return value


def _specs_by_category(payload: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    return {
        "training": cast(list[dict[str, object]], payload["training_specs"]),
        "final_evaluation": cast(list[dict[str, object]], payload["evaluation_specs"]),
        "reload_parity": cast(list[dict[str, object]], payload["reload_specs"]),
    }


def _trusted_payload(payload: object, expected_payload_sha256: object) -> dict[str, object]:
    expected = contracts.validate_sha256(
        expected_payload_sha256, "trusted expected payload SHA-256"
    )
    return payload_api.validate_payload(payload, expected_payload_sha256=expected)


def initial_result(payload: object) -> dict[str, object]:
    """Create a mutable execution template; it is not a valid returned result."""

    normalized = payload_api.validate_payload(payload)
    role = cast(str, normalized["role"])
    ledger = progress_api.ForwardLedger(role, _specs_by_category(normalized))
    trained = role != contracts.ROLE_UNCHANGED
    return {
        **{field: normalized[field] for field in _IDENTITY_FIELDS},
        "status": "running",
        "phase": "preflight",
        "failure": None,
        "provenance": provenance_api.initial_provenance(normalized),
        "evidence": {
            **ledger.snapshot(),
            "outputs": [],
            "selected_adapter_identity": None,
            "training": training_api.initial_training_evidence() if trained else None,
            "reload": output_api.initial_reload_evidence() if trained else None,
        },
    }


def _validate_failure(value: object, *, passed: bool) -> dict[str, str] | None:
    if passed:
        if value is not None:
            raise ValueError("passed result failure must be null")
        return None
    failure = _json_object(value, "failure")
    if set(failure) != {"stage", "type", "message"}:
        raise ValueError("failure must contain exactly stage, type, and message")
    return {
        "stage": _nonblank(failure["stage"], "failure stage", 128),
        "type": _nonblank(failure["type"], "failure type", 128),
        "message": _nonblank(failure["message"], "failure message", 2_000),
    }


def validate_result(
    value: object, *, payload: object, expected_payload_sha256: object
) -> dict[str, object]:
    """Authenticate the caller-trusted payload and detach a consistent result."""

    normalized_payload = _trusted_payload(payload, expected_payload_sha256)
    result = _json_object(value, "result")
    if set(result) != _RESULT_FIELDS:
        raise ValueError("result has an unexpected top-level schema")
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != contracts.SCHEMA_VERSION
    ):
        raise ValueError("result schema_version must be a strict integer 1")
    for field in _IDENTITY_FIELDS:
        if contracts.canonical_json(result[field]) != contracts.canonical_json(
            normalized_payload[field]
        ):
            raise ValueError(f"result {field} differs from the authenticated payload")

    status = result["status"]
    phase = result["phase"]
    if not isinstance(status, str) or status not in {"passed", "failed"}:
        raise ValueError("returned result status must be passed or failed")
    if not isinstance(phase, str) or phase not in _PHASES:
        raise ValueError("result phase is invalid")
    passed = status == "passed"
    if (passed and phase != "completed") or (not passed and phase == "completed"):
        raise ValueError("result status and phase are inconsistent")
    result["failure"] = _validate_failure(result["failure"], passed=passed)

    role = cast(str, normalized_payload["role"])
    trained = role != contracts.ROLE_UNCHANGED
    planned_training_forwards = len(cast(list[object], normalized_payload["training_specs"]))
    evidence = _json_object(result["evidence"], "evidence")
    if set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("evidence has an unexpected schema")
    progress_value = {
        field: evidence[field]
        for field in (
            "completed_forward_counts",
            "completed_input_token_counts",
            "pending_forward",
        )
    }
    progress = progress_api.validate_progress(
        progress_value,
        role=role,
        specs_by_category=_specs_by_category(normalized_payload),
        require_complete=passed or phase == "finalize",
    )
    evidence.update(progress)
    counts = cast(Mapping[str, int], progress["completed_forward_counts"])
    pending = cast(dict[str, object] | None, progress["pending_forward"])
    pending_category = None if pending is None else cast(str, pending["category"])
    evaluation_attempted = counts["final_evaluation"] > 0 or pending_category == "final_evaluation"
    reload_attempted = counts["reload_parity"] > 0 or pending_category == "reload_parity"
    any_call = counts["total"] > 0 or pending is not None

    if phase == "preflight" and (counts["total"] != 0 or pending is not None):
        raise ValueError("preflight phase requires zero completed forwards and no pending call")
    if role == contracts.ROLE_UNCHANGED:
        if phase in {"training", "reload"}:
            raise ValueError("unchanged role cannot enter training or reload phase")
        if evidence["training"] is not None or evidence["reload"] is not None:
            raise ValueError("unchanged role cannot contain training or reload evidence")
    else:
        if evidence["training"] is None or evidence["reload"] is None:
            raise ValueError("trained roles require training and reload evidence")

    reload_evidence: dict[str, object] | None = None
    reload_observed = False
    if trained:
        reload_evidence = _json_object(evidence["reload"], "reload evidence")
        reload_observed = bool(reload_evidence.get("outputs")) or any(
            reload_evidence.get(field) is not None
            for field in (
                "tensor_equal",
                "tensor_sha256",
                "winner_match_count",
                "max_candidate_logit_difference",
            )
        )

    if phase == "training" and (evaluation_attempted or reload_attempted):
        raise ValueError("training phase cannot contain evaluation or reload work")
    if phase == "final_evaluation" and reload_attempted:
        raise ValueError("final_evaluation phase cannot contain reload work")

    require_loaded = phase != "preflight" or any_call
    require_reload = trained and (
        phase in {"reload", "finalize"} or reload_attempted or reload_observed or passed
    )
    provenance = provenance_api.validate_provenance(
        result["provenance"],
        payload=normalized_payload,
        passed=passed,
        require_loaded=require_loaded,
        require_reload=require_reload,
    )
    identity = provenance_api.validate_selected_adapter_identity(
        evidence["selected_adapter_identity"],
        payload=normalized_payload,
        required=require_loaded,
    )

    if trained:
        training_passed = (
            passed
            or phase in {"final_evaluation", "reload", "finalize"}
            or evaluation_attempted
            or reload_attempted
        )
        training = training_api.validate_training_evidence(
            evidence["training"],
            run_id=normalized_payload["run_id"],
            completed_training_forwards=counts["training"],
            passed=training_passed,
        )
        evidence["training"] = training
        if pending_category == "training" and (
            training["initialization_verified"] is not True
            or training["adapter_inventory"] is None
            or training["optimizer"] is None
            or training["optimizer_initial_state_entries"] != 0
        ):
            raise ValueError(
                "pending training forward requires verified initialization, inventory, "
                "and fresh optimizer"
            )
        if (
            identity is not None
            and training["initial_tensor_sha256"] is not None
            and training["initial_tensor_sha256"] != identity["tensor_sha256"]
        ):
            raise ValueError("training initial tensor differs from selected adapter identity")
        updates = cast(int, training["optimizer_updates_completed"])
        if (
            pending_category == "training"
            and counts["training"] % contracts.MICROBATCHES_PER_UPDATE == 0
        ):
            if updates != counts["training"] // contracts.MICROBATCHES_PER_UPDATE:
                raise ValueError("pending batch boundary requires its completed optimizer update")
        if counts[
            "training"
        ] > contracts.UNSCORED_SAVE_UPDATE * contracts.MICROBATCHES_PER_UPDATE or (
            pending is not None
            and pending_category == "training"
            and cast(int, pending["index"])
            >= contracts.UNSCORED_SAVE_UPDATE * contracts.MICROBATCHES_PER_UPDATE
        ):
            paths = cast(Mapping[str, object], training["adapter_paths"])
            if str(contracts.UNSCORED_SAVE_UPDATE) not in paths:
                raise ValueError("training must retain the midpoint adapter snapshot")
        losses = cast(list[object], training["training_step_losses"])
        terminal_loss_gap = updates > 0 and len(losses) == updates - 1
        if terminal_loss_gap and (pending is not None or evaluation_attempted or reload_attempted):
            raise ValueError(
                "terminal training loss gap cannot coexist with a pending or later call"
            )
        final_tensor_sha256 = training["final_tensor_sha256"]
    else:
        training = None
        final_tensor_sha256 = None

    if phase == "final_evaluation" and trained:
        if counts["training"] != planned_training_forwards or not training_passed:
            raise ValueError("final_evaluation requires successful complete training")

    output_passed = passed
    outputs = output_api.validate_scored_outputs(
        evidence["outputs"],
        presentations=cast(list[dict[str, object]], normalized_payload["evaluation"]),
        specs=cast(list[dict[str, object]], normalized_payload["evaluation_specs"]),
        completed_forwards=counts["final_evaluation"],
        passed=output_passed,
    )
    evidence["outputs"] = outputs
    expected_evaluations = len(cast(list[object], normalized_payload["evaluation_specs"]))
    evaluation_complete = (
        counts["final_evaluation"] == expected_evaluations and len(outputs) == expected_evaluations
    )
    if len(outputs) != counts["final_evaluation"]:
        if pending is not None or reload_attempted or phase != "final_evaluation":
            raise ValueError(
                "terminal evaluation output gap must be the last work in final_evaluation"
            )

    if phase in {"reload", "finalize"}:
        if trained and (not training_passed or counts["training"] != planned_training_forwards):
            raise ValueError("reload and finalize require successful complete training")
        if not evaluation_complete:
            raise ValueError("reload and finalize require every final evaluation output")

    if trained:
        assert reload_evidence is not None
        if reload_attempted or reload_observed:
            if not evaluation_complete or final_tensor_sha256 is None:
                raise ValueError(
                    "reload work requires complete final outputs and a final tensor digest"
                )
        reload_passed = passed or phase == "finalize"
        checked_reload = output_api.validate_reload_evidence(
            reload_evidence,
            presentations=cast(list[dict[str, object]], normalized_payload["reload"]),
            specs=cast(list[dict[str, object]], normalized_payload["reload_specs"]),
            final_outputs=outputs,
            final_tensor_sha256=final_tensor_sha256,
            completed_forwards=counts["reload_parity"],
            passed=reload_passed,
        )
        if len(cast(list[object], checked_reload["outputs"])) != counts["reload_parity"]:
            if pending is not None or phase != "reload":
                raise ValueError("terminal reload output gap must remain in failed reload phase")
        evidence["reload"] = checked_reload

    result["provenance"] = provenance
    result["evidence"] = evidence
    return result
