"""Strict receipt validation for the bounded fresh evaluation."""
# ruff: noqa: E501

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments import fresh_eval_core as core
from experiments import mixture_training_contracts
from experiments.mixture_training_contracts import normalize_json_object, validate_sha256

_RESULT_FIELDS = {
    "schema_version",
    "experiment_id",
    "run_id",
    "nonce",
    "selection_sha256",
    "payload_sha256",
    "status",
    "phase",
    "provenance",
    "evidence",
    "failure",
}
_EVIDENCE_FIELDS = {
    "forward_counts",
    "input_token_counts",
    "outputs",
    "adapter_identity",
    "reload_parity",
    "unknown_work",
}
_COUNT_KEYS = {"base_evaluation", "training", "final_evaluation", "reload_parity", "total"}
_OUTPUT_FIELDS = {
    "presentation_id",
    "record_id",
    "dataset_id",
    "source_group_id",
    "request_hash",
    "order_index",
    "order_ids",
    "candidate_logits",
    "winner_option_id",
    "input_tokens",
    "prompt_sha256",
}


def validate_unknown_work(
    value: object, *, failure_stage: str | None, passed: bool
) -> dict[str, object] | None:
    """For lost work, record uncertainty rather than manufacturing a zero count."""

    if passed:
        if value is not None:
            raise ValueError("passed fresh evaluation cannot retain unknown work")
        return None
    lifecycle = failure_stage in {"modal_lifecycle", "worker_execution", "result_validation"}
    if value is None:
        if lifecycle:
            raise ValueError("lost fresh evaluation execution requires unknown work evidence")
        return None
    unknown = normalize_json_object(value, "unknown_work")
    if set(unknown) != {"possible_forwards", "reason"}:
        raise ValueError("unknown work has an unexpected schema")
    if type(unknown["possible_forwards"]) is not int or unknown["possible_forwards"] < 1:
        raise ValueError("unknown work must retain a positive possible-forward count")
    if (
        unknown["possible_forwards"] > core.MAX_TOTAL_FORWARDS
        or not isinstance(unknown["reason"], str)
        or not unknown["reason"].strip()
    ):
        raise ValueError("unknown work evidence is malformed")
    return unknown


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _validate_output_rows(
    value: object,
    presentations: Sequence[Mapping[str, object]],
    compiled_requests: Sequence[Mapping[str, object]],
    *,
    label: str,
    complete: bool,
) -> list[dict[str, object]]:
    if (
        not isinstance(value, list)
        or len(value) > len(presentations)
        or (complete and len(value) != len(presentations))
    ):
        raise ValueError(f"{label} outputs have an unexpected count")
    outputs: list[dict[str, object]] = []
    for output, presentation, compiled in zip(
        value, presentations, compiled_requests, strict=False
    ):
        row = normalize_json_object(output, label)
        if set(row) != _OUTPUT_FIELDS:
            raise ValueError(f"{label} output has an unexpected schema")
        identity = {
            key: presentation[key]
            for key in (
                "presentation_id",
                "record_id",
                "dataset_id",
                "source_group_id",
                "request_hash",
                "order_index",
                "order_ids",
            )
        }
        if {key: row[key] for key in identity} != identity:
            raise ValueError(f"{label} output identity differs from the frozen presentation")
        scores = row["candidate_logits"]
        if not isinstance(scores, list) or len(scores) != len(presentation["order_ids"]):
            raise ValueError(f"{label} candidate score dimensions are invalid")
        parsed = [_finite(score, f"{label} candidate score") for score in scores]
        expected_winner = min(
            option_id
            for option_id, score in zip(presentation["order_ids"], parsed, strict=True)
            if score == max(parsed)
        )
        if row["winner_option_id"] != expected_winner:
            raise ValueError(f"{label} winner violates the semantic tie rule")
        if (
            row["input_tokens"] != compiled["input_tokens"]
            or row["prompt_sha256"] != compiled["prompt_sha256"]
        ):
            raise ValueError(f"{label} output differs from the host-compiled request")
        outputs.append(row)
    return outputs


def _validate_counts(
    counts_value: object,
    tokens_value: object,
    outputs: Mapping[str, Sequence[Mapping[str, object]]],
    parity: Sequence[Mapping[str, object]],
    *,
    passed: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    counts = normalize_json_object(counts_value, "forward_counts")
    tokens = normalize_json_object(tokens_value, "input_token_counts")
    if set(counts) != _COUNT_KEYS or set(tokens) != _COUNT_KEYS:
        raise ValueError("fresh evaluation counts have an unexpected schema")
    normalized_counts: dict[str, int] = {}
    normalized_tokens: dict[str, int] = {}
    expected_counts = {
        "base_evaluation": len(outputs["base"]),
        "training": 0,
        "final_evaluation": len(outputs["adapter"]),
        "reload_parity": len(parity),
    }
    expected_tokens = {
        "base_evaluation": sum(int(row["input_tokens"]) for row in outputs["base"]),
        "training": 0,
        "final_evaluation": sum(int(row["input_tokens"]) for row in outputs["adapter"]),
        "reload_parity": sum(int(row["input_tokens"]) for row in parity),
    }
    for field, expected in expected_counts.items():
        if type(counts[field]) is not int or counts[field] < expected:
            raise ValueError(
                "fresh evaluation completed forward count differs from retained outputs"
            )
        if passed and counts[field] != expected:
            raise ValueError("passed fresh evaluation count differs from retained outputs")
        if not passed and counts[field] > expected + 1:
            raise ValueError("failed fresh evaluation has more than one unretained forward")
        if (
            type(tokens[field]) is not int
            or tokens[field] < expected_tokens[field]
            or tokens[field] > counts[field] * 2048
        ):
            raise ValueError("fresh evaluation input token count is malformed")
        if passed and tokens[field] != expected_tokens[field]:
            raise ValueError("passed fresh evaluation token count differs from retained outputs")
        normalized_counts[field] = counts[field]
        normalized_tokens[field] = tokens[field]
    total = sum(normalized_counts.values())
    total_tokens = sum(normalized_tokens.values())
    if (
        counts["total"] != total
        or tokens["total"] != total_tokens
        or total > core.MAX_TOTAL_FORWARDS
    ):
        raise ValueError("fresh evaluation total counts differ from retained evidence")
    if total_tokens > core.MAX_TOTAL_INPUT_TOKENS:
        raise ValueError("fresh evaluation total input tokens exceed the fixed budget")
    normalized_counts["total"] = total
    normalized_tokens["total"] = total_tokens
    if passed and normalized_counts != {
        "base_evaluation": 1400,
        "training": 0,
        "final_evaluation": 1400,
        "reload_parity": 24,
        "total": 2824,
    }:
        raise ValueError("passed fresh evaluation counts differ from the frozen plan")
    return normalized_counts, normalized_tokens


def _validate_provenance(
    value: object, payload: Mapping[str, object], *, passed: bool
) -> dict[str, object]:
    provenance = normalize_json_object(value, "fresh evaluation provenance")
    required = {
        "model_id",
        "model_revision",
        "versions",
        "source_file_sha256",
        "measured_source_file_sha256",
        "base_model",
        "cuda_device",
        "reload_tokenizer_file_sha256",
    }
    if set(provenance) != required:
        raise ValueError("fresh evaluation provenance has an unexpected schema")
    if (
        provenance["model_id"] != payload["pins"]["model_id"]
        or provenance["model_revision"] != payload["pins"]["model_revision"]
        or provenance["source_file_sha256"] != payload["pins"]["source_file_sha256"]
    ):
        raise ValueError("fresh evaluation provenance differs from frozen model or source pins")
    measured = provenance["measured_source_file_sha256"]
    if passed and measured != payload["pins"]["source_file_sha256"]:
        raise ValueError("passed fresh evaluation is missing measured remote source pins")
    versions = provenance["versions"]
    if passed and versions != mixture_training_contracts.RUNTIME_VERSION_PINS:
        raise ValueError("passed fresh evaluation runtime versions differ from the pinned image")
    base = provenance["base_model"]
    if passed:
        base = normalize_json_object(base, "fresh evaluation base-model provenance")
        hashes = base.get("tokenizer_file_sha256")
        if (
            base.get("model_id") != payload["pins"]["model_id"]
            or base.get("model_revision") != payload["pins"]["model_revision"]
            or base.get("effective_dtype") != "torch.bfloat16"
            or not isinstance(hashes, Mapping)
            or hashes.get("tokenizer.json") != payload["pins"]["tokenizer_file_sha256"]
            or provenance["reload_tokenizer_file_sha256"] != hashes
            or not isinstance(provenance["cuda_device"], str)
            or not provenance["cuda_device"].strip()
        ):
            raise ValueError("passed fresh evaluation base-model provenance is incomplete")
    return provenance


def _validate_adapter_identity(
    value: object, payload: Mapping[str, object], *, passed: bool
) -> dict[str, object] | None:
    if value is None and not passed:
        return None
    identity = normalize_json_object(value, "adapter_identity")
    required = {"snapshot_path", "files_sha256", "tensor_sha256", "reloaded_tensor_sha256", "dtype"}
    if set(identity) != required:
        raise ValueError("adapter identity has an unexpected schema")
    snapshot = payload["selection"]["snapshot"]
    if (
        identity["snapshot_path"] != snapshot["path"]
        or identity["files_sha256"] != snapshot["files_sha256"]
        or identity["tensor_sha256"] != payload["pins"]["adapter_tensor_sha256"]
        or identity["dtype"] != "torch.float32"
    ):
        raise ValueError("adapter identity differs from the selected saved adapter")
    reloaded = identity["reloaded_tensor_sha256"]
    if reloaded is not None:
        validate_sha256(reloaded, "reloaded adapter tensor SHA-256")
    if passed and reloaded != identity["tensor_sha256"]:
        raise ValueError("fresh reload tensor digest differs from the selected adapter")
    return identity


def validate_result(
    value: object, payload: object, *, root: str | Path | None = None
) -> dict[str, Any]:
    """Fail closed on malformed, incomplete, or budget-breaking remote receipts."""

    frozen = core.validate_payload(payload, root=root)
    result = normalize_json_object(value, "fresh evaluation result")
    if set(result) != _RESULT_FIELDS:
        raise ValueError("fresh evaluation result has an unexpected schema")
    for field in (
        "schema_version",
        "experiment_id",
        "run_id",
        "nonce",
        "selection_sha256",
        "payload_sha256",
    ):
        if result[field] != frozen[field]:
            raise ValueError(f"fresh evaluation result {field} differs from its payload")
    passed = result["status"] == "passed"
    if (
        result["status"] not in {"passed", "failed"}
        or not isinstance(result["phase"], str)
        or not result["phase"]
    ):
        raise ValueError("fresh evaluation result status or phase is invalid")
    failure = result["failure"]
    failure_stage: str | None = None
    if passed:
        if result["phase"] != "completed" or failure is not None:
            raise ValueError("passed fresh evaluation result is incomplete")
    else:
        failure = normalize_json_object(failure, "fresh evaluation failure")
        if set(failure) != {"stage", "type", "message"} or any(
            not isinstance(failure[key], str) or not failure[key].strip() for key in failure
        ):
            raise ValueError("failed fresh evaluation result lacks failure evidence")
        failure_stage = failure["stage"]
    evidence = normalize_json_object(result["evidence"], "fresh evaluation evidence")
    if set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("fresh evaluation evidence has an unexpected schema")
    output_value = normalize_json_object(evidence["outputs"], "fresh evaluation outputs")
    if set(output_value) != {"base", "adapter"}:
        raise ValueError("fresh evaluation outputs must include base and adapter")
    outputs = {
        state: _validate_output_rows(
            output_value[state],
            frozen["presentations"],
            frozen["compiled_requests"],
            label=state,
            complete=passed,
        )
        for state in ("base", "adapter")
    }
    parity_value = normalize_json_object(
        evidence["reload_parity"], "fresh evaluation reload parity"
    )
    if set(parity_value) != {"outputs", "max_candidate_logit_delta"}:
        raise ValueError("fresh evaluation reload parity has an unexpected schema")
    parity_compiled = [
        row
        for row in frozen["compiled_requests"]
        if row["presentation_id"] in {p["presentation_id"] for p in frozen["parity_presentations"]}
    ]
    parity = _validate_output_rows(
        parity_value["outputs"],
        frozen["parity_presentations"],
        parity_compiled,
        label="reload parity",
        complete=passed,
    )
    counts, tokens = _validate_counts(
        evidence["forward_counts"], evidence["input_token_counts"], outputs, parity, passed=passed
    )
    identity = _validate_adapter_identity(evidence["adapter_identity"], frozen, passed=passed)
    unknown = validate_unknown_work(
        evidence["unknown_work"], failure_stage=failure_stage, passed=passed
    )
    if passed:
        adapter_by_id = {row["presentation_id"]: row for row in outputs["adapter"]}
        deltas: list[float] = []
        for row in parity:
            expected = adapter_by_id.get(row["presentation_id"])
            if expected is None or row["winner_option_id"] != expected["winner_option_id"]:
                raise ValueError("fresh reload winner differs from the primary adapter result")
            deltas.extend(
                abs(float(a) - float(b))
                for a, b in zip(row["candidate_logits"], expected["candidate_logits"], strict=True)
            )
        measured = max(deltas, default=0.0)
        if (
            _finite(parity_value["max_candidate_logit_delta"], "reload logit delta") != measured
            or measured > 0.001
        ):
            raise ValueError(
                "fresh reload candidate-score delta differs or exceeds the fixed limit"
            )
    provenance = _validate_provenance(result["provenance"], frozen, passed=passed)
    return {
        **result,
        "failure": failure,
        "provenance": provenance,
        "evidence": {
            **evidence,
            "forward_counts": counts,
            "input_token_counts": tokens,
            "outputs": outputs,
            "reload_parity": {
                "outputs": parity,
                "max_candidate_logit_delta": parity_value["max_candidate_logit_delta"],
            },
            "adapter_identity": identity,
            "unknown_work": unknown,
        },
    }
