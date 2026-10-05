"""Strict result validation for the adapter-transfer experiment."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments import adapter_transfer_contracts as contracts
from experiments import adapter_transfer_core as core
from experiments.mixture_training_contracts import normalize_json_object, validate_sha256
from experiments.mixture_training_evidence import _validate_provenance
from experiments.mixture_training_outputs import _validate_counts, _validate_output_rows

_COUNT_FIELDS = {
    "base_evaluation",
    "training",
    "final_evaluation",
    "reload_parity",
    "total",
}
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
}
_ADAPTER_IDENTITY_FIELDS = {
    "snapshot_path",
    "files_sha256",
    "tensor_sha256",
    "reloaded_tensor_sha256",
    "dtype",
}
_OUTPUT_STATES = {"base", "adapter"}
_FAILURE_FIELDS = {"stage", "type", "message"}
_STAGE = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")


def validate_result(
    value: object, payload: object, *, root: str | Path | None = None
) -> dict[str, Any]:
    """Validate a complete result or a bounded failure with honest partial evidence."""

    normalized_payload = core.validate_payload(payload, root=root)
    result = normalize_json_object(value, "result")
    if set(result) != _RESULT_FIELDS:
        raise ValueError("result has an unexpected top-level schema")
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != contracts.SCHEMA_VERSION
    ):
        raise ValueError("result schema version is unsupported")
    for name in (
        "experiment_id",
        "run_id",
        "nonce",
        "selection_sha256",
        "payload_sha256",
    ):
        if result[name] != normalized_payload[name]:
            raise ValueError(f"result {name} does not match its payload")
    if result["experiment_id"] != contracts.EXPERIMENT_ID:
        raise ValueError("result experiment_id is unsupported")
    status = result["status"]
    if status not in {"passed", "failed"}:
        raise ValueError("result status must be passed or failed")
    passed = status == "passed"
    phase = result["phase"]
    if not isinstance(phase, str) or not phase or len(phase) > 64:
        raise ValueError("result phase must be a bounded nonblank string")
    failure = result["failure"]
    if passed:
        if phase != "completed" or failure is not None:
            raise ValueError("passed result must be completed and contain no failure")
    else:
        failure = normalize_json_object(failure, "failure")
        if set(failure) != _FAILURE_FIELDS or any(
            not isinstance(failure[key], str) or not failure[key].strip() for key in _FAILURE_FIELDS
        ):
            raise ValueError("failed result requires stage, type, and message")
        if phase != f"{failure['stage']}_failed"[:64]:
            raise ValueError("failed result phase does not match its last failure stage")
        result["failure"] = failure

    provenance = _validate_provenance(
        result["provenance"], normalized_payload, passed=passed, train=False
    )
    _validate_transfer_provenance(provenance, passed=passed)
    evidence = normalize_json_object(result["evidence"], "evidence")
    if set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("result evidence has an unexpected schema")
    evidence["forward_counts"], evidence["input_token_counts"] = _validate_transfer_counts(
        evidence["forward_counts"], evidence["input_token_counts"], passed=passed
    )

    outputs_value = normalize_json_object(evidence["outputs"], "outputs")
    if set(outputs_value) != _OUTPUT_STATES:
        raise ValueError("result outputs must contain base and adapter rows")
    expected_presentations = normalized_payload["presentations"]
    outputs = {
        state: _validate_output_rows(
            outputs_value[state],
            expected_presentations,
            label=f"outputs.{state}",
            require_complete=passed,
        )
        for state in ("base", "adapter")
    }
    parity_value = normalize_json_object(evidence["reload_parity"], "reload_parity")
    if set(parity_value) != {"outputs", "max_candidate_logit_delta"}:
        raise ValueError("reload parity evidence has an unexpected schema")
    parity_rows = _validate_output_rows(
        parity_value["outputs"],
        normalized_payload["parity_presentations"],
        label="reload_parity.outputs",
        require_complete=passed,
    )
    _validate_base_adapter_token_counts(outputs, passed=passed)
    evidence["adapter_identity"] = _validate_adapter_identity(
        evidence["adapter_identity"], normalized_payload, required=passed
    )
    delta = parity_value["max_candidate_logit_delta"]
    measured_delta = _validate_reload_parity(outputs, parity_rows, passed=passed)
    if passed:
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
            or float(delta) != measured_delta
            or measured_delta > contracts.MAX_LOGIT_DELTA
        ):
            raise ValueError("reload parity candidate-logit delta exceeds or differs from evidence")
    elif delta is not None:
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
            or float(delta) != measured_delta
        ):
            raise ValueError("partial reload parity delta does not match retained output evidence")

    _validate_retained_counts(
        evidence["forward_counts"],
        evidence["input_token_counts"],
        outputs,
        parity_rows,
        passed=passed,
    )
    if passed:
        expected_counts = {
            "base_evaluation": contracts.BASE_PRESENTATION_COUNT,
            "training": 0,
            "final_evaluation": contracts.ADAPTER_PRESENTATION_COUNT,
            "reload_parity": contracts.PARITY_PRESENTATION_COUNT,
            "total": contracts.MAX_TOTAL_FORWARDS,
        }
        counts = evidence["forward_counts"]
        if counts != expected_counts:
            raise ValueError("passed result forward counts differ from the fixed 272-forward plan")
        if evidence["input_token_counts"]["total"] > contracts.MAX_TOTAL_INPUT_TOKENS:
            raise ValueError("passed result exceeds the fixed input-token ceiling")
    evidence["outputs"] = outputs
    evidence["reload_parity"] = {
        "outputs": parity_rows,
        "max_candidate_logit_delta": delta,
    }
    result["provenance"] = provenance
    result["evidence"] = evidence
    return result


def _validate_transfer_provenance(value: Mapping[str, object], *, passed: bool) -> None:
    base = value.get("base_model")
    if base is not None:
        base = normalize_json_object(base, "base_model provenance")
        if base.get("effective_dtype") != "torch.bfloat16":
            raise ValueError("base model effective dtype must be BF16")
        tokenizer_hashes = base.get("tokenizer_file_sha256")
        if not isinstance(tokenizer_hashes, Mapping) or not tokenizer_hashes:
            raise ValueError("base model must identify the tokenizer files by SHA-256")
        for name, digest in tokenizer_hashes.items():
            if not isinstance(name, str) or not name:
                raise ValueError("tokenizer file names must be nonblank")
            validate_sha256(digest, f"tokenizer {name} SHA-256")
        reload_tokenizer = value.get("reload_tokenizer_file_sha256")
        if reload_tokenizer is not None and dict(reload_tokenizer) != dict(tokenizer_hashes):
            raise ValueError("fresh reload tokenizer hashes differ from the base tokenizer")
        if passed and reload_tokenizer != tokenizer_hashes:
            raise ValueError("passed result is missing fresh-reload tokenizer hash evidence")
    elif passed:
        raise ValueError("passed result is missing BF16 base-model evidence")


def _validate_transfer_counts(
    value: object, token_value: object, *, passed: bool
) -> tuple[dict[str, int | None], dict[str, int | None]]:
    counts, tokens = _validate_counts(value, token_value, passed_phase=None)
    maxima = {
        "base_evaluation": contracts.BASE_PRESENTATION_COUNT,
        "training": 0,
        "final_evaluation": contracts.ADAPTER_PRESENTATION_COUNT,
        "reload_parity": contracts.PARITY_PRESENTATION_COUNT,
    }
    for category, maximum in maxima.items():
        if counts[category] is not None and counts[category] > maximum:
            raise ValueError(f"{category} count exceeds the fixed adapter-transfer plan")
    if counts["training"] != 0 or tokens["training"] != 0:
        raise ValueError("adapter-transfer evaluation cannot perform training forwards")
    if counts["total"] is not None and counts["total"] > contracts.MAX_TOTAL_FORWARDS:
        raise ValueError("total forwards exceed the fixed adapter-transfer ceiling")
    if tokens["total"] is not None and tokens["total"] > contracts.MAX_TOTAL_INPUT_TOKENS:
        raise ValueError("input tokens exceed the fixed adapter-transfer ceiling")
    if passed and any(item is None for item in (*counts.values(), *tokens.values())):
        raise ValueError("passed result must retain every completed count and token sum")
    return counts, tokens


def _validate_adapter_identity(
    value: object, payload: Mapping[str, object], *, required: bool
) -> dict[str, object] | None:
    if value is None and not required:
        return None
    identity = normalize_json_object(value, "adapter_identity")
    if set(identity) != _ADAPTER_IDENTITY_FIELDS:
        raise ValueError("adapter identity has an unexpected schema")
    selection = payload["selection"]
    snapshot = selection["snapshot"]
    if (
        identity["snapshot_path"] != snapshot["path"]
        or identity["files_sha256"] != snapshot["files_sha256"]
        or identity["dtype"] != "torch.float32"
    ):
        raise ValueError("adapter identity differs from selected saved FP32 files")
    tensor_sha = validate_sha256(identity["tensor_sha256"], "saved adapter tensor SHA-256")
    reloaded_value = identity["reloaded_tensor_sha256"]
    reloaded_sha = (
        None
        if reloaded_value is None and not required
        else validate_sha256(reloaded_value, "reloaded adapter tensor SHA-256")
    )
    if required and tensor_sha != reloaded_sha:
        raise ValueError(
            "fresh reload adapter tensor digest differs from the selected saved adapter"
        )
    return {**identity, "tensor_sha256": tensor_sha, "reloaded_tensor_sha256": reloaded_sha}


def _validate_reload_parity(
    outputs: Mapping[str, list[dict[str, object]]],
    parity_rows: Sequence[Mapping[str, object]],
    *,
    passed: bool,
) -> float:
    base_by_id = {row["presentation_id"]: row for row in outputs["base"]}
    adapter_by_id = {row["presentation_id"]: row for row in outputs["adapter"]}
    deltas: list[float] = []
    for row in parity_rows:
        identity = row["presentation_id"]
        expected = adapter_by_id.get(identity)
        base = base_by_id.get(identity)
        if expected is None:
            if passed:
                raise ValueError("reload parity row has no primary adapter output")
            continue
        if passed and (
            row["input_tokens"] != expected["input_tokens"]
            or row["prompt_sha256"] != expected["prompt_sha256"]
            or row["winner_option_id"] != expected["winner_option_id"]
        ):
            raise ValueError(
                "reload parity tokenization, prompt, or winner differs from the adapter"
            )
        if (
            passed
            and base is not None
            and (
                row["input_tokens"] != base["input_tokens"]
                or row["prompt_sha256"] != base["prompt_sha256"]
            )
        ):
            raise ValueError("base and adapter tokenization or rendered prompt hashes differ")
        row_scores = row["candidate_logits"]
        expected_scores = expected["candidate_logits"]
        if not isinstance(row_scores, list) or not isinstance(expected_scores, list):
            raise ValueError("reload parity candidate logits are malformed")
        if len(row_scores) != len(expected_scores):
            raise ValueError("reload parity candidate dimensions differ from adapter output")
        deltas.extend(
            abs(float(left) - float(right))
            for left, right in zip(row_scores, expected_scores, strict=True)
        )
    if passed and len(deltas) != contracts.PARITY_PRESENTATION_COUNT * 2:
        raise ValueError("passed reload parity does not compare every candidate logit")
    return max(deltas, default=0.0)


def _validate_base_adapter_token_counts(
    outputs: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    passed: bool,
) -> None:
    for base, adapter in zip(outputs["base"], outputs["adapter"], strict=False):
        if passed and base["input_tokens"] != adapter["input_tokens"]:
            raise ValueError("base and adapter input token counts differ")


def _validate_retained_counts(
    counts: Mapping[str, int | None],
    tokens: Mapping[str, int | None],
    outputs: Mapping[str, Sequence[Mapping[str, object]]],
    parity: Sequence[Mapping[str, object]],
    *,
    passed: bool,
) -> None:
    retained = {
        "base_evaluation": outputs["base"],
        "training": (),
        "final_evaluation": outputs["adapter"],
        "reload_parity": parity,
    }
    for category, rows in retained.items():
        row_tokens = sum(int(row["input_tokens"]) for row in rows)
        if passed and counts[category] != len(rows):
            raise ValueError(f"{category} forward count differs from retained output evidence")
        if not passed and counts[category] is not None and counts[category] < len(rows):
            raise ValueError(f"{category} count is smaller than retained output evidence")
        if passed and tokens[category] != row_tokens:
            raise ValueError(f"{category} token sum differs from retained output evidence")
        if not passed and tokens[category] is not None and tokens[category] < row_tokens:
            raise ValueError(f"{category} token sum is smaller than retained output evidence")
        if rows and (counts[category] is None or tokens[category] is None):
            raise ValueError(f"{category} output rows require known forward and token counts")
    known_counts = [counts[key] for key in retained if counts[key] is not None]
    known_tokens = [tokens[key] for key in retained if tokens[key] is not None]
    if passed:
        if counts["total"] != sum(known_counts):
            raise ValueError("total forward count differs from retained phase counts")
        if tokens["total"] != sum(known_tokens):
            raise ValueError("total token sum differs from retained phase token sums")
    else:
        if counts["total"] is not None and counts["total"] < sum(known_counts):
            raise ValueError("total forward count is smaller than known phase counts")
        if tokens["total"] is not None and tokens["total"] < sum(known_tokens):
            raise ValueError("total token sum is smaller than known phase token sums")


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

    normalized_payload = core.validate_payload(payload)
    if not isinstance(stage, str) or _STAGE.fullmatch(stage) is None:
        raise ValueError("failure stage must be a bounded identifier")
    message = sanitize(error)
    if not isinstance(message, str) or not message.strip() or len(message) > 500:
        raise ValueError("sanitized failure message must be a bounded nonblank string")
    base_provenance = {
        "model_id": contracts.MODEL_ID,
        "model_revision": contracts.MODEL_REVISION,
        "versions": {},
        "source_file_sha256": normalized_payload["pins"]["source_file_sha256"],
        "measured_source_file_sha256": {},
        "base_model": None,
        "cuda_device": None,
        "reload_tokenizer_file_sha256": None,
    }
    if provenance is not None:
        base_provenance.update(normalize_json_object(provenance, "partial provenance"))
    counts = {key: None for key in _COUNT_FIELDS}
    counts["training"] = 0
    tokens = dict(counts)
    tokens["training"] = 0
    evidence_value: dict[str, object] = {
        "forward_counts": counts,
        "input_token_counts": tokens,
        "outputs": {"base": [], "adapter": []},
        "adapter_identity": None,
        "reload_parity": {"outputs": [], "max_candidate_logit_delta": None},
    }
    if evidence is not None:
        partial = normalize_json_object(evidence, "partial evidence")
        for key, item in partial.items():
            if key in {"forward_counts", "input_token_counts"}:
                merged = dict(evidence_value[key])
                if not isinstance(item, Mapping):
                    raise ValueError(f"partial {key} must be an object")
                merged.update(item)
                evidence_value[key] = merged
            elif key == "outputs":
                merged_outputs = dict(evidence_value[key])
                if not isinstance(item, Mapping):
                    raise ValueError("partial outputs must be an object")
                merged_outputs.update(item)
                evidence_value[key] = merged_outputs
            elif key == "reload_parity":
                merged_parity = dict(evidence_value[key])
                if not isinstance(item, Mapping):
                    raise ValueError("partial reload parity must be an object")
                merged_parity.update(item)
                evidence_value[key] = merged_parity
            else:
                evidence_value[key] = item
    result = {
        "schema_version": contracts.SCHEMA_VERSION,
        "experiment_id": normalized_payload["experiment_id"],
        "run_id": normalized_payload["run_id"],
        "nonce": normalized_payload["nonce"],
        "selection_sha256": normalized_payload["selection_sha256"],
        "payload_sha256": normalized_payload["payload_sha256"],
        "status": "failed",
        "phase": f"{stage}_failed"[:64],
        "provenance": base_provenance,
        "evidence": evidence_value,
        "failure": {"stage": stage, "type": type(error).__name__, "message": message},
    }
    return validate_result(result, normalized_payload)
