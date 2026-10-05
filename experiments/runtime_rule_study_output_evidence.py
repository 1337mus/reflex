"""Pure CPU validation for runtime-rule scoring and reload evidence."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import cast

from experiments import mixture_training_contracts as json_contracts
from experiments import runtime_rule_study_contracts as contracts
from experiments.mixture_training_outputs import _validate_output_rows
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest

_OUTPUT_IDENTITY_FIELDS = tuple(
    "presentation_id record_id dataset_id source_group_id "
    "request_hash order_index order_ids".split()
)
_SPEC_IDENTITY_FIELDS = ("presentation_id", "record_id", "request_hash")
_RELOAD_FIELDS = frozenset(
    "outputs tensor_equal tensor_sha256 winner_match_count max_candidate_logit_difference".split()
)
_FINAL_EVALUATION_COUNT = 3_946


def initial_reload_evidence() -> dict[str, object]:
    return {"outputs": [], **dict.fromkeys(_RELOAD_FIELDS - {"outputs"})}


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _normalized_descriptors(
    presentations: object, specs: object, *, label: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_presentations = _list(presentations, f"{label} presentations")
    raw_specs = _list(specs, f"{label} specs")
    if len(raw_presentations) != len(raw_specs):
        raise ValueError(f"{label} presentation and spec counts differ")

    normalized_presentations: list[dict[str, object]] = []
    normalized_specs: list[dict[str, object]] = []
    for index, (raw_presentation, raw_spec) in enumerate(
        zip(raw_presentations, raw_specs, strict=True)
    ):
        presentation = json_contracts.normalize_json_object(
            raw_presentation, f"{label} presentation {index}"
        )
        spec = json_contracts.normalize_json_object(raw_spec, f"{label} spec {index}")
        if not set((*_OUTPUT_IDENTITY_FIELDS, "request")).issubset(presentation):
            raise ValueError(f"{label} presentation {index} is missing output identity fields")
        if not set((*_SPEC_IDENTITY_FIELDS, "input_tokens", "prompt_sha256")).issubset(spec):
            raise ValueError(f"{label} spec {index} is missing compiled identity fields")

        if (
            any(
                not isinstance(presentation[key], str) or not presentation[key]
                for key in _OUTPUT_IDENTITY_FIELDS[:5]
            )
            or type(presentation["order_index"]) is not int
            or presentation["order_index"] < 0
            or not isinstance(presentation["order_ids"], list)
            or any(
                not isinstance(option_id, str) or not option_id
                for option_id in presentation["order_ids"]
            )
        ):
            raise ValueError(f"{label} presentation {index} has invalid output identity fields")
        spec_identity = {key: spec[key] for key in _SPEC_IDENTITY_FIELDS}
        presentation_identity = {key: presentation[key] for key in _SPEC_IDENTITY_FIELDS}
        if json_contracts.canonical_json(spec_identity) != json_contracts.canonical_json(
            presentation_identity
        ):
            raise ValueError(f"{label} spec {index} identity differs from its presentation")

        try:
            request = DecisionRequest.model_validate(presentation["request"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} presentation {index} request is invalid") from exc
        if (
            request.model_dump(mode="json") != presentation["request"]
            or request.request_hash != presentation["request_hash"]
            or [option.id for option in request.options] != presentation["order_ids"]
        ):
            raise ValueError(f"{label} presentation {index} request identity or order is invalid")

        if (
            type(spec["input_tokens"]) is not int
            or not 1 <= spec["input_tokens"] <= contracts.MAX_INPUT_TOKENS
        ):
            raise ValueError(f"{label} spec {index} input_tokens is outside the fixed limit")
        expected_prompt_sha256 = hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest()
        if (
            json_contracts.validate_sha256(
                spec["prompt_sha256"], f"{label} spec {index} prompt_sha256"
            )
            != expected_prompt_sha256
        ):
            raise ValueError(f"{label} spec {index} prompt hash differs from its request")

        normalized_presentations.append(presentation)
        normalized_specs.append(spec)
    if len({p["presentation_id"] for p in normalized_presentations}) < len(
        normalized_presentations
    ):
        raise ValueError(f"{label} presentation IDs must be unique")
    return normalized_presentations, normalized_specs


def _execution_status(
    completed_forwards: object, passed: object, *, maximum: int, label: str
) -> tuple[int, bool]:
    if type(completed_forwards) is not int or not 0 <= completed_forwards <= maximum:
        raise ValueError(
            f"{label} completed_forwards must be a strict integer from zero to {maximum}"
        )
    if type(passed) is not bool:
        raise ValueError(f"{label} passed must be a strict boolean")
    return completed_forwards, passed


def _check_output_length(
    actual: int, completed: int, planned: int, passed: bool, *, label: str
) -> None:
    allowed = {completed} | ({completed - 1} if completed else set())
    if passed and (completed != planned or actual != planned):
        raise ValueError(f"passed {label} must match the complete panel and forward count")
    if not passed and actual not in allowed:
        raise ValueError(f"failed {label} must be a completed prefix or one terminal row short")


def _check_canonical_identities(
    outputs: Sequence[object], presentations: Sequence[Mapping[str, object]], *, label: str
) -> None:
    for raw_output, presentation in zip(outputs, presentations, strict=False):
        if not isinstance(raw_output, Mapping):
            raise ValueError(f"{label} row must be an object")
        expected = {key: presentation[key] for key in _OUTPUT_IDENTITY_FIELDS}
        actual = {key: raw_output.get(key) for key in _OUTPUT_IDENTITY_FIELDS}
        if json_contracts.canonical_json(actual) != json_contracts.canonical_json(expected):
            raise ValueError(f"{label} identity differs canonically from its presentation")


def _check_output_specs(
    outputs: Sequence[Mapping[str, object]], specs: Sequence[Mapping[str, object]], *, label: str
) -> None:
    for index, (row, spec) in enumerate(zip(outputs, specs, strict=True)):
        if row["input_tokens"] != spec["input_tokens"]:
            raise ValueError(f"{label} row {index} input_tokens differ from its compiled spec")
        if row["prompt_sha256"] != spec["prompt_sha256"]:
            raise ValueError(f"{label} row {index} prompt hash differs from its compiled spec")


def validate_scored_outputs(
    value: object,
    *,
    presentations: object,
    specs: object,
    completed_forwards: object,
    passed: object,
) -> list[dict[str, object]]:
    """Validate detached scored output rows against their ordered panel and specs."""

    expected, compiled = _normalized_descriptors(presentations, specs, label="scored output")
    completed, succeeded = _execution_status(
        completed_forwards, passed, maximum=len(expected), label="scored outputs"
    )
    raw_outputs = _list(value, "scored outputs")
    _check_output_length(
        len(raw_outputs), completed, len(expected), succeeded, label="scored outputs"
    )
    _check_canonical_identities(raw_outputs, expected, label="scored outputs")

    normalized = _validate_output_rows(
        raw_outputs,
        expected,
        label="scored outputs",
        require_complete=succeeded,
    )
    _check_output_specs(normalized, compiled[: len(normalized)], label="scored outputs")
    return normalized


def _strict_nonnegative_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _matched_final_outputs(
    value: object,
    presentations: Sequence[Mapping[str, object]],
    specs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    final_outputs = _list(value, "final evaluation outputs")
    if len(final_outputs) != _FINAL_EVALUATION_COUNT:
        raise ValueError("reload evidence requires the complete final evaluation outputs")
    final_by_id: dict[str, Mapping[str, object]] = {}
    for index, raw_output in enumerate(final_outputs):
        if not isinstance(raw_output, Mapping):
            raise ValueError(f"final evaluation output {index} must be an object")
        presentation_id = raw_output.get("presentation_id")
        if not isinstance(presentation_id, str) or not presentation_id:
            raise ValueError("final evaluation outputs require nonempty presentation IDs")
        if presentation_id in final_by_id:
            raise ValueError("final evaluation presentation IDs must be unique")
        final_by_id[presentation_id] = raw_output

    counterparts: list[Mapping[str, object]] = []
    for presentation in presentations:
        presentation_id = cast(str, presentation["presentation_id"])
        counterpart = final_by_id.get(presentation_id)
        if counterpart is None:
            raise ValueError("final evaluation is missing a planned reload presentation")
        counterparts.append(counterpart)

    _check_canonical_identities(counterparts, presentations, label="final reload counterpart")
    validated = _validate_output_rows(
        counterparts,
        presentations,
        label="final reload counterparts",
        require_complete=True,
    )
    _check_output_specs(validated, specs, label="final reload counterparts")
    return validated


def validate_reload_evidence(
    value: object,
    *,
    presentations: object,
    specs: object,
    final_outputs: object,
    final_tensor_sha256: object,
    completed_forwards: object,
    passed: object,
) -> dict[str, object]:
    """Validate detached reload rows and their measured parity with final evaluation."""

    expected, compiled = _normalized_descriptors(presentations, specs, label="reload")
    if len(expected) != contracts.RELOAD_PARITY_COUNT:
        raise ValueError("reload presentations must contain the fixed 32-row plan")
    completed, succeeded = _execution_status(
        completed_forwards, passed, maximum=len(expected), label="reload"
    )

    evidence = json_contracts.normalize_json_object(value, "reload evidence")
    if set(evidence) != _RELOAD_FIELDS:
        raise ValueError("reload evidence has an unexpected schema")
    outputs_value = _list(evidence["outputs"], "reload outputs")
    tensor_equal = evidence["tensor_equal"]
    tensor_sha256 = evidence["tensor_sha256"]
    winner_match_count = evidence["winner_match_count"]
    max_difference = evidence["max_candidate_logit_difference"]

    final_digest = None
    if final_tensor_sha256 is not None:
        final_digest = json_contracts.validate_sha256(final_tensor_sha256, "final_tensor_sha256")
    if succeeded and final_digest is None:
        raise ValueError("passed reload requires a valid final tensor digest")
    if succeeded and final_digest == contracts.SELECTED_TENSOR_SHA256:
        raise ValueError(
            "passed reload final tensor digest must differ from the selected starting tensor"
        )

    if tensor_equal is not None and type(tensor_equal) is not bool:
        raise ValueError("reload tensor_equal must be a strict boolean or null")
    if tensor_sha256 is not None:
        tensor_sha256 = json_contracts.validate_sha256(tensor_sha256, "reload tensor_sha256")
    if (tensor_equal is None) != (tensor_sha256 is None):
        raise ValueError("reload tensor equality and digest observations must appear together")
    if tensor_equal is True and (final_digest is None or tensor_sha256 != final_digest):
        raise ValueError("true reload tensor equality requires the known final tensor digest")

    if winner_match_count is not None and (
        type(winner_match_count) is not int or not 0 <= winner_match_count <= len(expected)
    ):
        raise ValueError("reload winner_match_count must be a strict integer from 0 to 32")
    measured_difference = (
        None
        if max_difference is None
        else _strict_nonnegative_number(
            max_difference, label="reload max_candidate_logit_difference"
        )
    )
    if not outputs_value and (winner_match_count is not None or measured_difference is not None):
        raise ValueError("reload summaries cannot be present without scored reload outputs")

    evidence_started = bool(outputs_value or completed or succeeded) or any(
        item is not None
        for item in (tensor_equal, tensor_sha256, winner_match_count, measured_difference)
    )
    if evidence_started and final_digest is None:
        raise ValueError("reload evidence requires the valid final training tensor digest")
    counterparts = (
        _matched_final_outputs(final_outputs, expected, compiled) if evidence_started else []
    )

    _check_output_length(len(outputs_value), completed, len(expected), succeeded, label="reload")
    if succeeded:
        if tensor_equal is not True or tensor_sha256 != final_digest:
            raise ValueError("passed reload must confirm exact final tensor equality")
        if winner_match_count != len(expected) or measured_difference is None:
            raise ValueError("passed reload requires complete winner and logit summaries")
        if measured_difference > contracts.MAX_CANDIDATE_SCORE_DIFFERENCE:
            raise ValueError("passed reload logit difference exceeds the fixed tolerance")
    _check_canonical_identities(outputs_value, expected, label="reload outputs")

    normalized_outputs = _validate_output_rows(
        outputs_value,
        expected,
        label="reload outputs",
        require_complete=succeeded,
    )
    _check_output_specs(
        normalized_outputs, compiled[: len(normalized_outputs)], label="reload outputs"
    )

    if normalized_outputs:
        matched = 0
        maximum = 0.0
        pairs = zip(normalized_outputs, counterparts[: len(normalized_outputs)], strict=True)
        for row, final_row in pairs:
            matched += row["winner_option_id"] == final_row["winner_option_id"]
            for reload_logit, final_logit in zip(
                cast(list[int | float], row["candidate_logits"]),
                cast(list[int | float], final_row["candidate_logits"]),
                strict=True,
            ):
                difference = abs(float(reload_logit) - float(final_logit))
                if not math.isfinite(difference):
                    raise ValueError("reload candidate logit difference is nonfinite")
                maximum = max(maximum, difference)
        if winner_match_count is not None and winner_match_count != matched:
            raise ValueError("reload winner summary differs from measured outputs")
        if measured_difference is not None and measured_difference != maximum:
            raise ValueError("reload logit summary differs from measured outputs")

    evidence["outputs"] = normalized_outputs
    return evidence
