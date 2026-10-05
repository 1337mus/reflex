"""CPU validators for scored outputs and frozen forward-count receipts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from experiments.mixture_training_contracts import (
    MAX_INPUT_TOKENS,
    MAX_TOTAL_FORWARDS,
    canonical_json,
    normalize_json_object,
    validate_sha256,
)
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest

_COUNT_KEYS = {
    "base_evaluation",
    "training",
    "final_evaluation",
    "reload_parity",
    "total",
}
_OUTPUT_KEYS = {
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


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer greater than or equal to {minimum}")
    return value


def _finite_number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{label} is outside its finite range")
    return number


def _expected_output_identity(presentation: Mapping[str, object]) -> dict[str, object]:
    return {
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


def _validate_output_rows(
    value: object,
    presentations: Sequence[Mapping[str, object]],
    *,
    label: str,
    require_complete: bool,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    if require_complete and len(value) != len(presentations):
        raise ValueError(f"{label} does not contain every expected presentation")
    if len(value) > len(presentations):
        raise ValueError(f"{label} contains more rows than its evaluation panel")
    normalized: list[dict[str, object]] = []
    for index, item in enumerate(value):
        row = normalize_json_object(item, label)
        if set(row) != _OUTPUT_KEYS:
            raise ValueError(f"{label} row has an unexpected schema")
        expected = _expected_output_identity(presentations[index])
        actual = {key: row[key] for key in expected}
        if actual != expected:
            raise ValueError(f"{label} rows do not match evaluation identities and order")
        order_ids = expected["order_ids"]
        logits = row["candidate_logits"]
        if (
            not isinstance(order_ids, list)
            or not isinstance(logits, list)
            or len(logits) != len(order_ids)
        ):
            raise ValueError(f"{label} candidate logits do not match option dimensions")
        scores = [_finite_number(score, f"{label} candidate logit") for score in logits]
        winner = row["winner_option_id"]
        if not isinstance(winner, str) or not winner:
            raise ValueError(f"{label} winner_option_id must be a nonempty string")
        maximum = max(scores)
        expected_winner = min(
            option_id
            for option_id, score in zip(order_ids, scores, strict=True)
            if score == maximum
        )
        if winner != expected_winner:
            raise ValueError(f"{label} winner violates the lexicographic tie rule")
        token_count = _integer(row["input_tokens"], f"{label} input_tokens", minimum=1)
        if token_count > MAX_INPUT_TOKENS:
            raise ValueError(f"{label} input token count exceeds the frozen limit")
        prompt_hash = validate_sha256(row["prompt_sha256"], f"{label} prompt_sha256")
        request = DecisionRequest.model_validate(presentations[index]["request"])
        expected_prompt_hash = hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest()
        if prompt_hash != expected_prompt_hash:
            raise ValueError(f"{label} prompt hash differs from the canonical rendered request")
        normalized.append(row)
    return normalized


def _validate_artifact(
    value: object,
    outputs: Sequence[Mapping[str, object]],
    run_id: str,
    *,
    required: bool,
) -> dict[str, object] | None:
    if value is None and not required:
        return None
    artifact = normalize_json_object(value, "output_artifact")
    if set(artifact) != {"path", "sha256"}:
        raise ValueError("output_artifact has an unexpected schema")
    path = artifact["path"]
    if not isinstance(path, str) or not path.startswith(f"/artifacts/runs/{run_id}/"):
        raise ValueError("output artifact path must be inside this immutable run directory")
    parts = PurePosixPath(path).parts
    if ".." in parts or not PurePosixPath(path).is_absolute():
        raise ValueError("output artifact path is unsafe")
    digest = validate_sha256(artifact["sha256"], "output artifact SHA-256")
    expected = hashlib.sha256(canonical_json(list(outputs)) + b"\n").hexdigest()
    if digest != expected:
        raise ValueError("output artifact SHA-256 differs from canonical output bytes")
    return {"path": path, "sha256": digest}


def _validate_counts(
    value: object, token_value: object, *, passed_phase: str | None
) -> tuple[dict[str, int | None], dict[str, int | None]]:
    counts = normalize_json_object(value, "forward_counts")
    tokens = normalize_json_object(token_value, "input_token_counts")
    if set(counts) != _COUNT_KEYS or set(tokens) != _COUNT_KEYS:
        raise ValueError("forward or token counts have an unexpected schema")
    normalized_counts: dict[str, int | None] = {}
    normalized_tokens: dict[str, int | None] = {}
    for key in _COUNT_KEYS - {"total"}:
        count, token_count = counts[key], tokens[key]
        if count is not None:
            count = _integer(count, f"forward_counts.{key}")
            if count > MAX_TOTAL_FORWARDS:
                raise ValueError("forward count exceeds the experiment-wide limit")
        if token_count is not None:
            token_count = _integer(token_count, f"input_token_counts.{key}")
            if count is None:
                raise ValueError("token sum cannot be known when its forward count is unknown")
            if (count == 0 and token_count != 0) or token_count > count * MAX_INPUT_TOKENS:
                raise ValueError("input token sum is inconsistent with its completed forwards")
            if count > 0 and token_count < count:
                raise ValueError("each completed forward must consume at least one input token")
        normalized_counts[key] = count
        normalized_tokens[key] = token_count
    total = counts["total"]
    total_tokens = tokens["total"]
    if total is not None:
        total = _integer(total, "forward_counts.total")
        known_forwards = sum(item for item in normalized_counts.values() if item is not None)
        if total < known_forwards or total > MAX_TOTAL_FORWARDS:
            raise ValueError("total forwards do not match the known counts or frozen maximum")
        if all(item is not None for item in normalized_counts.values()) and total != known_forwards:
            raise ValueError("total forward count differs from its completed categories")
    if total_tokens is not None:
        total_tokens = _integer(total_tokens, "input_token_counts.total")
        known_tokens = sum(item for item in normalized_tokens.values() if item is not None)
        known_forwards = (
            total
            if total is not None
            else sum(item for item in normalized_counts.values() if item is not None)
        )
        if total_tokens < known_tokens or total_tokens > known_forwards * MAX_INPUT_TOKENS:
            raise ValueError("total token sum is inconsistent with completed forward evidence")
        if (
            all(item is not None for item in normalized_tokens.values())
            and total_tokens != known_tokens
        ):
            raise ValueError("total token sum differs from its completed categories")
    normalized_counts["total"] = total
    normalized_tokens["total"] = total_tokens
    if passed_phase == "initialize":
        expected = {
            "base_evaluation": 939,
            "training": 0,
            "final_evaluation": 0,
            "reload_parity": 0,
            "total": 939,
        }
        if normalized_counts != expected:
            raise ValueError("passed initialization forward counts differ from the frozen plan")
    elif passed_phase == "train":
        expected = {
            "base_evaluation": 0,
            "training": 1008,
            "final_evaluation": 4023,
            "reload_parity": 32,
            "total": 5063,
        }
        if normalized_counts != expected:
            raise ValueError("passed training forward counts differ from the frozen plan")
    if passed_phase is not None and any(
        value is None for value in (*normalized_counts.values(), *normalized_tokens.values())
    ):
        raise ValueError("passed results must retain all forward and input token counts")
    return normalized_counts, normalized_tokens


__all__ = ["_validate_artifact", "_validate_counts", "_validate_output_rows"]
