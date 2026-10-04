"""CPU helpers for a bounded, one-shot remote inference smoke."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .rendering import CompiledRequest, Tokenizer, compile_request
from .schema import DecisionRequest, Option
from .scoring import DecisionResult, score_candidates

MAX_INPUT_TOKENS = 2048
MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
LOGIT_TOLERANCE = 0.125
PROBABILITY_TOLERANCE = 0.02
MAX_FAILURE_MESSAGE_CHARS = 2000

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_AUTHORIZATION_PATTERN = re.compile(
    r"\b(authorization\s*[:=]\s*)(?:bearer|basic)\s+[^\s,;)}\]]+", re.IGNORECASE
)
_ASSIGNMENT_PATTERN = re.compile(
    r"\b(?:[\w.-]*(?:token|secret|password|passwd)|api[_-]?key|authorization)"
    r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;)}\]]+)",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"\b(bearer|basic)\s+[^\s,;)}\]]+", re.IGNORECASE)
_HF_TOKEN_PATTERN = re.compile(r"\bhf_[A-Za-z0-9_-]+\b")


@dataclass(frozen=True, slots=True)
class PaddedInputs:
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    position_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class ParityCheck:
    passed: bool
    logit_differences: tuple[float | None, ...]
    probability_differences: tuple[float | None, ...]
    max_abs_logit_difference: float | None
    max_abs_probability_difference: float | None
    top_id_agreement: bool | None
    logit_tolerance: float
    probability_tolerance: float
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArtifactReservation:
    destination: Path
    lock_path: Path


def sanitize_exception_message(exc: BaseException) -> str:
    """Keep useful remote error context while removing common credential forms."""

    try:
        message = str(exc)
    except Exception:
        message = "exception message unavailable"

    def sanitize_url(match: re.Match[str]) -> str:
        raw = match.group(0)
        suffix = ""
        while raw and raw[-1] in ".,;:!?)]}":
            suffix = raw[-1] + suffix
            raw = raw[:-1]
        try:
            parsed = urlsplit(raw)
            safe_netloc = parsed.netloc.rsplit("@", 1)[-1]
            safe_url = urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))
        except ValueError:
            safe_url = "[URL REDACTED]"
        return safe_url + suffix

    message = _URL_PATTERN.sub(sanitize_url, message)
    message = _AUTHORIZATION_PATTERN.sub(r"\1[REDACTED]", message)
    message = _ASSIGNMENT_PATTERN.sub("[REDACTED]", message)
    message = _BEARER_PATTERN.sub(r"\1 [REDACTED]", message)
    message = _HF_TOKEN_PATTERN.sub("[REDACTED]", message)
    return message[:MAX_FAILURE_MESSAGE_CHARS]


def synthetic_requests() -> tuple[DecisionRequest, DecisionRequest, DecisionRequest]:
    """Return fixed 2-option, 16-option, and reversed 2-option requests."""

    two_option = DecisionRequest(
        context="A small synthetic account transfer is awaiting review.",
        question="What should the reviewer do?",
        options=(
            Option(id="hold", label="Hold for review", description="Delay the transfer."),
            Option(id="close", label="Close the alert", description="Allow the transfer."),
        ),
    )
    sixteen_option = DecisionRequest(
        context="A synthetic task has sixteen possible routing destinations.",
        question="Which destination should receive the task?",
        options=tuple(
            Option(id=f"route-{index:02d}", label=f"Route {index:02d}") for index in range(1, 17)
        ),
    )
    reversed_two_option = two_option.model_copy(
        update={"options": tuple(reversed(two_option.options))}
    )
    return two_option, sixteen_option, reversed_two_option


def compile_smoke_request(request: DecisionRequest, tokenizer: Tokenizer) -> CompiledRequest:
    """Compile a smoke input using the repository's unchanged renderer."""

    return compile_request(request, tokenizer, max_tokens=MAX_INPUT_TOKENS)


def left_pad_inputs(sequences: Sequence[Sequence[int]], *, pad_token_id: int) -> PaddedInputs:
    """Left-pad inputs and derive positions from their attention masks."""

    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    if not sequences:
        raise ValueError("at least one input sequence is required")
    normalized: list[tuple[int, ...]] = []
    for sequence in sequences:
        if not sequence:
            raise ValueError("input sequences must not be empty")
        invalid_tokens = (
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in sequence
        )
        if any(invalid_tokens):
            raise ValueError("input token IDs must be nonnegative integers")
        normalized.append(tuple(sequence))

    width = max(len(sequence) for sequence in normalized)
    padded_ids: list[tuple[int, ...]] = []
    masks: list[tuple[int, ...]] = []
    positions: list[tuple[int, ...]] = []
    for sequence in normalized:
        padding = width - len(sequence)
        mask = (0,) * padding + (1,) * len(sequence)
        padded_ids.append((pad_token_id,) * padding + sequence)
        masks.append(mask)
        position_row: list[int] = []
        position = 0
        for attended in mask:
            position_row.append(max(position - 1, 0) if not attended else position)
            position += attended
        positions.append(tuple(position_row))

    if any(mask[-1] != 1 for mask in masks):
        raise ValueError("left padding must leave a real token in the final column")
    return PaddedInputs(tuple(padded_ids), tuple(masks), tuple(positions))


def compare_candidate_logits(
    request: DecisionRequest,
    single_logits: Sequence[float | int],
    batch_logits: Sequence[float | int],
) -> ParityCheck:
    """Compare single and padded-batch logits with fixed BF16 tolerances."""

    reasons: list[str] = []
    if len(single_logits) != len(request.options) or len(batch_logits) != len(request.options):
        reasons.append("candidate_count_mismatch")

    logit_differences = _finite_differences(single_logits, batch_logits)
    if any(value is None for value in logit_differences):
        reasons.append("non_finite_or_invalid_logit")

    single_result: DecisionResult | None = None
    batch_result: DecisionResult | None = None
    try:
        single_result = score_candidates(request, single_logits, model_revision=MODEL_REVISION)
        batch_result = score_candidates(request, batch_logits, model_revision=MODEL_REVISION)
    except (TypeError, ValueError, OverflowError):
        reasons.append("candidate_scores_invalid")

    probability_differences: tuple[float | None, ...] = ()
    top_id_agreement: bool | None = None
    if single_result is not None and batch_result is not None:
        probability_differences = tuple(
            _finite_difference(single.probability, batch.probability)
            for single, batch in zip(single_result.scores, batch_result.scores, strict=True)
        )
        if any(value is None for value in probability_differences):
            reasons.append("non_finite_or_invalid_probability")
        top_id_agreement = single_result.top_option_id == batch_result.top_option_id

    max_logit_difference = _maximum_absolute(logit_differences)
    max_probability_difference = _maximum_absolute(probability_differences)
    if max_logit_difference is not None and max_logit_difference > LOGIT_TOLERANCE:
        reasons.append("logit_tolerance_exceeded")
    if (
        max_probability_difference is not None
        and max_probability_difference > PROBABILITY_TOLERANCE
    ):
        reasons.append("probability_tolerance_exceeded")
    return ParityCheck(
        passed=not reasons,
        logit_differences=logit_differences,
        probability_differences=probability_differences,
        max_abs_logit_difference=max_logit_difference,
        max_abs_probability_difference=max_probability_difference,
        top_id_agreement=top_id_agreement,
        logit_tolerance=LOGIT_TOLERANCE,
        probability_tolerance=PROBABILITY_TOLERANCE,
        failure_reasons=tuple(reasons),
    )


def _finite_difference(left: object, right: object) -> float | None:
    if (
        isinstance(left, bool)
        or not isinstance(left, Real)
        or isinstance(right, bool)
        or not isinstance(right, Real)
    ):
        return None
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return None
    difference = right_value - left_value
    return difference if math.isfinite(difference) else None


def _finite_differences(
    left: Sequence[float | int], right: Sequence[float | int]
) -> tuple[float | None, ...]:
    if len(left) != len(right):
        return ()
    return tuple(_finite_difference(a, b) for a, b in zip(left, right, strict=True))


def _maximum_absolute(differences: Sequence[float | None]) -> float | None:
    finite = [abs(value) for value in differences if value is not None]
    return max(finite) if finite else None


@contextmanager
def reserve_output(path: str | Path) -> Iterator[ArtifactReservation]:
    """Reserve a new output and prove its parent can hold an atomic artifact."""

    destination = Path(path).expanduser().absolute()
    parent = destination.parent
    if not parent.is_dir():
        raise OSError("output parent directory must already exist")
    if os.path.lexists(destination):
        raise FileExistsError("output destination already exists")
    lock_path = parent / f".{destination.name}.reflex-smoke.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise FileExistsError("output destination is already reserved") from None
    os.close(descriptor)
    probe_path: Path | None = None
    try:
        if os.path.lexists(destination):
            raise FileExistsError("output destination already exists")
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.preflight-", dir=parent, delete=False
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(b"reflex smoke output preflight\n")
            probe.flush()
            os.fsync(probe.fileno())
        probe_path.unlink()
        yield ArtifactReservation(destination, lock_path)
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def write_json_artifact(reservation: ArtifactReservation, result: dict[str, object]) -> None:
    """Atomically create a strict JSON artifact without replacing any path."""

    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{reservation.destination.name}.",
            suffix=".tmp",
            dir=reservation.destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, reservation.destination)
        temporary_path.unlink()
        directory_fd = os.open(reservation.destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
