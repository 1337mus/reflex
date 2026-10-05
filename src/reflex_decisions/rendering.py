"""Deterministic prompt rendering and tokenizer-boundary checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from .schema import DecisionRequest

_SYMBOLS = "ABCDEFGHIJKLMNOP"
_ANSWER_SUFFIX = "\nAnswer:\n"


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class CompiledRequest:
    prompt: str
    input_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...]
    symbol_to_option_id: tuple[tuple[str, str], ...]
    request_hash: str
    schema_hash: str
    prompt_hash: str


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception as exc:
        raise ValueError(f"tokenizer could not encode prompt text: {exc}") from exc
    if not isinstance(token_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError("tokenizer.encode must return a list of nonnegative integer token IDs")
    return token_ids


def render_prompt(request: DecisionRequest) -> str:
    """Render the exact model prompt without requiring a tokenizer."""

    symbol_to_option_id = tuple(
        zip(_SYMBOLS, (option.id for option in request.options), strict=False)
    )
    payload = {
        "context": request.context,
        "question": request.question,
        "options": [
            {
                "symbol": symbol,
                "label": option.label,
                "description": option.description,
            }
            for (symbol, _), option in zip(symbol_to_option_id, request.options, strict=True)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + _ANSWER_SUFFIX


def compile_request(
    request: DecisionRequest,
    tokenizer: Tokenizer,
    *,
    max_tokens: int = 2048,
) -> CompiledRequest:
    """Render a request and verify every answer symbol is one boundary-safe token."""

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")

    symbol_to_option_id = tuple(
        zip(_SYMBOLS, (option.id for option in request.options), strict=False)
    )
    prompt = render_prompt(request)
    input_ids = _encode(tokenizer, prompt)
    if len(input_ids) > max_tokens:
        raise ValueError(f"rendered prompt is {len(input_ids)} tokens; limit is {max_tokens}")

    candidate_token_ids: list[int] = []
    for symbol, _ in symbol_to_option_id:
        combined_ids = _encode(tokenizer, prompt + symbol)
        if combined_ids[:-1] != input_ids or len(combined_ids) != len(input_ids) + 1:
            raise ValueError(f"candidate symbol {symbol!r} retokenizes the prompt boundary")
        candidate_token_ids.append(combined_ids[-1])

    if len(set(candidate_token_ids)) != len(candidate_token_ids):
        raise ValueError("candidate symbols must have distinct token IDs")

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return CompiledRequest(
        prompt=prompt,
        input_ids=tuple(input_ids),
        candidate_token_ids=tuple(candidate_token_ids),
        symbol_to_option_id=symbol_to_option_id,
        request_hash=request.request_hash,
        schema_hash=request.schema_hash,
        prompt_hash=prompt_hash,
    )
