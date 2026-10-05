"""CPU-only contracts shared by the COPA baseline launcher and its tests."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest

MODELS = ("qwen", "intern", "kev")
EXPECTED_AUXILIARY_FORWARD_COUNTS = {"qwen": 0, "intern": 1, "kev": 2}
RECORD_COUNT = 32
PRESENTATIONS_PER_RECORD = 2
PRESENTATIONS_PER_MODEL = RECORD_COUNT * PRESENTATIONS_PER_RECORD


def plan() -> dict[str, object]:
    return {
        "mode": "plan-only",
        "dataset": "copa-dev-pilot-v1",
        "records": RECORD_COUNT,
        "presentations_per_record": PRESENTATIONS_PER_RECORD,
        "presentations_per_model": PRESENTATIONS_PER_MODEL,
        "models": list(MODELS),
        "runtime": {
            "python": "3.12",
            "modal": "1.6.1",
            "torch": "2.14.1+cu130",
            "torchvision": "0.29.1+cu130",
            "transformers": "5.18.0",
            "peft": "0.21.0",
            "pydantic": "2.13.5",
            "pillow": "12.0.0",
            "use_hub_kernels": "NO",
        },
        "modal": {
            "app": "reflex-copa-development-baselines",
            "gpu": "A10",
            "cpu_physical_cores": [2.0, 2.0],
            "memory_mib": [16384, 16384],
            "max_containers": 3,
            "min_containers": 0,
            "buffer_containers": 0,
            "scaledown_window_seconds": 2,
            "retries": 0,
            "single_use_containers": True,
            "startup_timeout_seconds": 300,
            "timeout_seconds": 900,
            "serialized": True,
            "include_source": False,
        },
        "inputs": {
            "records": "data/processed/copa-dev-pilot-v1.jsonl",
            "manifest": "data/baselines/copa-dev-manifest.json",
        },
    }


def serialize_remote_records(
    records: tuple[DecisionRecord, ...],
) -> list[dict[str, object]]:
    """Send only record IDs and unlabeled requests to the model workers."""

    return [
        {"record_id": record.record_id, "request": record.request.model_dump(mode="json")}
        for record in records
    ]


def presentations(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create original and reversed requests while keeping semantic option IDs."""

    output: list[dict[str, object]] = []
    for row in records:
        if set(row) != {"record_id", "request"}:
            raise ValueError("remote records may contain only record_id and request")
        record_id = row["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("remote record ID must be nonblank")
        request = DecisionRequest.model_validate(row["request"])
        for index, options in enumerate((request.options, tuple(reversed(request.options)))):
            presented = request.model_copy(update={"options": options})
            output.append(
                {
                    "record_id": record_id,
                    "request_hash": presented.request_hash,
                    "permutation_index": index,
                    "option_ids": [option.id for option in presented.options],
                    "request": presented.model_dump(mode="json"),
                }
            )
    return output


def validate_presentation_result(
    presentation: dict[str, object], scored: object
) -> dict[str, object]:
    """Validate an adapter result before adding it to a run receipt."""

    if not isinstance(scored, dict) or set(scored) != {
        "raw_logits",
        "input_tokens",
        "prompt_sha256",
    }:
        raise ValueError("scorer output does not match the raw-logit contract")
    logits = scored["raw_logits"]
    option_ids = presentation["option_ids"]
    if not isinstance(option_ids, list) or any(not isinstance(value, str) for value in option_ids):
        raise ValueError("presentation option IDs must be strings")
    if len(option_ids) != len(set(option_ids)):
        raise ValueError("presentation option IDs must be unique")
    if not isinstance(logits, list) or len(logits) != len(option_ids):
        raise ValueError("scorer returned a different number of logits than options")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in logits
    ):
        raise ValueError("scorer returned non-finite candidate logits")
    input_tokens = scored["input_tokens"]
    if type(input_tokens) is not int or not 1 <= input_tokens <= 2048:
        raise ValueError("scorer input token count is outside the 1–2048 limit")
    prompt_sha256 = scored["prompt_sha256"]
    if not isinstance(prompt_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", prompt_sha256) is None:
        raise ValueError("scorer prompt digest must be a lowercase SHA-256")
    return {
        "record_id": presentation["record_id"],
        "request_hash": presentation["request_hash"],
        "permutation_index": presentation["permutation_index"],
        "option_ids": list(option_ids),
        "raw_logits": [float(value) for value in logits],
        "input_tokens": input_tokens,
        "prompt_sha256": prompt_sha256,
    }


def validate_adapter_provenance(
    model_name: str, adapter: Any, provenance: object
) -> dict[str, object]:
    if model_name not in MODELS or not isinstance(provenance, dict):
        raise ValueError("adapter returned invalid model provenance")
    revision = getattr(adapter, "MODEL_REVISION", getattr(adapter, "REVISION", None))
    if (
        provenance.get("model_id") != adapter.MODEL_ID
        or provenance.get("model_revision") != revision
    ):
        raise ValueError("adapter provenance does not match its pinned model revision")
    temperature = provenance.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("adapter temperature must be a finite positive number")
    if model_name in {"qwen", "intern"} and temperature != adapter.TEMPERATURE:
        raise ValueError("adapter temperature does not match its published value")
    return provenance


def comparison_passed(
    receipts: Mapping[str, object], expected_record_ids: list[str], *, lifecycle_ok: bool = True
) -> bool:
    if not lifecycle_ok:
        return False
    if (
        set(receipts) != set(MODELS)
        or len(expected_record_ids) != RECORD_COUNT
        or len(set(expected_record_ids)) != RECORD_COUNT
    ):
        return False
    expected = {
        (record_id, permutation_index)
        for record_id in expected_record_ids
        for permutation_index in (0, 1)
    }
    for name in MODELS:
        receipt = receipts[name]
        if not isinstance(receipt, dict) or receipt.get("status") != "passed":
            return False
        rows = receipt.get("presentations")
        if not isinstance(rows, list) or len(rows) != PRESENTATIONS_PER_MODEL:
            return False
        auxiliary_count = receipt.get("auxiliary_forward_count")
        if (
            type(receipt.get("scored_presentation_count")) is not int
            or receipt["scored_presentation_count"] != PRESENTATIONS_PER_MODEL
            or type(auxiliary_count) is not int
            or auxiliary_count != EXPECTED_AUXILIARY_FORWARD_COUNTS[name]
            or type(receipt.get("total_forward_count")) is not int
            or receipt["total_forward_count"] != PRESENTATIONS_PER_MODEL + auxiliary_count
        ):
            return False
        actual: set[tuple[str, int]] = set()
        for row in rows:
            if not isinstance(row, dict):
                return False
            record_id = row.get("record_id")
            permutation_index = row.get("permutation_index")
            if (
                not isinstance(record_id, str)
                or type(permutation_index) is not int
                or permutation_index not in (0, 1)
            ):
                return False
            actual.add((record_id, permutation_index))
        if actual != expected:
            return False
    return True
