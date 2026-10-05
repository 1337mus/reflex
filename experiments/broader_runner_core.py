"""CPU-only contracts shared by the bounded broader-baseline runner."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping
from pathlib import Path

from experiments import baseline_runner_core
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest

MODELS = baseline_runner_core.MODELS
DATASET_IDS = ("boolq-dev-pilot-v1", "snli-dev-pilot-v1")
RECORDS_PER_DATASET = 32
RECORD_COUNT = len(DATASET_IDS) * RECORDS_PER_DATASET
PRESENTATIONS_PER_MODEL = 256
PRESENTATIONS_TOTAL = 768
AUXILIARY_FORWARD_COUNTS = baseline_runner_core.EXPECTED_AUXILIARY_FORWARD_COUNTS
AUXILIARY_FORWARD_TOTAL = sum(AUXILIARY_FORWARD_COUNTS.values())
_OPTION_IDS = {
    DATASET_IDS[0]: ("yes", "no"),
    DATASET_IDS[1]: ("entailment", "neutral", "contradiction"),
}
SOURCE_EXPERIMENT_FILES = (
    "experiments/analyze_broader.py",
    "experiments/baseline_intern.py",
    "experiments/baseline_kev.py",
    "experiments/baseline_qwen.py",
    "experiments/baseline_runner_core.py",
    "experiments/modal_baseline.py",
    "experiments/modal_broader.py",
    "experiments/modal_smoke.py",
    "experiments/broader_runner_core.py",
)


def source_fingerprints() -> dict[str, str]:
    """Fingerprint the runner and inference dependencies, excluding unrelated experiments."""

    root = Path(__file__).resolve().parents[1]
    paths = [root / relative for relative in SOURCE_EXPERIMENT_FILES]
    paths.extend(sorted((root / "src" / "reflex_decisions").rglob("*.py")))
    if any(not path.is_file() for path in paths):
        raise ValueError("a required broader runner source file is missing")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def dataset_for_record_id(record_id: str) -> str:
    for dataset_id in DATASET_IDS:
        if record_id.startswith(f"{dataset_id}-"):
            return dataset_id
    raise ValueError("record ID does not identify an approved broader dataset")


def serialize_remote_records(records: tuple[DecisionRecord, ...]) -> list[dict[str, object]]:
    """Send only record IDs and unlabeled requests to remote model workers."""

    return [
        {"record_id": record.record_id, "request": record.request.model_dump(mode="json")}
        for record in records
    ]


def presentations(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create each option permutation in deterministic itertools order."""

    output: list[dict[str, object]] = []
    for row in records:
        if not isinstance(row, dict) or set(row) != {"record_id", "request"}:
            raise ValueError("remote records may contain only record_id and request")
        record_id = row["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("remote record ID must be nonblank")
        request = DecisionRequest.model_validate(row["request"])
        dataset_id = dataset_for_record_id(record_id)
        if tuple(option.id for option in request.options) != _OPTION_IDS[dataset_id]:
            raise ValueError("remote request does not preserve its approved semantic option order")
        for index, options in enumerate(itertools.permutations(request.options)):
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
    """Validate finite logits and bounded metadata before receipt serialization."""

    return baseline_runner_core.validate_presentation_result(presentation, scored)


def comparison_passed(
    receipts: Mapping[str, object],
    expected_records: list[dict[str, object]],
    *,
    lifecycle_ok: bool = True,
) -> bool:
    """Accept only three complete, exact-order receipts for the frozen 64-record pilot."""

    if not lifecycle_ok or set(receipts) != set(MODELS) or len(expected_records) != RECORD_COUNT:
        return False
    try:
        record_ids = [row["record_id"] for row in expected_records]
        if len(set(record_ids)) != RECORD_COUNT:
            return False
        dataset_counts = {dataset_id: 0 for dataset_id in DATASET_IDS}
        for row in expected_records:
            if not isinstance(row, dict) or set(row) != {"record_id", "request"}:
                return False
            dataset_counts[dataset_for_record_id(row["record_id"])] += 1
        if any(count != RECORDS_PER_DATASET for count in dataset_counts.values()):
            return False
        expected_rows = presentations(expected_records)
        if len(expected_rows) != PRESENTATIONS_PER_MODEL:
            return False
        expected_by_key = {
            (row["record_id"], row["permutation_index"]): row for row in expected_rows
        }
        expected_auxiliary = baseline_runner_core.EXPECTED_AUXILIARY_FORWARD_COUNTS
        for name in MODELS:
            receipt = receipts[name]
            if (
                not isinstance(receipt, dict)
                or receipt.get("model_name") != name
                or receipt.get("status") != "passed"
            ):
                return False
            rows = receipt.get("presentations")
            if not isinstance(rows, list) or len(rows) != PRESENTATIONS_PER_MODEL:
                return False
            auxiliary_count = expected_auxiliary[name]
            if (
                type(receipt.get("scored_presentation_count")) is not int
                or receipt["scored_presentation_count"] != PRESENTATIONS_PER_MODEL
                or type(receipt.get("auxiliary_forward_count")) is not int
                or receipt["auxiliary_forward_count"] != auxiliary_count
                or type(receipt.get("total_forward_count")) is not int
                or receipt["total_forward_count"] != PRESENTATIONS_PER_MODEL + auxiliary_count
            ):
                return False
            actual: set[tuple[str, int]] = set()
            for row in rows:
                if not isinstance(row, dict):
                    return False
                record_id, permutation_index = row.get("record_id"), row.get("permutation_index")
                if not isinstance(record_id, str) or type(permutation_index) is not int:
                    return False
                key = (record_id, permutation_index)
                if key not in expected_by_key or key in actual:
                    return False
                expected = expected_by_key[key]
                if (
                    row.get("request_hash") != expected["request_hash"]
                    or row.get("option_ids") != expected["option_ids"]
                ):
                    return False
                validate_presentation_result(
                    expected,
                    {key: row.get(key) for key in ("raw_logits", "input_tokens", "prompt_sha256")},
                )
                actual.add(key)
            if actual != set(expected_by_key):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True
