"""CPU verification of the frozen evaluation presentation panel."""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict
from collections.abc import Sequence

from experiments import mixture_training_data as mixture_data
from experiments import real_pilot_core
from experiments.mixture_training_contracts import (
    json_sha256,
    normalize_json_object,
    validate_sha256,
)
from reflex_decisions import snli_diagnostic
from reflex_decisions.schema import DecisionRequest

_PRESENTATION_FIELDS = {
    "presentation_id",
    "record_id",
    "dataset_id",
    "source_group_id",
    "request_hash",
    "order_index",
    "order_ids",
    "request",
}
_REAL_EVALUATION_COUNTS = dict(mixture_data._REAL_EVALUATION_PRESENTATION_COUNTS)
_EVALUATION_DATASET_COUNTS = {
    **_REAL_EVALUATION_COUNTS,
    snli_diagnostic.DATASET_ID: 1152,
    **mixture_data._SYNTHETIC_PRESENTATION_COUNTS,
}
_REAL_EVALUATION_RECORD_COUNTS = {
    dataset_id: count
    for dataset_id, count in real_pilot_core.DATASET_RECORD_COUNTS.items()
    if dataset_id not in real_pilot_core.TRAIN_DATASET_IDS
}
_SYNTHETIC_EVALUATION_RECORD_COUNTS = {
    dataset_id: count
    for dataset_id, count in mixture_data.SYNTHETIC_DATASET_COUNTS.items()
    if dataset_id in mixture_data.SYNTHETIC_EVALUATION_DATASET_IDS
}
_EVALUATION_RECORD_COUNTS = {
    **_REAL_EVALUATION_RECORD_COUNTS,
    snli_diagnostic.DATASET_ID: 192,
    **_SYNTHETIC_EVALUATION_RECORD_COUNTS,
}
_REAL_DATASET_IDS = set(real_pilot_core.DATASET_RECORD_COUNTS)
_SYNTHETIC_DATASET_IDS = set(mixture_data.SYNTHETIC_DATASET_COUNTS)
_REAL_NAMESPACE = "reflex-real-pilot-presentation-v1"


def _expected_orders(dataset_id: str, order_ids: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    base = tuple(order_ids)
    if dataset_id.startswith("dbpedia14-"):
        if len(base) != 14:
            raise ValueError("DBpedia14 evaluation requests must contain exactly 14 options")
        if dataset_id.endswith("-calibration"):
            return (base,)
        reverse = tuple(reversed(base))
        return tuple(
            dict.fromkeys(
                (
                    *(base[index:] + base[:index] for index in range(len(base))),
                    *(reverse[index:] + reverse[:index] for index in range(len(reverse))),
                )
            )
        )
    if dataset_id.startswith("sms-"):
        if len(base) != 2:
            raise ValueError("SMS evaluation requests must contain exactly two options")
        return (
            (base,) if dataset_id.endswith("-calibration") else tuple(itertools.permutations(base))
        )
    if dataset_id == "snli-pilot-v1-development":
        if len(base) != 3:
            raise ValueError("pilot SNLI evaluation requests must contain exactly three options")
        return (base,)
    if dataset_id == snli_diagnostic.DATASET_ID:
        if len(base) != 3:
            raise ValueError("balanced SNLI requests must contain exactly three options")
        return tuple(itertools.permutations(base))
    if dataset_id in _SYNTHETIC_DATASET_IDS:
        if dataset_id.startswith("synthetic-atomic-fact-inference-") and len(base) != 3:
            raise ValueError("synthetic fact inference requests must contain exactly three options")
        if dataset_id.startswith("synthetic-numeric-selection-") and len(base) not in (
            2,
            4,
            8,
            16,
        ):
            raise ValueError("synthetic numeric requests must contain 2, 4, 8, or 16 options")
        if dataset_id.endswith("-calibration"):
            return (base,)
        if len(base) in (2, 3):
            return tuple(itertools.permutations(base))
        if len(base) in (4, 8, 16):
            return tuple(base[index:] + base[:index] for index in range(len(base)))
    raise ValueError("evaluation dataset or option count is outside the frozen allowlist")


def _presentation_id(
    dataset_id: str, record_id: str, request_hash: str, order_ids: Sequence[str]
) -> str:
    if dataset_id in _REAL_DATASET_IDS:
        content = "\0".join((_REAL_NAMESPACE, record_id, *order_ids)).encode("utf-8")
        return hashlib.sha256(content).hexdigest()
    namespace = (
        "reflex-mixture-balanced-snli-v1"
        if dataset_id == snli_diagnostic.DATASET_ID
        else "reflex-mixture-synthetic-v1"
    )
    return json_sha256(
        {
            "namespace": namespace,
            "record_id": record_id,
            "request_hash": request_hash,
            "order_ids": list(order_ids),
        }
    )


def _validate_evaluation_rows(value: object, phase: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("evaluation_presentations must be a JSON array")
    expected_count = (
        mixture_data.EXPECTED_SYNTHETIC_PRESENTATIONS
        if phase == "initialize"
        else mixture_data.EXPECTED_EVALUATION_PRESENTATIONS
    )
    if len(value) != expected_count:
        raise ValueError("evaluation panel has the wrong frozen presentation count")

    rows: list[dict[str, object]] = []
    by_record: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_presentations: set[str] = set()
    dataset_counts: Counter[str] = Counter()
    transitions: set[str] = set()
    last_record: str | None = None
    group_dataset: dict[str, str] = {}
    for item in value:
        row = normalize_json_object(item, "evaluation presentation")
        if set(row) != _PRESENTATION_FIELDS:
            raise ValueError("evaluation presentation has an unexpected or label-bearing field")
        for name in (
            "presentation_id",
            "record_id",
            "dataset_id",
            "source_group_id",
            "request_hash",
        ):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError(f"evaluation {name} must be a nonempty string")
        if type(row["order_index"]) is not int or row["order_index"] < 0:
            raise ValueError("evaluation order_index must be a nonnegative integer")
        validate_sha256(row["request_hash"], "evaluation request_hash")
        order_ids = row["order_ids"]
        if (
            not isinstance(order_ids, list)
            or len(order_ids) < 2
            or len(order_ids) > 16
            or any(not isinstance(option_id, str) or not option_id for option_id in order_ids)
            or len(order_ids) != len(set(order_ids))
        ):
            raise ValueError("evaluation option IDs are malformed")
        request = DecisionRequest.model_validate(row["request"])
        if request.model_dump(mode="json") != row["request"]:
            raise ValueError("evaluation request is not canonical JSON")
        if list(option.id for option in request.options) != order_ids:
            raise ValueError("evaluation request option order differs from order_ids")
        if request.request_hash != row["request_hash"]:
            raise ValueError("evaluation request hash does not match its semantic request")
        expected_id = _presentation_id(
            row["dataset_id"], row["record_id"], row["request_hash"], order_ids
        )
        if row["presentation_id"] != expected_id:
            raise ValueError("evaluation presentation ID does not match its semantic row")
        if row["presentation_id"] in seen_presentations:
            raise ValueError("evaluation presentation IDs must be unique")
        seen_presentations.add(row["presentation_id"])
        record_id = row["record_id"]
        if record_id != last_record:
            if record_id in transitions:
                raise ValueError("evaluation records must be contiguous and ordered")
            transitions.add(record_id)
            last_record = record_id
        by_record[record_id].append(row)
        dataset_counts[row["dataset_id"]] += 1
        prior_dataset = group_dataset.setdefault(row["source_group_id"], row["dataset_id"])
        if prior_dataset != row["dataset_id"]:
            raise ValueError("evaluation source group appears in multiple datasets")
        rows.append(row)

    expected_counts = (
        dict(mixture_data._SYNTHETIC_PRESENTATION_COUNTS)
        if phase == "initialize"
        else _EVALUATION_DATASET_COUNTS
    )
    if dataset_counts != Counter(expected_counts):
        raise ValueError("evaluation dataset presentation counts differ from the frozen panel")
    expected_record_counts = (
        _SYNTHETIC_EVALUATION_RECORD_COUNTS if phase == "initialize" else _EVALUATION_RECORD_COUNTS
    )
    actual_record_counts = Counter(
        record_rows[0]["dataset_id"] for record_rows in by_record.values()
    )
    if actual_record_counts != Counter(expected_record_counts):
        raise ValueError("evaluation dataset record counts differ from the frozen panel")
    request_owners: dict[str, str] = {}
    balanced_records_by_group: dict[str, set[str]] = defaultdict(set)
    for record_rows in by_record.values():
        first = record_rows[0]
        if any(row["dataset_id"] != first["dataset_id"] for row in record_rows):
            raise ValueError("evaluation record changes dataset across its order panel")
        if first["dataset_id"] == snli_diagnostic.DATASET_ID:
            balanced_records_by_group[str(first["source_group_id"])].add(str(first["record_id"]))
        prior_record = request_owners.setdefault(first["request_hash"], first["record_id"])
        if prior_record != first["record_id"]:
            raise ValueError("evaluation semantic requests must be unique across records")
        base_request = DecisionRequest.model_validate(first["request"])
        base_options = {
            option.id: option.model_dump(mode="json") for option in base_request.options
        }
        expected = _expected_orders(first["dataset_id"], first["order_ids"])
        if len(record_rows) != len(expected):
            raise ValueError("evaluation record has an incomplete permutation panel")
        if any(
            row["dataset_id"] != first["dataset_id"]
            or row["source_group_id"] != first["source_group_id"]
            or row["request_hash"] != first["request_hash"]
            for row in record_rows
        ):
            raise ValueError("evaluation record identity changes across option orders")
        for order_index, row in enumerate(record_rows):
            request = DecisionRequest.model_validate(row["request"])
            current_options = {
                option.id: option.model_dump(mode="json") for option in request.options
            }
            if (
                row["order_index"] != order_index
                or tuple(row["order_ids"]) != expected[order_index]
                or current_options != base_options
                or request.context != base_request.context
                or request.question != base_request.question
            ):
                raise ValueError("evaluation order is not the exact expected permutation/rotation")

    if phase == "train" and (
        len(balanced_records_by_group) != snli_diagnostic.SELECTED_GROUP_COUNT
        or any(len(record_ids) != 3 for record_ids in balanced_records_by_group.values())
    ):
        raise ValueError("balanced SNLI panel must contain exactly 64 groups of three records")

    return rows


__all__ = ["_validate_evaluation_rows"]
