"""Local-data and receipt-output binding for fresh evaluation analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import cast

STATES = ("base", "adapter")
DATASET_COUNTS = {
    "hans-eval-v1": 300,
    "winogrande-dev-v1": 200,
    "arc-challenge-dev-v1": 200,
}
RECEIPT_FIELDS = frozenset(
    "schema_version experiment_id run_id status phase payload result raw_worker_result failure "
    "execution modal".split()
)
_STATES = STATES
_DATASET_COUNTS = DATASET_COUNTS


def winner_maps(
    payload: Mapping[str, object],
    result: Mapping[str, object],
    records: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, dict[int, str]]]:
    """Join validated model rows to the exact two-order semantic request panel."""

    presentations = payload.get("presentations")
    compiled_rows = payload.get("compiled_requests")
    evidence = result.get("evidence")
    outputs = evidence.get("outputs") if isinstance(evidence, Mapping) else None
    if (
        not isinstance(presentations, list)
        or len(presentations) != 2 * len(records)
        or not isinstance(compiled_rows, list)
        or len(compiled_rows) != len(presentations)
        or not isinstance(outputs, Mapping)
        or set(outputs) != set(_STATES)
    ):
        raise ValueError("receipt output panel is incomplete")

    expected: dict[str, Mapping[str, object]] = {}
    record_orders: dict[str, dict[int, list[str]]] = {record_id: {} for record_id in records}
    request_hashes: dict[str, str] = {}
    for row in presentations:
        if not isinstance(row, Mapping):
            raise ValueError("presentation identity is malformed")
        presentation_id, record_id = row.get("presentation_id"), row.get("record_id")
        dataset_id, group_id = row.get("dataset_id"), row.get("source_group_id")
        order, order_ids, request_hash = (
            row.get("order_index"),
            row.get("order_ids"),
            row.get("request_hash"),
        )
        if (
            not isinstance(presentation_id, str)
            or not isinstance(record_id, str)
            or record_id not in records
            or presentation_id in expected
            or not isinstance(dataset_id, str)
            or not isinstance(group_id, str)
            or type(order) is not int
            or order not in (0, 1)
            or not isinstance(order_ids, list)
            or any(not isinstance(option_id, str) for option_id in order_ids)
            or not isinstance(request_hash, str)
        ):
            raise ValueError("presentation identity is malformed or duplicated")
        record = records[record_id]
        option_ids = record.get("option_ids")
        if (
            dataset_id != record.get("dataset_id")
            or group_id != record.get("source_group_id")
            or request_hash != record.get("request_hash")
            or not isinstance(option_ids, list)
            or order_ids != (option_ids if order == 0 else option_ids[1:] + option_ids[:1])
            or presentation_id != f"fresh-eval-v1:{record_id}:order-{order}"
            or order in record_orders[record_id]
        ):
            raise ValueError("presentation differs from the local semantic request")
        record_orders[record_id][order] = order_ids
        prior_hash = request_hashes.setdefault(record_id, request_hash)
        if prior_hash != request_hash:
            raise ValueError("presentation request hash changes across option orders")
        expected[presentation_id] = row
    if any(set(orders) != {0, 1} for orders in record_orders.values()):
        raise ValueError("presentation panel has a missing option order")

    compiled: dict[str, Mapping[str, object]] = {}
    for row in compiled_rows:
        if not isinstance(row, Mapping):
            raise ValueError("compiled request identity is malformed")
        presentation_id = row.get("presentation_id")
        if (
            not isinstance(presentation_id, str)
            or presentation_id not in expected
            or presentation_id in compiled
            or any(
                row.get(key) != expected[presentation_id].get(key)
                for key in ("request_hash", "order_ids")
            )
            or type(row.get("input_tokens")) is not int
            or not isinstance(row.get("prompt_sha256"), str)
        ):
            raise ValueError("compiled request differs from the fixed panel")
        compiled[presentation_id] = row
    if set(compiled) != set(expected):
        raise ValueError("compiled request panel is incomplete")

    winner_maps: dict[str, dict[str, dict[int, str]]] = {}
    identity_fields = (
        "presentation_id",
        "record_id",
        "dataset_id",
        "source_group_id",
        "request_hash",
        "order_index",
        "order_ids",
    )
    for state in _STATES:
        state_rows = outputs[state]
        if not isinstance(state_rows, list) or len(state_rows) != len(expected):
            raise ValueError(f"{state} output panel is incomplete")
        seen: set[str] = set()
        state_winners = {record_id: {} for record_id in records}
        for row in state_rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{state} output identity is malformed")
            presentation_id = row.get("presentation_id")
            if (
                not isinstance(presentation_id, str)
                or presentation_id not in expected
                or presentation_id in seen
            ):
                raise ValueError(f"{state} output has a duplicate or unmatched identity")
            source = expected[presentation_id]
            compiled_row = compiled[presentation_id]
            if any(row.get(key) != source.get(key) for key in identity_fields):
                raise ValueError(f"{state} output identity differs from the fixed panel")
            if row.get("input_tokens") != compiled_row.get("input_tokens") or row.get(
                "prompt_sha256"
            ) != compiled_row.get("prompt_sha256"):
                raise ValueError(f"{state} output compilation differs from the pinned prompt")
            logits, winner = row.get("candidate_logits"), row.get("winner_option_id")
            if (
                not isinstance(logits, list)
                or len(logits) != len(source["order_ids"])
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in logits
                )
                or not isinstance(winner, str)
                or winner not in source["order_ids"]
            ):
                raise ValueError(f"{state} output winner or candidate scores are malformed")
            record_id, order = source["record_id"], source["order_index"]
            state_winners[record_id][order] = winner
            seen.add(presentation_id)
        if seen != set(expected) or any(set(orders) != {0, 1} for orders in state_winners.values()):
            raise ValueError(f"{state} output has a missing option order")
        winner_maps[state] = state_winners
    return winner_maps


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _record_sha256(record: object) -> str:
    dump = getattr(record, "model_dump", None)
    if callable(dump):
        try:
            value = dump(mode="json")
        except TypeError:
            value = dump()
    elif isinstance(record, Mapping):
        request = record.get("request")
        options = _field(request, "options")
        if not isinstance(options, (list, tuple)):
            raise ValueError("local record cannot be canonically hashed")
        value = {
            "record_id": record.get("record_id"),
            "dataset_id": record.get("dataset_id"),
            "source_group_id": record.get("source_group_id"),
            "request": {
                "context": _field(request, "context"),
                "question": _field(request, "question"),
                "options": [
                    {
                        "id": _field(option, "id"),
                        "label": _field(option, "label"),
                        "description": _field(option, "description"),
                    }
                    for option in options
                ],
            },
            "answer_id": record.get("answer_id"),
        }
    else:
        raise ValueError("local record cannot be canonically hashed")
    try:
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("local record cannot be canonically hashed") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _local_records(
    records: object,
) -> tuple[list[dict[str, object]], dict[str, Mapping[str, object]]]:
    """Normalize the private local panel without serializing request text."""

    if not isinstance(records, list) or len(records) != sum(_DATASET_COUNTS.values()):
        raise ValueError("local fresh-evaluation panel must contain exactly 700 records")
    normalized: list[dict[str, object]] = []
    by_id: dict[str, Mapping[str, object]] = {}
    for record in records:
        record_id = _field(record, "record_id")
        dataset_id = _field(record, "dataset_id")
        group_id = _field(record, "source_group_id")
        answer_id = _field(record, "answer_id")
        request = _field(record, "request")
        options = _field(request, "options")
        request_hash = _field(request, "request_hash")
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in by_id
            or dataset_id not in _DATASET_COUNTS
            or not isinstance(group_id, str)
            or not group_id
            or not isinstance(answer_id, str)
            or not isinstance(options, (list, tuple))
        ):
            raise ValueError("local fresh-evaluation record identity is malformed")
        option_ids = [_field(option, "id") for option in options]
        if (
            len(option_ids) < 2
            or any(not isinstance(option_id, str) or not option_id for option_id in option_ids)
            or len(set(option_ids)) != len(option_ids)
            or answer_id not in option_ids
            or not isinstance(request_hash, str)
            or len(request_hash) != 64
            or any(character not in "0123456789abcdef" for character in request_hash)
        ):
            raise ValueError("local fresh-evaluation request or gold identity is malformed")
        item = {
            "record_id": record_id,
            "dataset_id": dataset_id,
            "source_group_id": group_id,
            "answer_id": answer_id,
            "option_ids": option_ids,
            "request_hash": request_hash,
            "record_sha256": _record_sha256(record),
        }
        by_id[record_id] = item
        normalized.append(item)
    expected_order = sorted(
        normalized, key=lambda row: (cast(str, row["dataset_id"]), cast(str, row["record_id"]))
    )
    if normalized != expected_order:
        raise ValueError("local fresh-evaluation records are not sorted by task and record")
    counts: dict[str, int] = defaultdict(int)
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in normalized:
        dataset_id = cast(str, row["dataset_id"])
        group_id = cast(str, row["source_group_id"])
        counts[dataset_id] += 1
        groups[dataset_id][group_id] += 1
    if dict(counts) != _DATASET_COUNTS:
        raise ValueError("local fresh-evaluation dataset counts differ from the frozen panel")
    hans_group_sizes = list(groups["hans-eval-v1"].values())
    if len(hans_group_sizes) != 30 or any(size != 10 for size in hans_group_sizes):
        raise ValueError("HANS must contain 30 source subcases with 10 questions each")
    for dataset_id in ("winogrande-dev-v1", "arc-challenge-dev-v1"):
        task_groups = groups[dataset_id]
        if len(task_groups) != _DATASET_COUNTS[dataset_id] or any(
            size != 1 for size in task_groups.values()
        ):
            raise ValueError(f"{dataset_id} must use each question as its source group")
    return normalized, by_id


def _validate_host_metadata(
    metadata: object,
    records_by_id: Mapping[str, Mapping[str, object]],
    gpu_pins: object,
    *,
    core: object,
) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        raise ValueError("fresh-evaluation data loader metadata is malformed")
    required_hashes = (
        "manifest_sha256",
        "recipe_sha256",
        "records_sha256",
        "record_metadata_sha256",
    )
    for field in required_hashes:
        digest = metadata.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"fresh-evaluation metadata {field} is malformed")
    normalized_gpu_pins = core.validate_gpu_pins(gpu_pins)
    source_hashes = metadata.get("source_file_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or dict(source_hashes) != normalized_gpu_pins["data_file_sha256"]
    ):
        raise ValueError("local dataset file pins differ from the frozen GPU data pins")
    if (
        metadata["manifest_sha256"] != normalized_gpu_pins["source_manifest_sha256"]
        or metadata["records_sha256"] != normalized_gpu_pins["panel_sha256"]
    ):
        raise ValueError("local manifest or record pins differ from the frozen GPU data pins")
    expected_counts = {
        "total": 700,
        "by_dataset": dict(_DATASET_COUNTS),
        "presentations": 1400,
        "parity_presentations": 24,
    }
    if metadata.get("counts") != expected_counts:
        raise ValueError("fresh-evaluation host counts differ from the frozen panel")
    source_hash_copy: dict[str, str] = {}
    for path, digest in source_hashes.items():
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("local dataset source hash map is malformed")
        source_hash_copy[path] = digest
    metadata_rows = metadata.get("record_metadata_by_id")
    if not isinstance(metadata_rows, Mapping) or set(metadata_rows) != set(records_by_id):
        raise ValueError("host-only record metadata differs from the local panel")
    hans_metadata: dict[str, Mapping[str, object]] = {}
    for record_id, record in records_by_id.items():
        source = metadata_rows[record_id]
        if (
            not isinstance(source, Mapping)
            or source.get("dataset_id") != record["dataset_id"]
            or source.get("source_group_id") != record["source_group_id"]
            or source.get("semantic_request_sha256") != record["request_hash"]
            or source.get("record_sha256") != record["record_sha256"]
        ):
            raise ValueError(
                "host-only gold, request, or group metadata differs from local records"
            )
        if record["dataset_id"] == "hans-eval-v1":
            heuristic, subcase = source.get("heuristic"), source.get("subcase")
            if (
                not isinstance(heuristic, str)
                or not heuristic
                or not isinstance(subcase, str)
                or not subcase
            ):
                raise ValueError("HANS heuristic and subcase diagnostics are incomplete")
            hans_metadata[record_id] = source
    if len(hans_metadata) != DATASET_COUNTS["hans-eval-v1"]:
        raise ValueError("HANS diagnostic metadata does not cover all 300 records")
    hans_heuristics = {row["heuristic"] for row in hans_metadata.values()}
    hans_subcases = {row["subcase"] for row in hans_metadata.values()}
    subcase_groups: dict[str, set[str]] = defaultdict(set)
    group_subcases: dict[str, set[str]] = defaultdict(set)
    subcase_heuristics: dict[str, set[str]] = defaultdict(set)
    for row in hans_metadata.values():
        subcase_groups[cast(str, row["subcase"])].add(cast(str, row["source_group_id"]))
        group_subcases[cast(str, row["source_group_id"])].add(cast(str, row["subcase"]))
        subcase_heuristics[cast(str, row["subcase"])].add(cast(str, row["heuristic"]))
    if (
        len(hans_heuristics) != 3
        or len(hans_subcases) != 30
        or any(len(group_ids) != 1 for group_ids in subcase_groups.values())
        or any(len(subcases) != 1 for subcases in group_subcases.values())
        or any(len(heuristics) != 1 for heuristics in subcase_heuristics.values())
    ):
        raise ValueError("HANS diagnostics require exactly three heuristics and 30 subcases")
    selection = metadata.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("fresh-evaluation selection metadata is malformed")
    selection_digest = selection.get("selection_digest_sha256")
    if (
        not isinstance(selection_digest, str)
        or len(selection_digest) != 64
        or any(character not in "0123456789abcdef" for character in selection_digest)
    ):
        raise ValueError("fresh-evaluation selection digest is malformed")
    return {
        "dataset_source_file_sha256": source_hash_copy,
        "manifest_sha256": metadata["manifest_sha256"],
        "recipe_sha256": metadata["recipe_sha256"],
        "records_sha256": metadata["records_sha256"],
        "record_metadata_sha256": metadata["record_metadata_sha256"],
        "selection": dict(selection),
        "hans_record_metadata": hans_metadata,
    }
