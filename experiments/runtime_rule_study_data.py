"""Deterministic CPU schedules and request-only panels for the runtime-rule study."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

from experiments import mixture_training_data, training_rehearsal_core
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import Option

TrainingExample = training_rehearsal_core.TrainingExample

CONTINUED_PRACTICE = "continued_practice"
RUNTIME_MIX = "runtime_mix"
ARM_NAMES = (CONTINUED_PRACTICE, RUNTIME_MIX)
REAL_SEED = 20261010
OLD_SYNTHETIC_SEED = 20262010
SNLI_SEED = 20263010
ROUTING_SEED = 20264010
TOOL_SEED = 20265010
SHARED_EXAMPLES = 336
NEW_RECORDS_PER_FAMILY = 56
TRAINING_UPDATES = 336
MICROBATCHES_PER_UPDATE = 4
TRAINING_PRESENTATIONS = TRAINING_UPDATES * MICROBATCHES_PER_UPDATE
NEW_PRESENTATIONS_PER_FAMILY = 82
NEW_PANEL_PRESENTATIONS = 2 * NEW_PRESENTATIONS_PER_FAMILY
RELOAD_PER_FAMILY = 16
PRESENTATION_FIELDS = frozenset(
    {
        "presentation_id",
        "record_id",
        "dataset_id",
        "source_group_id",
        "request_hash",
        "order_index",
        "order_ids",
        "request",
    }
)
NEW_TRAIN_DATASET_IDS = {
    "routing": "routing-data-v1-train",
    "tool": "tool-data-v1-train",
}
NEW_DEVELOPMENT_DATASET_IDS = {
    "routing": "routing-data-v1-development",
    "tool": "tool-data-v1-development",
}
_TRAIN_MENU_COUNTS = {4: 19, 6: 18, 8: 19}
_DEVELOPMENT_MENU_COUNTS = {4: 5, 6: 5, 8: 4}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_record_values(records: Sequence[DecisionRecord], label: str) -> None:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of DecisionRecord values")
    if any(not isinstance(record, DecisionRecord) for record in records):
        raise TypeError(f"{label} must contain DecisionRecord values")


def _validate_new_pool(
    records: Sequence[DecisionRecord], *, family: str, split: str, label: str
) -> None:
    _validate_record_values(records, label)
    dataset_id = (NEW_TRAIN_DATASET_IDS if split == "train" else NEW_DEVELOPMENT_DATASET_IDS)[
        family
    ]
    expected_count = NEW_RECORDS_PER_FAMILY if split == "train" else 14
    if Counter(record.dataset_id for record in records) != Counter({dataset_id: expected_count}):
        raise ValueError(f"{label} must contain exactly {expected_count} records from {dataset_id}")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError(f"{label} record IDs must be unique")
    if len({record.request.request_hash for record in records}) != len(records):
        raise ValueError(f"{label} semantic requests must be unique")
    if len({record.source_group_id for record in records}) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} distinct source groups")
    menu_counts = Counter(len(record.request.options) for record in records)
    expected = _TRAIN_MENU_COUNTS if split == "train" else _DEVELOPMENT_MENU_COUNTS
    if menu_counts != Counter(expected):
        raise ValueError(f"{label} menu-size counts do not match the exact 4/6/8-option mix")


def _validate_training_pools(
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
    routing_train: Sequence[DecisionRecord],
    tool_train: Sequence[DecisionRecord],
) -> None:
    mixture_training_data._validate_training_bundles(real_train, synthetic_train, snli_train)
    _validate_new_pool(
        routing_train, family="routing", split="train", label="routing training data"
    )
    _validate_new_pool(tool_train, family="tool", split="train", label="tool training data")
    mixture_training_data._validate_disjoint_pools(
        (real_train, synthetic_train, snli_train, routing_train, tool_train)
    )


def _epoch(
    records: Sequence[DecisionRecord], *, seed: int, epoch: int
) -> tuple[TrainingExample, ...]:
    return training_rehearsal_core.epoch_examples(
        tuple(sorted(records, key=lambda record: record.record_id)), seed=seed, epoch=epoch
    )


def _interleave(streams: Sequence[Sequence[TrainingExample]]) -> tuple[TrainingExample, ...]:
    return tuple(example for items in zip(*streams, strict=True) for example in items)


def _flatten_updates(
    real: Sequence[TrainingExample],
    synthetic: Sequence[TrainingExample],
    snli: Sequence[TrainingExample],
    extra: Sequence[TrainingExample],
) -> tuple[TrainingExample, ...]:
    return tuple(
        example for batch in zip(real, synthetic, snli, extra, strict=True) for example in batch
    )


def _build_schedules(
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
    routing_train: Sequence[DecisionRecord],
    tool_train: Sequence[DecisionRecord],
) -> dict[str, tuple[TrainingExample, ...]]:
    real = _epoch(real_train, seed=REAL_SEED, epoch=0)[:SHARED_EXAMPLES]
    synthetic = _epoch(synthetic_train, seed=OLD_SYNTHETIC_SEED, epoch=0)[:SHARED_EXAMPLES]
    snli = _epoch(snli_train, seed=SNLI_SEED, epoch=0)[:SHARED_EXAMPLES]
    control_extra = _interleave((real[:112], synthetic[:112], snli[:112]))

    treatment_extra: list[TrainingExample] = []
    for epoch in range(3):
        routing = _epoch(routing_train, seed=ROUTING_SEED, epoch=epoch)
        tool = _epoch(tool_train, seed=TOOL_SEED, epoch=epoch)
        treatment_extra.extend(_interleave((routing, tool)))

    return {
        CONTINUED_PRACTICE: _flatten_updates(real, synthetic, snli, control_extra),
        RUNTIME_MIX: _flatten_updates(real, synthetic, snli, treatment_extra),
    }


def build_training_schedules(
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
    routing_train: Sequence[DecisionRecord],
    tool_train: Sequence[DecisionRecord],
) -> dict[str, tuple[TrainingExample, ...]]:
    """Build the fixed matched 336-update control and runtime-mix schedules."""

    _validate_training_pools(real_train, synthetic_train, snli_train, routing_train, tool_train)
    return _build_schedules(real_train, synthetic_train, snli_train, routing_train, tool_train)


def _schedule_exposure(
    examples: Sequence[TrainingExample], records_by_id: Mapping[str, DecisionRecord]
) -> dict[str, object]:
    datasets: Counter[str] = Counter()
    records: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    gold_positions: Counter[str] = Counter()
    for example in examples:
        record = records_by_id.get(example.record_id)
        if record is None:
            raise ValueError("training schedule references an unknown record ID")
        if not 0 <= example.gold_index < len(example.request.options):
            raise ValueError("training schedule gold index is outside its option order")
        if example.gold_option_id != example.request.options[example.gold_index].id:
            raise ValueError("training schedule gold mapping is inconsistent")
        datasets[record.dataset_id] += 1
        records[record.record_id] += 1
        option_counts[str(len(example.request.options))] += 1
        gold_positions[str(example.gold_index)] += 1
    return {
        "by_source": dict(sorted(datasets.items())),
        "by_record": dict(sorted(records.items())),
        "by_option_count": dict(sorted(option_counts.items(), key=lambda item: int(item[0]))),
        "by_gold_position": dict(sorted(gold_positions.items(), key=lambda item: int(item[0]))),
        "distinct_record_count": len(records),
        "repeated_record_count": sum(count > 1 for count in records.values()),
        "repeated_presentation_count": sum(count - 1 for count in records.values()),
        "max_record_exposure": max(records.values(), default=0),
    }


def audit_training_schedules(
    schedules: Mapping[str, Sequence[TrainingExample]],
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
    routing_train: Sequence[DecisionRecord],
    tool_train: Sequence[DecisionRecord],
) -> dict[str, object]:
    """Validate exact regenerated schedules and report canonical hashes/exposure."""

    expected = build_training_schedules(
        real_train, synthetic_train, snli_train, routing_train, tool_train
    )
    if not isinstance(schedules, Mapping) or set(schedules) != set(ARM_NAMES):
        raise ValueError("training schedule arms do not match the exact arm allowlist")
    normalized: dict[str, tuple[TrainingExample, ...]] = {}
    for arm in ARM_NAMES:
        try:
            normalized[arm] = tuple(schedules[arm])
        except TypeError as exc:
            raise ValueError(f"training schedule {arm} must be a sequence") from exc
        if len(normalized[arm]) != TRAINING_PRESENTATIONS:
            raise ValueError(f"training schedule {arm} has the wrong presentation count")
        if any(type(example) is not TrainingExample for example in normalized[arm]):
            raise ValueError(f"training schedule {arm} must contain TrainingExample values")
        if any(type(example.gold_index) is not int for example in normalized[arm]):
            raise ValueError(f"training schedule {arm} gold index must be an integer")
        if normalized[arm] != expected[arm]:
            raise ValueError(
                f"training schedule {arm} does not match the fixed stream order/options/gold"
            )

    records_by_id = {
        record.record_id: record
        for pool in (real_train, synthetic_train, snli_train, routing_train, tool_train)
        for record in pool
    }
    hashes: dict[str, str] = {}
    exposure: dict[str, dict[str, object]] = {}
    for arm in ARM_NAMES:
        examples = normalized[arm]
        canonical_rows = [
            {
                "record_id": example.record_id,
                "dataset_id": records_by_id[example.record_id].dataset_id,
                "source_group_id": records_by_id[example.record_id].source_group_id,
                "request_hash": example.request.request_hash,
                "option_ids": [option.id for option in example.request.options],
                "gold_option_id": example.gold_option_id,
                "gold_index": example.gold_index,
            }
            for example in examples
        ]
        hashes[arm] = _digest(canonical_rows)
        exposure[arm] = _schedule_exposure(examples, records_by_id)
    return {
        "presentation_count": TRAINING_PRESENTATIONS,
        "optimizer_update_count": TRAINING_UPDATES,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "schedule_sha256": hashes,
        "exposure": exposure,
    }


def _validate_development_pools(
    routing_development: Sequence[DecisionRecord],
    tool_development: Sequence[DecisionRecord],
) -> None:
    for family, records in (
        ("routing", routing_development),
        ("tool", tool_development),
    ):
        _validate_new_pool(
            records, family=family, split="development", label=f"{family} development data"
        )
    mixture_training_data._validate_disjoint_pools((routing_development, tool_development))


def _presentation(
    record: DecisionRecord, options: Sequence[Option], order_index: int, family: str
) -> dict[str, object]:
    order_ids = [option.id for option in options]
    request = record.request.model_copy(update={"options": tuple(options)})
    presentation_id = _digest(
        {
            "namespace": "runtime-rules-v1",
            "family": family,
            "record_id": record.record_id,
            "order_index": order_index,
            "request_hash": request.request_hash,
            "order_ids": order_ids,
        }
    )
    return {
        "presentation_id": presentation_id,
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "source_group_id": record.source_group_id,
        "request_hash": request.request_hash,
        "order_index": order_index,
        "order_ids": order_ids,
        "request": request.model_dump(mode="json"),
    }


def _build_new_evaluation_presentations(
    routing_development: Sequence[DecisionRecord],
    tool_development: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for family, records in (
        ("routing", routing_development),
        ("tool", tool_development),
    ):
        for record in sorted(records, key=lambda item: item.record_id):
            options = record.request.options
            for order_index in range(len(options)):
                rotated = options[order_index:] + options[:order_index]
                rows.append(_presentation(record, rotated, order_index, family))
    if len(rows) != NEW_PANEL_PRESENTATIONS:
        raise ValueError("new development panel must contain exactly 164 presentations")
    if len({row["presentation_id"] for row in rows}) != NEW_PANEL_PRESENTATIONS:
        raise ValueError("new development presentation IDs must be unique")
    return tuple(rows)


def build_new_evaluation_presentations(
    routing_development: Sequence[DecisionRecord],
    tool_development: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    """Build all cyclic menu rotations as label-free routing/tool requests."""

    _validate_development_pools(routing_development, tool_development)
    return _build_new_evaluation_presentations(routing_development, tool_development)


def _validate_panel_rows(
    presentations: Sequence[Mapping[str, object]], expected: Sequence[dict[str, object]]
) -> None:
    if not isinstance(presentations, Sequence) or isinstance(presentations, (str, bytes)):
        raise TypeError("development presentations must be a sequence of request-only rows")
    for row in presentations:
        if not isinstance(row, Mapping) or set(row) != PRESENTATION_FIELDS:
            raise ValueError("development presentation fields do not match the request-only schema")
        if type(row["order_index"]) is not int:
            raise ValueError("development presentation order_index must be an integer")
    actual_ids = [row.get("presentation_id") for row in presentations]
    if any(not isinstance(presentation_id, str) for presentation_id in actual_ids):
        raise ValueError("development presentation IDs must be strings")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("development presentations contain duplicate presentation IDs")
    expected_ids = [row["presentation_id"] for row in expected]
    if set(actual_ids) != set(expected_ids):
        if set(expected_ids) - set(actual_ids):
            raise ValueError("development presentations are missing expected rows")
        raise ValueError("development presentations contain unknown rows")
    if actual_ids != expected_ids:
        raise ValueError(
            "development presentation ordering does not match family/record/order order"
        )
    for actual, planned in zip(presentations, expected, strict=True):
        if actual != planned:
            if (
                actual.get("order_ids") != planned["order_ids"]
                or actual.get("request") != planned["request"]
            ):
                raise ValueError("development presentation options or request changed")
            if actual.get("request_hash") != planned["request_hash"]:
                raise ValueError("development presentation request hash changed")
            raise ValueError("development presentation metadata changed")


def audit_new_evaluation_presentations(
    presentations: Sequence[Mapping[str, object]],
    routing_development: Sequence[DecisionRecord],
    tool_development: Sequence[DecisionRecord],
) -> dict[str, object]:
    """Verify the complete deterministic panel and summarize its CPU-side exposure."""

    expected = build_new_evaluation_presentations(routing_development, tool_development)
    _validate_panel_rows(presentations, expected)
    records = (*routing_development, *tool_development)
    by_id = {record.record_id: record for record in records}
    family_counts: Counter[str] = Counter()
    menu_counts: Counter[str] = Counter()
    family_menu_counts: dict[str, Counter[str]] = {"routing": Counter(), "tool": Counter()}
    gold_positions: Counter[str] = Counter()
    for row in presentations:
        dataset_id = cast(str, row["dataset_id"])
        record_id = cast(str, row["record_id"])
        order_ids = cast(list[str], row["order_ids"])
        family = "routing" if dataset_id == NEW_DEVELOPMENT_DATASET_IDS["routing"] else "tool"
        record = by_id[record_id]
        count = len(order_ids)
        family_counts[family] += 1
        menu_counts[str(count)] += 1
        family_menu_counts[family][str(count)] += 1
        gold_positions[str(order_ids.index(record.answer_id))] += 1
    return {
        "panel_sha256": _digest(list(presentations)),
        "record_count": len(records),
        "source_group_count": len({record.source_group_id for record in records}),
        "presentation_count": len(presentations),
        "by_family": dict(sorted(family_counts.items())),
        "by_menu_count": dict(sorted(menu_counts.items(), key=lambda item: int(item[0]))),
        "by_family_menu_count": {
            family: dict(sorted(counts.items(), key=lambda item: int(item[0])))
            for family, counts in family_menu_counts.items()
        },
        "by_gold_position": dict(sorted(gold_positions.items(), key=lambda item: int(item[0]))),
    }


def reload_presentations(
    presentations: Sequence[Mapping[str, object]],
    routing_development: Sequence[DecisionRecord],
    tool_development: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    """Validate the complete panel, then choose 16 sorted parity rows per family."""

    audit_new_evaluation_presentations(presentations, routing_development, tool_development)
    selected: list[dict[str, object]] = []
    for _family, dataset_id in NEW_DEVELOPMENT_DATASET_IDS.items():
        family_rows = sorted(
            (row for row in presentations if row["dataset_id"] == dataset_id),
            key=lambda row: (row["record_id"], row["order_index"]),
        )
        selected.extend(dict(row) for row in family_rows[:RELOAD_PER_FAMILY])
    if len(selected) != 2 * RELOAD_PER_FAMILY:
        raise ValueError("reload panel must contain exactly 16 presentations per family")
    return tuple(selected)
