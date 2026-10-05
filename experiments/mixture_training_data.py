"""CPU-only schedules and evaluation panels for the controlled mixture candidate."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import permutations
from typing import Any

from experiments import real_pilot_core, training_rehearsal_core
from reflex_decisions import pilot_data, snli_diagnostic, synthetic_data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import Option

TrainingExample = training_rehearsal_core.TrainingExample

ARM_NAMES = ("synthetic_repeat", "snli_mix")
RUN_SUFFIXES = {"synthetic_repeat": "-syn", "snli_mix": "-snli"}
SCHEDULE_SEED = 20261007
INIT_SEED = 20261006
SHARED_SYNTHETIC_SEED = 20262007
SNLI_PRESENTATION_SEED = 20263007
SHARED_REAL_COUNT = 504
SYNTHETIC_COUNT = 500
SNLI_TRAIN_COUNT = 500
TRAINING_STREAM_COUNT = 3
TRAINING_PRESENTATIONS = 1512
MICROBATCHES_PER_UPDATE = 4
MAX_UPDATES = 378
UNSCORED_CHECKPOINT_UPDATE = 189
FINAL_CHECKPOINT_UPDATE = 378
UNIQUE_TRAINING_RECORD_COUNT = 1504

REAL_TRAIN_DATASET_COUNTS = {
    real_pilot_core.TRAIN_DATASET_IDS[0]: 252,
    real_pilot_core.TRAIN_DATASET_IDS[1]: 252,
}
SYNTHETIC_TRAIN_DATASET_COUNTS = {
    f"synthetic-{family}-v1-train": 300 if family == synthetic_data.FACT else 200
    for family in synthetic_data.SOURCE
}
SNLI_TRAIN_DATASET_COUNTS = {"snli-training-v1": SNLI_TRAIN_COUNT}
_SYNTHETIC_SPLIT_COUNTS = {
    synthetic_data.FACT: (300, 75, 75),
    synthetic_data.NUMERIC: (200, 50, 50),
}
SYNTHETIC_DATASET_COUNTS = {
    f"synthetic-{family}-v1-{split}": count
    for family, split_counts in _SYNTHETIC_SPLIT_COUNTS.items()
    for split, count in zip(synthetic_data.SPLITS, split_counts, strict=True)
}
REAL_EVALUATION_DATASET_IDS = frozenset(real_pilot_core.DATASET_RECORD_COUNTS).difference(
    real_pilot_core.TRAIN_DATASET_IDS
)
SYNTHETIC_EVALUATION_DATASET_IDS = frozenset(
    dataset_id for dataset_id in SYNTHETIC_DATASET_COUNTS if not dataset_id.endswith("-train")
)
EXPECTED_EVALUATION_PRESENTATIONS = 4023
EXPECTED_SYNTHETIC_PRESENTATIONS = 939

_REAL_EVALUATION_PRESENTATION_COUNTS = {
    "dbpedia14-pilot-v1-development": 1568,
    "sms-pilot-v1-development": 120,
    "dbpedia14-pilot-v1-calibration": 56,
    "sms-pilot-v1-calibration": 60,
    pilot_data.SNLI_DATASET_ID: 128,
}
_SYNTHETIC_PRESENTATION_COUNTS = {
    "synthetic-atomic-fact-inference-v1-development": 450,
    "synthetic-numeric-selection-v1-development": 364,
    "synthetic-atomic-fact-inference-v1-calibration": 75,
    "synthetic-numeric-selection-v1-calibration": 50,
}


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


def _validate_record_bundle(
    records: Sequence[DecisionRecord], expected_counts: Mapping[str, int], label: str
) -> None:
    if any(not isinstance(record, DecisionRecord) for record in records):
        raise TypeError(f"{label} must contain DecisionRecord values")
    if Counter(record.dataset_id for record in records) != Counter(expected_counts):
        raise ValueError(f"{label} dataset IDs or record counts do not match the exact allowlist")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError(f"{label} record IDs must be unique")
    if len({record.request.request_hash for record in records}) != len(records):
        raise ValueError(f"{label} semantic requests must be unique")


def _group_and_request_pools(
    bundles: Sequence[Sequence[DecisionRecord]],
) -> dict[str, tuple[set[str], set[str]]]:
    pools: dict[str, list[DecisionRecord]] = defaultdict(list)
    for bundle in bundles:
        for record in bundle:
            pools[record.dataset_id].append(record)
    return {
        dataset_id: (
            {record.source_group_id for record in records},
            {record.request.request_hash for record in records},
        )
        for dataset_id, records in pools.items()
    }


def _validate_disjoint_pools(bundles: Sequence[Sequence[DecisionRecord]]) -> None:
    records = [record for bundle in bundles for record in bundle]
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("record IDs must be unique across the data bundles")
    pools = _group_and_request_pools(bundles)
    names = tuple(pools)
    for index, left_name in enumerate(names):
        left_groups, left_requests = pools[left_name]
        for right_name in names[index + 1 :]:
            right_groups, right_requests = pools[right_name]
            if left_groups & right_groups:
                raise ValueError(
                    f"source groups overlap across data pools {left_name} and {right_name}"
                )
            if left_requests & right_requests:
                raise ValueError(
                    f"semantic requests overlap across data pools {left_name} and {right_name}"
                )


def _validate_training_bundles(
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
) -> None:
    _validate_record_bundle(real_train, REAL_TRAIN_DATASET_COUNTS, "real training data")
    _validate_record_bundle(
        synthetic_train, SYNTHETIC_TRAIN_DATASET_COUNTS, "synthetic training data"
    )
    _validate_record_bundle(snli_train, SNLI_TRAIN_DATASET_COUNTS, "SNLI training data")
    if len({record.source_group_id for record in snli_train}) != SNLI_TRAIN_COUNT:
        raise ValueError("SNLI training data must contain one record per source group")
    _validate_disjoint_pools((real_train, synthetic_train, snli_train))


def _training_streams(
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
) -> tuple[
    tuple[TrainingExample, ...],
    tuple[TrainingExample, ...],
    tuple[TrainingExample, ...],
    tuple[TrainingExample, ...],
]:
    """Create the fixed shared and arm-specific 504-example streams."""

    shared_real = training_rehearsal_core.epoch_examples(real_train, seed=SCHEDULE_SEED, epoch=0)
    shared_synthetic = (
        training_rehearsal_core.epoch_examples(synthetic_train, seed=SHARED_SYNTHETIC_SEED, epoch=0)
        + training_rehearsal_core.epoch_examples(
            synthetic_train, seed=SHARED_SYNTHETIC_SEED, epoch=1
        )[:4]
    )
    synthetic_extra = (
        training_rehearsal_core.epoch_examples(synthetic_train, seed=SHARED_SYNTHETIC_SEED, epoch=1)
        + training_rehearsal_core.epoch_examples(
            synthetic_train, seed=SHARED_SYNTHETIC_SEED, epoch=2
        )[:4]
    )
    snli_extra = (
        training_rehearsal_core.epoch_examples(snli_train, seed=SNLI_PRESENTATION_SEED, epoch=0)
        + training_rehearsal_core.epoch_examples(snli_train, seed=SNLI_PRESENTATION_SEED, epoch=1)[
            :4
        ]
    )
    return shared_real, shared_synthetic, synthetic_extra, snli_extra


def _flatten_triples(
    real_stream: Sequence[TrainingExample],
    synthetic_stream: Sequence[TrainingExample],
    extra_stream: Sequence[TrainingExample],
) -> tuple[TrainingExample, ...]:
    if any(
        len(stream) != SHARED_REAL_COUNT for stream in (real_stream, synthetic_stream, extra_stream)
    ):
        raise ValueError("each training stream must contain exactly 504 examples")
    return tuple(
        example
        for triple in zip(real_stream, synthetic_stream, extra_stream, strict=True)
        for example in triple
    )


def _fixed_training_schedules(
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
) -> dict[str, tuple[TrainingExample, ...]]:
    shared_real, shared_synthetic, synthetic_extra, snli_extra = _training_streams(
        real_train, synthetic_train, snli_train
    )
    return {
        "synthetic_repeat": _flatten_triples(shared_real, shared_synthetic, synthetic_extra),
        "snli_mix": _flatten_triples(shared_real, shared_synthetic, snli_extra),
    }


def build_training_schedules(
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
) -> dict[str, tuple[TrainingExample, ...]]:
    """Build the paired, deterministic 1,512-presentation training arms."""

    _validate_training_bundles(real_train, synthetic_train, snli_train)
    return _fixed_training_schedules(real_train, synthetic_train, snli_train)


def _training_examples_exposure(
    examples: Sequence[TrainingExample], record_by_id: Mapping[str, DecisionRecord]
) -> dict[str, object]:
    datasets: Counter[str] = Counter()
    records: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    gold_positions: Counter[str] = Counter()
    for example in examples:
        record = record_by_id.get(example.record_id)
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
        "gold_position_index_base": 0,
        "by_dataset": dict(sorted(datasets.items())),
        "by_record": dict(sorted(records.items())),
        "by_option_count": dict(sorted(option_counts.items(), key=lambda item: int(item[0]))),
        "by_gold_position": dict(sorted(gold_positions.items(), key=lambda item: int(item[0]))),
    }


def _validate_schedule_pair(
    schedules: Mapping[str, Sequence[TrainingExample]],
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
) -> None:
    real_ids = {record.record_id for record in real_train}
    synthetic_ids = {record.record_id for record in synthetic_train}
    if set(schedules) != set(ARM_NAMES):
        raise ValueError("training schedule arms do not match the exact arm allowlist")
    normalized = {arm: tuple(schedules[arm]) for arm in ARM_NAMES}
    if any(len(schedule) != TRAINING_PRESENTATIONS for schedule in normalized.values()):
        raise ValueError("training schedule has the wrong presentation count")
    expected = _fixed_training_schedules(real_train, synthetic_train, snli_train)
    if normalized != expected:
        raise ValueError("training schedule does not match the fixed stream order")
    if len(real_ids | synthetic_ids | {record.record_id for record in snli_train}) != (
        UNIQUE_TRAINING_RECORD_COUNT
    ):
        raise ValueError("training source union must contain exactly 1,504 unique records")


def build_training_schedule_audit(
    real_train: Sequence[DecisionRecord],
    synthetic_train: Sequence[DecisionRecord],
    snli_train: Sequence[DecisionRecord],
    schedules: Mapping[str, Sequence[TrainingExample]] | None = None,
) -> dict[str, object]:
    """Return deterministic arm digests and aggregate exposure counts."""

    planned = (
        schedules
        if schedules is not None
        else build_training_schedules(real_train, synthetic_train, snli_train)
    )
    _validate_training_bundles(real_train, synthetic_train, snli_train)
    _validate_schedule_pair(planned, real_train, synthetic_train, snli_train)
    record_by_id = {
        record.record_id: record for record in (*real_train, *synthetic_train, *snli_train)
    }
    schedule_digests: dict[str, str] = {}
    exposures: dict[str, dict[str, object]] = {}
    for arm in ARM_NAMES:
        examples = tuple(planned[arm])
        canonical_rows = [
            {
                "record_id": example.record_id,
                "dataset_id": record_by_id[example.record_id].dataset_id,
                "request_hash": example.request.request_hash,
                "order_ids": [option.id for option in example.request.options],
                "gold_option_id": example.gold_option_id,
                "gold_index": example.gold_index,
            }
            for example in examples
        ]
        schedule_digests[arm] = _digest(canonical_rows)
        exposures[arm] = _training_examples_exposure(examples, record_by_id)
    return {
        "presentation_count": TRAINING_PRESENTATIONS,
        "optimizer_update_count": MAX_UPDATES,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "schedule_sha256": schedule_digests,
        "exposure": exposures,
    }


def _presentation(
    record: DecisionRecord,
    options: Sequence[Option],
    order_index: int,
    *,
    namespace: str,
) -> dict[str, object]:
    order_ids = [option.id for option in options]
    request = record.request.model_copy(update={"options": tuple(options)})
    presentation_id = _digest(
        {
            "namespace": namespace,
            "record_id": record.record_id,
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


def _synthetic_option_orders(record: DecisionRecord) -> tuple[tuple[Option, ...], ...]:
    options = record.request.options
    option_count = len(options)
    if option_count in (2, 3):
        return tuple(tuple(order) for order in permutations(options))
    if option_count in (4, 8, 16):
        return tuple(tuple(options[index:] + options[:index]) for index in range(option_count))
    raise ValueError("synthetic evaluation records must have 2, 3, 4, 8, or 16 options")


def _synthetic_record_orders(record: DecisionRecord) -> tuple[tuple[Option, ...], ...]:
    if record.dataset_id.endswith("-calibration"):
        return (record.request.options,)
    if record.dataset_id.endswith("-development"):
        return _synthetic_option_orders(record)
    raise ValueError("synthetic evaluation record is outside the development/calibration allowlist")


def build_synthetic_evaluation_presentations(
    synthetic_records: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    """Select only the exact synthetic development/calibration IDs and expand orders."""

    _validate_record_bundle(synthetic_records, SYNTHETIC_DATASET_COUNTS, "synthetic bundle")
    _validate_disjoint_pools((synthetic_records,))
    presentations = tuple(
        _presentation(record, options, order_index, namespace="reflex-mixture-synthetic-v1")
        for record in synthetic_records
        if record.dataset_id in SYNTHETIC_EVALUATION_DATASET_IDS
        for order_index, options in enumerate(_synthetic_record_orders(record))
    )
    if Counter(row["dataset_id"] for row in presentations) != Counter(
        _SYNTHETIC_PRESENTATION_COUNTS
    ):
        raise ValueError("synthetic evaluation presentations do not match the fixed panel")
    if len(presentations) != EXPECTED_SYNTHETIC_PRESENTATIONS:
        raise ValueError("synthetic evaluation presentation count is incorrect")
    return presentations


def _balanced_option_orders(record: DecisionRecord) -> tuple[tuple[Option, ...], ...]:
    if record.dataset_id != snli_diagnostic.DATASET_ID or len(record.request.options) != 3:
        raise ValueError("balanced SNLI records must use the exact three-option dataset")
    return tuple(tuple(order) for order in permutations(record.request.options))


def _real_evaluation_presentations(
    real_records: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    all_real_rows = real_pilot_core.build_evaluation_presentations(real_records)
    record_by_id = {record.record_id: record for record in real_records}
    rows = tuple(
        {**row, "source_group_id": record_by_id[row["record_id"]].source_group_id}
        for row in all_real_rows
        if row["dataset_id"] in REAL_EVALUATION_DATASET_IDS
        and (row["dataset_id"] != pilot_data.SNLI_DATASET_ID or row["order_index"] == 0)
    )
    if Counter(row["dataset_id"] for row in rows) != Counter(_REAL_EVALUATION_PRESENTATION_COUNTS):
        raise ValueError("real evaluation presentations do not match the frozen panel")
    return tuple(rows)


def build_evaluation_presentations(
    real_records: Sequence[DecisionRecord],
    balanced_records: Sequence[DecisionRecord],
    synthetic_records: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    """Build the exact 4,023-row label-free development/calibration evaluation panel."""

    _validate_record_bundle(
        real_records, real_pilot_core.DATASET_RECORD_COUNTS, "real pilot bundle"
    )
    _validate_record_bundle(
        balanced_records,
        {snli_diagnostic.DATASET_ID: 192},
        "balanced SNLI bundle",
    )
    _validate_record_bundle(synthetic_records, SYNTHETIC_DATASET_COUNTS, "synthetic bundle")
    _validate_disjoint_pools((real_records, balanced_records, synthetic_records))

    real_rows = _real_evaluation_presentations(real_records)
    balanced_rows = tuple(
        _presentation(record, options, order_index, namespace="reflex-mixture-balanced-snli-v1")
        for record in balanced_records
        for order_index, options in enumerate(_balanced_option_orders(record))
    )
    synthetic_rows = build_synthetic_evaluation_presentations(synthetic_records)
    rows = (*real_rows, *balanced_rows, *synthetic_rows)
    if len(rows) != EXPECTED_EVALUATION_PRESENTATIONS:
        raise ValueError("evaluation panel does not match the fixed 4,023-presentation count")
    if len({row["presentation_id"] for row in rows}) != len(rows):
        raise ValueError("evaluation presentation IDs must be unique")
    return tuple(rows)


def _presentation_exposure(
    presentations: Sequence[Mapping[str, Any]], records: Sequence[DecisionRecord]
) -> dict[str, object]:
    record_by_id = {record.record_id: record for record in records}
    datasets: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    gold_positions: Counter[str] = Counter()
    for row in presentations:
        if set(row) != {
            "presentation_id",
            "record_id",
            "dataset_id",
            "source_group_id",
            "request_hash",
            "order_index",
            "order_ids",
            "request",
        }:
            raise ValueError("evaluation row contains an unexpected or label-bearing field")
        record_id = row.get("record_id")
        record = record_by_id.get(record_id) if isinstance(record_id, str) else None
        if record is None:
            raise ValueError("evaluation panel references an unknown record ID")
        order_ids = row.get("order_ids")
        if not isinstance(order_ids, list) or sorted(order_ids) != sorted(
            option.id for option in record.request.options
        ):
            raise ValueError("evaluation panel option order does not match its source record")
        if row.get("dataset_id") != record.dataset_id:
            raise ValueError("evaluation panel dataset does not match its source record")
        if row.get("source_group_id") != record.source_group_id:
            raise ValueError("evaluation panel group does not match its source record")
        if row.get("request_hash") != record.request.request_hash:
            raise ValueError("evaluation panel request hash does not match its source record")
        request_payload = row.get("request")
        if not isinstance(request_payload, dict) or set(request_payload) != {
            "context",
            "question",
            "options",
        }:
            raise ValueError("evaluation request payload must be label-free")
        expected_request = record.request.model_copy(
            update={
                "options": tuple(
                    option
                    for option_id in order_ids
                    for option in record.request.options
                    if option.id == option_id
                )
            }
        )
        if request_payload != expected_request.model_dump(mode="json"):
            raise ValueError("evaluation request payload does not match its semantic option order")
        datasets[record.dataset_id] += 1
        record_counts[record_id] += 1
        option_counts[str(len(order_ids))] += 1
        gold_positions[str(order_ids.index(record.answer_id))] += 1
    return {
        "gold_position_index_base": 0,
        "by_dataset": dict(sorted(datasets.items())),
        "by_record": dict(sorted(record_counts.items())),
        "by_option_count": dict(sorted(option_counts.items(), key=lambda item: int(item[0]))),
        "by_gold_position": dict(sorted(gold_positions.items(), key=lambda item: int(item[0]))),
    }


def build_evaluation_panel_audit(
    presentations: Sequence[Mapping[str, Any]], records: Sequence[DecisionRecord]
) -> dict[str, object]:
    """Hash the label-free panel and count aggregate gold-position exposures."""

    if len(presentations) != EXPECTED_EVALUATION_PRESENTATIONS:
        raise ValueError("evaluation panel has the wrong presentation count")
    exposure = _presentation_exposure(presentations, records)
    synthetic_count = sum(
        row.get("dataset_id") in SYNTHETIC_EVALUATION_DATASET_IDS for row in presentations
    )
    if synthetic_count != EXPECTED_SYNTHETIC_PRESENTATIONS:
        raise ValueError("evaluation panel does not contain exactly 939 synthetic presentations")
    return {
        "presentation_sha256": _digest(list(presentations)),
        "presentation_count": len(presentations),
        "new_synthetic_presentation_count": synthetic_count,
        "exposure": exposure,
    }
