"""CPU tests for the controlled-mixture schedule and evaluation panel."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pytest

from experiments import real_pilot_core
from reflex_decisions import pilot_data, snli_diagnostic, synthetic_data
from reflex_decisions.broader_data import SNLI_LABELS
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


@lru_cache(maxsize=1)
def _real_train() -> tuple[DecisionRecord, ...]:
    dbpedia_options = tuple(
        Option(id=f"class-{index}", label=f"Class {index}") for index in range(14)
    )
    sms_options = (Option(id="ham", label="Ham"), Option(id="spam", label="Spam"))
    rows = []
    for dataset_id, options in (
        (pilot_data.DBPEDIA_DATASET_IDS[0], dbpedia_options),
        (pilot_data.SMS_DATASET_IDS[0], sms_options),
    ):
        for index in range(252):
            rows.append(
                DecisionRecord(
                    record_id=f"{dataset_id}-fixture-{index}",
                    dataset_id=dataset_id,
                    source_group_id=f"group-{dataset_id}-{index}",
                    request=DecisionRequest(
                        context=f"Self-authored fixture context {dataset_id} {index}.",
                        question="Which option is correct?",
                        options=options,
                    ),
                    answer_id=options[index % len(options)].id,
                )
            )
    return tuple(rows)


@lru_cache(maxsize=1)
def _synthetic_train() -> tuple[DecisionRecord, ...]:
    candidate = synthetic_data.build_candidate()
    train_ids = {f"synthetic-{family}-v1-train" for family in synthetic_data.SOURCE}
    return tuple(record for record in candidate.records if record.dataset_id in train_ids)


@lru_cache(maxsize=1)
def _snli_train() -> tuple[DecisionRecord, ...]:
    options = tuple(Option(id=label, label=label) for label in SNLI_LABELS)
    return tuple(
        DecisionRecord(
            record_id=f"snli-training-v1-fixture-{index}",
            dataset_id="snli-training-v1",
            source_group_id=f"snli-training-group-{index}",
            request=DecisionRequest(
                context=f"Self-authored SNLI training fixture context {index}.",
                question="Which relation follows?",
                options=options,
            ),
            answer_id=SNLI_LABELS[index % len(SNLI_LABELS)],
        )
        for index in range(500)
    )


@lru_cache(maxsize=1)
def _real_bundle() -> tuple[DecisionRecord, ...]:
    rows = []
    for dataset_id, count in real_pilot_core.DATASET_RECORD_COUNTS.items():
        option_count = 14 if dataset_id.startswith("dbpedia14-") else 2
        if dataset_id == pilot_data.SNLI_DATASET_ID:
            option_count = 3
        options = tuple(
            Option(id=f"{dataset_id}-option-{index}", label=f"Option {index}")
            for index in range(option_count)
        )
        for index in range(count):
            rows.append(
                DecisionRecord(
                    record_id=f"{dataset_id}-fixture-{index}",
                    dataset_id=dataset_id,
                    source_group_id=f"group-{dataset_id}-{index}",
                    request=DecisionRequest(
                        context=f"Self-authored fixture context {dataset_id} {index}.",
                        question=f"Self-authored fixture question for {dataset_id}.",
                        options=options,
                    ),
                    answer_id=options[index % len(options)].id,
                )
            )
    return tuple(rows)


@lru_cache(maxsize=1)
def _balanced_snli() -> tuple[DecisionRecord, ...]:
    options = tuple(Option(id=label, label=label) for label in SNLI_LABELS)
    return tuple(
        DecisionRecord(
            record_id=f"{snli_diagnostic.DATASET_ID}-fixture-{group_index}-{label}",
            dataset_id=snli_diagnostic.DATASET_ID,
            source_group_id=f"balanced-group-{group_index}",
            request=DecisionRequest(
                context=(f"Self-authored balanced fixture context {group_index} {label}."),
                question="Which relation follows?",
                options=options,
            ),
            answer_id=label,
        )
        for group_index in range(snli_diagnostic.SELECTED_GROUP_COUNT)
        for label in SNLI_LABELS
    )


def test_three_stream_schedule_aligns_shared_rows_and_uses_twelve_slot_pattern() -> None:
    from experiments import mixture_training_data, training_rehearsal_core

    real, synthetic, snli = _real_train(), _synthetic_train(), _snli_train()
    schedules = mixture_training_data.build_training_schedules(real, synthetic, snli)
    control = schedules["synthetic_repeat"]
    mixture = schedules["snli_mix"]

    assert mixture_training_data.ARM_NAMES == ("synthetic_repeat", "snli_mix")
    assert mixture_training_data.TRAINING_PRESENTATIONS == 1512
    assert mixture_training_data.MICROBATCHES_PER_UPDATE == 4
    assert mixture_training_data.MAX_UPDATES == 378
    assert len(control) == len(mixture) == 1512
    assert all(
        control[index : index + 2] == mixture[index : index + 2] for index in range(0, 1512, 3)
    )
    assert all(
        example.gold_option_id == example.request.options[example.gold_index].id
        for schedule in schedules.values()
        for example in schedule
    )
    for schedule in schedules.values():
        for start in range(0, len(schedule), mixture_training_data.MICROBATCHES_PER_UPDATE):
            update = schedule[start : start + mixture_training_data.MICROBATCHES_PER_UPDATE]
            assert len(update) == 4
    record_by_id = {record.record_id: record for record in (*real, *synthetic, *snli)}
    for schedule in schedules.values():
        for start in range(0, len(schedule), 12):
            twelve_slots = schedule[start : start + 12]
            assert len(twelve_slots) == 12
            datasets = [record_by_id[example.record_id].dataset_id for example in twelve_slots]
            assert all(dataset.endswith("-train") for dataset in datasets[::3])
            assert all(dataset.startswith("synthetic-") for dataset in datasets[1::3])
            assert all(
                dataset.startswith("synthetic-")
                if schedule is control
                else dataset == "snli-training-v1"
                for dataset in datasets[2::3]
            )

    expected_streams = {
        "real": training_rehearsal_core.epoch_examples(real, seed=20261007, epoch=0),
        "shared_synthetic": (
            training_rehearsal_core.epoch_examples(synthetic, seed=20262007, epoch=0)
            + training_rehearsal_core.epoch_examples(synthetic, seed=20262007, epoch=1)[:4]
        ),
        "control_extra": (
            training_rehearsal_core.epoch_examples(synthetic, seed=20262007, epoch=1)
            + training_rehearsal_core.epoch_examples(synthetic, seed=20262007, epoch=2)[:4]
        ),
        "snli_extra": (
            training_rehearsal_core.epoch_examples(snli, seed=20263007, epoch=0)
            + training_rehearsal_core.epoch_examples(snli, seed=20263007, epoch=1)[:4]
        ),
    }
    assert control[0::3] == mixture[0::3] == expected_streams["real"]
    assert control[1::3] == mixture[1::3] == expected_streams["shared_synthetic"]
    assert control[2::3] == expected_streams["control_extra"]
    assert mixture[2::3] == expected_streams["snli_extra"]


def test_schedule_audit_is_deterministic_and_counts_deliberate_repeats() -> None:
    from experiments import mixture_training_data

    real, synthetic, snli = _real_train(), _synthetic_train(), _snli_train()
    schedules = mixture_training_data.build_training_schedules(real, synthetic, snli)
    audit = mixture_training_data.build_training_schedule_audit(real, synthetic, snli, schedules)

    assert audit == mixture_training_data.build_training_schedule_audit(
        real,
        synthetic,
        snli,
        mixture_training_data.build_training_schedules(real, synthetic, snli),
    )
    assert audit["presentation_count"] == 1512
    assert set(audit["schedule_sha256"]) == {"synthetic_repeat", "snli_mix"}
    assert len(set(record.record_id for record in (*real, *synthetic, *snli))) == 1504
    assert audit["exposure"]["synthetic_repeat"]["by_dataset"] == {
        "dbpedia14-pilot-v1-train": 252,
        "sms-pilot-v1-train": 252,
        "synthetic-atomic-fact-inference-v1-train": 604,
        "synthetic-numeric-selection-v1-train": 404,
    }
    assert audit["exposure"]["snli_mix"]["by_dataset"] == {
        "dbpedia14-pilot-v1-train": 252,
        "sms-pilot-v1-train": 252,
        "snli-training-v1": 504,
        "synthetic-atomic-fact-inference-v1-train": 301,
        "synthetic-numeric-selection-v1-train": 203,
    }
    control_synthetic_exposures = Counter(
        audit["exposure"]["synthetic_repeat"]["by_record"][record.record_id] for record in synthetic
    )
    assert control_synthetic_exposures == {2: 492, 3: 8}
    candidate_synthetic_exposures = Counter(
        audit["exposure"]["snli_mix"]["by_record"][record.record_id] for record in synthetic
    )
    assert candidate_synthetic_exposures == {1: 496, 2: 4}
    assert all(
        audit["exposure"]["snli_mix"]["by_record"][record.record_id] in (1, 2) for record in snli
    )
    assert (
        sum(
            count == 2
            for record in snli
            if (count := audit["exposure"]["snli_mix"]["by_record"][record.record_id])
        )
        == 4
    )
    assert not any(
        record.record_id.startswith("snli-training-v1") for record in schedules["synthetic_repeat"]
    )


def test_training_rejects_unknown_or_suffixed_dataset_ids() -> None:
    from experiments import mixture_training_data

    real = list(_real_train())
    real[0] = real[0].model_copy(update={"dataset_id": "dbpedia14-pilot-v1-train-extra"})

    with pytest.raises(ValueError, match="dataset IDs"):
        mixture_training_data.build_training_schedules(real, _synthetic_train(), _snli_train())

    synthetic = list(_synthetic_train())
    synthetic[0] = synthetic[0].model_copy(
        update={"dataset_id": "synthetic-atomic-fact-inference-v1-train-extra"}
    )
    with pytest.raises(ValueError, match="dataset IDs"):
        mixture_training_data.build_training_schedules(_real_train(), synthetic, _snli_train())


def test_schedule_contract_rejects_missing_or_wrong_snli_stream() -> None:
    from experiments import mixture_training_data

    with pytest.raises(TypeError):
        mixture_training_data.build_training_schedules(_real_train(), _synthetic_train())

    schedules = mixture_training_data.build_training_schedules(
        _real_train(), _synthetic_train(), _snli_train()
    )
    altered_candidate = list(schedules["snli_mix"])
    altered_candidate[2] = schedules["synthetic_repeat"][2]
    altered = dict(schedules)
    altered["snli_mix"] = tuple(altered_candidate)
    with pytest.raises(ValueError, match="fixed stream"):
        mixture_training_data.build_training_schedule_audit(
            _real_train(), _synthetic_train(), _snli_train(), altered
        )


def test_synthetic_evaluation_panel_has_exact_orders_and_label_free_rows() -> None:
    from experiments import mixture_training_data
    from experiments.mixture_training_evaluations import _validate_evaluation_rows

    candidate = synthetic_data.build_candidate()
    rows = mixture_training_data.build_synthetic_evaluation_presentations(candidate.records)

    assert len(rows) == 939
    assert _validate_evaluation_rows(list(rows), "initialize") == list(rows)
    assert Counter(row["dataset_id"] for row in rows) == {
        "synthetic-atomic-fact-inference-v1-development": 450,
        "synthetic-numeric-selection-v1-development": 364,
        "synthetic-atomic-fact-inference-v1-calibration": 75,
        "synthetic-numeric-selection-v1-calibration": 50,
    }
    assert all("answer_id" not in row and "gold_index" not in row for row in rows)
    assert all("source_group_id" in row and "request_hash" in row for row in rows)
    for record in candidate.records:
        if record.dataset_id.endswith("-development"):
            first = next(row for row in rows if row["record_id"] == record.record_id)
            assert first["order_index"] == 0
            assert first["order_ids"] == [option.id for option in record.request.options]
            assert first["request"]["options"] == record.request.model_dump(mode="json")["options"]
            if record.dataset_id.startswith("synthetic-atomic-fact"):
                orders = [row["order_ids"] for row in rows if row["record_id"] == record.record_id]
                assert len(orders) == 6
                assert {order.index(record.answer_id) for order in orders} == {0, 1, 2}


def test_full_evaluation_panel_has_frozen_counts_and_rejects_leakage() -> None:
    from experiments import mixture_training_data
    from experiments.mixture_training_evaluations import _validate_evaluation_rows

    real_bundle = _real_bundle()
    synthetic_bundle = synthetic_data.build_candidate().records
    balanced = _balanced_snli()
    rows = mixture_training_data.build_evaluation_presentations(
        real_bundle, balanced, synthetic_bundle
    )

    assert len(rows) == 4023
    assert Counter(row["dataset_id"] for row in rows) == {
        "dbpedia14-pilot-v1-development": 1568,
        "sms-pilot-v1-development": 120,
        "dbpedia14-pilot-v1-calibration": 56,
        "sms-pilot-v1-calibration": 60,
        "snli-pilot-v1-development": 128,
        "snli-balanced-v1-development": 1152,
        "synthetic-atomic-fact-inference-v1-development": 450,
        "synthetic-numeric-selection-v1-development": 364,
        "synthetic-atomic-fact-inference-v1-calibration": 75,
        "synthetic-numeric-selection-v1-calibration": 50,
    }
    assert all(
        set(row)
        == {
            "presentation_id",
            "record_id",
            "dataset_id",
            "source_group_id",
            "request_hash",
            "order_index",
            "order_ids",
            "request",
        }
        for row in rows
    )
    assert _validate_evaluation_rows(list(rows), "train") == list(rows)
    audit = mixture_training_data.build_evaluation_panel_audit(
        rows, (*real_bundle, *balanced, *synthetic_bundle)
    )
    assert audit["presentation_count"] == 4023
    assert audit["new_synthetic_presentation_count"] == 939
    assert audit == mixture_training_data.build_evaluation_panel_audit(
        mixture_training_data.build_evaluation_presentations(
            real_bundle, balanced, synthetic_bundle
        ),
        (*real_bundle, *balanced, *synthetic_bundle),
    )

    altered_panel = [dict(row) for row in rows]
    balanced_record_id = next(
        row["record_id"] for row in altered_panel if row["dataset_id"] == snli_diagnostic.DATASET_ID
    )
    for row in altered_panel:
        if row["record_id"] == balanced_record_id:
            row["source_group_id"] = "new-balanced-group"
    with pytest.raises(ValueError, match="64 groups of three records"):
        _validate_evaluation_rows(altered_panel, "train")

    leaked_group = real_bundle[0].source_group_id
    development_index = next(
        index
        for index, record in enumerate(real_bundle)
        if record.dataset_id == pilot_data.DBPEDIA_DATASET_IDS[1]
    )
    altered_real = list(real_bundle)
    altered_real[development_index] = altered_real[development_index].model_copy(
        update={"source_group_id": leaked_group}
    )
    with pytest.raises(ValueError, match="group"):
        mixture_training_data.build_evaluation_presentations(
            altered_real, balanced, synthetic_bundle
        )

    altered_real = list(real_bundle)
    altered_real[development_index] = altered_real[development_index].model_copy(
        update={"request": real_bundle[0].request}
    )
    with pytest.raises(ValueError, match="request"):
        mixture_training_data.build_evaluation_presentations(
            altered_real, balanced, synthetic_bundle
        )

    altered_real = list(real_bundle)
    altered_real[development_index] = altered_real[development_index].model_copy(
        update={"dataset_id": "dbpedia14-pilot-v1-development-extra"}
    )
    with pytest.raises(ValueError, match="dataset IDs"):
        mixture_training_data.build_evaluation_presentations(
            altered_real, balanced, synthetic_bundle
        )

    assert sum(audit["exposure"]["by_gold_position"].values()) == 4023
