from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from experiments import real_pilot_core, runtime_rule_study_data
from experiments.training_rehearsal_core import TrainingExample, epoch_examples
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _record(
    dataset_id: str,
    family: str,
    index: int,
    menu_size: int,
) -> DecisionRecord:
    options = tuple(
        Option(id=f"choice-{option}", label=f"Choice {option}", description=f"Details {option}")
        for option in range(menu_size)
    )
    request = DecisionRequest(
        context=f"{family} context {index}",
        question=f"{family} question {index}",
        options=options,
    )
    return DecisionRecord(
        record_id=f"{dataset_id}-{index:04d}",
        dataset_id=dataset_id,
        source_group_id=f"{dataset_id}-group-{index:04d}",
        request=request,
        answer_id=options[index % menu_size].id,
    )


def _training_pools() -> tuple[
    tuple[DecisionRecord, ...],
    tuple[DecisionRecord, ...],
    tuple[DecisionRecord, ...],
    tuple[DecisionRecord, ...],
    tuple[DecisionRecord, ...],
]:
    real = tuple(
        _record(dataset_id, dataset_id, index, 4)
        for dataset_id, count in zip(real_pilot_core.TRAIN_DATASET_IDS, (252, 252), strict=True)
        for index in range(count)
    )
    synthetic = tuple(
        _record(dataset_id, dataset_id, index, 6)
        for dataset_id, count in (
            ("synthetic-atomic-fact-inference-v1-train", 300),
            ("synthetic-numeric-selection-v1-train", 200),
        )
        for index in range(count)
    )
    snli = tuple(_record("snli-training-v1", "snli", i, 3) for i in range(500))

    def new_pool(family: str, dataset_id: str) -> tuple[DecisionRecord, ...]:
        menu_sizes = (4,) * 19 + (6,) * 18 + (8,) * 19
        return tuple(_record(dataset_id, family, index, menu_sizes[index]) for index in range(56))

    return (
        real,
        synthetic,
        snli,
        new_pool("routing", "routing-data-v1-train"),
        new_pool("tool", "tool-data-v1-train"),
    )


def _development_pools() -> tuple[tuple[DecisionRecord, ...], tuple[DecisionRecord, ...]]:
    menu_sizes = (4,) * 5 + (6,) * 5 + (8,) * 4
    return tuple(
        tuple(_record(dataset_id, family, index, menu_sizes[index]) for index in range(14))
        for family, dataset_id in (
            ("routing", "routing-data-v1-development"),
            ("tool", "tool-data-v1-development"),
        )
    )  # type: ignore[return-value]


def test_training_schedules_have_fixed_matched_streams_and_exposures() -> None:
    pools = _training_pools()
    schedules = runtime_rule_study_data.build_training_schedules(*pools)

    assert set(schedules) == {"continued_practice", "runtime_mix"}
    control = schedules["continued_practice"]
    treatment = schedules["runtime_mix"]
    assert len(control) == len(treatment) == 1_344
    assert all(isinstance(example, TrainingExample) for example in control + treatment)

    real, synthetic, snli, routing, tool = pools
    shared = (
        epoch_examples(tuple(sorted(real, key=lambda row: row.record_id)), seed=20261010, epoch=0)[
            :336
        ],
        epoch_examples(
            tuple(sorted(synthetic, key=lambda row: row.record_id)), seed=20262010, epoch=0
        )[:336],
        epoch_examples(tuple(sorted(snli, key=lambda row: row.record_id)), seed=20263010, epoch=0)[
            :336
        ],
    )
    expected_control_extra = tuple(
        example for row in zip(*(stream[:112] for stream in shared), strict=True) for example in row
    )
    expected_treatment_extra = tuple(
        example
        for epoch in range(3)
        for row in zip(
            epoch_examples(
                tuple(sorted(routing, key=lambda item: item.record_id)), seed=20264010, epoch=epoch
            ),
            epoch_examples(
                tuple(sorted(tool, key=lambda item: item.record_id)), seed=20265010, epoch=epoch
            ),
            strict=True,
        )
        for example in row
    )

    for update in range(336):
        control_batch = control[update * 4 : (update + 1) * 4]
        treatment_batch = treatment[update * 4 : (update + 1) * 4]
        assert control_batch[:3] == tuple(stream[update] for stream in shared)
        assert control_batch[:3] == treatment_batch[:3]
        assert control_batch[3] == expected_control_extra[update]
        assert treatment_batch[3] == expected_treatment_extra[update]
    new_train_ids = {record.record_id for record in (*pools[3], *pools[4])}
    treatment_new = Counter(
        example.record_id for example in treatment if example.record_id in new_train_ids
    )
    control_new = Counter(
        example.record_id for example in control if example.record_id in new_train_ids
    )
    assert treatment_new == Counter({record_id: 3 for record_id in new_train_ids})
    assert not control_new

    shuffled = tuple(tuple(reversed(pool)) for pool in pools)
    assert runtime_rule_study_data.build_training_schedules(*shuffled) == schedules


def test_training_schedule_audit_binds_gold_order_digest_and_exposures() -> None:
    pools = _training_pools()
    schedules = runtime_rule_study_data.build_training_schedules(*pools)
    report = runtime_rule_study_data.audit_training_schedules(schedules, *pools)
    shuffled_report = runtime_rule_study_data.audit_training_schedules(
        schedules, *(tuple(reversed(pool)) for pool in pools)
    )

    assert report == shuffled_report
    assert report["presentation_count"] == 1_344
    assert report["optimizer_update_count"] == 336
    assert report["microbatches_per_update"] == 4
    assert set(report["schedule_sha256"]) == {"continued_practice", "runtime_mix"}
    assert all(len(value) == 64 for value in report["schedule_sha256"].values())
    exposure = report["exposure"]
    assert exposure["continued_practice"]["distinct_record_count"] == 1_008
    assert exposure["continued_practice"]["repeated_record_count"] == 336
    assert exposure["continued_practice"]["repeated_presentation_count"] == 336
    assert exposure["runtime_mix"]["distinct_record_count"] == 1_120
    assert exposure["runtime_mix"]["repeated_record_count"] == 112
    assert exposure["runtime_mix"]["repeated_presentation_count"] == 224

    record_by_id = {record.record_id: record for pool in pools for record in pool}
    for schedule in schedules.values():
        for example in schedule:
            assert example.gold_option_id == record_by_id[example.record_id].answer_id
            assert example.request.options[example.gold_index].id == example.gold_option_id

    control = schedules["continued_practice"]
    changed_order = list(control)
    changed_order[0], changed_order[1] = changed_order[1], changed_order[0]
    with pytest.raises(ValueError, match="fixed stream order/options/gold"):
        runtime_rule_study_data.audit_training_schedules(
            {**schedules, "continued_practice": tuple(changed_order)}, *pools
        )

    first = control[0]
    wrong_gold = replace(first, gold_index=(first.gold_index + 1) % len(first.request.options))
    with pytest.raises(ValueError, match="fixed stream order/options/gold"):
        runtime_rule_study_data.audit_training_schedules(
            {**schedules, "continued_practice": (wrong_gold, *control[1:])}, *pools
        )

    altered_options = tuple(reversed(first.request.options))
    new_gold_index = tuple(option.id for option in altered_options).index(first.gold_option_id)
    wrong_options = replace(
        first,
        request=first.request.model_copy(update={"options": altered_options}),
        gold_index=new_gold_index,
    )
    with pytest.raises(ValueError, match="fixed stream order/options/gold"):
        runtime_rule_study_data.audit_training_schedules(
            {**schedules, "continued_practice": (wrong_options, *control[1:])}, *pools
        )

    with pytest.raises(ValueError, match="exact arm allowlist"):
        runtime_rule_study_data.audit_training_schedules(
            {**schedules, "unexpected": control}, *pools
        )

    unknown_record = replace(first, record_id="not-in-the-approved-pools")
    with pytest.raises(ValueError, match="fixed stream order/options/gold"):
        runtime_rule_study_data.audit_training_schedules(
            {**schedules, "continued_practice": (unknown_record, *control[1:])}, *pools
        )


@pytest.mark.parametrize("numeric_type", [bool, float], ids=["bool", "float"])
def test_training_schedule_audit_rejects_non_integer_gold_index(numeric_type: type) -> None:
    pools = _training_pools()
    schedules = runtime_rule_study_data.build_training_schedules(*pools)
    control = schedules["continued_practice"]
    source_index = next(
        index
        for index, example in enumerate(control)
        if numeric_type is float or example.gold_index in (0, 1)
    )
    example = control[source_index]
    malformed_index = numeric_type(example.gold_index)
    changed = replace(example, gold_index=malformed_index)
    malformed_control = (*control[:source_index], changed, *control[source_index + 1 :])
    assert (
        runtime_rule_study_data.audit_training_schedules(schedules, *pools)["presentation_count"]
        == 1_344
    )

    with pytest.raises(ValueError, match="gold index must be an integer"):
        runtime_rule_study_data.audit_training_schedules(
            {**schedules, "continued_practice": malformed_control}, *pools
        )


def test_training_schedule_digest_binds_source_group_identity() -> None:
    pools = _training_pools()
    schedules = runtime_rule_study_data.build_training_schedules(*pools)
    scheduled_record_id = schedules["continued_practice"][0].record_id
    changed_pools = list(pools)
    pool_index = next(
        index
        for index, pool in enumerate(pools)
        if any(record.record_id == scheduled_record_id for record in pool)
    )
    changed_pool = list(pools[pool_index])
    record_index = next(
        index
        for index, record in enumerate(changed_pool)
        if record.record_id == scheduled_record_id
    )
    changed_pool[record_index] = changed_pool[record_index].model_copy(
        update={"source_group_id": "fresh-unique-source-group"}
    )
    changed_pools[pool_index] = tuple(changed_pool)
    changed_schedules = runtime_rule_study_data.build_training_schedules(*changed_pools)

    assert changed_schedules == schedules
    original_report = runtime_rule_study_data.audit_training_schedules(schedules, *pools)
    changed_report = runtime_rule_study_data.audit_training_schedules(
        changed_schedules, *changed_pools
    )
    assert changed_report["schedule_sha256"] != original_report["schedule_sha256"]


def test_training_pool_validation_rejects_wrong_split_duplicates_and_group_overlap() -> None:
    pools = _training_pools()
    runtime_rule_study_data.build_training_schedules(*pools)

    routing_development = tuple(
        record.model_copy(update={"dataset_id": "routing-data-v1-development"})
        for record in pools[3]
    )
    with pytest.raises(ValueError, match="routing training data must contain exactly 56"):
        runtime_rule_study_data.build_training_schedules(*pools[:3], routing_development, pools[4])

    duplicate_id = list(pools[3])
    duplicate_id[1] = duplicate_id[1].model_copy(update={"record_id": duplicate_id[0].record_id})
    with pytest.raises(ValueError, match="record IDs must be unique"):
        runtime_rule_study_data.build_training_schedules(*pools[:3], tuple(duplicate_id), pools[4])

    duplicate_request = list(pools[3])
    duplicate_request[1] = duplicate_request[1].model_copy(
        update={"request": duplicate_request[0].request}
    )
    with pytest.raises(ValueError, match="semantic requests must be unique"):
        runtime_rule_study_data.build_training_schedules(
            *pools[:3], tuple(duplicate_request), pools[4]
        )

    duplicate_group = list(pools[3])
    duplicate_group[1] = duplicate_group[1].model_copy(
        update={"source_group_id": duplicate_group[0].source_group_id}
    )
    with pytest.raises(ValueError, match="56 distinct source groups"):
        runtime_rule_study_data.build_training_schedules(
            *pools[:3], tuple(duplicate_group), pools[4]
        )

    overlap = list(pools[4])
    overlap[0] = overlap[0].model_copy(update={"source_group_id": pools[3][0].source_group_id})
    with pytest.raises(ValueError, match="source groups overlap across data pools"):
        runtime_rule_study_data.build_training_schedules(*pools[:3], pools[3], tuple(overlap))


def test_new_development_panel_contains_every_stable_cyclic_order() -> None:
    routing, tool = _development_pools()
    presentations = runtime_rule_study_data.build_new_evaluation_presentations(routing, tool)
    reversed_inputs = runtime_rule_study_data.build_new_evaluation_presentations(
        tuple(reversed(routing)), tuple(reversed(tool))
    )

    assert len(presentations) == 164
    assert presentations == reversed_inputs
    assert len({row["presentation_id"] for row in presentations}) == 164
    assert [row["dataset_id"] for row in presentations[:82]] == ["routing-data-v1-development"] * 82
    assert [row["dataset_id"] for row in presentations[82:]] == ["tool-data-v1-development"] * 82
    assert all(set(row) == runtime_rule_study_data.PRESENTATION_FIELDS for row in presentations)
    assert all(set(row["request"]) == {"context", "question", "options"} for row in presentations)
    assert all(
        "answer_id" not in row and "source_group_id" not in row["request"] for row in presentations
    )

    records_by_id = {record.record_id: record for record in (*routing, *tool)}
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in presentations:
        grouped.setdefault(row["record_id"], []).append(row)
    for record_id, rows in grouped.items():
        record = records_by_id[record_id]
        original = [option.id for option in record.request.options]
        assert rows[0]["order_index"] == 0
        assert rows[0]["order_ids"] == original
        assert len(rows) == len(original)
        for order_index, row in enumerate(rows):
            assert row["order_index"] == order_index
            assert row["order_ids"] == original[order_index:] + original[:order_index]

    report = runtime_rule_study_data.audit_new_evaluation_presentations(
        presentations, routing, tool
    )
    assert report["record_count"] == 28
    assert report["source_group_count"] == 28
    assert report["presentation_count"] == 164
    assert report["by_family"] == {"routing": 82, "tool": 82}
    assert report["by_menu_count"] == {"4": 40, "6": 60, "8": 64}


def test_new_development_panel_audit_rejects_incomplete_mutated_and_wrong_split_rows() -> None:
    routing, tool = _development_pools()
    presentations = runtime_rule_study_data.build_new_evaluation_presentations(routing, tool)
    assert (
        runtime_rule_study_data.audit_new_evaluation_presentations(presentations, routing, tool)[
            "presentation_count"
        ]
        == 164
    )

    with pytest.raises(ValueError, match="missing expected rows"):
        runtime_rule_study_data.audit_new_evaluation_presentations(
            presentations[:-1], routing, tool
        )

    duplicated = (*presentations[:-1], presentations[0])
    with pytest.raises(ValueError, match="duplicate presentation IDs"):
        runtime_rule_study_data.audit_new_evaluation_presentations(duplicated, routing, tool)

    reordered = list(presentations)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="ordering"):
        runtime_rule_study_data.audit_new_evaluation_presentations(reordered, routing, tool)

    changed = [dict(row) for row in presentations]
    changed[0]["order_ids"] = list(reversed(changed[0]["order_ids"]))
    with pytest.raises(ValueError, match="options or request changed"):
        runtime_rule_study_data.audit_new_evaluation_presentations(changed, routing, tool)

    for split in ("train", "calibration", "test"):
        wrong_routing = tuple(
            record.model_copy(update={"dataset_id": f"routing-data-v1-{split}"})
            for record in routing
        )
        with pytest.raises(ValueError, match="routing development data must contain exactly 14"):
            runtime_rule_study_data.build_new_evaluation_presentations(wrong_routing, tool)


@pytest.mark.parametrize("numeric_type", [bool, float], ids=["bool", "float"])
def test_new_development_panel_audit_rejects_non_integer_order_index(numeric_type: type) -> None:
    routing, tool = _development_pools()
    presentations = runtime_rule_study_data.build_new_evaluation_presentations(routing, tool)
    assert (
        runtime_rule_study_data.audit_new_evaluation_presentations(presentations, routing, tool)[
            "presentation_count"
        ]
        == 164
    )
    malformed = [dict(row) for row in presentations]
    target = next(index for index, row in enumerate(malformed) if row["order_index"] == 1)
    malformed[target]["order_index"] = numeric_type(1)

    with pytest.raises(ValueError, match="order_index must be an integer"):
        runtime_rule_study_data.audit_new_evaluation_presentations(malformed, routing, tool)


def test_reload_panel_selects_sorted_sixteen_per_family_after_full_audit() -> None:
    routing, tool = _development_pools()
    presentations = runtime_rule_study_data.build_new_evaluation_presentations(routing, tool)
    reload_rows = runtime_rule_study_data.reload_presentations(presentations, routing, tool)

    assert len(reload_rows) == 32
    for _family, dataset_id in (
        ("routing", "routing-data-v1-development"),
        ("tool", "tool-data-v1-development"),
    ):
        expected = sorted(
            (row for row in presentations if row["dataset_id"] == dataset_id),
            key=lambda row: (row["record_id"], row["order_index"]),
        )[:16]
        actual = [row for row in reload_rows if row["dataset_id"] == dataset_id]
        assert actual == expected

    with pytest.raises(ValueError, match="ordering"):
        runtime_rule_study_data.reload_presentations(tuple(reversed(presentations)), routing, tool)
