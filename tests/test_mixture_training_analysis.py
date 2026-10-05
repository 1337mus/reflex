import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path

import pytest

from experiments import (
    mixture_training_analysis as analysis,
)
from experiments import (
    mixture_training_calibration as analysis_calibration,
)
from experiments import mixture_training_data as mixture_data
from experiments import (
    mixture_training_statistics as statistics,
)
from experiments.mixture_training_contracts import MODEL_ID, MODEL_REVISION, RUNTIME_VERSION_PINS
from reflex_decisions import calibration
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def test_paired_group_delta_is_mixture_minus_control() -> None:
    delta = analysis.paired_group_delta(
        {"a": Fraction(4, 5), "b": Fraction(1, 2)},
        {"a": Fraction(1, 5), "b": Fraction(3, 4)},
    )

    assert delta == {"a": Fraction(3, 5), "b": Fraction(-1, 4)}


def test_paired_group_delta_rejects_unpaired_group_sets() -> None:
    with pytest.raises(ValueError, match="same source groups"):
        analysis.paired_group_delta({"a": Fraction(1)}, {"b": Fraction(0)})


def test_fraction_thresholds_pass_at_boundary_without_rounding() -> None:
    threshold = Fraction(50, 56) - Fraction(1, 20)

    assert analysis.passes_minimum(threshold, threshold)
    assert not analysis.passes_minimum(threshold - Fraction(1, 10_000), threshold)


def test_one_task_regression_fails_even_when_real_task_mean_is_unchanged() -> None:
    selected = {
        "snli-pilot-v1-development": {
            "base": {"original-snli": Fraction(1)},
            "real_only": {"original-snli": Fraction(1)},
            "synthetic_mix": {"original-snli": Fraction(1)},
        },
        "snli-balanced-v1-development": {
            "base": {"balanced": Fraction(9, 10)},
            "real_only": {"balanced": Fraction(9, 10)},
            "synthetic_mix": {"balanced": Fraction(1)},
        },
        "dbpedia14-pilot-v1-development": {
            "base": {"dbpedia": Fraction(1, 2)},
            "real_only": {"dbpedia": Fraction(0)},
            "synthetic_mix": {"dbpedia": Fraction(1)},
        },
        "sms-pilot-v1-development": {
            "base": {"sms": Fraction(1, 2)},
            "real_only": {"sms": Fraction(1)},
            "synthetic_mix": {"sms": Fraction(0)},
        },
    }
    original = {
        "dbpedia14-pilot-v1-development": {
            name: {"dbpedia": Fraction(1)} for name in ("base", "real_only", "synthetic_mix")
        },
        "sms-pilot-v1-development": {
            name: {"sms": Fraction(1)} for name in ("base", "real_only", "synthetic_mix")
        },
        "snli-pilot-v1-development": {
            name: {"original-snli": Fraction(1)} for name in ("base", "real_only", "synthetic_mix")
        },
    }

    history = {
        dataset_id: {group_id: Fraction(1)}
        for dataset_id, group_id in (
            ("dbpedia14-pilot-v1-development", "dbpedia"),
            ("sms-pilot-v1-development", "sms"),
            ("snli-pilot-v1-development", "original-snli"),
        )
    }
    report, _bootstrap = statistics._gate_report(selected, original, history)
    components = report["components"]

    assert components["dbpedia_selected_mix_minus_real_only"]["passed"]
    assert not components["sms_selected_mix_minus_real_only"]["passed"]
    assert not report["overall_passed"]


def test_group_bootstrap_uses_all_five_paired_groups() -> None:
    group_ids = tuple(f"group-{index}" for index in range(5))
    selected = {
        dataset_id: {
            "base": {group_id: Fraction(0) for group_id in group_ids},
            "real_only": {group_id: Fraction(0) for group_id in group_ids},
            "synthetic_mix": {group_id: Fraction(0) for group_id in group_ids},
        }
        for dataset_id in (
            "dbpedia14-pilot-v1-development",
            "sms-pilot-v1-development",
            "snli-pilot-v1-development",
            "snli-balanced-v1-development",
        )
    }
    selected["snli-balanced-v1-development"]["synthetic_mix"] = {
        group_id: Fraction(1) if index < 3 else Fraction(0)
        for index, group_id in enumerate(group_ids)
    }
    original = {
        dataset_id: {
            name: {group_id: Fraction(1) for group_id in group_ids}
            for name in ("base", "real_only", "synthetic_mix")
        }
        for dataset_id in (
            "dbpedia14-pilot-v1-development",
            "sms-pilot-v1-development",
            "snli-pilot-v1-development",
        )
    }

    history = {
        dataset_id: {group_id: Fraction(1) for group_id in group_ids}
        for dataset_id in (
            "dbpedia14-pilot-v1-development",
            "sms-pilot-v1-development",
            "snli-pilot-v1-development",
        )
    }
    report, bootstrap = statistics._gate_report(selected, original, history)
    rng = random.Random(statistics.BOOTSTRAP_SEED)
    statistics._bootstrap_samples(5, rng)  # DBpedia groups
    statistics._bootstrap_samples(5, rng)  # SMS groups
    balanced_samples = statistics._bootstrap_samples(5, rng)
    delta = {
        group_id: selected["snli-balanced-v1-development"]["synthetic_mix"][group_id]
        - selected["snli-balanced-v1-development"]["real_only"][group_id]
        for group_id in group_ids
    }
    expected = statistics._bootstrap_interval(delta, balanced_samples)

    assert all(len(sample) == 5 for sample in balanced_samples)
    assert any(any(index >= 3 for index in sample) for sample in balanced_samples)
    assert bootstrap["intervals"][
        "balanced_snli_mix_minus_real_only"
    ] == statistics._interval_report(expected)


def test_zero_bootstrap_boundary_fails_strict_transfer_gate_exactly() -> None:
    numerators = [
        7,
        7,
        -3,
        -5,
        -5,
        8,
        -7,
        2,
        2,
        7,
        -4,
        10,
        8,
        -8,
        10,
        -8,
        -5,
        8,
        0,
        -5,
        9,
        -6,
        -7,
        -4,
        4,
        -4,
        -3,
        -8,
        5,
        7,
        6,
        6,
        5,
        5,
        5,
        9,
        6,
        0,
        5,
        -1,
        1,
        1,
        8,
        -3,
        -3,
        -5,
        2,
        5,
        3,
        1,
        -6,
        5,
        9,
        4,
        4,
        7,
        3,
        6,
        -6,
        -7,
        4,
        -5,
        6,
        -3,
    ]
    balanced_groups = tuple(f"balanced-{index:03}" for index in range(64))
    control = Fraction(8, 18)
    selected = {
        "dbpedia14-pilot-v1-development": {
            name: {f"dbpedia-{index:02}": Fraction(1) for index in range(56)}
            for name in ("base", "real_only", "synthetic_mix")
        },
        "sms-pilot-v1-development": {
            name: {f"sms-{index:02}": Fraction(1) for index in range(60)}
            for name in ("base", "real_only", "synthetic_mix")
        },
        "snli-balanced-v1-development": {
            "base": {group: control for group in balanced_groups},
            "real_only": {group: control for group in balanced_groups},
            "synthetic_mix": {
                group: Fraction(8 + numerator, 18)
                for group, numerator in zip(balanced_groups, numerators, strict=True)
            },
        },
        "snli-pilot-v1-development": {
            name: {f"snli-{index:03}": Fraction(1) for index in range(128)}
            for name in ("base", "real_only", "synthetic_mix")
        },
    }
    dbpedia_history = {f"dbpedia-{index:02}": Fraction(index < 50) for index in range(56)}
    sms_history = {f"sms-{index:02}": Fraction(index < 58) for index in range(60)}
    snli_history = {f"snli-{index:03}": Fraction(index < 57) for index in range(128)}
    original = {
        "dbpedia14-pilot-v1-development": {
            name: dict(dbpedia_history) for name in ("base", "real_only", "synthetic_mix")
        },
        "sms-pilot-v1-development": {
            name: dict(sms_history) for name in ("base", "real_only", "synthetic_mix")
        },
        "snli-balanced-v1-development": {
            name: {group: Fraction(1) for group in balanced_groups}
            for name in ("base", "real_only", "synthetic_mix")
        },
        "snli-pilot-v1-development": {
            name: dict(snli_history) for name in ("base", "real_only", "synthetic_mix")
        },
    }

    report, bootstrap = statistics._gate_report(
        selected,
        original,
        {
            "dbpedia14-pilot-v1-development": dbpedia_history,
            "sms-pilot-v1-development": sms_history,
            "snli-pilot-v1-development": snli_history,
        },
    )
    primary = report["components"]["balanced_snli_mix_minus_real_only"]
    assert primary["value"]["fraction"] == "89/1152"
    assert primary["paired_bootstrap_95"]["lower"] == 0.0
    assert not primary["passed"]
    assert not report["overall_passed"]

    rng = random.Random(statistics.BOOTSTRAP_SEED)
    statistics._bootstrap_samples(56, rng)
    statistics._bootstrap_samples(60, rng)
    samples = statistics._bootstrap_samples(64, rng)
    positive = statistics._bootstrap_interval(
        {group: Fraction(1, 18) for group in balanced_groups}, samples
    )
    assert positive["lower"] == Fraction(1, 18)
    assert statistics._gate(
        value=Fraction(1, 18),
        threshold=Fraction(1, 20),
        comparison="synthetic_mix - real_only",
        interval=positive,
        require_positive_lower=True,
    )["passed"]


def test_gate_report_rejects_incomplete_paired_group_sets() -> None:
    selected = {
        "snli-balanced-v1-development": {
            "base": {"a": Fraction(0)},
            "real_only": {"a": Fraction(0)},
            "synthetic_mix": {"a": Fraction(0), "b": Fraction(1)},
        }
    }

    with pytest.raises(ValueError, match="same source groups"):
        statistics._gate_report(selected, {}, {})


def test_historical_intervals_pair_source_groups_and_keep_frozen_point_anchors() -> None:
    dbpedia_groups = tuple(f"dbpedia-{index}" for index in range(56))
    sms_groups = tuple(f"sms-{index}" for index in range(60))
    snli_groups = tuple(f"snli-{index}" for index in range(128))
    balanced_groups = tuple(f"balanced-{index}" for index in range(64))
    selected = {
        "dbpedia14-pilot-v1-development": {
            name: {group: Fraction(0) for group in dbpedia_groups}
            for name in ("base", "real_only", "synthetic_mix")
        },
        "sms-pilot-v1-development": {
            name: {group: Fraction(0) for group in sms_groups}
            for name in ("base", "real_only", "synthetic_mix")
        },
        "snli-pilot-v1-development": {
            name: {group: Fraction(0) for group in snli_groups}
            for name in ("base", "real_only", "synthetic_mix")
        },
        "snli-balanced-v1-development": {
            name: {group: Fraction(0) for group in balanced_groups}
            for name in ("base", "real_only", "synthetic_mix")
        },
    }
    historical = {
        "dbpedia14-pilot-v1-development": {
            group: Fraction(index < 50) for index, group in enumerate(dbpedia_groups)
        },
        "sms-pilot-v1-development": {
            group: Fraction(index < 58) for index, group in enumerate(sms_groups)
        },
        "snli-pilot-v1-development": {
            group: Fraction(index < 57) for index, group in enumerate(snli_groups)
        },
    }
    original = {
        dataset_id: {name: dict(groups) for name, groups in states.items()}
        for dataset_id, states in selected.items()
    }
    # One DBpedia group improves. The gate point still compares to the frozen 50/56
    # accuracy; its interval resamples final-minus-historical rates by matching group.
    original["dbpedia14-pilot-v1-development"]["synthetic_mix"] = dict(
        historical["dbpedia14-pilot-v1-development"]
    )
    original["dbpedia14-pilot-v1-development"]["synthetic_mix"][dbpedia_groups[50]] = Fraction(1)

    report, bootstrap = statistics._gate_report(selected, original, historical)
    rng = random.Random(statistics.BOOTSTRAP_SEED)
    dbpedia_samples = statistics._bootstrap_samples(len(dbpedia_groups), rng)
    expected_interval = statistics._bootstrap_interval(
        {group: Fraction(group == dbpedia_groups[50]) for group in dbpedia_groups},
        dbpedia_samples,
    )
    component = report["components"]["dbpedia_original_anchor"]

    assert component["value"]["fraction"] == "1/56"
    assert component["minimum"]["fraction"] == "-1/20"
    assert bootstrap["intervals"]["dbpedia_original_anchor"] == statistics._interval_report(
        expected_interval
    )


def test_historical_reference_requires_one_record_per_anchor_group() -> None:
    real, _balanced, _synthetic, _panel, _pins = _fixture_records_and_panel()
    counts = {
        dataset_id: {
            row.source_group_id: {"correct": int(index < correct), "total": 1}
            for index, row in enumerate(row for row in real if row.dataset_id == dataset_id)
        }
        for dataset_id, (correct, _total) in analysis.baselines.HISTORICAL_ACCURACY_ANCHORS.items()
    }
    rates = analysis._historical_reference_rates(
        {"historical_reference_group_counts": counts}, real
    )
    assert {
        name: sum(group.values(), Fraction()) / len(group) for name, group in rates.items()
    } == {
        dataset_id: Fraction(correct, total)
        for dataset_id, (correct, total) in analysis.baselines.HISTORICAL_ACCURACY_ANCHORS.items()
    }

    dbpedia = [row for row in real if row.dataset_id == "dbpedia14-pilot-v1-development"]
    duplicated = [
        row.model_copy(update={"source_group_id": dbpedia[0].source_group_id})
        if row.record_id == dbpedia[1].record_id
        else row
        for row in real
    ]
    with pytest.raises(ValueError, match="one record per source group"):
        analysis._historical_reference_rates(
            {"historical_reference_group_counts": counts}, duplicated
        )


def _fixture_records_and_panel() -> tuple[object, ...]:
    def make_records(
        dataset_id: str, prefix: str, count: int, *, groups_of: int = 1
    ) -> tuple[DecisionRecord, ...]:
        records = []
        for index in range(count):
            options = (
                Option(id="gold", label="Gold"),
                Option(id="wrong-a", label="Wrong A"),
                Option(id="wrong-b", label="Wrong B"),
            )
            request = DecisionRequest(
                context=f"Fixture context {prefix} {index}.",
                question="Which option is supported?",
                options=options,
            )
            records.append(
                DecisionRecord(
                    record_id=f"{prefix}-{index}",
                    dataset_id=dataset_id,
                    source_group_id=f"{prefix}-group-{index // groups_of}",
                    request=request,
                    answer_id="gold",
                )
            )
        return tuple(records)

    real = (
        *make_records("dbpedia14-pilot-v1-development", "dbpedia-dev", 56),
        *make_records("sms-pilot-v1-development", "sms-dev", 60),
        *make_records("snli-pilot-v1-development", "snli-dev", 128),
        *make_records("dbpedia14-pilot-v1-calibration", "dbpedia-cal", 56),
        *make_records("sms-pilot-v1-calibration", "sms-cal", 60),
    )
    balanced = make_records("snli-balanced-v1-development", "balanced-dev", 192, groups_of=3)
    synthetic = (
        *make_records("synthetic-atomic-fact-inference-v1-development", "atomic-dev", 75),
        *make_records("synthetic-numeric-selection-v1-development", "numeric-dev", 50),
        *make_records("synthetic-atomic-fact-inference-v1-calibration", "atomic-cal", 75),
        *make_records("synthetic-numeric-selection-v1-calibration", "numeric-cal", 50),
    )
    records = (*real, *balanced, *synthetic)
    panel = []
    for record in records:
        presentation_id = f"{record.record_id}-presentation"
        panel.append(
            {
                "presentation_id": presentation_id,
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "source_group_id": record.source_group_id,
                "request_hash": record.request.request_hash,
                "order_index": 0,
                "order_ids": [option.id for option in record.request.options],
                "request": record.request.model_dump(mode="json"),
            }
        )
    if len(panel) != 802:
        raise AssertionError("portable analyzer fixture must contain 802 presentations")
    pins = {"data_file_sha256": {}, "protocol_sha256": "test", "source_file_sha256": {}}
    return real, balanced, synthetic, tuple(panel), pins


def _install_fixture_data(monkeypatch: pytest.MonkeyPatch) -> tuple[object, ...]:
    real, balanced, synthetic, panel, pins = _fixture_records_and_panel()
    monkeypatch.setattr(
        analysis.core,
        "load_local_data",
        lambda *_args: (real, balanced, synthetic, pins),
    )
    monkeypatch.setattr(
        mixture_data,
        "build_evaluation_presentations",
        lambda *_args: panel,
    )
    return real, balanced, synthetic, panel, pins


def test_calibration_weights_four_source_tasks_equally(monkeypatch: pytest.MonkeyPatch) -> None:
    real, balanced, synthetic, _panel, _pins = _fixture_records_and_panel()
    records = (*real, *balanced, *synthetic)
    calibration_counts = statistics._CALIBRATION_COUNTS
    calibration_records = [row for row in records if row.dataset_id in calibration_counts]
    outputs = {}
    for record in calibration_records:
        order_ids = [option.id for option in record.request.options]
        outputs[record.record_id] = {
            "record_id": record.record_id,
            "order_index": 0,
            "order_ids": order_ids,
            "candidate_logits": [float(index) for index in range(len(order_ids))],
        }
    captured: list[calibration.CalibrationExample] = []

    def capture_fit(examples: object) -> calibration.TemperatureFitResult:
        captured.extend(examples)  # type: ignore[arg-type]
        return calibration.TemperatureFitResult(
            temperature=1.0,
            raw_nll=1.0,
            fitted_nll=1.0,
            example_count=len(captured),
            grid_range=(0.05, 10.0),
            grid_count=82,
            boundary_hit=False,
        )

    monkeypatch.setattr(calibration, "fit_temperature", capture_fit)
    _fit, report = analysis_calibration._calibration_fit(outputs, records)

    task_weights = {task: 0.0 for task in calibration_counts}
    for record, example in zip(calibration_records, captured, strict=True):
        task_weights[record.dataset_id] += example.weight
    assert task_weights == pytest.approx({task: 1.0 for task in calibration_counts})
    assert report["fit_dataset_counts"] == calibration_counts
    assert report["fit_task_weight"] == {task: 1.0 for task in calibration_counts}


def test_incomplete_pair_is_reported_without_evaluating_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real, balanced, synthetic, _panel, _pins = _install_fixture_data(monkeypatch)
    monkeypatch.setattr(
        analysis,
        "_payloads_and_results",
        lambda *_args: (
            {"experiment_id": "incomplete", "status": "failed", "failure": None},
            {},
            {},
        ),
    )

    report = analysis.analyze_experiment({}, real, balanced, synthetic, {}, {}, root=Path.cwd())

    assert report["status"] == "failed"
    assert report["primary_transfer_contrast"]["status"] == "not_evaluated"
    assert not report["engineering_gate"]["evaluated"]
    assert report["failure"]["reasons"] == [
        "initialize result is missing",
        "real_only result is missing",
        "synthetic_mix result is missing",
    ]


def _success_analysis_fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[object, ...]:
    real, balanced, synthetic, panel, pins = _install_fixture_data(monkeypatch)
    monkeypatch.setattr(mixture_data, "EXPECTED_EVALUATION_PRESENTATIONS", len(panel))
    records = (*real, *balanced, *synthetic)
    records_by_id = {row.record_id: row for row in records}

    def make_output(presentation: dict[str, object], *, correct: bool) -> dict[str, object]:
        record = records_by_id[presentation["record_id"]]
        order_ids = list(presentation["order_ids"])
        winner = (
            record.answer_id
            if correct
            else next(option_id for option_id in order_ids if option_id != record.answer_id)
        )
        request = DecisionRequest.model_validate(presentation["request"])
        output = {
            key: presentation[key]
            for key in (
                "presentation_id",
                "record_id",
                "dataset_id",
                "source_group_id",
                "request_hash",
                "order_index",
                "order_ids",
            )
        }
        output.update(
            {
                "candidate_logits": [
                    1.0 if option_id == winner else 0.0 for option_id in order_ids
                ],
                "winner_option_id": winner,
                "input_tokens": 12,
                "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
            }
        )
        return output

    dbpedia_records = [row for row in real if row.dataset_id == "dbpedia14-pilot-v1-development"]
    sms_records = [row for row in real if row.dataset_id == "sms-pilot-v1-development"]
    original_snli_records = [row for row in real if row.dataset_id == "snli-pilot-v1-development"]
    historical_group_counts = {
        "dbpedia14-pilot-v1-development": {
            row.source_group_id: {"correct": int(index < 50), "total": 1}
            for index, row in enumerate(dbpedia_records)
        },
        "sms-pilot-v1-development": {
            row.source_group_id: {"correct": int(index < 58), "total": 1}
            for index, row in enumerate(sms_records)
        },
        "snli-pilot-v1-development": {
            row.source_group_id: {"correct": int(index < 57), "total": 1}
            for index, row in enumerate(original_snli_records)
        },
    }
    historical_accuracy_counts = {
        dataset_id: {
            "correct": sum(group["correct"] for group in groups.values()),
            "total": sum(group["total"] for group in groups.values()),
        }
        for dataset_id, groups in historical_group_counts.items()
    }
    old_snli_correct = {row.record_id for row in original_snli_records[:57]}
    reused_rows = [row for row in panel if not str(row["dataset_id"]).startswith("synthetic-")]
    initialization_rows = [row for row in panel if str(row["dataset_id"]).startswith("synthetic-")]
    reused = {
        row["presentation_id"]: make_output(
            row,
            correct=(
                row["dataset_id"] != "snli-pilot-v1-development"
                or row["record_id"] in old_snli_correct
            ),
        )
        for row in reused_rows
    }
    initialization_outputs = [make_output(row, correct=False) for row in initialization_rows]
    balanced_group_ids = sorted({row.source_group_id for row in balanced})
    retained_groups = set(balanced_group_ids[: len(balanced_group_ids) // 2])

    def arm_outputs(arm: str) -> list[dict[str, object]]:
        return [
            make_output(
                presentation,
                correct=(
                    arm != "real_only"
                    or presentation["dataset_id"] != "snli-balanced-v1-development"
                    or presentation["source_group_id"] in retained_groups
                ),
            )
            for presentation in panel
        ]

    base_identity = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_class": "Qwen3_5ForCausalLM",
        "config_class": "Qwen3_5TextConfig",
        "layer_count": 24,
        "tied_embeddings": True,
        "no_meta_parameters": True,
        "load_diagnostics": {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
        "tokenizer_file_sha256": {"tokenizer.json": "a" * 64},
        "effective_dtype": "torch.bfloat16",
        "attention_implementation": "eager",
        "use_kernels": False,
        "use_hub_kernels": "NO",
        "versions": {
            name: version for name, version in RUNTIME_VERSION_PINS.items() if name != "Pillow"
        },
    }
    init_provenance = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "versions": dict(RUNTIME_VERSION_PINS),
        "base_model": base_identity,
    }
    initialization = {"snapshot": "update-000", "tensor_sha256": "a" * 64}
    envelope = {"experiment_id": "analysis-success-path", "status": "passed"}
    payloads = {name: {"pins": pins} for name in ("initialize", "real_only", "synthetic_mix")}
    results = {
        "initialize": {
            "status": "passed",
            "provenance": init_provenance,
            "evidence": {
                "outputs": initialization_outputs,
                "initialization": initialization,
                "forward_counts": {"total": 939},
                "output_artifact": {"sha256": "b" * 64},
            },
        },
    }
    for arm in ("real_only", "synthetic_mix"):
        results[arm] = {
            "status": "passed",
            "provenance": dict(init_provenance),
            "evidence": {
                "outputs": arm_outputs(arm),
                "initialization": initialization,
                "forward_counts": {"total": 5063},
                "output_artifact": {"sha256": "c" * 64},
            },
        }

    monkeypatch.setattr(
        analysis.baselines,
        "validate_reused_base_outputs",
        lambda **_kwargs: (
            reused,
            {
                "source_receipt_sha256": {"r2": "a" * 64, "r1": "b" * 64},
                "reused_rows": len(reused),
                "by_source": {
                    "real_r2_base": len(reused_rows) - len(balanced),
                    "balanced_r1_base": len(balanced),
                },
                "presentation_id_lineage": {},
                "source_presentation_ids_preserved_in_lineage": 0,
                "historical_reference_group_counts": historical_group_counts,
                "historical_reference_accuracy_counts": historical_accuracy_counts,
            },
        ),
    )
    monkeypatch.setattr(
        analysis,
        "_payloads_and_results",
        lambda *_args: (envelope, payloads, results),
    )
    return real, balanced, synthetic, envelope, payloads, results


def test_analyze_success_path_transposes_state_metrics_for_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real, balanced, synthetic, _envelope, _payloads, _results = _success_analysis_fixture(
        monkeypatch
    )

    report = analysis.analyze_experiment({}, real, balanced, synthetic, {}, {}, root=Path.cwd())

    json.dumps(report, allow_nan=False)
    assert report["status"] == "passed"
    assert report["primary_transfer_contrast"]["group_mean_delta"]["fraction"] == "1/2"
    assert report["provenance"]["evaluation_presentations"] == 802
    assert report["provenance"]["total_forward_count"] == 11065
    assert report["provenance"]["base_and_arm_model_tokenizer_identity_identical"]
