from __future__ import annotations

import sys
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from random import Random

import pytest

from experiments import analyze_real_pilot as analyzer
from experiments import real_pilot_core as core
from experiments import training_rehearsal_core
from reflex_decisions import pilot_data
from reflex_decisions.data import DatasetSpec, DecisionRecord, SplitManifest
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def _synthetic_data() -> tuple[SplitManifest, tuple[DecisionRecord, ...]]:
    records: list[DecisionRecord] = []
    specs: list[DatasetSpec] = []
    source_by_dataset = {
        "dbpedia14": ("fancyzhx/dbpedia_14", "topic-classification"),
        "sms": ("uci-sms-spam-collection-228", "spam-detection"),
        "snli": ("stanford-snli-1.0", "natural-language-inference-three-way"),
    }
    for dataset_id, count in core.DATASET_RECORD_COUNTS.items():
        prefix = dataset_id.split("-", 1)[0]
        source_id, task_family = source_by_dataset[prefix]
        split = (
            "train"
            if dataset_id.endswith("-train")
            else "calibration"
            if dataset_id.endswith("-calibration")
            else "development"
        )
        specs.append(
            DatasetSpec(
                dataset_id=dataset_id,
                source_id=source_id,
                task_family=task_family,
                split=split,
                source_uri=f"local://synthetic/{dataset_id}",
                source_revision="synthetic-v1",
                license="synthetic fixture",
            )
        )
        option_ids = (
            tuple(f"topic-{index}" for index in range(14))
            if prefix == "dbpedia14"
            else ("ham", "spam")
            if prefix == "sms"
            else ("entailment", "neutral", "contradiction")
        )
        options = tuple(Option(id=option_id, label=option_id) for option_id in option_ids)
        for index in range(count):
            record_id = f"{dataset_id}-record-{index:03d}"
            records.append(
                DecisionRecord(
                    record_id=record_id,
                    dataset_id=dataset_id,
                    source_group_id=f"{dataset_id}-group-{index:03d}",
                    request=DecisionRequest(
                        context=f"Synthetic context for {record_id}",
                        question="Which option applies?",
                        options=options,
                    ),
                    answer_id=option_ids[index % len(option_ids)],
                )
            )
    manifest = SplitManifest(
        data_kind="benchmark",
        group_partitioned_sources=("fancyzhx/dbpedia_14", "uci-sms-spam-collection-228"),
        datasets=tuple(specs),
    )
    return manifest, tuple(records)


def _valid_receipt(
    payload: dict[str, object], records: tuple[DecisionRecord, ...]
) -> dict[str, object]:
    records_by_id = {record.record_id: record for record in records}
    outputs_by_state: dict[str, list[dict[str, object]]] = {"base": [], "final": []}
    for state in outputs_by_state:
        for row in payload["evaluation_presentations"]:
            record = records_by_id[row["record_id"]]
            order_ids = row["order_ids"]
            index = int(record.record_id.rsplit("-", 1)[1])
            chosen_id = record.answer_id
            if (
                state == "base"
                and record.dataset_id == "dbpedia14-pilot-v1-development"
                and index == 0
                and row["order_index"] == 0
            ):
                chosen_id = next(
                    option.id for option in record.request.options if option.id != chosen_id
                )
            if (
                state == "final"
                and record.dataset_id == "sms-pilot-v1-development"
                and index == 0
                and row["order_index"] == 0
            ):
                chosen_id = next(
                    option.id for option in record.request.options if option.id != chosen_id
                )
            logits = [4.0 if option_id == chosen_id else 0.0 for option_id in order_ids]
            request = DecisionRequest.model_validate(row["request"])
            outputs_by_state[state].append(
                {
                    "presentation_id": row["presentation_id"],
                    "record_id": row["record_id"],
                    "dataset_id": row["dataset_id"],
                    "request_hash": row["request_hash"],
                    "order_ids": order_ids,
                    "candidate_logits": logits,
                    "winner_option_id": chosen_id,
                    "input_tokens": 17,
                    "prompt_sha256": sha256(render_prompt(request).encode("utf-8")).hexdigest(),
                }
            )

    final_by_id = {row["presentation_id"]: row for row in outputs_by_state["final"]}
    parity_ids = sorted(final_by_id)[: core.RELOAD_PARITY_COUNT]
    run_dir = f"/artifacts/runs/{payload['run_id']}"
    source_hashes = payload["pins"]["source_file_sha256"]
    return {
        "schema_version": 1,
        "status": "passed",
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "pins": payload["pins"],
        "provenance": {
            "model_id": core.MODEL_ID,
            "model_revision": core.MODEL_REVISION,
            "source_file_sha256": source_hashes,
            "measured_source_file_sha256": source_hashes,
        },
        "evidence": {
            "optimizer_updates_completed": core.MAX_UPDATES,
            "forward_counts": {
                "base_evaluation": core.BASE_EVALUATION_COUNT,
                "training": core.TRAIN_FORWARD_COUNT,
                "final_evaluation": core.FINAL_EVALUATION_COUNT,
                "reload_parity": core.RELOAD_PARITY_COUNT,
                "total": core.MAX_FORWARD_COUNT,
            },
            "base_gradients_none": True,
            "adapter_inventory": {
                "module_count": training_rehearsal_core.EXPECTED_LORA_MODULES,
                "adapter_tensor_count": training_rehearsal_core.EXPECTED_ADAPTER_TENSORS,
                "trainable_parameter_count": training_rehearsal_core.EXPECTED_TRAINABLE_PARAMETERS,
                "base_parameters_frozen_bf16": True,
            },
            "training_step_losses": [
                {"update": update, "mean_loss": 1.0, "lo_ra_b_gradient_l1": 0.5}
                for update in range(1, core.MAX_UPDATES + 1)
            ],
            "adapter_update": {
                "changed_tensor_count": 1,
                "changed_tensor_names": ["base_model.layer.0.lora_A.default.weight"],
            },
            "adapter_paths": [
                {
                    "update": update,
                    "path": f"{run_dir}/adapter-update-{update:03d}",
                    "files_sha256": {
                        "adapter_model.safetensors": "f" * 64,
                        "adapter_config.json": "e" * 64,
                    },
                }
                for update in (126, 252)
            ],
            "base_outputs": outputs_by_state["base"],
            "final_outputs": outputs_by_state["final"],
            "output_artifacts": {
                name: {"path": f"{run_dir}/{name}.json", "sha256": "c" * 64}
                for name in ("base_outputs", "final_outputs")
            },
            "reload_parity": {
                "presentation_ids": parity_ids,
                "reloaded_logits": [
                    list(final_by_id[presentation_id]["candidate_logits"])
                    for presentation_id in parity_ids
                ],
                "adapter_tensor_keys_shapes_values_match": True,
                "winner_match": True,
                "max_abs_logit_diff": 0.0,
            },
        },
    }


@pytest.fixture(scope="module")
def synthetic_run() -> tuple[
    SplitManifest,
    tuple[DecisionRecord, ...],
    dict[str, object],
    dict[str, object],
]:
    manifest, records = _synthetic_data()
    source_hashes = {path: "a" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[core.PROTOCOL_PATH] = "e" * 64
    payload = core.build_remote_payload(
        manifest,
        records,
        run_id="cpu-analyzer-test",
        nonce="8d4bfbb3-778e-4894-8f18-d4dd7f1eb14b",
        records_sha256="b" * 64,
        manifest_sha256="c" * 64,
        recipe_sha256="d" * 64,
        protocol_sha256="e" * 64,
        source_file_sha256=source_hashes,
    )
    return manifest, records, payload, _valid_receipt(payload, records)


def test_analyzer_uses_semantic_ids_and_counts_order_changes_by_source_group(
    synthetic_run: tuple[
        SplitManifest,
        tuple[DecisionRecord, ...],
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    manifest, records, payload, receipt = synthetic_run

    report = analyzer.analyze_receipt(
        receipt,
        payload,
        records,
        manifest,
        records_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )

    dbpedia = report["states"]["base"]["order_analysis"]["dbpedia14-pilot-v1-development"]
    assert dbpedia["source_group_count"] == 56
    assert dbpedia["independent_unit_count"] == 56
    assert dbpedia["order_count"] == 28
    assert dbpedia["semantic_top_answer_flip_record_count"] == 1
    assert dbpedia["semantic_top_answer_flip_fraction"] == pytest.approx(1 / 56)
    assert dbpedia["correct_by_order_index"][0]["correct_count"] == 55
    assert dbpedia["correct_by_order_index"][1]["correct_count"] == 56

    paired_dbpedia = report["paired_comparison"]["per_dataset"]["dbpedia14-pilot-v1-development"]
    assert paired_dbpedia["base_wrong_final_correct"] == 1
    assert paired_dbpedia["base_correct_final_wrong"] == 0
    paired_sms = report["paired_comparison"]["per_dataset"]["sms-pilot-v1-development"]
    assert paired_sms["base_correct_final_wrong"] == 1
    assert paired_sms["base_wrong_final_correct"] == 0
    assert (
        report["states"]["base"]["metrics"]["temperature_1"]["dbpedia-sms-macro"]["dataset_count"]
        == 2
    )
    base_raw = report["states"]["base"]["metrics"]["temperature_1"][
        "dbpedia14-pilot-v1-development"
    ]
    base_fitted = report["states"]["base"]["metrics"]["fitted"]["dbpedia14-pilot-v1-development"]
    assert base_raw["accuracy"] == base_fitted["accuracy"]
    assert base_raw["macro_f1"] == base_fitted["macro_f1"]
    assert set(report["states"]["base"]["metrics"]["temperature_1"]) == {
        "dbpedia14-pilot-v1-development",
        "sms-pilot-v1-development",
        "snli-pilot-v1-development",
        "dbpedia-sms-macro",
    }
    assert report["states"]["base"]["calibration_fit"]["example_count"] == 116
    assert report["states"]["final"]["model_revision"].endswith("+adapter-sha256:" + "f" * 64)


def test_engineering_gate_uses_inclusive_accuracy_delta_boundaries() -> None:
    assert analyzer._minimum_delta_passed(0.05, minimum=0.05)
    assert not analyzer._minimum_delta_passed(0.049999, minimum=0.05)
    assert analyzer._minimum_delta_passed(-0.05, minimum=-0.05)
    assert not analyzer._minimum_delta_passed(-0.050001, minimum=-0.05)
    assert analyzer._minimum_delta_passed(Fraction(30 - 33, 60), minimum=Fraction(-1, 20))
    assert not analyzer._minimum_delta_passed(Fraction(29 - 33, 60), minimum=Fraction(-1, 20))


def test_order_flip_gate_requires_quarter_relative_drop_and_handles_zero_base() -> None:
    reduction, passed = analyzer._order_flip_reduction(0.20, 0.15)
    assert reduction == pytest.approx(0.25)
    assert passed
    reduction, passed = analyzer._order_flip_reduction(0.20, 0.150001)
    assert reduction == pytest.approx(0.249995)
    assert not passed
    assert analyzer._order_flip_reduction(0.0, 0.0) == (0.0, True)
    assert analyzer._order_flip_reduction(0.0, 0.01) == (None, False)
    base = (Fraction(0, 56) + Fraction(4, 60)) / 2
    final = (Fraction(0, 56) + Fraction(3, 60)) / 2
    reduction, passed = analyzer._order_flip_reduction(base, final)
    assert reduction == 0.25
    assert passed
    assert not analyzer._order_flip_reduction(base, Fraction(4, 60) / 2)[1]


def test_engineering_gate_uses_exact_supported_count_boundaries(
    synthetic_run: tuple[
        SplitManifest,
        tuple[DecisionRecord, ...],
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    manifest, records, payload, source_receipt = synthetic_run
    receipt = deepcopy(source_receipt)
    evidence = receipt["evidence"]
    assert isinstance(evidence, dict)
    records_by_id = {record.record_id: record for record in records}
    panel_by_id = {row["presentation_id"]: row for row in payload["evaluation_presentations"]}

    for state in ("base", "final"):
        output_rows = evidence[f"{state}_outputs"]
        assert isinstance(output_rows, list)
        for row in output_rows:
            assert isinstance(row, dict)
            record = records_by_id[row["record_id"]]
            dataset_id = record.dataset_id
            index = int(record.record_id.rsplit("-", 1)[1])
            panel_row = panel_by_id[row["presentation_id"]]
            order_index = int(panel_row["order_index"])
            should_be_wrong = False
            if dataset_id == "sms-pilot-v1-development":
                should_be_wrong = 10 <= index < 37 if state == "base" else index < 30
                if state == "base" and index < 4 and order_index == 1:
                    should_be_wrong = True
                if state == "final" and index < 3 and order_index == 1:
                    should_be_wrong = False
            chosen_id = record.answer_id
            if should_be_wrong:
                chosen_id = next(
                    option.id for option in record.request.options if option.id != chosen_id
                )
            row["winner_option_id"] = chosen_id
            row["candidate_logits"] = [
                4.0 if option_id == chosen_id else 0.0 for option_id in row["order_ids"]
            ]

    parity = evidence["reload_parity"]
    assert isinstance(parity, dict)
    final_rows = evidence["final_outputs"]
    assert isinstance(final_rows, list)
    final_by_id = {row["presentation_id"]: row for row in final_rows}
    parity["reloaded_logits"] = [
        list(final_by_id[presentation_id]["candidate_logits"])
        for presentation_id in parity["presentation_ids"]
    ]

    report = analyzer.analyze_receipt(
        receipt,
        payload,
        records,
        manifest,
        records_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )
    base_order = report["states"]["base"]["order_analysis"]
    final_order = report["states"]["final"]["order_analysis"]
    assert (
        base_order["sms-pilot-v1-development"]["original_order_correct_count"],
        final_order["sms-pilot-v1-development"]["original_order_correct_count"],
    ) == (33, 30)
    assert (
        base_order["dbpedia14-pilot-v1-development"]["semantic_top_answer_flip_record_count"] == 0
    )
    assert base_order["sms-pilot-v1-development"]["semantic_top_answer_flip_record_count"] == 4
    assert (
        final_order["dbpedia14-pilot-v1-development"]["semantic_top_answer_flip_record_count"] == 0
    )
    assert final_order["sms-pilot-v1-development"]["semantic_top_answer_flip_record_count"] == 3

    components = report["engineering_gate"]["components"]
    assert components["sms_accuracy_drop_limit"]["passed"]
    flip_gate = components["dbpedia_sms_macro_order_flip_reduction"]
    assert flip_gate["relative_reduction"] == 0.25
    assert flip_gate["passed"]


def test_temperature_fit_uses_equal_task_weight_and_calibration_labels_only(
    synthetic_run: tuple[
        SplitManifest,
        tuple[DecisionRecord, ...],
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    _manifest, records, payload, receipt = synthetic_run
    outputs = analyzer._output_maps(receipt)["base"]
    _panel_by_id, panel_by_key = analyzer._panel_maps(payload)
    baseline = analyzer._calibration_fit(records, outputs, panel_by_key)

    development_changed = tuple(
        record.model_copy(
            update={
                "answer_id": next(
                    option.id for option in record.request.options if option.id != record.answer_id
                )
            }
        )
        if record.dataset_id.endswith("-development") or record.dataset_id.startswith("snli-")
        else record
        for record in records
    )
    development_fit = analyzer._calibration_fit(development_changed, outputs, panel_by_key)
    assert development_fit == baseline

    calibration_changed = tuple(
        record.model_copy(
            update={
                "answer_id": next(
                    option.id for option in record.request.options if option.id != record.answer_id
                )
            }
        )
        if record.dataset_id.endswith("-calibration")
        else record
        for record in records
    )
    changed_fit = analyzer._calibration_fit(calibration_changed, outputs, panel_by_key)
    assert changed_fit["pool_sha256"] != baseline["pool_sha256"]
    assert changed_fit["temperature"] != baseline["temperature"]
    assert baseline["task_weighting"]["dbpedia14-pilot-v1-calibration"]["total_weight"] == 1.0
    assert baseline["task_weighting"]["sms-pilot-v1-calibration"]["total_weight"] == 1.0
    assert len(baseline["fit_membership"]["dbpedia14-pilot-v1-calibration"]) == 56
    assert len(baseline["fit_membership"]["sms-pilot-v1-calibration"]) == 60


def test_paired_bootstrap_is_deterministic_and_resamples_source_groups(
    synthetic_run: tuple[
        SplitManifest,
        tuple[DecisionRecord, ...],
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    _manifest, records, _payload, _receipt = synthetic_run
    dataset_records = tuple(
        record for record in records if record.dataset_id == "dbpedia14-pilot-v1-development"
    )
    base = {record.record_id: False for record in dataset_records}
    final = {record.record_id: True for record in dataset_records}

    first = analyzer._paired_bootstrap(dataset_records, base, final, Random(20261005))
    second = analyzer._paired_bootstrap(dataset_records, base, final, Random(20261005))

    assert first == second
    assert first["source_group_count"] == 56
    assert first["replicates"] == 2_000
    assert first["percentile_95"] == [1.0, 1.0]


def test_macro_bootstrap_resamples_dbpedia_and_sms_independently(
    synthetic_run: tuple[
        SplitManifest,
        tuple[DecisionRecord, ...],
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    _manifest, records, _payload, _receipt = synthetic_run
    datasets = {
        dataset_id: tuple(record for record in records if record.dataset_id == dataset_id)
        for dataset_id in analyzer._DEVELOPMENT_DATASETS[:2]
    }
    base = {
        dataset_id: {record.record_id: False for record in dataset_records}
        for dataset_id, dataset_records in datasets.items()
    }
    final = {
        dataset_id: {record.record_id: True for record in dataset_records}
        for dataset_id, dataset_records in datasets.items()
    }

    interval = analyzer._macro_bootstrap_interval(datasets, base, final, Random(20261005))

    assert interval["dataset_ids"] == list(analyzer._DEVELOPMENT_DATASETS[:2])
    assert interval["accuracy_delta_final_minus_base"] == 1.0
    assert interval["percentile_95"] == [1.0, 1.0]
    assert "independent source-group resampling" in interval["sampling"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_id", "wrong_pin", "nonfinite"])
def test_passed_receipt_rejects_incomplete_or_malformed_outputs(
    synthetic_run: tuple[
        SplitManifest,
        tuple[DecisionRecord, ...],
        dict[str, object],
        dict[str, object],
    ],
    problem: str,
) -> None:
    manifest, records, payload, receipt = synthetic_run
    malformed = deepcopy(receipt)
    final_outputs = malformed["evidence"]["final_outputs"]
    if problem == "missing":
        final_outputs.pop()
    elif problem == "duplicate":
        malformed["evidence"]["base_outputs"].append(
            deepcopy(malformed["evidence"]["base_outputs"][0])
        )
    elif problem == "wrong_id":
        final_outputs[0]["presentation_id"] = "0" * 64
    elif problem == "wrong_pin":
        malformed["pins"]["records_sha256"] = "9" * 64
    else:
        final_outputs[0]["candidate_logits"][0] = float("nan")

    with pytest.raises(ValueError):
        analyzer.analyze_receipt(
            malformed,
            payload,
            records,
            manifest,
            records_sha256="b" * 64,
            manifest_sha256="c" * 64,
        )


def test_prepared_data_pins_reject_a_copy_at_an_unapproved_recipe_path(
    tmp_path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    recipe = {
        "outputs": {
            "records_path": pilot_data.DEFAULT_RECORDS_PATH,
            "records_sha256": "b" * 64,
            "manifest_path": pilot_data.DEFAULT_MANIFEST_PATH,
            "manifest_sha256": "c" * 64,
        }
    }

    with pytest.raises(ValueError, match="recipe path"):
        analyzer._validate_data_pins(
            project_root / pilot_data.DEFAULT_RECORDS_PATH,
            project_root / pilot_data.DEFAULT_MANIFEST_PATH,
            tmp_path / "copied-recipe.json",
            recipe,
            records_sha256="b" * 64,
            manifest_sha256="c" * 64,
            recipe_sha256="d" * 64,
        )


def test_analyzer_requires_the_receipt_path_from_the_completed_run(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["analyze_real_pilot.py"])
    with pytest.raises(SystemExit) as exit_info:
        analyzer.main()

    assert exit_info.value.code == 2
    assert "the following arguments are required: --receipt" in capsys.readouterr().err
