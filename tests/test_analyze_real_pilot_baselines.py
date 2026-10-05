from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

from experiments import analyze_real_pilot as pilot_analysis
from experiments import analyze_real_pilot_baselines as analyzer
from experiments import real_pilot_baseline_core as baseline_core
from experiments import real_pilot_core as pilot_core
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option

DBPEDIA_CALIBRATION = "dbpedia14-pilot-v1-calibration"
SMS_CALIBRATION = "sms-pilot-v1-calibration"
DBPEDIA_DEVELOPMENT = "dbpedia14-pilot-v1-development"
SMS_DEVELOPMENT = "sms-pilot-v1-development"


def _record(dataset_id: str, record_id: str, option_count: int) -> DecisionRecord:
    options = tuple(
        Option(id=f"option-{index:02}", label=f"Option {index}") for index in range(option_count)
    )
    return DecisionRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        source_group_id=f"group-{record_id}",
        request=DecisionRequest(
            context=f"Context for {record_id}",
            question="Which option applies?",
            options=options,
        ),
        answer_id=options[0].id,
    )


def _calibration_fixture() -> tuple[
    tuple[DecisionRecord, ...],
    dict[str, dict[str, object]],
    dict[tuple[str, int], dict[str, object]],
]:
    records: list[DecisionRecord] = []
    outputs: dict[str, dict[str, object]] = {}
    panel: dict[tuple[str, int], dict[str, object]] = {}
    for dataset_id, count, option_count in (
        (DBPEDIA_CALIBRATION, 56, 14),
        (SMS_CALIBRATION, 60, 2),
    ):
        for index in range(count):
            record = _record(dataset_id, f"{dataset_id}-{index:03}", option_count)
            records.append(record)
            presentation_id = f"presentation-{record.record_id}"
            option_ids = [option.id for option in record.request.options]
            panel[(record.record_id, 0)] = {
                "presentation_id": presentation_id,
                "record_id": record.record_id,
                "dataset_id": dataset_id,
                "order_index": 0,
                "order_ids": option_ids,
            }
            outputs[presentation_id] = {
                "presentation_id": presentation_id,
                "candidate_logits": [
                    3.0 if option_index == index % option_count else -0.5
                    for option_index in range(option_count)
                ],
            }
    records.extend(
        (
            _record(DBPEDIA_DEVELOPMENT, "dbpedia-development-row", 14),
            _record(SMS_DEVELOPMENT, "sms-development-row", 2),
        )
    )
    return tuple(records), outputs, panel


def _reference_state(
    status: str, *, partial: list[dict[str, object]] | None = None
) -> dict[str, object]:
    state: dict[str, object] = {
        "status": status,
        "model_id": f"{status}-model",
        "model_revision": "revision",
        "forward_counts": {"scored": 1, "auxiliary": 0, "total": 1},
        "prompt_hash_kind": "test-hash",
        "provenance": {"test": True},
        "presentations": partial or [],
    }
    if status == "failed":
        state["failure"] = {"stage": "scoring", "type": "RuntimeError", "message": "stopped"}
    return state


def _reference_receipt(
    payload: dict[str, object],
    *,
    intern_status: str = "failed",
    kev_status: str = "failed",
) -> dict[str, object]:
    statuses = {intern_status, kev_status}
    envelope_status = (
        "passed" if statuses == {"passed"} else "partial" if "passed" in statuses else "failed"
    )
    return {
        "schema_version": baseline_core.SCHEMA_VERSION,
        "status": envelope_status,
        "payload": deepcopy(payload),
        "limits": deepcopy(baseline_core.MODAL_LIMITS),
        "provenance": {
            "modal_sdk_version": "1.0.0",
            "profile": "reflex-personal",
            "workspace": "rajath-61258",
            "hf_hub_disable_implicit_token": True,
        },
        "models": {
            "intern": _reference_state(intern_status, partial=[{"partial": 1}]),
            "kev": _reference_state(kev_status),
        },
    }


def test_calibration_fit_uses_only_local_calibration_labels() -> None:
    records, outputs, panel_by_key = _calibration_fixture()
    baseline = pilot_analysis._calibration_fit(records, outputs, panel_by_key)

    development_changed = tuple(
        record.model_copy(update={"answer_id": record.request.options[-1].id})
        if record.dataset_id in {DBPEDIA_DEVELOPMENT, SMS_DEVELOPMENT}
        else record
        for record in records
    )
    assert pilot_analysis._calibration_fit(development_changed, outputs, panel_by_key) == baseline

    calibration_changed = tuple(
        record.model_copy(update={"answer_id": record.request.options[-1].id})
        if record.dataset_id in {DBPEDIA_CALIBRATION, SMS_CALIBRATION}
        else record
        for record in records
    )
    changed = pilot_analysis._calibration_fit(calibration_changed, outputs, panel_by_key)
    assert changed["pool_sha256"] != baseline["pool_sha256"]
    assert baseline["example_count"] == 116
    assert baseline["task_weighting"][DBPEDIA_CALIBRATION]["total_weight"] == 1.0
    assert baseline["task_weighting"][SMS_CALIBRATION]["total_weight"] == 1.0


def test_reference_analysis_maps_fourteen_candidate_logits_by_semantic_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = DBPEDIA_DEVELOPMENT
    record = _record(dataset_id, "dbpedia-fourteen-row", 14)
    option_ids = [option.id for option in record.request.options]
    target = record.answer_id
    outputs: dict[str, dict[str, object]] = {}
    panel_by_id: dict[str, dict[str, object]] = {}
    panel_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for order_index in range(14):
        ordered_ids = option_ids[order_index:] + option_ids[:order_index]
        presentation_id = f"presentation-{order_index}"
        request = record.request.model_copy(
            update={
                "options": tuple(
                    next(option for option in record.request.options if option.id == option_id)
                    for option_id in ordered_ids
                )
            }
        )
        presentation = {
            "presentation_id": presentation_id,
            "record_id": record.record_id,
            "dataset_id": dataset_id,
            "order_index": order_index,
            "order_ids": ordered_ids,
            "request": request.model_dump(mode="json"),
        }
        panel_by_id[presentation_id] = presentation
        panel_by_key[(record.record_id, order_index)] = presentation
        outputs[presentation_id] = {
            "presentation_id": presentation_id,
            "candidate_logits": [
                10.0 if option_id == target else -1.0 for option_id in ordered_ids
            ],
        }

    monkeypatch.setattr(pilot_analysis, "_DEVELOPMENT_DATASETS", (dataset_id,))
    monkeypatch.setattr(pilot_analysis, "_calibration_fit", lambda *_args: {"temperature": 2.0})
    monkeypatch.setattr(pilot_analysis, "_prediction_rows", lambda *_args: ())
    temperatures: list[float] = []

    def evaluate(*_args: object, temperature: float, **_kwargs: object) -> dict[str, float]:
        temperatures.append(temperature)
        return {"temperature": temperature}

    monkeypatch.setattr(pilot_analysis, "_evaluate_dataset", evaluate)
    monkeypatch.setattr(
        pilot_analysis, "_macro_metrics", lambda metrics: {"datasets": len(metrics)}
    )
    state = {
        "status": "passed",
        "model_id": "intern",
        "model_revision": "intern-revision",
        "forward_counts": {"scored": 2572, "auxiliary": 1, "total": 2573},
        "prompt_hash_kind": "rendered_chat_utf8_sha256",
        "provenance": {},
        "presentations": list(outputs.values()),
    }

    report, correct = analyzer._analyze_reference_state(
        state,
        (record,),
        object(),  # type: ignore[arg-type]
        panel_by_id,
        panel_by_key,
        records_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )

    order_result = report["order_analysis"][dataset_id]
    assert order_result["order_count"] == 14
    assert order_result["original_order_correct_count"] == 1
    assert order_result["semantic_top_answer_flip_record_count"] == 0
    assert correct[dataset_id] == {record.record_id: True}
    assert temperatures == [1.0, 2.0]


def test_reference_envelope_rejects_pin_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    pins = {"records_sha256": "a" * 64}
    expected = {"run_id": "run", "nonce": "nonce", "pins": pins}
    receipt = _reference_receipt(expected)
    receipt["payload"]["pins"] = {"records_sha256": "b" * 64}
    monkeypatch.setattr(baseline_core, "validate_persisted_model_state", lambda value, **_kw: value)

    with pytest.raises(ValueError, match="identity, pins, or limits"):
        analyzer._validate_reference_envelope(receipt, expected)


def test_analyzer_rejects_a_reference_panel_that_differs_from_qwen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, nonce = "run", "nonce"
    pins = {"records_sha256": "a" * 64}
    qwen_panel = [{"presentation_id": "one"}]
    reference_panel = [{"presentation_id": "two"}]
    qwen_payload = {
        "run_id": run_id,
        "nonce": nonce,
        "pins": pins,
        "evaluation_presentations": qwen_panel,
    }
    reference_payload = {
        "run_id": run_id,
        "nonce": nonce,
        "pins": pins,
        "evaluation_presentations": reference_panel,
    }
    qwen_receipt = {"run_id": run_id, "nonce": nonce, "pins": pins}
    ref_receipt = _reference_receipt(reference_payload)
    monkeypatch.setattr(analyzer, "_build_qwen_payload", lambda *_args: qwen_payload)
    monkeypatch.setattr(analyzer, "_build_reference_payload", lambda *_args: reference_payload)
    monkeypatch.setattr(pilot_core, "validate_passed_receipt", lambda value, **_kw: value)
    monkeypatch.setattr(baseline_core, "validate_persisted_model_state", lambda value, **_kw: value)
    monkeypatch.setattr(
        pilot_analysis,
        "analyze_receipt",
        lambda *_args, **_kwargs: pytest.fail("panel mismatch must be rejected before analysis"),
    )

    with pytest.raises(ValueError, match="exact frozen panel"):
        analyzer.analyze_receipts(
            qwen_receipt,
            ref_receipt,
            (),
            object(),  # type: ignore[arg-type]
            hashes={},
        )


def test_failed_reference_state_is_retained_without_erasing_passing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, nonce = "run", "nonce"
    pins = {"records_sha256": "a" * 64}
    panel = [
        {
            "presentation_id": "one",
            "record_id": "record-one",
            "dataset_id": DBPEDIA_DEVELOPMENT,
            "order_index": 0,
        }
    ]
    payload = {
        "run_id": run_id,
        "nonce": nonce,
        "pins": pins,
        "evaluation_presentations": panel,
    }
    qwen_receipt = {"run_id": run_id, "nonce": nonce, "pins": pins}
    ref_receipt = _reference_receipt(payload, intern_status="failed", kev_status="passed")
    monkeypatch.setattr(analyzer, "_build_qwen_payload", lambda *_args: payload)
    monkeypatch.setattr(analyzer, "_build_reference_payload", lambda *_args: payload)
    monkeypatch.setattr(pilot_core, "validate_passed_receipt", lambda value, **_kw: value)
    monkeypatch.setattr(baseline_core, "validate_persisted_model_state", lambda value, **_kw: value)
    monkeypatch.setattr(
        pilot_analysis,
        "_panel_maps",
        lambda _payload: ({"one": panel[0]}, {("record-one", 0): panel[0]}),
    )
    monkeypatch.setattr(
        pilot_analysis,
        "analyze_receipt",
        lambda *_args, **_kwargs: {"model": {"revision": "qwen"}, "states": {"final": {}}},
    )
    monkeypatch.setattr(analyzer, "_qwen_correctness", lambda *_args: {})
    monkeypatch.setattr(
        analyzer,
        "_analyze_reference_state",
        lambda state, *_args, **_kwargs: ({"model_revision": state["model_revision"]}, {}),
    )
    monkeypatch.setattr(analyzer, "_paired_comparison", lambda *_args: {"paired": True})

    result = analyzer.analyze_receipts(
        qwen_receipt,
        ref_receipt,
        (),
        object(),  # type: ignore[arg-type]
        hashes={"records_sha256": "a" * 64, "manifest_sha256": "b" * 64},
    )

    assert result["models"]["intern"]["status"] == "failed"
    assert result["models"]["intern"]["failure"]["type"] == "RuntimeError"
    assert result["models"]["intern"]["partial_presentations"] == [{"partial": 1}]
    assert result["models"]["kev"] == {
        "status": "passed",
        "state": {"model_revision": "revision"},
    }
    assert set(result["paired_comparisons"]) == {"kev"}


def test_passed_models_remain_analyzable_under_failed_lifecycle_and_statuses_are_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, nonce = "run", "nonce"
    pins = {"records_sha256": "a" * 64}
    panel = [
        {
            "presentation_id": "one",
            "record_id": "record-one",
            "dataset_id": DBPEDIA_DEVELOPMENT,
            "order_index": 0,
        }
    ]
    payload = {
        "run_id": run_id,
        "nonce": nonce,
        "pins": pins,
        "evaluation_presentations": panel,
    }
    qwen_receipt = {"run_id": run_id, "nonce": nonce, "pins": pins}
    ref_receipt = _reference_receipt(payload, intern_status="passed", kev_status="passed")
    ref_receipt["status"] = "failed"
    lifecycle_failure = {"stage": "worker_execution", "type": "RuntimeError", "message": "teardown"}
    ref_receipt["failure"] = lifecycle_failure
    monkeypatch.setattr(analyzer, "_build_qwen_payload", lambda *_args: payload)
    monkeypatch.setattr(analyzer, "_build_reference_payload", lambda *_args: payload)
    monkeypatch.setattr(pilot_core, "validate_passed_receipt", lambda value, **_kw: value)
    monkeypatch.setattr(baseline_core, "validate_persisted_model_state", lambda value, **_kw: value)
    monkeypatch.setattr(
        pilot_analysis,
        "_panel_maps",
        lambda _payload: ({"one": panel[0]}, {("record-one", 0): panel[0]}),
    )
    monkeypatch.setattr(
        pilot_analysis,
        "analyze_receipt",
        lambda *_args, **_kwargs: {"model": {"revision": "qwen"}, "states": {"final": {}}},
    )
    monkeypatch.setattr(analyzer, "_qwen_correctness", lambda *_args: {})
    monkeypatch.setattr(
        analyzer,
        "_analyze_reference_state",
        lambda state, *_args, **_kwargs: (
            {"model_revision": state["model_revision"], "metrics": {"accuracy": 1.0}},
            {},
        ),
    )
    monkeypatch.setattr(analyzer, "_paired_comparison", lambda *_args: {"paired": True})

    result = analyzer.analyze_receipts(
        qwen_receipt,
        ref_receipt,
        (),
        object(),  # type: ignore[arg-type]
        hashes={"records_sha256": "a" * 64, "manifest_sha256": "b" * 64},
    )

    assert result["reference_run_failure"] == lifecycle_failure
    for model_name in ("intern", "kev"):
        assert result["models"][model_name] == {
            "status": "passed",
            "state": {"model_revision": "revision", "metrics": {"accuracy": 1.0}},
        }
    assert set(result["paired_comparisons"]) == {"intern", "kev"}

    with pytest.raises(ValueError, match="status"):
        analyzer._validate_reference_envelope({**ref_receipt, "status": "passed"}, payload)
    without_failure = deepcopy(ref_receipt)
    without_failure.pop("failure")
    with pytest.raises(ValueError, match="status"):
        analyzer._validate_reference_envelope(without_failure, payload)
    with pytest.raises(ValueError, match="status"):
        analyzer._validate_reference_envelope({**ref_receipt, "status": "partial"}, payload)


def test_cli_does_not_overwrite_an_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "existing-report.json"
    destination.write_text("keep this report\n", encoding="utf-8")
    monkeypatch.setattr(analyzer, "analyze_run", lambda *_args: {"new": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_real_pilot_baselines.py",
            "--qwen-receipt",
            str(tmp_path / "qwen.json"),
            "--reference-receipt",
            str(tmp_path / "reference.json"),
            "--output",
            str(destination),
        ],
    )

    with pytest.raises(SystemExit) as error:
        analyzer.main()

    assert error.value.code == 2
    assert destination.read_text(encoding="utf-8") == "keep this report\n"
