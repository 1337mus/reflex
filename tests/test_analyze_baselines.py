import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import analyze_baselines
from reflex_decisions import baseline_data
from reflex_decisions.data import DatasetSpec, DecisionRecord, SplitManifest
from reflex_decisions.schema import DecisionRequest, Option


def _fixture() -> tuple[SplitManifest, tuple[DecisionRecord, ...]]:
    manifest = SplitManifest(
        data_kind="fixture",
        datasets=(
            DatasetSpec(
                dataset_id="pilot",
                source_id="synthetic-copa",
                task_family="commonsense-causal-choice",
                split="test",
                source_uri="local://fixture/copa",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )
    records = tuple(
        DecisionRecord(
            record_id=f"r{index}",
            dataset_id="pilot",
            source_group_id=f"item-{index}",
            request=DecisionRequest(
                context=f"Context {index}",
                question="Which alternative is more plausible?",
                options=(Option(id="choice1", label="First"), Option(id="choice2", label="Second")),
            ),
            answer_id=answer,
        )
        for index, answer in ((1, "choice1"), (2, "choice2"))
    )
    return manifest, records


def _receipt(records: tuple[DecisionRecord, ...]) -> dict[str, object]:
    logits = {"r1": ([2.0, 0.0], [0.0, 2.0]), "r2": ([2.0, 0.0], [2.0, 0.0])}
    presentations = []
    for record in records:
        for permutation_index in (0, 1):
            option_ids = [option.id for option in record.request.options]
            if permutation_index:
                option_ids.reverse()
            presentations.append(
                {
                    "record_id": record.record_id,
                    "request_hash": record.request.request_hash,
                    "permutation_index": permutation_index,
                    "option_ids": option_ids,
                    "raw_logits": logits[record.record_id][permutation_index],
                    "input_tokens": 5,
                    "prompt_sha256": hashlib.sha256(
                        f"{record.record_id}:{permutation_index}".encode()
                    ).hexdigest(),
                }
            )
    return {
        "status": "passed",
        "provenance": {
            "model_id": "test/model",
            "model_revision": "test-revision",
            "temperature": 2.0,
        },
        "presentations": presentations,
    }


def _write_analysis_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[Path, Path, SplitManifest, tuple[DecisionRecord, ...], str, str]:
    manifest, records = _fixture()
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records_path.write_bytes(baseline_data.serialize_records(records))
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    records_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if monkeypatch is not None:

        def verify_fixture(records_arg: str | Path, manifest_arg: str | Path):
            assert Path(records_arg) == records_path
            assert Path(manifest_arg) == manifest_path
            return manifest, records

        monkeypatch.setattr(analyze_baselines, "verify_prepared_data", verify_fixture)
    return records_path, manifest_path, manifest, records, records_sha256, manifest_sha256


def test_analysis_reports_original_accuracy_order_sensitivity_and_wilson_interval() -> None:
    manifest, records = _fixture()
    result = analyze_baselines.analyze_model(
        "kev",
        _receipt(records),
        records,
        manifest,
        records_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )

    assert result["raw_report"]["overall"]["record_count"] == 2
    assert result["raw_report"]["overall"]["micro_accuracy"] == 0.5
    assert result["shipped_report"]["temperature"] == 2.0
    assert result["shipped_report"]["calibration_id"] == "model-native-temp"
    interval = result["accuracy_interval"]
    assert interval["correct"] == 1
    assert interval["record_count"] == 2
    assert interval["lower_95"] == pytest.approx(0.094531, abs=1e-6)
    assert interval["upper_95"] == pytest.approx(0.905469, abs=1e-6)
    assert result["order_sensitivity"]["forced_choice_flip_count"] == 1
    assert result["order_sensitivity"]["forced_choice_flip_fraction"] == 0.5
    assert result["order_sensitivity"]["mean_js_divergence_nats"] == pytest.approx(0.055472)
    assert result["order_sensitivity"]["original_order_accuracy"] == 0.5
    assert result["order_sensitivity"]["reversed_order_accuracy"] == 1.0


def test_qwen_shipped_report_does_not_claim_a_calibration_profile() -> None:
    manifest, records = _fixture()
    receipt = _receipt(records)
    receipt["provenance"]["temperature"] = 1.0

    result = analyze_baselines.analyze_model(
        "qwen",
        receipt,
        records,
        manifest,
        records_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )

    assert result["shipped_report"]["temperature"] == 1.0
    assert result["shipped_report"]["calibration_id"] is None


def test_analysis_rejects_a_presentation_with_the_wrong_request_hash() -> None:
    manifest, records = _fixture()
    receipt = _receipt(records)
    receipt["presentations"][0]["request_hash"] = "0" * 64

    with pytest.raises(ValueError, match="request_hash"):
        analyze_baselines.analyze_model(
            "kev",
            receipt,
            records,
            manifest,
            records_sha256="0" * 64,
            manifest_sha256="1" * 64,
        )


def test_analysis_rejects_a_wrong_reversed_option_order() -> None:
    manifest, records = _fixture()
    receipt = _receipt(records)
    receipt["presentations"][1]["option_ids"] = ["choice1", "choice2"]

    with pytest.raises(ValueError, match="approved permutation"):
        analyze_baselines.analyze_model(
            "kev",
            receipt,
            records,
            manifest,
            records_sha256="0" * 64,
            manifest_sha256="1" * 64,
        )


def test_analysis_rejects_duplicate_model_presentations() -> None:
    manifest, records = _fixture()
    receipt = _receipt(records)
    receipt["presentations"][3] = receipt["presentations"][0]

    with pytest.raises(ValueError, match="duplicate presentation"):
        analyze_baselines.analyze_model(
            "kev",
            receipt,
            records,
            manifest,
            records_sha256="0" * 64,
            manifest_sha256="1" * 64,
        )


def test_analysis_rejects_a_missing_model_presentation() -> None:
    manifest, records = _fixture()
    receipt = _receipt(records)
    receipt["presentations"].pop()

    with pytest.raises(ValueError, match="two presentations per source record"):
        analyze_baselines.analyze_model(
            "kev",
            receipt,
            records,
            manifest,
            records_sha256="0" * 64,
            manifest_sha256="1" * 64,
        )


def test_production_analysis_rejects_a_mislabeled_frozen_model_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, manifest_path, _, _, records_sha256, manifest_sha256 = _write_analysis_inputs(
        tmp_path, monkeypatch
    )
    run = {
        "status": "passed",
        "records_sha256": records_sha256,
        "manifest_sha256": manifest_sha256,
        "models": {
            name: {
                "status": "passed",
                "provenance": {
                    "model_id": "incorrect/model" if name == "qwen" else name,
                    "model_revision": "incorrect-revision",
                    "temperature": 1.0,
                },
                "presentations": [],
            }
            for name in analyze_baselines.MODELS
        },
    }
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen pin"):
        analyze_baselines.analyze_run(
            run_path,
            records_path,
            manifest_path,
        )


def test_production_analysis_accepts_exact_complete_synthetic_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path, manifest_path, _, records, records_sha256, manifest_sha256 = (
        _write_analysis_inputs(tmp_path, monkeypatch)
    )
    models = {}
    for name in analyze_baselines.MODELS:
        model_id, revision = analyze_baselines.MODEL_PINS[name]
        receipt = _receipt(records)
        receipt["provenance"] = {
            "model_id": model_id,
            "model_revision": revision,
            "temperature": 1.0 if name == "qwen" else 2.0,
        }
        models[name] = receipt
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "records_sha256": records_sha256,
                "manifest_sha256": manifest_sha256,
                "models": models,
            }
        ),
        encoding="utf-8",
    )

    result = analyze_baselines.analyze_run(run_path, records_path, manifest_path)

    assert result["record_count"] == 2
    assert set(result["per_model"]) == set(analyze_baselines.MODELS)
    assert all(
        value["shipped_report"]["overall"]["record_count"] == 2
        for value in result["per_model"].values()
    )
    assert len(result["paired_comparisons"]) == 3


def test_production_analysis_rejects_failed_or_partial_runs(tmp_path: Path) -> None:
    run_path = tmp_path / "failed-run.json"
    run_path.write_text(json.dumps({"status": "failed", "models": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="passed status"):
        analyze_baselines.analyze_run(
            run_path,
            tmp_path / "records-not-needed.jsonl",
            tmp_path / "manifest-not-needed.json",
        )


def test_production_analysis_rejects_runner_source_hash_mismatch(tmp_path: Path) -> None:
    records_path, manifest_path, _, _, _, _ = _write_analysis_inputs(tmp_path)
    run_path = tmp_path / "wrong-source-hash.json"
    run_path.write_text(
        json.dumps({"status": "passed", "records_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="records SHA-256"):
        analyze_baselines.analyze_run(
            run_path,
            records_path,
            manifest_path,
        )
