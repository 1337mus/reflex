"""Independent transfer-result analysis and its paired source-group bootstrap."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from experiments import adapter_transfer_core as core
from experiments import adapter_transfer_data as data
from experiments import analyze_adapter_transfer as analyzer
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("copa-dev-pilot-v1", "boolq-dev-pilot-v1")


def _records() -> tuple[DecisionRecord, ...]:
    options = (Option(id="a", label="A"), Option(id="b", label="B"))
    records = []
    for dataset_id in DATASETS:
        for index in range(32):
            source = f"{dataset_id}-group-{index:02d}"
            records.append(
                DecisionRecord(
                    record_id=f"{dataset_id}-record-{index:02d}",
                    dataset_id=dataset_id,
                    source_group_id=source,
                    request=DecisionRequest(
                        context=f"Context {dataset_id} {index}.",
                        question="Which option is correct?",
                        options=options,
                    ),
                    answer_id="a",
                )
            )
    return tuple(records)


def _outputs(presentations, records, *, model: str) -> list[dict[str, object]]:
    by_id = {record.record_id: record for record in records}
    rows = []
    for presentation in presentations:
        record = by_id[presentation["record_id"]]
        index = int(record.record_id[-2:])
        order = presentation["order_index"]
        correct_limit = {"base": (16, 12), "adapter": (20, 16)}[model][order]
        winner = "a" if index < correct_limit else "b"
        ids = presentation["order_ids"]
        rows.append(
            {
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
            | {
                "candidate_logits": [1.0 if option_id == winner else 0.0 for option_id in ids],
                "winner_option_id": winner,
                "input_tokens": 10,
                "prompt_sha256": hashlib.sha256(
                    presentation["presentation_id"].encode()
                ).hexdigest(),
            }
        )
    return rows


def _fixture(monkeypatch, *, status: str = "passed"):
    records = _records()
    presentations = data.build_presentations(records)
    selection = {"selection_id": "test-selection"}
    pins = {"source_file_sha256": {"tests/fixture": "a" * 64}}
    expected_payload = {
        "schema_version": 1,
        "experiment_id": "adapter-transfer-v1",
        "run_id": "adapter-transfer-2026-10-05-r1",
        "nonce": "e8f9ba22-4eef-4bf9-8b15-9daedcfab6bd",
        "selection_sha256": "b" * 64,
        "payload_sha256": "c" * 64,
        "selection": selection,
        "pins": pins,
        "presentations": presentations,
        "parity_presentations": data.parity_presentations(presentations),
    }
    calls: list[tuple[object, object]] = []

    def build_payload(run_id, nonce, local_presentations, local_selection, local_pins, *, root):
        assert run_id == expected_payload["run_id"]
        assert nonce == expected_payload["nonce"]
        assert local_presentations == presentations
        assert local_selection == selection
        assert local_pins == pins
        calls.append(("build", root))
        return expected_payload

    def validate_payload(saved, *, root):
        calls.append(("payload", root))
        return saved

    def load_inputs(root):
        calls.append(("load", root))
        return records, presentations, selection, pins

    output_rows = {
        "base": _outputs(presentations, records, model="base"),
        "adapter": _outputs(presentations, records, model="adapter"),
    }
    result = {
        "status": status,
        "phase": "completed" if status == "passed" else "adapter_evaluation_failed",
        "run_id": expected_payload["run_id"],
        "nonce": expected_payload["nonce"],
        "selection_sha256": expected_payload["selection_sha256"],
        "payload_sha256": expected_payload["payload_sha256"],
        "evidence": {
            "outputs": output_rows,
            "reload_parity": {"outputs": [], "max_candidate_logit_delta": 0.0},
            "forward_counts": {"base_evaluation": 128, "final_evaluation": 128},
        },
        "failure": None if status == "passed" else {"stage": "adapter_evaluation"},
    }
    receipt = {
        "schema_version": 1,
        "run_id": expected_payload["run_id"],
        "status": status,
        "phase": "complete" if status == "passed" else "adapter_evaluation_failed",
        "payload": deepcopy(expected_payload),
        "result": result,
        "failure": None if status == "passed" else {"stage": "adapter_evaluation"},
        "modal": {"profile": "reflex-personal"},
    }

    def validate_result(raw_result, payload, *, root):
        assert payload == expected_payload
        calls.append(("validate", root))
        return raw_result

    monkeypatch.setattr(data, "load_inputs", load_inputs)
    monkeypatch.setattr(core, "build_payload", build_payload)
    monkeypatch.setattr(core, "validate_payload", validate_payload)
    monkeypatch.setattr(core, "validate_result", validate_result)
    return receipt, records, presentations, expected_payload, calls


def _type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def test_reports_per_task_counts_group_means_and_paired_bootstrap(monkeypatch) -> None:
    receipt, _records_value, _presentations, _payload, calls = _fixture(monkeypatch)

    report = analyzer.analyze_receipt(receipt, root=ROOT)

    assert report["status"] == "analyzed"
    assert list(report["tasks"]) == sorted(DATASETS)
    assert report["run"]["selection"] == _payload["selection"]
    assert [event[0] for event in calls] == ["payload", "load", "build", "validate"]
    for task_id in DATASETS:
        task = report["tasks"][task_id]
        assert task["group_count"] == 32
        base = task["models"]["qwen_base"]
        adapter = task["models"]["adapter"]
        assert base["original_order"] == {
            "correct": 16,
            "total": 32,
            "exact_mean": {"numerator": 1, "denominator": 2},
            "accuracy": 0.5,
            "accuracy_percent": 50.0,
        }
        assert base["both_orders"]["correct"] == 28
        assert base["both_orders"]["total"] == 64
        assert base["both_orders"]["accuracy_percent"] == 43.75
        assert base["semantic_answer_flips"] == {"count": 4, "total": 32}
        assert adapter["original_order"]["correct"] == 20
        assert adapter["original_order"]["accuracy_percent"] == 62.5
        assert adapter["both_orders"]["correct"] == 36
        assert adapter["both_orders"]["accuracy_percent"] == 56.25
        assert adapter["semantic_answer_flips"] == {"count": 4, "total": 32}
        paired = task["paired_adapter_minus_base"]
        assert paired["original_order"]["exact_delta"] == {
            "numerator": 1,
            "denominator": 8,
        }
        assert paired["both_orders"]["exact_delta"] == {
            "numerator": 1,
            "denominator": 8,
        }
        assert paired["original_order"]["estimate"] == 1 / 8
        assert paired["both_orders"]["estimate"] == 1 / 8
        assert task["group_means"][0]["source_group_id"].endswith("00")
        assert task["group_means"][12]["qwen_base"]["original_order"] == {
            "numerator": 1,
            "denominator": 1,
        }
        assert report["bootstrap"] == {
            "unit": "source_group",
            "replicates": 2000,
            "seed": 20261009,
            "interval": "95% type-7 percentile",
        }

    rng = random.Random(20261009)
    for task_id in sorted(DATASETS):
        task = report["tasks"][task_id]
        paired = task["paired_adapter_minus_base"]
        deltas = [0.0] * 16 + [1.0] * 4 + [0.0] * 12
        original_draws, both_draws = [], []
        both_group_deltas = [0.0] * 12 + [0.5] * 8 + [0.0] * 12
        for _ in range(2000):
            indices = [rng.randrange(32) for _ in range(32)]
            original_draws.append(sum(deltas[index] for index in indices) / 32)
            both_draws.append(sum(both_group_deltas[index] for index in indices) / 32)
        assert paired["original_order"]["bootstrap_95"] == {
            "lower": _type7(original_draws, 0.025),
            "upper": _type7(original_draws, 0.975),
        }
        assert paired["both_orders"]["bootstrap_95"] == {
            "lower": _type7(both_draws, 0.025),
            "upper": _type7(both_draws, 0.975),
        }


def test_payload_must_match_local_selection_data_and_source_pins_before_result_validation(
    monkeypatch,
) -> None:
    receipt, _records_value, _presentations, _payload, calls = _fixture(monkeypatch)
    receipt["payload"]["pins"]["source_file_sha256"]["tests/fixture"] = "d" * 64

    with pytest.raises(ValueError, match="local payload"):
        analyzer.analyze_receipt(receipt, root=ROOT)

    assert [event[0] for event in calls] == ["payload", "load", "build"]


@pytest.mark.parametrize("damage", ["missing_adapter", "missing_row"])
def test_analyzer_rejects_missing_model_state_or_presentation(monkeypatch, damage) -> None:
    receipt, _records_value, _presentations, _payload, _calls = _fixture(monkeypatch)
    if damage == "missing_adapter":
        del receipt["result"]["evidence"]["outputs"]["adapter"]
    else:
        receipt["result"]["evidence"]["outputs"]["base"].pop()

    with pytest.raises(ValueError, match="model output|presentation rows"):
        analyzer.analyze_receipt(receipt, root=ROOT)


def test_failed_execution_is_preserved_without_learned_metrics(monkeypatch) -> None:
    receipt, _records_value, _presentations, _payload, _calls = _fixture(
        monkeypatch, status="failed"
    )
    receipt["result"]["evidence"]["outputs"]["adapter"] = []

    report = analyzer.analyze_receipt(receipt, root=ROOT)

    assert report["status"] == "failed"
    assert "tasks" not in report
    assert report["run"]["failure"] == {"stage": "adapter_evaluation"}
    assert report["run"]["partial_evidence"]["outputs"]["base"]


def test_failed_preflight_without_payload_is_preserved_without_scoring() -> None:
    receipt = {
        "schema_version": 1,
        "run_id": "adapter-transfer-2026-10-05-r1",
        "status": "failed",
        "phase": "host_preflight",
        "payload": None,
        "result": None,
        "failure": {"stage": "source_preflight", "type": "ValueError"},
        "modal": {"profile": "reflex-personal"},
    }

    report = analyzer.analyze_receipt(receipt, root=ROOT)

    assert report["status"] == "failed"
    assert "tasks" not in report
    assert report["run"]["failure"] == receipt["failure"]
    assert report["run"]["partial_evidence"] is None


def test_cli_refuses_to_overwrite_analysis_output(tmp_path, monkeypatch) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "analysis.json"
    output.write_text("keep me\n", encoding="utf-8")
    monkeypatch.setattr(
        analyzer, "analyze_receipt", lambda *_args, **_kwargs: pytest.fail("analysis started")
    )

    status = analyzer.main(
        ["--receipt", str(receipt_path), "--repo-root", str(tmp_path), "--output", str(output)]
    )

    assert status != 0
    assert output.read_text(encoding="utf-8") == "keep me\n"


def test_cli_persists_failed_analysis_to_a_new_output(tmp_path, monkeypatch) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text('{"run_id":"saved-run"}', encoding="utf-8")
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        analyzer,
        "analyze_receipt",
        lambda receipt, *, root: {"status": "failed", "run_id": receipt["run_id"]},
    )

    status = analyzer.main(
        ["--receipt", str(receipt_path), "--repo-root", str(tmp_path), "--output", str(output)]
    )

    assert status == 1
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "failed",
        "run_id": "saved-run",
    }


def test_cli_rejects_duplicate_receipt_keys_before_analysis(tmp_path, monkeypatch) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text('{"run_id":"first","run_id":"second"}', encoding="utf-8")
    output = tmp_path / "analysis.json"
    monkeypatch.setattr(
        analyzer, "analyze_receipt", lambda *_args, **_kwargs: pytest.fail("analysis started")
    )

    status = analyzer.main(
        ["--receipt", str(receipt_path), "--repo-root", str(tmp_path), "--output", str(output)]
    )

    assert status == 2
    assert not output.exists()


def test_import_does_not_load_gpu_or_modal_packages() -> None:
    script = """
import sys
from experiments import analyze_adapter_transfer
forbidden = ("torch", "transformers", "peft", "modal")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_script_entrypoint_runs_argument_parser() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.analyze_adapter_transfer", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--receipt" in completed.stdout
