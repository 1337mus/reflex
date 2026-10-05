from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experiments import analyze_fresh_eval as analysis
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _toy_task() -> tuple[list[dict[str, object]], dict[str, object], list[list[str]]]:
    records = [
        {
            "record_id": "q1",
            "dataset_id": "toy",
            "source_group_id": "g-a",
            "answer_id": "A",
            "option_ids": ["A", "B"],
        },
        {
            "record_id": "q2",
            "dataset_id": "toy",
            "source_group_id": "g-b",
            "answer_id": "A",
            "option_ids": ["A", "B"],
        },
        {
            "record_id": "q3",
            "dataset_id": "toy",
            "source_group_id": "g-b",
            "answer_id": "A",
            "option_ids": ["A", "B", "C"],
        },
    ]
    predictions = {
        "base": {
            "q1": {0: "A", 1: "A"},
            "q2": {0: "B", 1: "B"},
            "q3": {0: "B", 1: "C"},
        },
        "adapter": {
            "q1": {0: "B", 1: "B"},
            "q2": {0: "A", 1: "A"},
            "q3": {0: "A", 1: "B"},
        },
    }
    draws = [["g-a", "g-a"], ["g-a", "g-b"], ["g-b", "g-b"]]
    return records, predictions, draws


def test_task_metrics_use_semantic_winners_and_equal_group_means() -> None:
    records, predictions, draws = _toy_task()

    report = analysis._task_metrics("toy", records, predictions, bootstrap_draws=draws)

    assert report["models"]["base"]["original_order_accuracy"] == {
        "correct": 1,
        "total": 3,
        "accuracy": 1 / 3,
        "source_group_mean_accuracy": {"numerator": 1, "denominator": 2},
    }
    assert report["models"]["adapter"]["original_order_accuracy"]["accuracy"] == 2 / 3
    assert report["models"]["base"]["semantic_winner_changes"] == {"count": 1, "total": 3}
    assert report["models"]["adapter"]["semantic_winner_changes"] == {"count": 1, "total": 3}
    original = report["paired_adapter_minus_base"]["original_order"]
    both = report["paired_adapter_minus_base"]["both_orders"]
    assert original["group_mean_difference"] == {"numerator": 0, "denominator": 1}
    assert original["bootstrap_95"] == {"lower": -0.95, "upper": 0.95}
    assert both["group_mean_difference"] == {"numerator": -1, "denominator": 8}
    assert both["bootstrap_95"] == {"lower": -0.95625, "upper": 0.70625}


def test_record_digest_matches_canonical_decision_record_json() -> None:
    record = DecisionRecord(
        record_id="q1",
        dataset_id="toy",
        source_group_id="group-1",
        request=DecisionRequest(
            context="fixture context",
            question="Which option?",
            options=(Option(id="A", label="first"), Option(id="B", label="second")),
        ),
        answer_id="A",
    )
    canonical = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert analysis._record_sha256(record) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _one_record_evidence() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    record = {
        "record_id": "q1",
        "dataset_id": "toy",
        "source_group_id": "g1",
        "answer_id": "left",
        "option_ids": ["left", "right"],
        "request_hash": "a" * 64,
    }
    presentations = [
        {
            "presentation_id": "fresh-eval-v1:q1:order-0",
            "record_id": "q1",
            "dataset_id": "toy",
            "source_group_id": "g1",
            "request_hash": "a" * 64,
            "order_index": 0,
            "order_ids": ["left", "right"],
        },
        {
            "presentation_id": "fresh-eval-v1:q1:order-1",
            "record_id": "q1",
            "dataset_id": "toy",
            "source_group_id": "g1",
            "request_hash": "a" * 64,
            "order_index": 1,
            "order_ids": ["right", "left"],
        },
    ]
    compiled = [
        {
            "presentation_id": row["presentation_id"],
            "request_hash": row["request_hash"],
            "order_ids": row["order_ids"],
            "prompt_sha256": "b" * 64,
            "input_tokens": 5,
        }
        for row in presentations
    ]
    result = {
        "evidence": {
            "outputs": {
                state: [
                    {
                        **row,
                        "winner_option_id": winner,
                        "candidate_logits": [1.0, 0.0],
                        "input_tokens": 5,
                        "prompt_sha256": "b" * 64,
                    }
                    for row, winner in zip(presentations, winners, strict=True)
                ]
                for state, winners in (("base", ["left", "left"]), ("adapter", ["left", "right"]))
            }
        }
    }
    return record, {"presentations": presentations, "compiled_requests": compiled}, result


def test_winner_map_preserves_option_identity_after_rotation() -> None:
    record, payload, result = _one_record_evidence()

    winners = analysis._winner_maps(payload, result, {"q1": record})

    assert winners == {
        "base": {"q1": {0: "left", 1: "left"}},
        "adapter": {"q1": {0: "left", 1: "right"}},
    }


def test_winner_map_rejects_a_request_hash_changed_from_local_data() -> None:
    record, payload, result = _one_record_evidence()
    for row in payload["presentations"]:
        row["request_hash"] = "c" * 64
    for row in payload["compiled_requests"]:
        row["request_hash"] = "c" * 64
    for state_rows in result["evidence"]["outputs"].values():
        for row in state_rows:
            row["request_hash"] = "c" * 64

    import pytest

    with pytest.raises(ValueError, match="local semantic request"):
        analysis._winner_maps(payload, result, {"q1": record})


def test_winner_map_rejects_missing_and_duplicate_order_outputs() -> None:
    import pytest

    record, payload, result = _one_record_evidence()
    result["evidence"]["outputs"]["base"].pop()
    with pytest.raises(ValueError, match="output panel is incomplete"):
        analysis._winner_maps(payload, result, {"q1": record})

    record, payload, result = _one_record_evidence()
    base_rows = result["evidence"]["outputs"]["base"]
    base_rows[1] = dict(base_rows[0])
    with pytest.raises(ValueError, match="duplicate or unmatched identity"):
        analysis._winner_maps(payload, result, {"q1": record})


def test_analyzer_refuses_failed_receipts_before_loading_local_data(monkeypatch) -> None:
    import pytest

    receipt = {
        "schema_version": 1,
        "experiment_id": "fresh-eval-v1",
        "run_id": "failed-run",
        "status": "failed",
        "phase": "worker_execution_failed",
        "payload": None,
        "result": None,
        "raw_worker_result": None,
        "failure": {"stage": "worker_execution"},
        "execution": {"run_marker": "marker", "unknown_work": None},
        "modal": {
            "profile": "reflex-personal",
            "workspace": "rajath-61258",
            "environment": "main",
            "sdk_version": "1.6.1",
            "volume_read_only": True,
        },
    }

    def unexpected_loader(_root):
        raise AssertionError("failed receipts must be rejected before data loading")

    from reflex_decisions import fresh_eval_data

    monkeypatch.setattr(fresh_eval_data, "load_panel", unexpected_loader, raising=False)
    with pytest.raises(ValueError, match="only a complete passed"):
        analysis.analyze_receipt(receipt, root=".")


def _full_receipt_fixture() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    for dataset_id, count in (
        ("arc-challenge-dev-v1", 200),
        ("hans-eval-v1", 300),
        ("winogrande-dev-v1", 200),
    ):
        for index in range(count):
            group_id = (
                f"{dataset_id}-group-{index // 10:02d}"
                if dataset_id == "hans-eval-v1"
                else f"{dataset_id}-group-{index:03d}"
            )
            record_id = f"{dataset_id}-record-{index:03d}"
            records.append(
                {
                    "record_id": record_id,
                    "dataset_id": dataset_id,
                    "source_group_id": group_id,
                    "request": {
                        "context": f"fixture context {index}",
                        "question": "Which option?",
                        "options": [
                            {"id": "A", "label": "first", "description": None},
                            {"id": "B", "label": "second", "description": None},
                        ],
                        "request_hash": hashlib.sha256(record_id.encode()).hexdigest(),
                    },
                    "answer_id": "A",
                }
            )
    records.sort(key=lambda row: (row["dataset_id"], row["record_id"]))
    records_by_id = {row["record_id"]: row for row in records}
    source_file_sha256 = {"source.tar.gz": "1" * 64}
    gpu_pins = {
        "data_file_sha256": source_file_sha256,
        "dataset_counts": {
            "hans-eval-v1": 300,
            "winogrande-dev-v1": 200,
            "arc-challenge-dev-v1": 200,
        },
        "panel_sha256": "2" * 64,
        "source_manifest_sha256": "3" * 64,
    }
    record_metadata: dict[str, dict[str, object]] = {}
    for row in records:
        meta: dict[str, object] = {
            "dataset_id": row["dataset_id"],
            "source_group_id": row["source_group_id"],
            "semantic_request_sha256": row["request"]["request_hash"],
            "record_sha256": analysis._record_sha256(row),
        }
        if row["dataset_id"] == "hans-eval-v1":
            group_index = int(row["source_group_id"].rsplit("-", 1)[1])
            meta.update(
                {
                    "heuristic": f"heuristic-{group_index % 3}",
                    "subcase": f"subcase-{group_index:02d}",
                }
            )
        record_metadata[row["record_id"]] = meta
    metadata = {
        "gpu_pins": gpu_pins,
        "source_file_sha256": source_file_sha256,
        "manifest_sha256": "3" * 64,
        "recipe_sha256": "4" * 64,
        "records_sha256": "2" * 64,
        "record_metadata_sha256": "5" * 64,
        "counts": {
            "total": 700,
            "by_dataset": gpu_pins["dataset_counts"],
            "presentations": 1400,
            "parity_presentations": 24,
        },
        "selection": {"selection_digest_sha256": "6" * 64},
        "record_metadata_by_id": record_metadata,
    }

    presentations: list[dict[str, object]] = []
    compiled: list[dict[str, object]] = []
    for row in records:
        option_ids = ["A", "B"]
        for order in (0, 1):
            order_ids = option_ids if order == 0 else option_ids[1:] + option_ids[:1]
            presentation = {
                "presentation_id": f"fresh-eval-v1:{row['record_id']}:order-{order}",
                "record_id": row["record_id"],
                "dataset_id": row["dataset_id"],
                "source_group_id": row["source_group_id"],
                "request_hash": row["request"]["request_hash"],
                "order_index": order,
                "order_ids": order_ids,
            }
            presentations.append(presentation)
            compiled.append(
                {
                    "presentation_id": presentation["presentation_id"],
                    "request_hash": presentation["request_hash"],
                    "order_ids": order_ids,
                    "input_tokens": 5,
                    "prompt_sha256": "7" * 64,
                }
            )
    outputs: dict[str, list[dict[str, object]]] = {}
    for state in ("base", "adapter"):
        outputs[state] = []
        for presentation in presentations:
            winner = "A"
            outputs[state].append(
                {
                    **presentation,
                    "winner_option_id": winner,
                    "candidate_logits": [
                        1.0 if option_id == winner else 0.0
                        for option_id in presentation["order_ids"]
                    ],
                    "input_tokens": 5,
                    "prompt_sha256": "7" * 64,
                }
            )
    result = {
        "status": "passed",
        "run_id": "fresh-eval-test-run",
        "provenance": {"model_id": "test-model", "model_revision": "test-revision"},
        "evidence": {
            "outputs": outputs,
            "adapter_identity": {"tensor_sha256": "8" * 64},
            "reload_parity": {"max_candidate_logit_delta": 0.0},
        },
    }
    payload = {
        "run_id": "fresh-eval-test-run",
        "selection": {"selection_id": "fixture"},
        "selection_sha256": "9" * 64,
        "payload_sha256": "a" * 64,
        "pins": {"gpu_pins": gpu_pins, "source_file_sha256": {"experiments/fake.py": "b" * 64}},
        "presentations": presentations,
        "compiled_requests": compiled,
    }
    receipt = {
        "schema_version": 1,
        "experiment_id": "fresh-eval-v1",
        "run_id": "fresh-eval-test-run",
        "status": "passed",
        "phase": "complete",
        "payload": payload,
        "result": result,
        "raw_worker_result": result,
        "failure": None,
        "execution": {"run_marker": "fixture-marker", "unknown_work": None},
        "modal": {
            "profile": "reflex-personal",
            "workspace": "rajath-61258",
            "environment": "main",
            "sdk_version": "1.6.1",
            "volume_read_only": True,
        },
    }
    assert set(records_by_id) == set(record_metadata)
    return receipt, records, metadata


def _patch_full_receipt_fixture(monkeypatch, receipt, records, metadata) -> None:
    from experiments import fresh_eval_core as core
    from reflex_decisions import fresh_eval_data

    monkeypatch.setattr(core, "validate_payload", lambda payload, *, root: payload)
    monkeypatch.setattr(core, "validate_result", lambda result, payload, *, root: result)
    monkeypatch.setattr(
        fresh_eval_data, "load_panel", lambda _root: (records, metadata), raising=False
    )


def test_analyzer_rejects_changed_dataset_pins_and_gold_records(monkeypatch, tmp_path) -> None:
    import copy

    import pytest

    receipt, records, metadata = _full_receipt_fixture()
    changed_metadata = copy.deepcopy(metadata)
    changed_metadata["gpu_pins"]["data_file_sha256"]["source.tar.gz"] = "f" * 64
    _patch_full_receipt_fixture(monkeypatch, receipt, records, changed_metadata)
    with pytest.raises(ValueError, match="dataset pins differ"):
        analysis.analyze_receipt(receipt, root=tmp_path)

    receipt, records, metadata = _full_receipt_fixture()
    records[0]["answer_id"] = "B"
    _patch_full_receipt_fixture(monkeypatch, receipt, records, metadata)
    with pytest.raises(ValueError, match="gold, request, or group metadata"):
        analysis.analyze_receipt(receipt, root=tmp_path)


def test_analyzer_rejects_missing_or_malformed_hans_diagnostics(monkeypatch, tmp_path) -> None:
    import pytest

    receipt, records, metadata = _full_receipt_fixture()
    hans_id = next(
        record_id
        for record_id, row in metadata["record_metadata_by_id"].items()
        if row["dataset_id"] == "hans-eval-v1"
    )
    del metadata["record_metadata_by_id"][hans_id]["heuristic"]
    _patch_full_receipt_fixture(monkeypatch, receipt, records, metadata)
    with pytest.raises(ValueError, match="HANS heuristic and subcase diagnostics are incomplete"):
        analysis.analyze_receipt(receipt, root=tmp_path)

    receipt, records, metadata = _full_receipt_fixture()
    for row in metadata["record_metadata_by_id"].values():
        if row["dataset_id"] == "hans-eval-v1":
            row["heuristic"] = "only-one-heuristic"
    _patch_full_receipt_fixture(monkeypatch, receipt, records, metadata)
    with pytest.raises(ValueError, match="exactly three heuristics and 30 subcases"):
        analysis.analyze_receipt(receipt, root=tmp_path)

    receipt, records, metadata = _full_receipt_fixture()
    for row in metadata["record_metadata_by_id"].values():
        if row["dataset_id"] == "hans-eval-v1" and row["subcase"] == "subcase-29":
            row["subcase"] = "subcase-00"
    _patch_full_receipt_fixture(monkeypatch, receipt, records, metadata)
    with pytest.raises(ValueError, match="exactly three heuristics and 30 subcases"):
        analysis.analyze_receipt(receipt, root=tmp_path)


def test_analyzer_reports_only_task_metrics_with_fixed_hans_diagnostics(
    monkeypatch, tmp_path
) -> None:
    receipt, records, metadata = _full_receipt_fixture()
    _patch_full_receipt_fixture(monkeypatch, receipt, records, metadata)

    report = analysis.analyze_receipt(receipt, root=tmp_path)

    assert set(report["tasks"]) == {
        "hans-eval-v1",
        "winogrande-dev-v1",
        "arc-challenge-dev-v1",
    }
    assert report["tasks"]["hans-eval-v1"]["question_count"] == 300
    assert report["tasks"]["hans-eval-v1"]["source_group_count"] == 30
    assert report["tasks"]["winogrande-dev-v1"]["source_group_count"] == 200
    assert report["bootstrap"]["replicates"] == 2_000
    assert len(report["hans_diagnostics"]["heuristic"]) == 3
    assert len(report["hans_diagnostics"]["subcase"]) == 30
    assert "aggregate_score" not in report
    assert report["provenance"]["dataset_source_file_sha256"] == metadata["source_file_sha256"]


def test_module_cli_runs_and_refuses_a_failed_receipt(tmp_path) -> None:
    receipt = {
        "schema_version": 1,
        "experiment_id": "fresh-eval-v1",
        "run_id": "failed-cli-run",
        "status": "failed",
        "phase": "worker_execution_failed",
        "payload": None,
        "result": None,
        "raw_worker_result": None,
        "failure": {"stage": "worker_execution"},
        "execution": {"run_marker": "marker", "unknown_work": None},
        "modal": {
            "profile": "reflex-personal",
            "workspace": "rajath-61258",
            "environment": "main",
            "sdk_version": "1.6.1",
            "volume_read_only": True,
        },
    }
    receipt_path = tmp_path / "receipt.json"
    output_path = tmp_path / "report.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.analyze_fresh_eval",
            "--receipt",
            str(receipt_path),
            "--root",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "only a complete passed fresh-evaluation receipt" in completed.stderr
    assert not output_path.exists()
