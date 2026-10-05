import hashlib

import pytest

from experiments import analyze_broader
from experiments.broader_runner_core import presentations, serialize_remote_records
from reflex_decisions.data import DatasetSpec, DecisionRecord, SplitManifest
from reflex_decisions.schema import DecisionRequest, Option


def _fixture():
    manifest = SplitManifest(
        data_kind="fixture",
        datasets=(
            DatasetSpec(
                dataset_id="boolq-dev-pilot-v1",
                source_id="synthetic-boolq",
                task_family="reading-comprehension",
                split="development",
                source_uri="local://fixture/boolq",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
            DatasetSpec(
                dataset_id="snli-dev-pilot-v1",
                source_id="synthetic-snli",
                task_family="sentence-inference",
                split="development",
                source_uri="local://fixture/snli",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )
    records = (
        DecisionRecord(
            record_id="boolq-dev-pilot-v1-b1",
            dataset_id="boolq-dev-pilot-v1",
            source_group_id="boolq-group",
            request=DecisionRequest(
                context="Passage",
                question="Question?",
                options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
            ),
            answer_id="yes",
        ),
        DecisionRecord(
            record_id="snli-dev-pilot-v1-s1",
            dataset_id="snli-dev-pilot-v1",
            source_group_id="snli-group",
            request=DecisionRequest(
                context="Premise",
                question="Hypothesis",
                options=(
                    Option(id="entailment", label="Entailment"),
                    Option(id="neutral", label="Neutral"),
                    Option(id="contradiction", label="Contradiction"),
                ),
            ),
            answer_id="neutral",
        ),
    )
    remote = serialize_remote_records(records)
    presentations_expected = presentations(remote)
    receipts = []
    for row in presentations_expected:
        winner = {
            (records[0].record_id, 0): "yes",
            (records[0].record_id, 1): "no",
            (records[1].record_id, 0): "neutral",
            (records[1].record_id, 1): "entailment",
        }.get((row["record_id"], row["permutation_index"]), row["option_ids"][0])
        receipts.append(
            {
                "record_id": row["record_id"],
                "request_hash": row["request_hash"],
                "permutation_index": row["permutation_index"],
                "option_ids": row["option_ids"],
                "raw_logits": [
                    2.0 if option_id == winner else 0.0 for option_id in row["option_ids"]
                ],
                "input_tokens": 5,
                "prompt_sha256": hashlib.sha256(
                    f"{row['record_id']}:{row['permutation_index']}".encode()
                ).hexdigest(),
            }
        )
    receipt = {
        "status": "passed",
        "provenance": {
            "model_id": "fixture/model",
            "model_revision": "fixture-revision",
            "temperature": 2.0,
        },
        "presentations": receipts,
    }
    return manifest, records, receipt


def test_analysis_uses_original_rows_for_metrics_and_tracks_any_semantic_flip():
    manifest, records, receipt = _fixture()

    result = analyze_broader.analyze_model(
        "kev",
        receipt,
        records,
        manifest,
        records_sha256="0" * 64,
        manifest_sha256="1" * 64,
    )

    assert result["raw_report"]["overall"]["record_count"] == 2
    assert result["per_dataset"]["boolq-dev-pilot-v1"]["accuracy_interval"]["record_count"] == 1
    assert result["per_dataset"]["snli-dev-pilot-v1"]["accuracy_interval"]["correct"] == 1
    assert result["per_dataset"]["boolq-dev-pilot-v1"]["any_semantic_flip_group_count"] == 1
    assert result["per_dataset"]["snli-dev-pilot-v1"]["any_semantic_flip_group_count"] == 1
    assert result["permutation_accuracy"]["snli-dev-pilot-v1"]["independent_unit_count"] == 1
    assert len(result["permutation_accuracy"]["snli-dev-pilot-v1"]["permutations"]) == 6


def test_analysis_rejects_missing_duplicate_and_tampered_presentations():
    import copy

    manifest, records, receipt = _fixture()
    missing = copy.deepcopy(receipt)
    missing["presentations"].pop()
    duplicate = copy.deepcopy(receipt)
    duplicate["presentations"][1] = duplicate["presentations"][0]
    tampered = copy.deepcopy(receipt)
    tampered["presentations"][0]["option_ids"] = ["no", "yes"]

    for changed, message in (
        (missing, "exact complete"),
        (duplicate, "duplicate presentation"),
        (tampered, "request hash or option order"),
    ):
        with pytest.raises(ValueError, match=message):
            analyze_broader.analyze_model(
                "kev",
                changed,
                records,
                manifest,
                records_sha256="0" * 64,
                manifest_sha256="1" * 64,
            )


def test_analyze_run_validates_full_envelope_and_pairs_each_dataset(tmp_path, monkeypatch):
    import json

    from experiments import baseline_runner_core
    from experiments.broader_runner_core import MODELS, presentations, serialize_remote_records
    from experiments.modal_broader import (
        EXPECTED_PROTOCOL,
        EXPECTED_PROTOCOL_SHA256,
        EXPECTED_SOURCE_AMENDMENT,
        EXPECTED_SOURCE_AMENDMENT_SHA256,
        _plan,
        _source_fingerprints,
    )
    from reflex_decisions import broader_data

    datasets = (
        DatasetSpec(
            dataset_id="boolq-dev-pilot-v1",
            source_id="synthetic-boolq",
            task_family="reading-comprehension",
            split="development",
            source_uri="local://fixture/boolq",
            source_revision="fixture-v1",
            license="synthetic fixture",
        ),
        DatasetSpec(
            dataset_id="snli-dev-pilot-v1",
            source_id="synthetic-snli",
            task_family="sentence-inference",
            split="development",
            source_uri="local://fixture/snli",
            source_revision="fixture-v1",
            license="synthetic fixture",
        ),
    )
    manifest = SplitManifest(data_kind="benchmark", datasets=datasets)
    records = tuple(
        DecisionRecord(
            record_id=f"boolq-dev-pilot-v1-{index}",
            dataset_id="boolq-dev-pilot-v1",
            source_group_id=f"boolq-group-{index}",
            request=DecisionRequest(
                context=f"Passage {index}.",
                question=f"Question {index}?",
                options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
            ),
            answer_id="yes",
        )
        for index in range(32)
    ) + tuple(
        DecisionRecord(
            record_id=f"snli-dev-pilot-v1-{index}",
            dataset_id="snli-dev-pilot-v1",
            source_group_id=f"snli-group-{index}",
            request=DecisionRequest(
                context=f"Premise {index}.",
                question=f"Hypothesis {index}.",
                options=(
                    Option(id="entailment", label="Entailment"),
                    Option(id="neutral", label="Neutral"),
                    Option(id="contradiction", label="Contradiction"),
                ),
            ),
            answer_id="neutral",
        )
        for index in range(32)
    )
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    run_path = tmp_path / "run.json"
    records_path.write_bytes(b"synthetic records\n")
    manifest_path.write_bytes(b"synthetic manifest\n")
    monkeypatch.setattr(
        broader_data,
        "verify_prepared_data",
        lambda records_arg, manifest_arg: (manifest, records),
        raising=False,
    )

    all_rows = presentations(serialize_remote_records(records))
    model_receipts = {}
    for model_name in MODELS:
        pins = analyze_broader.MODEL_PINS[model_name]
        rows = []
        for row in all_rows:
            record = next(record for record in records if record.record_id == row["record_id"])
            rows.append(
                {
                    "record_id": row["record_id"],
                    "request_hash": row["request_hash"],
                    "permutation_index": row["permutation_index"],
                    "option_ids": row["option_ids"],
                    "raw_logits": [
                        2.0 if option_id == record.answer_id else 0.0
                        for option_id in row["option_ids"]
                    ],
                    "input_tokens": 5,
                    "prompt_sha256": "a" * 64,
                }
            )
        auxiliary = baseline_runner_core.EXPECTED_AUXILIARY_FORWARD_COUNTS[model_name]
        model_receipts[model_name] = {
            "model_name": model_name,
            "status": "passed",
            "provenance": {
                "model_id": pins[0],
                "model_revision": pins[1],
                "temperature": 1.0 if model_name == "qwen" else 2.0,
            },
            "presentations": rows,
            "scored_presentation_count": 256,
            "auxiliary_forward_count": auxiliary,
            "total_forward_count": 256 + auxiliary,
        }
    run_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "models": model_receipts,
                "provenance": {
                    "baseline_protocol": EXPECTED_PROTOCOL,
                    "baseline_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
                    "source_amendment": EXPECTED_SOURCE_AMENDMENT,
                    "source_amendment_sha256": EXPECTED_SOURCE_AMENDMENT_SHA256,
                    "source_file_sha256": _source_fingerprints(),
                    "modal_sdk_version": "1.6.1",
                    "profile": "reflex-personal",
                    "workspace": "rajath-61258",
                },
                "limits": _plan()["modal"],
            }
        ),
        encoding="utf-8",
    )

    result = analyze_broader.analyze_run(run_path, records_path, manifest_path)

    assert result["run_sha256"] == hashlib.sha256(run_path.read_bytes()).hexdigest()
    assert result["source_amendment_sha256"] == EXPECTED_SOURCE_AMENDMENT_SHA256
    assert len(result["paired_original_order"]) == 3
    for pair in result["paired_original_order"]:
        assert pair["overall"]["record_count"] == 64
        assert pair["per_dataset"]["boolq-dev-pilot-v1"]["record_count"] == 32
        assert pair["per_dataset"]["snli-dev-pilot-v1"]["record_count"] == 32

    tampered = json.loads(run_path.read_text(encoding="utf-8"))
    tampered["provenance"]["baseline_protocol_sha256"] = "0" * 64
    run_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol SHA-256"):
        analyze_broader.analyze_run(run_path, records_path, manifest_path)


def test_module_cli_help_runs_main():
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-m", "experiments.analyze_broader", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "usage:" in completed.stdout
