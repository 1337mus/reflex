"""Tests for independent receipt validation and source-group uncertainty."""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from pathlib import Path

import pytest

from experiments import analyze_snli_diagnostic as analyzer
from experiments import snli_diagnostic_core as core
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _records() -> tuple[DecisionRecord, ...]:
    from reflex_decisions.snli_diagnostic import DATASET_ID, SNLI_LABELS

    options = tuple(Option(id=label, label=label.title()) for label in SNLI_LABELS)
    records = []
    for group_index in range(64):
        group_id = hashlib.sha256(f"analysis-group-{group_index}".encode()).hexdigest()
        for label in SNLI_LABELS:
            pair_id = f"analysis-pair-{group_index}-{label}"
            record_id = f"{DATASET_ID}-{hashlib.sha256(pair_id.encode()).hexdigest()}"
            records.append(
                DecisionRecord(
                    record_id=record_id,
                    dataset_id=DATASET_ID,
                    source_group_id=group_id,
                    request=DecisionRequest(
                        context=f"Premise for group {group_index}.",
                        question=f"Does the premise support {label}?",
                        options=options,
                    ),
                    answer_id=label,
                )
            )
    return tuple(records)


def _payload(records: tuple[DecisionRecord, ...]) -> dict[str, object]:
    sources = {path: "d" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    sources[core.PROTOCOL_PATH] = core.EXPECTED_PROTOCOL_SHA256
    pins = {
        "records_sha256": core.EXPECTED_RECORDS_SHA256,
        "manifest_sha256": core.EXPECTED_MANIFEST_SHA256,
        "recipe_sha256": core.EXPECTED_RECIPE_SHA256,
        "protocol_sha256": core.EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": sources,
    }
    return core.build_payload(
        records,
        pins=pins,
        run_id=str(uuid.UUID(int=11)),
        nonce=str(uuid.UUID(int=12)),
    )


def _qwen_evidence() -> tuple[dict[str, object], dict[str, object]]:
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / "docs/verification/training-rehearsal.json").read_text())
    base = saved["provenance"]
    base_fields = {
        "model_id",
        "model_revision",
        "model_class",
        "config_class",
        "layer_count",
        "tied_embeddings",
        "no_meta_parameters",
        "load_diagnostics",
        "tokenizer_file_sha256",
        "effective_dtype",
        "device",
        "attention_implementation",
        "use_kernels",
        "use_hub_kernels",
        "versions",
    }
    inventory = dict(saved["evidence"]["adapter_inventory"])
    inventory["trainable_parameter_count"] = 0
    adapter = {
        "expected_file_sha256": dict(core.QWEN_ADAPTER_FILE_SHA256),
        "actual_file_sha256": dict(core.QWEN_ADAPTER_FILE_SHA256),
        "tensor_keys_shapes_values_match": True,
        "default_adapter_active": True,
        "adapter_unmerged": True,
        "inventory": inventory,
        "all_adapter_parameters_frozen": True,
    }
    return {key: base[key] for key in base_fields}, adapter


def _receipt(records: tuple[DecisionRecord, ...]) -> dict[str, object]:
    payload = _payload(records)
    record_by_id = {record.record_id: record for record in records}
    base, adapter = _qwen_evidence()
    source_hashes = payload["pins"]["source_file_sha256"]
    rows = []
    for presentation in payload["presentations"]:
        record = record_by_id[presentation["record_id"]]
        group_index = records.index(record) // 3
        if group_index < 32:
            winner = record.answer_id
        else:
            winner = next(
                option_id
                for option_id in core.MODEL_NAMES[:1] + ("neutral", "contradiction")
                if option_id in presentation["order_ids"] and option_id != record.answer_id
            )
        ids = presentation["order_ids"]
        logits = [2.0 if option_id == winner else 0.0 for option_id in ids]
        rows.append(
            {
                key: presentation[key]
                for key in (
                    "presentation_id",
                    "record_id",
                    "source_group_id",
                    "request_hash",
                    "order_index",
                    "order_ids",
                )
            }
            | {
                "candidate_logits": logits,
                "winner_option_id": winner,
                "input_tokens": 12,
                "prompt_sha256": "f" * 64,
            }
        )
    models = {}
    for name in core.MODEL_NAMES:
        pin = core.MODEL_PINS[name]
        if name in {"qwen_base", "qwen_final"}:
            models[name] = {
                "model_name": name,
                "status": "passed",
                "run_id": payload["run_id"],
                "nonce": payload["nonce"],
                "pins": payload["pins"],
                **pin,
                "forward_counts": {"scored": 1152, "auxiliary": 0, "total": 1152},
                "provenance": {
                    "measured_source_file_sha256": source_hashes,
                    "runtime_versions": dict(core.RUNTIME_VERSION_PINS),
                    "effective_dtype": "torch.bfloat16",
                    "scorer": {},
                    "base": base,
                    "adapter": adapter if name == "qwen_final" else None,
                },
                "presentations": rows,
            }
        else:
            auxiliary = None
            models[name] = {
                "model_name": name,
                "status": "failed",
                "run_id": payload["run_id"],
                "nonce": payload["nonce"],
                "pins": payload["pins"],
                **pin,
                "forward_counts": {"scored": 0, "auxiliary": auxiliary, "total": None},
                "provenance": {
                    "measured_source_file_sha256": {},
                    "runtime_versions": {},
                    "effective_dtype": None,
                    "scorer": {},
                    "base": None,
                    "adapter": None,
                },
                "presentations": [],
                "failure": {"stage": "not_started", "type": "NotStarted", "message": "not run"},
            }
    return {
        "schema_version": core.SCHEMA_VERSION,
        "status": "partial",
        "payload": payload,
        "models": models,
        "provenance": {
            "modal_profile": "reflex-personal",
            "workspace": "rajath-61258",
            "modal_sdk_version": "1.6.1",
            "limits": core.MODAL_LIMITS,
        },
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def test_group_bootstrap_preserves_all_rows_and_orders_together() -> None:
    records = _records()
    receipt = _receipt(records)

    report = analyzer.analyze_receipt(receipt, records)

    assert report["status"] == "analyzed"
    assert set(report["models"]) == {"qwen_base", "qwen_final"}
    assert set(report["model_failures"]) == {"intern", "kev"}
    base = report["models"]["qwen_base"]
    assert base["all_order"]["accuracy"] == 0.5
    assert base["all_order"]["group_equal_weight_accuracy"] == 0.5
    assert base["all_order"]["per_class_recall"]["entailment"]["recall"] == 0.5
    assert base["all_order"]["unique_max_coverage"] == 1.0
    assert base["all_order"]["tie_count"] == 0
    assert base["original_order"]["accuracy"] == 0.5
    assert base["record_behavior"]["semantic_flip_count"] == 0
    assert base["record_behavior"]["all_six_correct_record_count"] == 96

    good_group_ids = {records[index * 3].source_group_id for index in range(32)}
    group_ids = sorted({record.source_group_id for record in records})
    generator = random.Random(20261006)
    draws = [
        sum(18 for _ in range(64) if group_ids[generator.randrange(64)] in good_group_ids) / 1152
        for _ in range(2000)
    ]
    interval = base["all_order"]["bootstrap_95"]
    assert interval == {"lower": _quantile(draws, 0.025), "upper": _quantile(draws, 0.975)}
    assert interval["upper"] - interval["lower"] > 0.15
    assert report["paired_all_order_differences"]["qwen_final_minus_qwen_base"]["ci95"] == {
        "lower": 0.0,
        "upper": 0.0,
    }
    assert len(group_ids) == 64


def test_ties_keep_deterministic_winners_without_unique_max_accuracy() -> None:
    records = _records()
    receipt = _receipt(records)
    for name in ("qwen_base", "qwen_final"):
        outputs = receipt["models"][name]["presentations"]
        for output in outputs:
            output["candidate_logits"] = [0.0, 0.0, 0.0]
            output["winner_option_id"] = min(output["order_ids"])

    report = analyzer.analyze_receipt(receipt, records)

    metrics = report["models"]["qwen_base"]["all_order"]
    assert metrics["accuracy"] == 1 / 3
    assert metrics["unique_max_coverage"] == 0.0
    assert metrics["unique_max_count"] == 0
    assert metrics["unique_max_accuracy"] is None
    assert metrics["tie_count"] == 1152
    assert metrics["confusion"]["entailment"]["contradiction"] == 384


def test_analyzer_reconstructs_payload_from_local_records_and_rejects_drift() -> None:
    records = _records()
    receipt = _receipt(records)
    drifted = list(records)
    drifted[0] = drifted[0].model_copy(
        update={"request": drifted[0].request.model_copy(update={"context": "different premise"})}
    )

    with pytest.raises(ValueError, match="payload differs"):
        analyzer.analyze_receipt(receipt, tuple(drifted))
