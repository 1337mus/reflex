"""Tests for the label-free SNLI worker contract and saved receipts."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest

from experiments import snli_diagnostic_core as core
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _records() -> tuple[DecisionRecord, ...]:
    from reflex_decisions.snli_diagnostic import DATASET_ID, SNLI_LABELS

    options = tuple(Option(id=label, label=label.title()) for label in SNLI_LABELS)
    records = []
    for group_index in range(64):
        group_id = hashlib.sha256(f"group-{group_index}".encode()).hexdigest()
        for label in SNLI_LABELS:
            pair_id = f"pair-{group_index}-{label}"
            record_id = f"{DATASET_ID}-{hashlib.sha256(pair_id.encode()).hexdigest()}"
            records.append(
                DecisionRecord(
                    record_id=record_id,
                    dataset_id=DATASET_ID,
                    source_group_id=group_id,
                    request=DecisionRequest(
                        context=f"Premise {group_index}.",
                        question=f"Does this support {label}?",
                        options=options,
                    ),
                    answer_id=label,
                )
            )
    return tuple(records)


def _pins() -> dict[str, object]:
    sources = {path: "d" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    sources[core.PROTOCOL_PATH] = core.EXPECTED_PROTOCOL_SHA256
    return {
        "records_sha256": core.EXPECTED_RECORDS_SHA256,
        "manifest_sha256": core.EXPECTED_MANIFEST_SHA256,
        "recipe_sha256": core.EXPECTED_RECIPE_SHA256,
        "protocol_sha256": core.EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": sources,
    }


def _payload() -> dict[str, object]:
    return core.build_payload(
        _records(), pins=_pins(), run_id=str(uuid.UUID(int=1)), nonce=str(uuid.UUID(int=2))
    )


def _base_evidence() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "docs/verification/training-rehearsal.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))["provenance"]
    fields = {
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
    return {key: provenance[key] for key in fields}


def _adapter_evidence() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "docs/verification/training-rehearsal.json"
    inventory = json.loads(path.read_text(encoding="utf-8"))["evidence"]["adapter_inventory"]
    inventory = dict(inventory)
    inventory["trainable_parameter_count"] = 0
    return {
        "expected_file_sha256": dict(core.QWEN_ADAPTER_FILE_SHA256),
        "actual_file_sha256": dict(core.QWEN_ADAPTER_FILE_SHA256),
        "tensor_keys_shapes_values_match": True,
        "default_adapter_active": True,
        "adapter_unmerged": True,
        "inventory": inventory,
        "all_adapter_parameters_frozen": True,
    }


def _output_row(presentation: dict[str, object]) -> dict[str, object]:
    order_ids = presentation["order_ids"]
    logits = [1.0 if option_id == order_ids[0] else 0.0 for option_id in order_ids]
    return {
        key: presentation[key]
        for key in (
            "presentation_id",
            "record_id",
            "source_group_id",
            "request_hash",
            "order_index",
            "order_ids",
        )
    } | {
        "candidate_logits": logits,
        "winner_option_id": order_ids[0],
        "input_tokens": 12,
        "prompt_sha256": "f" * 64,
    }


def _failed(payload: dict[str, object], name: str) -> dict[str, object]:
    pin = core.MODEL_PINS[name]
    auxiliary = 0 if name.startswith("qwen_") else None
    return {
        "model_name": name,
        "status": "failed",
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "pins": payload["pins"],
        **pin,
        "forward_counts": {
            "scored": 0,
            "auxiliary": auxiliary,
            "total": auxiliary,
        },
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


def _qwen_base_result(payload: dict[str, object]) -> dict[str, object]:
    sources = payload["pins"]["source_file_sha256"]
    presentations = payload["presentations"]
    return {
        "model_name": "qwen_base",
        "status": "passed",
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "pins": payload["pins"],
        **core.MODEL_PINS["qwen_base"],
        "forward_counts": {"scored": 1152, "auxiliary": 0, "total": 1152},
        "provenance": {
            "measured_source_file_sha256": sources,
            "runtime_versions": dict(core.RUNTIME_VERSION_PINS),
            "effective_dtype": "torch.bfloat16",
            "scorer": {},
            "base": _base_evidence(),
            "adapter": None,
        },
        "presentations": [_output_row(row) for row in presentations],
    }


def _qwen_final_result(payload: dict[str, object]) -> dict[str, object]:
    result = _qwen_base_result(payload)
    result["model_name"] = "qwen_final"
    result["model_id"] = core.MODEL_PINS["qwen_final"]["model_id"]
    result["model_revision"] = core.MODEL_PINS["qwen_final"]["model_revision"]
    result["provenance"]["adapter"] = _adapter_evidence()
    return result


def test_payload_is_label_free_and_contains_all_original_tuple_permutations() -> None:
    payload = core.validate_payload(_payload())

    assert len(payload["presentations"]) == 1152
    rows = payload["presentations"]
    assert set(rows[0]) == {
        "presentation_id",
        "record_id",
        "source_group_id",
        "request_hash",
        "order_index",
        "order_ids",
        "request",
    }
    assert all("answer_id" not in row and "gold_label" not in row for row in rows)
    first_six = rows[:6]
    original = tuple(first_six[0]["order_ids"])
    import itertools

    assert [tuple(row["order_ids"]) for row in first_six] == list(itertools.permutations(original))


def test_payload_rejects_panel_drift_and_extra_target_fields() -> None:
    payload = _payload()
    changed = dict(payload)
    changed["panel_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="panel digest"):
        core.validate_payload(changed)

    rows = [dict(row) for row in payload["presentations"]]
    rows[0]["answer_id"] = "entailment"
    changed = dict(payload, presentations=rows)
    from experiments.snli_diagnostic_core import _json_sha256

    changed["panel_sha256"] = _json_sha256(rows)
    with pytest.raises(ValueError, match="schema"):
        core.validate_payload(changed)


def test_passed_model_result_binds_identity_outputs_and_provenance() -> None:
    payload = _payload()

    result = core.validate_model_result(_qwen_base_result(payload), payload, "qwen_base")

    assert result["status"] == "passed"
    assert len(result["presentations"]) == 1152
    tampered = _qwen_base_result(payload)
    first_row = tampered["presentations"][0]
    first_order = first_row["order_ids"]
    first_row["winner_option_id"] = next(
        option_id for option_id in first_order if option_id != first_row["winner_option_id"]
    )
    with pytest.raises(ValueError, match="winner"):
        core.validate_model_result(tampered, payload, "qwen_base")


def test_qwen_final_requires_the_saved_frozen_adapter_evidence() -> None:
    payload = _payload()

    result = core.validate_model_result(_qwen_final_result(payload), payload, "qwen_final")

    assert result["provenance"]["adapter"]["inventory"]["base_parameters_frozen_bf16"] is True


def test_failed_result_is_only_a_validated_prefix() -> None:
    payload = _payload()
    result = _failed(payload, "qwen_base")
    result["forward_counts"] = {"scored": 1, "auxiliary": 0, "total": 1}
    result["presentations"] = [_output_row(payload["presentations"][0])]

    assert core.validate_model_result(result, payload, "qwen_base")["forward_counts"] == {
        "scored": 1,
        "auxiliary": 0,
        "total": 1,
    }
    result["presentations"] = [_output_row(payload["presentations"][1])]
    with pytest.raises(ValueError, match="prefix"):
        core.validate_model_result(result, payload, "qwen_base")


def test_receipt_requires_every_model_and_lifecycle_failure_keeps_passed_state() -> None:
    payload = _payload()
    models = {name: _failed(payload, name) for name in core.MODEL_NAMES}
    models["qwen_base"] = _qwen_base_result(payload)
    receipt = {
        "schema_version": core.SCHEMA_VERSION,
        "status": "failed",
        "payload": payload,
        "models": models,
        "provenance": {
            "modal_profile": "reflex-personal",
            "workspace": "rajath-61258",
            "modal_sdk_version": "1.6.1",
            "limits": core.MODAL_LIMITS,
        },
        "failure": {"stage": "modal_lifecycle", "type": "RuntimeError", "message": "teardown"},
    }

    normalized = core.validate_receipt(receipt, payload)

    assert normalized["status"] == "failed"
    assert normalized["models"]["qwen_base"]["status"] == "passed"
    del receipt["models"]["kev"]
    with pytest.raises(ValueError, match="all four"):
        core.validate_receipt(receipt, payload)


def test_protocol_and_source_fingerprints_are_pinned() -> None:
    assert core.verify_protocol() == core.EXPECTED_PROTOCOL_SHA256
    fingerprints = core.source_fingerprints()
    assert set(fingerprints) == set(core.SOURCE_FINGERPRINT_PATHS)
    assert fingerprints[core.PROTOCOL_PATH] == core.EXPECTED_PROTOCOL_SHA256
    assert "experiments/__init__.py" in fingerprints
    assert "src/reflex_decisions/__init__.py" in fingerprints
    assert not any("synthetic_data" in path for path in fingerprints)
