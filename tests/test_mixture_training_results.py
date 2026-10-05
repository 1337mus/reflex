from __future__ import annotations

import hashlib

import pytest

from experiments import mixture_training_data
from experiments.mixture_training_contracts import (
    DATA_FILE_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    ORIGINAL_PROTOCOL_PATH,
    ORIGINAL_PROTOCOL_SHA256,
    PROTOCOL_PATH,
    RUNTIME_VERSION_PINS,
    SCHEMA_VERSION,
    SOURCE_FINGERPRINT_PATHS,
    canonical_json,
    json_sha256,
)
from experiments.mixture_training_core import _validate_group_separation
from experiments.mixture_training_evidence import (
    _validate_adapter_update,
    _validate_provenance,
    _validate_training_losses,
)
from experiments.mixture_training_outputs import (
    _validate_artifact,
    _validate_counts,
    _validate_output_rows,
)
from experiments.mixture_training_results import validate_result
from reflex_decisions import synthetic_data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def _source_fingerprints() -> dict[str, str]:
    return {
        path: EXPECTED_PROTOCOL_SHA256
        if path == PROTOCOL_PATH
        else ORIGINAL_PROTOCOL_SHA256
        if path == ORIGINAL_PROTOCOL_PATH
        else "a" * 64
        for path in SOURCE_FINGERPRINT_PATHS
    }


def _initialization_provenance() -> dict[str, object]:
    fingerprints = _source_fingerprints()
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "versions": dict(RUNTIME_VERSION_PINS),
        "source_file_sha256": fingerprints,
        "measured_source_file_sha256": fingerprints,
        "base_model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "versions": {
                name: version for name, version in RUNTIME_VERSION_PINS.items() if name != "Pillow"
            },
        },
        "cuda_device": "NVIDIA A100",
    }


def test_passed_initialization_does_not_require_training_optimizer() -> None:
    fingerprints = _source_fingerprints()
    payload = {"pins": {"source_file_sha256": fingerprints}}

    normalized = _validate_provenance(
        _initialization_provenance(), payload, passed=True, train=False
    )

    assert "optimizer" not in normalized


def test_initialization_cannot_claim_optimizer_training() -> None:
    fingerprints = _source_fingerprints()
    payload = {"pins": {"source_file_sha256": fingerprints}}
    provenance = _initialization_provenance()
    provenance["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
    }

    with pytest.raises(ValueError, match="must not claim optimizer training"):
        _validate_provenance(provenance, payload, passed=True, train=False)


def _scored_row() -> tuple[dict[str, object], dict[str, object]]:
    request = DecisionRequest(
        context="A short neutral context.",
        question="Which option is correct?",
        options=(Option(id="zeta", label="Zeta"), Option(id="alpha", label="Alpha")),
    )
    presentation = {
        "presentation_id": "presentation-1",
        "record_id": "dataset-record-1",
        "dataset_id": "dataset",
        "source_group_id": "group-1",
        "request_hash": request.request_hash,
        "order_index": 0,
        "order_ids": ["zeta", "alpha"],
        "request": request.model_dump(mode="json"),
    }
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
            "candidate_logits": [0.5, 0.5],
            "winner_option_id": "alpha",
            "input_tokens": 12,
            "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
        }
    )
    return presentation, output


def test_output_validation_recomputes_ties_and_binds_artifact_bytes() -> None:
    presentation, output = _scored_row()
    normalized = _validate_output_rows(
        [output], [presentation], label="outputs", require_complete=True
    )
    digest = hashlib.sha256(canonical_json(normalized) + b"\n").hexdigest()

    artifact = _validate_artifact(
        {"path": "/artifacts/runs/example-real/outputs.jsonl", "sha256": digest},
        normalized,
        "example-real",
        required=True,
    )

    assert artifact == {
        "path": "/artifacts/runs/example-real/outputs.jsonl",
        "sha256": digest,
    }
    tampered = dict(output, winner_option_id="zeta")
    with pytest.raises(ValueError, match="lexicographic tie rule"):
        _validate_output_rows([tampered], [presentation], label="outputs", require_complete=True)


def test_forward_counts_match_frozen_initialization_plan() -> None:
    counts = {
        "base_evaluation": 939,
        "training": 0,
        "final_evaluation": 0,
        "reload_parity": 0,
        "total": 939,
    }
    tokens = {
        "base_evaluation": 939,
        "training": 0,
        "final_evaluation": 0,
        "reload_parity": 0,
        "total": 939,
    }

    normalized, _ = _validate_counts(counts, tokens, passed_phase="initialize")

    assert normalized == counts
    incomplete_tokens = dict(tokens, reload_parity=None)
    with pytest.raises(ValueError, match="retain all forward and input token counts"):
        _validate_counts(counts, incomplete_tokens, passed_phase="initialize")
    invalid = dict(counts, total=940)
    with pytest.raises(ValueError, match="total forward count differs"):
        _validate_counts(invalid, tokens, passed_phase="initialize")


def test_training_and_evaluation_record_ids_must_be_disjoint() -> None:
    request = DecisionRequest(
        context="A neutral context.",
        question="Choose one.",
        options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
    )
    train_record = DecisionRecord(
        record_id="record-collision",
        dataset_id="fixture-train",
        source_group_id="train-group",
        request=request,
        answer_id="yes",
    )
    evaluation = {
        "record_id": "record-collision",
        "source_group_id": "different-evaluation-group",
        "request_hash": "b" * 64,
    }

    with pytest.raises(ValueError, match="record IDs must be disjoint"):
        _validate_group_separation((train_record,), [evaluation])


def test_adapter_update_names_are_bound_to_the_tensor_inventory() -> None:
    inventory = {
        "adapter_tensor_names": [
            "layer-a.lora_A.default.weight",
            "layer-a.lora_B.default.weight",
        ]
    }
    update = {
        "changed_tensor_count": 1,
        "changed_tensor_names": ["layer-a.lora_B.weight"],
    }

    assert _validate_adapter_update(update, inventory) == update
    with pytest.raises(ValueError, match="absent from the inventory"):
        _validate_adapter_update(
            {"changed_tensor_count": 1, "changed_tensor_names": ["missing"]}, inventory
        )
    with pytest.raises(ValueError, match="do not match the inventory"):
        _validate_adapter_update(
            {
                "changed_tensor_count": 2,
                "changed_tensor_names": ["layer-a.lora_B.weight"] * 2,
            },
            inventory,
        )
    with pytest.raises(ValueError, match="collide after default-adapter normalization"):
        _validate_adapter_update(
            {
                "changed_tensor_count": 2,
                "changed_tensor_names": [
                    "layer-a.lora_B.weight",
                    "layer-a.lora_B.default.weight",
                ],
            },
            inventory,
        )


def test_training_losses_allow_zero_steps_but_require_positive_run_totals() -> None:
    losses = [
        {
            "update": update,
            "mean_loss": 0.0,
            "lo_ra_b_gradient_l1": 0.0 if update == 1 else 1.0,
            "gradient_norm": 0.0 if update == 1 else 0.5,
        }
        for update in range(1, 253)
    ]

    normalized = _validate_training_losses(losses)

    assert normalized[0]["lo_ra_b_gradient_l1"] == 0.0
    assert normalized[-1]["gradient_norm"] == 0.5
    all_zero = [
        {
            "update": update,
            "mean_loss": 0.0,
            "lo_ra_b_gradient_l1": 0.0,
            "gradient_norm": 0.0,
        }
        for update in range(1, 253)
    ]
    with pytest.raises(ValueError, match="nonzero aggregate gradient evidence"):
        _validate_training_losses(all_zero)


def test_passed_initialization_result_is_consumable_without_optimizer_provenance() -> None:
    experiment_id = "init-result-validation"
    run_id = f"{experiment_id}-init"
    source_hashes = _source_fingerprints()
    pins = {
        "data_file_sha256": dict(DATA_FILE_SHA256),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": source_hashes,
    }
    presentations = list(
        mixture_training_data.build_synthetic_evaluation_presentations(
            synthetic_data.build_candidate().records
        )
    )
    unsigned_payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "nonce": "00000000-0000-4000-8000-000000000003",
        "phase": "initialize",
        "arm": None,
        "pins": pins,
        "train_records": [],
        "evaluation_presentations": presentations,
        "schedule_sha256": None,
        "initialization": None,
    }
    payload = dict(unsigned_payload, payload_sha256=json_sha256(unsigned_payload))
    outputs = []
    tokens_per_row = 12
    for presentation in presentations:
        request = DecisionRequest.model_validate(presentation["request"])
        order_ids = presentation["order_ids"]
        row = {
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
        logits = [float(index) for index in range(len(order_ids))]
        row.update(
            {
                "candidate_logits": logits,
                "winner_option_id": order_ids[-1],
                "input_tokens": tokens_per_row,
                "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
            }
        )
        outputs.append(row)
    output_digest = hashlib.sha256(canonical_json(outputs) + b"\n").hexdigest()
    initialization = {
        "snapshot": {
            "update": 0,
            "path": f"/artifacts/runs/{run_id}/adapter-update-000",
            "files_sha256": {
                "adapter_model.safetensors": "b" * 64,
                "adapter_config.json": "c" * 64,
            },
        },
        "tensor_sha256": "d" * 64,
    }
    evidence = {
        "forward_counts": {
            "base_evaluation": 939,
            "training": 0,
            "final_evaluation": 0,
            "reload_parity": 0,
            "total": 939,
        },
        "input_token_counts": {
            "base_evaluation": len(outputs) * tokens_per_row,
            "training": 0,
            "final_evaluation": 0,
            "reload_parity": 0,
            "total": len(outputs) * tokens_per_row,
        },
        "outputs": outputs,
        "output_artifact": {
            "path": f"/artifacts/runs/{run_id}/outputs.jsonl",
            "sha256": output_digest,
        },
        "initialization": initialization,
    }
    base_versions = {
        name: version for name, version in RUNTIME_VERSION_PINS.items() if name != "Pillow"
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "nonce": payload["nonce"],
        "phase": "initialize",
        "arm": None,
        "payload_sha256": payload["payload_sha256"],
        "status": "passed",
        "provenance": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "versions": dict(RUNTIME_VERSION_PINS),
            "source_file_sha256": source_hashes,
            "measured_source_file_sha256": source_hashes,
            "base_model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "versions": base_versions,
            },
            "cuda_device": "NVIDIA A100",
        },
        "evidence": evidence,
        "failure": None,
    }

    normalized = validate_result(result, payload)

    assert normalized["status"] == "passed"
    assert "optimizer" not in normalized["provenance"]

    partial_counts = {
        "base_evaluation": None,
        "training": None,
        "final_evaluation": None,
        "reload_parity": None,
        "total": None,
    }
    failed = dict(result)
    failed.update(
        {
            "phase": "source_verification_failed",
            "status": "failed",
            "evidence": {
                "forward_counts": dict(partial_counts),
                "input_token_counts": dict(partial_counts),
                "outputs": [],
                "output_artifact": None,
                "initialization": None,
            },
            "failure": {
                "stage": "source_verification",
                "type": "ValueError",
                "message": "uploaded source hash differs",
            },
        }
    )

    normalized_failure = validate_result(failed, payload)

    assert normalized_failure["status"] == "failed"
    assert normalized_failure["failure"]["stage"] == "source_verification"

    import sys

    from experiments import mixture_training_runtime

    before_modules = set(sys.modules)
    preflight = validate_result(mixture_training_runtime._initial_result(payload), payload)
    newly_imported = set(sys.modules) - before_modules
    forbidden = ("torch", "transformers", "peft", "modal")

    assert preflight["status"] == "failed"
    assert preflight["provenance"]["cuda_device"] is None
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in newly_imported
        for prefix in forbidden
    )
