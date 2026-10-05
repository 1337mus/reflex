from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pytest

from experiments import adapter_transfer_contracts as contracts
from experiments import adapter_transfer_core as core
from experiments import adapter_transfer_data as data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def _records() -> tuple[DecisionRecord, ...]:
    rows = []
    for dataset in ("copa-dev-pilot-v1", "boolq-dev-pilot-v1"):
        for index in range(32):
            rows.append(
                DecisionRecord(
                    record_id=f"{dataset}-{index:02d}",
                    dataset_id=dataset,
                    source_group_id=f"{dataset}-group-{index:02d}",
                    request=DecisionRequest(
                        context=f"Context {dataset} {index}.",
                        question=f"Question {dataset} {index}?",
                        options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
                    ),
                    answer_id="yes",
                )
            )
    return tuple(rows)


def _selection() -> dict[str, Any]:
    return {
        "selection_id": "adapter-transfer-v1",
        "training_run_id": "natural-reasoning-2026-10-04-r1",
        "arm": "snli_mix",
        "update": 378,
        "training_receipt_sha256": "1" * 64,
        "analysis_sha256": "2" * 64,
        "agreement_sha256": "3" * 64,
        "storage_verification_sha256": "4" * 64,
        "snapshot": {
            "update": 378,
            "path": "/artifacts/runs/natural-reasoning-2026-10-04-r1-snli/adapter-update-378",
            "files_sha256": {
                "adapter_config.json": "5" * 64,
                "adapter_model.safetensors": "6" * 64,
            },
        },
    }


def _payload(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(contracts, "EXPECTED_SELECTION_SHA256", "a" * 64)
    from experiments.mixture_training_contracts import json_sha256

    selection = _selection()
    monkeypatch.setattr(
        contracts,
        "EXPECTED_CANONICAL_SELECTION_SHA256",
        json_sha256(selection),
        raising=False,
    )
    source_hashes = {
        path: f"{index:064x}" for index, path in enumerate(contracts.SOURCE_FINGERPRINT_PATHS, 1)
    }
    source_hashes[contracts.PROTOCOL_PATH] = contracts.EXPECTED_PROTOCOL_SHA256
    monkeypatch.setattr(core, "source_fingerprints", lambda root=None: source_hashes)
    from experiments import mixture_training_contracts

    training_hashes = dict(mixture_training_contracts.DATA_FILE_SHA256)
    data_hashes = dict(contracts.PANEL_FILE_SHA256)
    pins = {
        "data_file_sha256": data_hashes,
        "training_data_file_sha256": training_hashes,
        "panel_sha256": {
            "copa": data_hashes[contracts.COPA_RECORDS_PATH],
            "boolq": contracts.BOOLQ_PANEL_SHA256,
        },
        "selection_file_sha256": "a" * 64,
        "protocol_sha256": contracts.EXPECTED_PROTOCOL_SHA256,
        "training_protocol_sha256": mixture_training_contracts.EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": source_hashes,
    }
    return core.build_payload(
        "test-transfer", str(uuid.uuid4()), data.build_presentations(_records()), selection, pins
    )


def _resign(payload: dict[str, Any]) -> dict[str, Any]:
    from experiments.mixture_training_contracts import json_sha256

    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return {**unsigned, "payload_sha256": json_sha256(unsigned)}


def _output(presentation: dict[str, Any]) -> dict[str, Any]:
    request = DecisionRequest.model_validate(presentation["request"])
    order_ids = presentation["order_ids"]
    winner = min(order_ids)
    scores = [1.0 if option_id == winner else 0.0 for option_id in order_ids]
    return {
        "presentation_id": presentation["presentation_id"],
        "record_id": presentation["record_id"],
        "dataset_id": presentation["dataset_id"],
        "source_group_id": presentation["source_group_id"],
        "request_hash": presentation["request_hash"],
        "order_index": presentation["order_index"],
        "order_ids": order_ids,
        "candidate_logits": scores,
        "winner_option_id": winner,
        "input_tokens": 5,
        "prompt_sha256": hashlib.sha256(render_prompt(request).encode()).hexdigest(),
    }


def _provenance(payload: dict[str, Any]) -> dict[str, Any]:
    from experiments import mixture_training_contracts

    base_versions = {
        key: value
        for key, value in mixture_training_contracts.RUNTIME_VERSION_PINS.items()
        if key != "Pillow"
    }
    tokenizer_hashes = {"tokenizer.json": "d" * 64}
    return {
        "model_id": contracts.MODEL_ID,
        "model_revision": contracts.MODEL_REVISION,
        "versions": dict(mixture_training_contracts.RUNTIME_VERSION_PINS),
        "source_file_sha256": payload["pins"]["source_file_sha256"],
        "measured_source_file_sha256": payload["pins"]["source_file_sha256"],
        "base_model": {
            "model_id": contracts.MODEL_ID,
            "model_revision": contracts.MODEL_REVISION,
            "effective_dtype": "torch.bfloat16",
            "versions": base_versions,
            "tokenizer_file_sha256": tokenizer_hashes,
        },
        "cuda_device": "NVIDIA A10",
        "reload_tokenizer_file_sha256": tokenizer_hashes,
    }


def _passed_result(payload: dict[str, Any]) -> dict[str, Any]:
    outputs = {
        name: [_output(row) for row in payload["presentations"]] for name in ("base", "adapter")
    }
    parity = [_output(row) for row in payload["parity_presentations"]]
    adapter_identity = {
        "snapshot_path": payload["selection"]["snapshot"]["path"],
        "files_sha256": payload["selection"]["snapshot"]["files_sha256"],
        "tensor_sha256": "e" * 64,
        "reloaded_tensor_sha256": "e" * 64,
        "dtype": "torch.float32",
    }
    counts = {
        "base_evaluation": 128,
        "training": 0,
        "final_evaluation": 128,
        "reload_parity": 16,
        "total": 272,
    }
    tokens = {
        "base_evaluation": 640,
        "training": 0,
        "final_evaluation": 640,
        "reload_parity": 80,
        "total": 1360,
    }
    return {
        "schema_version": contracts.SCHEMA_VERSION,
        "experiment_id": contracts.EXPERIMENT_ID,
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "selection_sha256": payload["selection_sha256"],
        "payload_sha256": payload["payload_sha256"],
        "status": "passed",
        "phase": "completed",
        "provenance": _provenance(payload),
        "evidence": {
            "forward_counts": counts,
            "input_token_counts": tokens,
            "outputs": outputs,
            "adapter_identity": adapter_identity,
            "reload_parity": {"outputs": parity, "max_candidate_logit_delta": 0.0},
        },
        "failure": None,
    }


def test_payload_is_request_only_hash_bound_and_rejects_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(monkeypatch)

    normalized = core.validate_payload(payload)
    assert len(normalized["presentations"]) == 128
    assert len(normalized["parity_presentations"]) == 16
    assert all(
        "answer_id" not in row and "source_group_id" in row for row in normalized["presentations"]
    )
    assert all("answer_id" not in row["request"] for row in normalized["presentations"])
    tampered = {**payload, "run_id": "different-transfer"}
    with pytest.raises(ValueError, match="payload SHA-256"):
        core.validate_payload(tampered)


def test_payload_rejects_unpinned_selection_and_wrong_source_or_panel_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(monkeypatch)
    bad_selection = {**payload, "selection": {**payload["selection"], "arm": "other"}}
    with pytest.raises(ValueError, match="pinned final SNLI adapter"):
        core.validate_payload(_resign(bad_selection))
    bad_path = {
        **payload,
        "selection": {
            **payload["selection"],
            "snapshot": {
                **payload["selection"]["snapshot"],
                "path": "/artifacts/runs/natural-reasoning-2026-10-04-r1-syn/adapter-update-378",
            },
        },
    }
    with pytest.raises(ValueError, match="pinned final SNLI checkpoint"):
        core.validate_payload(_resign(bad_path))
    bad_pin = {
        **payload,
        "pins": {**payload["pins"], "panel_sha256": {"copa": "0" * 64, "boolq": "1" * 64}},
    }
    with pytest.raises(ValueError, match="panel SHA-256"):
        core.validate_payload(_resign(bad_pin))


def test_payload_rejects_rehashed_selection_content_not_matching_the_pinned_canonical_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.mixture_training_contracts import json_sha256

    payload = _payload(monkeypatch)
    changed_selection = {
        **payload["selection"],
        "analysis_sha256": "9" * 64,
    }
    forged = _resign(
        {
            **payload,
            "selection": changed_selection,
            "selection_sha256": json_sha256(changed_selection),
        }
    )

    with pytest.raises(ValueError, match="canonical selection SHA-256"):
        core.validate_payload(forged)


def test_completed_result_revalidates_rows_counts_identity_and_reload_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(monkeypatch)
    result = _passed_result(payload)

    normalized = core.validate_result(result, payload)

    assert normalized["evidence"]["forward_counts"]["total"] == 272
    bad = {**result, "payload_sha256": "0" * 64}
    with pytest.raises(ValueError, match="payload_sha256"):
        core.validate_result(bad, payload)
    bad_identity = {
        **result,
        "evidence": {
            **result["evidence"],
            "outputs": {
                **result["evidence"]["outputs"],
                "adapter": result["evidence"]["outputs"]["adapter"][:-1],
            },
        },
    }
    with pytest.raises(ValueError, match="every expected presentation"):
        core.validate_result(bad_identity, payload)


@pytest.mark.parametrize("category", ["base_evaluation", "final_evaluation", "reload_parity"])
def test_passed_result_requires_exact_token_sums_for_every_retained_phase(
    monkeypatch: pytest.MonkeyPatch, category: str
) -> None:
    payload = _payload(monkeypatch)
    result = _passed_result(payload)
    token_counts = dict(result["evidence"]["input_token_counts"])
    token_counts[category] += 1
    token_counts["total"] += 1
    forged = {
        **result,
        "evidence": {
            **result["evidence"],
            "input_token_counts": token_counts,
        },
    }

    with pytest.raises(
        ValueError, match=f"{category} token sum differs from retained output evidence"
    ):
        core.validate_result(forged, payload)


def test_base_and_adapter_token_counts_match_for_all_aligned_evaluations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(monkeypatch)
    result = _passed_result(payload)
    evidence = result["evidence"]
    outputs = evidence["outputs"]
    adapter_rows = list(outputs["adapter"])
    adapter_row = dict(adapter_rows[20])
    adapter_row["input_tokens"] += 1
    adapter_rows[20] = adapter_row
    token_counts = dict(evidence["input_token_counts"])
    token_counts["final_evaluation"] += 1
    token_counts["total"] += 1
    forged = {
        **result,
        "evidence": {
            **evidence,
            "outputs": {**outputs, "adapter": adapter_rows},
            "input_token_counts": token_counts,
        },
    }

    with pytest.raises(ValueError, match="base and adapter input token counts differ"):
        core.validate_result(forged, payload)


def test_failed_result_preserves_valid_rows_when_reload_parity_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(monkeypatch)
    result = _passed_result(payload)
    evidence = result["evidence"]
    parity = list(evidence["reload_parity"]["outputs"])
    parity_row = dict(parity[0])
    scores = list(parity_row["candidate_logits"])
    scores.reverse()
    parity_row["candidate_logits"] = scores
    parity_row["winner_option_id"] = parity_row["order_ids"][scores.index(max(scores))]
    parity[0] = parity_row
    failed = {
        **result,
        "status": "failed",
        "phase": "reload_parity_failed",
        "failure": {"stage": "reload_parity", "type": "ValueError", "message": "parity mismatch"},
        "evidence": {
            **evidence,
            "reload_parity": {
                "outputs": parity,
                "max_candidate_logit_delta": 1.0,
            },
        },
    }

    normalized = core.validate_result(failed, payload)

    assert normalized["status"] == "failed"
    assert normalized["evidence"]["reload_parity"]["outputs"][0] == parity_row
    assert normalized["evidence"]["reload_parity"]["max_candidate_logit_delta"] == 1.0


def test_failed_result_preserves_valid_rows_when_base_adapter_token_counts_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(monkeypatch)
    result = _passed_result(payload)
    evidence = result["evidence"]
    outputs = evidence["outputs"]
    adapter_rows = list(outputs["adapter"])
    adapter_row = dict(adapter_rows[20])
    adapter_row["input_tokens"] += 1
    adapter_rows[20] = adapter_row
    token_counts = dict(evidence["input_token_counts"])
    token_counts["final_evaluation"] += 1
    token_counts["total"] += 1
    failed = {
        **result,
        "status": "failed",
        "phase": "final_evaluation_failed",
        "failure": {"stage": "final_evaluation", "type": "ValueError", "message": "token mismatch"},
        "evidence": {
            **evidence,
            "outputs": {**outputs, "adapter": adapter_rows},
            "input_token_counts": token_counts,
        },
    }

    normalized = core.validate_result(failed, payload)

    assert normalized["status"] == "failed"
    assert normalized["evidence"]["outputs"]["adapter"][20]["input_tokens"] == 6


def test_failed_result_allows_unknown_fresh_reload_tensor_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(monkeypatch)
    result = _passed_result(payload)
    evidence = result["evidence"]
    identity = {**evidence["adapter_identity"], "reloaded_tensor_sha256": None}
    failed = {
        **result,
        "status": "failed",
        "phase": "reload_parity_failed",
        "failure": {"stage": "reload_parity", "type": "RuntimeError", "message": "reload stopped"},
        "evidence": {**evidence, "adapter_identity": identity},
    }

    normalized = core.validate_result(failed, payload)

    assert normalized["evidence"]["adapter_identity"]["tensor_sha256"] == "e" * 64
    assert normalized["evidence"]["adapter_identity"]["reloaded_tensor_sha256"] is None


def test_failed_result_preserves_unknown_counts_and_partial_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(monkeypatch)
    partial = _output(payload["presentations"][0])
    evidence = {
        "forward_counts": {
            "base_evaluation": 1,
            "training": 0,
            "final_evaluation": None,
            "reload_parity": None,
            "total": 1,
        },
        "input_token_counts": {
            "base_evaluation": 5,
            "training": 0,
            "final_evaluation": None,
            "reload_parity": None,
            "total": 5,
        },
        "outputs": {"base": [partial], "adapter": []},
        "adapter_identity": None,
        "reload_parity": {"outputs": [], "max_candidate_logit_delta": None},
    }
    failed = core.build_failed_result(
        payload,
        stage="base_evaluation",
        error=RuntimeError("secret/path"),
        sanitize=lambda exc: "scoring failed",
        provenance=None,
        evidence=evidence,
    )

    normalized = core.validate_result(failed, payload)
    assert normalized["status"] == "failed"
    assert normalized["evidence"]["outputs"]["base"] == [partial]
    assert normalized["evidence"]["forward_counts"]["final_evaluation"] is None
    assert normalized["evidence"]["forward_counts"]["training"] == 0

    false_zero = {
        **failed,
        "evidence": {
            **failed["evidence"],
            "forward_counts": {**failed["evidence"]["forward_counts"], "total": 0},
        },
    }
    with pytest.raises(ValueError, match="total forwards do not match"):
        core.validate_result(false_zero, payload)
