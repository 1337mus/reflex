from __future__ import annotations

import uuid

import pytest

from experiments import baseline_intern, baseline_kev, real_pilot_core
from experiments import real_pilot_baseline_core as core
from reflex_decisions import pilot_data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _fixture_records() -> tuple[object, tuple[DecisionRecord, ...]]:
    manifest = pilot_data.build_manifest()
    records: list[DecisionRecord] = []
    for dataset_id, count in real_pilot_core.DATASET_RECORD_COUNTS.items():
        option_count = 14 if dataset_id.startswith("dbpedia14-") else 2
        if dataset_id == "snli-pilot-v1-development":
            option_count = 3
        for index in range(count):
            options = tuple(
                Option(id=f"option-{option_index:02}", label=f"Option {option_index}")
                for option_index in range(option_count)
            )
            records.append(
                DecisionRecord(
                    record_id=f"{dataset_id}-{index:03}",
                    dataset_id=dataset_id,
                    source_group_id=f"source-group-{dataset_id}-{index:03}",
                    request=DecisionRequest(
                        context=f"Synthetic context {dataset_id}-{index}",
                        question="Choose the best option.",
                        options=options,
                    ),
                    answer_id=options[0].id,
                )
            )
    return manifest, tuple(records)


def _build_payload() -> dict[str, object]:
    manifest, records = _fixture_records()
    source_hashes = {path: "a" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[real_pilot_core.PROTOCOL_PATH] = real_pilot_core.EXPECTED_PROTOCOL_SHA256
    source_hashes[core.REFERENCE_PROTOCOL_PATH] = core.EXPECTED_REFERENCE_PROTOCOL_SHA256
    return core.build_evaluation_payload(
        manifest,
        records,
        run_id=str(uuid.UUID(int=1)),
        nonce=str(uuid.UUID(int=2)),
        records_sha256=pilot_data.EXPECTED_RECORDS_SHA256,
        manifest_sha256=pilot_data.EXPECTED_MANIFEST_SHA256,
        recipe_sha256=pilot_data.EXPECTED_RECIPE_SHA256,
        protocol_sha256=real_pilot_core.EXPECTED_PROTOCOL_SHA256,
        reference_protocol_sha256=core.EXPECTED_REFERENCE_PROTOCOL_SHA256,
        source_file_sha256=source_hashes,
    )


def test_reference_protocol_matches_its_frozen_sha256() -> None:
    assert core.verify_reference_protocol() == core.EXPECTED_REFERENCE_PROTOCOL_SHA256


def test_runtime_version_validator_pins_hub_tokenizers_and_safetensors() -> None:
    versions = _runtime_versions()
    versions.update(
        {
            "huggingface-hub": "1.33.0",
            "tokenizers": "0.23.2",
            "safetensors": "0.8.0",
        }
    )
    assert core._validate_runtime_versions(versions) == versions

    for package, wrong_version in (
        ("huggingface-hub", "0.36.0"),
        ("tokenizers", "0.22.1"),
        ("safetensors", "0.6.2"),
    ):
        incompatible = dict(versions)
        incompatible[package] = wrong_version
        with pytest.raises(ValueError, match="frozen Modal image"):
            core._validate_runtime_versions(incompatible)


def _scorer_provenance(model_name: str) -> dict[str, object]:
    if model_name == "intern":
        return {
            "model_id": baseline_intern.MODEL_ID,
            "model_revision": baseline_intern.REVISION,
            "source_commit": baseline_intern.SOURCE_COMMIT,
            "module_sha256": baseline_intern.MODULE_SHA256,
            "temperature": baseline_intern.TEMPERATURE,
            "auxiliary_forward_count": 1,
            "calibration_parity_passed": True,
            "versions": {
                "torch": "2.14.1+cu130",
                "transformers": "5.18.0",
                "huggingface_hub": "0.36.0",
            },
        }
    return {
        "model_id": baseline_kev.MODEL_ID,
        "model_revision": baseline_kev.REVISION,
        "base_model_id": baseline_kev.BASE_MODEL_ID,
        "base_revision": baseline_kev.BASE_REVISION,
        "source_commit": baseline_kev.SOURCE_COMMIT,
        "source_sha256": dict(baseline_kev.SOURCE_SHA256),
        "safe_checkpoint_module_sha256": baseline_kev.SAFE_CHECKPOINT_SHA256,
        "temperature": 1.0,
        "auxiliary_forward_count": 2,
        "temperature_parity_max_abs": 0.0,
    }


def _runtime_versions() -> dict[str, str]:
    return {
        "torch": "2.14.1+cu130",
        "torchvision": "0.29.1+cu130",
        "transformers": "5.18.0",
        "peft": "0.21.0",
        "Pillow": "12.0.0",
        "pydantic": "2.13.5",
        "huggingface-hub": "1.33.0",
        "tokenizers": "0.23.2",
        "safetensors": "0.8.0",
    }


def _model_receipt(
    payload: dict[str, object], model_name: str = "intern", *, complete: bool = True
) -> dict[str, object]:
    model = core.MODEL_PINS[model_name]
    rows = payload["evaluation_presentations"]
    assert isinstance(rows, list)
    outputs: list[dict[str, object]] = []
    for presentation in rows:
        option_ids = list(presentation["order_ids"])
        logits = [
            10.0 if option_id in {"option-03", "option-10"} else 0.0 for option_id in option_ids
        ]
        winner = min(
            option_id
            for option_id, logit in zip(option_ids, logits, strict=True)
            if logit == max(logits)
        )
        outputs.append(
            {
                "presentation_id": presentation["presentation_id"],
                "record_id": presentation["record_id"],
                "dataset_id": presentation["dataset_id"],
                "request_hash": presentation["request_hash"],
                "order_index": presentation["order_index"],
                "order_ids": option_ids,
                "candidate_logits": logits,
                "winner_option_id": winner,
                "input_tokens": 64,
                "prompt_sha256": "b" * 64,
            }
        )
    if not complete:
        outputs.pop()
    auxiliary = core.AUXILIARY_FORWARD_COUNTS[model_name]
    scored = len(outputs)
    return {
        "schema_version": 1,
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "pins": payload["pins"],
        "model_name": model_name,
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "status": "passed",
        "forward_counts": {"scored": scored, "auxiliary": auxiliary, "total": scored + auxiliary},
        "prompt_hash_kind": core.PROMPT_HASH_KINDS[model_name],
        "provenance": {
            "measured_source_file_sha256": dict(payload["pins"]["source_file_sha256"]),
            "scorer": _scorer_provenance(model_name),
            "runtime_versions": _runtime_versions(),
            "use_hub_kernels": "NO",
            "hf_hub_disable_implicit_token": True,
        },
        "presentations": outputs,
    }


def test_evaluation_payload_is_exact_label_free_presentation_panel() -> None:
    payload = _build_payload()

    rows = payload["evaluation_presentations"]
    assert isinstance(rows, list)
    assert len(rows) == 2572
    assert set(payload) == {
        "schema_version",
        "run_id",
        "nonce",
        "pins",
        "evaluation_presentations",
        "evaluation_payload_sha256",
    }
    assert all(
        set(row)
        == {
            "presentation_id",
            "record_id",
            "dataset_id",
            "request_hash",
            "order_index",
            "order_ids",
            "request",
        }
        for row in rows
    )
    assert all("answer_id" not in row and "label" not in row for row in rows)

    dbpedia_rows = [
        row
        for row in rows
        if row["dataset_id"] == "dbpedia14-pilot-v1-development"
        and row["record_id"] == "dbpedia14-pilot-v1-development-000"
    ]
    assert len(dbpedia_rows) == 28
    base_ids = tuple(dbpedia_rows[0]["order_ids"])
    forward = tuple(base_ids[index:] + base_ids[:index] for index in range(14))
    reversed_ids = tuple(reversed(base_ids))
    reverse = tuple(reversed_ids[index:] + reversed_ids[:index] for index in range(14))
    expected_orders = tuple(dict.fromkeys((*forward, *reverse)))
    assert tuple(tuple(row["order_ids"]) for row in dbpedia_rows) == expected_orders
    for row in dbpedia_rows:
        request = DecisionRequest.model_validate(row["request"])
        assert tuple(option.id for option in request.options) == tuple(row["order_ids"])


def test_passed_model_receipt_validates_exact_semantic_output_membership() -> None:
    payload = _build_payload()
    state = _model_receipt(payload)

    normalized = core.validate_model_receipt(state, expected_payload=payload, model_name="intern")

    assert normalized["status"] == "passed"
    outputs = normalized["presentations"]
    assert isinstance(outputs, list)
    assert len(outputs) == 2572
    dbpedia = [row for row in outputs if row["dataset_id"] == "dbpedia14-pilot-v1-development"]
    assert len(dbpedia) == 56 * 28
    assert all(row["winner_option_id"] == "option-03" for row in dbpedia)


def test_passed_model_receipt_rejects_missing_or_duplicate_presentations() -> None:
    payload = _build_payload()
    missing = _model_receipt(payload)
    missing["presentations"].pop()
    duplicate = _model_receipt(payload)
    duplicate["presentations"][-1] = dict(duplicate["presentations"][0])

    for receipt in (missing, duplicate):
        try:
            core.validate_model_receipt(receipt, expected_payload=payload, model_name="intern")
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete or duplicate reference outputs must be rejected")


def test_passed_model_receipt_rejects_protocol_pin_and_measured_source_mismatches() -> None:
    payload = _build_payload()
    bad_pin = _model_receipt(payload)
    bad_pin["pins"]["reference_protocol_sha256"] = "f" * 64
    bad_measurement = _model_receipt(payload)
    bad_measurement["provenance"]["measured_source_file_sha256"][
        "experiments/baseline_intern.py"
    ] = "f" * 64

    for receipt in (bad_pin, bad_measurement):
        try:
            core.validate_model_receipt(receipt, expected_payload=payload, model_name="intern")
        except ValueError as error:
            assert "pin" in str(error).lower() or "source" in str(error).lower()
        else:
            raise AssertionError("a changed reference pin or measured source must be rejected")


def test_failed_model_receipt_preserves_valid_partial_rows() -> None:
    payload = _build_payload()
    state = _model_receipt(payload)
    state["status"] = "failed"
    state["presentations"] = state["presentations"][:17]
    state["forward_counts"] = {"scored": 17, "auxiliary": 1, "total": 18}
    state["failure"] = {
        "stage": "inference",
        "type": "TimeoutError",
        "message": "scorer stopped before completing the panel",
    }

    normalized = core.validate_failed_model_receipt(
        state, expected_payload=payload, model_name="intern"
    )

    assert normalized["status"] == "failed"
    assert normalized["forward_counts"] == {"scored": 17, "auxiliary": 1, "total": 18}
    assert normalized["presentations"] == state["presentations"]
    assert normalized["failure"] == state["failure"]


def test_failed_intern_receipt_accepts_lazy_parity_not_yet_observed() -> None:
    payload = _build_payload()
    state = _model_receipt(payload)
    state["status"] = "failed"
    state["presentations"] = []
    state["forward_counts"] = {"scored": 0, "auxiliary": 0, "total": 0}
    state["provenance"]["scorer"].update(
        {"auxiliary_forward_count": 0, "calibration_parity_passed": False}
    )
    state["failure"] = {
        "stage": "inference",
        "type": "RuntimeError",
        "message": "first scoring call failed before calibration parity completed",
    }

    normalized = core.validate_failed_model_receipt(
        state, expected_payload=payload, model_name="intern"
    )

    assert normalized["forward_counts"] == {"scored": 0, "auxiliary": 0, "total": 0}
    assert normalized["provenance"]["scorer"]["calibration_parity_passed"] is False


def test_failed_intern_receipt_rejects_integer_parity_values() -> None:
    payload = _build_payload()
    for parity_passed in (0, 1):
        state = _model_receipt(payload)
        state["status"] = "failed"
        state["presentations"] = []
        state["forward_counts"] = {"scored": 0, "auxiliary": 0, "total": 0}
        state["provenance"]["scorer"].update(
            {"auxiliary_forward_count": parity_passed, "calibration_parity_passed": parity_passed}
        )
        state["failure"] = {
            "stage": "inference",
            "type": "RuntimeError",
            "message": "calibration parity must be a boolean",
        }

        with pytest.raises(ValueError, match="parity"):
            core.validate_failed_model_receipt(state, expected_payload=payload, model_name="intern")


def test_persisted_model_state_revalidates_against_top_level_identity_and_panel() -> None:
    payload = _build_payload()
    worker_state = _model_receipt(payload)
    normalized = core.validate_model_receipt(
        worker_state, expected_payload=payload, model_name="intern"
    )

    revalidated = core.validate_persisted_model_state(
        normalized, expected_payload=payload, model_name="intern"
    )

    assert revalidated == normalized
