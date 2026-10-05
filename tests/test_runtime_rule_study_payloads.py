from __future__ import annotations

import hashlib
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from itertools import permutations
from typing import Any

import pytest

from experiments import (
    adapter_transfer_contracts,
    mixture_training_contracts,
    mixture_training_data,
    runtime_rule_study_cache,
    runtime_rule_study_contracts,
    runtime_rule_study_inputs,
    runtime_rule_study_payload_rows,
    runtime_rule_study_payloads,
    runtime_rule_study_tokens,
)
from experiments import (
    runtime_rule_study_data as study,
)
from experiments.runtime_rule_study_payloads import build_worker_payload
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def _record(record_id: str, dataset_id: str, group_id: str, option_count: int) -> DecisionRecord:
    options = tuple(
        Option(id=f"{record_id}-option-{index}", label=f"Option {index}")
        for index in range(option_count)
    )
    return DecisionRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        source_group_id=group_id,
        request=DecisionRequest(
            context=f"Self-authored payload fixture context for {record_id}.",
            question=f"Which option applies to {record_id}?",
            options=options,
        ),
        answer_id=options[0].id,
    )


def _training_pools() -> tuple[tuple[DecisionRecord, ...], ...]:
    real = tuple(
        _record(
            f"real-{dataset_id}-{index:03d}",
            dataset_id,
            f"real-group-{dataset_id}-{index:03d}",
            14 if dataset_id.startswith("dbpedia14-") else 2,
        )
        for dataset_id, count in mixture_training_data.REAL_TRAIN_DATASET_COUNTS.items()
        for index in range(count)
    )
    synthetic = tuple(
        _record(
            f"synthetic-{dataset_id}-{index:03d}",
            dataset_id,
            f"synthetic-group-{dataset_id}-{index:03d}",
            4,
        )
        for dataset_id, count in mixture_training_data.SYNTHETIC_TRAIN_DATASET_COUNTS.items()
        for index in range(count)
    )
    snli = tuple(
        _record(
            f"snli-training-{index:03d}",
            "snli-training-v1",
            f"snli-training-group-{index:03d}",
            3,
        )
        for index in range(500)
    )
    sizes = (4,) * 19 + (6,) * 18 + (8,) * 19
    routing = tuple(
        _record(
            f"routing-train-{index:03d}",
            study.NEW_TRAIN_DATASET_IDS["routing"],
            f"routing-train-group-{index:03d}",
            size,
        )
        for index, size in enumerate(sizes)
    )
    tool = tuple(
        _record(
            f"tool-train-{index:03d}",
            study.NEW_TRAIN_DATASET_IDS["tool"],
            f"tool-train-group-{index:03d}",
            size,
        )
        for index, size in enumerate(sizes)
    )
    return real, synthetic, snli, routing, tool


def _development_pools() -> tuple[tuple[DecisionRecord, ...], tuple[DecisionRecord, ...]]:
    sizes = (4,) * 5 + (6,) * 5 + (8,) * 4
    routing = tuple(
        _record(
            f"routing-development-{index:02d}",
            study.NEW_DEVELOPMENT_DATASET_IDS["routing"],
            f"routing-development-group-{index:02d}",
            size,
        )
        for index, size in enumerate(sizes)
    )
    tool = tuple(
        _record(
            f"tool-development-{index:02d}",
            study.NEW_DEVELOPMENT_DATASET_IDS["tool"],
            f"tool-development-group-{index:02d}",
            size,
        )
        for index, size in enumerate(sizes)
    )
    return routing, tool


def _old_retention_rows() -> list[dict[str, object]]:
    record_shapes = {
        "dbpedia14-pilot-v1-development": ((56, 14),),
        "sms-pilot-v1-development": ((60, 2),),
        "snli-balanced-v1-development": ((192, 3),),
        "synthetic-atomic-fact-inference-v1-development": ((75, 3),),
        "synthetic-numeric-selection-v1-development": (
            (14, 2),
            (12, 4),
            (12, 8),
            (12, 16),
        ),
        "boolq-dev-pilot-v1": ((32, 2),),
        "copa-dev-pilot-v1": ((32, 2),),
    }
    rows = []
    for dataset_id, batches in record_shapes.items():
        for batch_index, (count, option_count) in enumerate(batches):
            for index in range(count):
                record = _record(
                    f"old-{dataset_id}-{batch_index}-{index:03d}",
                    dataset_id,
                    f"old-group-{dataset_id}-{batch_index}-{index:03d}",
                    option_count,
                )
                options = record.request.options
                if dataset_id.startswith("dbpedia14-"):
                    reverse = tuple(reversed(options))
                    orders = tuple(
                        options[index:] + options[:index] for index in range(option_count)
                    )
                    orders += tuple(
                        reverse[index:] + reverse[:index] for index in range(option_count)
                    )
                elif dataset_id in {
                    "snli-balanced-v1-development",
                    "synthetic-atomic-fact-inference-v1-development",
                }:
                    orders = tuple(permutations(options))
                else:
                    orders = tuple(
                        options[index:] + options[:index] for index in range(option_count)
                    )
                for order_index, ordered in enumerate(orders):
                    request = record.request.model_copy(update={"options": ordered})
                    order_ids = [option.id for option in ordered]
                    rows.append(
                        {
                            "presentation_id": f"old-{record.record_id}-{order_index}",
                            "record_id": record.record_id,
                            "dataset_id": record.dataset_id,
                            "source_group_id": record.source_group_id,
                            "request_hash": request.request_hash,
                            "order_index": order_index,
                            "order_ids": order_ids,
                            "request": request.model_dump(mode="json"),
                        }
                    )
    return rows


def _compiled_spec(
    request: DecisionRequest, record_id: str, presentation_id: str | None = None
) -> dict[str, object]:
    spec: dict[str, object] = {
        "request_json_sha256": mixture_training_contracts.json_sha256(
            request.model_dump(mode="json")
        ),
        "request_hash": request.request_hash,
        "schema_hash": request.schema_hash,
        "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
        "input_tokens": 20,
        "input_ids_sha256": "f" * 64,
        "candidate_token_ids": list(range(101, 101 + len(request.options))),
        "symbol_to_option_id": {
            chr(ord("A") + index): option.id for index, option in enumerate(request.options)
        },
        "record_id": record_id,
    }
    if presentation_id is not None:
        spec["presentation_id"] = presentation_id
    return spec


def _fixture_selection() -> dict[str, object]:
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
            "path": adapter_transfer_contracts._SELECTED_SNAPSHOT_PATH,
            "files_sha256": {
                "adapter_config.json": "5" * 64,
                "adapter_model.safetensors": "6" * 64,
            },
        },
    }


@lru_cache(maxsize=1)
def _payload_fixture() -> dict[str, Any]:
    training_pools = _training_pools()
    schedules = study.build_training_schedules(*training_pools)
    audit = study.audit_training_schedules(schedules, *training_pools)
    routing_development, tool_development = _development_pools()
    new_rows = list(study.build_new_evaluation_presentations(routing_development, tool_development))
    old_rows = _old_retention_rows()
    evaluation = [*old_rows, *new_rows]
    reload = list(study.reload_presentations(new_rows, routing_development, tool_development))
    training_specs = {
        role: [_compiled_spec(example.request, example.record_id) for example in schedules[role]]
        for role in study.ARM_NAMES
    }
    evaluation_specs = [
        _compiled_spec(
            DecisionRequest.model_validate(row["request"]),
            str(row["record_id"]),
            str(row["presentation_id"]),
        )
        for row in evaluation
    ]
    unchanged_specs = deepcopy(evaluation_specs[-len(new_rows) :])
    reload_specs = [
        _compiled_spec(
            DecisionRequest.model_validate(row["request"]),
            str(row["record_id"]),
            str(row["presentation_id"]),
        )
        for row in reload
    ]
    spec_groups = (
        ("training_continued_practice", training_specs[study.CONTINUED_PRACTICE], 1),
        ("training_runtime_mix", training_specs[study.RUNTIME_MIX], 1),
        ("evaluation_both_arms", evaluation_specs, 2),
        ("unchanged_adapter", unchanged_specs, 1),
        ("reload_both_arms", reload_specs, 2),
    )
    token_counts = {
        name: multiplier * sum(int(spec["input_tokens"]) for spec in specs)
        for name, specs, multiplier in spec_groups
    }
    token_counts["total"] = sum(token_counts.values())
    unsigned_compilation: dict[str, object] = {
        "training": training_specs,
        "evaluation": evaluation_specs,
        "unchanged": unchanged_specs,
        "reload": reload_specs,
        "schedule_sha256": dict(audit["schedule_sha256"]),
        "counts": dict(runtime_rule_study_tokens.EXPECTED_COUNTS),
        "input_token_counts": token_counts,
        "max_input_tokens": 20,
    }
    compilation = {
        **unsigned_compilation,
        "compilation_sha256": mixture_training_contracts.json_sha256(unsigned_compilation),
    }

    old_source_hashes = {
        path: hashlib.sha256(path.encode("utf-8")).hexdigest()
        for path in adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS
    }
    selection_path = adapter_transfer_contracts.SELECTION_PATH
    selection_hash = adapter_transfer_contracts.EXPECTED_SELECTION_SHA256
    protocol_path = runtime_rule_study_inputs._STUDY_PROTOCOL_PATH
    protocol_hash = runtime_rule_study_inputs._STUDY_PROTOCOL_SHA256
    summary_path = runtime_rule_study_inputs._TRANSFER_SUMMARY_PATH
    summary_hash = runtime_rule_study_inputs._TRANSFER_SUMMARY_SHA256
    fixed_input_hashes = {
        **mixture_training_contracts.DATA_FILE_SHA256,
        **runtime_rule_study_inputs._NEW_FILE_SHA256,
        **adapter_transfer_contracts.PANEL_FILE_SHA256,
        selection_path: selection_hash,
        protocol_path: protocol_hash,
        summary_path: summary_hash,
    }
    input_hashes = {**old_source_hashes, **fixed_input_hashes}
    source_hashes = {
        **old_source_hashes,
        protocol_path: protocol_hash,
        "pyproject.toml": "7" * 64,
        "uv.lock": "8" * 64,
        "experiments/runtime_rule_study_data.py": "9" * 64,
        "experiments/runtime_rule_study_inputs.py": "a" * 64,
        "experiments/runtime_rule_study_cache.py": "b" * 64,
        "experiments/runtime_rule_study_tokens.py": "c" * 64,
        "experiments/runtime_rule_study_contracts.py": "d" * 64,
        "experiments/runtime_rule_study_bundle.py": "e" * 64,
        "experiments/runtime_rule_study_statistics.py": "f" * 64,
        "experiments/runtime_rule_study_payloads.py": "1" * 64,
        "experiments/runtime_rule_study_payload_rows.py": "2" * 64,
        "experiments/runtime_rule_study_payload_metadata.py": "3" * 64,
    }
    selection = _fixture_selection()
    tokenizer_files = {
        "tokenizer.json": runtime_rule_study_tokens.TOKENIZER_SHA256,
        "tokenizer_config.json": "4" * 64,
        "vocab.json": "5" * 64,
        "merges.txt": "6" * 64,
    }
    runtime_versions = dict(runtime_rule_study_contracts.RUNTIME_VERSION_PINS)
    natural_receipt_path = runtime_rule_study_cache.NATURAL_RECEIPT_PATH
    natural_receipt_hash = runtime_rule_study_cache.NATURAL_RECEIPT_SHA256
    transfer_receipt_path = runtime_rule_study_cache.TRANSFER_RECEIPT_PATH
    transfer_receipt_hash = runtime_rule_study_cache.TRANSFER_RECEIPT_SHA256
    retention_cache = {
        "receipt_file_sha256": {
            natural_receipt_path: natural_receipt_hash,
            transfer_receipt_path: transfer_receipt_hash,
        },
        "outputs_sha256": "5" * 64,
        "cache_sha256": "6" * 64,
        "source_file_sha256": old_source_hashes,
    }
    unsigned_bundle: dict[str, object] = {
        "schema_version": runtime_rule_study_contracts.SCHEMA_VERSION,
        "experiment_id": runtime_rule_study_contracts.EXPERIMENT_ID,
        "protocol_sha256": runtime_rule_study_contracts.PROTOCOL_SHA256,
        "model_id": runtime_rule_study_contracts.MODEL_ID,
        "model_revision": runtime_rule_study_contracts.MODEL_REVISION,
        "selection": selection,
        "selected_tensor_sha256": runtime_rule_study_contracts.SELECTED_TENSOR_SHA256,
        "tokenizer_file_sha256": tokenizer_files,
        "runtime_versions": runtime_versions,
        "input_file_sha256": input_hashes,
        "retention_cache": retention_cache,
        "training_pools": [
            [record.model_dump(mode="json") for record in pool] for pool in training_pools
        ],
        "evaluation": evaluation,
        "unchanged": new_rows,
        "reload": reload,
        "compilation": compilation,
    }
    bundle = {
        **unsigned_bundle,
        "bundle_sha256": mixture_training_contracts.json_sha256(unsigned_bundle),
    }
    return {
        "bundle": bundle,
        "bundle_sha256": bundle["bundle_sha256"],
        "source_file_sha256": source_hashes,
        "source_commit": "e" * 40,
        "study_id": "toy-study",
        "nonce": "00000000-0000-4000-8000-000000000001",
        "schedules": schedules,
    }


@pytest.fixture(autouse=True)
def _toy_selection_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_hash = mixture_training_contracts.json_sha256(_fixture_selection())
    monkeypatch.setattr(
        adapter_transfer_contracts, "EXPECTED_CANONICAL_SELECTION_SHA256", fixture_hash
    )


@pytest.fixture(scope="module")
def payload_fixture() -> dict[str, Any]:
    return _payload_fixture()


def _build_payload(fixture: dict[str, Any], role: str) -> dict[str, object]:
    return build_worker_payload(
        deepcopy(fixture["bundle"]),
        expected_bundle_sha256=fixture["bundle_sha256"],
        study_id=fixture["study_id"],
        nonce=fixture["nonce"],
        role=role,
        source_commit=fixture["source_commit"],
        source_file_sha256=deepcopy(fixture["source_file_sha256"]),
    )


def _resign_compilation_and_bundle(bundle: dict[str, Any]) -> str:
    compilation = bundle["compilation"]
    unsigned_compilation = {
        key: value for key, value in compilation.items() if key != "compilation_sha256"
    }
    compilation["compilation_sha256"] = mixture_training_contracts.json_sha256(unsigned_compilation)
    unsigned_bundle = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    bundle["bundle_sha256"] = mixture_training_contracts.json_sha256(unsigned_bundle)
    return str(bundle["bundle_sha256"])


def _resign_payload(payload: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = mixture_training_contracts.json_sha256(unsigned)


def test_unchanged_payload_contains_only_new_development_requests(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, "unchanged")

    assert payload["role"] == "unchanged"
    assert payload["run_id"] == f"{payload_fixture['study_id']}-unchanged"
    assert payload["training_pools"] == []
    assert payload["training_specs"] == []
    assert payload["optimizer"] is None
    assert payload["reload"] == []
    assert payload["reload_specs"] == []
    assert len(payload["evaluation"]) == len(payload["evaluation_specs"]) == 164
    assert {row["dataset_id"] for row in payload["evaluation"]} == {
        study.NEW_DEVELOPMENT_DATASET_IDS["routing"],
        study.NEW_DEVELOPMENT_DATASET_IDS["tool"],
    }
    assert all("answer_id" not in row for row in payload["evaluation"])
    assert all("answer_id" not in row["request"] for row in payload["evaluation"])


@pytest.mark.parametrize("role", [study.CONTINUED_PRACTICE, study.RUNTIME_MIX])
def test_training_payload_contains_only_its_arm_and_fixed_panels(
    payload_fixture: dict[str, Any], role: str
) -> None:
    payload = _build_payload(payload_fixture, role)

    assert payload["role"] == role
    expected_suffix = "-control" if role == study.CONTINUED_PRACTICE else "-runtime"
    assert payload["run_id"] == f"{payload_fixture['study_id']}{expected_suffix}"
    assert len(payload["training_pools"]) == 5
    assert len(payload["training_specs"]) == 1_344
    assert len(payload["evaluation"]) == len(payload["evaluation_specs"]) == 3_946
    assert len(payload["reload"]) == len(payload["reload_specs"]) == 32
    assert payload["optimizer"] == runtime_rule_study_contracts.optimizer_settings()
    examples = runtime_rule_study_payloads.training_examples(payload)
    assert len(examples) == study.TRAINING_PRESENTATIONS
    assert [example.record_id for example in examples[:4]] == [
        spec["record_id"] for spec in payload["training_specs"][:4]
    ]


def test_payload_accepts_historical_numeric_selection_two_and_sixteen_option_shapes(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, study.RUNTIME_MIX)
    numeric_rows = [
        row
        for row in payload["evaluation"]
        if row["dataset_id"] == "synthetic-numeric-selection-v1-development"
    ]
    numeric_record_sizes = {
        row["record_id"]: len(row["order_ids"]) for row in numeric_rows if row["order_index"] == 0
    }

    assert len(numeric_rows) == 364
    assert Counter(numeric_record_sizes.values()) == Counter({2: 14, 4: 12, 8: 12, 16: 12})


def test_builder_returns_detached_json_values(payload_fixture: dict[str, Any]) -> None:
    fixture_copy = deepcopy(payload_fixture)
    payload = _build_payload(fixture_copy, "unchanged")
    original_context = payload["evaluation"][0]["request"]["context"]

    fixture_copy["bundle"]["unchanged"][0]["request"]["context"] = "mutated input"
    assert payload["evaluation"][0]["request"]["context"] == original_context
    payload["evaluation"][0]["request"]["context"] = "mutated output"
    clean_payload = _build_payload(payload_fixture, "unchanged")
    assert clean_payload["evaluation"][0]["request"]["context"] == original_context


def test_payload_digest_can_be_checked_against_a_trusted_value(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, "unchanged")

    with pytest.raises(ValueError, match="trusted expected digest"):
        runtime_rule_study_payloads.validate_payload(payload, expected_payload_sha256="0" * 64)


def test_rehashed_payload_still_rejects_answer_bearing_evaluation_request(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, "unchanged")
    payload["evaluation"][0]["request"]["answer_id"] = "leaked-label"
    _resign_payload(payload)

    with pytest.raises(ValueError, match="only context, question, and options"):
        runtime_rule_study_payloads.validate_payload(payload)


def test_rehashed_payload_rejects_compiled_request_identity_drift(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, "unchanged")
    payload["evaluation_specs"][0]["prompt_sha256"] = "0" * 64
    _resign_payload(payload)

    with pytest.raises(ValueError, match="identity differs from its source row"):
        runtime_rule_study_payloads.validate_payload(payload)


def test_rehashed_payload_rejects_run_id_suffix_drift(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, "unchanged")
    payload["run_id"] = "some-other-run"
    _resign_payload(payload)

    with pytest.raises(ValueError, match="fixed role suffix"):
        runtime_rule_study_payloads.validate_payload(payload)


def test_rehashed_payload_rejects_numeric_aliases_in_fixed_contracts(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, study.RUNTIME_MIX)
    payload["worker_settings"]["retries"] = False
    payload["optimizer"]["max_gradient_norm"] = True
    _resign_payload(payload)

    with pytest.raises(ValueError, match="worker settings"):
        runtime_rule_study_payloads.validate_payload(payload)


def test_rehashed_payload_rejects_optimizer_numeric_alias(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, study.RUNTIME_MIX)
    payload["optimizer"]["max_gradient_norm"] = True
    _resign_payload(payload)

    with pytest.raises(ValueError, match="optimizer"):
        runtime_rule_study_payloads.validate_payload(payload)


def test_payload_rows_reject_optimizer_numeric_alias_with_canonical_comparison(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, study.RUNTIME_MIX)
    payload["optimizer"]["max_gradient_norm"] = True

    with pytest.raises(ValueError, match="optimizer"):
        runtime_rule_study_payload_rows.validate_role_rows(
            payload, study.RUNTIME_MIX, payload["study"]["schedule_sha256"]
        )


def test_rehashed_payload_rejects_boolean_forward_count(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, "unchanged")
    payload["forward_counts"]["final_evaluation"] = True
    _resign_payload(payload)

    with pytest.raises(ValueError, match="forward counts"):
        runtime_rule_study_payloads.validate_payload(payload)


def test_rehashed_payload_rejects_boolean_zero_forward_count(
    payload_fixture: dict[str, Any],
) -> None:
    payload = _build_payload(payload_fixture, "unchanged")
    payload["forward_counts"]["training"] = False
    _resign_payload(payload)

    with pytest.raises(ValueError, match="strict integers"):
        runtime_rule_study_payloads.validate_payload(payload)


def test_training_builder_rejects_regenerated_schedule_drift(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    selected_record_id = payload_fixture["schedules"][study.CONTINUED_PRACTICE][0].record_id
    first = next(
        row
        for pool in bundle["training_pools"]
        for row in pool
        if row["record_id"] == selected_record_id
    )
    first["answer_id"] = next(
        option["id"] for option in first["request"]["options"] if option["id"] != first["answer_id"]
    )
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="regenerated schedules differ"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role=study.CONTINUED_PRACTICE,
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_rejects_rehashed_evaluation_task_drift(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    bundle["evaluation"][0]["dataset_id"] = study.NEW_DEVELOPMENT_DATASET_IDS["routing"]
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="retention evaluation record identity"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role=study.RUNTIME_MIX,
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_rejects_rehashed_retention_task_interleaving(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    evaluation = bundle["evaluation"]
    first_record_id = evaluation[0]["record_id"]
    moved_group = [row for row in evaluation if row["record_id"] == first_record_id]
    del evaluation[: len(moved_group)]
    first_sms_index = next(
        index
        for index, row in enumerate(evaluation)
        if row["dataset_id"] == "sms-pilot-v1-development"
    )
    first_sms_record_id = evaluation[first_sms_index]["record_id"]
    insert_at = first_sms_index
    while evaluation[insert_at]["record_id"] == first_sms_record_id:
        insert_at += 1
    evaluation[insert_at:insert_at] = moved_group
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="retention task order or grouping"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role=study.CONTINUED_PRACTICE,
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_rejects_cross_panel_duplicate_presentation_id(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    duplicate_id = bundle["evaluation"][3_782]["presentation_id"]
    bundle["evaluation"][0]["presentation_id"] = duplicate_id
    bundle["compilation"]["evaluation"][0]["presentation_id"] = duplicate_id
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="presentation IDs must be unique across the full panel"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role=study.RUNTIME_MIX,
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_rejects_cross_panel_duplicate_record_id(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    evaluation = bundle["evaluation"]
    original_record_id = evaluation[3_782]["record_id"]
    retained_record_id = evaluation[0]["record_id"]
    presentation_updates: dict[str, tuple[str, str]] = {}
    for row in evaluation[3_782:]:
        if row["record_id"] != original_record_id:
            continue
        old_presentation_id = row["presentation_id"]
        row["record_id"] = retained_record_id
        row["presentation_id"] = runtime_rule_study_payload_rows._new_presentation_id(row)
        presentation_updates[old_presentation_id] = (
            row["presentation_id"],
            retained_record_id,
        )

    for collection_name in ("evaluation", "unchanged", "reload"):
        for row in bundle[collection_name]:
            update = presentation_updates.get(row["presentation_id"])
            if update is not None:
                row["presentation_id"], row["record_id"] = update
    for collection_name in ("evaluation", "unchanged", "reload"):
        for spec in bundle["compilation"][collection_name]:
            update = presentation_updates.get(spec.get("presentation_id"))
            if update is not None:
                spec["presentation_id"], spec["record_id"] = update

    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="record IDs overlap between retention and new panels"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role=study.RUNTIME_MIX,
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_unchanged_builder_rejects_rehashed_new_family_interleaving(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    new_rows = bundle["evaluation"][-164:]
    first_routing_record = new_rows[0]["record_id"]
    moved_group = [row for row in new_rows if row["record_id"] == first_routing_record]
    new_rows = [row for row in new_rows if row["record_id"] != first_routing_record]
    first_tool_index = next(
        index
        for index, row in enumerate(new_rows)
        if row["dataset_id"] == study.NEW_DEVELOPMENT_DATASET_IDS["tool"]
    )
    first_tool_record = new_rows[first_tool_index]["record_id"]
    insert_at = first_tool_index
    while new_rows[insert_at]["record_id"] == first_tool_record:
        insert_at += 1
    new_rows[insert_at:insert_at] = moved_group
    bundle["evaluation"][-164:] = new_rows
    bundle["unchanged"] = deepcopy(new_rows)
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="new development task order"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role="unchanged",
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_rejects_rehashed_candidate_token_collision(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    first_spec = bundle["compilation"]["unchanged"][0]
    first_spec["candidate_token_ids"][1] = first_spec["candidate_token_ids"][0]
    bundle["compilation"]["evaluation"][-164]["candidate_token_ids"][1] = first_spec[
        "candidate_token_ids"
    ][0]
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="candidate token IDs"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role="unchanged",
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_rejects_rehashed_missing_tokenizer_file(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    del bundle["tokenizer_file_sha256"]["vocab.json"]
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="exact four tokenizer files"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role="unchanged",
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_rejects_rehashed_extra_tokenizer_file(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    bundle["tokenizer_file_sha256"]["unexpected.json"] = "7" * 64
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="exact four tokenizer files"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role="unchanged",
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_validates_every_rehashed_tokenizer_digest(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    bundle["tokenizer_file_sha256"]["merges.txt"] = "not-a-sha256"
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="tokenizer_file_sha256 merges.txt must be a lowercase"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role="unchanged",
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_training_builder_rejects_rehashed_reload_identity_drift(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    bundle["reload"][0]["presentation_id"] = "incorrect-reload-presentation"
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="reload parity rows differ"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role=study.RUNTIME_MIX,
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_training_builder_rejects_rehashed_reload_spec_drift(
    payload_fixture: dict[str, Any],
) -> None:
    bundle = deepcopy(payload_fixture["bundle"])
    bundle["compilation"]["reload"][0]["candidate_token_ids"][0] = 999_999
    trusted_digest = _resign_compilation_and_bundle(bundle)

    with pytest.raises(ValueError, match="reload specs do not match the selected evaluation specs"):
        build_worker_payload(
            bundle,
            expected_bundle_sha256=trusted_digest,
            study_id=payload_fixture["study_id"],
            nonce=payload_fixture["nonce"],
            role=study.RUNTIME_MIX,
            source_commit=payload_fixture["source_commit"],
            source_file_sha256=payload_fixture["source_file_sha256"],
        )


def test_payload_builder_checks_uploaded_historical_source_hashes(
    payload_fixture: dict[str, Any],
) -> None:
    source_hashes = deepcopy(payload_fixture["source_file_sha256"])
    path = adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS[0]
    source_hashes[path] = "0" * 64

    with pytest.raises(ValueError, match="upload and retention source hashes disagree"):
        _build_payload({**payload_fixture, "source_file_sha256": source_hashes}, "unchanged")


def test_payload_accepts_structurally_valid_future_runtime_source(
    payload_fixture: dict[str, Any],
) -> None:
    source_hashes = deepcopy(payload_fixture["source_file_sha256"])
    source_hashes["experiments/runtime_rule_study_payload_results.py"] = "d" * 64

    payload = _build_payload({**payload_fixture, "source_file_sha256": source_hashes}, "unchanged")

    assert payload["source_file_sha256"]["experiments/runtime_rule_study_payload_results.py"] == (
        "d" * 64
    )


def test_payload_source_allowlist_rejects_data_and_missing_minimum_paths(
    payload_fixture: dict[str, Any],
) -> None:
    source_hashes = deepcopy(payload_fixture["source_file_sha256"])
    source_hashes["data/training/secret.json"] = "d" * 64
    with pytest.raises(ValueError, match="source path is outside the fixed upload file classes"):
        _build_payload({**payload_fixture, "source_file_sha256": source_hashes}, "unchanged")

    source_hashes = deepcopy(payload_fixture["source_file_sha256"])
    del source_hashes["experiments/runtime_rule_study_bundle.py"]
    with pytest.raises(ValueError, match="omits required historical or study source paths"):
        _build_payload({**payload_fixture, "source_file_sha256": source_hashes}, "unchanged")


def test_builder_rejects_unknown_worker_role() -> None:
    with pytest.raises(ValueError, match="role"):
        build_worker_payload(
            {},
            expected_bundle_sha256="a" * 64,
            study_id="toy-study",
            nonce="00000000-0000-4000-8000-000000000001",
            role="unknown",
            source_commit="b" * 40,
            source_file_sha256={},
        )


def test_builder_rejects_abbreviated_source_commit_before_bundle_verification() -> None:
    with pytest.raises(ValueError, match="full lowercase 40-character Git SHA"):
        build_worker_payload(
            {},
            expected_bundle_sha256="a" * 64,
            study_id="toy-study",
            nonce="00000000-0000-4000-8000-000000000001",
            role="unchanged",
            source_commit="b" * 39,
            source_file_sha256={},
        )


def test_builder_rejects_parent_traversal_source_path_before_bundle_verification() -> None:
    with pytest.raises(ValueError, match="source path is unsafe"):
        build_worker_payload(
            {},
            expected_bundle_sha256="a" * 64,
            study_id="toy-study",
            nonce="00000000-0000-4000-8000-000000000001",
            role="unchanged",
            source_commit="b" * 40,
            source_file_sha256={"experiments/../secret.py": "c" * 64},
        )


def test_builder_rejects_unsafe_study_id_before_bundle_verification() -> None:
    with pytest.raises(ValueError, match="study_id must be a safe run ID"):
        build_worker_payload(
            {},
            expected_bundle_sha256="a" * 64,
            study_id="unsafe/study",
            nonce="00000000-0000-4000-8000-000000000001",
            role="unchanged",
            source_commit="b" * 40,
            source_file_sha256={},
        )


def test_builder_rejects_study_id_that_overflows_derived_run_id() -> None:
    with pytest.raises(ValueError, match="run_id must be a safe run ID"):
        build_worker_payload(
            {},
            expected_bundle_sha256="a" * 64,
            study_id="s" * 55,
            nonce="00000000-0000-4000-8000-000000000001",
            role="unchanged",
            source_commit="b" * 40,
            source_file_sha256={},
        )


def test_builder_rejects_noncanonical_nonce_before_bundle_verification() -> None:
    with pytest.raises(ValueError, match="nonce must be a canonical UUID"):
        build_worker_payload(
            {},
            expected_bundle_sha256="a" * 64,
            study_id="toy-study",
            nonce="00000000000040008000000000000001",
            role="unchanged",
            source_commit="b" * 40,
            source_file_sha256={},
        )


def test_builder_requires_trusted_bundle_sha256() -> None:
    with pytest.raises(ValueError, match="expected bundle SHA-256 must be a lowercase"):
        build_worker_payload(
            {},
            expected_bundle_sha256="A" * 64,
            study_id="toy-study",
            nonce="00000000-0000-4000-8000-000000000001",
            role="unchanged",
            source_commit="b" * 40,
            source_file_sha256={},
        )
