import sys
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256

import pytest

from experiments import real_pilot_core as core
from reflex_decisions.data import DatasetSpec, DecisionRecord, SplitManifest
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def _training_records() -> tuple[DecisionRecord, ...]:
    records = []
    for index in range(504):
        dataset_id = "dbpedia14-pilot-v1-train" if index < 252 else "sms-pilot-v1-train"
        if dataset_id.startswith("dbpedia14-"):
            options = tuple(
                Option(id=f"topic-{topic}", label=f"Topic {topic}") for topic in range(14)
            )
        else:
            options = (Option(id="ham", label="Ham"), Option(id="spam", label="Spam"))
        answer_id = options[index % 2].id
        records.append(
            DecisionRecord(
                record_id=f"train-{index:03d}",
                dataset_id=dataset_id,
                source_group_id=f"group-{index:03d}",
                request=DecisionRequest(
                    context=f"Synthetic training example {index}",
                    question="Which label applies?",
                    options=options,
                ),
                answer_id=answer_id,
            )
        )
    return tuple(records)


def _evaluation_records() -> tuple[DecisionRecord, ...]:
    option_counts = {
        "dbpedia14-pilot-v1-development": 14,
        "dbpedia14-pilot-v1-calibration": 14,
        "sms-pilot-v1-development": 2,
        "sms-pilot-v1-calibration": 2,
        "snli-pilot-v1-development": 3,
    }
    record_counts = {
        "dbpedia14-pilot-v1-development": 56,
        "dbpedia14-pilot-v1-calibration": 56,
        "sms-pilot-v1-development": 60,
        "sms-pilot-v1-calibration": 60,
        "snli-pilot-v1-development": 128,
    }
    records = []
    for dataset_id, count in record_counts.items():
        options = tuple(
            Option(id=f"option-{index}", label=f"Option {index}")
            for index in range(option_counts[dataset_id])
        )
        for index in range(count):
            records.append(
                DecisionRecord(
                    record_id=f"{dataset_id}-record-{index:03d}",
                    dataset_id=dataset_id,
                    source_group_id=f"{dataset_id}-group-{index:03d}",
                    request=DecisionRequest(
                        context=f"Synthetic evaluation example {dataset_id}-{index}",
                        question="Which option applies?",
                        options=options,
                    ),
                    answer_id=options[index % len(options)].id,
                )
            )
    return tuple(records)


def _pilot_manifest() -> SplitManifest:
    datasets = []
    split_by_dataset = {
        "dbpedia14-pilot-v1-train": ("fancyzhx/dbpedia_14", "train", "topic-classification"),
        "dbpedia14-pilot-v1-development": (
            "fancyzhx/dbpedia_14",
            "development",
            "topic-classification",
        ),
        "dbpedia14-pilot-v1-calibration": (
            "fancyzhx/dbpedia_14",
            "calibration",
            "topic-classification",
        ),
        "sms-pilot-v1-train": (
            "uci-sms-spam-collection-228",
            "train",
            "spam-detection",
        ),
        "sms-pilot-v1-development": (
            "uci-sms-spam-collection-228",
            "development",
            "spam-detection",
        ),
        "sms-pilot-v1-calibration": (
            "uci-sms-spam-collection-228",
            "calibration",
            "spam-detection",
        ),
        "snli-pilot-v1-development": (
            "stanford-snli-1.0",
            "development",
            "natural-language-inference-three-way",
        ),
    }
    for dataset_id, (source_id, split, task_family) in split_by_dataset.items():
        datasets.append(
            DatasetSpec(
                dataset_id=dataset_id,
                source_id=source_id,
                task_family=task_family,
                split=split,
                source_uri=f"local://fixture/{dataset_id}",
                source_revision="fixture-v1",
                license="fixture license",
            )
        )
    return SplitManifest(
        data_kind="benchmark",
        group_partitioned_sources=(
            "fancyzhx/dbpedia_14",
            "uci-sms-spam-collection-228",
        ),
        datasets=tuple(datasets),
    )


def test_training_schedule_is_deterministic_two_epochs_and_remaps_gold() -> None:
    records = _training_records()

    schedule = core.build_training_schedule(records)

    assert schedule == core.build_training_schedule(records)
    assert len(schedule) == 1008
    assert Counter(example.record_id for example in schedule) == {
        record.record_id: 2 for record in records
    }
    assert all(
        example.request.options[example.gold_index].id == example.gold_option_id
        for example in schedule
    )


def test_evaluation_panel_has_fixed_counts_and_balanced_option_positions() -> None:
    records = _training_records() + _evaluation_records()

    presentations = core.build_evaluation_presentations(records)

    assert len(presentations) == 2572
    assert len({row["presentation_id"] for row in presentations}) == 2572
    presentations_per_record = Counter(row["record_id"] for row in presentations)
    dataset_multiplicity = {
        "dbpedia14-pilot-v1-development": 28,
        "sms-pilot-v1-development": 2,
        "dbpedia14-pilot-v1-calibration": 1,
        "sms-pilot-v1-calibration": 1,
        "snli-pilot-v1-development": 6,
    }
    for record in _evaluation_records():
        assert presentations_per_record[record.record_id] == dataset_multiplicity[record.dataset_id]

    dbpedia_orders = [
        row["order_ids"]
        for row in presentations
        if row["dataset_id"] == "dbpedia14-pilot-v1-development"
        and row["record_id"].endswith("record-000")
    ]
    assert len(dbpedia_orders) == len({tuple(order) for order in dbpedia_orders}) == 28
    by_position = defaultdict(Counter)
    for order in dbpedia_orders:
        for position, option_id in enumerate(order):
            by_position[position][option_id] += 1
    assert all(set(counts.values()) == {2} for counts in by_position.values())

    calibration = [row for row in presentations if "-calibration" in row["dataset_id"]]
    assert len(calibration) == 116
    assert all(row["order_index"] == 0 for row in calibration)
    assert all("answer_id" not in row and "gold_option_id" not in row for row in presentations)
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
        for row in presentations
    )


def test_pilot_data_contract_has_two_train_tasks_and_snli_development() -> None:
    manifest = _pilot_manifest()
    records = _training_records() + _evaluation_records()

    audit = core.validate_pilot_data(manifest, records)

    assert audit.record_count == 864
    assert dict(audit.split_counts) == {
        "train": 504,
        "development": 244,
        "calibration": 116,
        "test": 0,
    }

    with pytest.raises(ValueError, match="source-group split policy"):
        core.validate_pilot_data(
            manifest.model_copy(update={"group_partitioned_sources": ()}), records
        )


def test_remote_payload_separates_train_gold_from_evaluation_and_pins_identity() -> None:
    manifest = _pilot_manifest()
    records = _training_records() + _evaluation_records()
    nonce = "8d4bfbb3-778e-4894-8f18-d4dd7f1eb14b"
    source_hashes = {path: "a" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[core.PROTOCOL_PATH] = "e" * 64
    payload = core.build_remote_payload(
        manifest,
        records,
        run_id="cpu-contract-test",
        nonce=nonce,
        records_sha256="b" * 64,
        manifest_sha256="c" * 64,
        recipe_sha256="d" * 64,
        protocol_sha256="e" * 64,
        source_file_sha256=source_hashes,
    )

    assert len(payload["train_records"]) == 504
    assert len(payload["evaluation_presentations"]) == 2572
    assert all("answer_id" in row for row in payload["train_records"])
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
        for row in payload["evaluation_presentations"]
    )
    assert (
        core.validate_remote_payload(
            payload,
            expected_identity={
                "run_id": "cpu-contract-test",
                "nonce": nonce,
                "pins": payload["pins"],
            },
        )
        == payload
    )

    labeled_evaluation = deepcopy(payload)
    labeled_evaluation["evaluation_presentations"][0]["answer_id"] = "hidden"
    with pytest.raises(ValueError, match="request-only"):
        core.validate_remote_payload(labeled_evaluation)

    stale_nonce = deepcopy(payload)
    stale_nonce["nonce"] = "d349bf85-9c89-4cbc-8c06-47a8d83291c4"
    with pytest.raises(ValueError, match="identity or pins"):
        core.validate_remote_payload(
            stale_nonce,
            expected_identity={
                "run_id": "cpu-contract-test",
                "nonce": nonce,
                "pins": payload["pins"],
            },
        )


def _valid_receipt(payload: dict[str, object]) -> dict[str, object]:
    evaluation = payload["evaluation_presentations"]
    outputs = []
    for row in evaluation:
        order_ids = row["order_ids"]
        logits = [float(index) for index in range(len(order_ids))]
        outputs.append(
            {
                "presentation_id": row["presentation_id"],
                "record_id": row["record_id"],
                "dataset_id": row["dataset_id"],
                "request_hash": row["request_hash"],
                "order_ids": order_ids,
                "candidate_logits": logits,
                "winner_option_id": order_ids[-1],
                "input_tokens": 17,
                "prompt_sha256": sha256(
                    render_prompt(DecisionRequest.model_validate(row["request"])).encode("utf-8")
                ).hexdigest(),
            }
        )
    by_id = {row["presentation_id"]: row for row in outputs}
    parity_ids = sorted(by_id)[: core.RELOAD_PARITY_COUNT]
    parity_logits = [
        list(by_id[presentation_id]["candidate_logits"]) for presentation_id in parity_ids
    ]
    run_dir = f"/artifacts/runs/{payload['run_id']}"
    return {
        "schema_version": 1,
        "status": "passed",
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "pins": payload["pins"],
        "provenance": {
            "model_id": core.MODEL_ID,
            "model_revision": core.MODEL_REVISION,
            "source_file_sha256": payload["pins"]["source_file_sha256"],
            "measured_source_file_sha256": payload["pins"]["source_file_sha256"],
        },
        "evidence": {
            "optimizer_updates_completed": core.MAX_UPDATES,
            "forward_counts": {
                "base_evaluation": core.BASE_EVALUATION_COUNT,
                "training": core.TRAIN_FORWARD_COUNT,
                "final_evaluation": core.FINAL_EVALUATION_COUNT,
                "reload_parity": core.RELOAD_PARITY_COUNT,
                "total": core.MAX_FORWARD_COUNT,
            },
            "base_gradients_none": True,
            "adapter_inventory": {
                "module_count": 60,
                "adapter_tensor_count": 120,
                "trainable_parameter_count": 2_015_232,
                "base_parameters_frozen_bf16": True,
            },
            "training_step_losses": [
                {"update": update, "mean_loss": 1.0, "lo_ra_b_gradient_l1": 0.5}
                for update in range(1, core.MAX_UPDATES + 1)
            ],
            "adapter_update": {
                "changed_tensor_count": 1,
                "changed_tensor_names": ["base_model.layer.0.lora_A.default.weight"],
            },
            "adapter_paths": [
                {
                    "update": update,
                    "path": f"{run_dir}/adapter-update-{update:03d}",
                    "files_sha256": {
                        "adapter_model.safetensors": "a" * 64,
                        "adapter_config.json": "b" * 64,
                    },
                }
                for update in (126, 252)
            ],
            "base_outputs": outputs,
            "final_outputs": outputs,
            "output_artifacts": {
                name: {"path": f"{run_dir}/{name}.json", "sha256": "c" * 64}
                for name in ("base_outputs", "final_outputs")
            },
            "reload_parity": {
                "presentation_ids": parity_ids,
                "reloaded_logits": parity_logits,
                "adapter_tensor_keys_shapes_values_match": True,
                "winner_match": True,
                "max_abs_logit_diff": 0.0,
            },
        },
    }


def test_passed_receipt_requires_complete_outputs_and_reload_parity() -> None:
    import ast
    import inspect

    from experiments import modal_real_pilot

    manifest = _pilot_manifest()
    records = _training_records() + _evaluation_records()
    nonce = "8d4bfbb3-778e-4894-8f18-d4dd7f1eb14b"
    source_hashes = {path: "a" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[core.PROTOCOL_PATH] = "e" * 64
    payload = core.build_remote_payload(
        manifest,
        records,
        run_id="receipt-contract-test",
        nonce=nonce,
        records_sha256="b" * 64,
        manifest_sha256="c" * 64,
        recipe_sha256="d" * 64,
        protocol_sha256="e" * 64,
        source_file_sha256=source_hashes,
    )

    receipt = _valid_receipt(payload)
    worker_ast = ast.parse(inspect.getsource(modal_real_pilot._remote_train))
    produced_parity_keys = {
        key.value
        for node in ast.walk(worker_ast)
        if isinstance(node, ast.Dict)
        for parent_key, value in zip(node.keys, node.values, strict=True)
        if isinstance(parent_key, ast.Constant)
        and parent_key.value == "reload_parity"
        and isinstance(value, ast.Dict)
        for key in value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert produced_parity_keys == set(receipt["evidence"]["reload_parity"])
    assert core.validate_passed_receipt(receipt, expected_payload=payload) == receipt

    incomplete = deepcopy(receipt)
    incomplete["evidence"]["final_outputs"].pop()
    with pytest.raises(ValueError, match="every evaluation presentation"):
        core.validate_passed_receipt(incomplete, expected_payload=payload)

    changed_parity = deepcopy(receipt)
    changed_parity["evidence"]["reload_parity"]["reloaded_logits"][0][0] = 0.1
    with pytest.raises(ValueError, match="winner|difference"):
        core.validate_passed_receipt(changed_parity, expected_payload=payload)


def test_recovery_receipt_is_strict_and_stale_nonce_is_rejected() -> None:
    pins = {
        "records_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "recipe_sha256": "d" * 64,
        "protocol_sha256": "e" * 64,
        "source_file_sha256": {
            **{path: "a" * 64 for path in core.SOURCE_FINGERPRINT_PATHS},
            core.PROTOCOL_PATH: "e" * 64,
        },
    }
    nonce = "8d4bfbb3-778e-4894-8f18-d4dd7f1eb14b"
    identity = {"run_id": "recovery-contract-test", "nonce": nonce, "pins": pins}
    progress = {
        "schema_version": 1,
        **identity,
        "status": "running",
        "provenance": {"measured_source_file_sha256": pins["source_file_sha256"]},
        "evidence": {},
    }

    recovered = core.validate_recovery_receipt(progress, expected_identity=identity)

    assert recovered["status"] == "failed"
    assert recovered["phase"] == "recovered_after_remote_exception"
    stale_identity = {**identity, "nonce": "d349bf85-9c89-4cbc-8c06-47a8d83291c4"}
    with pytest.raises(ValueError, match="identity or pins"):
        core.validate_recovery_receipt(progress, expected_identity=stale_identity)


def test_failed_receipt_requires_launch_identity_and_measured_source_map() -> None:
    manifest = _pilot_manifest()
    records = _training_records() + _evaluation_records()
    nonce = "8d4bfbb3-778e-4894-8f18-d4dd7f1eb14b"
    source_hashes = {path: "a" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[core.PROTOCOL_PATH] = "e" * 64
    payload = core.build_remote_payload(
        manifest,
        records,
        run_id="failed-receipt-contract-test",
        nonce=nonce,
        records_sha256="b" * 64,
        manifest_sha256="c" * 64,
        recipe_sha256="d" * 64,
        protocol_sha256="e" * 64,
        source_file_sha256=source_hashes,
    )
    receipt = {
        "schema_version": 1,
        "status": "failed",
        "run_id": payload["run_id"],
        "nonce": nonce,
        "pins": payload["pins"],
        "phase": "training_update_8",
        "failure": {
            "stage": "training",
            "type": "RuntimeError",
            "message": "remote failure with partial evidence",
        },
        "provenance": {
            "model_id": core.MODEL_ID,
            "model_revision": core.MODEL_REVISION,
            "source_file_sha256": source_hashes,
            "measured_source_file_sha256": source_hashes,
        },
        "evidence": {"optimizer_updates_completed": 8, "training_step_losses": []},
        "limits": core.MODAL_LIMITS,
    }

    preserved = core.validate_failed_receipt(receipt, expected_payload=payload)

    assert preserved["status"] == "failed"
    assert preserved["evidence"] == receipt["evidence"]
    assert preserved["failure"] == receipt["failure"]

    stale_source = deepcopy(receipt)
    stale_source["provenance"]["measured_source_file_sha256"][core.PROTOCOL_PATH] = "f" * 64
    with pytest.raises(ValueError, match="source provenance"):
        core.validate_failed_receipt(stale_source, expected_payload=payload)

    missing_identity = deepcopy(receipt)
    del missing_identity["nonce"]
    with pytest.raises(ValueError, match="unexpected schema"):
        core.validate_failed_receipt(missing_identity, expected_payload=payload)


def test_worker_collision_does_not_overwrite_another_runs_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    import builtins

    from experiments import modal_real_pilot

    manifest = _pilot_manifest()
    records = _training_records() + _evaluation_records()
    nonce = "8d4bfbb3-778e-4894-8f18-d4dd7f1eb14b"
    source_hashes = {path: "a" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[core.PROTOCOL_PATH] = "e" * 64
    payload = core.build_remote_payload(
        manifest,
        records,
        run_id="existing-run",
        nonce=nonce,
        records_sha256="b" * 64,
        manifest_sha256="c" * 64,
        recipe_sha256="d" * 64,
        protocol_sha256="e" * 64,
        source_file_sha256=source_hashes,
    )
    run_dir = tmp_path / "runs" / "existing-run"
    run_dir.mkdir(parents=True)
    marker = run_dir / "progress.json"
    marker.write_text('{"nonce":"prior-attempt"}\n', encoding="utf-8")
    writes: list[object] = []
    commits: list[object] = []
    monkeypatch.setattr(modal_real_pilot, "VOLUME_ROOT", tmp_path)
    monkeypatch.setattr(
        modal_real_pilot,
        "_measure_remote_source_fingerprints",
        lambda: source_hashes,
    )
    monkeypatch.setattr(
        modal_real_pilot.modal_train_rehearsal,
        "_write_progress",
        lambda *args, **kwargs: writes.append(args),
    )
    monkeypatch.setattr(
        modal_real_pilot.modal_train_rehearsal,
        "_commit_volume",
        lambda: commits.append(True),
    )
    original_import = builtins.__import__
    torch_imports: list[str] = []

    def track_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            torch_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_torch_import)

    result = modal_real_pilot._remote_train(payload)

    assert result["status"] == "failed"
    assert marker.read_text(encoding="utf-8") == '{"nonce":"prior-attempt"}\n'
    assert writes == []
    assert commits == []
    assert torch_imports == []


def test_launch_preserves_authenticated_remote_failure_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    from experiments import modal_real_pilot

    manifest = _pilot_manifest()
    records = _training_records() + _evaluation_records()
    run_id = "remote-failure-test"
    source_hashes = {path: "a" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[core.PROTOCOL_PATH] = "e" * 64
    files = {
        "records": tmp_path / "records.jsonl",
        "manifest": tmp_path / "manifest.json",
        "recipe": tmp_path / "recipe.json",
    }
    for path in files.values():
        path.write_text("fixture", encoding="utf-8")

    partial_evidence = {
        "optimizer_updates_completed": 16,
        "forward_counts": {"training": 64, "total": 2636},
        "training_step_losses": [{"update": 16, "mean_loss": 0.75, "lo_ra_b_gradient_l1": 0.125}],
    }
    remote_failure = {
        "schema_version": 1,
        "status": "failed",
        "run_id": run_id,
        "nonce": "",
        "pins": {},
        "phase": "training_update_16",
        "failure": {
            "stage": "training",
            "type": "RuntimeError",
            "message": "remote failure with partial evidence",
        },
        "provenance": {},
        "evidence": partial_evidence,
        "limits": core.MODAL_LIMITS,
    }

    class FakeImage:
        def env(self, *_args, **_kwargs):
            return self

        def pip_install(self, *_args, **_kwargs):
            return self

        def add_local_python_source(self, *_args, **_kwargs):
            return self

        def add_local_file(self, *_args, **_kwargs):
            return self

    class FakeRemoteFunction:
        def remote(self, payload):
            remote_failure.update(
                {
                    "nonce": payload["nonce"],
                    "pins": payload["pins"],
                    "provenance": {
                        "model_id": core.MODEL_ID,
                        "model_revision": core.MODEL_REVISION,
                        "source_file_sha256": payload["pins"]["source_file_sha256"],
                        "measured_source_file_sha256": payload["pins"]["source_file_sha256"],
                    },
                }
            )
            return remote_failure

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **_kwargs):
            return lambda _function: FakeRemoteFunction()

        def run(self):
            return nullcontext()

    fake_modal = SimpleNamespace(
        Image=SimpleNamespace(debian_slim=lambda **_kwargs: FakeImage()),
        Volume=SimpleNamespace(from_name=lambda _name: object()),
        App=FakeApp,
        enable_output=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(
        modal_real_pilot,
        "_load_local_data",
        lambda *_args: (manifest, records, {"fixture": True}),
    )
    monkeypatch.setattr(modal_real_pilot.core, "verify_protocol", lambda: "e" * 64)
    monkeypatch.setattr(modal_real_pilot.core, "source_fingerprints", lambda: source_hashes)
    monkeypatch.setattr(modal_real_pilot.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(modal_real_pilot.importlib.metadata, "version", lambda _name: "1.6.1")
    written_receipts: list[dict[str, object]] = []
    monkeypatch.setattr(
        modal_real_pilot.modal_smoke.smoke,
        "write_json_artifact",
        lambda _reservation, receipt: written_receipts.append(receipt),
    )
    for name in modal_real_pilot.modal_smoke.CREDENTIAL_OVERRIDES:
        monkeypatch.delenv(name, raising=False)

    exit_code = modal_real_pilot._launch(
        profile="reflex-personal",
        workspace="rajath-61258",
        run_id=run_id,
        records_path=str(files["records"]),
        manifest_path=str(files["manifest"]),
        recipe_path=str(files["recipe"]),
        reservation=SimpleNamespace(destination=tmp_path / "receipt.json"),
    )

    assert exit_code == 1
    assert len(written_receipts) == 1
    receipt = written_receipts[0]
    assert receipt["status"] == "failed"
    assert receipt["failure"] == remote_failure["failure"]
    assert receipt["evidence"] == partial_evidence
    assert receipt["phase"] == "training_update_16"


def test_runner_import_and_default_plan_do_not_import_modal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from experiments import modal_real_pilot

    assert "modal" not in sys.modules
    assert modal_real_pilot.main([]) == 0
    assert "modal" not in sys.modules
    assert '"mode": "plan-only"' in capsys.readouterr().out


def test_plan_and_protocol_verification_use_the_frozen_protocol_hash() -> None:
    expected = "88d3bb26a984c95f118dea7cd794d9ee82e38db31e87ecb265a52d49bd37632b"

    assert core.plan()["pins"]["protocol_sha256"] == expected
    assert core.verify_protocol() == expected
