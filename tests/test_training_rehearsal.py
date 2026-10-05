import importlib.util
from collections import Counter
from pathlib import Path

from experiments import training_rehearsal_core
from reflex_decisions.data import audit_splits


def test_prepared_recipe_is_exactly_two_self_authored_train_only_tasks():
    manifest, records = training_rehearsal_core.build_fixture()

    assert manifest.data_kind == "fixture"
    assert {dataset.split for dataset in manifest.datasets} == {"train"}
    assert {dataset.source_id for dataset in manifest.datasets} == {"reflex-authored-rehearsal-v1"}
    assert len(records) == 64
    assert Counter(record.dataset_id for record in records) == {
        "rehearsal-support-v1": 32,
        "rehearsal-infra-v1": 32,
    }
    assert len({record.record_id for record in records}) == 64
    assert len({record.request.request_hash for record in records}) == 64
    assert Counter(record.source_group_id for record in records) == {
        group_id: 4 for group_id in {record.source_group_id for record in records}
    }
    assert len({record.source_group_id for record in records}) == 16
    assert audit_splits(manifest, records).record_count == 64
    assert all(record.record_id.startswith("reflex-authored-rehearsal-v1-") for record in records)


def test_training_examples_reorder_options_but_keep_gold_semantics():
    manifest, records = training_rehearsal_core.build_fixture()
    record = records[0]
    reversed_ids = [option.id for option in reversed(record.request.options)]

    example = training_rehearsal_core.make_example(record, reversed_ids)

    assert example.record_id == record.record_id
    assert example.gold_option_id == record.answer_id
    assert example.gold_index == reversed_ids.index(record.answer_id)
    assert [option.id for option in example.request.options] == reversed_ids


def test_epoch_examples_are_seeded_and_shuffle_record_and_option_order():
    _manifest, records = training_rehearsal_core.build_fixture()

    first = training_rehearsal_core.epoch_examples(records, seed=20261004, epoch=0)
    repeated = training_rehearsal_core.epoch_examples(records, seed=20261004, epoch=0)
    next_epoch = training_rehearsal_core.epoch_examples(records, seed=20261004, epoch=1)

    assert first == repeated
    assert len(first) == len(records)
    assert {example.record_id for example in first} == {record.record_id for record in records}
    assert [example.record_id for example in first] != [example.record_id for example in next_epoch]
    assert any(
        [option.id for option in example.request.options]
        != [
            option.id
            for option in next(
                record for record in records if record.record_id == example.record_id
            ).request.options
        ]
        for example in first
    )
    assert all(
        example.request.options[example.gold_index].id == example.gold_option_id
        for example in first + next_epoch
    )


def test_plan_pins_training_mechanics_and_forward_budget_without_launching():
    plan = training_rehearsal_core.plan()

    assert plan["mode"] == "plan-only"
    assert plan["purpose"] == "synthetic training mechanics and fixture memorization only"
    assert plan["records"] == 64
    assert plan["train_forwards_max"] == 512
    assert plan["max_forward_count"] == 1280
    assert plan["optimizer_updates_max"] == 128
    assert plan["microbatches_per_update"] == 4
    assert plan["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 0.0005,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
        "seed": 20261004,
    }
    assert plan["data_sha256"] == {
        "records": training_rehearsal_core.EXPECTED_RECORDS_SHA256,
        "manifest": training_rehearsal_core.EXPECTED_MANIFEST_SHA256,
        "protocol": training_rehearsal_core.EXPECTED_PROTOCOL_SHA256,
    }
    assert plan["modal"]["gpu"] == "A10"
    assert plan["modal"]["max_containers"] == 1


def test_each_task_has_eight_problem_types_with_four_templates_and_correct_gold():
    manifest, records = training_rehearsal_core.build_fixture()

    for dataset_id, expected_gold in (
        ("rehearsal-support-v1", {"billing", "account"}),
        ("rehearsal-infra-v1", {"network", "storage"}),
    ):
        task_records = [record for record in records if record.dataset_id == dataset_id]
        type_to_records = {}
        for record in task_records:
            _prefix, problem = record.source_group_id.split("-", 1)
            pattern = record.record_id.rsplit("-", 1)[1]
            type_to_records.setdefault(problem, []).append((pattern, record))

        assert len(type_to_records) == 8
        for examples in type_to_records.values():
            assert {pattern for pattern, _record in examples} == {"1", "2", "3", "4"}
        assert {record.answer_id for record in task_records} == expected_gold


def test_prepared_training_files_match_the_pinned_recipe(tmp_path):
    manifest, records = training_rehearsal_core.build_fixture()
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    records_path.write_bytes(training_rehearsal_core.serialize_records(records))
    manifest_path.write_bytes(training_rehearsal_core.serialize_manifest(manifest))

    verified_manifest, verified_records = training_rehearsal_core.verify_prepared_data(
        records_path, manifest_path
    )

    assert verified_manifest == manifest
    assert verified_records == records


def test_checked_in_recipe_and_protocol_match_the_pinned_hashes():
    manifest, records = training_rehearsal_core.verify_prepared_data()

    assert manifest.data_kind == "fixture"
    assert len(records) == 64
    assert training_rehearsal_core.verify_protocol() == (
        training_rehearsal_core.EXPECTED_PROTOCOL_SHA256
    )


def test_run_id_validator_rejects_paths_and_unsafe_characters():
    for run_id in ("../escape", "/absolute", "two words", "line\nbreak", "x" * 65):
        try:
            training_rehearsal_core.validate_run_id(run_id)
        except ValueError:
            continue
        raise AssertionError(f"unsafe run ID was accepted: {run_id!r}")


def test_remote_payload_verifier_accepts_only_the_exact_canonical_recipe():
    manifest, records = training_rehearsal_core.build_fixture()
    record_bytes = training_rehearsal_core.serialize_records(records)
    manifest_bytes = training_rehearsal_core.serialize_manifest(manifest)

    verified_manifest, verified_records = training_rehearsal_core.verify_prepared_bytes(
        record_bytes, manifest_bytes
    )

    assert verified_manifest == manifest
    assert verified_records == records
    altered_records = record_bytes.replace(b"Billing support", b"billing support", 1)
    try:
        training_rehearsal_core.verify_prepared_bytes(altered_records, manifest_bytes)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("altered training payload was accepted")


def test_remote_source_attestation_requires_exact_caller_and_measured_maps(monkeypatch):
    launcher = load_training_launcher()
    expected = training_rehearsal_core.source_fingerprints()
    measured = dict(expected)
    monkeypatch.setattr(launcher, "_measure_remote_source_fingerprints", lambda: measured)

    assert launcher._verify_remote_source_fingerprints(expected) == expected
    for supplied in ({}, {"arbitrary.py": "a" * 64}, {**expected, "extra.py": "b" * 64}):
        try:
            launcher._verify_remote_source_fingerprints(supplied)
        except ValueError:
            pass
        else:
            raise AssertionError("empty or arbitrary source map was accepted")

    changed = dict(expected)
    changed[next(iter(changed))] = "0" * 64
    monkeypatch.setattr(launcher, "_measure_remote_source_fingerprints", lambda: measured)
    try:
        launcher._verify_remote_source_fingerprints(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("source drift was not rejected")


def test_remote_source_module_paths_map_to_canonical_fingerprint_keys():
    launcher = load_training_launcher()

    assert (
        launcher._canonical_source_key("/root/reflex_decisions/schema.py")
        == "src/reflex_decisions/schema.py"
    )
    assert (
        launcher._canonical_source_key("/root/experiments/modal_train_rehearsal.py")
        == "experiments/modal_train_rehearsal.py"
    )


def test_progress_json_parser_rejects_duplicate_keys_and_nonstandard_constants():
    launcher = load_training_launcher()

    for raw in (b'{"status":"failed","status":"passed"}', b'{"accuracy":NaN}'):
        try:
            launcher._strict_json_object(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("non-strict progress JSON was accepted")


def test_training_launcher_default_prints_a_plan_without_cloud_or_torch_imports(
    monkeypatch, capsys
):
    import builtins

    launcher = load_training_launcher()
    original_import = builtins.__import__

    def forbid_paid_or_model_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("torch"):
            raise AssertionError(f"default planning imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_paid_or_model_import)

    assert launcher.main([]) == 0
    plan = __import__("json").loads(capsys.readouterr().out)
    assert plan["mode"] == "plan-only"
    assert plan["model_revision"] == training_rehearsal_core.MODEL_REVISION


def test_training_auth_failure_writes_a_sanitized_receipt_before_modal_import(
    tmp_path, monkeypatch, capsys
):
    import builtins
    import json

    launcher = load_training_launcher()
    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", lambda *_args: False)
    original_import = builtins.__import__

    def forbid_modal_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("torch"):
            raise AssertionError(f"authentication failure imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_modal_import)
    output = tmp_path / "failed-run.json"

    assert launcher.main(["--launch", "--run-id", "unit-test-run", "--output", str(output)]) == 1

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "auth"
    assert receipt["provenance"]["records_sha256"] == (
        training_rehearsal_core.EXPECTED_RECORDS_SHA256
    )
    assert "unit-test-run" not in capsys.readouterr().err


def test_credential_override_failure_is_labeled_and_stops_before_profile_check(
    tmp_path, monkeypatch
):
    import builtins
    import json

    launcher = load_training_launcher()
    monkeypatch.setenv("MODAL_TOKEN_ID", "do-not-copy-this-value")

    def forbidden_profile_check(*_args):
        raise AssertionError("profile check must not run with a credential override")

    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", forbidden_profile_check)
    original_import = builtins.__import__

    def forbid_modal_import(name, *args, **kwargs):
        if name == "modal" or name.startswith("torch"):
            raise AssertionError(f"credential preflight imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_modal_import)
    output = tmp_path / "credential-failure.json"

    assert launcher.main(["--launch", "--run-id", "unit-test-run", "--output", str(output)]) == 1

    text = output.read_text(encoding="utf-8")
    assert "do-not-copy-this-value" not in text
    assert json.loads(text)["failure"]["stage"] == "environment"


def test_existing_volume_run_id_is_never_rewritten_or_committed(tmp_path, monkeypatch):
    launcher = load_training_launcher()
    source_hashes = training_rehearsal_core.source_fingerprints()
    artifact_root = tmp_path / "volume"
    existing_run = artifact_root / "runs" / "already-used"
    existing_run.mkdir(parents=True)
    progress = existing_run / "progress.json"
    progress.write_bytes(b"keep existing receipt byte-for-byte\n")
    before = progress.read_bytes()
    manifest, records = training_rehearsal_core.build_fixture()
    commits = []
    progress_writes = []
    monkeypatch.setattr(launcher, "VOLUME_ROOT", artifact_root)
    monkeypatch.setattr(launcher, "_verify_remote_source_fingerprints", lambda supplied: supplied)
    monkeypatch.setattr(launcher, "_commit_volume", lambda: commits.append(True))
    monkeypatch.setattr(
        launcher,
        "_write_progress",
        lambda *_args: progress_writes.append(True),
    )

    receipt = launcher._remote_train(
        {
            "run_id": "already-used",
            "attempt_id": "123e4567-e89b-12d3-a456-426614174000",
            "records_jsonl": training_rehearsal_core.serialize_records(records).decode("utf-8"),
            "manifest_json": training_rehearsal_core.serialize_manifest(manifest).decode("utf-8"),
            "protocol_sha256": training_rehearsal_core.EXPECTED_PROTOCOL_SHA256,
            "source_file_sha256": source_hashes,
        }
    )

    assert receipt["status"] == "failed"
    assert progress.read_bytes() == before
    assert progress_writes == []
    assert commits == []


def test_base_snapshot_download_is_anonymous_and_revision_pinned(monkeypatch):
    launcher = load_training_launcher()
    received = {}

    def fake_snapshot_download(**kwargs):
        received.update(kwargs)
        return "/temporary/model-snapshot"

    assert launcher._download_base_snapshot(fake_snapshot_download) == "/temporary/model-snapshot"
    assert received == {
        "repo_id": training_rehearsal_core.MODEL_ID,
        "revision": training_rehearsal_core.MODEL_REVISION,
        "token": False,
    }


def test_modal_teardown_failure_keeps_the_returned_remote_evidence(tmp_path, monkeypatch):
    import json
    import uuid

    launcher = load_training_launcher()
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda _name: "1.6.1")
    remote_receipt = None

    class FakeImage:
        @classmethod
        def debian_slim(cls, **_kwargs):
            return cls()

        def env(self, *_args, **_kwargs):
            return self

        def pip_install(self, *_args, **_kwargs):
            return self

        def add_local_python_source(self, *_args, **_kwargs):
            return self

        def add_local_file(self, *_args, **_kwargs):
            return self

    class FakeRunner:
        def remote(self, payload):
            nonlocal remote_receipt
            assert str(uuid.UUID(payload["attempt_id"])) == payload["attempt_id"]
            remote_receipt = make_valid_passed_receipt(
                "lifecycle-test",
                payload.get("attempt_id"),
                payload.get("source_file_sha256", {}),
            )
            return remote_receipt

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **_kwargs):
            return lambda _function: FakeRunner()

        def run(self):
            class FailingContext:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    raise RuntimeError("application teardown failed")

            return FailingContext()

    class FakeVolume:
        @classmethod
        def from_name(cls, *_args, **_kwargs):
            return cls()

    class FakeModal:
        Image = FakeImage
        App = FakeApp
        Volume = FakeVolume

        @staticmethod
        def enable_output():
            from contextlib import nullcontext

            return nullcontext()

    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: FakeModal)
    output = tmp_path / "teardown-receipt.json"
    with launcher.smoke.reserve_output(output) as reservation:
        assert (
            launcher._launch(
                profile="reflex-personal",
                workspace="rajath-61258",
                run_id="lifecycle-test",
                records_path=training_rehearsal_core.DEFAULT_RECORDS,
                manifest_path=training_rehearsal_core.DEFAULT_MANIFEST,
                reservation=reservation,
            )
            == 1
        )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "remote_training"
    assert receipt["attempt_id"] == remote_receipt["attempt_id"]
    assert receipt["evidence"]["optimizer_updates_completed"] == 16
    assert receipt["evidence"]["forward_count"] == 448


def test_remote_exception_recovers_only_pinned_progress_and_preserves_failure(
    tmp_path, monkeypatch
):
    import json

    launcher = load_training_launcher()
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.setattr(launcher.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(launcher.importlib.metadata, "version", lambda _name: "1.6.1")
    payload_seen = {}
    read_paths = []

    class FakeImage:
        @classmethod
        def debian_slim(cls, **_kwargs):
            return cls()

        def env(self, *_args, **_kwargs):
            return self

        def pip_install(self, *_args, **_kwargs):
            return self

        def add_local_python_source(self, *_args, **_kwargs):
            return self

        def add_local_file(self, *_args, **_kwargs):
            return self

    class FakeRunner:
        def remote(self, payload):
            payload_seen.update(payload)
            raise RuntimeError("remote result transport failed")

    class FakeApp:
        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **_kwargs):
            return lambda _function: FakeRunner()

        def run(self):
            from contextlib import nullcontext

            return nullcontext()

    class FakeVolume:
        @classmethod
        def from_name(cls, *_args, **_kwargs):
            return cls()

        def read_file(self, path):
            read_paths.append(path)
            receipt = make_valid_passed_receipt(
                payload_seen["run_id"],
                payload_seen["attempt_id"],
                payload_seen["source_file_sha256"],
            )
            return iter([json.dumps(receipt, allow_nan=False).encode("utf-8")])

    class FakeModal:
        Image = FakeImage
        App = FakeApp
        Volume = FakeVolume

        @staticmethod
        def enable_output():
            from contextlib import nullcontext

            return nullcontext()

    monkeypatch.setattr(launcher.importlib, "import_module", lambda _name: FakeModal)
    output = tmp_path / "progress-recovery.json"
    with launcher.smoke.reserve_output(output) as reservation:
        assert (
            launcher._launch(
                profile="reflex-personal",
                workspace="rajath-61258",
                run_id="progress-recovery",
                records_path=training_rehearsal_core.DEFAULT_RECORDS,
                manifest_path=training_rehearsal_core.DEFAULT_MANIFEST,
                reservation=reservation,
            )
            == 1
        )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert read_paths == ["runs/progress-recovery/progress.json"]
    assert receipt["status"] == "failed"
    assert receipt["failure"]["message"] == "remote result transport failed"
    assert receipt["attempt_id"] == payload_seen["attempt_id"]
    assert receipt["evidence"]["optimizer_updates_completed"] == 16


def test_passed_receipt_requires_matching_pins_and_complete_evidence():
    import copy
    import uuid

    launcher = load_training_launcher()
    source_hashes = training_rehearsal_core.source_fingerprints()
    expected = {
        "run_id": "receipt-validation",
        "attempt_id": str(uuid.uuid4()),
        "source_file_sha256": source_hashes,
    }
    valid = make_valid_passed_receipt(expected["run_id"], expected["attempt_id"], source_hashes)

    assert launcher._validate_remote_receipt(valid, expected) == valid

    partial_failure = {
        "schema_version": 1,
        "status": "failed",
        "run_id": expected["run_id"],
        "attempt_id": expected["attempt_id"],
    }
    assert launcher._validate_remote_receipt(partial_failure, expected) == partial_failure
    try:
        launcher._validate_remote_receipt(partial_failure, expected, require_full_provenance=True)
    except ValueError:
        pass
    else:
        raise AssertionError("progress receipt without full provenance was accepted")

    unrelated = copy.deepcopy(valid)
    unrelated["run_id"] = "somebody-elses-run"
    wrong_source = copy.deepcopy(valid)
    wrong_source["provenance"]["measured_source_file_sha256"] = {
        **source_hashes,
        next(iter(source_hashes)): "0" * 64,
    }
    minimal_passed = {"schema_version": 1, "status": "passed", "run_id": expected["run_id"]}
    unrelated_failure = {**partial_failure, "run_id": "somebody-elses-run"}
    for receipt in (unrelated, wrong_source, minimal_passed, unrelated_failure):
        try:
            launcher._validate_remote_receipt(receipt, expected)
        except ValueError:
            pass
        else:
            raise AssertionError("unverified passed receipt was accepted")


def make_valid_passed_receipt(run_id, attempt_id, source_hashes):
    return {
        "schema_version": 1,
        "status": "passed",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "phase": "completed",
        "provenance": {
            "model_id": training_rehearsal_core.MODEL_ID,
            "model_revision": training_rehearsal_core.MODEL_REVISION,
            "seed": training_rehearsal_core.SEED,
            "records_sha256": training_rehearsal_core.EXPECTED_RECORDS_SHA256,
            "manifest_sha256": training_rehearsal_core.EXPECTED_MANIFEST_SHA256,
            "protocol_sha256": training_rehearsal_core.EXPECTED_PROTOCOL_SHA256,
            "source_file_sha256": source_hashes,
            "measured_source_file_sha256": source_hashes,
        },
        "evidence": {
            "optimizer_updates_completed": 16,
            "train_forward_count": 64,
            "forward_count": 448,
            "max_forward_count": training_rehearsal_core.MAX_FORWARD_COUNT,
            "diagnostics": [{"update": 16, "accuracy": 0.96}],
            "memorization_threshold_met": True,
            "lo_ra_b_gradient_l1_total": 1.0,
            "adapter_update": {"changed_tensor_count": 1, "changed_tensor_names": ["lora"]},
            "reload_parity": {
                "presentation_count": 128,
                "max_candidate_logit_difference": 0.0001,
                "winner_mismatches": 0,
                "adapter_tensor_keys_shapes_values_match": True,
                "adapter_tensors_unmerged_fp32": True,
                "model_eval_mode": True,
                "use_cache": False,
            },
            "adapter_paths": [
                {
                    "update": 16,
                    "path": f"/artifacts/runs/{run_id}/adapter-update-016",
                    "files_sha256": {
                        "adapter_model.safetensors": "a" * 64,
                        "adapter_config.json": "b" * 64,
                    },
                }
            ],
        },
    }


def test_adapter_snapshot_config_records_portable_base_id_and_revision(tmp_path):
    import json

    launcher = load_training_launcher()

    class FakeAdapter:
        def save_pretrained(self, directory, *, safe_serialization):
            assert safe_serialization is True
            (Path(directory) / "adapter_model.safetensors").write_bytes(b"fake safetensors")
            (Path(directory) / "adapter_config.json").write_text(
                '{"base_model_name_or_path":"/container/cache/snapshot","revision":null}\n',
                encoding="utf-8",
            )

    snapshot = launcher._save_adapter_snapshot(FakeAdapter(), tmp_path, 16)
    config = json.loads(
        (tmp_path / "adapter-update-016" / "adapter_config.json").read_text(encoding="utf-8")
    )

    assert snapshot["path"] == str(tmp_path / "adapter-update-016")
    assert config["base_model_name_or_path"] == training_rehearsal_core.MODEL_ID
    assert config["revision"] == training_rehearsal_core.MODEL_REVISION


def load_training_launcher():
    path = Path(__file__).parents[1] / "experiments" / "modal_train_rehearsal.py"
    spec = importlib.util.spec_from_file_location("modal_train_rehearsal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
