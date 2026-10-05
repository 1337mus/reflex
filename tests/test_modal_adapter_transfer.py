"""Host-side plan and lifecycle tests for the adapter transfer evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import uuid
from contextlib import nullcontext

from experiments import adapter_transfer_core as core
from experiments import adapter_transfer_data as data
from experiments import modal_adapter_transfer as runner
from experiments import modal_smoke
from tests.test_adapter_transfer_core import _output, _payload


def test_plan_only_reports_frozen_profile_without_importing_modal_or_torch(
    tmp_path, monkeypatch, capsys
) -> None:
    output = tmp_path / "adapter-transfer.json"
    original_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name in {"modal", "torch"}:
            raise AssertionError(f"plan-only mode imported {name}")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    plan = runner.plan(runner.RUN_ID, str(output))

    assert plan["mode"] == "plan-only"
    assert plan["experiment_id"] == "adapter-transfer-v1"
    assert plan["run_id"] == runner.RUN_ID
    assert plan["output"] == str(output)
    assert plan["presentations"] == 128
    assert plan["parity_presentations"] == 16
    assert plan["max_total_forwards"] == 272
    assert plan["max_total_input_tokens"] == 557056
    assert plan["profile"] == "reflex-personal"
    assert plan["workspace"] == "rajath-61258"
    assert plan["modal_sdk_version"] == "1.6.1"
    assert plan["modal"]["gpu"] == "A10"
    assert plan["modal"]["timeout"] == 900
    assert plan["modal"]["startup_timeout"] == 300
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_module_entrypoint_displays_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.modal_adapter_transfer", "--help"],
        cwd=runner.PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert "--execute" in completed.stdout
    assert "--output OUTPUT" in completed.stdout


def test_modal_image_preserves_the_pinned_source_tree_layout() -> None:
    class FakeImage:
        environment: dict[str, str] = {}
        files: list[tuple[str, str]] = []

        @classmethod
        def debian_slim(cls, *, python_version: str):
            assert python_version == "3.12"
            cls.environment = {}
            cls.files = []
            return cls()

        def env(self, values: dict[str, str]):
            self.environment = values
            return self

        def pip_install(self, *_packages: str, extra_index_url: str):
            assert extra_index_url == "https://download.pytorch.org/whl/cu130"
            return self

        def add_local_file(self, local_path: str, *, remote_path: str):
            self.files.append((local_path, remote_path))
            return self

    image = runner._build_modal_image(
        type("FakeModal", (), {"Image": FakeImage}), runner.PROJECT_ROOT
    )

    assert image.environment["PYTHONPATH"] == "/root:/root/src"
    assert image.environment["USE_HUB_KERNELS"] == "NO"
    assert image.files == [
        (str(runner.PROJECT_ROOT / relative), f"/root/{relative}")
        for relative in runner.contracts.SOURCE_FINGERPRINT_PATHS
    ]
    assert (
        str(runner.PROJECT_ROOT / "experiments/adapter_transfer_results.py"),
        "/root/experiments/adapter_transfer_results.py",
    ) in image.files


def test_dry_run_reserves_and_writes_payload_without_remote_imports(
    tmp_path, monkeypatch, capsys
) -> None:
    output = tmp_path / "adapter-transfer.json"
    payload = {"run_id": runner.RUN_ID, "selection_sha256": "a" * 64}
    observed: list[tuple[str, bool]] = []

    def load_payload(args, project_root) -> dict[str, object]:
        lock = tmp_path / ".adapter-transfer.json.reflex-smoke.lock"
        observed.append((args.run_id, lock.exists()))
        return payload

    monkeypatch.setattr(runner, "_load_payload", load_payload)
    original_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name in {"modal", "torch"}:
            raise AssertionError(f"dry-run imported {name}")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    assert runner.main(["--run-id", runner.RUN_ID, "--output", str(output)]) == 0
    receipt = json.loads(output.read_text())

    assert observed == [(runner.RUN_ID, True)]
    assert receipt["status"] == "dry_run"
    assert receipt["phase"] == "preflight_complete"
    assert receipt["payload"] == payload
    assert receipt["result"] is None
    assert receipt["failure"] is None
    assert receipt["modal"]["profile"] == "reflex-personal"
    assert json.loads(capsys.readouterr().out) == {
        "status": "dry_run",
        "artifact": str(output),
    }


def test_payload_preflight_builds_and_validates_with_a_fresh_nonce(tmp_path, monkeypatch) -> None:
    records = (object(),)
    presentations = [{"presentation_id": "p0"}]
    selection = {"selection_id": "adapter-transfer-v1"}
    pins = {"protocol_sha256": "c" * 64}
    payload = {"run_id": runner.RUN_ID, "nonce": ""}
    events: list[str] = []

    def load_inputs(project_root):
        events.append("inputs")
        assert project_root == tmp_path
        return records, presentations, selection, pins

    def build_payload(run_id, nonce, rows, selected, input_pins, *, root):
        events.append("build")
        assert run_id == runner.RUN_ID
        assert rows is presentations
        assert selected is selection
        assert input_pins is pins
        assert root == tmp_path
        payload["nonce"] = nonce
        return payload

    def validate_payload(value, *, root):
        events.append("validate")
        assert value is payload
        assert root == tmp_path
        return value

    monkeypatch.setattr(data, "load_inputs", load_inputs)
    monkeypatch.setattr(core, "build_payload", build_payload)
    monkeypatch.setattr(core, "validate_payload", validate_payload)

    import argparse

    actual = runner._load_payload(argparse.Namespace(run_id=runner.RUN_ID), tmp_path)

    assert events == ["inputs", "build", "validate"]
    assert actual is payload
    assert uuid.UUID(payload["nonce"])
    assert payload["nonce"] != runner.RUN_ID


def test_profile_mismatch_returns_payload_bound_failure_before_modal_import(
    tmp_path, monkeypatch
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": "adapter-transfer-v1",
        "run_id": runner.RUN_ID,
        "nonce": "c8b5fb54-591a-4541-aeca-6c1b2b5e5c89",
        "selection_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "pins": {"source_file_sha256": {}},
    }
    failed_result = {
        "run_id": runner.RUN_ID,
        "status": "failed",
        "phase": "profile_verification_failed",
        "failure": {"stage": "profile_verification", "message": "wrong profile"},
    }
    evidence_seen: list[dict[str, object]] = []

    def build_failed(_payload, *, stage, error, sanitize, provenance=None, evidence=None):
        assert stage == "profile_verification"
        evidence_seen.append(evidence)
        return failed_result

    monkeypatch.setattr(core, "build_failed_result", build_failed)
    monkeypatch.setattr(core, "validate_result", lambda value, _payload: value)
    monkeypatch.setattr(runner, "_load_payload", lambda *_args: payload)
    monkeypatch.setattr(modal_smoke, "_verify_profile", lambda *_args: False)
    for name in modal_smoke.CREDENTIAL_OVERRIDES:
        monkeypatch.delenv(name, raising=False)

    def unexpected_version(_name: str) -> str:
        raise AssertionError("SDK lookup must follow profile verification")

    monkeypatch.setattr(importlib.metadata, "version", unexpected_version)
    original_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name == "modal":
            raise AssertionError("profile mismatch must stop before importing Modal")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    receipt = runner._launch(argparse.Namespace(run_id=runner.RUN_ID, root=str(tmp_path)))

    assert receipt["status"] == "failed"
    assert receipt["phase"] == "profile_verification_failed"
    assert receipt["payload"] == payload
    assert receipt["result"]["status"] == "failed"
    assert receipt["result"]["failure"]["stage"] == "profile_verification"
    assert receipt["modal"]["sdk_version"] is None
    assert evidence_seen[0]["forward_counts"] == {
        "base_evaluation": 0,
        "training": 0,
        "final_evaluation": 0,
        "reload_parity": 0,
        "total": 0,
    }


def test_app_context_exit_failure_keeps_completed_worker_result(tmp_path, monkeypatch) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": "adapter-transfer-v1",
        "run_id": runner.RUN_ID,
        "nonce": "c8b5fb54-591a-4541-aeca-6c1b2b5e5c89",
        "selection_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "pins": {"source_file_sha256": {}},
    }
    completed = {"run_id": runner.RUN_ID, "status": "passed", "phase": "completed"}
    monkeypatch.setattr(runner, "_load_payload", lambda *_args: payload)
    monkeypatch.setattr(modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner, "_build_modal_image", lambda *_args: object())
    monkeypatch.setattr(core, "validate_result", lambda value, _payload: value)
    for name in modal_smoke.CREDENTIAL_OVERRIDES:
        monkeypatch.delenv(name, raising=False)

    class FakeVolume:
        def with_mount_options(self, *, read_only: bool):
            assert read_only is True
            return self

    class FakeFunction:
        def remote(self, received):
            assert received is payload
            return completed

    class FakeApp:
        def __init__(self, _name: str, *, image) -> None:
            assert image is not None

        def function(self, **options):
            assert options["gpu"] == "A10"
            assert options["retries"] == 0
            return lambda _worker: FakeFunction()

        def run(self):
            class FailingExit:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    raise RuntimeError("app teardown failed")

            return FailingExit()

    fake_modal = type(
        "FakeModal",
        (),
        {
            "Volume": type("Volume", (), {"from_name": staticmethod(lambda _name: FakeVolume())}),
            "App": FakeApp,
            "enable_output": staticmethod(nullcontext),
        },
    )
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "modal":
            return fake_modal
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    receipt = runner._launch(argparse.Namespace(run_id=runner.RUN_ID, root=str(tmp_path)))

    assert receipt["status"] == "failed"
    assert receipt["phase"] == "modal_lifecycle_failed"
    assert receipt["payload"] == payload
    assert receipt["result"] is completed
    assert receipt["result"]["status"] == "passed"
    assert receipt["failure"]["stage"] == "modal_lifecycle"


def test_execute_reserves_receipt_before_preflight_and_preserves_failure(
    tmp_path, monkeypatch, capsys
) -> None:
    output = tmp_path / "adapter-transfer.json"
    expected = {
        "schema_version": 1,
        "run_id": runner.RUN_ID,
        "status": "failed",
        "phase": "selection_preflight",
        "payload": None,
        "result": None,
        "failure": {"stage": "selection_preflight", "type": "ValueError", "message": "pending"},
        "provenance": {},
    }

    def fail_preflight(args) -> dict[str, object]:
        assert args.run_id == runner.RUN_ID
        assert (tmp_path / ".adapter-transfer.json.reflex-smoke.lock").exists()
        return expected

    monkeypatch.setattr(runner, "_launch", fail_preflight)

    assert runner.main(["--execute", "--run-id", runner.RUN_ID, "--output", str(output)]) == 1

    assert json.loads(output.read_text()) == expected
    assert not (tmp_path / ".adapter-transfer.json.reflex-smoke.lock").exists()
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "artifact": str(output),
    }


def test_invalid_worker_provenance_does_not_discard_valid_partial_evidence(monkeypatch) -> None:
    payload = _payload(monkeypatch)
    presentation = payload["presentations"][0]
    output = _output(presentation)
    partial_evidence = {
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
        "outputs": {"base": [output], "adapter": []},
        "adapter_identity": None,
        "reload_parity": {"outputs": [], "max_candidate_logit_delta": None},
    }
    partial_result = runner.runtime._initial_result(payload)
    partial_result.update(
        status="failed",
        phase="base_evaluation_failed",
        failure={"stage": "base_evaluation", "type": "RuntimeError", "message": "scoring stopped"},
        evidence=partial_evidence,
    )
    partial_result["provenance"]["model_id"] = "untrusted worker provenance"

    failed = runner._failed_worker_result(
        payload,
        "result_validation",
        ValueError("worker result provenance did not match the pinned model"),
        known_unstarted=False,
        partial=partial_result,
    )

    assert failed["status"] == "failed"
    assert failed["failure"]["stage"] == "result_validation"
    assert failed["evidence"]["outputs"]["base"] == [output]
    assert failed["evidence"]["forward_counts"]["base_evaluation"] == 1
    assert failed["provenance"]["model_id"] == runner.contracts.MODEL_ID
