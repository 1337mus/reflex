"""CPU-only lifecycle tests for the paired real-pilot reference workers."""

from __future__ import annotations

import importlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from experiments import baseline_intern
from experiments import modal_real_pilot_baselines as runner
from reflex_decisions.schema import DecisionRequest
from tests.test_real_pilot_baseline_core import _build_payload


def test_default_command_prints_plan_without_importing_modal_or_torch(monkeypatch, capsys) -> None:
    original_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name in {"modal", "torch"}:
            raise AssertionError(f"plan-only mode imported {name}")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    assert runner.main([]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan-only"
    assert plan["models"] == ["intern", "kev"]
    assert plan["modal"]["gpu"] == "A10"


def test_worker_rejects_measured_source_mismatch_before_loading_scorer(monkeypatch) -> None:
    payload = _build_payload()
    measured_hashes = {path: "b" * 64 for path in runner.core.SOURCE_FINGERPRINT_PATHS}
    monkeypatch.setattr(runner, "_measure_remote_source_fingerprints", lambda: measured_hashes)
    loaded: list[str] = []
    monkeypatch.setattr(runner, "_load_scorer", lambda model_name: loaded.append(model_name))

    receipt = runner._remote_score(payload, "intern")

    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "source_verification"
    assert receipt["provenance"]["measured_source_file_sha256"] == measured_hashes
    assert loaded == []


def test_worker_retains_completed_rows_when_a_later_score_fails(monkeypatch) -> None:
    payload = _build_payload()
    source_hashes = payload["pins"]["source_file_sha256"]
    monkeypatch.setattr(runner, "_measure_remote_source_fingerprints", lambda: source_hashes)
    monkeypatch.setattr(runner, "_runtime_versions", lambda: dict(runner.RUNTIME_VERSION_PINS))
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")
    calls = 0

    def score(request: DecisionRequest) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("scoring stopped")
        return {
            "raw_logits": [1.0, *([0.0] * (len(request.options) - 1))],
            "input_tokens": 12,
            "prompt_sha256": "b" * 64,
        }

    adapter_provenance = {
        "model_id": baseline_intern.MODEL_ID,
        "model_revision": baseline_intern.REVISION,
        "temperature": baseline_intern.TEMPERATURE,
        "source_commit": baseline_intern.SOURCE_COMMIT,
        "module_sha256": baseline_intern.MODULE_SHA256,
        "calibration_parity_passed": True,
        "auxiliary_forward_count": 1,
        "versions": {},
    }
    monkeypatch.setattr(
        runner,
        "_load_scorer",
        lambda model_name: (baseline_intern, score, adapter_provenance),
    )

    receipt = runner._remote_score(payload, "intern")

    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "inference"
    assert receipt["forward_counts"] == {"scored": 1, "auxiliary": 1, "total": 2}
    assert len(receipt["presentations"]) == 1
    assert (
        receipt["presentations"][0]["winner_option_id"]
        == payload["evaluation_presentations"][0]["order_ids"][0]
    )
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_worker_defers_intern_parity_validation_and_roundtrips_persisted_state(
    monkeypatch,
) -> None:
    payload = _build_payload()
    source_hashes = payload["pins"]["source_file_sha256"]
    monkeypatch.setattr(runner, "_measure_remote_source_fingerprints", lambda: source_hashes)
    monkeypatch.setattr(runner, "_runtime_versions", lambda: dict(runner.RUNTIME_VERSION_PINS))
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0")

    provenance: dict[str, object] = {
        "model_id": baseline_intern.MODEL_ID,
        "model_revision": baseline_intern.REVISION,
        "temperature": baseline_intern.TEMPERATURE,
        "source_commit": baseline_intern.SOURCE_COMMIT,
        "module_sha256": baseline_intern.MODULE_SHA256,
        "calibration_parity_passed": False,
        "auxiliary_forward_count": 0,
        "versions": {"torch": "2.14.1+cu130", "transformers": "5.18.0"},
    }
    scored_calls = 0

    def score(request: DecisionRequest) -> dict[str, object]:
        nonlocal scored_calls
        if scored_calls == 0:
            assert provenance["calibration_parity_passed"] is False
            assert provenance["auxiliary_forward_count"] == 0
            provenance["calibration_parity_passed"] = True
            provenance["auxiliary_forward_count"] = 1
        scored_calls += 1
        return {
            "raw_logits": [float(index) for index, _option in enumerate(request.options)],
            "input_tokens": 12,
            "prompt_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        runner,
        "_load_scorer",
        lambda _model_name: (baseline_intern, score, provenance),
    )

    receipt = runner._remote_score(payload, "intern")
    persisted = runner._persisted_model_state(
        receipt, expected_payload=payload, model_name="intern"
    )

    assert receipt["status"] == "passed", receipt.get("failure")
    assert receipt["forward_counts"] == {"scored": 2572, "auxiliary": 1, "total": 2573}
    assert receipt["provenance"]["scorer"]["calibration_parity_passed"] is True
    assert persisted["status"] == "passed"
    assert persisted["forward_counts"] == receipt["forward_counts"]
    assert persisted["provenance"]["scorer"]["auxiliary_forward_count"] == 1


def test_modal_function_options_pin_two_single_use_a10_workers() -> None:
    options = runner._function_options()

    assert options == {
        "gpu": "A10",
        "cpu": (2.0, 2.0),
        "memory": (16384, 16384),
        "max_containers": 2,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window": 2,
        "retries": 0,
        "single_use_containers": True,
        "serialized": False,
        "include_source": False,
        "startup_timeout": 300,
        "timeout": 3600,
    }


def test_modal_image_uploads_only_fingerprinted_sources_and_protocols(tmp_path) -> None:
    class ImageBuilder:
        def __init__(self) -> None:
            self.files: list[tuple[str, str]] = []
            self.environment: dict[str, str] = {}
            self.packages: tuple[str, ...] = ()

        def env(self, values: dict[str, str]) -> ImageBuilder:
            self.environment = values
            return self

        def pip_install(self, *packages: str, **_options: object) -> ImageBuilder:
            self.packages = packages
            return self

        def add_local_file(self, path: str, *, remote_path: str) -> ImageBuilder:
            self.files.append((path, remote_path))
            return self

    image = ImageBuilder()

    class Modal:
        class Image:
            @staticmethod
            def debian_slim(*, python_version: str) -> ImageBuilder:
                assert python_version == "3.12"
                return image

    runner._build_modal_image(Modal, tmp_path)

    assert {Path(local).relative_to(tmp_path).as_posix() for local, _ in image.files} == set(
        runner.core.SOURCE_FINGERPRINT_PATHS
    )
    assert all(remote.startswith("/root/") for _, remote in image.files)
    assert all("data/" not in local for local, _ in image.files)
    assert image.environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert "huggingface-hub==1.33.0" in image.packages
    assert "tokenizers==0.23.2" in image.packages
    assert "safetensors==0.8.0" in image.packages
    assert runner.RUNTIME_VERSION_PINS["huggingface-hub"] == "1.33.0"
    assert runner.RUNTIME_VERSION_PINS["tokenizers"] == "0.23.2"
    assert runner.RUNTIME_VERSION_PINS["safetensors"] == "0.8.0"


def test_worker_calls_spawn_together_and_collect_both_outcomes() -> None:
    events: list[str] = []

    class Call:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def get(self) -> str:
            events.append(f"get:{self.model_name}")
            if self.model_name == "intern":
                raise TimeoutError("Intern worker failed")
            return "kev receipt"

    class Function:
        def spawn(self, _payload: object, model_name: str) -> Call:
            events.append(f"spawn:{model_name}")
            return Call(model_name)

    results, failures = runner._collect_worker_calls(Function(), {})

    assert events == ["spawn:intern", "spawn:kev", "get:intern", "get:kev"]
    assert results == {"kev": "kev receipt"}
    assert isinstance(failures["intern"], TimeoutError)


def test_launch_teardown_failure_marks_receipt_failed_and_cli_returns_nonzero(
    monkeypatch, tmp_path, capsys
) -> None:
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    recipe = tmp_path / "recipe.json"
    records.write_text("records\n", encoding="utf-8")
    manifest.write_text("manifest\n", encoding="utf-8")
    recipe.write_text("recipe\n", encoding="utf-8")
    payload = {"run_id": "test-run"}
    states = {"intern": {"status": "passed"}, "kev": {"status": "passed"}}

    monkeypatch.setattr(
        runner.pilot_data, "verify_prepared_data", lambda *_args: (None, None, None)
    )
    monkeypatch.setattr(runner.core.real_pilot_core, "verify_protocol", lambda *_args: "a" * 64)
    monkeypatch.setattr(runner.core, "verify_reference_protocol", lambda *_args: "b" * 64)
    monkeypatch.setattr(runner.core, "source_fingerprints", lambda *_args: {})
    monkeypatch.setattr(runner.core, "build_evaluation_payload", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(runner.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner, "_build_modal_image", lambda *_args: "image")
    monkeypatch.setattr(runner, "_collect_model_states", lambda *_args: states)

    class Run:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            raise RuntimeError("app teardown failed")

    class App:
        def function(self, **_kwargs: object):
            return lambda function: function

        def run(self) -> Run:
            return Run()

    app = App()

    class Modal:
        def App(self, *_args: object, **_kwargs: object) -> App:
            return app

        @staticmethod
        def enable_output():
            return nullcontext()

    original_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None):
        if name == "modal":
            return Modal()
        return original_import_module(name, package)

    monkeypatch.setattr(runner.importlib, "import_module", import_module)
    for name in runner.modal_smoke.CREDENTIAL_OVERRIDES:
        monkeypatch.delenv(name, raising=False)

    receipt = runner._launch(SimpleNamespace(records=records, manifest=manifest, recipe=recipe))

    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "worker_execution"
    assert receipt["models"] == states

    monkeypatch.setattr(runner, "_launch", lambda _args: receipt)
    output = tmp_path / "teardown.json"
    assert runner.main(["--launch", "--output", str(output)]) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def test_explicit_launch_reserves_a_new_output_and_never_overwrites(
    monkeypatch, tmp_path, capsys
) -> None:
    output = tmp_path / "reference-run.json"
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "payload": {},
        "models": {},
        "provenance": {},
        "limits": {},
    }
    launches: list[bool] = []
    monkeypatch.setattr(runner, "_launch", lambda args: launches.append(True) or receipt)

    assert runner.main(["--launch", "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert json.loads(saved)["status"] == "passed"
    assert runner.main(["--launch", "--output", str(output)]) == 1

    assert output.read_bytes() == saved
    assert launches == [True]
