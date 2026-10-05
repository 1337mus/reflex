from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from threading import Barrier, Lock

from experiments import mixture_training_modal_host as modal_host
from experiments import modal_mixture_training as runner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_preflight_loads_and_transmits_exact_three_source_training_union(
    monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace

    from experiments import mixture_training_data as data

    real_id = next(iter(data.REAL_TRAIN_DATASET_COUNTS))
    synthetic_id = next(iter(data.SYNTHETIC_TRAIN_DATASET_COUNTS))
    snli_id = next(iter(data.SNLI_TRAIN_DATASET_COUNTS))
    records = (
        SimpleNamespace(record_id="real", dataset_id=real_id),
        SimpleNamespace(record_id="synthetic", dataset_id=synthetic_id),
        SimpleNamespace(record_id="snli", dataset_id=snli_id),
    )
    loaded = (records[:1], (), records[1:2], records[2:], {"pins": True})
    monkeypatch.setattr(runner.core, "load_natural_reasoning_data", lambda _root: loaded)
    monkeypatch.setattr(data, "build_synthetic_evaluation_presentations", lambda _rows: ())
    monkeypatch.setattr(data, "build_evaluation_presentations", lambda *_rows: ())
    built: list[dict[str, object]] = []

    def build_payload(**kwargs: object) -> dict[str, object]:
        built.append(kwargs)
        return {"phase": kwargs["phase"]}

    monkeypatch.setattr(runner.core, "build_payload", build_payload)

    _init, training_records, _evaluations, pins = runner._make_payloads("reasoning-run", tmp_path)

    assert tuple(record.record_id for record in training_records) == ("real", "synthetic", "snli")
    assert built[0]["train_records"] == ()
    assert pins == {"pins": True}


def test_arm_payloads_use_canonical_names_and_run_suffixes(monkeypatch) -> None:
    from experiments import mixture_training_data as data

    calls: list[dict[str, object]] = []

    def build_payload(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"arm": kwargs["arm"], "run_id": kwargs["run_id"]}

    monkeypatch.setattr(runner.core, "build_payload", build_payload)

    payloads = runner._make_arm_payloads("reasoning-run", (), (), {}, {"init": True})

    assert tuple(payloads) == data.ARM_NAMES
    assert [payload["run_id"] for payload in payloads.values()] == [
        f"reasoning-run{data.RUN_SUFFIXES[arm]}" for arm in data.ARM_NAMES
    ]
    assert [call["train_records"] for call in calls] == [(), ()]


def test_lifecycle_validates_init_before_running_both_arms_concurrently() -> None:
    init_payload = {"nonce": "init-nonce", "phase": "initialize"}
    calls: list[str] = []
    calls_lock = Lock()
    arms_barrier = Barrier(2)

    def validate(raw: object, payload: dict[str, object]) -> dict[str, object]:
        assert isinstance(raw, dict)
        assert raw["nonce"] == payload["nonce"]
        return raw

    def invoke_init(payload: dict[str, object]) -> dict[str, object]:
        with calls_lock:
            calls.append("initialize")
        return {
            "status": "passed",
            "nonce": payload["nonce"],
            "evidence": {"initialization": {"snapshot": "update-000", "tensor_sha256": "a"}},
        }

    def make_arm_payloads(initialization: object) -> dict[str, dict[str, object]]:
        assert initialization == {"snapshot": "update-000", "tensor_sha256": "a"}
        return {
            "synthetic_repeat": {
                "nonce": "synthetic-nonce",
                "phase": "train",
                "arm": "synthetic_repeat",
            },
            "snli_mix": {
                "nonce": "mix-nonce",
                "phase": "train",
                "arm": "snli_mix",
            },
        }

    def invoke_arm(arm: str, payload: dict[str, object]) -> dict[str, object]:
        with calls_lock:
            assert "initialize" in calls
            calls.append(arm)
        arms_barrier.wait(timeout=2)
        return {"status": "passed", "nonce": payload["nonce"], "evidence": {"arm": arm}}

    outcome = runner._execute_lifecycle(
        init_payload=init_payload,
        invoke_init=invoke_init,
        make_arm_payloads=make_arm_payloads,
        invoke_arm=invoke_arm,
        validate_result=validate,
    )

    assert calls[0] == "initialize"
    assert set(calls[1:]) == {"synthetic_repeat", "snli_mix"}
    assert outcome["results"]["initialize"]["status"] == "passed"
    assert outcome["results"]["synthetic_repeat"]["evidence"] == {"arm": "synthetic_repeat"}
    assert outcome["results"]["snli_mix"]["evidence"] == {"arm": "snli_mix"}


def test_failed_initialization_prevents_both_training_arms() -> None:
    arm_calls: list[str] = []
    payload_builds: list[object] = []
    init_payload = {"nonce": "init-nonce", "phase": "initialize"}

    outcome = runner._execute_lifecycle(
        init_payload=init_payload,
        invoke_init=lambda payload: {
            "status": "failed",
            "nonce": payload["nonce"],
            "evidence": {"initialization": None},
        },
        make_arm_payloads=lambda initialization: payload_builds.append(initialization) or {},
        invoke_arm=lambda arm, payload: arm_calls.append(arm) or {},
        validate_result=lambda raw, payload: raw,
    )

    assert outcome["results"]["initialize"]["status"] == "failed"
    assert payload_builds == []
    assert arm_calls == []
    assert set(outcome["payloads"]) == {"initialize"}


def test_initialization_exception_prevents_arms_and_keeps_unknown_counts() -> None:
    arm_calls: list[str] = []
    init_payload = {"nonce": "init-nonce", "phase": "initialize"}

    def invoke_init(_payload: dict[str, object]) -> object:
        raise TimeoutError("init timed out")

    outcome = runner._execute_lifecycle(
        init_payload=init_payload,
        invoke_init=invoke_init,
        make_arm_payloads=lambda _initialization: {},
        invoke_arm=lambda arm, _payload: arm_calls.append(arm) or {},
        validate_result=lambda raw, _payload: raw,
        make_failed_result=lambda payload, stage, error: {
            "status": "failed",
            "nonce": payload["nonce"],
            "forward_counts": {"total": None},
            "failure": {"stage": stage, "type": type(error).__name__},
        },
    )

    assert arm_calls == []
    assert outcome["results"]["initialize"]["forward_counts"]["total"] is None
    assert outcome["results"]["initialize"]["failure"]["type"] == "TimeoutError"


def test_one_arm_exception_does_not_discard_sibling_result() -> None:
    init_payload = {"nonce": "init-nonce", "phase": "initialize"}
    arm_payloads = {
        "synthetic_repeat": {
            "nonce": "synthetic-nonce",
            "phase": "train",
            "arm": "synthetic_repeat",
        },
        "snli_mix": {
            "nonce": "mix-nonce",
            "phase": "train",
            "arm": "snli_mix",
        },
    }

    def validate(raw: object, payload: dict[str, object]) -> dict[str, object]:
        assert isinstance(raw, dict)
        assert raw["nonce"] == payload["nonce"]
        return raw

    def invoke_arm(arm: str, payload: dict[str, object]) -> dict[str, object]:
        if arm == "synthetic_repeat":
            raise TimeoutError("synthetic repeat worker timed out")
        return {"status": "passed", "nonce": payload["nonce"], "evidence": {"done": True}}

    outcome = runner._execute_lifecycle(
        init_payload=init_payload,
        invoke_init=lambda payload: {
            "status": "passed",
            "nonce": payload["nonce"],
            "evidence": {"initialization": {"snapshot": "update-000", "tensor_sha256": "a"}},
        },
        make_arm_payloads=lambda _initialization: arm_payloads,
        invoke_arm=invoke_arm,
        validate_result=validate,
    )

    assert outcome["results"]["synthetic_repeat"]["status"] == "failed"
    assert outcome["results"]["snli_mix"]["status"] == "passed"
    assert outcome["results"]["snli_mix"]["evidence"] == {"done": True}


def test_bad_recovery_nonce_falls_back_to_original_error_and_unknown_counts() -> None:
    class Volume:
        def read_file(self, _path: str) -> list[bytes]:
            return [b'{"nonce":"forged"}']

    payload = {"run_id": "experiment-init", "nonce": "expected"}

    def validate(raw: object, expected: dict[str, object]) -> dict[str, object]:
        assert isinstance(raw, dict)
        if raw.get("nonce") != expected["nonce"]:
            raise ValueError("recovered progress nonce did not match")
        return raw

    def failed(_payload: dict[str, object], stage: str, error: BaseException) -> dict[str, object]:
        return {
            "status": "failed",
            "failure": {"stage": stage, "type": type(error).__name__},
            "forward_counts": {"total": None},
        }

    result = modal_host._failed_after_remote_error(
        TimeoutError("remote worker timed out"),
        payload,
        Volume(),
        validate_result=validate,
        make_failed_result=failed,
        sanitize=str,
    )

    assert result["status"] == "failed"
    assert result["failure"] == {"stage": "remote_execution", "type": "TimeoutError"}
    assert result["forward_counts"]["total"] is None


def test_modal_workers_have_distinct_names_and_independent_limits(monkeypatch, tmp_path) -> None:
    class ImageBuilder:
        def env(self, _values: dict[str, str]) -> ImageBuilder:
            return self

        def pip_install(self, *_packages: str, **_options: object) -> ImageBuilder:
            return self

        def add_local_file(self, _path: str, *, remote_path: str) -> ImageBuilder:
            raise AssertionError(f"unexpected source package: {remote_path}")

    class ImageFactory:
        @staticmethod
        def debian_slim(*, python_version: str) -> ImageBuilder:
            assert python_version == "3.12"
            return ImageBuilder()

    class FakeApp:
        def __init__(self, _name: str, *, image: ImageBuilder) -> None:
            assert isinstance(image, ImageBuilder)
            self.options: list[dict[str, object]] = []

        def function(self, **options: object):
            self.options.append(options)
            return lambda function: function

    class FakeModal:
        Image = ImageFactory
        App = FakeApp

    monkeypatch.setattr(modal_host.core, "SOURCE_FINGERPRINT_PATHS", ())

    app, initialize, arm_functions = modal_host._register_functions(FakeModal, tmp_path, object())

    assert (
        initialize
        is modal_host.importlib.import_module("experiments.mixture_training_runtime").remote_worker
    )
    assert set(arm_functions) == {"synthetic_repeat", "snli_mix"}
    assert all(function is initialize for function in arm_functions.values())
    assert [options["name"] for options in app.options] == [
        "initialize",
        "synthetic_repeat",
        "snli_mix",
    ]
    assert [options["timeout"] for options in app.options] == [900, 3600, 3600]
    assert all(options["max_containers"] == 1 for options in app.options)
    assert all(options["serialized"] is True for options in app.options)
    assert all(options["include_source"] is False for options in app.options)


def test_default_cli_dry_run_does_not_import_model_or_modal_packages(tmp_path) -> None:
    output = tmp_path / "mixture-plan.json"
    script = f"""
import sys
from experiments import modal_mixture_training as runner
runner._sdk_version = lambda: "1.6.1"
# Keep the dry-run CLI check independent of ignored local training datasets.
runner._make_payloads = lambda *_args: ({{"phase": "initialize"}}, (), (), {{}})
main = runner.main
status = main(["--experiment-id", "dry-run-check", "--output", {str(output)!r}])
assert status == 0
forbidden = ("torch", "transformers", "peft", "modal")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.is_file()


def test_missing_data_returns_failure_receipt_before_modal_or_model_import(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "failed-preflight.json"
    monkeypatch.setattr(runner, "_sdk_version", lambda: "1.6.1")

    status = runner.main(
        [
            "--experiment-id",
            "missing-data-check",
            "--root",
            str(tmp_path / "empty-root"),
            "--output",
            str(output),
        ]
    )

    assert status == 1
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["payloads"] == {}
    assert receipt["results"] == {}
    assert receipt["failure"]["stage"] == "preflight_or_execution"
    assert not any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in ("torch", "transformers", "peft", "modal")
    )


def _run_main_with_fake_modal(
    tmp_path, monkeypatch, *, fail_app_exit: bool, fail_arm_payloads: bool
) -> tuple[int, dict[str, object]]:
    output = tmp_path / "failed-execution.json"
    init_payload = {
        "experiment_id": "exit-check",
        "run_id": "exit-check-init",
        "nonce": "init",
        "phase": "initialize",
        "arm": None,
        "pins": {},
        "payload_sha256": "init-hash",
    }
    arm_payloads = {
        "synthetic_repeat": {
            "experiment_id": "exit-check",
            "run_id": "exit-check-syn",
            "nonce": "real",
            "phase": "train",
            "arm": "synthetic_repeat",
            "pins": {},
            "payload_sha256": "real-hash",
        },
        "snli_mix": {
            "experiment_id": "exit-check",
            "run_id": "exit-check-snli",
            "nonce": "mix",
            "phase": "train",
            "arm": "snli_mix",
            "pins": {},
            "payload_sha256": "mix-hash",
        },
    }

    class AppContext:
        app_id = "ap-exit-check"

        def run(self):
            class ExitFailure:
                def __enter__(self):
                    return None

                def __exit__(self, exception_type, *_args):
                    if fail_app_exit and exception_type is None:
                        raise RuntimeError("app shutdown failed")
                    return False

            return ExitFailure()

    class RemoteFunction:
        def remote(self, payload: dict[str, object]) -> dict[str, object]:
            if payload["phase"] == "initialize":
                return {
                    "status": "passed",
                    "nonce": payload["nonce"],
                    "evidence": {"initialization": {"snapshot": "snapshot-1"}},
                }
            return {
                "status": "passed",
                "nonce": payload["nonce"],
                "evidence": {"arm": payload["arm"]},
            }

    app = AppContext()
    modal = type(
        "FakeModal",
        (),
        {
            "Volume": type(
                "FakeVolume",
                (),
                {"from_name": staticmethod(lambda *_args, **_kwargs: object())},
            ),
            "enable_output": staticmethod(lambda: __import__("contextlib").nullcontext()),
        },
    )
    modules = {
        "experiments.modal_smoke": type(
            "FakeModalSmoke", (), {"_verify_profile": staticmethod(lambda *_args: True)}
        ),
        "modal": modal,
    }

    def make_arm_payloads(*_args):
        if fail_arm_payloads:
            raise RuntimeError("training payload construction failed")
        return arm_payloads

    monkeypatch.setattr(runner, "_sdk_version", lambda: "1.6.1")
    monkeypatch.setattr(
        runner,
        "_make_payloads",
        lambda *_args: (init_payload, (), (), {}),
    )
    monkeypatch.setattr(runner, "_make_arm_payloads", make_arm_payloads)
    monkeypatch.setattr(runner.core, "validate_result", lambda raw, _payload: raw)
    monkeypatch.setattr(modal_host.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(modal_host.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(
        modal_host,
        "_register_functions",
        lambda *_args: (
            app,
            RemoteFunction(),
            {arm: RemoteFunction() for arm in ("synthetic_repeat", "snli_mix")},
        ),
    )

    status = runner.main(
        [
            "--experiment-id",
            "exit-check",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--execute",
        ]
    )
    receipt = json.loads(output.read_text(encoding="utf-8"))
    return status, receipt


def test_main_preserves_worker_results_when_modal_app_exit_fails(tmp_path, monkeypatch) -> None:
    status, receipt = _run_main_with_fake_modal(
        tmp_path, monkeypatch, fail_app_exit=True, fail_arm_payloads=False
    )
    assert status == 1
    assert receipt["status"] == "failed"
    assert receipt["modal"]["app_id"] == "ap-exit-check"
    assert set(receipt["payloads"]) == {"initialize", "synthetic_repeat", "snli_mix"}
    assert set(receipt["results"]) == {"initialize", "synthetic_repeat", "snli_mix"}
    assert all(result["status"] == "passed" for result in receipt["results"].values())
    assert receipt["failure"]["stage"] == "modal_context_exit"


def test_main_preserves_init_result_when_arm_payload_building_fails(tmp_path, monkeypatch) -> None:
    status, receipt = _run_main_with_fake_modal(
        tmp_path, monkeypatch, fail_app_exit=False, fail_arm_payloads=True
    )

    assert status == 1
    assert receipt["status"] == "failed"
    assert receipt["modal"]["app_id"] == "ap-exit-check"
    assert set(receipt["payloads"]) == {"initialize"}
    assert set(receipt["results"]) == {"initialize"}
    assert receipt["results"]["initialize"]["status"] == "passed"
    assert receipt["failure"]["stage"] == "preflight_or_execution"
