from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def test_function_options_are_the_single_bounded_a10_worker() -> None:
    from experiments import modal_fresh_eval as runner

    assert runner._function_options() == {
        "gpu": "A10",
        "cpu": (2.0, 2.0),
        "memory": (16384, 16384),
        "max_containers": 1,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window": 2,
        "retries": 0,
        "single_use_containers": True,
        "serialized": False,
        "include_source": False,
        "startup_timeout": 300,
        "timeout": 1800,
    }


def test_run_marker_path_is_independent_of_receipt_output() -> None:
    from pathlib import Path

    from experiments import modal_fresh_eval as runner

    root = Path("/project")
    assert runner._run_marker_path(root, "fresh-eval-2026-10-05-r1") == (
        root / "artifacts" / "fresh-eval-2026-10-05-r1.fresh-eval.run-marker.json"
    )


def test_launch_binds_volume_and_app_run_to_main_environment(monkeypatch, tmp_path) -> None:
    from experiments import modal_fresh_eval as runner

    captured: dict[str, object] = {}
    payload = {"payload": "frozen"}
    result = {
        "status": "failed",
        "phase": "preflight_failed",
        "failure": {"stage": "preflight"},
        "evidence": {"unknown_work": None},
    }
    monkeypatch.setenv("MODAL_ENVIRONMENT", "unexpected")
    monkeypatch.setattr(runner, "_load_payload", lambda *_args: payload)
    monkeypatch.setattr(runner.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner, "_build_modal_image", lambda *_args: object())
    monkeypatch.setattr(runner.core, "validate_result", lambda *_args, **_kwargs: result)

    class Function:
        def remote(self, _payload):
            return result

    class App:
        def function(self, **_kwargs):
            return lambda _worker: Function()

        def run(self, **kwargs):
            captured["app_run"] = kwargs
            return nullcontext()

    def from_name(_name, **kwargs):
        captured["volume"] = kwargs
        return SimpleNamespace(with_mount_options=lambda **_kwargs: object())

    modal = SimpleNamespace(
        Volume=SimpleNamespace(from_name=from_name),
        App=lambda *_args, **_kwargs: App(),
        enable_output=nullcontext,
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: modal)

    runner._launch(SimpleNamespace(run_id="fresh-eval-2026-10-05-r1"), tmp_path, "marker")

    assert captured == {
        "volume": {"environment_name": "main"},
        "app_run": {"environment_name": "main"},
    }


def test_launch_records_unknown_work_when_remote_is_interrupted(monkeypatch, tmp_path) -> None:
    from experiments import modal_fresh_eval as runner

    payload = {"payload": "frozen"}
    failure = {
        "status": "failed",
        "phase": "modal_lifecycle_failed",
        "failure": {"stage": "modal_lifecycle"},
        "evidence": {
            "unknown_work": {
                "possible_forwards": 2824,
                "reason": "remote execution may have started before its result was unavailable",
            }
        },
    }
    monkeypatch.setattr(runner, "_load_payload", lambda *_args: payload)
    monkeypatch.setattr(runner.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner, "_build_modal_image", lambda *_args: object())
    monkeypatch.setattr(runner, "_worker_failure", lambda *_args: failure)

    class Function:
        def remote(self, _payload):
            raise KeyboardInterrupt("interrupted")

    class App:
        def function(self, **_kwargs):
            return lambda _worker: Function()

        def run(self, **_kwargs):
            return nullcontext()

    modal = SimpleNamespace(
        Volume=SimpleNamespace(
            from_name=lambda _name, **_kwargs: SimpleNamespace(
                with_mount_options=lambda **_kwargs: object()
            )
        ),
        App=lambda *_args, **_kwargs: App(),
        enable_output=nullcontext,
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: modal)

    receipt = runner._launch(SimpleNamespace(run_id="fresh-eval-2026-10-05-r1"), tmp_path, "marker")

    assert receipt["status"] == "failed"
    assert receipt["raw_worker_result"] is None
    assert receipt["execution"]["unknown_work"]["possible_forwards"] == 2824


def test_worker_failure_uses_exception_type_when_keyboard_interrupt_message_is_blank(
    monkeypatch,
) -> None:
    from experiments import modal_fresh_eval as runner

    payload = {
        "schema_version": 1,
        "experiment_id": "fresh-eval-v1",
        "run_id": "fresh-eval-2026-10-05-r1",
        "nonce": "nonce",
        "selection_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "pins": {
            "model_id": "Qwen/Qwen3.5-0.8B-Base",
            "model_revision": "revision",
            "source_file_sha256": {},
        },
    }
    monkeypatch.setattr(runner.core, "validate_result", lambda result, _payload: result)

    result = runner._worker_failure(payload, "modal_lifecycle", KeyboardInterrupt())

    assert result["failure"]["message"] == "KeyboardInterrupt"
    assert result["evidence"]["unknown_work"]["possible_forwards"] == 2824


@pytest.mark.parametrize("raise_on_exit", [False, True])
@pytest.mark.parametrize("validation_error", [ValueError("bad"), KeyboardInterrupt()])
def test_launch_retains_raw_worker_result_and_marks_unknown_work(
    monkeypatch, tmp_path, raise_on_exit: bool, validation_error: BaseException
) -> None:
    from experiments import modal_fresh_eval as runner

    payload = {"payload": "frozen"}
    raw_result = {"unexpected": "worker receipt"}
    failure = {
        "status": "failed",
        "phase": "result_validation_failed",
        "failure": {"stage": "result_validation"},
        "evidence": {"unknown_work": {"possible_forwards": 2824, "reason": "unknown"}},
    }
    monkeypatch.setattr(runner, "_load_payload", lambda *_args: payload)
    monkeypatch.setattr(runner.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.1")
    monkeypatch.setattr(runner, "_build_modal_image", lambda *_args: object())
    monkeypatch.setattr(runner, "_worker_failure", lambda *_args: failure)
    monkeypatch.setattr(
        runner.core,
        "validate_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(validation_error),
    )

    class Function:
        def remote(self, received):
            assert received is payload
            return raw_result

    class App:
        def function(self, **_kwargs):
            return lambda _worker: Function()

        def run(self, **_kwargs):
            if not raise_on_exit:
                return nullcontext()

            class ExitFailure:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    raise RuntimeError("app context failed after result")

            return ExitFailure()

    modal = SimpleNamespace(
        Volume=SimpleNamespace(
            from_name=lambda _name, **_kwargs: SimpleNamespace(
                with_mount_options=lambda **_kwargs: object()
            )
        ),
        App=lambda *_args, **_kwargs: App(),
        enable_output=nullcontext,
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: modal)

    receipt = runner._launch(SimpleNamespace(run_id="fresh-eval-2026-10-05-r1"), tmp_path, "marker")

    assert receipt["raw_worker_result"] == raw_result
    assert receipt["execution"]["unknown_work"]["possible_forwards"] == 2824
