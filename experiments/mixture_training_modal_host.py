"""Modal packaging, bounded workers, and best-effort progress recovery."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from experiments import mixture_training_core as core
from experiments import mixture_training_runtime_helpers as helpers

VOLUME_NAME = "reflex-rehearsal-artifacts"
APP_NAME = "reflex-controlled-mixture-training"
EXPECTED_MODAL_VERSION = "1.6.1"
_CREDENTIAL_OVERRIDES = (
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_OAUTH_REFRESH_TOKEN",
    "MODAL_OAUTH_CLIENT_ID",
    "MODAL_OAUTH_CLIENT_SECRET",
)


def _remote_progress(volume: Any, payload: dict[str, object]) -> dict[str, object] | None:
    run_id = str(payload["run_id"])
    path = f"runs/{core.validate_safe_run_id(run_id)}/progress.json"
    chunks = volume.read_file(path)
    if isinstance(chunks, bytes):
        raise ValueError("Modal progress read did not return a byte stream")
    pieces = list(chunks)
    if any(not isinstance(chunk, bytes) for chunk in pieces):
        raise ValueError("Modal progress stream contained a non-byte chunk")
    return helpers.strict_json_object(b"".join(pieces))


def _failed_after_remote_error(
    error: BaseException,
    payload: dict[str, object],
    volume: Any,
    *,
    validate_result: Callable[[object, dict[str, object]], dict[str, object]],
    make_failed_result: Callable[[dict[str, object], str, BaseException], dict[str, object]],
    sanitize: Callable[[BaseException], str],
) -> dict[str, object]:
    try:
        recovered = _remote_progress(volume, payload)
        if recovered is not None:
            result = validate_result(recovered, payload)
            result["status"] = "failed"
            result["failure"] = {
                "stage": "remote_execution",
                "type": type(error).__name__,
                "message": sanitize(error),
            }
            return validate_result(result, payload)
    except Exception:
        # Recovery is best effort. Keep the original remote exception as the cause.
        pass
    return make_failed_result(payload, "remote_execution", error)


def _image(modal: Any, root: Path) -> Any:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .env({"USE_HUB_KERNELS": "NO", "PYTHONPATH": "/root:/root/src"})
        .pip_install(
            "torch==2.14.1+cu130",
            "torchvision==0.29.1+cu130",
            "transformers==5.18.0",
            "peft==0.21.0",
            "Pillow==12.0.0",
            "pydantic==2.13.5",
            "huggingface-hub==1.33.0",
            "tokenizers==0.23.2",
            "safetensors==0.8.0",
            extra_index_url="https://download.pytorch.org/whl/cu130",
        )
    )
    for relative in core.SOURCE_FINGERPRINT_PATHS:
        image = image.add_local_file(
            str((root / relative).resolve(strict=True)),
            remote_path=f"/root/{relative}",
        )
    return image


def _register_functions(modal: Any, root: Path, volume: Any) -> tuple[Any, Any, Any, Any]:
    runtime = importlib.import_module("experiments.mixture_training_runtime")
    app = modal.App(APP_NAME, image=_image(modal, root))

    def register(function_name: str, timeout: int) -> Any:
        return app.function(
            name=function_name,
            gpu="A10",
            cpu=(2.0, 2.0),
            memory=(16384, 16384),
            max_containers=1,
            min_containers=0,
            buffer_containers=0,
            scaledown_window=2,
            retries=0,
            single_use_containers=True,
            serialized=True,
            include_source=False,
            startup_timeout=300,
            timeout=timeout,
            volumes={"/artifacts": volume},
        )(runtime.remote_worker)

    return (
        app,
        register("initialize", 900),
        register("real_only", 3600),
        register("synthetic_mix", 3600),
    )


def launch_modal(
    *,
    init_payload: dict[str, object],
    make_arm_payloads: Callable[[object], Mapping[str, dict[str, object]]],
    validate_result: Callable[[object, dict[str, object]], dict[str, object]],
    make_failed_result: Callable[[dict[str, object], str, BaseException], dict[str, object]],
    execute_lifecycle: Callable[..., dict[str, object]],
    sanitize: Callable[[BaseException], str],
    profile: str,
    workspace: str,
    root: Path,
    receipt: dict[str, object],
    invalid_responses: dict[str, object],
    response_lock: Lock,
) -> dict[str, object]:
    """Launch one init worker, then two separately bounded concurrent arm workers."""

    if profile != core.PROFILE or workspace != core.WORKSPACE:
        raise ValueError("only the pinned personal Modal profile and workspace are allowed")
    if importlib.metadata.version("modal") != EXPECTED_MODAL_VERSION:
        raise RuntimeError("Modal 1.6.1 is required")
    if any(os.environ.get(name, "") for name in _CREDENTIAL_OVERRIDES):
        raise RuntimeError("Modal credential environment overrides are not accepted")

    modal_smoke = importlib.import_module("experiments.modal_smoke")
    if not modal_smoke._verify_profile(profile, workspace):
        raise RuntimeError("Modal token info did not match the expected workspace")
    modal = importlib.import_module("modal")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    app, init_function, real_function, mix_function = _register_functions(modal, root, volume)

    def invoke(function: Any, payload: dict[str, object]) -> object:
        try:
            return function.remote(payload)
        except Exception as error:
            return _failed_after_remote_error(
                error,
                payload,
                volume,
                validate_result=validate_result,
                make_failed_result=make_failed_result,
                sanitize=sanitize,
            )

    def checked(raw: object, payload: dict[str, object]) -> dict[str, object]:
        key = str(payload["arm"] or "initialize")
        try:
            result = validate_result(raw, payload)
        except Exception:
            with response_lock:
                invalid_responses[key] = raw
            raise
        remember_result(result, payload)
        return result

    receipt_payloads = receipt.get("payloads")
    if not isinstance(receipt_payloads, dict):
        receipt_payloads = {}
        receipt["payloads"] = receipt_payloads
    receipt_results = receipt.get("results")
    if not isinstance(receipt_results, dict):
        receipt_results = {}
        receipt["results"] = receipt_results
    receipt_payloads["initialize"] = init_payload

    def remember_result(result: dict[str, object], payload: dict[str, object]) -> None:
        key = str(payload.get("arm") or "initialize")
        with response_lock:
            receipt_results[key] = result

    def make_recorded_arm_payloads(initialization: object) -> Mapping[str, dict[str, object]]:
        arm_payloads = dict(make_arm_payloads(initialization))
        receipt_payloads.update(arm_payloads)
        return arm_payloads

    def make_recorded_failed_result(
        payload: dict[str, object], stage: str, error: BaseException
    ) -> dict[str, object]:
        result = make_failed_result(payload, stage, error)
        remember_result(result, payload)
        return result

    lifecycle: dict[str, object] | None = None
    try:
        with modal.enable_output():
            with app.run():
                app_id = getattr(app, "app_id", None)
                if isinstance(app_id, str):
                    receipt["modal"]["app_id"] = app_id
                lifecycle = execute_lifecycle(
                    init_payload=init_payload,
                    invoke_init=lambda payload: invoke(init_function, payload),
                    make_arm_payloads=make_recorded_arm_payloads,
                    invoke_arm=lambda arm, payload: invoke(
                        real_function if arm == "real_only" else mix_function, payload
                    ),
                    validate_result=checked,
                    make_failed_result=make_recorded_failed_result,
                )
    except Exception as error:
        if lifecycle is not None:
            receipt["status"] = "failed"
            receipt["failure"] = {
                "stage": "modal_context_exit",
                "type": type(error).__name__,
                "message": sanitize(error),
            }
        raise
    if lifecycle is None:
        raise RuntimeError("Modal app exited without completing the mixture lifecycle")
    return lifecycle
