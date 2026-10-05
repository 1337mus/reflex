"""Plan or explicitly launch the pinned adapter transfer evaluation on Modal."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import uuid
from pathlib import Path

from experiments import adapter_transfer_contracts as contracts
from experiments import adapter_transfer_core as core
from experiments import adapter_transfer_runtime as runtime
from experiments import modal_smoke
from reflex_decisions import smoke

PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
VOLUME_NAME = "reflex-rehearsal-artifacts"
VOLUME_ROOT = "/artifacts"
EXPECTED_MODAL_VERSION = "1.6.1"
RUN_ID = "adapter-transfer-2026-10-05-r1"
EXPERIMENT_ID = contracts.EXPERIMENT_ID
SCHEMA_VERSION = contracts.SCHEMA_VERSION
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _function_options() -> dict[str, object]:
    return {
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
        "timeout": 900,
    }


def plan(run_id: str, output: str) -> dict[str, object]:
    limits = _function_options()
    return {
        "mode": "plan-only",
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "output": output,
        "model_id": smoke.MODEL_ID,
        "model_revision": smoke.MODEL_REVISION,
        "presentations": 128,
        "parity_presentations": 16,
        "max_total_forwards": 272,
        "max_total_input_tokens": 557056,
        "profile": PROFILE,
        "workspace": WORKSPACE,
        "modal_sdk_version": EXPECTED_MODAL_VERSION,
        "modal": limits,
    }


def _load_payload(args: argparse.Namespace, project_root) -> dict[str, object]:
    from experiments import adapter_transfer_core as core
    from experiments import adapter_transfer_data as data

    _records, presentations, selection, pins = data.load_inputs(project_root)
    nonce = str(uuid.uuid4())
    payload = core.build_payload(
        args.run_id,
        nonce,
        presentations,
        selection,
        pins,
        root=project_root,
    )
    return core.validate_payload(payload, root=project_root)


def _host_failure(run_id: str, stage: str, error: BaseException) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "failed",
        "phase": stage,
        "payload": None,
        "result": None,
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": smoke.sanitize_exception_message(error),
        },
        "modal": {
            "profile": PROFILE,
            "workspace": WORKSPACE,
            "sdk_version": None,
            "limits": _function_options(),
        },
    }


def _dry_run_receipt(run_id: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "dry_run",
        "phase": "preflight_complete",
        "payload": payload,
        "result": None,
        "failure": None,
        "modal": {
            "profile": PROFILE,
            "workspace": WORKSPACE,
            "sdk_version": None,
            "limits": _function_options(),
        },
    }


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": smoke.sanitize_exception_message(error),
    }


def _receipt(
    run_id: str,
    payload: dict[str, object],
    result: dict[str, object],
    *,
    modal_version: str | None,
    lifecycle_failure: BaseException | None = None,
) -> dict[str, object]:
    failure = _failure("modal_lifecycle", lifecycle_failure) if lifecycle_failure else None
    status = "passed" if result.get("status") == "passed" and failure is None else "failed"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "phase": (
            "complete"
            if status == "passed"
            else ("modal_lifecycle_failed" if failure else result.get("phase"))
        ),
        "payload": payload,
        "result": result,
        "failure": failure or result.get("failure"),
        "modal": {
            "profile": PROFILE,
            "workspace": WORKSPACE,
            "sdk_version": modal_version,
            "limits": _function_options(),
        },
    }


def _failed_worker_result(
    payload: dict[str, object],
    stage: str,
    error: BaseException,
    *,
    known_unstarted: bool,
    partial: object = None,
) -> dict[str, object]:
    initial = runtime._initial_result(payload)
    provenance = initial["provenance"] if known_unstarted else None
    evidence = initial["evidence"] if known_unstarted else None
    if isinstance(partial, dict):
        provenance = partial.get("provenance", provenance)
        evidence = partial.get("evidence", evidence)
    try:
        failed = core.build_failed_result(
            payload,
            stage=stage,
            error=error,
            sanitize=smoke.sanitize_exception_message,
            provenance=provenance,
            evidence=evidence,
        )
    except Exception:
        if not isinstance(partial, dict) or not isinstance(partial.get("evidence"), dict):
            raise
        # A bad provenance field must not erase otherwise valid worker evidence.
        failed = core.build_failed_result(
            payload,
            stage=stage,
            error=error,
            sanitize=smoke.sanitize_exception_message,
            provenance=initial["provenance"],
            evidence=partial["evidence"],
        )
    return core.validate_result(failed, payload)


def _remote_file_path(relative: str) -> str:
    return f"/root/{relative}"


def _build_modal_image(modal, project_root: Path):
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .env(
            {
                "PYTHONPATH": "/root:/root/src",
                "USE_HUB_KERNELS": "NO",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            }
        )
        .pip_install(
            *(
                f"{package}=={version}"
                for package, version in runtime.mixture_contracts.RUNTIME_VERSION_PINS.items()
            ),
            extra_index_url="https://download.pytorch.org/whl/cu130",
        )
    )
    for relative in contracts.SOURCE_FINGERPRINT_PATHS:
        image = image.add_local_file(
            str(project_root / relative), remote_path=_remote_file_path(relative)
        )
    return image


def _launch(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(args.root).expanduser().resolve() if args.root else PROJECT_ROOT
    payload: dict[str, object] | None = None
    modal_version: str | None = None
    stage = "data_preflight"
    try:
        payload = _load_payload(args, project_root)
        stage = "credential_preflight"
        if any(os.environ.get(name, "") for name in modal_smoke.CREDENTIAL_OVERRIDES):
            raise RuntimeError("Modal credential environment overrides are not accepted")
        stage = "profile_verification"
        if not modal_smoke._verify_profile(PROFILE, WORKSPACE):
            raise RuntimeError("Modal token info did not match the fixed personal workspace")
        stage = "modal_version"
        modal_version = importlib.metadata.version("modal")
        if modal_version != EXPECTED_MODAL_VERSION:
            raise RuntimeError("Modal SDK 1.6.1 is required")
        stage = "modal_setup"
        modal = importlib.import_module("modal")
        volume = modal.Volume.from_name(VOLUME_NAME).with_mount_options(read_only=True)
        image = _build_modal_image(modal, project_root)
        app = modal.App("reflex-adapter-transfer-evaluation", image=image)
        function = app.function(**_function_options(), volumes={runtime.VOLUME_ROOT: volume})(
            runtime.remote_worker
        )
        stage = "worker_execution"
        worker_result: dict[str, object] | None = None
        raw_result: object = None
        lifecycle_failure: BaseException | None = None
        validation_failure: BaseException | None = None
        try:
            with modal.enable_output(), app.run():
                raw_result = function.remote(payload)
                if isinstance(raw_result, dict):
                    try:
                        worker_result = core.validate_result(raw_result, payload)
                    except Exception as exc:
                        validation_failure = exc
                else:
                    validation_failure = ValueError("Modal worker returned no result object")
        except Exception as exc:
            lifecycle_failure = exc
        if worker_result is None:
            if validation_failure is not None:
                try:
                    worker_result = _failed_worker_result(
                        payload,
                        "result_validation",
                        validation_failure,
                        known_unstarted=False,
                        partial=raw_result,
                    )
                except Exception:
                    worker_result = _failed_worker_result(
                        payload,
                        "result_validation",
                        validation_failure,
                        known_unstarted=False,
                    )
            else:
                error = lifecycle_failure or RuntimeError("Modal worker returned no result")
                worker_result = _failed_worker_result(
                    payload,
                    "modal_lifecycle" if lifecycle_failure else stage,
                    error,
                    known_unstarted=False,
                )
        return _receipt(
            args.run_id,
            payload,
            worker_result,
            modal_version=modal_version,
            lifecycle_failure=lifecycle_failure,
        )
    except Exception as exc:
        if payload is None:
            return _host_failure(args.run_id, stage, exc)
        result = _failed_worker_result(payload, stage, exc, known_unstarted=True)
        return _receipt(args.run_id, payload, result, modal_version=modal_version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", help="project root containing pinned inputs")
    parser.add_argument("--execute", action="store_true", help="explicitly launch remote inference")
    args = parser.parse_args(argv)
    try:
        with smoke.reserve_output(args.output) as reservation:
            try:
                if args.execute:
                    receipt = _launch(args)
                else:
                    project_root = (
                        Path(args.root).expanduser().resolve() if args.root else PROJECT_ROOT
                    )
                    payload = _load_payload(args, project_root)
                    receipt = _dry_run_receipt(args.run_id, payload)
            except Exception as exc:
                stage = "host_preflight" if args.execute else "data_preflight"
                receipt = _host_failure(args.run_id, stage, exc)
            smoke.write_json_artifact(reservation, receipt)
        status = receipt.get("status")
        print(json.dumps({"status": status, "artifact": str(reservation.destination)}))
        return 0 if status in {"passed", "dry_run"} else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "failure": {
                        "type": type(exc).__name__,
                        "message": smoke.sanitize_exception_message(exc),
                    },
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
