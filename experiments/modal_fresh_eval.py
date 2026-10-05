"""Explicit launcher for the bounded, inference-only fresh evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import uuid
from pathlib import Path

from experiments import fresh_eval_core as core
from experiments import fresh_eval_runtime as runtime
from experiments import modal_smoke
from reflex_decisions import smoke

PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
ENVIRONMENT = "main"
VOLUME_NAME = "reflex-rehearsal-artifacts"
VOLUME_ROOT = "/artifacts"
EXPECTED_MODAL_VERSION = "1.6.1"
RUN_ID = "fresh-eval-2026-10-05-r1"
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
        "timeout": 1800,
    }


def plan(run_id: str, output: str) -> dict[str, object]:
    return {
        "mode": "plan-only",
        "experiment_id": core.EXPERIMENT_ID,
        "run_id": run_id,
        "output": output,
        "presentations": core.BASE_PRESENTATION_COUNT,
        "parity_presentations": core.PARITY_PRESENTATION_COUNT,
        "max_total_forwards": core.MAX_TOTAL_FORWARDS,
        "max_total_input_tokens": core.MAX_TOTAL_INPUT_TOKENS,
        "profile": PROFILE,
        "workspace": WORKSPACE,
        "environment": ENVIRONMENT,
        "modal_sdk_version": EXPECTED_MODAL_VERSION,
        "tokenizer_file_sha256": core.TOKENIZER_FILE_SHA256,
        "tokenizers_version": core.TOKENIZERS_VERSION,
        "modal": _function_options(),
    }


def _host_tokenizer() -> object:
    """Use an already-cached, revision-pinned tokenizer; never download a model locally."""

    if importlib.metadata.version("tokenizers") != core.TOKENIZERS_VERSION:
        raise RuntimeError("local tokenizers package differs from the reviewed version")
    from tokenizers import Tokenizer

    path = PROJECT_ROOT / ".cache/qwen-tokenizer.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != core.TOKENIZER_FILE_SHA256:
        raise RuntimeError("cached tokenizer file differs from its reviewed SHA-256")
    tokenizer = Tokenizer.from_file(str(path))

    class PinnedTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    return PinnedTokenizer()


def _exception_message(error: BaseException) -> str:
    return smoke.sanitize_exception_message(error).strip() or type(error).__name__


def _run_marker_path(project_root: Path, run_id: str) -> Path:
    return project_root / "artifacts" / f"{run_id}.fresh-eval.run-marker.json"


def _load_payload(args: argparse.Namespace, project_root: Path) -> dict[str, object]:
    payload, _host_metadata = core.prepare_payload(
        project_root, args.run_id, str(uuid.uuid4()), _host_tokenizer()
    )
    return core.validate_payload(payload, root=project_root)


def _remote_file_path(relative: str) -> str:
    return f"/root/{relative}"


def _build_modal_image(modal: object, project_root: Path):
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
    for relative in core.SOURCE_FINGERPRINT_PATHS:
        image = image.add_local_file(
            str(project_root / relative), remote_path=_remote_file_path(relative)
        )
    return image


def _worker_failure(
    payload: dict[str, object], stage: str, error: BaseException
) -> dict[str, object]:
    result = runtime._initial_result(payload)
    result.update(
        {
            "phase": f"{stage}_failed",
            "failure": {
                "stage": stage,
                "type": type(error).__name__,
                "message": _exception_message(error),
            },
        }
    )
    if stage in {"modal_lifecycle", "worker_execution", "result_validation"}:
        result["evidence"]["unknown_work"] = {
            "possible_forwards": core.MAX_TOTAL_FORWARDS,
            "reason": "remote execution may have started before its result was unavailable",
        }
    return core.validate_result(result, payload)


def _receipt(
    run_id: str,
    payload: dict[str, object],
    result: dict[str, object],
    *,
    modal_version: str | None,
    run_marker: str | None,
    raw_worker_result: object = None,
) -> dict[str, object]:
    passed = result.get("status") == "passed"
    return {
        "schema_version": core.SCHEMA_VERSION,
        "experiment_id": core.EXPERIMENT_ID,
        "run_id": run_id,
        "status": "passed" if passed else "failed",
        "phase": "complete" if passed else result.get("phase"),
        "payload": payload,
        "result": result,
        "raw_worker_result": raw_worker_result,
        "failure": None if passed else result.get("failure"),
        "execution": {
            "run_marker": run_marker,
            "unknown_work": result.get("evidence", {}).get("unknown_work"),
        },
        "modal": {
            "profile": PROFILE,
            "workspace": WORKSPACE,
            "environment": ENVIRONMENT,
            "sdk_version": modal_version,
            "limits": _function_options(),
            "volume_read_only": True,
        },
    }


def _launch(
    args: argparse.Namespace, project_root: Path, run_marker: str | None
) -> dict[str, object]:
    payload: dict[str, object] | None = None
    version: str | None = None
    stage = "data_preflight"
    try:
        payload = _load_payload(args, project_root)
        stage = "credential_preflight"
        if any(os.environ.get(name, "") for name in modal_smoke.CREDENTIAL_OVERRIDES):
            raise RuntimeError("Modal credential environment overrides are not accepted")
        stage = "profile_verification"
        if not modal_smoke._verify_profile(PROFILE, WORKSPACE):
            raise RuntimeError("Modal profile did not match the fixed personal workspace")
        stage = "modal_version"
        version = importlib.metadata.version("modal")
        if version != EXPECTED_MODAL_VERSION:
            raise RuntimeError("Modal SDK 1.6.1 is required")
        modal = importlib.import_module("modal")
        volume = modal.Volume.from_name(
            VOLUME_NAME, environment_name=ENVIRONMENT
        ).with_mount_options(read_only=True)
        app = modal.App("reflex-fresh-evaluation", image=_build_modal_image(modal, project_root))
        function = app.function(**_function_options(), volumes={VOLUME_ROOT: volume})(
            runtime.remote_worker
        )
        stage = "worker_execution"
        raw_result: object = None
        try:
            with modal.enable_output(), app.run(environment_name=ENVIRONMENT):
                raw_result = function.remote(payload)
        except (Exception, KeyboardInterrupt) as exc:
            result = _worker_failure(payload, "modal_lifecycle", exc)
        else:
            try:
                result = core.validate_result(raw_result, payload, root=project_root)
            except (Exception, KeyboardInterrupt) as exc:
                result = _worker_failure(payload, "result_validation", exc)
        return _receipt(
            args.run_id,
            payload,
            result,
            modal_version=version,
            run_marker=run_marker,
            raw_worker_result=raw_result,
        )
    except Exception as exc:
        if payload is None:
            return {
                "schema_version": core.SCHEMA_VERSION,
                "experiment_id": core.EXPERIMENT_ID,
                "run_id": args.run_id,
                "status": "failed",
                "phase": f"{stage}_failed",
                "payload": None,
                "result": None,
                "failure": {
                    "stage": stage,
                    "type": type(exc).__name__,
                    "message": _exception_message(exc),
                },
                "execution": {"run_marker": run_marker, "unknown_work": None},
                "modal": {
                    "profile": PROFILE,
                    "workspace": WORKSPACE,
                    "environment": ENVIRONMENT,
                    "sdk_version": version,
                    "limits": _function_options(),
                    "volume_read_only": True,
                },
            }
        return _receipt(
            args.run_id,
            payload,
            _worker_failure(payload, stage, exc),
            modal_version=version,
            run_marker=run_marker,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(args.root).expanduser().resolve() if args.root else PROJECT_ROOT
    try:
        with smoke.reserve_output(args.output) as reservation:
            run_marker: str | None = None
            if args.execute:
                marker_path = _run_marker_path(project_root, args.run_id)
                with smoke.reserve_output(marker_path) as marker:
                    smoke.write_json_artifact(
                        marker, {"experiment_id": core.EXPERIMENT_ID, "run_id": args.run_id}
                    )
                    run_marker = str(marker.destination)
                receipt = _launch(args, project_root, run_marker)
            else:
                payload = _load_payload(args, project_root)
                receipt = _receipt(
                    args.run_id,
                    payload,
                    runtime._initial_result(payload),
                    modal_version=None,
                    run_marker=None,
                )
                receipt["status"] = "dry_run"
                receipt["phase"] = "preflight_complete"
                receipt["failure"] = None
            smoke.write_json_artifact(reservation, receipt)
        print(json.dumps({"status": receipt["status"], "artifact": str(reservation.destination)}))
        return 0 if receipt["status"] in {"passed", "dry_run"} else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "failure": {
                        "type": type(exc).__name__,
                        "message": _exception_message(exc),
                    },
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
