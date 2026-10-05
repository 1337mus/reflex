"""CPU planning and guarded, reference-first Modal execution for targeted training."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from experiments.targeted_training_contracts import (
    BUNDLE_MANIFEST_SHA256,
    ENVIRONMENT,
    MODAL_SDK_VERSION,
    PROFILE,
    ROLE_RESOURCES,
    TOKENIZER_FILE_SHA256,
    WORKSPACE,
)
from experiments.targeted_training_core import (
    EXPERIMENT_ID,
    ROLE_CONTROL,
    ROLE_TREATMENT,
    ROLE_UNCHANGED,
)

VOLUME_NAME = "reflex-rehearsal-artifacts"
VOLUME_ROOT = "/artifacts"
_REMOTE_CANCEL_TIMEOUT_SECONDS = 5.0
_REMOTE_CANCEL_BATCH_GRACE_SECONDS = 1.0
COORDINATOR_EXECUTION = {
    "cpu": 1,
    "memory_mib": 4096,
    "timeout_seconds": 9000,
    "startup_timeout_seconds": 300,
    "max_containers": 1,
    "max_inputs": 1,
    "scaledown_window_seconds": 2,
    "retries": 0,
    "single_use_containers": True,
}


def _modal_reference_worker(payload: dict[str, object]) -> dict[str, object]:
    _claim_worker_role(payload, ROLE_UNCHANGED)
    from experiments import targeted_training_runtime as runtime

    return runtime.run_worker(payload)


def _modal_control_worker(payload: dict[str, object]) -> dict[str, object]:
    _claim_worker_role(payload, ROLE_CONTROL)
    import modal

    from experiments import targeted_training_runtime as runtime

    try:
        return runtime.run_worker(payload)
    finally:
        modal.Volume.from_name(VOLUME_NAME, environment_name="main").commit()


def _modal_treatment_worker(payload: dict[str, object]) -> dict[str, object]:
    _claim_worker_role(payload, ROLE_TREATMENT)
    import modal

    from experiments import targeted_training_runtime as runtime

    try:
        return runtime.run_worker(payload)
    finally:
        modal.Volume.from_name(VOLUME_NAME, environment_name="main").commit()


def _claim_worker_role(payload: dict[str, object], role: str) -> None:
    """Reject duplicate provider entry before any model package or forward pass."""
    import modal

    from experiments import targeted_training_core as core
    from experiments.targeted_training_coordinator import CLAIM_DICT_NAME, claim_once

    checked = core.validate_role_payload(payload)
    if checked["role"] != role:
        raise ValueError("worker role differs from registered Modal function")
    store = modal.Dict.from_name(
        CLAIM_DICT_NAME, environment_name=ENVIRONMENT, create_if_missing=True
    )
    if not claim_once(
        store,
        checked["run_id"],
        role,
        {"payload_sha256": checked["payload_sha256"], "claimed_at_utc": _now_utc()},
    ):
        raise ValueError("duplicate role attempt")


def _build_modal_image(modal: Any, snapshot_root: Path, source_manifest: Mapping[str, str]) -> Any:
    """Build a source-bounded image; caller has already verified manifest hashes."""
    from experiments.mixture_training_contracts import RUNTIME_VERSION_PINS

    if not source_manifest:
        raise ValueError("source image manifest is empty")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .env({"PYTHONPATH": "/root:/root/src", "USE_HUB_KERNELS": "NO"})
        .pip_install(
            *(f"{name}=={version}" for name, version in RUNTIME_VERSION_PINS.items()),
            extra_index_url="https://download.pytorch.org/whl/cu130",
        )
    )
    for relative, digest in sorted(source_manifest.items()):
        local = (snapshot_root / relative).resolve(strict=True)
        if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source snapshot digest differs: {relative}")
        image = image.add_local_file(str(local), remote_path=f"/root/{relative}")
    return image


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _committed_source(root: Path, manifest: Mapping[str, str]) -> str:
    if _git(root, "status", "--porcelain"):
        raise ValueError("planning requires a clean committed source tree")
    commit = _git(root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("source commit is malformed")
    for relative, digest in manifest.items():
        committed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=root, capture_output=True, check=True
        ).stdout
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"source pin differs from commit: {relative}")
    return commit


def _verified_bundle(root: Path, inputs: object) -> str:
    """Bind regenerated host inputs to the already audited, label-separated bundle."""
    from experiments import targeted_training_data as data

    directory = root / "data/processed/targeted-reasoning-v1"
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    if digest != BUNDLE_MANIFEST_SHA256:
        raise ValueError("targeted bundle manifest differs from independent audit pin")
    manifest = json.loads(manifest_bytes)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("input_file_sha256") != getattr(inputs, "file_sha256", None)
    ):
        raise ValueError("targeted bundle input fingerprints differ")
    hashes = manifest.get("bundle_file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {
        "training-records.jsonl",
        "evaluation-records.jsonl",
        "presentations.json",
        "schedules.json",
        "audit.json",
    }:
        raise ValueError("targeted bundle file set differs")
    training = tuple(
        record
        for pool in ("real", "synthetic", "snli", "hans", "winogrande")
        for record in sorted(inputs.training_pools[pool], key=lambda item: item.record_id)
    )
    regenerated = {
        "training-records.jsonl": data._record_jsonl(training),
        "evaluation-records.jsonl": data._record_jsonl(inputs.evaluation_records),
        "presentations.json": data._json_file_bytes(list(inputs.presentations)),
        "schedules.json": data._json_file_bytes(
            {role: list(inputs.schedules[role]) for role in ("control", "treatment")}
        ),
        "audit.json": data._json_file_bytes(inputs.audit),
    }
    for name, content in regenerated.items():
        expected = hashes[name]
        if (
            not isinstance(expected, str)
            or hashlib.sha256(content).hexdigest() != expected
            or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"targeted bundle file differs from rebuilt inputs: {name}")
    return digest


def _load_cpu_tokenizer(root: Path) -> object:
    """Use only the pinned local tokenizer JSON, without model package imports."""
    path = (root / ".cache/qwen-tokenizer.json").resolve(strict=True)
    path.relative_to(root)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_FILE_SHA256["tokenizer.json"]:
        raise ValueError("cached CPU tokenizer differs from pinned tokenizer.json")
    if importlib.metadata.version("tokenizers") != "0.23.2":
        raise ValueError("CPU tokenizers package must be pinned 0.23.2")
    from tokenizers import Tokenizer

    backend = Tokenizer.from_str(raw.decode("utf-8"))

    class PinnedTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return list(backend.encode(text, add_special_tokens=add_special_tokens).ids)

    return PinnedTokenizer()


def _create_modal_workers(
    modal: Any, image: Any, run_id: str = "definition-check"
) -> tuple[Any, Mapping[str, Any], Any]:
    """Create exact single-use functions; no remote work runs until caller invokes them."""
    volume = modal.Volume.from_name(VOLUME_NAME, environment_name="main")
    readonly = volume.with_mount_options(read_only=True)
    app = modal.App(f"reflex-targeted-training-{run_id}", image=image)

    workers: dict[str, Any] = {}
    for role in (ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT):
        resource = ROLE_RESOURCES[role]
        options = {
            "gpu": "A10",
            "cpu": (float(resource.cpu), float(resource.cpu)),
            "memory": (resource.memory_mib, resource.memory_mib),
            "timeout": resource.timeout_seconds,
            "startup_timeout": resource.startup_timeout_seconds,
            "max_containers": 1,
            "min_containers": 0,
            "buffer_containers": 0,
            "scaledown_window": 2,
            "retries": 0,
            "single_use_containers": True,
            "include_source": False,
            "serialized": False,
            "volumes": {VOLUME_ROOT: readonly if role == ROLE_UNCHANGED else volume},
        }
        entry = {
            ROLE_UNCHANGED: _modal_reference_worker,
            ROLE_CONTROL: _modal_control_worker,
            ROLE_TREATMENT: _modal_treatment_worker,
        }[role]
        workers[role] = app.function(**options)(entry)
    from experiments.targeted_training_coordinator import _modal_coordinator_entry

    coordinator = app.function(
        cpu=(1.0, 1.0),
        memory=(4096, 4096),
        timeout=9000,
        startup_timeout=300,
        max_containers=1,
        min_containers=0,
        buffer_containers=0,
        scaledown_window=2,
        retries=0,
        single_use_containers=True,
        include_source=False,
        serialized=False,
        volumes={VOLUME_ROOT: volume},
    )(_modal_coordinator_entry)
    return app, workers, coordinator


def build_receipt(
    plan: Mapping[str, object], role_payloads: Mapping[str, object], roles: Mapping[str, object]
) -> dict[str, object]:
    """Build the analyzer-facing immutable host receipt without exposing metrics."""
    status = (
        "passed"
        if set(roles) == {ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT}
        and all(isinstance(item, Mapping) and "validated_result" in item for item in roles.values())
        else "failed"
    )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "run_id": plan["run_id"],
        "status": status,
        "plan": dict(plan),
        "role_payloads": dict(role_payloads),
        "roles": dict(roles),
    }


def _dispatch_detached(
    app: Any,
    workers: Mapping[str, Any],
    coordinator: Any,
    plan: Mapping[str, object],
    role_payloads: Mapping[str, object],
    destination: Path,
) -> dict[str, object]:
    """Save app identity before one spawn; never retry an ambiguous acknowledgement."""
    app_id: str | None = None
    spawn_attempted = False
    try:
        with app.run(detach=True, environment_name=ENVIRONMENT):
            app_id = getattr(app, "app_id", None)
            if not isinstance(app_id, str) or not app_id:
                raise ValueError("Modal app ID is missing before coordinator spawn")
            _write_json_exclusive(
                destination / "app.json",
                {
                    "app_id": app_id,
                    "run_id": plan["run_id"],
                    "plan_sha256": plan.get("plan_sha256"),
                    "source_commit": plan.get("source_commit"),
                },
            )
            spawn_attempted = True
            call = coordinator.spawn(dict(plan), dict(role_payloads), app_id, dict(workers))
            call_id = _require_call_id(call)
            dispatch = {
                "status": "dispatched",
                "app_id": app_id,
                "run_id": plan["run_id"],
                "plan_sha256": plan.get("plan_sha256"),
                "coordinator_call_id": call_id,
            }
            _write_json_exclusive(destination / "dispatch.json", dispatch)
            return dispatch
        return {
            "status": "unknown" if spawn_attempted else "failed",
            "app_id": app_id,
            "run_id": plan["run_id"],
            "plan_sha256": plan.get("plan_sha256"),
            "failure_type": "SuppressedContextException",
        }
    except BaseException as error:
        status = "unknown" if spawn_attempted else "failed"
        return {
            "status": status,
            "app_id": app_id,
            "run_id": plan["run_id"],
            "plan_sha256": plan.get("plan_sha256"),
            "failure_type": type(error).__name__,
        }


def prepare_plan(
    root: str | Path,
    run_id: str,
    output_dir: str | Path,
    *,
    load_inputs: Any = None,
    tokenizer: Any = None,
) -> dict[str, object]:
    """CPU-only immutable plan preparation; injectable for fixture verification."""
    from experiments import targeted_training_core as core

    project_root = Path(root).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ValueError("plan output directory must be new and exclusive")
    if load_inputs is None:
        from experiments.targeted_training_data import load_inputs as load_inputs
    if tokenizer is None:
        tokenizer = _load_cpu_tokenizer(project_root)
    inputs = load_inputs(project_root)
    bundle_sha256 = _verified_bundle(project_root, inputs)
    manifest = core.targeted_source_manifest(project_root)
    source_commit = _committed_source(project_root, manifest)
    payloads = {
        role: core.build_role_payload(inputs, role, tokenizer, run_id=run_id, source_pins=manifest)
        for role in (ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT)
    }
    for payload in payloads.values():
        core.validate_role_payload(payload)
    plan = build_launch_plan(
        run_id,
        manifest,
        payloads,
        source_commit=source_commit,
        bundle_manifest_sha256=bundle_sha256,
    )
    destination.mkdir(parents=True)
    snapshot = destination / "source-snapshot"
    for relative in manifest:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / relative, target)
    for role, payload in payloads.items():
        (destination / f"{role}-payload.json").write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (destination / "plan.json").write_text(
        json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    return {"plan": plan, "role_payloads": payloads, "path": str(destination)}


def launch_plan(
    plan_path: str | Path,
    expected_plan_sha256: str,
    clearance_path: str | Path,
    output_dir: str | Path,
    *,
    root: str | Path = ".",
) -> dict[str, object]:
    """Launch a reviewed immutable plan only after concrete local clearances."""
    from experiments import targeted_training_core as core

    plan_file = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("reviewed plan digest does not match")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if _canonical_digest(body) != expected_plan_sha256:
        raise ValueError("plan is not canonical")
    clearance = json.loads(Path(clearance_path).resolve(strict=True).read_text(encoding="utf-8"))
    _validate_clearance(plan, clearance)
    snapshot = plan_file.parent / "source-snapshot"
    manifest = plan.get("source_pins")
    if not isinstance(manifest, dict):
        raise ValueError("plan source manifest is malformed")
    project_root = Path(root).expanduser().resolve(strict=True)
    if core.targeted_source_manifest(project_root) != manifest:
        raise ValueError("live source manifest differs from reviewed plan")
    for relative, digest in manifest.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("source snapshot path is malformed")
        path = (snapshot / relative).resolve()
        path.relative_to(snapshot.resolve())
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("source snapshot differs from reviewed plan")
    bundle_path = project_root / "data/processed/targeted-reasoning-v1/manifest.json"
    bundle_bytes = bundle_path.read_bytes()
    if hashlib.sha256(bundle_bytes).hexdigest() != plan["bundle_manifest_sha256"]:
        raise ValueError("targeted bundle manifest differs from reviewed plan")
    bundle = json.loads(bundle_bytes)
    for filename, digest in bundle["bundle_file_sha256"].items():
        if hashlib.sha256((bundle_path.parent / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("targeted bundle file differs from reviewed manifest")
    if _committed_source(project_root, manifest) != plan["source_commit"]:
        raise ValueError("launch source commit differs from reviewed plan")
    payloads = {
        role: json.loads((plan_file.parent / f"{role}-payload.json").read_text(encoding="utf-8"))
        for role in (ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT)
    }
    if set(plan.get("role_payload_sha256", {})) != set(payloads):
        raise ValueError("reviewed role payload digest set is incomplete")
    for role, payload in payloads.items():
        if (
            core.validate_role_payload(payload)["payload_sha256"]
            != plan["role_payload_sha256"][role]
        ):
            raise ValueError(f"{role} payload differs from reviewed plan")
        if payload["run_id"] != plan["run_id"] or payload["source_pins"] != manifest:
            raise ValueError(f"{role} payload source/run binding differs")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise ValueError("launch output directory must be new and exclusive")
    run_marker = project_root / "artifacts" / f"{plan['run_id']}.targeted-attempt.json"
    if run_marker.exists():
        raise ValueError("run ID already has an attempt marker")
    _modal_preflight(plan["run_id"])
    destination.mkdir(parents=True, exist_ok=False)
    marker = {
        "run_id": plan["run_id"],
        "plan_sha256": expected_plan_sha256,
        "source_commit": plan["source_commit"],
        "created_at_utc": _now_utc(),
    }
    _write_json_exclusive(run_marker, marker)
    _write_json_exclusive(destination / "attempt.json", marker)

    try:
        import modal

        image = _build_modal_image(modal, snapshot, manifest)
        app, workers, coordinator = _create_modal_workers(modal, image, plan["run_id"])
        dispatch = _dispatch_detached(app, workers, coordinator, plan, payloads, destination)
    except BaseException as error:
        dispatch = {
            "status": "failed",
            "run_id": plan["run_id"],
            "plan_sha256": plan["plan_sha256"],
            "failure_type": type(error).__name__,
        }
    if dispatch["status"] != "dispatched":
        _write_json_exclusive(destination / "dispatch-failure.json", dispatch)
    return dispatch


def recover_plan(
    plan_path: str | Path,
    expected_plan_sha256: str,
    output_dir: str | Path,
    *,
    root: str | Path = ".",
    read_terminal: Any = None,
    read_identity: Any = None,
    provider: Any = None,
    call_status: Any = None,
) -> dict[str, object]:
    """Observe one prior attempt; never define, spawn, or replay a remote function."""
    from experiments.targeted_training_coordinator import (
        finalize_remote_receipt,
        validate_remote_receipt_identity,
    )

    plan_file = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if (
        not isinstance(plan, dict)
        or plan.get("plan_sha256") != expected_plan_sha256
        or _canonical_digest({key: value for key, value in plan.items() if key != "plan_sha256"})
        != expected_plan_sha256
    ):
        raise ValueError("recovery plan identity differs")
    destination = Path(output_dir).resolve(strict=True)
    attempt = json.loads((destination / "attempt.json").read_text(encoding="utf-8"))
    app_marker = json.loads((destination / "app.json").read_text(encoding="utf-8"))
    run_marker_path = (
        Path(root).resolve(strict=True) / "artifacts" / f"{plan['run_id']}.targeted-attempt.json"
    )
    run_marker = json.loads(run_marker_path.read_text(encoding="utf-8"))
    expected = {
        "run_id": plan["run_id"],
        "plan_sha256": expected_plan_sha256,
        "source_commit": plan["source_commit"],
    }
    if (
        not isinstance(attempt, dict)
        or not isinstance(app_marker, dict)
        or not isinstance(run_marker, dict)
        or attempt != run_marker
        or any(
            attempt.get(key) != value or app_marker.get(key) != value
            for key, value in expected.items()
        )
        or not isinstance(app_marker.get("app_id"), str)
        or re.fullmatch(r"ap-[A-Za-z0-9]+", app_marker["app_id"]) is None
    ):
        raise ValueError("recovery attempt or app identity differs")
    dispatch_file = destination / "dispatch.json"
    dispatch = (
        json.loads(dispatch_file.read_text(encoding="utf-8")) if dispatch_file.exists() else None
    )
    if dispatch is not None and (
        not isinstance(dispatch, dict)
        or dispatch.get("status") != "dispatched"
        or dispatch.get("app_id") != app_marker["app_id"]
        or dispatch.get("run_id") != plan["run_id"]
        or dispatch.get("plan_sha256") != expected_plan_sha256
        or not isinstance(dispatch.get("coordinator_call_id"), str)
    ):
        raise ValueError("recovery dispatch identity differs")
    marker = {**app_marker}
    if dispatch is not None:
        marker["coordinator_call_id"] = dispatch["coordinator_call_id"]
    if (
        read_terminal is None
        or provider is None
        or call_status is None
        or (dispatch is None and read_identity is None)
    ):
        _modal_preflight(plan["run_id"], require_unused=False)
        read_terminal = read_terminal or _read_remote_terminal
        read_identity = read_identity or _read_remote_identity
        provider = provider or _ModalProvider()
        call_status = call_status or _coordinator_call_status
    if dispatch is None:
        remote_identity = read_identity(plan["run_id"], app_marker["app_id"])
        if remote_identity is not None:
            expected_identity = {
                **expected,
                "app_id": app_marker["app_id"],
                "coordinator_call_id": remote_identity.get("coordinator_call_id")
                if isinstance(remote_identity, dict)
                else None,
            }
            if (
                not isinstance(remote_identity, dict)
                or remote_identity != expected_identity
                or not isinstance(remote_identity["coordinator_call_id"], str)
                or re.fullmatch(r"fc-[A-Za-z0-9]+", remote_identity["coordinator_call_id"]) is None
            ):
                raise ValueError("recovery remote coordinator identity differs")
            marker["coordinator_call_id"] = remote_identity["coordinator_call_id"]
    terminal = read_terminal(plan["run_id"], app_marker["app_id"])
    if terminal is None:
        observation = provider.observe(app_marker["app_id"])
        call_state = (
            call_status(marker["coordinator_call_id"])
            if "coordinator_call_id" in marker
            else "unknown"
        )
        stopped = (
            observation.get("state") == "APP_STATE_STOPPED" and observation.get("n_tasks") == 0
        )
        created = datetime.fromisoformat(attempt["created_at_utc"].replace("Z", "+00:00"))
        expired = created.tzinfo is None or datetime.now(UTC) - created > timedelta(seconds=9300)
        if stopped or call_state in ("failed", "complete") or expired:
            terminal = read_terminal(plan["run_id"], app_marker["app_id"])
            if terminal is None:
                teardown = provider.stop(app_marker["app_id"]) if not stopped else observation
                outcome = {
                    "status": "incomplete",
                    "app_id": app_marker["app_id"],
                    "teardown": teardown,
                }
                _write_json_exclusive(destination / "incomplete.json", outcome)
                return outcome
        else:
            return {"status": "pending", "app_id": app_marker["app_id"], "call_state": call_state}
    if not isinstance(terminal, dict):
        raise ValueError("remote terminal receipt is malformed")
    payloads = {
        role: json.loads((plan_file.parent / f"{role}-payload.json").read_text(encoding="utf-8"))
        for role in (ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT)
    }
    validate_remote_receipt_identity(plan, payloads, marker, terminal)
    teardown = provider.verify_teardown(app_marker["app_id"])
    receipt = finalize_remote_receipt(plan, payloads, marker, terminal, teardown)
    _write_json_exclusive(destination / "receipt.json", receipt)
    return {
        "status": receipt["status"],
        "app_id": app_marker["app_id"],
        "receipt": str(destination / "receipt.json"),
    }


def _read_remote_terminal(run_id: str, app_id: str) -> dict[str, object] | None:
    import modal

    from experiments.targeted_training_coordinator import EVIDENCE_DIRECTORY

    volume = modal.Volume.from_name(VOLUME_NAME, environment_name=ENVIRONMENT)
    path = f"/{EVIDENCE_DIRECTORY}/{run_id}/{app_id}/terminal.json"
    try:
        raw = b"".join(volume.read_file(path))
    except FileNotFoundError:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("remote terminal receipt is malformed")
    return value


def _read_remote_identity(run_id: str, app_id: str) -> dict[str, object] | None:
    import modal

    from experiments.targeted_training_coordinator import EVIDENCE_DIRECTORY

    volume = modal.Volume.from_name(VOLUME_NAME, environment_name=ENVIRONMENT)
    path = f"/{EVIDENCE_DIRECTORY}/{run_id}/{app_id}/identity.json"
    try:
        raw = b"".join(volume.read_file(path))
    except FileNotFoundError:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("remote coordinator identity is malformed")
    return value


def _coordinator_call_status(call_id: str) -> str:
    import modal

    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
    except TimeoutError as error:
        return "failed" if hasattr(error, "__line_cache__") else "pending"
    except (
        modal.exception.FunctionTimeoutError,
        modal.exception.InternalFailure,
        modal.exception.RemoteError,
        modal.exception.ExecutionError,
        modal.exception.OutputExpiredError,
    ):
        return "failed"
    except modal.exception.Error as error:
        return "failed" if hasattr(error, "__line_cache__") else "unknown"
    except Exception as error:
        return "failed" if hasattr(error, "__line_cache__") else "unknown"
    return "complete"


def main(argv: list[str] | None = None) -> int:
    """Prepare a CPU-only plan or execute an explicitly cleared reviewed plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    mode = parser.add_subparsers(dest="mode", required=True)
    planning = mode.add_parser("plan")
    planning.add_argument("--run-id", required=True)
    planning.add_argument("--output-dir", required=True)
    launch = mode.add_parser("launch")
    launch.add_argument("--plan", required=True)
    launch.add_argument("--expected-plan-sha256", required=True)
    launch.add_argument("--clearance", required=True)
    launch.add_argument("--output-dir", required=True)
    recovery = mode.add_parser("recover")
    recovery.add_argument("--plan", required=True)
    recovery.add_argument("--expected-plan-sha256", required=True)
    recovery.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.mode == "plan":
        prepared = prepare_plan(args.root, args.run_id, args.output_dir)
        print(
            _json_line(
                {
                    "plan": str(Path(prepared["path"]) / "plan.json"),
                    "plan_sha256": prepared["plan"]["plan_sha256"],
                }
            ).strip()
        )
        return 0
    if args.mode == "recover":
        result = recover_plan(args.plan, args.expected_plan_sha256, args.output_dir, root=args.root)
        print(_json_line(result).strip())
        return 0 if result["status"] in ("pending", "passed") else 1
    dispatch = launch_plan(
        args.plan, args.expected_plan_sha256, args.clearance, args.output_dir, root=args.root
    )
    print(
        _json_line(
            {
                "status": dispatch["status"],
                "app_id": dispatch.get("app_id"),
                "dispatch": str(Path(args.output_dir).resolve() / "dispatch.json"),
            }
        ).strip()
    )
    return 0 if dispatch["status"] == "dispatched" else 1


def _canonical_digest(value: object) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("launch plan is not JSON-safe") from error
    return hashlib.sha256(encoded).hexdigest()


def _json_line(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json_line(value))
        stream.flush()
        os.fsync(stream.fileno())


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _fresh_utc(value: object, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label} clearance time is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} clearance time is malformed") from error
    if parsed.tzinfo is None or not -timedelta(minutes=5) <= datetime.now(
        UTC
    ) - parsed <= timedelta(minutes=30):
        raise ValueError(f"{label} clearance is stale or future-dated")


def _validate_clearance(plan: Mapping[str, object], value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "plan_sha256",
        "source_commit",
        "source_pins_sha256",
        "ci",
        "account",
        "budget",
    }:
        raise ValueError("launch clearance has an unexpected schema")
    if (
        value["plan_sha256"] != plan["plan_sha256"]
        or value["source_commit"] != plan["source_commit"]
        or value["source_pins_sha256"] != plan["source_pins_sha256"]
    ):
        raise ValueError("launch clearance is not bound to reviewed source and plan")
    ci, account, budget = value["ci"], value["account"], value["budget"]
    if (
        not isinstance(ci, dict)
        or not {"head_sha", "status", "conclusion", "workflow_url", "evidence_sha256"} <= set(ci)
        or (
            ci["head_sha"] != plan["source_commit"]
            or ci["status"] != "completed"
            or ci["conclusion"] != "success"
        )
    ):
        raise ValueError("exact-source CI clearance is missing")
    if (
        not isinstance(account, dict)
        or not {"profile", "workspace", "environment", "checked_at_utc", "evidence_sha256"}
        <= set(account)
        or any(account[key] != plan[key] for key in ("profile", "workspace", "environment"))
    ):
        raise ValueError("personal Modal account clearance differs")
    if (
        not isinstance(budget, dict)
        or not {
            "status",
            "checked_at_utc",
            "estimated_max_usd",
            "approved_max_usd",
            "evidence_sha256",
        }
        <= set(budget)
        or budget["status"] != "cleared"
    ):
        raise ValueError("fresh budget clearance is missing")
    for label, item in (("ci", ci), ("account", account), ("budget", budget)):
        evidence = item["evidence_sha256"]
        if not isinstance(evidence, str) or re.fullmatch(r"[0-9a-f]{64}", evidence) is None:
            raise ValueError(f"{label} evidence digest is malformed")
    if not isinstance(ci["workflow_url"], str) or not ci["workflow_url"].startswith("https://"):
        raise ValueError("CI evidence URL is malformed")
    _fresh_utc(account["checked_at_utc"], "account")
    _fresh_utc(budget["checked_at_utc"], "budget")
    try:
        estimate = Decimal(str(budget["estimated_max_usd"]))
        approved = Decimal(str(budget["approved_max_usd"]))
    except InvalidOperation as error:
        raise ValueError("budget amounts are malformed") from error
    if not estimate.is_finite() or not approved.is_finite() or estimate <= 0 or approved < estimate:
        raise ValueError("budget estimate exceeds its explicit clearance")


def _modal_preflight(run_id: str, *, require_unused: bool = True) -> None:
    """Verify the fixed profile and inspect active apps before defining a worker."""
    from experiments import modal_smoke

    if any(os.environ.get(key) for key in modal_smoke.CREDENTIAL_OVERRIDES):
        raise ValueError("Modal credential overrides are not accepted")
    if importlib.metadata.version("modal") != MODAL_SDK_VERSION:
        raise ValueError("Modal SDK version differs from reviewed plan")
    if not modal_smoke._verify_profile(PROFILE, WORKSPACE):
        raise ValueError("Modal profile is not the fixed personal workspace")

    async def check_apps() -> None:
        import modal
        from modal_proto import api_pb2

        modal.config._set_profile(PROFILE)
        client = await modal.client._Client.from_env()
        workspace = modal.workspace._Workspace.from_context(client=client)
        await workspace.hydrate()
        if workspace.name != WORKSPACE:
            raise ValueError("Modal credentials do not identify the personal workspace")
        if require_unused:
            response = await client._stub.AppList(
                api_pb2.AppListRequest(environment_name=ENVIRONMENT)
            )
            for app in response.apps:
                if f"reflex-targeted-training-{run_id}" in (
                    getattr(app, "name", None),
                    getattr(app, "description", None),
                ):
                    raise ValueError("run ID already has a Modal app")

    asyncio.run(check_apps())


class _ModalProvider:
    def observe(self, app_id: str) -> dict[str, object]:
        return asyncio.run(self._observe(app_id))

    async def _observe(self, app_id: str) -> dict[str, object]:
        import modal
        from modal_proto import api_pb2

        if not isinstance(app_id, str) or re.fullmatch(r"ap-[A-Za-z0-9]+", app_id) is None:
            raise ValueError("Modal app ID is malformed")
        modal.config._set_profile(PROFILE)
        client = await modal.client._Client.from_env()
        lifecycle = await asyncio.wait_for(
            client._stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id)), timeout=5
        )
        tasks = await asyncio.wait_for(
            client._stub.TaskList(
                api_pb2.TaskListRequest(environment_name=ENVIRONMENT, app_id=app_id)
            ),
            timeout=5,
        )
        if any(getattr(task, "app_id", None) != app_id for task in tasks.tasks):
            raise ValueError("TaskList returned a different app ID")
        return {
            "state": api_pb2.AppState.Name(lifecycle.lifecycle.app_state),
            "n_tasks": len(tasks.tasks),
        }

    def stop(self, app_id: str) -> dict[str, object]:
        return asyncio.run(self._teardown(app_id, force_stop=True))

    def verify_teardown(self, app_id: str) -> dict[str, object]:
        return asyncio.run(self._teardown(app_id, force_stop=False))

    async def _teardown(self, app_id: str, *, force_stop: bool) -> dict[str, object]:
        import modal
        from modal_proto import api_pb2

        if not isinstance(app_id, str) or not app_id:
            raise ValueError("Modal app ID is missing for teardown")
        modal.config._set_profile(PROFILE)
        client = await modal.client._Client.from_env()
        observations: list[dict[str, object]] = []
        errors: list[dict[str, str]] = []
        stop_requested = False

        async def stop() -> None:
            nonlocal stop_requested
            stop_requested = True
            try:
                await asyncio.wait_for(
                    client._stub.AppStop(
                        api_pb2.AppStopRequest(app_id=app_id, source=api_pb2.APP_STOP_SOURCE_CLI)
                    ),
                    timeout=5,
                )
            except BaseException as error:
                errors.append({"stage": "AppStop", "type": type(error).__name__})

        if force_stop:
            await stop()
        for attempt in range(1, 7):
            observation: dict[str, object] = {"attempt": attempt}
            try:
                lifecycle = await asyncio.wait_for(
                    client._stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id)),
                    timeout=5,
                )
                observation["state"] = api_pb2.AppState.Name(lifecycle.lifecycle.app_state)
            except BaseException as error:
                errors.append({"stage": "AppGetLifecycle", "type": type(error).__name__})
            try:
                task_list = await asyncio.wait_for(
                    client._stub.TaskList(
                        api_pb2.TaskListRequest(environment_name=ENVIRONMENT, app_id=app_id)
                    ),
                    timeout=5,
                )
                if any(getattr(task, "app_id", None) != app_id for task in task_list.tasks):
                    raise ValueError("TaskList returned a different app ID")
                observation["n_tasks"] = len(task_list.tasks)
            except BaseException as error:
                errors.append({"stage": "TaskList", "type": type(error).__name__})
            observations.append(observation)
            if observation.get("state") == "APP_STATE_STOPPED" and observation.get("n_tasks") == 0:
                return {
                    "app_id": app_id,
                    "verified": True,
                    "stop_requested": stop_requested,
                    "observations": observations,
                    "errors": errors,
                }
            if attempt == 1 and not stop_requested:
                await stop()
            if attempt < 6:
                await asyncio.sleep(2)
        return {
            "app_id": app_id,
            "verified": False,
            "stop_requested": stop_requested,
            "observations": observations,
            "errors": errors,
        }


def build_launch_plan(
    run_id: object,
    source_pins: object,
    role_payloads: Mapping[str, object] | None = None,
    *,
    source_commit: str | None = None,
    bundle_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Create an immutable, role-minimal execution plan without launching Modal."""

    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", run_id) is None
    ):
        raise ValueError("run ID is malformed")
    if not isinstance(source_pins, Mapping) or not source_pins:
        raise ValueError("exact source pins are required before planning")
    if source_commit is not None and re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    if bundle_manifest_sha256 is not None and bundle_manifest_sha256 != BUNDLE_MANIFEST_SHA256:
        raise ValueError("bundle manifest differs from independent audit pin")
    roles = []
    for role in (ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT):
        resources = ROLE_RESOURCES[role]
        roles.append(
            {
                "role": role,
                "reference_first": role == ROLE_UNCHANGED,
                "resources": {
                    "timeout_seconds": resources.timeout_seconds,
                    "startup_timeout_seconds": resources.startup_timeout_seconds,
                    "cpu": resources.cpu,
                    "memory_mib": resources.memory_mib,
                    "max_containers": resources.max_containers,
                    "max_inputs": resources.max_inputs,
                    "scaledown_window_seconds": resources.scaledown_window_seconds,
                    "retries": resources.retries,
                },
            }
        )
    digests = {}
    if role_payloads is not None:
        if set(role_payloads) != {ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT}:
            raise ValueError("plan requires all role payloads")
        for role, payload in role_payloads.items():
            if not isinstance(payload, Mapping) or not isinstance(
                payload.get("payload_sha256"), str
            ):
                raise ValueError("role payload digest is malformed")
            digests[role] = payload["payload_sha256"]
    body = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "source_commit": source_commit,
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "source_pins": dict(source_pins),
        "source_pins_sha256": _canonical_digest(source_pins),
        "profile": PROFILE,
        "workspace": WORKSPACE,
        "environment": ENVIRONMENT,
        "modal_sdk_version": MODAL_SDK_VERSION,
        "execution": {"detach": True, "coordinator": dict(COORDINATOR_EXECUTION)},
        "roles": roles,
        "role_payload_sha256": digests,
    }
    return body | {"plan_sha256": _canonical_digest(body)}


def retain_raw_result(
    raw_result: object, validation_error: BaseException | None = None
) -> dict[str, object]:
    """Store JSON-safe raw worker evidence even when lifecycle validation fails."""

    try:
        normalized = json.loads(json.dumps(raw_result, allow_nan=False))
    except (TypeError, ValueError) as error:
        normalized = {"unserializable": type(raw_result).__name__}
        validation_error = validation_error or error
    receipt: dict[str, object] = {"raw_worker_result": normalized}
    if validation_error is not None:
        receipt["validation_failure"] = {
            "type": type(validation_error).__name__,
            "message": str(validation_error),
        }
    return receipt


def _cancel_remote_bounded(calls: Mapping[str, object]) -> dict[str, str]:
    """Bound concurrent Modal cancellation, including a coroutine that ignores cancellation."""
    errors: dict[str, str] = {}
    completed: set[str] = set()
    lock = threading.Lock()

    async def cancel_one(role: str, call: object) -> None:
        try:
            await asyncio.wait_for(
                call.cancel.aio(terminate_containers=True),
                timeout=_REMOTE_CANCEL_TIMEOUT_SECONDS,
            )
        except BaseException as error:
            with lock:
                errors[role] = type(error).__name__
        finally:
            with lock:
                completed.add(role)

    def run_batch() -> None:
        try:

            async def batch() -> None:
                await asyncio.gather(*(cancel_one(role, call) for role, call in calls.items()))

            asyncio.run(batch())
        except BaseException as error:
            with lock:
                errors["cancel_batch"] = type(error).__name__

    thread = threading.Thread(target=run_batch, name="targeted-remote-cancel", daemon=True)
    thread.start()
    thread.join(timeout=_REMOTE_CANCEL_TIMEOUT_SECONDS + _REMOTE_CANCEL_BATCH_GRACE_SECONDS)
    with lock:
        result = dict(errors)
        for role in calls:
            if role not in completed:
                result[role] = "TimeoutError"
    return result


def _execute_modal_roles(
    app: Any,
    workers: Mapping[str, Any],
    role_payloads: Mapping[str, object],
    persist: Any,
    provider: Any,
    validate_result: Any,
) -> dict[str, object]:
    """Reference-first Modal lifecycle with durable raw evidence and no retries.

    ``persist`` is called after every irreversible provider transition, making
    this directly testable with fake calls and usable by the CLI receipt writer.
    """
    if set(workers) != {ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT}:
        raise ValueError("Modal workers are incomplete")
    if set(role_payloads) != set(workers) or not callable(persist) or not callable(validate_result):
        raise ValueError("Modal lifecycle inputs are malformed")
    roles: dict[str, object] = {}
    calls: dict[str, Any] = {}
    futures: dict[object, str] = {}
    app_id = getattr(app, "app_id", None)
    if not isinstance(app_id, str) or not app_id:
        raise ValueError("Modal app ID is missing")
    persist("app", {"app_id": app_id})

    def cancel_all(reason: BaseException) -> None:
        try:
            errors = _cancel_remote_bounded(calls)
        except BaseException as error:
            errors = {"cancel_dispatch": type(error).__name__}
        try:
            teardown = provider.stop(app_id)
            persist("emergency_teardown", teardown)
        except BaseException as error:
            errors["app_stop"] = type(error).__name__
        persist("cancellation", {"reason": type(reason).__name__, "errors": errors})

    def receive(role: str, raw: object) -> None:
        roles[role] = retain_raw_result(raw)
        persist("roles", roles)
        roles[role]["validated_result"] = validate_result(role_payloads[role], raw)
        persist("roles", roles)
        if roles[role]["validated_result"].get("status") != "passed":
            raise ValueError(f"{role} result did not pass")

    try:
        reference = workers[ROLE_UNCHANGED].spawn(role_payloads[ROLE_UNCHANGED])
        calls[ROLE_UNCHANGED] = reference
        _require_call_id(reference)
        persist("calls", {ROLE_UNCHANGED: reference.object_id})
        raw = reference.get(timeout=ROLE_RESOURCES[ROLE_UNCHANGED].timeout_seconds + 360)
        receive(ROLE_UNCHANGED, raw)
        for role in (ROLE_CONTROL, ROLE_TREATMENT):
            calls[role] = workers[role].spawn(role_payloads[role])
            _require_call_id(calls[role])
            persist("calls", {name: call.object_id for name, call in calls.items()})
        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="targeted-training")
        try:
            futures = {
                pool.submit(
                    calls[role].get, timeout=ROLE_RESOURCES[role].timeout_seconds + 360
                ): role
                for role in (ROLE_CONTROL, ROLE_TREATMENT)
            }
            pending = set(futures)
            deadline = (
                time.monotonic()
                + max(
                    ROLE_RESOURCES[ROLE_CONTROL].timeout_seconds,
                    ROLE_RESOURCES[ROLE_TREATMENT].timeout_seconds,
                )
                + 360
            )
            while pending:
                completed, pending = wait(
                    pending,
                    timeout=max(0, deadline - time.monotonic()),
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    raise TimeoutError("arm result wait exceeded its fixed deadline")
                for future in completed:
                    role = futures[future]
                    receive(role, future.result())
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return {
            "status": "passed",
            "roles": roles,
            "calls": {role: call.object_id for role, call in calls.items()},
            "app_id": app_id,
        }
    except BaseException as error:
        for future, role in futures.items():
            if future.done() and role not in roles:
                try:
                    roles[role] = retain_raw_result(future.result())
                except BaseException:
                    pass
        cancel_all(error)
        for role in calls:
            roles.setdefault(role, {"execution": {"unresolved": True}})
        persist("roles", roles)
        return {
            "status": "failed",
            "roles": roles,
            "app_id": app_id,
            "failure": {"type": type(error).__name__},
        }


def _require_call_id(call: object) -> str:
    identifier = getattr(call, "object_id", None)
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Modal FunctionCall ID is missing")
    return identifier


if __name__ == "__main__":
    raise SystemExit(main())
