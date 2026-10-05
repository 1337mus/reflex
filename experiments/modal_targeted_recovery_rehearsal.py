"""Tiny CPU-only Modal rehearsal for detached targeted-run recovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

RUN_ID = "disconnect-rehearsal-2026-10-05-r1"
ROLES = ("unchanged", "control", "treatment")
REFERENCE_DELAY_SECONDS = 10
PINNED_PYDANTIC_VERSION = "2.13.5"


def _cpu_limits(timeout: int) -> dict[str, object]:
    return {
        "cpu": 1,
        "memory_mib": 1024,
        "gpu": None,
        "retries": 0,
        "single_use_containers": True,
        "max_containers": 1,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window_seconds": 2,
        "timeout_seconds": timeout,
        "startup_timeout_seconds": 60,
    }


STAGE_LIMITS = _cpu_limits(120)
COORDINATOR_LIMITS = _cpu_limits(180)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_spec(source_commit: str, source_manifest: Mapping[str, str]) -> dict[str, object]:
    """Build the fixed, data-free rehearsal spec from exact source pins."""
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("rehearsal source commit must be a full lowercase Git SHA")
    if not source_manifest or any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for path, digest in source_manifest.items()
    ):
        raise ValueError("rehearsal source manifest is malformed")
    from experiments import modal_targeted_training as launch
    from experiments.targeted_training_contracts import (
        ENVIRONMENT,
        MODAL_SDK_VERSION,
        PROFILE,
        WORKSPACE,
    )
    from experiments.targeted_training_core import EXPERIMENT_ID

    body: dict[str, object] = {
        "schema_version": 1,
        "rehearsal_id": "targeted-recovery-disconnect-v1",
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
        "source_commit": source_commit,
        "source_manifest": dict(sorted(source_manifest.items())),
        "source_manifest_sha256": launch._canonical_digest(dict(sorted(source_manifest.items()))),
        "profile": PROFILE,
        "workspace": WORKSPACE,
        "environment": ENVIRONMENT,
        "modal_sdk_version": MODAL_SDK_VERSION,
        "pydantic_version": PINNED_PYDANTIC_VERSION,
        "roles": list(ROLES),
        "reference_delay_seconds": REFERENCE_DELAY_SECONDS,
        "stage": dict(STAGE_LIMITS),
        "coordinator": dict(COORDINATOR_LIMITS),
    }
    return body | {"plan_sha256": launch._canonical_digest(body)}


def validate_clearance(spec: Mapping[str, object], ci_proof: object, clearance: object) -> None:
    """Require exact-commit CI and fresh account/budget evidence reviewed by root."""
    from experiments import modal_targeted_training as launch

    if (
        not isinstance(ci_proof, Mapping)
        or set(ci_proof) != {"head_sha", "status", "conclusion", "workflow_url", "evidence_sha256"}
        or not isinstance(clearance, Mapping)
        or set(clearance)
        != {
            "reviewed_by",
            "reviewed_at_utc",
            "plan_sha256",
            "source_commit",
            "source_manifest_sha256",
            "account",
            "budget",
        }
        or clearance.get("reviewed_by") != "root"
        or clearance.get("plan_sha256") != spec.get("plan_sha256")
        or clearance.get("source_commit") != spec.get("source_commit")
        or clearance.get("source_manifest_sha256") != spec.get("source_manifest_sha256")
    ):
        raise ValueError("root clearance does not match the exact rehearsal spec")
    launch._fresh_utc(clearance.get("reviewed_at_utc"), "root review")
    pinned = dict(spec)
    pinned["source_pins_sha256"] = spec["source_manifest_sha256"]
    launch._validate_clearance(
        pinned,
        {
            "plan_sha256": spec["plan_sha256"],
            "source_commit": spec["source_commit"],
            "source_pins_sha256": spec["source_manifest_sha256"],
            "ci": dict(ci_proof),
            "account": clearance["account"],
            "budget": clearance["budget"],
        },
    )


def begin_attempt(output_dir: str | Path, spec: Mapping[str, object]) -> Path:
    """Write the single-use local latch before any Modal app operation."""
    from experiments import modal_targeted_training as launch

    directory = Path(output_dir).resolve(strict=True)
    marker = directory / "attempt.json"
    try:
        launch._write_json_exclusive(
            marker,
            {
                "run_id": spec["run_id"],
                "plan_sha256": spec["plan_sha256"],
                "source_commit": spec["source_commit"],
                "created_at_utc": launch._now_utc(),
            },
        )
    except FileExistsError as error:
        raise ValueError("rehearsal attempt already exists; never resubmit") from error
    return marker


def build_payloads(spec: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Create three fixed synthetic markers bound to the reviewed spec."""
    return {
        role: {
            "run_id": spec["run_id"],
            "role": role,
            "marker": {
                "unchanged": "synthetic-reference",
                "control": "synthetic-control",
                "treatment": "synthetic-treatment",
            }[role],
            "plan_sha256": spec["plan_sha256"],
            "source_manifest_sha256": spec["source_manifest_sha256"],
        }
        for role in ROLES
    }


def verify_spec(spec_dir: str | Path, source_root: str | Path = ".") -> dict[str, object]:
    """Check spec, live committed source, and lock pin before dispatch."""
    from experiments import modal_targeted_training as launch
    from experiments import targeted_training_core as core

    directory = Path(spec_dir).expanduser().resolve(strict=True)
    root = Path(source_root).expanduser().resolve(strict=True)
    spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec != build_spec(
        spec.get("source_commit"), spec.get("source_manifest")
    ):
        raise ValueError("rehearsal spec digest or fixed contract differs")
    manifest = spec["source_manifest"]
    current = core.targeted_source_manifest(root)
    current["experiments/modal_targeted_recovery_rehearsal.py"] = _sha256(
        (root / "experiments/modal_targeted_recovery_rehearsal.py").read_bytes()
    )
    if current != manifest or launch._committed_source(root, manifest) != spec["source_commit"]:
        raise ValueError("live committed source differs from rehearsal spec")
    locked = next(
        package["version"]
        for package in tomllib.loads((root / "uv.lock").read_text())["package"]
        if package["name"] == "pydantic"
    )
    if locked != spec["pydantic_version"]:
        raise ValueError("rehearsal pydantic pin differs from uv.lock")
    return spec


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("rehearsal timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("rehearsal timestamp is malformed") from error
    if parsed.tzinfo is None:
        raise ValueError("rehearsal timestamp must include a timezone")
    return parsed.astimezone(UTC)


def validate_stage_result(payload: Mapping[str, object], raw: object) -> dict[str, object]:
    """Accept only the exact synthetic stage identity and timing fields."""
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {*payload, "started_at_utc", "finished_at_utc"}
        or any(raw.get(key) != value for key, value in payload.items())
    ):
        raise ValueError("stage output is not the expected synthetic marker")
    started = _timestamp(raw.get("started_at_utc"))
    finished = _timestamp(raw.get("finished_at_utc"))
    if finished <= started:
        raise ValueError("synthetic stage timestamps are out of order")
    return {
        "status": "passed",
        "role": payload["role"],
        "started_at_utc": raw["started_at_utc"],
        "finished_at_utc": raw["finished_at_utc"],
    }


def validate_terminal(
    spec: Mapping[str, object],
    payloads: Mapping[str, object],
    dispatch: Mapping[str, object],
    observed_exit: Mapping[str, object],
    terminal: Mapping[str, object],
) -> dict[str, object]:
    """Validate the data-free raw receipt and prove the disconnect timing."""
    if (
        dict(spec) != build_spec(spec.get("source_commit"), spec.get("source_manifest"))
        or dict(payloads) != build_payloads(spec)
        or terminal.get("status") != "pending_shutdown"
        or terminal.get("experiment_id") != spec.get("experiment_id")
        or terminal.get("run_id") != spec.get("run_id")
        or terminal.get("plan") != dict(spec)
        or terminal.get("role_payloads") != dict(payloads)
        or dispatch.get("run_id") != spec.get("run_id")
        or dispatch.get("plan_sha256") != spec.get("plan_sha256")
        or dispatch.get("source_manifest_sha256") != spec.get("source_manifest_sha256")
        or dispatch.get("coordinator_call_id") is None
        or any(
            observed_exit.get(key) != dispatch.get(key)
            for key in (
                "run_id",
                "app_id",
                "coordinator_call_id",
                "plan_sha256",
                "source_manifest_sha256",
            )
        )
    ):
        raise ValueError("rehearsal app, run, plan, or source identity differs")
    app_id = dispatch.get("app_id")
    lifecycle = terminal.get("lifecycle")
    roles = terminal.get("roles")
    calls = lifecycle.get("calls") if isinstance(lifecycle, Mapping) else None
    if (
        not isinstance(app_id, str)
        or re.fullmatch(r"ap-[A-Za-z0-9]+", app_id) is None
        or not isinstance(lifecycle, Mapping)
        or lifecycle.get("app_id") != app_id
        or lifecycle.get("coordinator_call_id") != observed_exit.get("coordinator_call_id")
        or lifecycle.get("status") != "passed"
        or not isinstance(roles, Mapping)
        or set(roles) != set(ROLES)
        or not isinstance(calls, Mapping)
        or set(calls) != set(ROLES)
        or len(set(calls.values())) != len(ROLES)
    ):
        raise ValueError("rehearsal terminal role or app evidence is incomplete")
    exited = _timestamp(observed_exit.get("client_exited_at_utc"))
    timing: dict[str, tuple[datetime, datetime, str]] = {}
    summary_roles: dict[str, dict[str, object]] = {}
    for role in ROLES:
        record = roles[role]
        call_id = calls[role]
        raw = record.get("raw_worker_result") if isinstance(record, Mapping) else None
        payload = payloads[role]
        if (
            not isinstance(call_id, str)
            or not call_id.startswith("fc-")
            or not isinstance(raw, Mapping)
        ):
            raise ValueError(f"rehearsal {role} result identity differs")
        validated = record.get("validated_result")
        if (
            not isinstance(validated, Mapping)
            or validated.get("status") != "passed"
            or validated.get("role") != role
        ):
            raise ValueError(f"rehearsal {role} result was not validated")
        validate_stage_result(payload, raw)
        started = _timestamp(raw.get("started_at_utc"))
        finished = _timestamp(raw.get("finished_at_utc"))
        if finished <= started or finished <= exited:
            raise ValueError(f"rehearsal {role} did not finish after client exit")
        timing[role] = (started, finished, call_id)
        summary_roles[role] = {
            "call_id": call_id,
            "started_at_utc": raw["started_at_utc"],
            "finished_at_utc": raw["finished_at_utc"],
        }
    reference_finished = timing["unchanged"][1]
    if any(timing[role][0] < reference_finished for role in ROLES[1:]):
        raise ValueError("rehearsal arms started before the reference finished")
    arms_overlapped = max(timing[role][0] for role in ROLES[1:]) < min(
        timing[role][1] for role in ROLES[1:]
    )
    return {
        "app_id": app_id,
        "run_id": spec["run_id"],
        "plan_sha256": spec["plan_sha256"],
        "source_manifest_sha256": spec["source_manifest_sha256"],
        "client_exited_at_utc": observed_exit["client_exited_at_utc"],
        "arms_overlapped": arms_overlapped,
        "roles": summary_roles,
    }


def _run_synthetic_stage(payload: dict[str, object], expected_role: str) -> dict[str, object]:
    role = payload.get("role")
    marker = {
        "unchanged": "synthetic-reference",
        "control": "synthetic-control",
        "treatment": "synthetic-treatment",
    }.get(role)
    if (
        set(payload) != {"run_id", "role", "marker", "plan_sha256", "source_manifest_sha256"}
        or role != expected_role
        or marker is None
        or payload["marker"] != marker
        or payload["run_id"] != RUN_ID
    ):
        raise ValueError("remote stage input is not the fixed synthetic payload")
    started = datetime.now(UTC).isoformat()
    time.sleep(REFERENCE_DELAY_SECONDS if role == "unchanged" else 4)
    return payload | {"started_at_utc": started, "finished_at_utc": datetime.now(UTC).isoformat()}


def _reference_stage(payload: dict[str, object]) -> dict[str, object]:
    return _run_synthetic_stage(payload, "unchanged")


def _control_stage(payload: dict[str, object]) -> dict[str, object]:
    return _run_synthetic_stage(payload, "control")


def _treatment_stage(payload: dict[str, object]) -> dict[str, object]:
    return _run_synthetic_stage(payload, "treatment")


def _modal_coordinator_entry(
    spec: dict[str, object],
    payloads: dict[str, object],
    app_id: str,
    workers: dict[str, object],
) -> dict[str, object]:
    import modal

    from experiments import targeted_training_coordinator as coordinator
    from experiments.modal_targeted_training import ENVIRONMENT, VOLUME_NAME

    volume = modal.Volume.from_name(VOLUME_NAME, environment_name=ENVIRONMENT)
    claims = modal.Dict.from_name(
        coordinator.CLAIM_DICT_NAME, environment_name=ENVIRONMENT, create_if_missing=True
    )
    evidence = coordinator.VolumeEvidence(Path("/artifacts"), spec["run_id"], app_id, volume)
    return coordinator.execute_coordinator(
        spec,
        payloads,
        app_id,
        workers,
        coordinator_call_id=modal.current_function_call_id(),
        claim=lambda identity: coordinator.claim_once(
            claims, spec["run_id"], "coordinator", identity
        ),
        persist=evidence.persist,
        stop_app=lambda _app_id: {"deferred_to_host": True},
        validate_result=validate_stage_result,
    )


def _modal_options(limits: Mapping[str, object]) -> dict[str, object]:
    return {
        "cpu": (limits["cpu"], limits["cpu"]),
        "memory": (limits["memory_mib"], limits["memory_mib"]),
        "gpu": limits["gpu"],
        "timeout": limits["timeout_seconds"],
        "startup_timeout": limits["startup_timeout_seconds"],
        "retries": 0,
        "single_use_containers": True,
        "max_containers": 1,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window": 2,
        "serialized": False,
        "include_source": False,
    }


def _build_image(modal: object, root: Path, spec: Mapping[str, object]) -> object:
    image = modal.Image.debian_slim(python_version="3.12").env({"PYTHONPATH": "/root"})
    image = image.pip_install(f"pydantic=={spec['pydantic_version']}")
    for relative, digest in spec["source_manifest"].items():
        source = (root / relative).resolve(strict=True)
        if _sha256(source.read_bytes()) != digest:
            raise ValueError(f"source changed before image definition: {relative}")
        image = image.add_local_file(str(source), remote_path=f"/root/{relative}")
    return image


def _create_app(modal: object, image: object) -> tuple[object, object, dict[str, object]]:
    from experiments.modal_targeted_training import ENVIRONMENT, VOLUME_NAME

    app = modal.App(
        "reflex-targeted-training-disconnect-rehearsal-2026-10-05-r1",
        image=image,
        include_source=False,
    )
    options = _modal_options(STAGE_LIMITS)
    workers = {
        role: app.function(image=image, **options)(entry)
        for role, entry in zip(
            ROLES, (_reference_stage, _control_stage, _treatment_stage), strict=True
        )
    }
    volume = modal.Volume.from_name(VOLUME_NAME, environment_name=ENVIRONMENT)
    coordinator = app.function(
        image=image,
        volumes={"/artifacts": volume},
        **_modal_options(COORDINATOR_LIMITS),
    )(_modal_coordinator_entry)
    return app, coordinator, workers


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: object) -> None:
    from experiments import modal_targeted_training as launch

    if path.exists():
        if _read_json(path) != value:
            raise ValueError(f"rehearsal evidence already differs: {path.name}")
        return
    launch._write_json_exclusive(path, value)


def _dispatch_client(spec_dir: str, root: str, ci_path: str, clearance_path: str) -> None:
    import modal

    from experiments import modal_targeted_training as launch

    directory, project_root = Path(spec_dir), Path(root).resolve(strict=True)
    spec = verify_spec(directory, project_root)
    validate_clearance(spec, _read_json(Path(ci_path)), _read_json(Path(clearance_path)))
    begin_attempt(directory, spec)
    if importlib.metadata.version("modal") != spec["modal_sdk_version"]:
        raise ValueError("Modal SDK version differs from reviewed rehearsal spec")
    launch._modal_preflight(RUN_ID)
    image = _build_image(modal, project_root, spec)
    app, coordinator, workers = _create_app(modal, image)
    payloads = build_payloads(spec)
    with app.run(detach=True, environment_name=spec["environment"]):
        app_id = app.app_id
        if not isinstance(app_id, str) or re.fullmatch(r"ap-[A-Za-z0-9]+", app_id) is None:
            raise ValueError("detached rehearsal app ID is missing")
        launch._write_json_exclusive(
            directory / "app-id.json",
            {"app_id": app_id, "run_id": RUN_ID, "plan_sha256": spec["plan_sha256"]},
        )
        if not all(getattr(worker, "_is_hydrated", False) for worker in workers.values()):
            raise ValueError("CPU stage handles were not hydrated by app startup")
        call = coordinator.spawn(spec, payloads, app_id, workers)
        record = {
            "run_id": RUN_ID,
            "app_id": app_id,
            "coordinator_call_id": launch._require_call_id(call),
            "plan_sha256": spec["plan_sha256"],
            "source_manifest_sha256": spec["source_manifest_sha256"],
        }
        launch._write_json_exclusive(directory / "dispatch.json", record)
        os._exit(0)


def dispatch(
    spec_dir: str | Path,
    ci_path: str | Path,
    clearance_path: str | Path,
    source_root: str | Path = ".",
) -> dict[str, object]:
    """Run a detached dispatch client in a child, then record its actual exit."""
    from experiments import modal_targeted_training as launch

    directory = Path(spec_dir).expanduser().resolve(strict=True)
    root = Path(source_root).expanduser().resolve(strict=True)
    spec = verify_spec(directory, root)
    validate_clearance(spec, _read_json(Path(ci_path)), _read_json(Path(clearance_path)))
    if any((directory / name).exists() for name in ("attempt.json", "dispatch-observed.json")):
        raise ValueError("rehearsal already has a dispatch attempt; never resubmit")
    code = (
        "import sys; from experiments.modal_targeted_recovery_rehearsal import _dispatch_client; "
        "_dispatch_client(*sys.argv[1:])"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(directory),
            str(root),
            str(Path(ci_path).resolve()),
            str(Path(clearance_path).resolve()),
        ],
        cwd=root,
    )
    exit_code = process.wait()
    record = (
        _read_json(directory / "dispatch.json") if (directory / "dispatch.json").exists() else {}
    )
    observation = {
        **{
            key: record[key]
            for key in (
                "run_id",
                "app_id",
                "coordinator_call_id",
                "plan_sha256",
                "source_manifest_sha256",
            )
            if key in record
        },
        "client_pid": process.pid,
        "client_exit_code": exit_code,
        "client_exited_at_utc": launch._now_utc(),
        "acknowledged": exit_code == 0 and bool(record.get("coordinator_call_id")),
    }
    launch._write_json_exclusive(directory / "dispatch-observed.json", observation)
    if observation["acknowledged"] is not True:
        raise RuntimeError(
            "coordinator acknowledgement is unknown; recover manually and never resubmit"
        )
    return observation


def _volume_json(volume: object, path: str) -> object:
    chunks = volume.read_file(path)
    raw = b"".join(chunks)
    if not raw:
        raise ValueError("Modal Volume returned an empty rehearsal receipt")
    return json.loads(raw)


def collect(spec_dir: str | Path, source_root: str | Path = ".") -> dict[str, object]:
    """Recover committed synthetic evidence directly, then stop and verify the app."""
    import modal

    from experiments import modal_targeted_training as launch
    from experiments.targeted_training_coordinator import EVIDENCE_DIRECTORY

    directory = Path(spec_dir).expanduser().resolve(strict=True)
    root = Path(source_root).expanduser().resolve(strict=True)
    spec = verify_spec(directory, root)
    dispatch_record = _read_json(directory / "dispatch.json")
    observed = _read_json(directory / "dispatch-observed.json")
    if (
        not isinstance(dispatch_record, dict)
        or not isinstance(observed, dict)
        or observed.get("acknowledged") is not True
    ):
        raise ValueError("dispatch acknowledgement is missing; collection cannot authorize work")
    modal.config._set_profile(spec["profile"])
    volume = modal.Volume.from_name(launch.VOLUME_NAME, environment_name=spec["environment"])
    volume_path = f"{EVIDENCE_DIRECTORY}/{RUN_ID}/{dispatch_record['app_id']}/terminal.json"
    terminal = _volume_json(volume, volume_path)
    if not isinstance(terminal, dict):
        raise ValueError("Modal Volume terminal receipt is malformed")
    summary = validate_terminal(spec, build_payloads(spec), dispatch_record, observed, terminal)
    _write_once(directory / "raw-synthetic-receipt.json", terminal)
    teardown = launch._ModalProvider().verify_teardown(summary["app_id"])
    stopped = any(
        isinstance(item, dict)
        and item.get("state") == "APP_STATE_STOPPED"
        and type(item.get("n_tasks")) is int
        and item["n_tasks"] == 0
        for item in teardown.get("observations", [])
    )
    if teardown.get("verified") is not True or not stopped:
        raise RuntimeError("rehearsal app teardown is not verified stopped with zero tasks")
    proof = summary | {
        "verified": True,
        "coordinator_call_id": dispatch_record["coordinator_call_id"],
        "teardown": teardown,
    }
    _write_once(directory / "rehearsal-proof.json", proof)
    return proof


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dispatch_parser = commands.add_parser("dispatch")
    dispatch_parser.add_argument("--spec-dir", required=True)
    dispatch_parser.add_argument("--ci-proof", required=True)
    dispatch_parser.add_argument("--clearance", required=True)
    dispatch_parser.add_argument("--root", default=".")
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--spec-dir", required=True)
    collect_parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    if args.command == "dispatch":
        result = dispatch(args.spec_dir, args.ci_proof, args.clearance, args.root)
    else:
        result = collect(args.spec_dir, args.root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
