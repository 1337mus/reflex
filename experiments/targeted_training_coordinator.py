"""Durable, single-attempt CPU coordinator for targeted Modal training."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from experiments.targeted_training_core import EXPERIMENT_ID, ROLES

EVIDENCE_DIRECTORY = "targeted-coordinator"
CLAIM_DICT_NAME = "reflex-targeted-execution-claims"


def claim_once(store: Any, run_id: str, role: str, evidence: Mapping[str, object]) -> bool:
    """Claim one run/role with Modal's server-side if-not-exists operation."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", run_id) is None or role not in (
        *ROLES,
        "coordinator",
    ):
        raise ValueError("remote claim identity is malformed")
    return bool(store.put(f"{EXPERIMENT_ID}:{run_id}:{role}", dict(evidence), skip_if_exists=True))


class VolumeEvidence:
    """Persist every transition before allowing the next remote action."""

    def __init__(self, mount: Path, run_id: str, app_id: str, volume: Any) -> None:
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", run_id) is None
            or re.fullmatch(r"ap-[A-Za-z0-9]+", app_id) is None
        ):
            raise ValueError("coordinator evidence identity is malformed")
        self.path = mount / EVIDENCE_DIRECTORY / run_id / app_id
        self.path.mkdir(parents=True, exist_ok=True)
        self.volume = volume

    def persist(self, kind: str, evidence: object) -> None:
        from experiments import modal_targeted_training as launch

        if kind == "identity":
            launch._write_json_exclusive(self.path / "identity.json", evidence)
        if kind == "terminal":
            launch._write_json_exclusive(self.path / "terminal.json", evidence)
        event = {"kind": kind, "at_utc": launch._now_utc(), "evidence": evidence}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.volume.commit()


def validate_remote_inputs(
    plan: Mapping[str, object], role_payloads: Mapping[str, object], app_id: str
) -> None:
    """Check the complete frozen execution contract before claiming or spawning."""
    from experiments import modal_targeted_training as launch
    from experiments import targeted_training_core as core

    if not isinstance(app_id, str) or re.fullmatch(r"ap-[A-Za-z0-9]+", app_id) is None:
        raise ValueError("remote app identity is malformed")
    if plan.get("experiment_id") != EXPERIMENT_ID or plan.get(
        "plan_sha256"
    ) != core.canonical_sha256({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise ValueError("remote plan identity differs")
    if set(role_payloads) != set(ROLES):
        raise ValueError("remote role payload set differs")
    for role in ROLES:
        payload = core.validate_role_payload(role_payloads[role])
        if (
            payload["role"] != role
            or payload["run_id"] != plan.get("run_id")
            or payload["source_pins"] != plan.get("source_pins")
        ):
            raise ValueError("remote role source or run identity differs")
    expected = launch.build_launch_plan(
        plan.get("run_id"),
        plan.get("source_pins"),
        role_payloads,
        source_commit=plan.get("source_commit"),
        bundle_manifest_sha256=plan.get("bundle_manifest_sha256"),
    )
    if dict(plan) != expected:
        raise ValueError("remote plan differs from fixed execution contract")


def finalize_remote_receipt(
    plan: Mapping[str, object],
    role_payloads: Mapping[str, object],
    marker: Mapping[str, object],
    terminal: Mapping[str, object],
    teardown: Mapping[str, object],
    *,
    validate_complete: Callable[[dict[str, object]], object] | None = None,
) -> dict[str, object]:
    """Promote remote raw evidence only after independent app teardown proof."""
    validate_remote_receipt_identity(plan, role_payloads, marker, terminal)
    app_id = marker.get("app_id")
    lifecycle = terminal["lifecycle"]
    assert isinstance(app_id, str) and isinstance(lifecycle, dict)
    if (
        teardown.get("app_id") != app_id
        or teardown.get("verified") is not True
        or not isinstance(teardown.get("observations"), list)
        or not any(
            isinstance(item, dict)
            and item.get("state") == "APP_STATE_STOPPED"
            and type(item.get("n_tasks")) is int
            and item["n_tasks"] == 0
            for item in teardown["observations"]
        )
    ):
        raise ValueError("app shutdown is not verified stopped with zero tasks")
    receipt = dict(terminal)
    receipt["lifecycle"] = {**lifecycle, "teardown": dict(teardown)}
    if terminal.get("status") == "pending_shutdown" and lifecycle.get("status") == "passed":
        receipt["status"] = "passed"
        if validate_complete is None:
            from experiments.analyze_targeted_training import validate_complete_receipt

            validate_complete_receipt(
                receipt, expected_plan_sha256=plan["plan_sha256"], plan=dict(plan)
            )
        else:
            validate_complete(receipt)
    else:
        receipt["status"] = "failed"
    return receipt


def validate_remote_receipt_identity(
    plan: Mapping[str, object],
    role_payloads: Mapping[str, object],
    marker: Mapping[str, object],
    terminal: Mapping[str, object],
) -> None:
    """Check remote and local bindings before asking Modal to stop any app."""
    app_id = marker.get("app_id")
    if (
        not isinstance(app_id, str)
        or re.fullmatch(r"ap-[A-Za-z0-9]+", app_id) is None
        or marker.get("run_id") != plan.get("run_id")
        or marker.get("plan_sha256") != plan.get("plan_sha256")
        or marker.get("source_commit") != plan.get("source_commit")
        or terminal.get("experiment_id") != EXPERIMENT_ID
        or terminal.get("run_id") != plan.get("run_id")
        or terminal.get("plan") != dict(plan)
        or terminal.get("role_payloads") != dict(role_payloads)
    ):
        raise ValueError("remote receipt or attempt identity differs")
    lifecycle = terminal.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("app_id") != app_id:
        raise ValueError("remote app identity differs")
    call_id = lifecycle.get("coordinator_call_id")
    if (
        not isinstance(call_id, str)
        or re.fullmatch(r"fc-[A-Za-z0-9]+", call_id) is None
        or marker.get("coordinator_call_id", call_id) != call_id
    ):
        raise ValueError("remote coordinator call identity differs")


def execute_coordinator(
    plan: Mapping[str, object],
    role_payloads: Mapping[str, object],
    app_id: str,
    workers: Mapping[str, Any],
    *,
    claim: Callable[[dict[str, object]], bool],
    persist: Callable[[str, object], None],
    stop_app: Callable[[str], object],
    validate_result: Callable[[object, object], dict[str, object]],
    coordinator_call_id: str | None = None,
) -> dict[str, object]:
    from experiments import modal_targeted_training as launch

    identity = {
        "run_id": plan["run_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan.get("source_commit"),
        "app_id": app_id,
        "coordinator_call_id": coordinator_call_id,
    }
    try:
        if not claim(identity):
            return {"status": "duplicate", **identity}
        persist("identity", identity)

        class DeferredStop:
            def stop(self, _app_id: str) -> dict[str, object]:
                return {"deferred_until_terminal_commit": True}

        lifecycle = launch._execute_modal_roles(
            type("BoundApp", (), {"app_id": app_id})(),
            workers,
            role_payloads,
            persist,
            DeferredStop(),
            validate_result,
        )
        terminal = launch.build_receipt(plan, role_payloads, lifecycle["roles"])
        terminal["status"] = "pending_shutdown" if lifecycle["status"] == "passed" else "failed"
        terminal["lifecycle"] = {
            **{key: value for key, value in lifecycle.items() if key != "roles"},
            "coordinator_call_id": coordinator_call_id,
        }
        persist("terminal", terminal)
        if lifecycle["status"] != "passed":
            stop_app(app_id)
        return lifecycle
    except BaseException:
        try:
            stop_app(app_id)
        except BaseException:
            pass
        raise


def _modal_coordinator_entry(
    plan: dict[str, object],
    role_payloads: dict[str, object],
    app_id: str,
    workers: Mapping[str, Any],
) -> dict[str, object]:
    """Detached CPU entrypoint; all GPU sequencing and durable events live here."""
    import modal

    from experiments import modal_targeted_training as launch
    from experiments import targeted_training_core as core

    volume = modal.Volume.from_name(launch.VOLUME_NAME, environment_name=launch.ENVIRONMENT)
    evidence = VolumeEvidence(Path(launch.VOLUME_ROOT), str(plan.get("run_id")), app_id, volume)
    call_id = modal.current_function_call_id()

    def stop_app(identifier: str) -> object:
        return launch._ModalProvider().stop(identifier)

    try:
        validate_remote_inputs(plan, role_payloads, app_id)
    except BaseException as error:
        try:
            evidence.persist("preflight_failure", {"type": type(error).__name__})
        finally:
            try:
                stop_app(app_id)
            except BaseException:
                pass
        raise
    store = modal.Dict.from_name(
        CLAIM_DICT_NAME, environment_name=launch.ENVIRONMENT, create_if_missing=True
    )
    result = execute_coordinator(
        plan,
        role_payloads,
        app_id,
        workers,
        claim=lambda identity: claim_once(
            store,
            str(plan["run_id"]),
            "coordinator",
            {**identity, "claimed_at_utc": launch._now_utc()},
        ),
        persist=evidence.persist,
        stop_app=stop_app,
        validate_result=core.validate_completed_result,
        coordinator_call_id=call_id,
    )
    return {"status": result["status"], "run_id": plan["run_id"], "app_id": app_id}
