from __future__ import annotations

import pytest

from experiments import modal_targeted_training as launch


def test_duplicate_coordinator_claim_never_spawns_workers() -> None:
    from experiments import targeted_training_coordinator as coordinator

    plan = launch.build_launch_plan("claim-run", {"source": "a" * 64})
    spawned: list[str] = []

    class Worker:
        def spawn(self, _payload: object) -> None:
            spawned.append("gpu")

    result = coordinator.execute_coordinator(
        plan,
        {role: {} for role in ("unchanged", "control", "treatment")},
        "ap-claim",
        {role: Worker() for role in ("unchanged", "control", "treatment")},
        claim=lambda _identity: False,
        persist=lambda _kind, _evidence: None,
        stop_app=lambda _app_id: None,
        validate_result=lambda _payload, raw: raw,
    )
    assert result["status"] == "duplicate"
    assert spawned == []


def test_claimed_coordinator_runs_reference_then_arms_and_persists_raw_terminal() -> None:
    from experiments import targeted_training_coordinator as coordinator

    roles = ("unchanged", "control", "treatment")
    payloads = {role: {"role": role, "payload_sha256": role} for role in roles}
    plan = launch.build_launch_plan("complete-run", {"source": "a" * 64}, payloads)
    events: list[tuple[str, object]] = []
    spawned: list[str] = []

    class Call:
        def __init__(self, role: str):
            self.object_id = f"fc-{role}"
            self.role = role

        def get(self, **_kwargs: object) -> dict[str, str]:
            return {"status": "passed", "role": self.role}

    class Worker:
        def __init__(self, role: str):
            self.role = role

        def spawn(self, _payload: object) -> Call:
            if self.role != "unchanged":
                assert any(kind == "roles" and "unchanged" in value for kind, value in events)
            spawned.append(self.role)
            return Call(self.role)

    result = coordinator.execute_coordinator(
        plan,
        payloads,
        "ap-complete",
        {role: Worker(role) for role in roles},
        claim=lambda _identity: True,
        persist=lambda kind, value: events.append((kind, value)),
        stop_app=lambda _app_id: None,
        validate_result=lambda _payload, raw: raw,
        coordinator_call_id="fc-coordinator",
    )
    assert result["status"] == "passed"
    assert spawned == list(roles)
    terminal = [value for kind, value in events if kind == "terminal"]
    assert len(terminal) == 1
    assert terminal[0]["lifecycle"]["status"] == "passed"
    assert terminal[0]["lifecycle"]["coordinator_call_id"] == "fc-coordinator"
    assert terminal[0]["roles"]["unchanged"]["raw_worker_result"]["role"] == "unchanged"


def test_persistence_failure_cancels_started_call_and_requests_stop() -> None:
    from experiments import targeted_training_coordinator as coordinator

    class Cancel:
        async def aio(self, **_kwargs: object) -> None:
            call.cancelled = True

    class Call:
        object_id = "fc-ref"
        cancel = Cancel()
        cancelled = False

        def get(self, **_kwargs: object) -> dict[str, str]:
            return {"status": "passed"}

    call = Call()

    class Worker:
        def spawn(self, _payload: object) -> Call:
            return call

    stopped: list[str] = []
    plan = launch.build_launch_plan("storage-run", {"source": "a" * 64})

    def persist(kind: str, _value: object) -> None:
        if kind == "roles":
            raise OSError("volume commit failed")

    with pytest.raises(OSError, match="volume"):
        coordinator.execute_coordinator(
            plan,
            {role: {} for role in ("unchanged", "control", "treatment")},
            "ap-storage",
            {role: Worker() for role in ("unchanged", "control", "treatment")},
            claim=lambda _identity: True,
            persist=persist,
            stop_app=stopped.append,
            validate_result=lambda _payload, raw: raw,
        )
    assert call.cancelled
    assert stopped == ["ap-storage"]


def test_recovery_rejects_changed_terminal_and_unverified_shutdown() -> None:
    from experiments import targeted_training_coordinator as coordinator

    plan = launch.build_launch_plan("recover-run", {"source": "a" * 64})
    marker = {
        "run_id": "recover-run",
        "plan_sha256": plan["plan_sha256"],
        "app_id": "ap-recover",
    }
    terminal = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "run_id": "recover-run",
        "status": "pending_shutdown",
        "plan": plan,
        "role_payloads": {},
        "roles": {},
        "lifecycle": {
            "status": "passed",
            "app_id": "ap-recover",
            "coordinator_call_id": "fc-actual",
            "calls": {},
        },
    }
    invoked: list[object] = []
    teardown = {"app_id": "ap-recover", "verified": False, "observations": []}
    with pytest.raises(ValueError, match="stopped|shutdown|teardown"):
        coordinator.finalize_remote_receipt(
            plan, {}, marker, terminal, teardown, validate_complete=invoked.append
        )
    assert invoked == []
    changed = dict(terminal, run_id="other-run")
    teardown = {
        "app_id": "ap-recover",
        "verified": True,
        "observations": [{"state": "APP_STATE_STOPPED", "n_tasks": 0}],
    }
    with pytest.raises(ValueError, match="identity"):
        coordinator.finalize_remote_receipt(
            plan, {}, marker, changed, teardown, validate_complete=invoked.append
        )
    assert invoked == []
    changed_marker = dict(marker, source_commit="other-source")
    with pytest.raises(ValueError, match="identity"):
        coordinator.finalize_remote_receipt(
            plan, {}, changed_marker, terminal, teardown, validate_complete=invoked.append
        )
    changed_marker = dict(marker, coordinator_call_id="fc-different")
    with pytest.raises(ValueError, match="identity"):
        coordinator.finalize_remote_receipt(
            plan, {}, changed_marker, terminal, teardown, validate_complete=invoked.append
        )


def test_volume_evidence_commits_each_transition_and_terminal(tmp_path) -> None:
    import json

    from experiments import targeted_training_coordinator as coordinator

    class Volume:
        commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = Volume()
    evidence = coordinator.VolumeEvidence(tmp_path, "volume-run", "ap-volume", volume)
    evidence.persist("calls", {"unchanged": "fc-reference"})
    terminal = {"status": "failed", "roles": {"unchanged": {"raw_worker_result": {"x": 1}}}}
    evidence.persist("terminal", terminal)
    assert volume.commits == 2
    assert json.loads((evidence.path / "terminal.json").read_text()) == terminal
    lines = (evidence.path / "events.jsonl").read_text().splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["calls", "terminal"]


def test_remote_claim_is_one_atomic_insert_per_role() -> None:
    from experiments import targeted_training_coordinator as coordinator

    calls: list[tuple[object, object, object]] = []

    class Dict:
        def put(self, key: object, value: object, *, skip_if_exists: bool) -> bool:
            calls.append((key, value, skip_if_exists))
            return len(calls) == 1

    store = Dict()
    assert coordinator.claim_once(store, "run-1", "unchanged", {"payload_sha256": "a" * 64})
    assert not coordinator.claim_once(store, "run-1", "unchanged", {"payload_sha256": "a" * 64})
    assert calls[0][0] == calls[1][0]
    assert all(item[2] is True for item in calls)


def test_remote_preflight_rejects_changed_plan_before_claim() -> None:
    from experiments import targeted_training_coordinator as coordinator

    plan = launch.build_launch_plan("immutable-run", {"source": "a" * 64})
    plan["run_id"] = "changed-run"
    with pytest.raises(ValueError, match="plan"):
        coordinator.validate_remote_inputs(plan, {}, "ap-immutable")


def test_independent_process_finishes_roles_after_dispatch_process_exits(tmp_path) -> None:
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path

    child = """
import sys, time
from pathlib import Path
from experiments import modal_targeted_training as launch
from experiments.targeted_training_coordinator import VolumeEvidence, execute_coordinator
time.sleep(0.4)
class Volume:
    def commit(self): pass
class Call:
    def __init__(self, role): self.object_id, self.role = "fc-" + role, role
    def get(self, **_kwargs): return {"status": "passed", "role": self.role}
class Worker:
    def __init__(self, role): self.role = role
    def spawn(self, _payload): return Call(self.role)
roles = ("unchanged", "control", "treatment")
payloads = {role: {"role": role, "payload_sha256": role} for role in roles}
plan = launch.build_launch_plan("disconnect-run", {"source": "a" * 64}, payloads)
evidence = VolumeEvidence(Path(sys.argv[1]), "disconnect-run", "ap-disconnect", Volume())
execute_coordinator(plan, payloads, "ap-disconnect", {r: Worker(r) for r in roles},
    claim=lambda _identity: True, persist=evidence.persist, stop_app=lambda _id: None,
    validate_result=lambda _payload, raw: raw, coordinator_call_id="fc-coordinator")
"""
    dispatcher = """
import subprocess, sys
subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
"""
    subprocess.run(
        [sys.executable, "-c", dispatcher, child, str(tmp_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        timeout=5,
    )
    terminal_path = tmp_path / "targeted-coordinator/disconnect-run/ap-disconnect/terminal.json"
    assert not terminal_path.exists()
    deadline = time.monotonic() + 5
    while not terminal_path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    terminal = json.loads(terminal_path.read_text())
    assert terminal["status"] == "pending_shutdown"
    assert set(terminal["roles"]) == {"unchanged", "control", "treatment"}
