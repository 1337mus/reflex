from __future__ import annotations

import pytest


def test_plan_orders_reference_before_train_arms() -> None:
    from experiments import modal_targeted_training as modal

    plan = modal.build_launch_plan("run-1", {"source": "pin"})
    assert [role["role"] for role in plan["roles"]] == ["unchanged", "control", "treatment"]
    assert plan["roles"][0]["reference_first"] is True


def test_launch_rejects_stale_plan_digest_before_modal_preflight(tmp_path) -> None:
    from experiments import modal_targeted_training as modal

    plan = modal.build_launch_plan("run-1", {"source": "pin"})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(modal._json_line(plan))
    with pytest.raises(ValueError, match="digest"):
        modal.launch_plan(
            plan_path, "0" * 64, tmp_path / "missing", tmp_path / "output", root=tmp_path
        )


def test_execute_modal_roles_reference_failure_blocks_arms_and_cancels() -> None:
    from experiments import modal_targeted_training as modal

    class Cancel:
        def __init__(self, owner):
            self.owner = owner

        async def aio(self, **kwargs):
            self.owner.cancelled = kwargs

    class Call:
        object_id = "ref-call"

        def __init__(self):
            self.cancel = Cancel(self)

        def get(self, **_kwargs):
            raise RuntimeError("reference failed")

    ref = Call()

    class Worker:
        def __init__(self, call):
            self.call = call
            self.spawned = 0

        def spawn(self, _payload):
            self.spawned += 1
            return self.call

    control, treatment = Worker(Call()), Worker(Call())
    events = []

    class Provider:
        def stop(self, app_id):
            events.append({"app_id": app_id})
            return {"verified": True}

    receipt = modal._execute_modal_roles(
        type("App", (), {"app_id": "app-1"})(),
        {"unchanged": Worker(ref), "control": control, "treatment": treatment},
        {"unchanged": {}, "control": {}, "treatment": {}},
        lambda *event: events.append(event),
        Provider(),
        lambda _payload, raw: raw,
    )
    assert receipt["status"] == "failed"
    assert control.spawned == treatment.spawned == 0
    assert ref.cancelled == {"terminate_containers": True}


def test_execute_modal_roles_cancels_both_arms_after_failed_result() -> None:
    from experiments import modal_targeted_training as modal

    class Cancel:
        def __init__(self, owner):
            self.owner = owner

        async def aio(self, **_kwargs):
            self.owner.cancelled = True

    class Call:
        def __init__(self, identifier, raw):
            self.object_id = identifier
            self.raw = raw
            self.cancelled = False
            self.cancel = Cancel(self)

        def get(self, **_kwargs):
            return self.raw

    class Worker:
        def __init__(self, call):
            self.call = call

        def spawn(self, _payload):
            return self.call

    reference = Call("ref", {"status": "passed"})
    control = Call("control", {"status": "failed", "outputs": ["partial"]})
    treatment = Call("treatment", {"status": "passed"})
    events = []

    class Provider:
        def stop(self, app_id):
            events.append(("stop", app_id))
            return {"verified": True}

    receipt = modal._execute_modal_roles(
        type("App", (), {"app_id": "app-2"})(),
        {
            "unchanged": Worker(reference),
            "control": Worker(control),
            "treatment": Worker(treatment),
        },
        {"unchanged": {}, "control": {}, "treatment": {}},
        lambda kind, evidence: events.append((kind, evidence)),
        Provider(),
        lambda _payload, raw: raw,
    )
    assert receipt["status"] == "failed"
    assert control.cancelled and treatment.cancelled
    assert any(
        kind == "roles" and "raw_worker_result" in evidence.get("control", {})
        for kind, evidence in events
        if isinstance(evidence, dict)
    )
    assert ("stop", "app-2") in events


def test_stuck_remote_cancel_reaches_app_stop_within_deadline(monkeypatch) -> None:
    import asyncio
    import threading
    import time

    from experiments import modal_targeted_training as modal

    monkeypatch.setattr(modal, "_REMOTE_CANCEL_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(modal, "_REMOTE_CANCEL_BATCH_GRACE_SECONDS", 0.02)
    release = threading.Event()
    events = []

    class Cancel:
        async def aio(self, **_kwargs):
            while not release.is_set():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    continue

    class Call:
        object_id = "fc-stuck"
        cancel = Cancel()

        def get(self, **_kwargs):
            raise RuntimeError("synthetic get failure")

    class Worker:
        def spawn(self, _payload):
            return Call()

    class Provider:
        def stop(self, app_id):
            events.append(("stop", app_id))
            return {"verified": True}

    start = time.monotonic()
    try:
        result = modal._execute_modal_roles(
            type("App", (), {"app_id": "ap-stuck"})(),
            {role: Worker() for role in ("unchanged", "control", "treatment")},
            {role: {} for role in ("unchanged", "control", "treatment")},
            lambda kind, evidence: events.append((kind, evidence)),
            Provider(),
            lambda _payload, raw: raw,
        )
    finally:
        release.set()
    assert time.monotonic() - start < 0.5
    assert result["status"] == "failed"
    assert ("stop", "ap-stuck") in events
    assert any(
        kind == "cancellation" and evidence["errors"]["unchanged"] == "TimeoutError"
        for kind, evidence in events
        if isinstance(evidence, dict)
    )


def test_clearance_requires_matching_commit_ci_and_fresh_budget() -> None:
    from datetime import UTC, datetime

    from experiments import modal_targeted_training as modal

    commit = "a" * 40
    plan = modal.build_launch_plan("run-2", {"source": "b" * 64}, source_commit=commit)
    now = datetime.now(UTC).isoformat()
    clearance = {
        "plan_sha256": plan["plan_sha256"],
        "source_commit": commit,
        "source_pins_sha256": plan["source_pins_sha256"],
        "ci": {
            "head_sha": commit,
            "status": "completed",
            "conclusion": "success",
            "workflow_url": "https://example.test/run",
            "evidence_sha256": "c" * 64,
        },
        "account": {
            "profile": plan["profile"],
            "workspace": plan["workspace"],
            "environment": plan["environment"],
            "checked_at_utc": now,
            "evidence_sha256": "d" * 64,
        },
        "budget": {
            "status": "cleared",
            "checked_at_utc": now,
            "estimated_max_usd": "5",
            "approved_max_usd": "10",
            "evidence_sha256": "e" * 64,
        },
    }
    modal._validate_clearance(plan, clearance)
    clearance["ci"]["head_sha"] = "f" * 40
    with pytest.raises(ValueError, match="CI"):
        modal._validate_clearance(plan, clearance)
