from __future__ import annotations

import pytest


def test_plan_orders_reference_before_train_arms() -> None:
    from experiments import modal_targeted_training as modal

    plan = modal.build_launch_plan("run-1", {"source": "pin"})
    assert [role["role"] for role in plan["roles"]] == ["unchanged", "control", "treatment"]
    assert plan["roles"][0]["reference_first"] is True


def test_plan_binds_single_use_detached_cpu_coordinator_limits() -> None:
    from experiments import modal_targeted_training as modal

    plan = modal.build_launch_plan("run-1", {"source": "pin"})
    assert plan["execution"] == {
        "detach": True,
        "coordinator": {
            "cpu": 1,
            "memory_mib": 4096,
            "timeout_seconds": 9000,
            "startup_timeout_seconds": 300,
            "max_containers": 1,
            "max_inputs": 1,
            "scaledown_window_seconds": 2,
            "retries": 0,
            "single_use_containers": True,
        },
    }


def test_worker_refuses_duplicate_claim_before_runtime_import(monkeypatch) -> None:
    from experiments import modal_targeted_training as modal

    def duplicate(_payload, _role):
        raise ValueError("duplicate role attempt")

    monkeypatch.setattr(modal, "_claim_worker_role", duplicate, raising=False)
    with pytest.raises(ValueError, match="duplicate"):
        modal._modal_reference_worker({"role": "unchanged"})


def test_modal_registration_adds_one_bounded_cpu_coordinator() -> None:
    from experiments import modal_targeted_training as modal

    definitions: list[tuple[dict[str, object], object]] = []

    class Volume:
        def with_mount_options(self, **_kwargs):
            return self

        @classmethod
        def from_name(cls, *_args, **_kwargs):
            return cls()

    class App:
        def __init__(self, *_args, **_kwargs):
            pass

        def function(self, **options):
            def register(entry):
                definitions.append((options, entry))
                return entry

            return register

    fake_modal = type("Modal", (), {"Volume": Volume, "App": App})
    _app, workers, coordinator = modal._create_modal_workers(fake_modal, object(), "run")
    assert set(workers) == {"unchanged", "control", "treatment"}
    assert coordinator is definitions[-1][1]
    options = definitions[-1][0]
    assert "gpu" not in options
    assert options["cpu"] == (1.0, 1.0)
    assert options["memory"] == (4096, 4096)
    assert options["timeout"] == 9000
    assert options["retries"] == 0
    assert options["single_use_containers"] is True


def test_detached_dispatch_saves_app_id_before_one_spawn(tmp_path) -> None:
    import json

    from experiments import modal_targeted_training as modal

    class Run:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    class App:
        app_id = "ap-dispatch"

        def run(self, **kwargs):
            assert kwargs["detach"] is True
            return Run()

    class Coordinator:
        spawned = 0

        def spawn(self, *_args):
            assert json.loads((tmp_path / "app.json").read_text())["app_id"] == "ap-dispatch"
            self.spawned += 1
            return type("Call", (), {"object_id": "fc-coordinator"})()

    worker = object()
    coordinator = Coordinator()
    result = modal._dispatch_detached(
        App(), {"unchanged": worker}, coordinator, {"run_id": "run"}, {}, tmp_path
    )
    assert result["status"] == "dispatched"
    assert coordinator.spawned == 1
    assert result["coordinator_call_id"] == "fc-coordinator"


def test_detached_dispatch_suppressed_interrupt_returns_unknown_without_retry(tmp_path) -> None:
    from experiments import modal_targeted_training as modal

    class Run:
        def __enter__(self):
            return None

        def __exit__(self, error_type, *_args):
            assert error_type is KeyboardInterrupt
            return True

    class App:
        app_id = "ap-suppressed"

        def run(self, **_kwargs):
            return Run()

    class Coordinator:
        spawned = 0

        def spawn(self, *_args):
            self.spawned += 1
            raise KeyboardInterrupt()

    coordinator = Coordinator()
    result = modal._dispatch_detached(
        App(), {}, coordinator, {"run_id": "suppressed"}, {}, tmp_path
    )
    assert result["status"] == "unknown"
    assert result["app_id"] == "ap-suppressed"
    assert coordinator.spawned == 1
    assert (tmp_path / "app.json").exists()


def test_recovery_without_terminal_reports_pending_without_worker_launch(
    tmp_path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace

    from experiments import modal_targeted_training as modal

    plan = modal.build_launch_plan("recover-pending", {"source": "a" * 64})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(modal._json_line(plan))
    output = tmp_path / "attempt"
    output.mkdir()
    marker = {
        "run_id": plan["run_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_commit"],
        "created_at_utc": modal._now_utc(),
    }
    (output / "attempt.json").write_text(modal._json_line(marker))
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "recover-pending.targeted-attempt.json").write_text(
        modal._json_line(marker)
    )
    (output / "app.json").write_text(modal._json_line({**marker, "app_id": "ap-pending"}))
    (output / "dispatch.json").write_text(
        modal._json_line(
            {
                "status": "dispatched",
                "app_id": "ap-pending",
                "run_id": plan["run_id"],
                "plan_sha256": plan["plan_sha256"],
                "coordinator_call_id": "fc-pending",
            }
        )
    )

    class Provider:
        def observe(self, _app_id):
            return {"state": "APP_STATE_DETACHED", "n_tasks": 1}

        def stop(self, _app_id):
            raise AssertionError("pending recovery stopped work")

    result = modal.recover_plan(
        plan_path,
        plan["plan_sha256"],
        output,
        root=tmp_path,
        read_terminal=lambda _run_id, _app_id: None,
        provider=Provider(),
        call_status=lambda _call_id: "pending",
    )
    assert result["status"] == "pending"
    assert not (output / "receipt.json").exists()

    class ModalError(Exception):
        pass

    class ResourceExhaustedError(ModalError):
        pass

    class ServiceError(ModalError):
        pass

    class Call:
        error = ResourceExhaustedError
        queried = 0

        def get(self, **_kwargs):
            Call.queried += 1
            raise self.error("temporary provider error")

    fake_modal = SimpleNamespace(
        FunctionCall=SimpleNamespace(from_id=lambda _id: Call()),
        exception=SimpleNamespace(
            Error=ModalError,
            ConnectionError=type("ConnectionError", (ModalError,), {}),
            AuthError=type("AuthError", (ModalError,), {}),
            NotFoundError=type("NotFoundError", (ModalError,), {}),
            FunctionTimeoutError=type("FunctionTimeoutError", (ModalError,), {}),
            InternalFailure=type("InternalFailure", (ModalError,), {}),
            RemoteError=type("RemoteError", (ModalError,), {}),
            ExecutionError=type("ExecutionError", (ModalError,), {}),
            OutputExpiredError=type("OutputExpiredError", (ModalError,), {}),
        ),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    for error_type in (ResourceExhaustedError, ServiceError):
        Call.error = error_type
        result = modal.recover_plan(
            plan_path,
            plan["plan_sha256"],
            output,
            root=tmp_path,
            read_terminal=lambda _run_id, _app_id: None,
            provider=Provider(),
            call_status=modal._coordinator_call_status,
        )
        assert result["status"] == "pending"
        assert result["call_state"] == "unknown"
    Call.error = TimeoutError
    result = modal.recover_plan(
        plan_path,
        plan["plan_sha256"],
        output,
        root=tmp_path,
        read_terminal=lambda _run_id, _app_id: None,
        provider=Provider(),
        call_status=modal._coordinator_call_status,
    )
    assert result["status"] == "pending"
    assert result["call_state"] == "pending"
    stopped: list[str] = []

    class StoppingProvider(Provider):
        def stop(self, app_id):
            stopped.append(app_id)
            return {"app_id": app_id, "verified": True}

    class RemoteWorkerError(RuntimeError):
        __line_cache__ = {}

    class RemoteWorkerTimeout(TimeoutError):
        __line_cache__ = {}

    class RemoteServiceError(ServiceError):
        __line_cache__ = {}

    for error_type in (
        fake_modal.exception.FunctionTimeoutError,
        RemoteWorkerError,
        RemoteWorkerTimeout,
        RemoteServiceError,
    ):
        Call.error = error_type
        result = modal.recover_plan(
            plan_path,
            plan["plan_sha256"],
            output,
            root=tmp_path,
            read_terminal=lambda _run_id, _app_id: None,
            provider=StoppingProvider(),
            call_status=modal._coordinator_call_status,
        )
        assert result["status"] == "incomplete"
        assert stopped[-1] == "ap-pending"
        (output / "incomplete.json").unlink()
    (output / "dispatch.json").unlink()
    remote_identity = {
        "run_id": plan["run_id"],
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_commit"],
        "app_id": "ap-pending",
        "coordinator_call_id": "fc-pending",
    }
    Call.error = fake_modal.exception.FunctionTimeoutError
    result = modal.recover_plan(
        plan_path,
        plan["plan_sha256"],
        output,
        root=tmp_path,
        read_terminal=lambda _run_id, _app_id: None,
        read_identity=lambda _run_id, _app_id: remote_identity,
        provider=StoppingProvider(),
        call_status=modal._coordinator_call_status,
    )
    assert result["status"] == "incomplete"
    assert stopped[-1] == "ap-pending"
    (output / "incomplete.json").unlink()
    queried_before = Call.queried
    stopped_before = len(stopped)
    with pytest.raises(ValueError, match="identity"):
        modal.recover_plan(
            plan_path,
            plan["plan_sha256"],
            output,
            root=tmp_path,
            read_terminal=lambda _run_id, _app_id: None,
            read_identity=lambda _run_id, _app_id: {**remote_identity, "app_id": "ap-other"},
            provider=StoppingProvider(),
            call_status=modal._coordinator_call_status,
        )
    assert Call.queried == queried_before
    assert len(stopped) == stopped_before


@pytest.mark.parametrize(
    ("observed_state", "n_tasks"),
    [("APP_STATE_STOPPED", 0), ("APP_STATE_DETACHED", 1)],
)
def test_recovery_rereads_terminal_before_declaring_completed_call_incomplete(
    tmp_path, monkeypatch, observed_state, n_tasks
) -> None:
    from experiments import modal_targeted_training as modal
    from experiments import targeted_training_coordinator as coordinator

    plan = modal.build_launch_plan("race-run", {"source": "a" * 64})
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    plan_path = plan_dir / "plan.json"
    plan_path.write_text(modal._json_line(plan))
    payloads = {role: {} for role in ("unchanged", "control", "treatment")}
    for role, payload in payloads.items():
        (plan_dir / f"{role}-payload.json").write_text(modal._json_line(payload))
    output = tmp_path / "attempt"
    output.mkdir()
    marker = {
        "run_id": "race-run",
        "plan_sha256": plan["plan_sha256"],
        "source_commit": plan["source_commit"],
        "created_at_utc": modal._now_utc(),
    }
    (output / "attempt.json").write_text(modal._json_line(marker))
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "race-run.targeted-attempt.json").write_text(modal._json_line(marker))
    (output / "app.json").write_text(modal._json_line({**marker, "app_id": "ap-race"}))
    (output / "dispatch.json").write_text(
        modal._json_line(
            {
                "status": "dispatched",
                "run_id": "race-run",
                "plan_sha256": plan["plan_sha256"],
                "app_id": "ap-race",
                "coordinator_call_id": "fc-race",
            }
        )
    )
    terminal = {
        "status": "pending_shutdown",
        "experiment_id": plan["experiment_id"],
        "run_id": "race-run",
        "plan": plan,
        "role_payloads": payloads,
        "lifecycle": {"status": "passed", "app_id": "ap-race", "coordinator_call_id": "fc-race"},
    }
    reads: list[int] = []

    def read_terminal(_run_id, _app_id):
        reads.append(1)
        return None if len(reads) == 1 else terminal

    class Provider:
        def observe(self, _app_id):
            return {"state": observed_state, "n_tasks": n_tasks}

        def stop(self, _app_id):
            raise AssertionError("racing completion must use terminal receipt")

        def verify_teardown(self, app_id):
            return {
                "app_id": app_id,
                "verified": True,
                "observations": [{"state": "APP_STATE_STOPPED", "n_tasks": 0}],
            }

    monkeypatch.setattr(coordinator, "finalize_remote_receipt", lambda *_args: {"status": "passed"})
    result = modal.recover_plan(
        plan_path,
        plan["plan_sha256"],
        output,
        root=tmp_path,
        read_terminal=read_terminal,
        provider=Provider(),
        call_status=lambda _id: "complete",
    )
    assert result["status"] == "passed"
    assert len(reads) == 2
    assert not (output / "incomplete.json").exists()


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
