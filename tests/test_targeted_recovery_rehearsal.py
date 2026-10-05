import hashlib
import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from experiments import modal_targeted_recovery_rehearsal as rehearsal


def test_rehearsal_spec_has_fixed_cpu_only_reference_first_bounds() -> None:
    spec = rehearsal.build_spec(
        source_commit="a" * 40,
        source_manifest={"experiments/targeted_training_coordinator.py": "b" * 64},
    )

    assert spec["run_id"] == "disconnect-rehearsal-2026-10-05-r1"
    assert spec["roles"] == ["unchanged", "control", "treatment"]
    assert spec["reference_delay_seconds"] == 10
    assert spec["stage"] == {
        "cpu": 1,
        "memory_mib": 1024,
        "gpu": None,
        "retries": 0,
        "single_use_containers": True,
        "max_containers": 1,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window_seconds": 2,
        "timeout_seconds": 120,
        "startup_timeout_seconds": 60,
    }
    assert spec["coordinator"] == {
        "cpu": 1,
        "memory_mib": 1024,
        "gpu": None,
        "retries": 0,
        "single_use_containers": True,
        "max_containers": 1,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window_seconds": 2,
        "timeout_seconds": 180,
        "startup_timeout_seconds": 60,
    }


def test_role_payloads_contain_only_synthetic_identity_and_source_bindings() -> None:
    spec = rehearsal.build_spec(
        source_commit="a" * 40,
        source_manifest={"experiments/targeted_training_coordinator.py": "b" * 64},
    )

    payloads = rehearsal.build_payloads(spec)

    assert tuple(payloads) == ("unchanged", "control", "treatment")
    assert [payload["marker"] for payload in payloads.values()] == [
        "synthetic-reference",
        "synthetic-control",
        "synthetic-treatment",
    ]
    for role, payload in payloads.items():
        assert set(payload) == {
            "run_id",
            "role",
            "marker",
            "plan_sha256",
            "source_manifest_sha256",
        }
        assert payload["run_id"] == spec["run_id"]
        assert payload["role"] == role
        assert payload["plan_sha256"] == spec["plan_sha256"]
        assert payload["source_manifest_sha256"] == spec["source_manifest_sha256"]


def test_terminal_validation_proves_disconnect_order_and_parallel_arms() -> None:
    spec = rehearsal.build_spec(
        source_commit="a" * 40,
        source_manifest={"experiments/targeted_training_coordinator.py": "b" * 64},
    )
    payloads = rehearsal.build_payloads(spec)
    app_id = "ap-rehearsal"
    times = {
        "unchanged": ("2026-10-05T00:00:00+00:00", "2026-10-05T00:00:10+00:00"),
        "control": ("2026-10-05T00:00:10+00:00", "2026-10-05T00:00:14+00:00"),
        "treatment": ("2026-10-05T00:00:10+00:00", "2026-10-05T00:00:14+00:00"),
    }
    roles = {}
    for role, payload in payloads.items():
        started, finished = times[role]
        raw = {
            **payload,
            "started_at_utc": started,
            "finished_at_utc": finished,
        }
        roles[role] = {
            "raw_worker_result": raw,
            "validated_result": {"status": "passed", "role": role},
        }
    terminal = {
        "status": "pending_shutdown",
        "experiment_id": spec["experiment_id"],
        "run_id": spec["run_id"],
        "plan": spec,
        "role_payloads": payloads,
        "roles": roles,
        "lifecycle": {
            "status": "passed",
            "app_id": app_id,
            "coordinator_call_id": "fc-coordinator",
            "calls": {role: f"fc-{role}" for role in payloads},
        },
    }
    dispatch = {
        "run_id": spec["run_id"],
        "app_id": app_id,
        "coordinator_call_id": "fc-coordinator",
        "plan_sha256": spec["plan_sha256"],
        "source_manifest_sha256": spec["source_manifest_sha256"],
    }
    observed_exit = {
        "client_exited_at_utc": "2026-10-05T00:00:02+00:00",
        "run_id": spec["run_id"],
        "app_id": app_id,
        "coordinator_call_id": "fc-coordinator",
        "plan_sha256": spec["plan_sha256"],
        "source_manifest_sha256": spec["source_manifest_sha256"],
    }

    summary = rehearsal.validate_terminal(spec, payloads, dispatch, observed_exit, terminal)

    assert summary["app_id"] == app_id
    assert summary["roles"]["unchanged"]["finished_at_utc"] == times["unchanged"][1]
    assert summary["roles"]["control"]["call_id"] == "fc-control"

    terminal["lifecycle"]["coordinator_call_id"] = "fc-other-coordinator"
    with pytest.raises(ValueError, match="evidence"):
        rehearsal.validate_terminal(spec, payloads, dispatch, observed_exit, terminal)
    terminal["lifecycle"]["coordinator_call_id"] = "fc-coordinator"

    roles["treatment"]["raw_worker_result"]["started_at_utc"] = "2026-10-05T00:00:14+00:00"
    roles["treatment"]["raw_worker_result"]["finished_at_utc"] = "2026-10-05T00:00:18+00:00"
    summary = rehearsal.validate_terminal(spec, payloads, dispatch, observed_exit, terminal)
    assert summary["arms_overlapped"] is False


def test_modal_function_options_apply_exact_cpu_and_memory_bounds() -> None:
    options = rehearsal._modal_options(rehearsal.STAGE_LIMITS)

    assert options["cpu"] == (1, 1)
    assert options["memory"] == (1024, 1024)


def test_coordinator_passes_its_modal_call_id_to_production(monkeypatch) -> None:
    from experiments import targeted_training_coordinator as coordinator

    call_ids = []
    modal_stub = SimpleNamespace(
        current_function_call_id=lambda: "fc-coordinator",
        Volume=SimpleNamespace(from_name=lambda *_args, **_kwargs: object()),
        Dict=SimpleNamespace(from_name=lambda *_args, **_kwargs: object()),
    )
    monkeypatch.setitem(sys.modules, "modal", modal_stub)
    monkeypatch.setattr(
        coordinator,
        "VolumeEvidence",
        lambda *_args: SimpleNamespace(persist=lambda _terminal: None),
    )
    monkeypatch.setattr(
        coordinator,
        "execute_coordinator",
        lambda *_args, **kwargs: call_ids.append(kwargs["coordinator_call_id"]),
    )
    spec = rehearsal.build_spec(
        source_commit="a" * 40,
        source_manifest={"experiments/targeted_training_coordinator.py": "b" * 64},
    )

    rehearsal._modal_coordinator_entry(spec, rehearsal.build_payloads(spec), "ap-test", {})

    assert call_ids == ["fc-coordinator"]


def test_stage_result_validator_rejects_non_synthetic_fields() -> None:
    spec = rehearsal.build_spec(
        source_commit="a" * 40,
        source_manifest={"experiments/targeted_training_coordinator.py": "b" * 64},
    )
    payload = rehearsal.build_payloads(spec)["unchanged"]
    result = {
        **payload,
        "started_at_utc": "2026-10-05T00:00:00+00:00",
        "finished_at_utc": "2026-10-05T00:00:10+00:00",
    }

    validated = rehearsal.validate_stage_result(payload, result)
    assert validated["status"] == "passed"
    assert validated["role"] == "unchanged"
    try:
        rehearsal.validate_stage_result(payload, result | {"model": "forbidden"})
    except ValueError as error:
        assert "synthetic" in str(error)
    else:
        raise AssertionError("non-synthetic stage output must be rejected")


def test_dispatch_clearance_binds_ci_and_root_review_to_exact_spec() -> None:
    spec = rehearsal.build_spec(
        source_commit="a" * 40,
        source_manifest={"experiments/targeted_training_coordinator.py": "b" * 64},
    )
    checked_at = datetime.now(UTC).isoformat()
    digest = "c" * 64
    ci_proof = {
        "head_sha": spec["source_commit"],
        "status": "completed",
        "conclusion": "success",
        "workflow_url": "https://github.com/example/reflex/actions/runs/123",
        "evidence_sha256": digest,
    }
    clearance = {
        "reviewed_by": "root",
        "reviewed_at_utc": checked_at,
        "plan_sha256": spec["plan_sha256"],
        "source_commit": spec["source_commit"],
        "source_manifest_sha256": spec["source_manifest_sha256"],
        "account": {
            "profile": spec["profile"],
            "workspace": spec["workspace"],
            "environment": spec["environment"],
            "checked_at_utc": checked_at,
            "evidence_sha256": digest,
        },
        "budget": {
            "status": "cleared",
            "checked_at_utc": checked_at,
            "estimated_max_usd": "1.25",
            "approved_max_usd": "2.50",
            "evidence_sha256": digest,
        },
    }

    rehearsal.validate_clearance(spec, ci_proof, clearance)
    with pytest.raises(ValueError, match="CI|commit"):
        rehearsal.validate_clearance(spec, ci_proof | {"head_sha": "d" * 40}, clearance)


def test_attempt_marker_prevents_dispatch_resubmission(tmp_path) -> None:
    spec = rehearsal.build_spec(
        source_commit="a" * 40,
        source_manifest={"experiments/targeted_training_coordinator.py": "b" * 64},
    )
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()

    rehearsal.begin_attempt(output_dir, spec)
    marker = (output_dir / "attempt.json").read_text()
    with pytest.raises(ValueError, match="attempt|resubmit"):
        rehearsal.begin_attempt(output_dir, spec)

    assert (output_dir / "attempt.json").read_text() == marker


def test_dispatch_spec_validation_rejects_changed_live_source_before_modal_use(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    source_files = {
        "experiments/targeted_training_core.py": "core\n",
        "experiments/modal_targeted_training.py": "launcher\n",
        "experiments/targeted_training_coordinator.py": "coordinator\n",
        "experiments/modal_targeted_recovery_rehearsal.py": "rehearsal\n",
        "uv.lock": 'version = 1\n\n[[package]]\nname = "pydantic"\nversion = "2.13.5"\n',
    }
    for relative, content in source_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source_manifest = {
        relative: hashlib.sha256(content.encode()).hexdigest()
        for relative, content in source_files.items()
    }

    def current_manifest(_root):
        return {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in source_manifest
            if relative != "experiments/modal_targeted_recovery_rehearsal.py"
        }

    monkeypatch.setattr(
        "experiments.targeted_training_core.targeted_source_manifest", current_manifest
    )
    monkeypatch.setattr(
        "experiments.modal_targeted_training._committed_source",
        lambda _root, _manifest: "a" * 40,
    )
    spec_dir = tmp_path / "frozen"
    spec_dir.mkdir()
    (spec_dir / "spec.json").write_text(json.dumps(rehearsal.build_spec("a" * 40, source_manifest)))
    (root / "experiments/targeted_training_coordinator.py").write_text("tampered\n")

    with pytest.raises(ValueError, match="source"):
        rehearsal.verify_spec(spec_dir, root)
