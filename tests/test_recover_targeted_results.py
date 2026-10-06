from __future__ import annotations

from datetime import UTC

import pytest

from experiments import recover_targeted_results as recovery


def test_strict_json_loads_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        recovery.strict_json_loads(b'{"role":"control","role":"treatment"}')


def test_strict_json_loads_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        recovery.strict_json_loads(b'{"score":NaN}')


def test_resolve_regular_file_rejects_path_traversal(tmp_path) -> None:
    with pytest.raises(ValueError, match="path"):
        recovery._resolve_regular_file(tmp_path, "../outside.json")


def _canonical_hash(value: object) -> str:
    import hashlib
    import json

    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def _event_fixture():
    from copy import deepcopy
    from datetime import datetime, timedelta

    roles = ("unchanged", "control", "treatment")
    identities = {
        "run_id": "toy-r2",
        "source_commit": "a" * 40,
        "plan_sha256": "",
        "app_id": "ap-toy",
        "coordinator_call_id": "fc-coordinator",
    }
    calls = {role: f"fc-{role}" for role in roles}
    payloads = {
        role: {"role": role, "payload_sha256": _canonical_hash({"role": role})} for role in roles
    }
    raw = {
        role: {
            "status": "passed",
            "experiment_id": "targeted-reasoning-v1",
            "run_id": "toy-r2",
            "role": role,
            "payload_sha256": payloads[role]["payload_sha256"],
        }
        for role in roles
    }
    wrappers = {
        role: {"raw_worker_result": raw[role]}
        | ({"validated_result": raw[role]} if role == "unchanged" else {})
        for role in roles
    }
    plan = {
        "experiment_id": "targeted-reasoning-v1",
        "run_id": "toy-r2",
        "role_payload_sha256": {role: payloads[role]["payload_sha256"] for role in roles},
    }
    plan["plan_sha256"] = _canonical_hash(plan)
    identities["plan_sha256"] = plan["plan_sha256"]
    life = {
        "status": "failed",
        "app_id": "ap-toy",
        "coordinator_call_id": "fc-coordinator",
        "failure": {"type": "KeyboardInterrupt"},
    }
    receipt = {
        "schema_version": 1,
        "status": "failed",
        "experiment_id": "targeted-reasoning-v1",
        "run_id": "toy-r2",
        "plan": plan,
        "role_payloads": payloads,
        "roles": wrappers,
        "lifecycle": life
        | {
            "teardown": {
                "app_id": "ap-toy",
                "verified": True,
                "observations": [{"state": "APP_STATE_STOPPED", "n_tasks": 0}],
            }
        },
    }
    events = []
    start = datetime(2026, 10, 5, tzinfo=UTC)

    def emit(kind: str, evidence: object) -> None:
        at = start + timedelta(seconds=len(events))
        events.append({"kind": kind, "at_utc": at.isoformat(), "evidence": evidence})

    emit("identity", identities)
    emit("app", {"app_id": "ap-toy"})
    emit("calls", {"unchanged": calls["unchanged"]})
    emit("roles", {"unchanged": {"raw_worker_result": raw["unchanged"]}})
    emit("roles", {"unchanged": deepcopy(wrappers["unchanged"])})
    emit("calls", {"unchanged": calls["unchanged"], "control": calls["control"]})
    emit("calls", calls)
    emit(
        "roles",
        {
            "unchanged": deepcopy(wrappers["unchanged"]),
            "treatment": {"raw_worker_result": raw["treatment"]},
        },
    )
    emit("emergency_teardown", {"deferred_until_terminal_commit": True})
    emit("cancellation", {"reason": "KeyboardInterrupt", "errors": {}})
    emit("roles", deepcopy(wrappers))
    emit(
        "terminal",
        {key: value for key, value in receipt.items() if key != "lifecycle"} | {"lifecycle": life},
    )
    pins = {
        "identities": identities | {"experiment_id": "targeted-reasoning-v1"},
        "calls": calls,
        "role_payload_sha256": {role: payloads[role]["payload_sha256"] for role in roles},
        "raw_worker_result_sha256": {role: _canonical_hash(raw[role]) for role in roles},
        "stored_validated_result_sha256": {
            role: _canonical_hash(raw[role]) if role == "unchanged" else None for role in roles
        },
    }
    return events, receipt, pins, calls


def test_event_sequence_reconstructs_three_stable_role_calls() -> None:
    events, receipt, pins, calls = _event_fixture()

    result = recovery._validate_event_sequence(events, pins, receipt)

    assert result["calls"] == calls
    assert result["validated_roles"] == ("unchanged",)


def test_event_sequence_rejects_call_snapshot_that_drops_a_role() -> None:
    from copy import deepcopy

    events, receipt, pins, _ = _event_fixture()
    changed = deepcopy(events)
    del changed[5]["evidence"]["unchanged"]

    with pytest.raises(ValueError, match="call.*regress"):
        recovery._validate_event_sequence(changed, pins, receipt)


def test_event_jsonl_rejects_an_unterminated_final_record(tmp_path) -> None:
    event_file = tmp_path / "events.jsonl"
    event_file.write_bytes(b'{"kind":"identity","at_utc":"2026-10-05T00:00:00Z","evidence":{}}')

    with pytest.raises(ValueError, match="event.*line|truncated"):
        list(recovery._strict_jsonl_records(event_file))


def test_event_jsonl_rejects_blank_records_and_nonfinite_numbers(tmp_path) -> None:
    event_file = tmp_path / "events.jsonl"
    event_file.write_bytes(b'{"kind":"identity"}\n\n')
    with pytest.raises(ValueError, match="blank"):
        list(recovery._strict_jsonl_records(event_file))
    event_file.write_bytes(b'{"kind":"identity","bad":NaN}\n')
    with pytest.raises(ValueError, match="malformed"):
        list(recovery._strict_jsonl_records(event_file))


def test_event_sequence_rejects_result_drift_in_a_later_snapshot() -> None:
    from copy import deepcopy

    events, receipt, pins, _ = _event_fixture()
    changed = deepcopy(events)
    changed[7]["evidence"]["treatment"]["raw_worker_result"]["role"] = "control"

    with pytest.raises(ValueError, match="raw result"):
        recovery._validate_event_sequence(changed, pins, receipt)


def test_event_sequence_rejects_terminal_role_drift() -> None:
    from copy import deepcopy

    events, receipt, pins, _ = _event_fixture()
    changed = deepcopy(events)
    changed[-1]["evidence"]["roles"]["control"]["raw_worker_result"]["status"] = "failed"

    with pytest.raises(ValueError, match="terminal.*receipt"):
        recovery._validate_event_sequence(changed, pins, receipt)


def test_shutdown_rejects_boolean_as_zero_task_count(tmp_path) -> None:
    import hashlib
    import json

    app = {
        "app_id": recovery.IDENTITIES["app_id"],
        "state": "APP_STATE_STOPPED",
        "n_running_tasks": 0,
    }
    raw = json.dumps(
        {"checked_at_utc": "2026-10-06T00:00:00Z", "apps": [app]},
        sort_keys=True,
    ).encode()
    (tmp_path / "shutdown.json").write_bytes(raw)
    pins = {
        "shutdown": {
            "source_metadata_path": "shutdown.json",
            "source_metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "observed": app,
        }
    }
    receipt = {
        "lifecycle": {
            "teardown": {
                "app_id": app["app_id"],
                "verified": True,
                "observations": [{"state": "APP_STATE_STOPPED", "n_tasks": False}],
            }
        }
    }

    with pytest.raises(ValueError, match="receipt teardown"):
        recovery._validate_shutdown(tmp_path, pins, receipt, "2026-10-05T23:00:00Z")


def test_frozen_receipt_gate_rejects_projection_without_mutating_receipt(monkeypatch) -> None:
    from copy import deepcopy

    from experiments import analyze_targeted_training as analysis

    receipt = {
        "status": "failed",
        "lifecycle": {"status": "failed", "failure": {"type": "KeyboardInterrupt"}},
        "roles": {role: {"raw_worker_result": {"role": role}} for role in recovery.ROLES},
    }
    original = deepcopy(receipt)
    observed = {}

    def reject_projection(view, *, expected_plan_sha256, plan):
        observed["view"] = view
        assert view["status"] == "passed"
        assert view["lifecycle"]["status"] == "passed"
        assert set(view["lifecycle"]["calls"]) == set(recovery.ROLES)
        assert all("validated_result" in item for item in view["roles"].values())
        raise ValueError("frozen validator rejected projection")

    monkeypatch.setattr(analysis, "validate_complete_receipt", reject_projection)
    with pytest.raises(ValueError, match="frozen validator"):
        recovery._validate_compatibility_projection(receipt, recovery.CALLS, {})
    assert receipt == original
    assert observed["view"] is not receipt


def test_clearance_failure_precedes_admission_and_host_loader(tmp_path, monkeypatch) -> None:
    import hashlib
    import sys
    from types import SimpleNamespace

    amendment = tmp_path / "amendment.json"
    clearance = tmp_path / "clearance.json"
    amendment.write_text("{}\n")
    clearance.write_text("{}\n")
    calls = {"admission": 0, "loader": 0}

    def reject_clearance(*_args):
        raise ValueError("approval missing")

    def forbidden_admission(*_args):
        calls["admission"] += 1
        pytest.fail("admission continued after clearance failure")

    monkeypatch.setattr(recovery, "_validate_clearance", reject_clearance)
    monkeypatch.setattr(recovery, "_validate_admission", forbidden_admission)
    monkeypatch.setitem(
        sys.modules,
        "experiments.targeted_training_data",
        SimpleNamespace(load_inputs=lambda _root: calls.__setitem__("loader", calls["loader"] + 1)),
    )
    code = recovery.main(
        [
            "--root",
            str(tmp_path),
            "--amendment",
            amendment.name,
            "--expected-amendment-sha256",
            hashlib.sha256(amendment.read_bytes()).hexdigest(),
            "--clearance",
            clearance.name,
            "--output",
            "out",
            "--analyze",
        ]
    )

    assert code == 2
    assert calls == {"admission": 0, "loader": 0}
    assert not (tmp_path / "out").exists()


def _clearance_fixture(tmp_path):
    import hashlib
    import json

    source = "b" * 40
    checks = [
        "clearance_mismatch_blocks_gold_loader",
        "missing_approval_blocks_gold_loader",
        "identity_mismatch_blocks_gold_loader",
        "call_provenance_mismatch_blocks_gold_loader",
        "valid_provenance_reaches_sentinel_control",
    ]
    required = [
        "experiments/recover_targeted_results.py",
        "tests/test_recover_targeted_results.py",
        "docs/targeted-r2-recovery-amendment.md",
        "docs/verification/targeted-r2-recovery-amendment.json",
        ".context/targeted-r2-recovery-tdd.md",
        ".context/recount_targeted_recovery.py",
        ".context/targeted-r2-independent-recovery-self-test.json",
    ]
    hashes = {}
    head_blobs = {}
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("self-test.json"):
            checker_hash = hashlib.sha256(b"checker").hexdigest()
            body = {
                "status": "passed",
                "scope": "synthetic_only",
                "real_inputs_opened": False,
                "score_values_emitted": False,
                "real_gold_loader_calls": 0,
                "rejected_cases_loader_calls": 0,
                "synthetic_sentinel_calls": 1,
                "checker_sha256": checker_hash,
                "checks": checks,
            }
            path.write_text(json.dumps(body, sort_keys=True) + "\n")
        elif relative.endswith("recount_targeted_recovery.py"):
            path.write_bytes(b"checker")
        else:
            path.write_text("reviewed fixture\n")
        body = path.read_bytes()
        hashes[relative] = hashlib.sha256(body).hexdigest()
        if relative in recovery.REVIEWED_TRACKED_FILES:
            head_blobs[relative] = body
    checker_hash = hashes[".context/recount_targeted_recovery.py"]
    pins = {"evidence_file_sha256": {".context/recount_targeted_recovery.py": checker_hash}}
    clearance = {
        "schema_version": 1,
        "amendment_sha256": "a" * 64,
        "source_commit": source,
        "ci": {"status": "passed", "source_commit": source},
        "root_review": {
            "status": "approved",
            "source_commit": source,
            "files_sha256": hashes,
        },
        "independent_review": {
            "status": "approved",
            "source_commit": source,
            "files_sha256": hashes,
        },
        "independent_checker": {
            "status": "passed",
            "source_commit": source,
            "sha256": checker_hash,
        },
    }
    return clearance, pins, source, head_blobs


def _stub_git_signature(monkeypatch, source: str, head_blobs) -> list[list[str]]:
    from types import SimpleNamespace

    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[3] == "rev-parse":
            return SimpleNamespace(stdout=source)
        if command[3] == "verify-commit":
            return SimpleNamespace(stdout="")
        assert command[3] == "show"
        return SimpleNamespace(stdout=head_blobs[command[4].removeprefix("HEAD:")])

    monkeypatch.setattr(recovery.subprocess, "run", run)
    return commands


def test_analysis_gate_validates_real_clearance_shape_then_loads_sentinel(
    tmp_path, monkeypatch
) -> None:
    clearance, pins, source, head_blobs = _clearance_fixture(tmp_path)
    commands = _stub_git_signature(monkeypatch, source, head_blobs)
    monkeypatch.setattr(recovery, "_validate_admission", lambda *_args: {"admitted": True})
    loaded = []

    admission, sentinel = recovery._analysis_gate_then_load(
        tmp_path,
        pins,
        clearance["amendment_sha256"],
        clearance,
        lambda root: loaded.append(root) or "sentinel-gold",
    )

    assert admission == {"admitted": True}
    assert sentinel == "sentinel-gold"
    assert loaded == [tmp_path]
    assert [command[3] for command in commands] == [
        "rev-parse",
        "verify-commit",
        "show",
        "show",
        "show",
        "show",
    ]


def test_missing_same_source_ci_blocks_before_admission_or_loader(tmp_path, monkeypatch) -> None:
    clearance, pins, source, head_blobs = _clearance_fixture(tmp_path)
    clearance["ci"]["status"] = "missing"
    _stub_git_signature(monkeypatch, source, head_blobs)
    monkeypatch.setattr(
        recovery,
        "_validate_admission",
        lambda *_args: pytest.fail("admission ran without exact-source CI"),
    )
    loaded = []

    with pytest.raises(ValueError, match="same-source CI"):
        recovery._analysis_gate_then_load(
            tmp_path,
            pins,
            clearance["amendment_sha256"],
            clearance,
            lambda root: loaded.append(root),
        )
    assert loaded == []


def test_failed_admission_blocks_host_loader_after_clearance(tmp_path, monkeypatch) -> None:
    clearance, pins, source, head_blobs = _clearance_fixture(tmp_path)
    _stub_git_signature(monkeypatch, source, head_blobs)

    def reject_admission(*_args):
        raise ValueError("admission rejected")

    monkeypatch.setattr(recovery, "_validate_admission", reject_admission)
    loaded = []
    with pytest.raises(ValueError, match="admission rejected"):
        recovery._analysis_gate_then_load(
            tmp_path,
            pins,
            clearance["amendment_sha256"],
            clearance,
            lambda root: loaded.append(root),
        )
    assert loaded == []


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("root_review", "review bindings"),
        ("independent_review", "review bindings"),
        ("wrong_source", "signed source binding"),
        ("changed_reviewed_file", "signed HEAD"),
        ("signature_failure", "git"),
    ],
)
def test_clearance_rejections_precede_admission_and_loader(
    tmp_path, monkeypatch, case, message
) -> None:
    import hashlib
    import subprocess
    from types import SimpleNamespace

    clearance, pins, source, head_blobs = _clearance_fixture(tmp_path)
    if case in {"root_review", "independent_review"}:
        clearance[case]["status"] = "missing"
        _stub_git_signature(monkeypatch, source, head_blobs)
    elif case == "wrong_source":
        clearance["source_commit"] = "c" * 40
        _stub_git_signature(monkeypatch, source, head_blobs)
    elif case == "changed_reviewed_file":
        relative = "docs/targeted-r2-recovery-amendment.md"
        changed = tmp_path / relative
        changed.write_text("reviewed working tree differs from signed HEAD\n")
        changed_hash = hashlib.sha256(changed.read_bytes()).hexdigest()
        clearance["root_review"]["files_sha256"][relative] = changed_hash
        clearance["independent_review"]["files_sha256"][relative] = changed_hash
        _stub_git_signature(monkeypatch, source, head_blobs)
    else:

        def fail_signature(command, **_kwargs):
            if command[3] == "rev-parse":
                return SimpleNamespace(stdout=source)
            raise subprocess.CalledProcessError(1, command)

        monkeypatch.setattr(recovery.subprocess, "run", fail_signature)

    monkeypatch.setattr(
        recovery,
        "_validate_admission",
        lambda *_args: pytest.fail("admission ran after clearance rejection"),
    )
    loaded = []
    exception = subprocess.CalledProcessError if case == "signature_failure" else ValueError
    with pytest.raises(exception, match=message):
        recovery._analysis_gate_then_load(
            tmp_path,
            pins,
            clearance["amendment_sha256"],
            clearance,
            lambda root: loaded.append(root),
        )
    assert loaded == []


def test_pinned_source_or_data_bytes_must_match(tmp_path) -> None:
    import hashlib

    pinned = tmp_path / "record.jsonl"
    pinned.write_bytes(b"safe fixture bytes\n")
    digest = hashlib.sha256(pinned.read_bytes()).hexdigest()
    recovery._checked_hash_map(tmp_path, {"record.jsonl": digest})
    pinned.write_bytes(b"changed fixture bytes\n")
    with pytest.raises(ValueError, match="digest differs"):
        recovery._checked_hash_map(tmp_path, {"record.jsonl": digest})
