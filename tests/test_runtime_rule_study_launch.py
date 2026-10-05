from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_launch as launch

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
COMMIT = "a" * 40
SOURCE_HASH = hashlib.sha256(b"fixture source\n").hexdigest()
BUNDLE_HASH = "c" * 64


def _stamp(value: datetime = NOW) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _payload(study_id: str, nonce: str, role: str, source_map: Mapping[str, str]) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "study_id": study_id,
        "nonce": nonce,
        "role": role,
        "source_commit": COMMIT,
        "source_file_sha256": dict(source_map),
        "bundle_sha256": BUNDLE_HASH,
        "worker_settings": contracts.worker_settings(role),
    }
    return {**unsigned, "payload_sha256": contracts.json_sha256(unsigned)}


def _github_evidence(
    *, now: datetime = NOW, commit: str = COMMIT, signature: bool = True
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checked_at_utc": _stamp(now),
        "repository": "1337mus/reflex",
        "visibility": "private",
        "branch": "main",
        "branch_sha": commit,
        "commit_sha": commit,
        "signature": {"verified": signature, "status": "verified"},
        "ci": {
            "workflow_path": ".github/workflows/ci.yml",
            "head_sha": commit,
            "status": "completed",
            "conclusion": "success",
            "run_id": 12345,
        },
    }


def _modal_evidence(
    *, now: datetime = NOW, metered_cost: str = "100.00", apps: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checked_at_utc": _stamp(now),
        "sdk_version": "1.6.1",
        "profile": "reflex-personal",
        "workspace": "rajath-61258",
        "billing_cycle": "2026-10",
        "billing": {
            "metered_cost": metered_cost,
            "billed_cost": "93.00",
            "adjustments": {"credit": "-2.00"},
        },
        "rates_usd": {
            "gpu_hour_a10g": "1.10",
            "cpu_hour": "0.05",
            "mem_gib_hour": "0.01",
        },
        "rate_evidence": {
            "source_url": "https://modal.com/pricing",
            "checked_at_utc": _stamp(now - timedelta(hours=1)),
        },
        "apps": [] if apps is None else apps,
    }


@pytest.fixture
def prepared_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[launch.PlanReceipt, Path, Path]:
    source_root = tmp_path / "repo"
    source_file = source_root / "experiments" / "fake.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("fixture source\n", encoding="utf-8")
    monkeypatch.setattr(launch.source_files, "SOURCE_PATHS", ("experiments/fake.py",))

    def freeze(root: str | Path, commit: str, destination: str | Path) -> dict[str, str]:
        assert commit == COMMIT
        root_path = Path(root)
        destination_path = Path(destination)
        destination_path.mkdir()
        target = destination_path / "experiments" / "fake.py"
        target.parent.mkdir()
        target.write_bytes((root_path / "experiments" / "fake.py").read_bytes())
        return {"experiments/fake.py": hashlib.sha256(target.read_bytes()).hexdigest()}

    monkeypatch.setattr(launch.source_files, "freeze_sources", freeze)
    monkeypatch.setattr(
        launch.source_files,
        "measure_sources",
        lambda root: {
            "experiments/fake.py": hashlib.sha256(
                (Path(root) / "experiments" / "fake.py").read_bytes()
            ).hexdigest()
        },
    )
    monkeypatch.setattr(launch, "_build_bundle", lambda _root: {"bundle_sha256": BUNDLE_HASH})

    def build_payload(
        _bundle: object,
        *,
        expected_bundle_sha256: str,
        study_id: str,
        nonce: str,
        role: str,
        source_commit: str,
        source_file_sha256: Mapping[str, str],
    ) -> dict[str, Any]:
        assert expected_bundle_sha256 == BUNDLE_HASH
        assert source_commit == COMMIT
        return _payload(study_id, nonce, role, source_file_sha256)

    def validate_payload(
        value: object, *, expected_payload_sha256: str | None = None
    ) -> dict[str, Any]:
        assert isinstance(value, Mapping)
        result = dict(value)
        digest = result["payload_sha256"]
        unsigned = {key: item for key, item in result.items() if key != "payload_sha256"}
        if digest != contracts.json_sha256(unsigned):
            raise ValueError("payload digest mismatch")
        if expected_payload_sha256 is not None and digest != expected_payload_sha256:
            raise ValueError("trusted payload digest mismatch")
        return result

    monkeypatch.setattr(launch.payloads, "build_worker_payload", build_payload)
    monkeypatch.setattr(launch.payloads, "validate_payload", validate_payload)
    out = tmp_path / "study-plan"
    receipt = launch.create_launch_plan(
        source_root,
        "launch-test-1",
        COMMIT,
        out,
        now_utc=NOW,
    )
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    return receipt, source_root, attempts


def test_plan_is_canonical_authenticated_and_never_grants_launch_authority(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
) -> None:
    receipt, _source_root, _attempts = prepared_plan
    plan = launch.load_launch_plan(receipt.plan_path, receipt.plan_sha256)

    assert plan.launch_authority is False
    assert plan.source_snapshot_path == receipt.plan_path.parent / "source"
    assert set(plan.role_payload_paths) == set(contracts.ROLES)
    assert plan.role_payload_sha256 == receipt.payload_sha256
    saved = json.loads(receipt.plan_path.read_text(encoding="utf-8"))
    assert saved["launch_authority"] is False
    assert saved["source_snapshot_path"] == "source"
    assert saved["source_file_sha256"] == {"experiments/fake.py": SOURCE_HASH}


def test_plan_hash_mismatch_refuses_to_load(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
) -> None:
    receipt, _source_root, _attempts = prepared_plan

    with pytest.raises(ValueError, match="expected SHA-256"):
        launch.load_launch_plan(receipt.plan_path, "f" * 64)


def test_payload_file_tampering_is_detected_before_provider_calls(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
) -> None:
    receipt, source_root, attempts = prepared_plan
    payload_path = receipt.plan_path.parent / "payloads" / "unchanged.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["study_id"] = "changed-study"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    calls: list[str] = []

    with pytest.raises(ValueError, match="payload file"):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: calls.append("github") or _github_evidence(),
            modal_probe=lambda: calls.append("modal") or _modal_evidence(),
            now_utc=NOW,
        )

    assert calls == []
    assert list(attempts.iterdir()) == []


def test_local_source_drift_fails_without_probing_or_creating_marker(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
) -> None:
    receipt, source_root, attempts = prepared_plan
    source_file = source_root / "experiments" / "fake.py"
    source_file.write_text("drifted source\n", encoding="utf-8")
    calls: list[str] = []

    with pytest.raises(ValueError, match="local source bytes"):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: calls.append("github") or _github_evidence(),
            modal_probe=lambda: calls.append("modal") or _modal_evidence(),
            now_utc=NOW,
        )

    assert calls == []
    assert list(attempts.iterdir()) == []


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        (_github_evidence(commit="d" * 40), "selected private main commit"),
        (_github_evidence(signature=False), "signature is not verified"),
        (_github_evidence(now=NOW - timedelta(minutes=6)), "stale or too far"),
    ],
)
def test_github_authority_failures_do_not_create_attempt_marker(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
    evidence: dict[str, Any],
    match: str,
) -> None:
    receipt, source_root, attempts = prepared_plan

    with pytest.raises(ValueError, match=match):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: evidence,
            modal_probe=_modal_evidence,
            now_utc=NOW,
        )

    assert list(attempts.iterdir()) == []


@pytest.mark.parametrize(
    ("mutations", "match"),
    [
        ({"sdk_version": "1.6.0"}, "SDK version"),
        ({"profile": "default"}, "profile or workspace"),
        ({"checked_at_utc": _stamp(NOW - timedelta(minutes=6))}, "stale or too far"),
        ({"metered_cost": "291.00"}, "cannot cover projected exposure and reserve"),
    ],
)
def test_modal_identity_freshness_and_budget_fail_closed(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
    mutations: dict[str, str],
    match: str,
) -> None:
    receipt, source_root, attempts = prepared_plan
    modal = _modal_evidence()
    if "metered_cost" in mutations:
        modal["billing"]["metered_cost"] = mutations["metered_cost"]
    else:
        modal.update(mutations)

    with pytest.raises(ValueError, match=match):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: _github_evidence(),
            modal_probe=lambda: modal,
            now_utc=NOW,
        )

    assert list(attempts.iterdir()) == []


def test_active_study_app_with_no_running_tasks_blocks_and_is_preserved_in_probe(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
) -> None:
    receipt, source_root, attempts = prepared_plan
    modal = _modal_evidence(
        apps=[
            {
                "name": "runtime-rule-study-another-id",
                "app_id": "ap-active",
                "state": "APP_STATE_RUNNING",
                "n_running_tasks": 0,
            }
        ]
    )

    with pytest.raises(ValueError, match="another runtime-rule study app"):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: _github_evidence(),
            modal_probe=lambda: modal,
            now_utc=NOW,
        )

    assert list(attempts.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("metered_cost", True, "finite decimal string"),
        ("gpu_hour_a10g", "NaN", "finite decimal string"),
        ("app_tasks", True, "strict non-negative integer"),
    ],
)
def test_modal_evidence_rejects_boolean_aliases_and_nonfinite_costs(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
    field: str,
    value: object,
    match: str,
) -> None:
    receipt, source_root, attempts = prepared_plan
    modal = _modal_evidence()
    if field == "metered_cost":
        modal["billing"]["metered_cost"] = value
    elif field == "gpu_hour_a10g":
        modal["rates_usd"][field] = value
    else:
        modal["apps"] = [
            {
                "name": "runtime-rule-study-queued",
                "app_id": "ap-queued",
                "state": "APP_STATE_QUEUED",
                "n_running_tasks": value,
            }
        ]

    with pytest.raises(ValueError, match=match):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: _github_evidence(),
            modal_probe=lambda: modal,
            now_utc=NOW,
        )

    assert list(attempts.iterdir()) == []


def test_stopped_modal_app_still_marks_the_exact_study_id_as_used(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
) -> None:
    receipt, source_root, attempts = prepared_plan
    modal = _modal_evidence(
        apps=[
            {
                "name": "runtime-rule-study-launch-test-1",
                "app_id": "ap-stopped",
                "state": "APP_STATE_STOPPED",
                "n_running_tasks": 0,
            }
        ]
    )

    with pytest.raises(ValueError, match="study ID was already used"):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: _github_evidence(),
            modal_probe=lambda: modal,
            now_utc=NOW,
        )

    assert list(attempts.iterdir()) == []


def test_success_authorizes_once_and_preserves_provider_evidence(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path],
) -> None:
    receipt, source_root, attempts = prepared_plan
    authorization = launch.authorize_launch(
        receipt.plan_path,
        receipt.plan_sha256,
        source_root=source_root,
        attempts_dir=attempts,
        github_probe=lambda _commit: _github_evidence(),
        modal_probe=_modal_evidence,
        now_utc=NOW,
    )
    marker = json.loads(authorization.marker_path.read_text(encoding="utf-8"))

    assert marker["launch_authority"] is True
    assert marker["github_evidence"]["ci"]["head_sha"] == COMMIT
    assert marker["modal_evidence"]["sdk_version"] == "1.6.1"
    assert marker["modal_evidence"]["billing"]["adjustments"]["credit"] == "-2.00"
    assert Decimal(marker["budget"]["projected_exposure_usd"]) == Decimal("3.41")
    assert Decimal(marker["budget"]["projected_exposure_usd"]) <= Decimal("10")
    assert launch.load_launch_plan(receipt.plan_path, receipt.plan_sha256).launch_authority is False

    with pytest.raises(ValueError, match="already has a launch-attempt marker"):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: pytest.fail("duplicate ID must fail before probing"),
            modal_probe=lambda: pytest.fail("duplicate ID must fail before probing"),
            now_utc=NOW,
        )


def test_probe_clocks_are_read_after_callbacks_and_accept_utc_offset_form(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, source_root, attempts = prepared_plan
    clock_values = iter(
        [
            NOW + timedelta(seconds=60),
            NOW + timedelta(seconds=61),
            NOW + timedelta(seconds=62),
        ]
    )

    class AdvancingDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            value = next(clock_values)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(launch.authority, "datetime", AdvancingDateTime)
    github = _github_evidence(now=NOW + timedelta(seconds=45))
    github["checked_at_utc"] = (NOW + timedelta(seconds=45)).isoformat()
    modal = _modal_evidence(now=NOW + timedelta(seconds=60))
    modal["checked_at_utc"] = (NOW + timedelta(seconds=60)).isoformat()
    modal["rate_evidence"]["checked_at_utc"] = (NOW - timedelta(hours=1)).isoformat()

    authorization = launch.authorize_launch(
        receipt.plan_path,
        receipt.plan_sha256,
        source_root=source_root,
        attempts_dir=attempts,
        github_probe=lambda _commit: github,
        modal_probe=lambda: modal,
    )

    assert authorization.authorized_at_utc == _stamp(NOW + timedelta(seconds=62))


def test_github_evidence_is_rechecked_after_a_slow_modal_probe(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, source_root, attempts = prepared_plan
    clock_values = iter(
        [
            NOW + timedelta(seconds=30),
            NOW + timedelta(minutes=6, seconds=5),
            NOW + timedelta(minutes=6, seconds=10),
        ]
    )

    class AdvancingDateTime(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            value = next(clock_values)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(launch.authority, "datetime", AdvancingDateTime)
    github = _github_evidence(now=NOW)
    modal = _modal_evidence(now=NOW + timedelta(minutes=6))

    with pytest.raises(ValueError, match="GitHub checked_at_utc is stale"):
        launch.authorize_launch(
            receipt.plan_path,
            receipt.plan_sha256,
            source_root=source_root,
            attempts_dir=attempts,
            github_probe=lambda _commit: github,
            modal_probe=lambda: modal,
        )

    assert list(attempts.iterdir()) == []


def test_symlink_payload_is_rejected_even_when_it_points_to_valid_json(
    prepared_plan: tuple[launch.PlanReceipt, Path, Path], tmp_path: Path
) -> None:
    receipt, _source_root, _attempts = prepared_plan
    payload = receipt.plan_path.parent / "payloads" / "unchanged.json"
    original = payload.read_bytes()
    outside = tmp_path / "valid-payload.json"
    outside.write_bytes(original)
    payload.unlink()
    payload.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|symlink component"):
        launch.load_launch_plan(receipt.plan_path, receipt.plan_sha256)
