from __future__ import annotations

import ast
import asyncio
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from experiments import modal_runtime_rule_study as host
from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_launch as launch_api
from experiments import runtime_rule_study_receipts as receipts_api
from experiments import runtime_rule_study_results as results_api

_RAW_TAG = "$runtime_rule_study_raw"
_STUDY_ID = "host-test-001"
_PLAN_SHA = "a" * 64
_COMMIT = "b" * 40


class _Call:
    def __init__(self, role: str, fixture: _RuntimeFixture) -> None:
        self.role = role
        self.object_id = f"fc-{role}"
        self.fixture = fixture
        self.get = SimpleNamespace(aio=self._get)
        self.cancel = SimpleNamespace(aio=self._cancel)
        self.cancel_count = 0

    async def _get(self, *, timeout: float) -> object:
        self.fixture.events.append(("get", self.role, timeout))
        if self.role in self.fixture.wait_errors:
            raise RuntimeError(f"transport-{self.role}")
        if self.role in self.fixture.wait_for_cancel:
            await self.fixture.cancel_signal.wait()
        return self.fixture.raw_results[self.role]

    async def _cancel(self, *, terminate_containers: bool) -> object:
        assert terminate_containers is True
        self.cancel_count += 1
        self.fixture.events.append(("cancel", self.role, terminate_containers))
        self.fixture.cancel_signal.set()
        return {"cancelled": self.role}


class _Function:
    def __init__(self, role: str, fixture: _RuntimeFixture) -> None:
        self.role = role
        self.fixture = fixture
        self.spawn = SimpleNamespace(aio=self._spawn)

    async def _spawn(self, payload: object, expected_digest: str) -> _Call:
        assert isinstance(payload, Mapping)
        assert expected_digest == self.fixture.plan.role_payload_sha256[self.role]
        self.fixture.events.append(("spawn", self.role, expected_digest))
        if self.role in self.fixture.spawn_errors:
            raise RuntimeError(f"spawn-{self.role}")
        call = _Call(self.role, self.fixture)
        self.fixture.calls[self.role] = call
        return call


class _Run:
    def __init__(self, fixture: _RuntimeFixture) -> None:
        self.fixture = fixture

    def aio(self, *, name: str, environment_name: str) -> _Run:
        self.fixture.events.append(("run", name, environment_name))
        assert environment_name == "main"
        return self

    async def __aenter__(self) -> _FakeApp:
        self.fixture.app.app_id = self.fixture.visible_app_id
        self.fixture.events.append(("app_enter", self.fixture.app_id))
        if self.fixture.startup_error is not None:
            raise self.fixture.startup_error
        return self.fixture.app

    async def __aexit__(self, *_exc: object) -> None:
        self.fixture.events.append(("app_exit", self.fixture.app_id))
        if self.fixture.exit_error is not None:
            raise self.fixture.exit_error


class _FakeApp:
    def __init__(self, fixture: _RuntimeFixture) -> None:
        self.app_id: str | None = None
        self.run = _Run(fixture)


class _FakeProvider:
    def __init__(self, fixture: _RuntimeFixture) -> None:
        self.fixture = fixture

    def github_probe(self, source_commit: str) -> dict[str, object]:
        self.fixture.events.append(("github_probe", source_commit))
        return {"commit": source_commit}

    def make_modal_probe(self, rates: Mapping[str, object], study_id: str) -> Any:
        self.fixture.events.append(("make_modal_probe", study_id, dict(rates)))
        return lambda: {"study_id": study_id}

    def build_app(self, plan: Any) -> tuple[_FakeApp, dict[str, _Function]]:
        assert plan.study_id == _STUDY_ID
        self.fixture.events.append(("build_app", plan.study_id))
        functions = {role: _Function(role, self.fixture) for role in contracts.ROLES}
        return self.fixture.app, functions

    async def verify_teardown(self, app_id: str) -> dict[str, object]:
        self.fixture.events.append(("teardown", app_id))
        if self.fixture.teardown_error is not None:
            raise self.fixture.teardown_error
        return {"schema_version": 1, "app_id": app_id, "verified": self.fixture.teardown_verified}

    async def wait_for_idle_app(self, app_id: str) -> dict[str, object]:
        self.fixture.events.append(("wait_idle", app_id))
        return {
            "schema_version": 1,
            "app_id": app_id,
            "verified": self.fixture.idle_verified,
            "observations": [{"attempt": 1, "n_tasks": 0 if self.fixture.idle_verified else 1}],
            "errors": [],
        }


class _RuntimeFixture:
    def __init__(self, root: Path) -> None:
        self.events: list[tuple[object, ...]] = []
        self.cancel_signal = asyncio.Event()
        self.calls: dict[str, _Call] = {}
        self.wait_errors: set[str] = set()
        self.wait_for_cancel: set[str] = set()
        self.spawn_errors: set[str] = set()
        self.raw_results: dict[str, object] = {
            role: {"status": "passed", "role": role} for role in contracts.ROLES
        }
        self.app_id = "ap-host-test"
        self.visible_app_id: str | None = self.app_id
        self.idle_verified = True
        self.startup_error: BaseException | None = None
        self.exit_error: BaseException | None = None
        self.teardown_error: BaseException | None = None
        self.teardown_verified = True
        self.app = _FakeApp(self)
        self.provider = _FakeProvider(self)
        self.plan_dir = root / "plan"
        payload_dir = self.plan_dir / "payloads"
        payload_dir.mkdir(parents=True)
        self.plan_file = self.plan_dir / "launch-plan.json"
        self.plan_file.write_text("{}", encoding="utf-8")
        payload_paths: dict[str, Path] = {}
        payload_file_digests: dict[str, str] = {}
        payload_digests: dict[str, str] = {}
        for index, role in enumerate(contracts.ROLES):
            payload = {
                "role": role,
                "worker_settings": {
                    "timeout_seconds": 900 if index == 0 else 3_600,
                    "startup_timeout_seconds": 300,
                },
            }
            path = payload_dir / f"{role}.json"
            payload_bytes = contracts.canonical_json(payload)
            path.write_bytes(payload_bytes)
            payload_paths[role] = path
            payload_file_digests[role] = hashlib.sha256(payload_bytes).hexdigest()
            payload_digests[role] = str(index + 1) * 64
        self.plan = SimpleNamespace(
            plan_path=self.plan_file,
            plan_sha256=_PLAN_SHA,
            study_id=_STUDY_ID,
            source_commit=_COMMIT,
            role_payload_paths=payload_paths,
            role_payload_file_sha256=payload_file_digests,
            role_payload_sha256=payload_digests,
        )
        self.attempts = root / "attempts"
        self.attempts.mkdir()
        self.receipts = root / "receipts"
        self.receipts.mkdir()
        self.rates_file = root / "rates.json"
        rates: dict[str, object] = {
            "rates_usd": {"gpu_hour_a10g": "1.10", "cpu_hour": "0.05", "mem_gib_hour": "0.01"},
            "rate_evidence": {
                "source_url": "https://modal.com/pricing",
                "checked_at_utc": "2026-10-05T12:00:00Z",
            },
            "scope": "root-verified fake rates",
            "source_html_sha256": "d" * 64,
        }
        self.rates_bytes = json.dumps(rates, sort_keys=True).encode()
        self.rates_file.write_bytes(self.rates_bytes)
        self.rates_sha = hashlib.sha256(self.rates_bytes).hexdigest()

    @property
    def receipt_path(self) -> Path:
        return self.receipts / f"{_STUDY_ID}.host-receipt.json"


@pytest.fixture
def setup_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _RuntimeFixture:
    fixture = _RuntimeFixture(tmp_path)

    def load_plan(path: str | Path, digest: str) -> Any:
        assert Path(path) == fixture.plan_file
        assert digest == _PLAN_SHA
        return fixture.plan

    def authorize(*args: object, **kwargs: object) -> Any:
        assert args[0] == fixture.plan_file
        assert kwargs["source_root"] == tmp_path
        assert kwargs["attempts_dir"] == fixture.attempts
        github_probe = cast(Any, kwargs["github_probe"])
        modal_probe = cast(Any, kwargs["modal_probe"])
        github_probe(_COMMIT)
        modal_probe()
        marker = fixture.attempts / f"{_STUDY_ID}.launch-attempt.json"
        marker.write_text("authorized", encoding="utf-8", errors="strict")
        fixture.events.append(("authorized", marker))
        return SimpleNamespace(
            study_id=_STUDY_ID,
            plan_sha256=_PLAN_SHA,
            attempt_id="attempt-001",
            authorized_at_utc="2026-10-05T12:00:00Z",
        )

    def validate_result(
        raw: object, *, payload: Mapping[str, object], expected_payload_sha256: object
    ) -> dict[str, object]:
        role = cast(str, payload["role"])
        assert expected_payload_sha256 == fixture.plan.role_payload_sha256[role]
        if not isinstance(raw, Mapping) or any(type(key) is not str for key in raw):
            raise ValueError("malformed worker return")
        if raw.get("role") != role or raw.get("status") not in {"passed", "failed"}:
            raise ValueError("worker identity/status mismatch")
        return dict(raw)

    monkeypatch.setattr(launch_api, "load_launch_plan", load_plan)
    monkeypatch.setattr(launch_api, "authorize_launch", authorize)
    monkeypatch.setattr(host, "_provider_module", lambda: fixture.provider)
    monkeypatch.setattr(results_api, "validate_result", validate_result)
    return fixture


def _execute(fixture: _RuntimeFixture) -> host.ExecutionReceipt:
    return host.execute_launch(
        fixture.plan_file,
        _PLAN_SHA,
        project_root=fixture.plan_dir.parent,
        attempts_dir=fixture.attempts,
        receipt_path=fixture.receipt_path,
        rates_path=fixture.rates_file,
        expected_rates_sha256=fixture.rates_sha,
    )


def _saved_receipt(fixture: _RuntimeFixture) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(fixture.receipt_path.read_text(encoding="utf-8")))


def test_success_spawns_unchanged_first_and_both_arms_before_waiting(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    result = _execute(fixture)

    assert result.status == "passed"
    assert [event[1] for event in fixture.events if event[0] == "spawn"] == list(contracts.ROLES)
    spawn_positions = [index for index, event in enumerate(fixture.events) if event[0] == "spawn"]
    get_positions = [index for index, event in enumerate(fixture.events) if event[0] == "get"]
    training_spawn_positions = [
        index
        for index, event in enumerate(fixture.events)
        if event[0] == "spawn" and event[1] != contracts.ROLE_UNCHANGED
    ]
    training_get_positions = [
        index
        for index, event in enumerate(fixture.events)
        if event[0] == "get" and event[1] != contracts.ROLE_UNCHANGED
    ]
    assert max(training_spawn_positions) < min(training_get_positions)
    assert min(get_positions) < min(training_spawn_positions)
    assert max(spawn_positions) > min(get_positions)
    unchanged_get = max(
        index for index, event in enumerate(fixture.events) if event[:2] == ("get", "unchanged")
    )
    idle_check = fixture.events.index(("wait_idle", fixture.app_id))
    assert unchanged_get < idle_check < min(training_spawn_positions)
    assert all(event[-1] == "main" for event in fixture.events if event[0] == "run")
    assert fixture.events.index(
        ("authorized", fixture.attempts / f"{_STUDY_ID}.launch-attempt.json")
    ) < fixture.events.index(("build_app", _STUDY_ID))
    saved = _saved_receipt(fixture)
    assert saved["status"] == "passed"
    assert saved["unchanged_idle"]["verified"] is True
    assert saved["unchanged_idle"]["evidence"]["observations"][0]["n_tasks"] == 0
    assert [saved["calls"][role]["state"] for role in contracts.ROLES] == ["passed"] * 3
    for role in contracts.ROLES:
        assert saved["calls"][role]["raw_result"] == fixture.raw_results[role]


def test_payload_drift_after_authorization_prevents_app_construction(
    setup_runtime: _RuntimeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = setup_runtime
    authorize = launch_api.authorize_launch

    def authorize_then_drift(*args: object, **kwargs: object) -> Any:
        authorization = authorize(*args, **kwargs)
        fixture.plan.role_payload_paths[contracts.ROLE_UNCHANGED].write_text(
            '{"role":"changed-after-authorization"}', encoding="utf-8"
        )
        return authorization

    monkeypatch.setattr(launch_api, "authorize_launch", authorize_then_drift)

    result = _execute(fixture)

    assert result.status == "failed"
    assert not any(event[0] == "build_app" for event in fixture.events)
    assert not fixture.calls
    saved = _saved_receipt(fixture)
    assert saved["status"] == "failed"
    assert saved["failures"][0]["message"] == (
        "unchanged authenticated payload changed before app construction"
    )


def test_authorization_identity_mismatch_writes_failure_receipt(
    setup_runtime: _RuntimeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = setup_runtime
    authorize = launch_api.authorize_launch

    def authorize_with_wrong_identity(*args: object, **kwargs: object) -> Any:
        authorization = authorize(*args, **kwargs)
        return SimpleNamespace(
            study_id="different-study",
            plan_sha256=authorization.plan_sha256,
            attempt_id=authorization.attempt_id,
            authorized_at_utc=authorization.authorized_at_utc,
        )

    monkeypatch.setattr(launch_api, "authorize_launch", authorize_with_wrong_identity)

    result = _execute(fixture)

    assert result.status == "failed"
    assert not any(event[0] == "build_app" for event in fixture.events)
    saved = _saved_receipt(fixture)
    assert saved["status"] == "failed"
    assert saved["study_id"] == "different-study"
    assert saved["failures"][0]["stage"] == "lifecycle"


def test_unchanged_failure_does_not_start_either_training_role(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.raw_results[contracts.ROLE_UNCHANGED] = {"role": "unchanged", "status": "failed"}

    result = _execute(fixture)

    assert result.status == "failed"
    assert [event[1] for event in fixture.events if event[0] == "spawn"] == ["unchanged"]
    assert fixture.events[-1] == ("teardown", fixture.app_id)
    assert _saved_receipt(fixture)["all_results_passed"] is False


def test_failed_wait_cancels_its_handle_and_retains_completed_sibling(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.wait_errors.add("continued_practice")
    fixture.wait_for_cancel.add("runtime_mix")

    result = _execute(fixture)

    assert result.status == "failed"
    assert fixture.calls["continued_practice"].cancel_count == 1
    assert fixture.calls["runtime_mix"].cancel_count == 1
    saved = _saved_receipt(fixture)
    assert saved["calls"]["continued_practice"]["state"] == "wait_failed"
    assert saved["calls"]["continued_practice"]["cancel"]["status"] == "returned"
    assert saved["calls"]["runtime_mix"]["raw_result"] == fixture.raw_results["runtime_mix"]
    assert saved["calls"]["runtime_mix"]["state"] == "passed"


def test_spawn_failure_cancels_prior_handle_without_retrying(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.spawn_errors.add("runtime_mix")

    result = _execute(fixture)

    assert result.status == "failed"
    assert fixture.calls["continued_practice"].cancel_count == 1
    assert "runtime_mix" not in fixture.calls
    assert [event[1] for event in fixture.events if event[0] == "spawn"] == [
        "unchanged",
        "continued_practice",
        "runtime_mix",
    ]


def test_worker_failure_cancels_unresolved_sibling(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.raw_results["continued_practice"] = {
        "role": "continued_practice",
        "status": "failed",
    }
    fixture.wait_for_cancel.add("runtime_mix")

    result = _execute(fixture)

    assert result.status == "failed"
    assert fixture.calls["continued_practice"].cancel_count == 0
    assert fixture.calls["runtime_mix"].cancel_count == 1
    saved = _saved_receipt(fixture)
    assert saved["calls"]["continued_practice"]["state"] == "worker_failed"
    assert saved["calls"]["runtime_mix"]["raw_result"] == fixture.raw_results["runtime_mix"]


def test_unverified_idle_blocks_training_and_is_kept_in_receipt(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.idle_verified = False

    result = _execute(fixture)

    assert result.status == "failed"
    assert [event[1] for event in fixture.events if event[0] == "spawn"] == ["unchanged"]
    saved = _saved_receipt(fixture)
    assert saved["unchanged_idle"]["verified"] is False
    assert saved["unchanged_idle"]["evidence"]["observations"][0]["n_tasks"] == 1


def test_missing_app_id_blocks_training_and_cannot_claim_teardown(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.visible_app_id = None

    result = _execute(fixture)

    assert result.status == "failed"
    assert [event[1] for event in fixture.events if event[0] == "spawn"] == ["unchanged"]
    assert not any(event[0] in {"wait_idle", "teardown"} for event in fixture.events)
    saved = _saved_receipt(fixture)
    assert saved["app_id"] is None
    assert saved["unchanged_idle"]["state"] == "unverified"
    assert saved["teardown"]["verified"] is False


def test_malformed_mapping_and_nonfinite_float_are_persisted_with_tags(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.raw_results["unchanged"] = {1: "integer-key", "non_finite": float("nan")}

    result = _execute(fixture)

    assert result.status == "failed"
    raw = _saved_receipt(fixture)["calls"]["unchanged"]["raw_result"]
    encoded = raw[_RAW_TAG]
    assert encoded["kind"] == "mapping"
    assert encoded["entries"][0] == [1, "integer-key"]
    assert encoded["entries"][1][1] == {_RAW_TAG: {"kind": "float", "value": "nan"}}


def test_unknown_raw_value_is_only_a_bounded_sanitized_summary(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime

    class SecretLike:
        def __repr__(self) -> str:
            return "token=do-not-persist " + "x" * 2_000

    fixture.raw_results["unchanged"] = SecretLike()
    result = _execute(fixture)

    assert result.status == "failed"
    summary = _saved_receipt(fixture)["calls"]["unchanged"]["raw_result"][_RAW_TAG]
    assert summary["kind"] == "summary"
    assert summary["type"].endswith("SecretLike")
    assert "do-not-persist" not in summary["repr"]
    assert len(summary["repr"]) <= receipts_api.RAW_SUMMARY_LIMIT
    assert "not preserved" in _saved_receipt(fixture)["raw_value_encoding"]["unknown_custom_value"]


@pytest.mark.parametrize(
    "case",
    [
        "large_integer",
        "surrogate_value",
        "surrogate_key",
        "exception_surrogate",
        "unknown_repr_surrogate",
    ],
)
def test_lifecycle_persists_receipts_for_malformed_observations(
    setup_runtime: _RuntimeFixture, case: str
) -> None:
    fixture = setup_runtime
    surrogate = chr(0xD800)

    class SurrogateRepr:
        def __repr__(self) -> str:
            return surrogate

    if case == "large_integer":
        malformed: object = 10**5000
        fixture.raw_results[contracts.ROLE_UNCHANGED] = malformed
    elif case == "surrogate_value":
        malformed = surrogate
        fixture.raw_results[contracts.ROLE_UNCHANGED] = malformed
    elif case == "surrogate_key":
        malformed = {surrogate: "value"}
        fixture.raw_results[contracts.ROLE_UNCHANGED] = malformed
    elif case == "exception_surrogate":
        fixture.startup_error = RuntimeError(surrogate)
        malformed = surrogate
    else:
        malformed = SurrogateRepr()
        fixture.raw_results[contracts.ROLE_UNCHANGED] = malformed

    result = _execute(fixture)

    assert result.status == "failed"
    assert fixture.events[-1] == ("teardown", fixture.app_id)
    saved = _saved_receipt(fixture)
    assert saved["status"] == "failed"
    encoding = cast(dict[str, str], saved["raw_value_encoding"])
    assert encoding["malformed_integer"] == (
        "values exceeding JSON decimal digit limits use signed hexadecimal tags"
    )
    assert encoding["malformed_string"] == (
        "non-UTF-8-encodable strings use tagged Python ASCII literal representations"
    )
    assert encoding["malformed_mapping"] == (
        "tagged entries preserve non-string or non-UTF-8-encodable string keys"
    )
    calls = cast(dict[str, dict[str, object]], saved["calls"])
    unchanged_record = calls[contracts.ROLE_UNCHANGED]
    if case == "large_integer":
        raw = cast(dict[str, object], unchanged_record["raw_result"])
        encoded_integer = cast(dict[str, object], raw[_RAW_TAG])
        assert encoded_integer["kind"] == "integer"
        assert int(cast(str, encoded_integer["value"]), 16) == malformed
    elif case == "surrogate_value":
        raw = cast(dict[str, object], unchanged_record["raw_result"])
        encoded_string = cast(dict[str, object], raw[_RAW_TAG])
        assert encoded_string["kind"] == "string"
        assert ast.literal_eval(cast(str, encoded_string["value"])) == malformed
    elif case == "surrogate_key":
        raw = cast(dict[str, object], unchanged_record["raw_result"])
        encoded_mapping = cast(dict[str, object], raw[_RAW_TAG])
        assert encoded_mapping["kind"] == "mapping"
        entries = cast(list[list[object]], encoded_mapping["entries"])
        encoded_key = cast(dict[str, object], entries[0][0])
        key_tag = cast(dict[str, object], encoded_key[_RAW_TAG])
        assert ast.literal_eval(cast(str, key_tag["value"])) == surrogate
        assert entries[0][1] == "value"
    elif case == "exception_surrogate":
        failures = cast(list[dict[str, object]], saved["failures"])
        encoded_message = cast(dict[str, object], failures[0]["message"])
        message_tag = cast(dict[str, object], encoded_message[_RAW_TAG])
        assert ast.literal_eval(cast(str, message_tag["value"])) == malformed
    else:
        raw = cast(dict[str, object], unchanged_record["raw_result"])
        summary = cast(dict[str, object], raw[_RAW_TAG])
        encoded_repr = cast(dict[str, object], summary["repr"])
        repr_tag = cast(dict[str, object], encoded_repr[_RAW_TAG])
        assert repr_tag["kind"] == "string"
        assert ast.literal_eval(cast(str, repr_tag["value"])) == surrogate


def test_two_waiter_cleanup_outcomes_remain_attached_to_their_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTask:
        def __init__(self, role: str, order_hash: int) -> None:
            self.role = role
            self.order_hash = order_hash
            self.was_cancelled = False

        def __hash__(self) -> int:
            return self.order_hash

        def done(self) -> bool:
            return self.was_cancelled

        def cancel(self) -> bool:
            self.was_cancelled = True
            return True

        def result(self) -> object:
            return None

    late = FakeTask("late", 1)
    early = FakeTask("early", 0)
    waiters: dict[str, Any] = {"late": late, "early": early}
    records: dict[str, dict[str, object]] = {"late": {}, "early": {}}

    async def fake_wait(tasks: set[object], *, timeout: float) -> tuple[set[object], set[object]]:
        assert timeout == host._LOCAL_WAITER_SETTLE_SECONDS
        return set(), set(tasks)

    async def fake_gather(*tasks: object, return_exceptions: bool) -> list[BaseException]:
        assert return_exceptions is True
        return [RuntimeError(f"cleanup-{cast(Any, task).role}") for task in tasks]

    async def exercise() -> None:
        with monkeypatch.context() as scoped:
            scoped.setattr(asyncio, "wait", fake_wait)
            scoped.setattr(asyncio, "gather", fake_gather)
            await host._settle_waiters(
                cast(Any, waiters),
                {},
                SimpleNamespace(role_payload_sha256={}),
                records,
                [],
            )

    asyncio.run(exercise())

    late_error = cast(dict[str, object], records["late"]["waiter_cleanup"])
    early_error = cast(dict[str, object], records["early"]["waiter_cleanup"])
    assert late_error["message"] == "cleanup-late"
    assert early_error["message"] == "cleanup-early"


def test_hanging_remote_cancel_is_bounded_and_does_not_skip_other_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCancel:
        def __init__(self, role: str) -> None:
            self.role = role

        async def aio(self, *, terminate_containers: bool) -> dict[str, str]:
            assert terminate_containers is True
            started.append(self.role)
            if self.role == "stuck":
                await asyncio.Event().wait()
            return {"cancelled": self.role}

    started: list[str] = []
    handles = {role: SimpleNamespace(cancel=FakeCancel(role)) for role in ("stuck", "next")}
    records: dict[str, dict[str, object]] = {role: {} for role in handles}

    async def exercise() -> None:
        await asyncio.wait_for(host._cancel_remote(handles, set(handles), records), timeout=0.2)

    monkeypatch.setattr(host, "_REMOTE_CANCEL_TIMEOUT_SECONDS", 0.005, raising=False)
    asyncio.run(exercise())

    assert set(started) == set(handles)
    stuck_cancel = cast(dict[str, object], records["stuck"]["cancel"])
    stuck_error = cast(dict[str, str], stuck_cancel["error"])
    next_cancel = cast(dict[str, object], records["next"]["cancel"])
    assert stuck_cancel["status"] == "failed"
    assert stuck_error["type"] == "TimeoutError"
    assert next_cancel["status"] == "returned"


def test_startup_failure_with_assigned_app_id_still_verifies_teardown(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.startup_error = RuntimeError("startup failed")

    result = _execute(fixture)

    assert result.status == "failed"
    assert fixture.events[-1] == ("teardown", fixture.app_id)
    assert _saved_receipt(fixture)["app_id"] == fixture.app_id


def test_app_exit_failure_still_verifies_teardown_after_all_results(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.exit_error = RuntimeError("app exit failed")

    result = _execute(fixture)

    assert result.status == "failed"
    saved = _saved_receipt(fixture)
    assert [saved["calls"][role]["state"] for role in contracts.ROLES] == ["passed"] * 3
    assert saved["all_results_passed"] is False
    assert fixture.events[-1] == ("teardown", fixture.app_id)


def test_teardown_failure_prevents_success_and_is_preserved(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.teardown_error = RuntimeError("teardown unavailable")

    result = _execute(fixture)

    assert result.status == "failed"
    saved = _saved_receipt(fixture)
    assert saved["all_results_passed"] is True
    assert saved["teardown"]["verified"] is False
    assert saved["teardown"]["error"]["type"] == "RuntimeError"


def test_cancelled_lifecycle_writes_receipt_and_tears_down_before_reraising(
    setup_runtime: _RuntimeFixture,
) -> None:
    fixture = setup_runtime
    fixture.startup_error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _execute(fixture)

    assert fixture.receipt_path.exists()
    assert fixture.events[-1] == ("teardown", fixture.app_id)
    assert _saved_receipt(fixture)["status"] == "failed"


@pytest.mark.parametrize("path_kind", ["receipt", "attempt"])
def test_evidence_paths_inside_plan_tree_are_rejected_before_authorization(
    setup_runtime: _RuntimeFixture, path_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = setup_runtime
    if path_kind == "receipt":
        receipt_path = fixture.plan_dir / "launch-plan.json"
        attempts_dir = fixture.attempts
    else:
        receipt_path = fixture.receipt_path
        attempts_dir = fixture.plan_dir
    authorized = False

    def fail_authorize(*_args: object, **_kwargs: object) -> None:
        nonlocal authorized
        authorized = True

    monkeypatch.setattr(launch_api, "authorize_launch", fail_authorize)
    with pytest.raises(ValueError, match="outside the exact launch plan tree"):
        host.execute_launch(
            fixture.plan_file,
            _PLAN_SHA,
            project_root=fixture.plan_dir.parent,
            attempts_dir=attempts_dir,
            receipt_path=receipt_path,
            rates_path=fixture.rates_file,
            expected_rates_sha256=fixture.rates_sha,
        )
    assert authorized is False


def test_existing_receipt_path_is_rejected_before_attempt_consumption(
    setup_runtime: _RuntimeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = setup_runtime
    fixture.receipt_path.write_text("previous evidence", encoding="utf-8")
    authorized = False

    def fail_authorize(*_args: object, **_kwargs: object) -> None:
        nonlocal authorized
        authorized = True

    monkeypatch.setattr(launch_api, "authorize_launch", fail_authorize)
    with pytest.raises(ValueError, match="receipt path already exists"):
        _execute(fixture)
    assert authorized is False


def test_existing_attempt_marker_is_rejected_before_authorization(
    setup_runtime: _RuntimeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = setup_runtime
    marker = fixture.attempts / f"{_STUDY_ID}.launch-attempt.json"
    marker.write_text("previous attempt", encoding="utf-8")
    authorized = False

    def fail_authorize(*_args: object, **_kwargs: object) -> None:
        nonlocal authorized
        authorized = True

    monkeypatch.setattr(launch_api, "authorize_launch", fail_authorize)
    with pytest.raises(ValueError, match="attempt path already exists"):
        _execute(fixture)
    assert authorized is False
    assert not any(event[0] == "build_app" for event in fixture.events)


@pytest.mark.parametrize("path_kind", ["lexical", "resolved"])
def test_attempt_marker_and_receipt_collision_fails_before_provider_calls(
    setup_runtime: _RuntimeFixture, path_kind: str
) -> None:
    fixture = setup_runtime
    attempt_path = fixture.attempts / f"{_STUDY_ID}.launch-attempt.json"
    if path_kind == "lexical":
        receipt_path = attempt_path
    else:
        alias_root = fixture.plan_dir.parent / "attempts-alias"
        alias_root.symlink_to(fixture.plan_dir.parent, target_is_directory=True)
        receipt_path = alias_root / "attempts" / attempt_path.name
        assert receipt_path != attempt_path
        assert receipt_path.parent.resolve() == attempt_path.parent.resolve()
        assert not receipt_path.parent.is_symlink()

    with pytest.raises(ValueError, match="attempt and receipt paths must differ"):
        host.execute_launch(
            fixture.plan_file,
            _PLAN_SHA,
            project_root=fixture.plan_dir.parent,
            attempts_dir=fixture.attempts,
            receipt_path=receipt_path,
            rates_path=fixture.rates_file,
            expected_rates_sha256=fixture.rates_sha,
        )

    assert fixture.events == []
    assert not attempt_path.exists()
    assert not fixture.calls


def test_rates_hash_mismatch_fails_before_provider_import_or_authorization(
    setup_runtime: _RuntimeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = setup_runtime
    provider_loaded = False
    authorized = False

    def no_provider() -> Any:
        nonlocal provider_loaded
        provider_loaded = True
        return fixture.provider

    def no_authorize(*_args: object, **_kwargs: object) -> None:
        nonlocal authorized
        authorized = True

    monkeypatch.setattr(host, "_provider_module", no_provider)
    monkeypatch.setattr(launch_api, "authorize_launch", no_authorize)
    with pytest.raises(ValueError, match="caller-supplied SHA-256"):
        host.execute_launch(
            fixture.plan_file,
            _PLAN_SHA,
            project_root=fixture.plan_dir.parent,
            attempts_dir=fixture.attempts,
            receipt_path=fixture.receipt_path,
            rates_path=fixture.rates_file,
            expected_rates_sha256="e" * 64,
        )
    assert provider_loaded is False
    assert authorized is False


def test_receipt_writer_never_overwrites_existing_evidence(tmp_path: Path) -> None:
    receipt = tmp_path / "immutable.json"
    receipt.write_text("earlier", encoding="utf-8")

    with pytest.raises(FileExistsError):
        receipts_api.write_receipt(receipt, {"new": True})

    assert receipt.read_text(encoding="utf-8") == "earlier"


def test_plan_mode_never_imports_or_calls_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan_receipt = SimpleNamespace(plan_path=tmp_path / "launch-plan.json", plan_sha256=_PLAN_SHA)
    monkeypatch.setattr(launch_api, "create_launch_plan", lambda *_args: plan_receipt)
    monkeypatch.setattr(
        host,
        "_provider_module",
        lambda: pytest.fail("plan-only path loaded providers"),
    )

    status = host.main(
        [
            "--project-root",
            str(tmp_path),
            "--study-id",
            _STUDY_ID,
            "--source-commit",
            _COMMIT,
            "--output-dir",
            str(tmp_path / "new-plan"),
        ]
    )

    assert status == 0
    assert json.loads(capsys.readouterr().out) == {
        "plan_path": str(plan_receipt.plan_path),
        "plan_sha256": _PLAN_SHA,
    }
