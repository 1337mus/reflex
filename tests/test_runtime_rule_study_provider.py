from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_provider as provider
from experiments import runtime_rule_study_sources as source_files


class _FakeAPI:
    APP_STATE_EPHEMERAL = "APP_STATE_EPHEMERAL"
    APP_STATE_STOPPED = "APP_STATE_STOPPED"
    APP_STOP_SOURCE_CLI = "APP_STOP_SOURCE_CLI"

    class AppState:
        @staticmethod
        def Name(state: str) -> str:
            return state

    @staticmethod
    def AppListRequest(*, environment_name: str) -> SimpleNamespace:
        return SimpleNamespace(environment_name=environment_name)

    @staticmethod
    def AppGetLifecycleRequest(*, app_id: str) -> SimpleNamespace:
        return SimpleNamespace(app_id=app_id)

    @staticmethod
    def TaskListRequest(*, environment_name: str, app_id: str) -> SimpleNamespace:
        return SimpleNamespace(environment_name=environment_name, app_id=app_id)

    @staticmethod
    def AppStopRequest(*, app_id: str, source: str) -> SimpleNamespace:
        return SimpleNamespace(app_id=app_id, source=source)


def _fake_modal_sdk(
    *, apps: list[SimpleNamespace], workspace_name: str = "rajath-61258"
) -> tuple[SimpleNamespace, dict[str, object]]:
    events: dict[str, object] = {"profiles": [], "cycles": []}

    class FakeStub:
        async def AppList(self, request: SimpleNamespace) -> SimpleNamespace:
            assert request.environment_name == "main"
            events["app_list_request"] = request
            return SimpleNamespace(apps=apps)

    class FakeClient:
        _stub = FakeStub()

        @classmethod
        async def from_env(cls) -> FakeClient:
            events["client_created"] = True
            return cls()

    class FakeBilling:
        async def summary(self, *, cycle: str) -> SimpleNamespace:
            cast_cycles = events["cycles"]
            assert isinstance(cast_cycles, list)
            cast_cycles.append(cycle)
            return SimpleNamespace(
                metered_cost=Decimal("23.40"),
                billed_cost=Decimal("22.15"),
                adjustments={"credit": Decimal("-1.25")},
            )

    class FakeWorkspace:
        name = workspace_name
        billing = FakeBilling()

        @classmethod
        def from_context(cls, *, client: FakeClient) -> FakeWorkspace:
            events["workspace_client"] = client
            return cls()

        async def hydrate(self) -> FakeWorkspace:
            events["workspace_hydrated"] = True
            return self

    def set_profile(profile: str) -> None:
        profiles = events["profiles"]
        assert isinstance(profiles, list)
        profiles.append(profile)

    sdk = SimpleNamespace(
        config=SimpleNamespace(_set_profile=set_profile),
        client=SimpleNamespace(_Client=FakeClient),
        workspace=SimpleNamespace(_Workspace=FakeWorkspace),
        modal=SimpleNamespace(),
    )
    return sdk, events


def _install_fake_modal_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider, "_load_modal_api", lambda: _FakeAPI)


def test_modal_protobuf_loader_uses_the_separate_modal_proto_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules: list[str] = []

    def fake_import(name: str) -> object:
        modules.append(name)
        return _FakeAPI

    monkeypatch.setattr(importlib, "import_module", fake_import)

    assert provider._load_modal_api() is _FakeAPI
    assert modules == ["modal_proto.api_pb2"]


def _app(name: str, app_id: str, state: str, running_tasks: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        app_id=app_id,
        state=state,
        n_running_tasks=running_tasks,
    )


def _root_rate_snapshot() -> dict[str, object]:
    return {
        "rates_usd": {
            "gpu_hour_a10g": "1.101600",
            "cpu_hour": "0.0471600",
            "mem_gib_hour": "0.00799200",
        },
        "rate_evidence": {
            "source_url": "https://modal.com/pricing",
            "checked_at_utc": "2026-10-05T12:21:46.084981+00:00",
        },
        "scope": "root-only note",
        "source_html_sha256": "d" * 64,
    }


def _github_responses(
    source_commit: str,
    *,
    repository: dict[str, object] | None = None,
    branch: dict[str, object] | None = None,
    commit: dict[str, object] | None = None,
    runs: list[dict[str, object]] | None = None,
) -> tuple[object, ...]:
    return (
        repository or {"full_name": "1337mus/reflex", "visibility": "private", "private": True},
        branch or {"name": "main", "commit": {"sha": source_commit}},
        commit
        or {
            "sha": source_commit,
            "commit": {"verification": {"verified": True, "reason": "valid"}},
        },
        runs
        or [
            {
                "databaseId": 7412,
                "headSha": source_commit,
                "status": "completed",
                "conclusion": "success",
                "createdAt": "2026-10-05T12:00:00Z",
            }
        ],
    )


def _patch_gh_json(monkeypatch: pytest.MonkeyPatch, responses: tuple[object, ...]) -> None:
    replies = iter(responses)
    monkeypatch.setattr(provider, "_gh_json", lambda _args: next(replies))


def test_github_probe_returns_exact_evidence_for_the_requested_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "a" * 40
    responses = iter(
        (
            {
                "full_name": "1337mus/reflex",
                "visibility": "private",
                "private": True,
            },
            {"name": "main", "commit": {"sha": source_commit}},
            {
                "sha": source_commit,
                "commit": {
                    "verification": {"verified": True, "reason": "valid"},
                },
            },
            [
                {
                    "databaseId": 7412,
                    "headSha": source_commit,
                    "status": "completed",
                    "conclusion": "success",
                    "createdAt": "2026-10-05T12:00:00Z",
                }
            ],
        )
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(next(responses)), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = provider.github_probe(source_commit)

    assert result.keys() == {
        "schema_version",
        "checked_at_utc",
        "repository",
        "visibility",
        "branch",
        "branch_sha",
        "commit_sha",
        "signature",
        "ci",
    }
    assert result == {
        "schema_version": 1,
        "checked_at_utc": result["checked_at_utc"],
        "repository": "1337mus/reflex",
        "visibility": "private",
        "branch": "main",
        "branch_sha": source_commit,
        "commit_sha": source_commit,
        "signature": {"verified": True, "status": "verified"},
        "ci": {
            "workflow_path": ".github/workflows/ci.yml",
            "head_sha": source_commit,
            "status": "completed",
            "conclusion": "success",
            "run_id": 7412,
        },
    }
    checked_at = datetime.fromisoformat(str(result["checked_at_utc"]).replace("Z", "+00:00"))
    assert checked_at.utcoffset() == timedelta(0)
    assert [call for call, _kwargs in calls] == [
        ["gh", "api", "repos/1337mus/reflex"],
        ["gh", "api", "repos/1337mus/reflex/branches/main"],
        ["gh", "api", f"repos/1337mus/reflex/commits/{source_commit}"],
        [
            "gh",
            "run",
            "list",
            "--repo",
            "1337mus/reflex",
            "--workflow",
            ".github/workflows/ci.yml",
            "--commit",
            source_commit,
            "--json",
            "databaseId,headSha,status,conclusion,createdAt",
            "--limit",
            "100",
        ],
    ]
    assert all(kwargs["timeout"] == 30 for _call, kwargs in calls)


def test_github_probe_rejects_an_invalid_commit_without_calling_gh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        provider.github_probe("a" * 39)

    assert calls == []


@pytest.mark.parametrize(
    ("repo", "branch", "commit", "runs"),
    [
        (
            {"full_name": "other/reflex", "visibility": "private", "private": True},
            None,
            None,
            None,
        ),
        (
            {"full_name": "1337mus/reflex", "visibility": "public", "private": False},
            None,
            None,
            None,
        ),
        (
            None,
            {"name": "main", "commit": {"sha": "b" * 40}},
            None,
            None,
        ),
        (
            None,
            None,
            {
                "sha": "a" * 40,
                "commit": {
                    "verification": {"verified": True, "reason": "unsigned"},
                    "status": "verified",
                },
            },
            None,
        ),
        (
            None,
            None,
            None,
            [
                {
                    "databaseId": 7412,
                    "headSha": "b" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "createdAt": "2026-10-05T12:00:00Z",
                }
            ],
        ),
        (
            None,
            None,
            None,
            [
                {
                    "databaseId": 7411,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "success",
                    "createdAt": "2026-10-05T11:00:00Z",
                },
                {
                    "databaseId": 7412,
                    "headSha": "a" * 40,
                    "status": "completed",
                    "conclusion": "failure",
                    "createdAt": "2026-10-05T12:00:00Z",
                },
            ],
        ),
    ],
)
def test_github_probe_rejects_repo_branch_signature_or_latest_ci_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    repo: dict[str, object] | None,
    branch: dict[str, object] | None,
    commit: dict[str, object] | None,
    runs: list[dict[str, object]] | None,
) -> None:
    source_commit = "a" * 40
    _patch_gh_json(
        monkeypatch,
        _github_responses(
            source_commit,
            repository=repo,
            branch=branch,
            commit=commit,
            runs=runs,
        ),
    )

    with pytest.raises(ValueError):
        provider.github_probe(source_commit)


def test_modal_probe_uses_injected_rates_and_keeps_zero_task_study_apps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_id = "provider-test-001"
    apps = [
        _app("runtime-rule-study-prior", "ap-prior", _FakeAPI.APP_STATE_EPHEMERAL, 0),
        _app(
            f"runtime-rule-study-{study_id}",
            "ap-current",
            _FakeAPI.APP_STATE_STOPPED,
            0,
        ),
        _app("other-active", "ap-active", _FakeAPI.APP_STATE_EPHEMERAL, 0),
        _app("other-stopped", "ap-idle", _FakeAPI.APP_STATE_STOPPED, 0),
        _app("other-with-tasks", "ap-taskful", _FakeAPI.APP_STATE_STOPPED, 2),
    ]
    sdk, events = _fake_modal_sdk(apps=apps)
    _install_fake_modal_api(monkeypatch)
    checked_at = datetime(2026, 10, 5, 12, 22, tzinfo=UTC)
    monkeypatch.setattr(provider, "_load_modal_sdk", lambda: sdk, raising=False)
    monkeypatch.setattr(provider, "_require_modal_sdk_version", lambda: "1.6.1")
    monkeypatch.setattr(provider, "_utcnow", lambda: checked_at, raising=False)

    result = provider.make_modal_probe(_root_rate_snapshot(), study_id)()

    assert not hasattr(sdk, "api_pb2")
    assert result == {
        "schema_version": 1,
        "checked_at_utc": "2026-10-05T12:22:00Z",
        "sdk_version": "1.6.1",
        "profile": "reflex-personal",
        "workspace": "rajath-61258",
        "billing_cycle": "2026-10",
        "billing": {
            "metered_cost": "23.40",
            "billed_cost": "22.15",
            "adjustments": {"credit": "-1.25"},
        },
        "rates_usd": _root_rate_snapshot()["rates_usd"],
        "rate_evidence": _root_rate_snapshot()["rate_evidence"],
        "apps": [
            {
                "name": f"runtime-rule-study-{study_id}",
                "app_id": "ap-current",
                "state": _FakeAPI.APP_STATE_STOPPED,
                "n_running_tasks": 0,
            },
            {
                "name": "runtime-rule-study-prior",
                "app_id": "ap-prior",
                "state": _FakeAPI.APP_STATE_EPHEMERAL,
                "n_running_tasks": 0,
            },
            {
                "name": "other-active",
                "app_id": "ap-active",
                "state": _FakeAPI.APP_STATE_EPHEMERAL,
                "n_running_tasks": 0,
            },
            {
                "name": "other-with-tasks",
                "app_id": "ap-taskful",
                "state": _FakeAPI.APP_STATE_STOPPED,
                "n_running_tasks": 2,
            },
        ],
    }
    assert events["profiles"] == ["reflex-personal"]
    assert events["cycles"] == ["2026-10"]
    assert "source_html_sha256" not in result


def test_modal_probe_rejects_credential_overrides_before_loading_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-only-credential")
    monkeypatch.setattr(
        provider,
        "_load_modal_sdk",
        lambda: pytest.fail("SDK must not load while credential overrides are set"),
    )

    probe = provider.make_modal_probe(_root_rate_snapshot(), "provider-test-002")
    with pytest.raises(ValueError, match="credential environment overrides"):
        probe()


def test_modal_probe_rejects_wrong_sdk_version_before_client_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        provider,
        "_require_modal_sdk_version",
        lambda: (_ for _ in ()).throw(RuntimeError("Modal SDK 1.6.1 is required")),
    )
    monkeypatch.setattr(
        provider,
        "_load_modal_sdk",
        lambda: _record_load(events),
    )

    with pytest.raises(RuntimeError, match="1.6.1"):
        provider.make_modal_probe(_root_rate_snapshot(), "provider-test-003")()

    assert events == []


def _record_load(events: list[str]) -> SimpleNamespace:
    events.append("loaded")
    return SimpleNamespace()


def test_modal_probe_rejects_wrong_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, events = _fake_modal_sdk(apps=[], workspace_name="another-workspace")
    _install_fake_modal_api(monkeypatch)
    monkeypatch.setattr(provider, "_load_modal_sdk", lambda: sdk)
    monkeypatch.setattr(provider, "_require_modal_sdk_version", lambda: "1.6.1")

    with pytest.raises(ValueError, match="expected workspace"):
        provider.make_modal_probe(_root_rate_snapshot(), "provider-test-004")()

    assert events["client_created"] is True
    assert "app_list_request" not in events
    assert "cycles" in events and events["cycles"] == []


def test_modal_probe_keeps_unnamed_active_apps_without_copying_descriptions() -> None:
    records = provider._relevant_apps(
        [
            SimpleNamespace(
                name="",
                description="free-form private description",
                app_id="ap-unnamed",
                state=_FakeAPI.APP_STATE_EPHEMERAL,
                n_running_tasks=0,
            ),
            SimpleNamespace(
                name="",
                description="runtime-rule-study-provider-test-007",
                app_id="ap-study",
                state=_FakeAPI.APP_STATE_STOPPED,
                n_running_tasks=0,
            ),
        ],
        _FakeAPI,
        "provider-test-007",
    )

    assert records == [
        {
            "name": "runtime-rule-study-provider-test-007",
            "app_id": "ap-study",
            "state": _FakeAPI.APP_STATE_STOPPED,
            "n_running_tasks": 0,
        },
        {
            "name": "<unnamed>",
            "app_id": "ap-unnamed",
            "state": _FakeAPI.APP_STATE_EPHEMERAL,
            "n_running_tasks": 0,
        },
    ]


def test_modal_probe_rejects_invalid_billing_without_positive_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, _events = _fake_modal_sdk(apps=[])
    _install_fake_modal_api(monkeypatch)

    class InvalidBilling:
        async def summary(self, *, cycle: str) -> SimpleNamespace:
            return SimpleNamespace(
                metered_cost=Decimal("NaN"),
                billed_cost=Decimal("1.00"),
                adjustments={},
            )

    class InvalidWorkspace:
        name = "rajath-61258"
        billing = InvalidBilling()

        @classmethod
        def from_context(cls, *, client: object) -> InvalidWorkspace:
            return cls()

        async def hydrate(self) -> InvalidWorkspace:
            return self

    sdk.workspace._Workspace = InvalidWorkspace
    monkeypatch.setattr(provider, "_load_modal_sdk", lambda: sdk)
    monkeypatch.setattr(provider, "_require_modal_sdk_version", lambda: "1.6.1")

    with pytest.raises(ValueError, match="metered cost"):
        provider.make_modal_probe(_root_rate_snapshot(), "provider-test-005")()


def _install_fake_modal(monkeypatch: pytest.MonkeyPatch, sdk: SimpleNamespace) -> None:
    monkeypatch.setattr(provider, "_load_modal_sdk", lambda: sdk)
    _install_fake_modal_api(monkeypatch)
    monkeypatch.setattr(provider, "_require_modal_sdk_version", lambda: "1.6.1")


def _app_plan(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, str]]:
    root = tmp_path / "plan" / "source"
    source_paths = (
        "experiments/runtime_rule_study_provider.py",
        "docs/runtime-rule-study-protocol.md",
    )
    root.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for relative in source_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"fake-source:{relative}\n".encode()
        path.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    plan = SimpleNamespace(
        plan_path=root.parent / "launch-plan.json",
        source_snapshot_path=root,
        source_file_sha256=hashes,
        study_id="provider-test-006",
    )
    return plan, hashes


def test_build_app_uploads_only_hashed_source_files_and_uses_fixed_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, _hashes = _app_plan(tmp_path)
    source_paths = tuple(plan.source_file_sha256)
    monkeypatch.setattr(source_files, "SOURCE_PATHS", source_paths)
    monkeypatch.setattr(provider, "_require_modal_sdk_version", lambda: "1.6.1")
    events: dict[str, object] = {"files": [], "functions": [], "mounts": []}

    class FakeImage:
        def env(self, value: dict[str, str]) -> FakeImage:
            events["env"] = value
            return self

        def pip_install(self, *packages: str, extra_index_url: str) -> FakeImage:
            events["packages"] = packages
            events["extra_index_url"] = extra_index_url
            return self

        def add_local_file(self, path: str, *, remote_path: str) -> FakeImage:
            files = events["files"]
            assert isinstance(files, list)
            files.append((path, remote_path))
            return self

    class FakeFunction:
        pass

    class FakeApp:
        def __init__(self, name: str, *, image: FakeImage) -> None:
            events["app_name"] = name
            events["image"] = image

        def function(self, **kwargs: object) -> Callable[[object], FakeFunction]:
            functions = events["functions"]
            assert isinstance(functions, list)
            functions.append(kwargs)
            return lambda _target: FakeFunction()

    class FakeVolume:
        def __init__(self, read_only: bool = False) -> None:
            self.read_only = read_only

        def with_mount_options(self, *, read_only: bool) -> FakeVolume:
            mounts = events["mounts"]
            assert isinstance(mounts, list)
            mounts.append(read_only)
            return FakeVolume(read_only=read_only)

    class VolumeFactory:
        @staticmethod
        def from_name(name: str, *, environment_name: str) -> FakeVolume:
            events["volume_name"] = name
            events["volume_environment"] = environment_name
            return FakeVolume()

    sdk = SimpleNamespace(
        config=SimpleNamespace(_set_profile=lambda profile: events.__setitem__("profile", profile)),
        Image=SimpleNamespace(debian_slim=lambda *, python_version: FakeImage()),
        Volume=VolumeFactory,
        App=FakeApp,
    )
    monkeypatch.setattr(provider, "_load_modal_sdk", lambda: sdk)
    monkeypatch.setitem(
        sys.modules,
        "experiments.runtime_rule_study_runtime",
        SimpleNamespace(remote_worker=object()),
    )

    app, functions = provider.build_app(plan)

    assert isinstance(app, FakeApp)
    assert tuple(functions) == contracts.ROLES
    assert events["app_name"] == "runtime-rule-study-provider-test-006"
    assert events["profile"] == "reflex-personal"
    assert events["volume_name"] == contracts.MODAL_VOLUME
    assert events["volume_environment"] == "main"
    assert events["env"] == {
        "USE_HUB_KERNELS": "NO",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "PYTHONPATH": "/root:/root/src",
    }
    assert events["extra_index_url"] == "https://download.pytorch.org/whl/cu130"
    assert events["packages"] == tuple(
        f"{name}=={version}" for name, version in contracts.RUNTIME_VERSION_PINS.items()
    )
    files = events["files"]
    assert isinstance(files, list)
    assert [remote for _local, remote in files] == [f"/root/{path}" for path in source_paths]
    assert all(Path(local).is_file() for local, _remote in files)
    function_rows = events["functions"]
    assert isinstance(function_rows, list)
    assert [row["name"] for row in function_rows] == [
        f"runtime_rule_study_{role}" for role in contracts.ROLES
    ]
    assert all(
        row["include_source"] is False and row["serialized"] is False for row in function_rows
    )
    assert all(row["max_containers"] == 1 for row in function_rows)
    assert all(
        row["cpu"] == (2.0, 2.0) and row["memory"] == (16_384, 16_384) for row in function_rows
    )
    assert all(
        row["min_containers"] == 0 and row["buffer_containers"] == 0 for row in function_rows
    )
    assert all(row["scaledown_window"] == 2 and row["retries"] == 0 for row in function_rows)
    assert all(row["single_use_containers"] is True for row in function_rows)
    volumes = [row["volumes"]["/artifacts"] for row in function_rows]
    assert [volume.read_only for volume in volumes] == [True, False, False]
    assert events["mounts"] == [True]


def test_build_app_rehashes_snapshot_before_loading_modal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan, _hashes = _app_plan(tmp_path)
    monkeypatch.setattr(
        source_files,
        "SOURCE_PATHS",
        tuple(plan.source_file_sha256),
    )
    first_path = plan.source_snapshot_path / next(iter(plan.source_file_sha256))
    first_path.write_text("changed after plan", encoding="utf-8")
    monkeypatch.setattr(
        provider,
        "_load_modal_sdk",
        lambda: pytest.fail("source mismatch must fail before the Modal SDK loads"),
    )

    with pytest.raises(ValueError, match="source snapshot"):
        provider.build_app(plan)


class _FakeTeardownStub:
    def __init__(
        self,
        lifecycle: Sequence[object],
        tasks: Sequence[object],
        *,
        stop_error: Exception | None = None,
    ) -> None:
        self.lifecycle = iter(lifecycle)
        self.tasks = iter(tasks)
        self.stop_error = stop_error
        self.lifecycle_requests: list[SimpleNamespace] = []
        self.task_requests: list[SimpleNamespace] = []
        self.stop_requests: list[SimpleNamespace] = []
        self.block_tasks = False
        self.block_lifecycle_calls = 0
        self.block_stop = False

    async def AppGetLifecycle(self, request: SimpleNamespace) -> object:
        self.lifecycle_requests.append(request)
        if self.block_lifecycle_calls > 0:
            self.block_lifecycle_calls -= 1
            await asyncio.Future[object]()
        value = next(self.lifecycle)
        if isinstance(value, Exception):
            raise value
        return value

    async def TaskList(self, request: SimpleNamespace) -> object:
        self.task_requests.append(request)
        if self.block_tasks:
            await asyncio.Future[object]()
        value = next(self.tasks)
        if isinstance(value, Exception):
            raise value
        return value

    async def AppStop(self, request: SimpleNamespace) -> None:
        self.stop_requests.append(request)
        if self.block_stop:
            await asyncio.Future[None]()
        if self.stop_error is not None:
            raise self.stop_error


def _lifecycle(state: str) -> SimpleNamespace:
    return SimpleNamespace(lifecycle=SimpleNamespace(app_state=state))


def _task_list(count: int) -> SimpleNamespace:
    return SimpleNamespace(tasks=[SimpleNamespace(app_id="ap-created") for _ in range(count)])


def _teardown_sdk(stub: _FakeTeardownStub) -> SimpleNamespace:
    class FakeClient:
        _stub = stub

        @classmethod
        async def from_env(cls) -> FakeClient:
            return cls()

    events: dict[str, object] = {"profiles": []}

    def select_profile(profile: str) -> None:
        profiles = events["profiles"]
        assert isinstance(profiles, list)
        profiles.append(profile)

    return SimpleNamespace(
        config=SimpleNamespace(_set_profile=select_profile),
        client=SimpleNamespace(_Client=FakeClient),
        events=events,
    )


def test_wait_for_idle_app_observes_active_then_zero_tasks_without_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FakeTeardownStub([], [_task_list(1), _task_list(0)])
    sdk = _teardown_sdk(stub)
    _install_fake_modal(monkeypatch, sdk)
    sleeps: list[float] = []

    async def no_wait(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    result = asyncio.run(provider.wait_for_idle_app("ap-created"))

    assert result["schema_version"] == 1
    assert result["app_id"] == "ap-created"
    assert result["verified"] is True
    observations = cast(list[dict[str, object]], result["observations"])
    assert [item["n_tasks"] for item in observations] == [1, 0]
    assert [item["task_app_ids"] for item in observations] == [["ap-created"], []]
    assert [request.app_id for request in stub.task_requests] == ["ap-created"] * 2
    assert all(request.environment_name == "main" for request in stub.task_requests)
    assert stub.lifecycle_requests == []
    assert stub.stop_requests == []
    assert sleeps == [provider._IDLE_POLL_INTERVAL_SECONDS]
    assert sdk.events["profiles"] == ["reflex-personal"]


def test_wait_for_idle_app_fails_closed_after_the_bounded_poll_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FakeTeardownStub([], [_task_list(1) for _ in range(provider._IDLE_POLL_ATTEMPTS)])
    _install_fake_modal(monkeypatch, _teardown_sdk(stub))
    sleeps: list[float] = []

    async def no_wait(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    result = asyncio.run(provider.wait_for_idle_app("ap-created"))

    assert result["verified"] is False
    observations = cast(list[dict[str, object]], result["observations"])
    assert len(observations) == provider._IDLE_POLL_ATTEMPTS
    assert all(item["n_tasks"] == 1 for item in observations)
    assert len(stub.task_requests) == provider._IDLE_POLL_ATTEMPTS
    assert sleeps == [provider._IDLE_POLL_INTERVAL_SECONDS] * (provider._IDLE_POLL_ATTEMPTS - 1)
    assert stub.stop_requests == []


def test_wait_for_idle_app_preserves_rpc_failures_and_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("fixture task-list failure")
    stub = _FakeTeardownStub([], [failure for _ in range(provider._IDLE_POLL_ATTEMPTS)])
    _install_fake_modal(monkeypatch, _teardown_sdk(stub))

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    result = asyncio.run(provider.wait_for_idle_app("ap-created"))

    assert result["verified"] is False
    observations = cast(list[dict[str, object]], result["observations"])
    errors = cast(list[dict[str, object]], result["errors"])
    assert len(observations) == provider._IDLE_POLL_ATTEMPTS
    assert all(item["n_tasks"] is None for item in observations)
    assert all(item["task_app_ids"] is None for item in observations)
    assert [error["stage"] for error in errors] == ["task_list"] * provider._IDLE_POLL_ATTEMPTS
    assert [error["attempt"] for error in errors] == list(
        range(1, provider._IDLE_POLL_ATTEMPTS + 1)
    )
    assert all(error["message"] == "fixture task-list failure" for error in errors)
    assert stub.stop_requests == []


def test_wait_for_idle_app_rejects_a_task_returned_for_another_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatch = SimpleNamespace(tasks=[SimpleNamespace(app_id="ap-other")])
    stub = _FakeTeardownStub([], [mismatch])
    _install_fake_modal(monkeypatch, _teardown_sdk(stub))
    monkeypatch.setattr(provider, "_IDLE_POLL_ATTEMPTS", 1)

    result = asyncio.run(provider.wait_for_idle_app("ap-created"))

    assert result["verified"] is False
    observations = cast(list[dict[str, object]], result["observations"])
    errors = cast(list[dict[str, object]], result["errors"])
    assert observations[0]["n_tasks"] is None
    assert observations[0]["task_app_ids"] == ["ap-other"]
    assert errors[0]["message"] == "Modal task list returned a task outside the requested app"
    assert stub.stop_requests == []


def test_wait_for_idle_app_bounds_each_task_list_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FakeTeardownStub([], [])
    stub.block_tasks = True
    _install_fake_modal(monkeypatch, _teardown_sdk(stub))
    monkeypatch.setattr(provider, "_IDLE_POLL_ATTEMPTS", 1)
    monkeypatch.setattr(provider, "_RPC_TIMEOUT_SECONDS", 0.001)

    result = asyncio.run(provider.wait_for_idle_app("ap-created"))

    assert result["verified"] is False
    observations = cast(list[dict[str, object]], result["observations"])
    errors = cast(list[dict[str, object]], result["errors"])
    assert observations[0]["n_tasks"] is None
    assert len(errors) == 1
    assert errors[0]["type"] == "TimeoutError"
    assert errors[0]["message"] == "Modal TaskList RPC exceeded 0.001 seconds"


@pytest.mark.parametrize("app_id", ["", " ap-created"])
def test_wait_for_idle_app_rejects_invalid_app_id_before_sdk_load(
    monkeypatch: pytest.MonkeyPatch, app_id: str
) -> None:
    monkeypatch.setattr(
        provider,
        "_load_modal_sdk",
        lambda: pytest.fail("invalid app ID must be rejected before SDK load"),
    )

    with pytest.raises(ValueError, match="app_id must be non-empty text"):
        asyncio.run(provider.wait_for_idle_app(app_id))


def test_verify_teardown_stops_only_target_and_confirms_stopped_zero_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FakeTeardownStub(
        [
            _lifecycle(_FakeAPI.APP_STATE_EPHEMERAL),
            _lifecycle(_FakeAPI.APP_STATE_STOPPED),
        ],
        [_task_list(1), _task_list(0)],
    )
    sdk = _teardown_sdk(stub)
    _install_fake_modal(monkeypatch, sdk)
    sleeps: list[float] = []

    async def no_wait(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    result = asyncio.run(provider.verify_teardown("ap-created"))

    assert result["verified"] is True
    assert result["stop_requested"] is True
    assert result["app_id"] == "ap-created"
    observations = cast(list[dict[str, object]], result["observations"])
    assert [item["state"] for item in observations] == [
        _FakeAPI.APP_STATE_EPHEMERAL,
        _FakeAPI.APP_STATE_STOPPED,
    ]
    assert [item["n_tasks"] for item in observations] == [1, 0]
    assert [request.app_id for request in stub.lifecycle_requests] == ["ap-created"] * 2
    assert [request.app_id for request in stub.task_requests] == ["ap-created"] * 2
    assert all(request.environment_name == "main" for request in stub.task_requests)
    assert [request.app_id for request in stub.stop_requests] == ["ap-created"]
    assert stub.stop_requests[0].source == _FakeAPI.APP_STOP_SOURCE_CLI
    assert sleeps == [provider._TEARDOWN_POLL_INTERVAL_SECONDS]
    assert sdk.events["profiles"] == ["reflex-personal"]


def test_verify_teardown_waits_through_the_full_bounded_shutdown_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        _FakeAPI.APP_STATE_EPHEMERAL,
        *(["APP_STATE_STOPPING"] * (provider._TEARDOWN_POLLS - 1)),
        _FakeAPI.APP_STATE_STOPPED,
    ]
    lifecycle = [_lifecycle(state) for state in states]
    tasks = [_task_list(1), *([_task_list(0)] * provider._TEARDOWN_POLLS)]
    stub = _FakeTeardownStub(lifecycle, tasks)
    _install_fake_modal(monkeypatch, _teardown_sdk(stub))
    sleeps: list[float] = []

    async def no_wait(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    result = asyncio.run(provider.verify_teardown("ap-created"))

    assert result["verified"] is True
    observations = cast(list[dict[str, object]], result["observations"])
    assert len(observations) == provider._TEARDOWN_POLLS + 1
    assert observations[-1]["state"] == _FakeAPI.APP_STATE_STOPPED
    assert observations[-1]["n_tasks"] == 0
    assert sleeps == [provider._TEARDOWN_POLL_INTERVAL_SECONDS] * provider._TEARDOWN_POLLS


def test_verify_teardown_bounds_lifecycle_and_stop_rpcs_but_continues_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FakeTeardownStub(
        [_lifecycle(_FakeAPI.APP_STATE_STOPPED)],
        [_task_list(1), _task_list(0)],
    )
    stub.block_lifecycle_calls = 1
    stub.block_stop = True
    _install_fake_modal(monkeypatch, _teardown_sdk(stub))
    monkeypatch.setattr(provider, "_TEARDOWN_POLLS", 1)
    monkeypatch.setattr(provider, "_RPC_TIMEOUT_SECONDS", 0.001)
    sleeps: list[float] = []

    async def no_wait(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    result = asyncio.run(provider.verify_teardown("ap-created"))

    assert result["verified"] is True
    assert result["stop_requested"] is True
    observations = cast(list[dict[str, object]], result["observations"])
    errors = cast(list[dict[str, object]], result["errors"])
    assert [item["state"] for item in observations] == [None, _FakeAPI.APP_STATE_STOPPED]
    assert [item["n_tasks"] for item in observations] == [1, 0]
    assert [error["stage"] for error in errors] == ["app_lifecycle", "app_stop"]
    assert [error["message"] for error in errors] == [
        "Modal AppGetLifecycle RPC exceeded 0.001 seconds",
        "Modal AppStop RPC exceeded 0.001 seconds",
    ]
    assert len(stub.lifecycle_requests) == len(stub.task_requests) == 2
    assert [request.app_id for request in stub.stop_requests] == ["ap-created"]
    assert sleeps == [provider._TEARDOWN_POLL_INTERVAL_SECONDS]


def test_verify_teardown_does_not_stop_an_already_stopped_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FakeTeardownStub([_lifecycle(_FakeAPI.APP_STATE_STOPPED)], [_task_list(0)])
    _install_fake_modal(monkeypatch, _teardown_sdk(stub))

    result = asyncio.run(provider.verify_teardown("ap-created"))

    assert result["verified"] is True
    assert result["stop_requested"] is False
    assert stub.stop_requests == []
    assert len(stub.lifecycle_requests) == len(stub.task_requests) == 1


def test_verify_teardown_preserves_rpc_failures_and_never_confirms_unknown_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("fixture transport error")
    stub = _FakeTeardownStub(
        [failure] * (provider._TEARDOWN_POLLS + 1),
        [failure] * (provider._TEARDOWN_POLLS + 1),
        stop_error=RuntimeError("fixture stop error"),
    )
    sdk = _teardown_sdk(stub)
    _install_fake_modal(monkeypatch, sdk)

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    result = asyncio.run(provider.verify_teardown("ap-missing"))

    assert result["verified"] is False
    assert result["stop_requested"] is True
    observations = cast(list[object], result["observations"])
    errors = cast(list[dict[str, object]], result["errors"])
    assert len(observations) == provider._TEARDOWN_POLLS + 1
    assert {error["stage"] for error in errors} == {
        "app_lifecycle",
        "task_list",
        "app_stop",
    }
    assert any(error["stage"] == "app_stop" for error in errors)
    assert all(request.app_id == "ap-missing" for request in stub.lifecycle_requests)
    assert all(request.app_id == "ap-missing" for request in stub.task_requests)
    assert [request.app_id for request in stub.stop_requests] == ["ap-missing"]


def test_verify_teardown_rejects_credential_overrides_before_client_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-only-credential")
    monkeypatch.setattr(
        provider,
        "_load_modal_sdk",
        lambda: pytest.fail("SDK must not load while credential overrides are set"),
    )

    with pytest.raises(ValueError, match="credential environment overrides"):
        asyncio.run(provider.verify_teardown("ap-created"))
