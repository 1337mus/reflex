"""CPU-safe adapters for the runtime-rule study's personal providers."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_sources as source_files

if TYPE_CHECKING:
    from experiments.runtime_rule_study_launch import LaunchPlan

REPOSITORY = "1337mus/reflex"
WORKFLOW_PATH = ".github/workflows/ci.yml"
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_GH_TIMEOUT_SECONDS = 30
_MODAL_VERSION = "1.6.1"
_PROFILE = "reflex-personal"
_WORKSPACE = "rajath-61258"
_ENVIRONMENT = "main"
_IMAGE_EXTRA_INDEX = "https://download.pytorch.org/whl/cu130"
_CREDENTIAL_OVERRIDES = (
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_OAUTH_REFRESH_TOKEN",
    "MODAL_OAUTH_CLIENT_ID",
    "MODAL_OAUTH_CLIENT_SECRET",
)
_TEARDOWN_POLLS = 5
_TEARDOWN_POLL_INTERVAL_SECONDS = 2.0
_IDLE_POLL_ATTEMPTS = 6
_IDLE_POLL_INTERVAL_SECONDS = 2.0
_RPC_TIMEOUT_SECONDS = 5.0


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _gh_json(arguments: Sequence[str]) -> object:
    completed = subprocess.run(
        ["gh", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=_GH_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0 or not isinstance(completed.stdout, str):
        raise RuntimeError("GitHub evidence query failed")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("GitHub evidence query returned malformed JSON") from exc


def _required_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _utcnow() -> datetime:
    """Return an aware UTC timestamp; kept separate for deterministic fake tests."""

    return datetime.now(UTC)


def _require_no_credential_overrides() -> None:
    if any(os.environ.get(name) for name in _CREDENTIAL_OVERRIDES):
        raise ValueError("Modal credential environment overrides must be unset")


def _require_modal_sdk_version() -> str:
    try:
        version = importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Modal SDK 1.6.1 is required") from exc
    if version != _MODAL_VERSION:
        raise RuntimeError("Modal SDK 1.6.1 is required")
    return version


def _load_modal_sdk() -> Any:
    return importlib.import_module("modal")


def _load_modal_api() -> Any:
    """Load the generated protobuf module from Modal's separate package."""

    return importlib.import_module("modal_proto.api_pb2")


def _select_modal_profile(modal: Any) -> None:
    modal.config._set_profile(_PROFILE)


def _decimal_text(value: object, label: str, *, allow_negative: bool = False) -> str:
    if type(value) is not Decimal or not value.is_finite():
        raise ValueError(f"Modal {label} must be a finite decimal")
    if not allow_negative and value < 0:
        raise ValueError(f"Modal {label} must be non-negative")
    return format(value, "f")


def _validated_rate_snapshot(value: object) -> tuple[dict[str, str], dict[str, str]]:
    snapshot = _object(value, "root rate snapshot")
    if set(snapshot) != {"rates_usd", "rate_evidence", "scope", "source_html_sha256"}:
        raise ValueError("root rate snapshot has an unexpected schema")
    rates = _object(snapshot["rates_usd"], "root rate snapshot rates")
    rate_names = {"gpu_hour_a10g", "cpu_hour", "mem_gib_hour"}
    if set(rates) != rate_names:
        raise ValueError("root rate snapshot does not contain the fixed rates")
    checked_rates: dict[str, str] = {}
    for name in sorted(rate_names):
        raw = rates[name]
        if type(raw) is not str:
            raise ValueError(f"root rate {name} must be decimal text")
        try:
            amount = Decimal(raw)
        except Exception as exc:
            raise ValueError(f"root rate {name} must be decimal text") from exc
        if not amount.is_finite() or amount <= 0:
            raise ValueError(f"root rate {name} must be positive and finite")
        checked_rates[name] = raw

    provenance = _object(snapshot["rate_evidence"], "root rate provenance")
    if set(provenance) != {"source_url", "checked_at_utc"} or any(
        type(provenance[key]) is not str or not provenance[key]
        for key in ("source_url", "checked_at_utc")
    ):
        raise ValueError("root rate provenance has an unexpected schema")
    source_url = provenance["source_url"]
    checked_at = provenance["checked_at_utc"]
    if (
        type(snapshot["source_html_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", snapshot["source_html_sha256"]) is None
    ):
        raise ValueError("root rate snapshot source digest is invalid")
    if type(snapshot["scope"]) is not str or not snapshot["scope"]:
        raise ValueError("root rate snapshot scope is invalid")
    return checked_rates, {
        "source_url": cast(str, source_url),
        "checked_at_utc": cast(str, checked_at),
    }


def _modal_state_name(api: Any, state: object) -> str:
    if type(state) is str:
        return state
    try:
        value = api.AppState.Name(state)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Modal returned an unknown app lifecycle state") from exc
    if type(value) is not str or not value:
        raise ValueError("Modal returned an unknown app lifecycle state")
    return value


def _relevant_apps(apps: object, api: Any, study_id: str) -> list[dict[str, object]]:
    if isinstance(apps, (str, bytes)) or not isinstance(apps, Sequence):
        raise ValueError("Modal app list did not contain an app sequence")
    expected_name = f"runtime-rule-study-{study_id}"
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for app in apps:
        raw_name = getattr(app, "name", None)
        description = getattr(app, "description", None)
        if type(raw_name) is str and raw_name:
            name = raw_name
        elif type(description) is str and description.startswith("runtime-rule-study-"):
            name = description
        else:
            # The AppList protobuf permits unnamed ephemeral Apps. Preserve those
            # active records without copying arbitrary free-form descriptions.
            name = "<unnamed>"
        app_id = _required_string(getattr(app, "app_id", None), "Modal app ID")
        state = _modal_state_name(api, getattr(app, "state", None))
        task_count = getattr(app, "n_running_tasks", None)
        if type(task_count) is not int or task_count < 0:
            raise ValueError("Modal app running task count must be a non-negative integer")
        if app_id in seen_ids:
            raise ValueError("Modal app list contains a duplicate app ID")
        seen_ids.add(app_id)
        if name.startswith("runtime-rule-study-") or state != "APP_STATE_STOPPED" or task_count > 0:
            records.append(
                {
                    "name": name,
                    "app_id": app_id,
                    "state": state,
                    "n_running_tasks": task_count,
                }
            )
    return sorted(records, key=lambda row: row["name"] != expected_name)


async def _read_modal_probe(rate_snapshot: object, study_id: str) -> dict[str, object]:
    rates, rate_evidence = _validated_rate_snapshot(rate_snapshot)
    _require_no_credential_overrides()
    sdk_version = _require_modal_sdk_version()
    modal = _load_modal_sdk()
    _select_modal_profile(modal)

    client = await modal.client._Client.from_env()
    workspace = modal.workspace._Workspace.from_context(client=client)
    await workspace.hydrate()
    if workspace.name != _WORKSPACE:
        raise ValueError("active Modal credentials do not identify the expected workspace")

    api = _load_modal_api()
    apps_response = await client._stub.AppList(api.AppListRequest(environment_name=_ENVIRONMENT))
    apps = _relevant_apps(getattr(apps_response, "apps", None), api, study_id)

    cycle_time = _utcnow()
    if cycle_time.tzinfo is None or cycle_time.utcoffset() is None:
        raise ValueError("UTC clock must return an aware timestamp")
    billing_cycle = cycle_time.astimezone(UTC).strftime("%Y-%m")
    summary = await workspace.billing.summary(cycle=billing_cycle)
    metered_cost = _decimal_text(getattr(summary, "metered_cost", None), "metered cost")
    billed_cost = _decimal_text(getattr(summary, "billed_cost", None), "billed cost")
    adjustments_value = getattr(summary, "adjustments", None)
    if not isinstance(adjustments_value, Mapping) or any(
        type(key) is not str or not key for key in adjustments_value
    ):
        raise ValueError("Modal billing adjustments must be a string-to-decimal map")
    adjustments = {
        key: _decimal_text(amount, f"billing adjustment {key}", allow_negative=True)
        for key, amount in sorted(adjustments_value.items())
    }

    checked_at = _utcnow()
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("UTC clock must return an aware timestamp")
    checked_at_utc = checked_at.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "checked_at_utc": checked_at_utc,
        "sdk_version": sdk_version,
        "profile": _PROFILE,
        "workspace": workspace.name,
        "billing_cycle": billing_cycle,
        "billing": {
            "metered_cost": metered_cost,
            "billed_cost": billed_cost,
            "adjustments": adjustments,
        },
        "rates_usd": rates,
        "rate_evidence": rate_evidence,
        "apps": apps,
    }


def _sync_callback(coroutine_factory: Callable[[], Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_factory())
    raise RuntimeError("Modal provider probes must run before the async lifecycle starts")


def make_modal_probe(root_rate_snapshot: object, study_id: str) -> Callable[[], dict[str, object]]:
    """Capture read-only Modal account evidence without requesting compute."""

    if type(study_id) is not str or not study_id or study_id.strip() != study_id:
        raise ValueError("study_id must be non-empty text")
    rates, provenance = _validated_rate_snapshot(root_rate_snapshot)
    snapshot = {
        "rates_usd": rates,
        "rate_evidence": provenance,
        "scope": "validated injected root rate evidence",
        "source_html_sha256": _object(root_rate_snapshot, "root rate snapshot")[
            "source_html_sha256"
        ],
    }

    def probe() -> dict[str, object]:
        return cast(
            dict[str, object], _sync_callback(lambda: _read_modal_probe(snapshot, study_id))
        )

    return probe


def _plan_source_root(plan: LaunchPlan) -> Path:
    source_root = Path(plan.source_snapshot_path)
    expected_root = Path(plan.plan_path).parent / "source"
    if source_root != expected_root:
        raise ValueError("launch plan source snapshot path differs from its fixed plan directory")
    expected_hashes = source_files.validate_source_map(plan.source_file_sha256)
    if source_files.measure_sources(source_root) != expected_hashes:
        raise ValueError("frozen source snapshot differs from the launch plan source hashes")
    return source_root


def build_app(plan: LaunchPlan) -> tuple[Any, dict[str, Any]]:
    """Configure a local Modal App from exact frozen source files; never start it."""

    source_root = _plan_source_root(plan)
    _require_no_credential_overrides()
    _require_modal_sdk_version()
    modal = _load_modal_sdk()
    _select_modal_profile(modal)

    pins = [f"{name}=={version}" for name, version in contracts.RUNTIME_VERSION_PINS.items()]
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .env(
            {
                "USE_HUB_KERNELS": "NO",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "PYTHONPATH": "/root:/root/src",
            }
        )
        .pip_install(*pins, extra_index_url=_IMAGE_EXTRA_INDEX)
    )
    for relative in source_files.SOURCE_PATHS:
        image = image.add_local_file(
            str((source_root / relative).resolve(strict=True)),
            remote_path=f"/root/{relative}",
        )

    volume = modal.Volume.from_name(
        contracts.MODAL_VOLUME,
        environment_name=_ENVIRONMENT,
    )
    read_only_volume = volume.with_mount_options(read_only=True)
    app = modal.App(f"runtime-rule-study-{plan.study_id}", image=image)
    from experiments import runtime_rule_study_runtime as runtime

    functions: dict[str, Any] = {}
    for role in contracts.ROLES:
        settings = contracts.worker_settings(role)
        cpu_cores = cast(int, settings["cpu"])
        memory_mib = cast(int, settings["memory_mib"])
        min_containers = cast(int, settings["min_containers"])
        buffer_containers = cast(int, settings["buffer_containers"])
        scaledown_window_seconds = cast(int, settings["scaledown_window_seconds"])
        retries = cast(int, settings["retries"])
        startup_timeout_seconds = cast(int, settings["startup_timeout_seconds"])
        timeout_seconds = cast(int, settings["timeout_seconds"])
        function_volume = read_only_volume if role == contracts.ROLE_UNCHANGED else volume
        function = app.function(
            name=f"runtime_rule_study_{role}",
            gpu="A10",
            cpu=(float(cpu_cores), float(cpu_cores)),
            memory=(int(memory_mib), int(memory_mib)),
            max_containers=1,
            min_containers=int(min_containers),
            buffer_containers=int(buffer_containers),
            scaledown_window=int(scaledown_window_seconds),
            retries=int(retries),
            single_use_containers=bool(settings["single_use"]),
            serialized=False,
            include_source=False,
            startup_timeout=int(startup_timeout_seconds),
            timeout=int(timeout_seconds),
            volumes={"/artifacts": function_volume},
        )(runtime.remote_worker)
        functions[role] = function
    return app, functions


def _safe_error(error: BaseException) -> dict[str, str]:
    message = str(error)
    for name in _CREDENTIAL_OVERRIDES:
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"(?i)(token|secret|authorization)=?[^\s,;]+", r"\1=[redacted]", message)
    return {"type": type(error).__name__, "message": message[:500]}


async def _bounded_modal_rpc(operation: Any, method: str) -> Any:
    try:
        return await asyncio.wait_for(operation, timeout=_RPC_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise TimeoutError(f"Modal {method} RPC exceeded {_RPC_TIMEOUT_SECONDS:g} seconds") from exc


async def _idle_observation(
    client: Any, api: Any, app_id: str, attempt: int, errors: list[dict[str, object]]
) -> dict[str, object]:
    n_tasks: int | None = None
    task_app_ids: list[object] | None = None
    try:
        task_response = await _bounded_modal_rpc(
            client._stub.TaskList(
                api.TaskListRequest(environment_name=_ENVIRONMENT, app_id=app_id)
            ),
            "TaskList",
        )
        tasks = getattr(task_response, "tasks", None)
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise ValueError("Modal task list response did not include a task sequence")
        task_app_ids = [cast(object, getattr(task, "app_id", None)) for task in tasks]
        if any(
            type(task_app_id) is not str or task_app_id != app_id for task_app_id in task_app_ids
        ):
            raise ValueError("Modal task list returned a task outside the requested app")
        n_tasks = len(tasks)
    except Exception as exc:
        errors.append({"stage": "task_list", "attempt": attempt, **_safe_error(exc)})

    checked_at = _utcnow()
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("UTC clock must return an aware timestamp")
    checked_at_utc = checked_at.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "attempt": attempt,
        "checked_at_utc": checked_at_utc,
        "n_tasks": n_tasks,
        "task_app_ids": task_app_ids,
    }


async def _wait_for_idle_app(app_id: str) -> dict[str, object]:
    _require_no_credential_overrides()
    _require_modal_sdk_version()
    modal = _load_modal_sdk()
    _select_modal_profile(modal)
    api = _load_modal_api()
    client = await modal.client._Client.from_env()
    errors: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    verified = False
    for attempt in range(1, _IDLE_POLL_ATTEMPTS + 1):
        observation = await _idle_observation(client, api, app_id, attempt, errors)
        observations.append(observation)
        if observation.get("n_tasks") == 0:
            verified = True
            break
        if attempt < _IDLE_POLL_ATTEMPTS:
            await asyncio.sleep(_IDLE_POLL_INTERVAL_SECONDS)
    return {
        "schema_version": 1,
        "app_id": app_id,
        "verified": verified,
        "observations": observations,
        "errors": errors,
    }


async def wait_for_idle_app(app_id: str) -> dict[str, object]:
    """Wait for the supplied app's task list to reach zero without stopping it."""

    if type(app_id) is not str or not app_id or app_id.strip() != app_id:
        raise ValueError("app_id must be non-empty text")
    return await _wait_for_idle_app(app_id)


async def _teardown_observation(
    client: Any, api: Any, app_id: str, attempt: int, errors: list[dict[str, object]]
) -> dict[str, object]:
    state: str | None = None
    n_tasks: int | None = None
    try:
        response = await _bounded_modal_rpc(
            client._stub.AppGetLifecycle(api.AppGetLifecycleRequest(app_id=app_id)),
            "AppGetLifecycle",
        )
        lifecycle = getattr(response, "lifecycle", None)
        if lifecycle is None:
            raise ValueError("Modal lifecycle response did not include lifecycle data")
        state = _modal_state_name(api, getattr(lifecycle, "app_state", None))
    except Exception as exc:
        errors.append({"stage": "app_lifecycle", "attempt": attempt, **_safe_error(exc)})
    try:
        task_response = await _bounded_modal_rpc(
            client._stub.TaskList(
                api.TaskListRequest(environment_name=_ENVIRONMENT, app_id=app_id)
            ),
            "TaskList",
        )
        tasks = getattr(task_response, "tasks", None)
        if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
            raise ValueError("Modal task list response did not include a task sequence")
        if any(getattr(task, "app_id", None) != app_id for task in tasks):
            raise ValueError("Modal task list returned a task outside the requested app")
        n_tasks = len(tasks)
    except Exception as exc:
        errors.append({"stage": "task_list", "attempt": attempt, **_safe_error(exc)})

    checked_at = _utcnow()
    checked_at_utc = checked_at.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "attempt": attempt,
        "checked_at_utc": checked_at_utc,
        "state": state,
        "n_tasks": n_tasks,
    }


def _confirmed_stopped(observation: Mapping[str, object]) -> bool:
    return observation.get("state") == "APP_STATE_STOPPED" and observation.get("n_tasks") == 0


async def _verify_teardown(app_id: str) -> dict[str, object]:
    _require_no_credential_overrides()
    _require_modal_sdk_version()
    modal = _load_modal_sdk()
    _select_modal_profile(modal)
    api = _load_modal_api()
    client = await modal.client._Client.from_env()
    errors: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    first = await _teardown_observation(client, api, app_id, 1, errors)
    observations.append(first)
    stop_requested = False
    if not _confirmed_stopped(first):
        stop_requested = True
        try:
            await _bounded_modal_rpc(
                client._stub.AppStop(
                    api.AppStopRequest(
                        app_id=app_id,
                        source=api.APP_STOP_SOURCE_CLI,
                    )
                ),
                "AppStop",
            )
        except Exception as exc:
            errors.append({"stage": "app_stop", "attempt": 1, **_safe_error(exc)})

        for attempt in range(2, _TEARDOWN_POLLS + 2):
            await asyncio.sleep(_TEARDOWN_POLL_INTERVAL_SECONDS)
            observation = await _teardown_observation(client, api, app_id, attempt, errors)
            observations.append(observation)
            if _confirmed_stopped(observation):
                break

    verified = any(_confirmed_stopped(observation) for observation in observations)
    return {
        "schema_version": 1,
        "app_id": app_id,
        "verified": verified,
        "stop_requested": stop_requested,
        "observations": observations,
        "errors": errors,
    }


async def verify_teardown(app_id: str) -> dict[str, object]:
    """Verify only the supplied app and its tasks, stopping only that app if needed."""

    if type(app_id) is not str or not app_id or app_id.strip() != app_id:
        raise ValueError("app_id must be non-empty text")
    return await _verify_teardown(app_id)


def github_probe(source_commit: str) -> dict[str, object]:
    """Read exact repository, branch, signature, and CI evidence with the local gh CLI."""

    if type(source_commit) is not str or _SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a full lowercase Git SHA")

    repository = _object(_gh_json(("api", f"repos/{REPOSITORY}")), "repository")
    branch = _object(_gh_json(("api", f"repos/{REPOSITORY}/branches/main")), "main branch")
    commit = _object(_gh_json(("api", f"repos/{REPOSITORY}/commits/{source_commit}")), "commit")
    runs_value = _gh_json(
        (
            "run",
            "list",
            "--repo",
            REPOSITORY,
            "--workflow",
            WORKFLOW_PATH,
            "--commit",
            source_commit,
            "--json",
            "databaseId,headSha,status,conclusion,createdAt",
            "--limit",
            "100",
        )
    )

    full_name = _required_string(repository.get("full_name"), "repository full_name")
    visibility = _required_string(repository.get("visibility"), "repository visibility")
    if full_name != REPOSITORY or repository.get("private") is not True or visibility != "private":
        raise ValueError("repository is not the expected private personal repository")

    branch_name = _required_string(branch.get("name"), "branch name")
    branch_commit = _object(branch.get("commit"), "branch commit")
    branch_sha = _required_string(branch_commit.get("sha"), "branch SHA")
    if branch_name != "main" or branch_sha != source_commit:
        raise ValueError("main branch does not point at the requested source commit")

    commit_sha = _required_string(commit.get("sha"), "commit SHA")
    commit_details = _object(commit.get("commit"), "commit details")
    verification = _object(commit_details.get("verification"), "commit verification")
    if (
        commit_sha != source_commit
        or verification.get("verified") is not True
        or verification.get("reason") != "valid"
    ):
        raise ValueError("requested source commit does not have a valid verified signature")

    if not isinstance(runs_value, list):
        raise ValueError("workflow run query did not return a list")
    runs = [_object(run, "workflow run") for run in runs_value]
    matching_runs = [run for run in runs if run.get("headSha") == source_commit]
    if not matching_runs:
        raise ValueError("no CPU CI run exists for the requested source commit")
    selected = max(
        matching_runs,
        key=lambda run: _required_string(run.get("createdAt"), "workflow run createdAt"),
    )
    run_id = selected.get("databaseId")
    if (
        type(run_id) is not int
        or run_id <= 0
        or selected.get("status") != "completed"
        or selected.get("conclusion") != "success"
    ):
        raise ValueError("latest workflow run for the requested commit did not succeed")

    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "repository": full_name,
        "visibility": visibility,
        "branch": branch_name,
        "branch_sha": branch_sha,
        "commit_sha": commit_sha,
        "signature": {"verified": True, "status": "verified"},
        "ci": {
            "workflow_path": WORKFLOW_PATH,
            "head_sha": source_commit,
            "status": "completed",
            "conclusion": "success",
            "run_id": run_id,
        },
    }
