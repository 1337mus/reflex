"""Validate live launch evidence, budget the fixed run, and consume one attempt ID."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from stat import S_ISDIR
from typing import Any, Protocol
from urllib.parse import urlsplit

from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_sources as source_files

MAX_PROBE_AGE = timedelta(minutes=5)
MAX_FUTURE_SKEW = timedelta(seconds=30)
MAX_RATE_AGE = timedelta(hours=24)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


class PlanAuthorityFields(Protocol):
    @property
    def plan_sha256(self) -> str: ...

    @property
    def study_id(self) -> str: ...

    @property
    def source_commit(self) -> str: ...

    @property
    def source_file_sha256(self) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class AuthorizationReceipt:
    marker_path: Path
    plan_sha256: str
    study_id: str
    attempt_id: str
    authorized_at_utc: str
    projected_exposure_usd: str
    allocation_usd: str
    reserve_usd: str


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    decoded = contracts.strict_json_loads(contracts.canonical_json(value))
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _timestamp(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp")
    if value.endswith("Z"):
        text = value[:-1] + "+00:00"
    elif value.endswith("+00:00"):
        text = value
    else:
        raise ValueError(f"{label} must use UTC Z or +00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be UTC")
    return parsed


def _timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _decimal(value: object, label: str, *, allow_negative: bool = False) -> Decimal:
    if type(value) is not str or _DECIMAL.fullmatch(value) is None:
        raise ValueError(f"{label} must be a finite decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be a finite decimal string") from exc
    if not number.is_finite() or (number < 0 and not allow_negative):
        raise ValueError(f"{label} must be finite and non-negative")
    return number


def _freshness(value: object, label: str, now: datetime, age: timedelta) -> None:
    checked = _timestamp(value, label)
    if checked > now + MAX_FUTURE_SKEW or now - checked > age:
        raise ValueError(f"{label} is stale or too far in the future")


def _github_evidence(value: object, commit: str, now: datetime) -> dict[str, Any]:
    evidence = _json_object(value, "GitHub probe evidence")
    if set(evidence) != {
        "schema_version",
        "checked_at_utc",
        "repository",
        "visibility",
        "branch",
        "branch_sha",
        "commit_sha",
        "signature",
        "ci",
    }:
        raise ValueError("GitHub probe evidence has an unexpected schema")
    if type(evidence["schema_version"]) is not int or evidence["schema_version"] != 1:
        raise ValueError("GitHub probe schema version is unsupported")
    _freshness(evidence["checked_at_utc"], "GitHub checked_at_utc", now, MAX_PROBE_AGE)
    if (
        evidence["repository"] != "1337mus/reflex"
        or evidence["visibility"] != "private"
        or evidence["branch"] != "main"
        or evidence["branch_sha"] != commit
        or evidence["commit_sha"] != commit
    ):
        raise ValueError("GitHub evidence does not prove the selected private main commit")
    signature = evidence["signature"]
    if not isinstance(signature, Mapping) or set(signature) != {"verified", "status"}:
        raise ValueError("GitHub signature evidence has an unexpected schema")
    if (
        type(signature["verified"]) is not bool
        or signature["verified"] is not True
        or signature["status"] != "verified"
    ):
        raise ValueError("selected GitHub commit signature is not verified")
    ci = evidence["ci"]
    if not isinstance(ci, Mapping) or set(ci) != {
        "workflow_path",
        "head_sha",
        "status",
        "conclusion",
        "run_id",
    }:
        raise ValueError("GitHub CI evidence has an unexpected schema")
    run_id = ci["run_id"]
    if (
        ci["workflow_path"] != ".github/workflows/ci.yml"
        or ci["head_sha"] != commit
        or ci["status"] != "completed"
        or ci["conclusion"] != "success"
        or type(run_id) is not int
        or run_id < 1
    ):
        raise ValueError("GitHub evidence does not prove successful exact-commit CI")
    return evidence


def _modal_evidence(
    value: object, study_id: str, now: datetime
) -> tuple[dict[str, Any], dict[str, Decimal]]:
    evidence = _json_object(value, "Modal probe evidence")
    if set(evidence) != {
        "schema_version",
        "checked_at_utc",
        "sdk_version",
        "profile",
        "workspace",
        "billing_cycle",
        "billing",
        "rates_usd",
        "rate_evidence",
        "apps",
    }:
        raise ValueError("Modal probe evidence has an unexpected schema")
    if type(evidence["schema_version"]) is not int or evidence["schema_version"] != 1:
        raise ValueError("Modal probe schema version is unsupported")
    _freshness(evidence["checked_at_utc"], "Modal checked_at_utc", now, MAX_PROBE_AGE)
    if evidence["sdk_version"] != "1.6.1":
        raise ValueError("Modal SDK version must be 1.6.1")
    if evidence["profile"] != "reflex-personal" or evidence["workspace"] != "rajath-61258":
        raise ValueError("Modal profile or workspace identity is not authorized")
    if evidence["billing_cycle"] != now.strftime("%Y-%m"):
        raise ValueError("Modal billing evidence is not for the current UTC cycle")

    billing = evidence["billing"]
    if not isinstance(billing, Mapping) or set(billing) != {
        "metered_cost",
        "billed_cost",
        "adjustments",
    }:
        raise ValueError("Modal billing evidence has an unexpected schema")
    _decimal(billing["metered_cost"], "metered_cost")
    _decimal(billing["billed_cost"], "billed_cost")
    adjustments = billing["adjustments"]
    if not isinstance(adjustments, Mapping) or any(type(key) is not str for key in adjustments):
        raise ValueError("Modal billing adjustments must be a decimal-string map")
    for key, amount in adjustments.items():
        _decimal(amount, f"billing adjustment {key}", allow_negative=True)

    rates_value = evidence["rates_usd"]
    rate_names = {"gpu_hour_a10g", "cpu_hour", "mem_gib_hour"}
    if not isinstance(rates_value, Mapping) or set(rates_value) != rate_names:
        raise ValueError("Modal rate evidence has an unexpected schema")
    rates = {name: _decimal(rates_value[name], name) for name in rate_names}
    if any(rate <= 0 for rate in rates.values()):
        raise ValueError("Modal resource rates must be positive")

    provenance = evidence["rate_evidence"]
    if not isinstance(provenance, Mapping) or set(provenance) != {"source_url", "checked_at_utc"}:
        raise ValueError("Modal rate provenance has an unexpected schema")
    source_url = provenance["source_url"]
    if type(source_url) is not str:
        raise ValueError("rate source URL must be text")
    parsed_url = urlsplit(source_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname is None
        or not (parsed_url.hostname == "modal.com" or parsed_url.hostname.endswith(".modal.com"))
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError("rate provenance must cite an official HTTPS Modal source")
    _freshness(provenance["checked_at_utc"], "rate checked_at_utc", now, MAX_RATE_AGE)

    apps = evidence["apps"]
    if not isinstance(apps, list):
        raise ValueError("Modal app evidence must be an array")
    seen_ids: set[str] = set()
    expected_name = f"runtime-rule-study-{study_id}"
    for app in apps:
        if not isinstance(app, Mapping) or set(app) != {
            "name",
            "app_id",
            "state",
            "n_running_tasks",
        }:
            raise ValueError("Modal app evidence has an unexpected schema")
        if any(type(app[key]) is not str or not app[key] for key in ("name", "app_id", "state")):
            raise ValueError("Modal app identity fields must be non-empty strings")
        tasks = app["n_running_tasks"]
        if type(tasks) is not int or tasks < 0:
            raise ValueError("Modal app running task count must be a strict non-negative integer")
        if app["app_id"] in seen_ids:
            raise ValueError("Modal app evidence contains a duplicate app ID")
        seen_ids.add(app["app_id"])
        if app["name"] == expected_name:
            raise ValueError("study ID was already used by a Modal app")
        if app["name"].startswith("runtime-rule-study-") and (
            app["state"] != "APP_STATE_STOPPED" or tasks > 0
        ):
            raise ValueError("another runtime-rule study app is active or queued")
    return evidence, rates


def _projected_exposure(
    payload_by_role: Mapping[str, Mapping[str, Any]], rates: Mapping[str, Decimal]
) -> Decimal:
    total = Decimal(0)
    for role in contracts.ROLES:
        settings = payload_by_role[role].get("worker_settings")
        if not isinstance(settings, Mapping):
            raise ValueError("payload worker settings are unavailable for budget estimation")
        timeout = settings.get("timeout_seconds")
        startup = settings.get("startup_timeout_seconds")
        scaledown = settings.get("scaledown_window_seconds")
        cpu = settings.get("cpu")
        memory_mib = settings.get("memory_mib")
        if (
            settings.get("gpu") != "A10"
            or type(timeout) is not int
            or timeout <= 0
            or type(startup) is not int
            or startup < 0
            or type(scaledown) is not int
            or scaledown < 0
            or type(cpu) is not int
            or cpu <= 0
            or type(memory_mib) is not int
            or memory_mib <= 0
        ):
            raise ValueError("payload worker resources are invalid for budget estimation")
        hours = Decimal(timeout + startup + scaledown) / Decimal(3600)
        total += hours * (
            rates["gpu_hour_a10g"]
            + Decimal(cpu) * rates["cpu_hour"]
            + Decimal(memory_mib) / Decimal(1024) * rates["mem_gib_hour"]
        )
    return total.quantize(Decimal("0.01"), rounding=ROUND_CEILING)


def _verify_current_sources(plan: PlanAuthorityFields, source_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="runtime-rule-study-source-check-") as temporary:
        destination = Path(temporary) / "frozen"
        measured = source_files.freeze_sources(source_root, plan.source_commit, destination)
        if source_files.validate_source_map(measured) != dict(plan.source_file_sha256):
            raise ValueError("current local source bytes differ from the authenticated plan")


def _marker_write(path: Path, value: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        remaining = memoryview(value)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("exclusive attempt marker write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def authorize_launch(
    plan: PlanAuthorityFields,
    role_payloads: Mapping[str, Mapping[str, Any]],
    *,
    source_root: str | Path,
    attempts_dir: str | Path,
    github_probe: Callable[[str], Mapping[str, object]],
    modal_probe: Callable[[], Mapping[str, object]],
    allocation_usd: Decimal,
    reserve_usd: Decimal,
    now_utc: datetime | None,
) -> AuthorizationReceipt:
    """Recheck authority with per-probe clocks, then exclusively consume one ID."""

    fixed_now = now_utc
    if fixed_now is not None and (
        not isinstance(fixed_now, datetime)
        or fixed_now.tzinfo is None
        or fixed_now.utcoffset() != timedelta(0)
    ):
        raise ValueError("now_utc must be timezone-aware UTC")

    def now() -> datetime:
        return fixed_now if fixed_now is not None else datetime.now(UTC)

    attempts = Path(attempts_dir)
    try:
        info = attempts.lstat()
    except OSError as exc:
        raise ValueError("attempts directory must already exist") from exc
    if attempts.is_symlink() or not attempts.is_dir() or not S_ISDIR(info.st_mode):
        raise ValueError("attempts directory must be a real directory")
    marker = attempts / f"{plan.study_id}.launch-attempt.json"
    if os.path.lexists(marker):
        raise ValueError("study ID already has a launch-attempt marker")
    _verify_current_sources(plan, Path(source_root))

    github_raw = github_probe(plan.source_commit)
    github = _github_evidence(github_raw, plan.source_commit, now())
    modal_raw = modal_probe()
    modal, rates = _modal_evidence(modal_raw, plan.study_id, now())

    if type(allocation_usd) is not Decimal or type(reserve_usd) is not Decimal:
        raise ValueError("allocation and reserve must be Decimal values")
    allocation, reserve = allocation_usd, reserve_usd
    if not allocation.is_finite() or not reserve.is_finite() or allocation <= 0 or reserve <= 0:
        raise ValueError("allocation and reserve must be finite and positive")
    exposure = _projected_exposure(role_payloads, rates)
    metered = _decimal(modal["billing"]["metered_cost"], "metered_cost")
    if exposure > reserve or metered + reserve > allocation:
        raise ValueError("current allocation cannot cover projected exposure and reserve")

    marker_time = now()
    _freshness(github["checked_at_utc"], "GitHub checked_at_utc", marker_time, MAX_PROBE_AGE)
    _freshness(modal["checked_at_utc"], "Modal checked_at_utc", marker_time, MAX_PROBE_AGE)
    if modal["billing_cycle"] != marker_time.strftime("%Y-%m"):
        raise ValueError("Modal billing evidence is not for the current UTC cycle")
    authorized_at = _timestamp_text(marker_time)
    attempt_id = str(uuid.uuid4())
    marker_document = {
        "schema_version": 1,
        "launch_authority": True,
        "study_id": plan.study_id,
        "attempt_id": attempt_id,
        "plan_sha256": plan.plan_sha256,
        "source_commit": plan.source_commit,
        "authorized_at_utc": authorized_at,
        "github_evidence": github,
        "modal_evidence": modal,
        "budget": {
            "allocation_usd": format(allocation, "f"),
            "reserve_usd": format(reserve, "f"),
            "metered_cost_usd": format(metered, "f"),
            "projected_exposure_usd": format(exposure, "f"),
            "note": "workspace estimate; not a provider cap or per-run invoice",
        },
    }
    _marker_write(marker, contracts.canonical_json(marker_document))
    return AuthorizationReceipt(
        marker_path=marker,
        plan_sha256=plan.plan_sha256,
        study_id=plan.study_id,
        attempt_id=attempt_id,
        authorized_at_utc=authorized_at,
        projected_exposure_usd=format(exposure, "f"),
        allocation_usd=format(allocation, "f"),
        reserve_usd=format(reserve, "f"),
    )
