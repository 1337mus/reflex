"""Create immutable CPU launch plans and authorize one bounded study attempt."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Any

from experiments import (
    runtime_rule_study_authority as authority,
)
from experiments import (
    runtime_rule_study_bundle as bundle_builder,
)
from experiments import (
    runtime_rule_study_cache as cache,
)
from experiments import (
    runtime_rule_study_contracts as contracts,
)
from experiments import (
    runtime_rule_study_inputs as inputs,
)
from experiments import (
    runtime_rule_study_payloads as payloads,
)
from experiments import (
    runtime_rule_study_sources as source_files,
)
from experiments import (
    runtime_rule_study_tokens as tokens,
)

AuthorizationReceipt = authority.AuthorizationReceipt

PLAN_FILENAME = "launch-plan.json"
PLAN_SCHEMA_VERSION = 1
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_FIELDS = {
    "schema_version",
    "experiment_id",
    "study_id",
    "nonce",
    "created_at_utc",
    "source_commit",
    "source_file_sha256",
    "source_snapshot_path",
    "protocol_sha256",
    "bundle_sha256",
    "role_payloads",
    "launch_authority",
}


@dataclass(frozen=True, slots=True)
class PlanReceipt:
    plan_path: Path
    plan_sha256: str
    study_id: str
    payload_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    plan_path: Path
    plan_sha256: str
    study_id: str
    nonce: str
    source_commit: str
    source_file_sha256: Mapping[str, str]
    source_snapshot_path: Path
    bundle_sha256: str
    role_payload_paths: Mapping[str, Path]
    role_payload_file_sha256: Mapping[str, str]
    role_payload_sha256: Mapping[str, str]
    launch_authority: bool


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    decoded = contracts.strict_json_loads(contracts.canonical_json(value))
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _git_sha(value: object, label: str) -> str:
    if type(value) is not str or _GIT_SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase Git SHA")
    return value


def _utc_now(value: datetime | None) -> datetime:
    now = datetime.now(UTC) if value is None else value
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ValueError("now_utc must be timezone-aware UTC")
    return now


def _timestamp(value: object, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be UTC")
    return parsed


def _timestamp_text(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def _relative_parts(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not str or not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"{label} must be a safe relative path")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must be a safe relative path")
    return parts


def _read_relative(root: Path, relative: str) -> bytes:
    """Read one regular file below root without following any saved-path symlink."""

    parts = _relative_parts(relative, "saved path")
    try:
        info = root.lstat()
        if not root.is_dir() or root.is_symlink() or not S_ISDIR(info.st_mode):
            raise ValueError("plan directory must be a real directory")
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError("plan directory is unavailable") from exc
    try:
        parent_fd = descriptor
        for part in parts[:-1]:
            try:
                child_info = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if not _is_directory_mode(child_info.st_mode):
                    raise ValueError("saved path has a non-directory or symlink component")
                child_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise ValueError("saved path has an unavailable or symlink component") from exc
            if parent_fd != descriptor:
                os.close(parent_fd)
            parent_fd = child_fd
        try:
            leaf = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if not _is_regular_mode(leaf.st_mode):
                raise ValueError("saved path is not a regular file")
            file_fd = os.open(
                parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
            )
        except OSError as exc:
            raise ValueError("saved file is unavailable or is a symlink") from exc
        try:
            chunks: list[bytes] = []
            while block := os.read(file_fd, 1024 * 1024):
                chunks.append(block)
            return b"".join(chunks)
        finally:
            os.close(file_fd)
    finally:
        if parent_fd != descriptor:
            os.close(parent_fd)
        os.close(descriptor)


def _is_directory_mode(mode: int) -> bool:
    return S_ISDIR(mode)


def _is_regular_mode(mode: int) -> bool:
    return S_ISREG(mode)


def _assert_exact_tree(root: Path, files: set[str]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("saved artifact root must be a real directory")
    expected_dirs = {
        "/".join(parts[:index])
        for filename in files
        for parts in (_relative_parts(filename, "saved path"),)
        for index in range(1, len(parts))
    }
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        prefix = "" if relative_dir == Path(".") else relative_dir.as_posix()
        for name in dirs:
            entry = current_path / name
            if entry.is_symlink():
                raise ValueError("saved artifact tree contains a symlink")
            actual_dirs.add(f"{prefix}/{name}".strip("/"))
        for name in names:
            entry = current_path / name
            if entry.is_symlink() or not entry.is_file():
                raise ValueError("saved artifact tree contains a symlink or non-regular file")
            actual_files.add(f"{prefix}/{name}".strip("/"))
    if actual_files != files or actual_dirs != expected_dirs:
        raise ValueError("saved artifact tree differs from its exact frozen file set")


def _write_exclusive(path: Path, value: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive plan file write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _build_bundle(project_root: Path) -> dict[str, Any]:
    study_inputs = inputs.load_study_inputs(project_root)
    tokenizer = tokens.load_cpu_tokenizer(project_root)
    compilation = tokens.compile_study_inputs(study_inputs, tokenizer)
    cached = cache.load_cached_retention(project_root, study_inputs, compilation)
    return bundle_builder.build_study_bundle(project_root, study_inputs, compilation, cached)


def _payload_path(role: str) -> str:
    return f"payloads/{role}.json"


def create_launch_plan(
    project_root: str | Path,
    study_id: str,
    source_commit: str,
    output_dir: str | Path,
    *,
    now_utc: datetime | None = None,
) -> PlanReceipt:
    """Freeze source and role payloads into a new plan directory; never grant authority."""

    checked_study = contracts.validate_safe_run_id(study_id, "study_id")
    checked_commit = _git_sha(source_commit, "source_commit")
    now = _utc_now(now_utc)
    root = Path(project_root)
    output = Path(output_dir)
    if output.name in {"", ".", ".."}:
        raise ValueError("launch plan output must name a new directory")
    if os.path.lexists(output):
        raise ValueError("launch plan output directory must not already exist")
    try:
        source_root = root.resolve(strict=True)
        parent = output.parent.resolve(strict=True)
        if not source_root.is_dir() or not parent.is_dir():
            raise ValueError("project root and plan parent must be directories")
    except OSError as exc:
        raise ValueError("project root or plan parent is unavailable") from exc
    output = parent / output.name
    output.mkdir(mode=0o700)
    try:
        source_dir = output / "source"
        source_hashes = source_files.validate_source_map(
            source_files.freeze_sources(source_root, checked_commit, source_dir)
        )
        _assert_exact_tree(source_dir, set(source_files.SOURCE_PATHS))

        bundle = _build_bundle(source_root)
        bundle_digest = contracts.validate_sha256(bundle.get("bundle_sha256"), "bundle SHA-256")
        nonce = str(uuid.uuid4())
        payload_dir = output / "payloads"
        payload_dir.mkdir(mode=0o700)
        payload_refs: dict[str, dict[str, str]] = {}
        for role in contracts.ROLES:
            payload = payloads.build_worker_payload(
                bundle,
                expected_bundle_sha256=bundle_digest,
                study_id=checked_study,
                nonce=nonce,
                role=role,
                source_commit=checked_commit,
                source_file_sha256=source_hashes,
            )
            validated = payloads.validate_payload(payload)
            payload_bytes = contracts.canonical_json(validated)
            payload_digest = contracts.validate_sha256(
                validated.get("payload_sha256"), "payload SHA-256"
            )
            payload_ref = {
                "path": _payload_path(role),
                "file_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "payload_sha256": payload_digest,
            }
            payload_refs[role] = payload_ref
            _write_exclusive(output / payload_ref["path"], payload_bytes)

        document: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "experiment_id": contracts.EXPERIMENT_ID,
            "study_id": checked_study,
            "nonce": nonce,
            "created_at_utc": _timestamp_text(now),
            "source_commit": checked_commit,
            "source_file_sha256": source_hashes,
            "source_snapshot_path": "source",
            "protocol_sha256": contracts.PROTOCOL_SHA256,
            "bundle_sha256": bundle_digest,
            "role_payloads": payload_refs,
            "launch_authority": False,
        }
        _validate_plan_document(document)
        plan_bytes = contracts.canonical_json(document)
        plan_path = output / PLAN_FILENAME
        _write_exclusive(plan_path, plan_bytes)
        _assert_exact_tree(
            output,
            {
                PLAN_FILENAME,
                *(f"source/{path}" for path in source_files.SOURCE_PATHS),
                *(_payload_path(role) for role in contracts.ROLES),
            },
        )
        return PlanReceipt(
            plan_path=plan_path,
            plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
            study_id=checked_study,
            payload_sha256={role: payload_refs[role]["payload_sha256"] for role in contracts.ROLES},
        )
    except BaseException:
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output, ignore_errors=True)
        raise


def _validate_plan_document(value: object) -> dict[str, Any]:
    plan = _json_object(value, "launch plan")
    if set(plan) != _PLAN_FIELDS:
        raise ValueError("launch plan has an unexpected schema")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("launch plan schema version is unsupported")
    if plan["experiment_id"] != contracts.EXPERIMENT_ID:
        raise ValueError("launch plan experiment identity is unsupported")
    contracts.validate_safe_run_id(plan["study_id"], "study_id")
    contracts.validate_uuid(plan["nonce"], "nonce")
    _timestamp(plan["created_at_utc"], "created_at_utc")
    _git_sha(plan["source_commit"], "source_commit")
    hashes = source_files.validate_source_map(plan["source_file_sha256"])
    if plan["source_snapshot_path"] != "source":
        raise ValueError("source snapshot path differs from the fixed plan directory")
    if plan["protocol_sha256"] != contracts.PROTOCOL_SHA256:
        raise ValueError("launch plan protocol differs from the frozen study")
    bundle_sha = contracts.validate_sha256(plan["bundle_sha256"], "bundle SHA-256")
    if type(plan["launch_authority"]) is not bool or plan["launch_authority"] is not False:
        raise ValueError("saved plan cannot contain launch authority")
    role_refs = plan["role_payloads"]
    if not isinstance(role_refs, Mapping) or set(role_refs) != set(contracts.ROLES):
        raise ValueError("launch plan role payloads differ from the fixed roles")
    for role in contracts.ROLES:
        ref = _json_object(role_refs[role], f"{role} payload reference")
        if set(ref) != {"path", "file_sha256", "payload_sha256"}:
            raise ValueError("launch plan payload reference has an unexpected schema")
        if ref["path"] != _payload_path(role):
            raise ValueError("launch plan payload path differs from the fixed relative path")
        _sha256(ref["file_sha256"], "payload file SHA-256")
        _sha256(ref["payload_sha256"], "payload SHA-256")
    plan["source_file_sha256"] = hashes
    plan["bundle_sha256"] = bundle_sha
    return plan


def _load_plan(
    plan_path: str | Path, expected_plan_sha256: str
) -> tuple[LaunchPlan, dict[str, Any], dict[str, dict[str, Any]]]:
    expected_digest = _sha256(expected_plan_sha256, "expected plan SHA-256")
    path = Path(plan_path)
    if path.name != PLAN_FILENAME or path.is_symlink():
        raise ValueError("plan path must name a regular launch plan file")
    directory = path.parent
    plan_bytes = _read_relative(directory, PLAN_FILENAME)
    actual_digest = hashlib.sha256(plan_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("launch plan file differs from its expected SHA-256")
    document = contracts.strict_json_loads(plan_bytes)
    if not isinstance(document, Mapping) or contracts.canonical_json(document) != plan_bytes:
        raise ValueError("launch plan must use canonical JSON encoding")
    plan = _validate_plan_document(document)
    expected_files = {
        PLAN_FILENAME,
        *(f"source/{path}" for path in source_files.SOURCE_PATHS),
        *(_payload_path(role) for role in contracts.ROLES),
    }
    _assert_exact_tree(directory, expected_files)
    frozen_sources = source_files.measure_sources(directory / "source")
    if frozen_sources != plan["source_file_sha256"]:
        raise ValueError("frozen source bytes differ from the authenticated plan")

    loaded_payloads: dict[str, dict[str, Any]] = {}
    for role in contracts.ROLES:
        ref = plan["role_payloads"][role]
        payload_bytes = _read_relative(directory, ref["path"])
        if hashlib.sha256(payload_bytes).hexdigest() != ref["file_sha256"]:
            raise ValueError(f"{role} payload file differs from its authenticated SHA-256")
        parsed = contracts.strict_json_loads(payload_bytes)
        if not isinstance(parsed, Mapping) or contracts.canonical_json(parsed) != payload_bytes:
            raise ValueError(f"{role} payload must use canonical JSON encoding")
        payload = payloads.validate_payload(parsed, expected_payload_sha256=ref["payload_sha256"])
        if (
            payload.get("role") != role
            or payload.get("study_id") != plan["study_id"]
            or payload.get("nonce") != plan["nonce"]
            or payload.get("source_commit") != plan["source_commit"]
            or payload.get("source_file_sha256") != plan["source_file_sha256"]
            or payload.get("bundle_sha256") != plan["bundle_sha256"]
        ):
            raise ValueError(f"{role} payload identity differs from its authenticated plan")
        loaded_payloads[role] = payload

    payload_refs = plan["role_payloads"]
    directory = path.parent
    receipt = LaunchPlan(
        plan_path=path,
        plan_sha256=actual_digest,
        study_id=plan["study_id"],
        nonce=plan["nonce"],
        source_commit=plan["source_commit"],
        source_file_sha256=plan["source_file_sha256"],
        source_snapshot_path=directory / plan["source_snapshot_path"],
        bundle_sha256=plan["bundle_sha256"],
        role_payload_paths={
            role: directory / payload_refs[role]["path"] for role in contracts.ROLES
        },
        role_payload_file_sha256={
            role: payload_refs[role]["file_sha256"] for role in contracts.ROLES
        },
        role_payload_sha256={
            role: payload_refs[role]["payload_sha256"] for role in contracts.ROLES
        },
        launch_authority=False,
    )
    return receipt, plan, loaded_payloads


def load_launch_plan(plan_path: str | Path, expected_plan_sha256: str) -> LaunchPlan:
    """Authenticate the plan, frozen tree, and all concrete role payloads."""

    receipt, _document, _payloads = _load_plan(plan_path, expected_plan_sha256)
    return receipt


def authorize_launch(
    plan_path: str | Path,
    expected_plan_sha256: str,
    *,
    source_root: str | Path,
    attempts_dir: str | Path,
    github_probe: Callable[[str], Mapping[str, object]],
    modal_probe: Callable[[], Mapping[str, object]],
    allocation_usd: Decimal = Decimal("300"),
    reserve_usd: Decimal = Decimal("10"),
    now_utc: datetime | None = None,
) -> AuthorizationReceipt:
    """Recheck live authority and exclusively consume the study ID before any launch."""

    plan, _document, role_payloads = _load_plan(plan_path, expected_plan_sha256)
    return authority.authorize_launch(
        plan,
        role_payloads,
        source_root=source_root,
        attempts_dir=attempts_dir,
        github_probe=github_probe,
        modal_probe=modal_probe,
        allocation_usd=allocation_usd,
        reserve_usd=reserve_usd,
        now_utc=now_utc,
    )
