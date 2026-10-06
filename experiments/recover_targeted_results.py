"""Offline admission and recovery for the saved R2 worker results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from experiments.targeted_training_core import ROLES, canonical_sha256

EVENT_KIND_SEQUENCE = (
    "identity",
    "app",
    "calls",
    "roles",
    "roles",
    "calls",
    "calls",
    "roles",
    "emergency_teardown",
    "cancellation",
    "roles",
    "terminal",
)
IDENTITIES = {
    "experiment_id": "targeted-reasoning-v1",
    "run_id": "targeted-reasoning-2026-10-05-r2",
    "source_commit": "a7c4cf41e36142bf489239852f97275eee4fd046",
    "plan_sha256": "01de7e3f5224aad57b0874deafa0204acfd1c8e1906a8a62aa18679e999390c9",
    "app_id": "ap-y5K159rl0jgFEaJ49H0AsB",
    "coordinator_call_id": "fc-01M472YVGG5ZBQEQDV3FBTCYKG",
}
CALLS = {
    "unchanged": "fc-01M472Z3AG7Q7JSRGP77FNYB8W",
    "control": "fc-01M473FAHRVPS92296J3JWCCKE",
    "treatment": "fc-01M473FCTFZS6HYQ791Y0E1T9D",
}
RECEIPT_PATH = "artifacts/targeted-reasoning-2026-10-05-r2-run/receipt.json"
EVENTS_PATH = "artifacts/targeted-reasoning-2026-10-05-r2-run/remote-events.jsonl"
PLAN_PATH = "artifacts/targeted-reasoning-2026-10-05-r2-plan/plan.json"
UNCERTAINTY = (
    "Complete, uniquely selected recorded worker results are admitted; available evidence "
    "cannot establish provider-wide absence of extra attempts or physical forwards."
)
REVIEWED_TRACKED_FILES = (
    "experiments/recover_targeted_results.py",
    "tests/test_recover_targeted_results.py",
    "docs/targeted-r2-recovery-amendment.md",
    "docs/verification/targeted-r2-recovery-amendment.json",
)


def strict_json_loads(raw: bytes) -> object:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("JSON is malformed") from exc


def _strict_jsonl_records(path: str | Path) -> Iterable[Mapping[str, object]]:
    """Yield strictly framed JSON object records from an event log."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("event file cannot be read") from exc
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("event line framing is truncated")
    for line_number, line in enumerate(raw[:-1].split(b"\n"), start=1):
        if not line.strip():
            raise ValueError(f"event line {line_number} is blank")
        try:
            record = strict_json_loads(line)
        except ValueError as exc:
            raise ValueError(f"event line {line_number} is malformed") from exc
        if not isinstance(record, Mapping):
            raise ValueError(f"event line {line_number} is not an object")
        yield record


def _resolve_regular_file(root: str | Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("pinned path is malformed")
    if any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError("pinned path must stay within root")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("pinned path must stay within root")
    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError("pinned root is not a directory")
    path = (root_path / Path(*candidate.parts)).resolve(strict=True)
    try:
        path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("pinned path escapes root") from exc
    if not path.is_file():
        raise ValueError("pinned path is not a regular file")
    return path


def _utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("event timestamp is malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("event timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("event timestamp must be UTC")
    return parsed


def _validate_event_sequence(
    events: Iterable[object], pins: Mapping[str, object], receipt: Mapping[str, object]
) -> dict[str, object]:
    identities = pins.get("identities")
    fixed_calls = pins.get("calls")
    payload_hashes = pins.get("role_payload_sha256")
    raw_hashes = pins.get("raw_worker_result_sha256")
    wrappers = receipt.get("roles")
    if not all(
        isinstance(value, Mapping)
        for value in (identities, fixed_calls, payload_hashes, raw_hashes, wrappers)
    ):
        raise ValueError("amendment event pins are malformed")
    stored_hashes = {
        role: canonical_sha256(wrappers[role]["validated_result"])
        if isinstance(wrappers.get(role), Mapping) and "validated_result" in wrappers[role]
        else None
        for role in ROLES
    }
    if any(
        set(value) != set(ROLES)
        for value in (fixed_calls, payload_hashes, raw_hashes, stored_hashes)
    ):
        raise ValueError("amendment role pins are incomplete")
    payloads = receipt.get("role_payloads")
    lifecycle = receipt.get("lifecycle")
    plan = receipt.get("plan")
    if not all(isinstance(value, Mapping) for value in (wrappers, payloads, lifecycle, plan)):
        raise ValueError("failed receipt identity is incomplete")
    if set(wrappers) != set(ROLES) or set(payloads) != set(ROLES):
        raise ValueError("failed receipt role set differs")
    expected_raw: dict[str, Mapping[str, object]] = {}
    for role in ROLES:
        payload = payloads[role]
        wrapper = wrappers[role]
        if not isinstance(payload, Mapping) or not isinstance(wrapper, Mapping):
            raise ValueError("failed receipt role data is malformed")
        raw = wrapper.get("raw_worker_result")
        if not isinstance(raw, Mapping) or canonical_sha256(raw) != raw_hashes[role]:
            raise ValueError("failed receipt raw result pin differs")
        if payload.get("payload_sha256") != payload_hashes[role]:
            raise ValueError("failed receipt payload pin differs")
        stored = wrapper.get("validated_result")
        wanted_stored_hash = stored_hashes[role]
        if wanted_stored_hash is None:
            if "validated_result" in wrapper:
                raise ValueError("failed receipt has an unexpected stored validation")
        elif not isinstance(stored, Mapping) or canonical_sha256(stored) != wanted_stored_hash:
            raise ValueError("failed receipt stored validation pin differs")
        expected_raw[role] = raw

    expected_identity = {
        key: identities[key]
        for key in ("run_id", "plan_sha256", "source_commit", "app_id", "coordinator_call_id")
    }
    calls: dict[str, str] = {}
    prior_role_set: set[str] = set()
    prior_validated: set[str] = set()
    seen_validated: set[str] = set()
    previous_time: datetime | None = None
    reference_validated_index: int | None = None
    first_arm_call_index: int | None = None
    terminal_at: str | None = None
    terminal_seen = False
    count = 0
    roles_snapshots = 0
    for count, value in enumerate(events, start=1):
        if not isinstance(value, Mapping) or set(value) != {"kind", "at_utc", "evidence"}:
            raise ValueError("event record schema is malformed")
        kind = value.get("kind")
        if count > len(EVENT_KIND_SEQUENCE) or kind != EVENT_KIND_SEQUENCE[count - 1]:
            raise ValueError("event kind or order differs from the fixed R2 trace")
        timestamp = _utc_timestamp(value.get("at_utc"))
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError("event timestamps are not strictly increasing")
        previous_time = timestamp
        evidence = value.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("event evidence is malformed")
        if kind == "identity":
            if dict(evidence) != expected_identity:
                raise ValueError("event identity differs from the fixed run")
        elif kind == "app":
            if set(evidence) != {"app_id"} or evidence.get("app_id") != identities.get("app_id"):
                raise ValueError("event app identity differs")
        elif kind == "calls":
            if not evidence or not set(evidence) <= set(ROLES):
                raise ValueError("call event role set is malformed")
            if not set(calls) <= set(evidence):
                raise ValueError("call snapshot regressed")
            old = dict(calls)
            for role, call_id in evidence.items():
                if not isinstance(call_id, str) or not call_id.startswith("fc-"):
                    raise ValueError("call event identity is malformed")
                if role in calls and calls[role] != call_id:
                    raise ValueError("role call identity changed")
                if role not in calls and (
                    call_id in calls.values() or call_id == identities.get("coordinator_call_id")
                ):
                    raise ValueError("role call identity is reused")
                calls[role] = call_id
            if any(calls.get(role) != call_id for role, call_id in old.items()):
                raise ValueError("role call history is not monotone")
            if (
                any(role in evidence for role in ("control", "treatment"))
                and first_arm_call_index is None
            ):
                first_arm_call_index = count
        elif kind == "roles":
            roles_snapshots += 1
            current_roles = set(evidence)
            if (
                not current_roles
                or not current_roles <= set(ROLES)
                or not prior_role_set <= current_roles
            ):
                raise ValueError("role result snapshot membership regressed")
            current_validated: set[str] = set()
            for role, wrapper in evidence.items():
                if role not in calls or not isinstance(wrapper, Mapping):
                    raise ValueError("role result has no prior call or malformed wrapper")
                if (
                    set(wrapper) - {"raw_worker_result", "validated_result"}
                    or "raw_worker_result" not in wrapper
                ):
                    raise ValueError("role result wrapper schema is malformed")
                raw = wrapper["raw_worker_result"]
                if not isinstance(raw, Mapping) or canonical_sha256(raw) != raw_hashes[role]:
                    raise ValueError("event raw result differs from receipt")
                if canonical_sha256(raw) != canonical_sha256(expected_raw[role]):
                    raise ValueError("event raw result changed between records")
                if (
                    raw.get("role") != role
                    or raw.get("run_id") != identities.get("run_id")
                    or raw.get("experiment_id") != identities.get("experiment_id")
                    or raw.get("payload_sha256") != payload_hashes[role]
                    or raw.get("status") != "passed"
                ):
                    raise ValueError("event raw result identity differs")
                if "validated_result" in wrapper:
                    validated = wrapper["validated_result"]
                    if (
                        stored_hashes[role] is None
                        or not isinstance(validated, Mapping)
                        or canonical_sha256(validated) != raw_hashes[role]
                        or canonical_sha256(validated) != canonical_sha256(raw)
                    ):
                        raise ValueError("event stored validation differs from raw result")
                    current_validated.add(role)
                    seen_validated.add(role)
                if role in prior_validated and role not in current_validated:
                    raise ValueError("stored validation disappeared from a later snapshot")
            if "unchanged" in current_validated and reference_validated_index is None:
                reference_validated_index = count
            prior_role_set = current_roles
            prior_validated = current_validated
        elif kind == "emergency_teardown":
            if dict(evidence) != {"deferred_until_terminal_commit": True}:
                raise ValueError("emergency teardown event schema differs")
        elif kind == "cancellation":
            if (
                set(evidence) != {"reason", "errors"}
                or evidence.get("reason") != "KeyboardInterrupt"
            ):
                raise ValueError("cancellation evidence differs")
            errors = evidence.get("errors")
            if not isinstance(errors, Mapping) or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in errors.items()
            ):
                raise ValueError("cancellation error ledger is malformed")
        elif kind == "terminal":
            terminal_seen = True
            terminal_at = value["at_utc"]
            required_terminal = {
                "schema_version",
                "status",
                "experiment_id",
                "run_id",
                "plan",
                "role_payloads",
                "roles",
                "lifecycle",
            }
            if set(evidence) != required_terminal:
                raise ValueError("terminal event schema differs")
            if any(
                evidence.get(key) != receipt.get(key)
                for key in (
                    "schema_version",
                    "experiment_id",
                    "run_id",
                    "plan",
                    "role_payloads",
                    "roles",
                )
            ):
                raise ValueError("terminal event immutable receipt fields differ")
            if evidence.get("status") != "failed" or receipt.get("status") != "failed":
                raise ValueError("terminal status is not the pinned failure")
            terminal_lifecycle = evidence.get("lifecycle")
            receipt_lifecycle = receipt.get("lifecycle")
            if not isinstance(terminal_lifecycle, Mapping) or not isinstance(
                receipt_lifecycle, Mapping
            ):
                raise ValueError("terminal lifecycle is malformed")
            if set(receipt_lifecycle) != set(terminal_lifecycle) | {"teardown"}:
                raise ValueError("receipt lifecycle gained unexpected fields")
            if any(receipt_lifecycle.get(key) != item for key, item in terminal_lifecycle.items()):
                raise ValueError("terminal lifecycle differs from final receipt")
    if count != len(EVENT_KIND_SEQUENCE) or not terminal_seen or terminal_at is None:
        raise ValueError("event trace is incomplete")
    if calls != dict(fixed_calls) or set(calls) != set(ROLES):
        raise ValueError("recorded role call map is incomplete or differs from pins")
    if set(expected_raw) != set(ROLES) or prior_role_set != set(ROLES):
        raise ValueError("not all fixed role results were recorded")
    expected_stored = {role for role in ROLES if stored_hashes[role] is not None}
    if seen_validated != expected_stored:
        raise ValueError("stored validation history differs from the pinned receipt")
    if (
        reference_validated_index is None
        or first_arm_call_index is None
        or reference_validated_index >= first_arm_call_index
    ):
        raise ValueError("reference validation did not precede arm calls")
    if lifecycle.get("status") != "failed" or lifecycle.get("failure") != {
        "type": "KeyboardInterrupt"
    }:
        raise ValueError("original controller failure differs")
    if "calls" in lifecycle:
        raise ValueError("original receipt unexpectedly contains host role calls")
    return {
        "calls": calls,
        "validated_roles": tuple(sorted(seen_validated)),
        "event_count": count,
        "role_snapshot_count": roles_snapshots,
        "terminal_at_utc": terminal_at,
    }


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_pinned(root: Path, relative: object) -> tuple[Path, bytes]:
    path = _resolve_regular_file(root, relative)
    return path, path.read_bytes()


def _json_object(raw: bytes, message: str) -> Mapping[str, object]:
    value = strict_json_loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def _checked_hash_map(root: Path, entries: object) -> Mapping[str, object]:
    if not isinstance(entries, Mapping) or not entries:
        raise ValueError("pinned file map is empty or malformed")
    for relative, expected in entries.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("pinned file digest is malformed")
        _path, raw = _read_pinned(root, relative)
        if _sha256(raw) != expected:
            raise ValueError("pinned file digest differs")
    return entries


def _validate_shutdown(
    root: Path, pins: Mapping[str, object], receipt: Mapping[str, object], terminal_at: str
) -> str:
    shutdown = pins.get("shutdown")
    if not isinstance(shutdown, Mapping):
        raise ValueError("shutdown pin is malformed")
    path, raw = _read_pinned(root, shutdown.get("source_metadata_path"))
    if _sha256(raw) != shutdown.get("source_metadata_sha256"):
        raise ValueError("shutdown metadata digest differs")
    observed = shutdown.get("observed")
    account = _json_object(raw, "shutdown metadata is malformed")
    apps = account.get("apps")
    if not isinstance(observed, Mapping) or not isinstance(apps, list) or len(apps) != 1:
        raise ValueError("shutdown observation inventory differs")
    app = apps[0]
    if not isinstance(app, Mapping) or dict(app) != dict(observed):
        raise ValueError("shutdown observation differs")
    if (
        app.get("app_id") != IDENTITIES["app_id"]
        or app.get("state") != "APP_STATE_STOPPED"
        or type(app.get("n_running_tasks")) is not int
        or app.get("n_running_tasks") != 0
        or _utc_timestamp(account.get("checked_at_utc")) <= _utc_timestamp(terminal_at)
    ):
        raise ValueError("shutdown evidence does not prove later stopped zero-task state")
    life = receipt.get("lifecycle")
    teardown = life.get("teardown") if isinstance(life, Mapping) else None
    observations = teardown.get("observations") if isinstance(teardown, Mapping) else None
    if (
        not isinstance(teardown, Mapping)
        or teardown.get("app_id") != IDENTITIES["app_id"]
        or teardown.get("verified") is not True
        or not isinstance(observations, list)
        or not any(
            isinstance(item, Mapping)
            and item.get("state") == "APP_STATE_STOPPED"
            and type(item.get("n_tasks")) is int
            and item.get("n_tasks") == 0
            for item in observations
        )
    ):
        raise ValueError("receipt teardown does not prove stopped zero-task state")
    evidence = pins.get("evidence_file_sha256")
    if not isinstance(evidence, Mapping):
        raise ValueError("supplemental evidence pins are malformed")
    lifecycle_path = ".context/targeted-r2-saved-recovery-lifecycle.json"
    account_path = ".context/targeted-r2-saved-recovery-account.json"
    if lifecycle_path not in evidence or account_path not in evidence:
        raise ValueError("latest closure evidence is not pinned")
    _checked_hash_map(
        root, {lifecycle_path: evidence[lifecycle_path], account_path: evidence[account_path]}
    )
    lifecycle = _json_object(
        _read_pinned(root, lifecycle_path)[1], "latest lifecycle evidence is malformed"
    )
    latest = _json_object(
        _read_pinned(root, account_path)[1], "latest account evidence is malformed"
    )
    if (
        lifecycle.get("app_id") != IDENTITIES["app_id"]
        or lifecycle.get("app_state") != "APP_STATE_STOPPED"
        or _utc_timestamp(lifecycle.get("stopped_at_utc")) <= _utc_timestamp(terminal_at)
        or _utc_timestamp(lifecycle.get("checked_at_utc")) <= _utc_timestamp(terminal_at)
        or latest.get("apps") != []
    ):
        raise ValueError("latest lifecycle does not preserve stopped state")

    adapter = pins.get("adapter_storage")
    if not isinstance(adapter, Mapping):
        raise ValueError("adapter metadata pin is malformed")
    _adapter_path, adapter_raw = _read_pinned(root, adapter.get("metadata_path"))
    if _sha256(adapter_raw) != adapter.get("metadata_sha256"):
        raise ValueError("adapter metadata digest differs")
    metadata = _json_object(adapter_raw, "adapter storage metadata is malformed")
    roles, descriptors = metadata.get("roles"), adapter.get("roles")
    names = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
    if (
        not isinstance(roles, Mapping)
        or not isinstance(descriptors, Mapping)
        or set(roles) != {"control", "treatment"}
        or set(descriptors) != set(roles)
    ):
        raise ValueError("adapter role inventory differs")
    for role in ("control", "treatment"):
        actual, expected = roles[role], descriptors[role]
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise ValueError("adapter descriptor is malformed")
        sizes = actual.get("files_size_bytes")
        expected_path = f"/artifacts/runs/{IDENTITIES['run_id']}-{role}/adapter-update-1200"
        if (
            actual.get("path") != expected_path
            or expected.get("path") != expected_path
            or actual.get("matches_saved_sizes") is not True
            or not isinstance(sizes, Mapping)
            or set(sizes) != names
            or dict(sizes) != expected.get("files_size_bytes")
            or any(type(size) is not int or size <= 0 for size in sizes.values())
        ):
            raise ValueError("adapter path or saved size inventory differs")
    return _sha256(raw)


def _validate_admission(
    root: Path, pins: Mapping[str, object], amendment_sha256: str
) -> dict[str, object]:
    if type(pins.get("schema_version")) is not int or pins.get("schema_version") != 1:
        raise ValueError("amendment schema is unsupported")
    if pins.get("identities") != IDENTITIES or pins.get("calls") != CALLS:
        raise ValueError("fixed R2 identity or calls differ")
    if pins.get("provider_attempt_uncertainty") != UNCERTAINTY:
        raise ValueError("provider-attempt limitation differs")
    receipt_pin, event_pin, plan_pin = (pins.get(k) for k in ("receipt", "events", "plan"))
    for pin, expected_path, expected_hash_key in (
        (receipt_pin, RECEIPT_PATH, "sha256"),
        (event_pin, EVENTS_PATH, "sha256"),
        (plan_pin, PLAN_PATH, "canonical_sha256"),
    ):
        if (
            not isinstance(pin, Mapping)
            or pin.get("path") != expected_path
            or not isinstance(pin.get(expected_hash_key), str)
        ):
            raise ValueError("fixed R2 evidence paths are malformed")
    if (
        receipt_pin["sha256"] != "f2ac56eb3d21192b0b6ac33a87ae9bce1ca559e6c5fa0f2cb6c5bc9f79730fda"
        or event_pin["sha256"] != "6f5142d831545a2db3ea3384a92e823423c98b6621c1306f3af1d31e41a75c23"
    ):
        raise ValueError("original receipt or event pin differs")
    for field in ("source_file_sha256", "input_file_sha256", "evidence_file_sha256"):
        _checked_hash_map(root, pins.get(field))
    _checker_path, checker_raw = _read_pinned(root, ".context/recount_targeted_recovery.py")
    if pins["evidence_file_sha256"].get(".context/recount_targeted_recovery.py") != _sha256(
        checker_raw
    ):
        raise ValueError("independent checker evidence hash differs")

    _plan_file, plan_raw = _read_pinned(root, plan_pin["path"])
    plan = _json_object(plan_raw, "pinned plan is malformed")
    if (
        plan.get("source_commit") != IDENTITIES["source_commit"]
        or plan.get("run_id") != IDENTITIES["run_id"]
        or plan.get("experiment_id") != IDENTITIES["experiment_id"]
        or plan.get("plan_sha256") != IDENTITIES["plan_sha256"]
        or canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"})
        != plan_pin["canonical_sha256"]
        or plan_pin["canonical_sha256"] != IDENTITIES["plan_sha256"]
    ):
        raise ValueError("canonical R2 plan identity differs")
    source_map = plan.get("source_pins")
    if not isinstance(source_map, Mapping) or dict(source_map) != pins["source_file_sha256"]:
        raise ValueError("source map differs from the fixed plan")
    from experiments import targeted_training_core as core

    if core.canonical_sha256(source_map) != plan.get(
        "source_pins_sha256"
    ) or core.targeted_source_manifest(root) != dict(source_map):
        raise ValueError("verified source snapshot differs from the plan")
    receipt_raw = _read_pinned(root, receipt_pin["path"])[1]
    if _sha256(receipt_raw) != receipt_pin["sha256"]:
        raise ValueError("original receipt bytes differ")
    receipt = _json_object(receipt_raw, "original receipt is malformed")
    if (
        set(receipt)
        != {
            "schema_version",
            "status",
            "experiment_id",
            "run_id",
            "plan",
            "role_payloads",
            "roles",
            "lifecycle",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "failed"
        or receipt.get("experiment_id") != IDENTITIES["experiment_id"]
        or receipt.get("run_id") != IDENTITIES["run_id"]
        or receipt.get("plan") != plan
    ):
        raise ValueError("original failed receipt identity differs")
    life = receipt.get("lifecycle")
    if (
        not isinstance(life, Mapping)
        or set(life) != {"status", "app_id", "coordinator_call_id", "failure", "teardown"}
        or life.get("status") != "failed"
        or life.get("app_id") != IDENTITIES["app_id"]
        or life.get("coordinator_call_id") != IDENTITIES["coordinator_call_id"]
        or life.get("failure") != {"type": "KeyboardInterrupt"}
    ):
        raise ValueError("original controller failure differs")

    payloads, wrappers = receipt.get("role_payloads"), receipt.get("roles")
    if (
        not isinstance(payloads, Mapping)
        or not isinstance(wrappers, Mapping)
        or set(payloads) != set(ROLES)
        or set(wrappers) != set(ROLES)
    ):
        raise ValueError("original receipt role inventory differs")
    payload_hashes, raw_hashes = (
        pins.get("role_payload_sha256"),
        pins.get("raw_worker_result_sha256"),
    )
    if (
        not isinstance(payload_hashes, Mapping)
        or not isinstance(raw_hashes, Mapping)
        or set(payload_hashes) != set(ROLES)
        or set(raw_hashes) != set(ROLES)
        or plan.get("role_payload_sha256") != payload_hashes
    ):
        raise ValueError("fixed role digest maps are incomplete")
    if pins.get("input_file_sha256") != payloads[ROLES[0]].get("input_file_sha256"):
        raise ValueError("fixed input file map differs")
    for role in ROLES:
        payload, wrapper = payloads[role], wrappers[role]
        if not isinstance(payload, Mapping) or not isinstance(wrapper, Mapping):
            raise ValueError("original role record is malformed")
        if (
            payload.get("role") != role
            or payload.get("run_id") != IDENTITIES["run_id"]
            or payload.get("experiment_id") != IDENTITIES["experiment_id"]
            or payload.get("source_pins") != source_map
            or payload.get("input_file_sha256") != pins["input_file_sha256"]
            or payload.get("payload_sha256") != payload_hashes[role]
            or core.canonical_sha256({k: v for k, v in payload.items() if k != "payload_sha256"})
            != payload_hashes[role]
        ):
            raise ValueError("original payload identity or digest differs")
        raw = wrapper.get("raw_worker_result")
        if not isinstance(raw, Mapping) or core.canonical_sha256(raw) != raw_hashes[role]:
            raise ValueError("recorded worker result hash differs")
        if ("validated_result" in wrapper) != (role == "unchanged"):
            raise ValueError("original stored validation provenance differs")
        if (
            "validated_result" in wrapper
            and core.canonical_sha256(wrapper["validated_result"]) != raw_hashes[role]
        ):
            raise ValueError("original stored validation differs")
        accepted = core.validate_completed_result(dict(payload), dict(raw))
        if core.canonical_sha256(accepted) != raw_hashes[role]:
            raise ValueError("frozen worker validator returned a different result")

    event_raw = _read_pinned(root, event_pin["path"])[1]
    if _sha256(event_raw) != event_pin["sha256"]:
        raise ValueError("original event bytes differ")
    events = list(_strict_jsonl_records(_read_pinned(root, event_pin["path"])[0]))
    reconstructed = _validate_event_sequence(events, pins, receipt)
    shutdown_sha = _validate_shutdown(root, pins, receipt, str(reconstructed["terminal_at_utc"]))
    bundle = pins.get("bundle")
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get("manifest_path"), str):
        raise ValueError("bundle pin is malformed")
    manifest_path, manifest_raw = _read_pinned(root, bundle["manifest_path"])
    if _sha256(manifest_raw) != bundle.get("manifest_sha256") or plan.get(
        "bundle_manifest_sha256"
    ) != bundle.get("manifest_sha256"):
        raise ValueError("bundle manifest digest differs")
    manifest = _json_object(manifest_raw, "bundle manifest is malformed")
    files = bundle.get("file_sha256")
    if (
        not isinstance(files, Mapping)
        or set(files) != set(manifest.get("bundle_file_sha256", {}))
        or manifest.get("bundle_file_sha256") != files
    ):
        raise ValueError("bundle file inventory differs")
    bundle_root = manifest_path.parent
    _checked_hash_map(
        root,
        {str((bundle_root / name).relative_to(root)): digest for name, digest in files.items()},
    )
    _validate_compatibility_projection(receipt, reconstructed["calls"], plan)
    return {
        "receipt": receipt,
        "plan": plan,
        "calls": dict(reconstructed["calls"]),
        "validated_roles": list(ROLES),
        "terminal_at_utc": reconstructed["terminal_at_utc"],
        "shutdown_evidence_sha256": shutdown_sha,
        "amendment_sha256": amendment_sha256,
    }


def _validate_clearance(
    root: Path, clearance: Mapping[str, object], amendment_sha: str, pins: Mapping[str, object]
) -> str:
    source = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(root), "verify-commit", source],
        check=True,
        capture_output=True,
        text=True,
    )
    if (
        type(clearance.get("schema_version")) is not int
        or clearance.get("schema_version") != 1
        or clearance.get("amendment_sha256") != amendment_sha
        or clearance.get("source_commit") != source
    ):
        raise ValueError("clearance amendment or signed source binding differs")
    ci = clearance.get("ci")
    if (
        not isinstance(ci, Mapping)
        or ci.get("status") != "passed"
        or ci.get("source_commit") != source
    ):
        raise ValueError("same-source CI clearance is missing")
    root_review, independent = clearance.get("root_review"), clearance.get("independent_review")
    if not isinstance(root_review, Mapping) or not isinstance(independent, Mapping):
        raise ValueError("final review clearance is missing")
    hashes = root_review.get("files_sha256")
    if (
        root_review.get("status") != "approved"
        or independent.get("status") != "approved"
        or root_review.get("source_commit") != source
        or independent.get("source_commit") != source
        or hashes != independent.get("files_sha256")
    ):
        raise ValueError("root and independent review bindings differ")
    required_files = {
        "experiments/recover_targeted_results.py",
        "tests/test_recover_targeted_results.py",
        "docs/targeted-r2-recovery-amendment.md",
        "docs/verification/targeted-r2-recovery-amendment.json",
        ".context/targeted-r2-recovery-tdd.md",
        ".context/recount_targeted_recovery.py",
        ".context/targeted-r2-independent-recovery-self-test.json",
    }
    if not isinstance(hashes, Mapping) or not required_files <= set(hashes):
        raise ValueError("review file hashes are incomplete")
    _checked_hash_map(root, hashes)
    for relative in REVIEWED_TRACKED_FILES:
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if _sha256(committed) != hashes[relative]:
            raise ValueError("reviewed tracked file differs from signed HEAD")
    checker = clearance.get("independent_checker")
    if (
        not isinstance(checker, Mapping)
        or checker.get("status") != "passed"
        or checker.get("source_commit") != source
        or checker.get("sha256")
        != pins["evidence_file_sha256"].get(".context/recount_targeted_recovery.py")
    ):
        raise ValueError("independent checker clearance is missing")
    self_test = _json_object(
        _read_pinned(root, ".context/targeted-r2-independent-recovery-self-test.json")[1],
        "independent checker self-test is malformed",
    )
    expected_checks = {
        "clearance_mismatch_blocks_gold_loader",
        "missing_approval_blocks_gold_loader",
        "identity_mismatch_blocks_gold_loader",
        "call_provenance_mismatch_blocks_gold_loader",
        "valid_provenance_reaches_sentinel_control",
    }
    if (
        self_test.get("status") != "passed"
        or self_test.get("scope") != "synthetic_only"
        or self_test.get("real_inputs_opened") is not False
        or self_test.get("score_values_emitted") is not False
        or self_test.get("checker_sha256") != checker.get("sha256")
        or not expected_checks <= set(self_test.get("checks", []))
    ):
        raise ValueError("independent checker synthetic tests are incomplete")
    return source


def _analysis_gate_then_load(
    root: Path,
    pins: Mapping[str, object],
    amendment_sha: str,
    clearance: Mapping[str, object],
    loader: Callable[[Path], object] | None = None,
) -> tuple[dict[str, object], object]:
    """Keep both clearance and full admission ahead of the host-answer loader."""
    _validate_clearance(root, clearance, amendment_sha, pins)
    admission = _validate_admission(root, pins, amendment_sha)
    if loader is None:
        from experiments import targeted_training_data as data

        inputs = data.load_inputs(root)
    else:
        inputs = loader(root)
    return admission, inputs


def _compatibility_view(
    receipt: Mapping[str, object], calls: Mapping[str, str]
) -> dict[str, object]:
    from copy import deepcopy

    view = deepcopy(dict(receipt))
    view["status"] = "passed"
    lifecycle = view["lifecycle"]
    lifecycle["status"] = "passed"
    lifecycle.pop("failure", None)
    lifecycle["calls"] = dict(calls)
    for _role, wrapper in view["roles"].items():
        wrapper["validated_result"] = deepcopy(wrapper["raw_worker_result"])
    return view


def _validate_compatibility_projection(
    receipt: Mapping[str, object], calls: Mapping[str, str], plan: Mapping[str, object]
) -> None:
    """Run the frozen execution gate before any host-answer loader is reachable."""
    from experiments import analyze_targeted_training as analysis

    analysis.validate_complete_receipt(
        _compatibility_view(receipt, calls),
        expected_plan_sha256=IDENTITIES["plan_sha256"],
        plan=plan,
    )


def _recovery_provenance(
    pins: Mapping[str, object], admission: Mapping[str, object]
) -> dict[str, object]:
    bundle = pins["bundle"]
    adapter = pins["adapter_storage"]
    return {
        "reconstructed_calls": dict(admission["calls"]),
        "validated_roles": list(admission["validated_roles"]),
        "original_failure_type": "KeyboardInterrupt",
        "shutdown_evidence_sha256": admission["shutdown_evidence_sha256"],
        "provider_attempt_uncertainty": pins["provider_attempt_uncertainty"],
        "original_events_sha256": pins["events"]["sha256"],
        "original_plan_sha256": pins["plan"]["canonical_sha256"],
        "source_commit": pins["identities"]["source_commit"],
        "role_payload_sha256": dict(pins["role_payload_sha256"]),
        "raw_worker_result_sha256": dict(pins["raw_worker_result_sha256"]),
        "input_file_sha256": dict(pins["input_file_sha256"]),
        "source_file_sha256": dict(pins["source_file_sha256"]),
        "bundle": {
            "manifest_sha256": bundle["manifest_sha256"],
            "file_sha256": dict(bundle["file_sha256"]),
        },
        "adapter_storage_metadata_sha256": adapter["metadata_sha256"],
    }


def _exclusive_json(output_dir: Path, filename: str, value: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / filename).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline R2 saved-result recovery admission")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--expected-amendment-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--clearance", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        amendment_raw = _read_pinned(root, str(args.amendment))[1]
        amendment_sha = _sha256(amendment_raw)
        if amendment_sha != args.expected_amendment_sha256:
            raise ValueError("amendment digest differs")
        pins = _json_object(amendment_raw, "amendment is malformed")
        if args.output.is_absolute():
            output_dir = args.output.resolve(strict=False)
        else:
            output_dir = (root / args.output).resolve(strict=False)
        output_dir.relative_to(root)
        if output_dir.exists():
            raise ValueError("output directory already exists")
        if args.analyze:
            if args.clearance is None:
                raise ValueError("analysis clearance is required")
            clearance = _json_object(
                _read_pinned(root, str(args.clearance))[1], "clearance is malformed"
            )
            admission, inputs = _analysis_gate_then_load(root, pins, amendment_sha, clearance)
        else:
            admission = _validate_admission(root, pins, amendment_sha)
        if not args.analyze:
            proof = {
                "schema_version": 1,
                "status": "admitted",
                "original_controller_status": "failed",
                "recovery_admission": "passed",
                "recorded_worker_execution_validated": True,
                "experiment_id": IDENTITIES["experiment_id"],
                "run_id": IDENTITIES["run_id"],
                "plan_sha256": IDENTITIES["plan_sha256"],
                "original_receipt_sha256": pins["receipt"]["sha256"],
                "amendment_sha256": amendment_sha,
                "recovery_provenance": _recovery_provenance(pins, admission),
            }
            _exclusive_json(output_dir, "admission-proof.json", proof)
            print("saved-result recovery admission passed")
            return 0
        from experiments import analyze_targeted_training as analysis

        report = analysis.analyze_receipt(
            _compatibility_view(admission["receipt"], admission["calls"]),
            inputs,
            expected_plan_sha256=IDENTITIES["plan_sha256"],
            plan=admission["plan"],
        )
        report.pop("execution_passed", None)
        report.update(
            {
                "schema_version": 1,
                "original_controller_status": "failed",
                "recovery_admission": "passed",
                "recorded_worker_execution_validated": True,
                "original_receipt_sha256": pins["receipt"]["sha256"],
                "amendment_sha256": amendment_sha,
                "recovery_provenance": _recovery_provenance(pins, admission),
            }
        )
        _exclusive_json(output_dir, "recovery-report.json", report)
        print("recovered analysis completed")
        return 0
    except Exception:
        print("R2 recovery failed; no accepted output was written", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
