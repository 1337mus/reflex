"""Plan and execute one explicitly authorized runtime-rule study attempt."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from stat import S_ISREG
from typing import Any, cast

from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_launch as launch_api
from experiments import runtime_rule_study_receipts as receipts_api
from experiments import runtime_rule_study_results as results_api
from reflex_decisions.smoke import sanitize_exception_message

_GET_TRANSPORT_ALLOWANCE_SECONDS = 30
_LOCAL_WAITER_SETTLE_SECONDS = 1.0
_REMOTE_CANCEL_TIMEOUT_SECONDS = 10.0
_RATE_FIELDS = {"rates_usd", "rate_evidence", "scope", "source_html_sha256"}


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    path: Path
    study_id: str
    attempt_id: str
    status: str


def _provider_module() -> Any:
    """Load provider adapters only after explicit execute-mode authorization."""

    from experiments import runtime_rule_study_provider

    return runtime_rule_study_provider


def _utc_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_error(stage: str, error: BaseException) -> dict[str, str]:
    error_type = type(error).__name__[:128] or "Exception"
    try:
        message = sanitize_exception_message(error)
        if not isinstance(message, str) or not message.strip():
            message = f"{error_type} during {stage}."
    except Exception:
        message = f"{error_type} during {stage}; detail unavailable."
    return {"stage": stage[:128], "type": error_type, "message": message[:2_000]}


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_evidence_paths(
    *, plan_dir: Path, attempts_dir: Path, receipt_path: Path, study_id: str
) -> tuple[Path, Path]:
    plan_root = plan_dir.resolve(strict=True)
    attempts_input = Path(os.path.abspath(attempts_dir))
    receipt_input = Path(os.path.abspath(receipt_path))
    attempt_path = attempts_input / f"{study_id}.launch-attempt.json"
    resolved_destinations: list[Path] = []
    for candidate, label in ((attempt_path, "attempt"), (receipt_input, "receipt")):
        parent = candidate.parent
        try:
            real_parent = parent.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{label} parent directory must already exist") from exc
        if not parent.is_dir() or parent.is_symlink():
            raise ValueError(f"{label} parent must be a real directory")
        lexical = Path(os.path.abspath(candidate))
        resolved = real_parent / candidate.name
        if _is_within(lexical, plan_root) or _is_within(resolved, plan_root):
            raise ValueError(f"{label} path must be outside the exact launch plan tree")
        resolved_destinations.append(resolved)
    if attempt_path == receipt_input or resolved_destinations[0] == resolved_destinations[1]:
        raise ValueError("attempt and receipt paths must differ")
    for candidate, label in ((attempt_path, "attempt"), (receipt_input, "receipt")):
        if os.path.lexists(candidate):
            raise ValueError(f"{label} path already exists; evidence is immutable")
    return attempt_path, receipt_input


def _read_rates(path: Path, expected_sha256: str) -> dict[str, object]:
    contracts.validate_sha256(expected_sha256, "expected rates file SHA-256")
    try:
        info = path.lstat()
        if path.is_symlink() or not path.is_file() or not S_ISREG(info.st_mode):
            raise ValueError("rates file must be a regular non-symlink file")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError("root rates file is unavailable") from exc
    try:
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("root rates file differs from its caller-supplied SHA-256")
    value = contracts.strict_json_loads(raw)
    if not isinstance(value, Mapping) or set(value) != _RATE_FIELDS:
        raise ValueError("root rates file has an unexpected schema")
    return cast(dict[str, object], dict(value))


def _read_payloads(plan: Any) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for role in contracts.ROLES:
        path = Path(plan.role_payload_paths[role])
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"{role} authenticated payload is not a regular file")
            chunks: list[bytes] = []
            while block := os.read(descriptor, 1024 * 1024):
                chunks.append(block)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        if hashlib.sha256(raw).hexdigest() != plan.role_payload_file_sha256[role]:
            raise ValueError(f"{role} authenticated payload changed before app construction")
        value = contracts.strict_json_loads(raw)
        if contracts.canonical_json(value) != raw:
            raise ValueError(f"{role} authenticated payload is not canonical JSON")
        if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
            raise ValueError(f"{role} authenticated payload is not an object")
        payloads[role] = cast(dict[str, object], dict(value))
    return payloads


def _worker_timeout(payload: Mapping[str, object]) -> float:
    settings = payload.get("worker_settings")
    if not isinstance(settings, Mapping):
        raise ValueError("authenticated payload has no worker settings")
    timeout = settings.get("timeout_seconds")
    startup = settings.get("startup_timeout_seconds")
    if type(timeout) is not int or type(startup) is not int or timeout <= 0 or startup <= 0:
        raise ValueError("authenticated worker timeout settings are invalid")
    return float(timeout + startup + _GET_TRANSPORT_ALLOWANCE_SECONDS)


def _app_id(value: object) -> str | None:
    try:
        candidate = cast(Any, value).app_id
    except Exception:
        return None
    if type(candidate) is str and candidate and candidate.strip() == candidate:
        return candidate
    return None


def _call_id(call: object) -> str:
    value = cast(Any, call).object_id
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError("Modal FunctionCall returned an invalid object_id")
    return value


async def _cancel_remote(
    handles: Mapping[str, object], unresolved: set[str], call_records: dict[str, dict[str, object]]
) -> None:
    for role in tuple(unresolved):
        call = handles[role]
        record = call_records[role]
        if "cancel" in record:
            continue
        try:
            cancel = cast(Any, call).cancel
            outcome = await asyncio.wait_for(
                cancel.aio(terminate_containers=True), timeout=_REMOTE_CANCEL_TIMEOUT_SECONDS
            )
            record["cancel"] = {"status": "returned", "outcome": receipts_api.encode_raw(outcome)}
        except BaseException as error:
            record["cancel"] = {"status": "failed", "error": _safe_error("remote_cancel", error)}
        # A cancellation request is an observation, not proof of completion.


async def _settle_waiters(
    waiters: Mapping[str, asyncio.Task[object]],
    payloads: Mapping[str, Mapping[str, object]],
    plan: Any,
    call_records: dict[str, dict[str, object]],
    failures: list[dict[str, str]],
) -> None:
    pending = {task for task in waiters.values() if not task.done()}
    if pending:
        await asyncio.wait(pending, timeout=_LOCAL_WAITER_SETTLE_SECONDS)
    for role, task in waiters.items():
        record = call_records[role]
        if (
            task.done()
            and not record.get("raw_result_observed")
            and record.get("state") != "wait_failed"
        ):
            try:
                raw = task.result()
            except BaseException as error:
                record["state"] = "wait_failed"
                record["wait_error"] = _safe_error("wait", error)
                failures.append(_safe_error(f"{role}_wait", error))
            else:
                _accept_result(role, raw, payloads, plan, call_records, failures)
    pending_waiters = [(role, task) for role, task in waiters.items() if not task.done()]
    for _role, task in pending_waiters:
        task.cancel()
    if pending_waiters:
        settled = await asyncio.gather(
            *(task for _role, task in pending_waiters), return_exceptions=True
        )
        for (role, _task), outcome in zip(pending_waiters, settled, strict=True):
            if isinstance(outcome, asyncio.CancelledError):
                call_records[role]["waiter_cleanup"] = {"status": "cancelled_after_remote_cancel"}
            elif isinstance(outcome, BaseException):
                call_records[role]["waiter_cleanup"] = _safe_error("waiter_cleanup", outcome)
            else:
                _accept_result(role, outcome, payloads, plan, call_records, failures)


def _record_raw(record: dict[str, object], raw: object) -> None:
    record["raw_result"] = receipts_api.encode_raw(raw)
    record["raw_result_observed"] = True


def _accept_result(
    role: str,
    raw: object,
    payloads: Mapping[str, Mapping[str, object]],
    plan: Any,
    call_records: dict[str, dict[str, object]],
    failures: list[dict[str, str]],
) -> bool:
    record = call_records[role]
    _record_raw(record, raw)
    try:
        validated = results_api.validate_result(
            raw,
            payload=payloads[role],
            expected_payload_sha256=plan.role_payload_sha256[role],
        )
    except Exception as error:
        record["state"] = "validation_failed"
        record["validation_error"] = _safe_error("result_validation", error)
        failures.append(_safe_error(f"{role}_validation", error))
        return False
    record["validated_result"] = validated
    if validated.get("status") != "passed":
        record["state"] = "worker_failed"
        failures.append(
            _safe_error(f"{role}_worker", RuntimeError("worker returned a validated failed result"))
        )
        return False
    record["state"] = "passed"
    return True


async def _observe_roles(
    functions: Mapping[str, object],
    roles: tuple[str, ...],
    payloads: Mapping[str, Mapping[str, object]],
    plan: Any,
    handles: dict[str, object],
    unresolved: set[str],
    call_records: dict[str, dict[str, object]],
    failures: list[dict[str, str]],
) -> bool:
    for role in roles:
        record = call_records[role]
        try:
            spawn = cast(Any, functions[role]).spawn
            handle = await spawn.aio(payloads[role], plan.role_payload_sha256[role])
            handles[role] = handle
            unresolved.add(role)
            record["call_id"] = _call_id(handle)
            record["state"] = "spawned"
        except BaseException as error:
            record["state"] = "spawn_failed"
            record["spawn_error"] = _safe_error("spawn", error)
            failures.append(_safe_error(f"{role}_spawn", error))
            await _cancel_remote(handles, unresolved, call_records)
            if not isinstance(error, Exception):
                raise
            return False

    async def wait_for_role(role: str) -> object:
        handle = handles[role]
        get = cast(Any, handle).get
        value = await get.aio(timeout=_worker_timeout(payloads[role]))
        unresolved.discard(role)
        return value

    waiters: dict[str, asyncio.Task[object]] = {
        role: asyncio.create_task(wait_for_role(role), name=f"runtime-study-{role}-get")
        for role in roles
    }
    pending = set(waiters.values())
    has_failure = False
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for role in roles:
                task = waiters[role]
                if task not in done:
                    continue
                record = call_records[role]
                try:
                    raw = task.result()
                except BaseException as error:
                    record["state"] = "wait_failed"
                    record["wait_error"] = _safe_error("wait", error)
                    failures.append(_safe_error(f"{role}_wait", error))
                    has_failure = True
                    continue
                if not _accept_result(role, raw, payloads, plan, call_records, failures):
                    has_failure = True
            if has_failure:
                # Include a sibling that completed in the same event-loop turn.
                await asyncio.sleep(0)
                for role in roles:
                    task = waiters[role]
                    record = call_records[role]
                    if task.done() and not record.get("raw_result_observed"):
                        try:
                            raw = task.result()
                        except BaseException as error:
                            record["state"] = "wait_failed"
                            record["wait_error"] = _safe_error("wait", error)
                            failures.append(_safe_error(f"{role}_wait", error))
                        else:
                            _accept_result(role, raw, payloads, plan, call_records, failures)
                await _cancel_remote(handles, unresolved, call_records)
                await _settle_waiters(waiters, payloads, plan, call_records, failures)
                return False
    except BaseException:
        await _cancel_remote(handles, unresolved, call_records)
        await _settle_waiters(waiters, payloads, plan, call_records, failures)
        raise
    return not has_failure


async def _run_in_app(
    app: object,
    functions: Mapping[str, object],
    provider: Any,
    plan: Any,
    payloads: Mapping[str, Mapping[str, object]],
    app_name: str,
    call_records: dict[str, dict[str, object]],
    failures: list[dict[str, str]],
    app_state: dict[str, str | None],
    idle_state: dict[str, object],
) -> bool:
    handles: dict[str, object] = {}
    unresolved: set[str] = set()
    try:
        scope = cast(Any, app).run.aio(name=app_name, environment_name="main")
        async with scope as active_app:
            app_state["app_id"] = _app_id(active_app) or _app_id(app)
            unchanged_ok = await _observe_roles(
                functions,
                (contracts.ROLE_UNCHANGED,),
                payloads,
                plan,
                handles,
                unresolved,
                call_records,
                failures,
            )
            if not unchanged_ok:
                return False
            app_id = app_state["app_id"]
            if app_id is None:
                idle_state.update(
                    {"verified": False, "state": "unverified", "reason": "app ID unavailable"}
                )
                failures.append(
                    _safe_error(
                        "unchanged_idle", RuntimeError("app ID is required before training")
                    )
                )
                return False
            idle_state["state"] = "checking"
            try:
                idle_raw = await provider.wait_for_idle_app(app_id)
            except BaseException as error:
                idle_state.update(
                    {
                        "verified": False,
                        "state": "failed",
                        "error": _safe_error("unchanged_idle", error),
                    }
                )
                failures.append(_safe_error("unchanged_idle", error))
                if not isinstance(error, Exception):
                    raise
                return False
            idle_encoded = receipts_api.encode_raw(idle_raw)
            idle_verified = (
                isinstance(idle_raw, Mapping)
                and type(idle_raw.get("verified")) is bool
                and idle_raw.get("verified") is True
                and idle_raw.get("app_id") == app_id
            )
            idle_state.update(
                {
                    "verified": idle_verified,
                    "state": "verified" if idle_verified else "failed",
                    "evidence": idle_encoded,
                }
            )
            if not idle_verified:
                failures.append(
                    _safe_error(
                        "unchanged_idle",
                        RuntimeError("unchanged app did not reach verified zero-task idle"),
                    )
                )
                return False
            trained_roles = tuple(
                role for role in contracts.ROLES if role != contracts.ROLE_UNCHANGED
            )
            return await _observe_roles(
                functions,
                trained_roles,
                payloads,
                plan,
                handles,
                unresolved,
                call_records,
                failures,
            )
    except BaseException:
        # A context startup/exit exception may occur after Modal assigned the app ID.
        app_state["app_id"] = app_state["app_id"] or _app_id(app)
        if unresolved:
            await _cancel_remote(handles, unresolved, call_records)
        raise


async def _execute_authorized(
    *,
    initial_plan: Any,
    plan_path: Path,
    expected_plan_sha256: str,
    receipt_path: Path,
    authorization: Any,
    provider: Any,
) -> ExecutionReceipt:
    plan = initial_plan
    payloads: dict[str, dict[str, object]] = {}
    app: object | None = None
    functions: Mapping[str, object] = {}
    app_state: dict[str, str | None] = {"app_id": None}
    failures: list[dict[str, str]] = []
    teardown: dict[str, object] = {"verified": False, "state": "not_attempted"}
    idle: dict[str, object] = {"verified": False, "state": "not_attempted"}
    call_records: dict[str, dict[str, object]] = {
        role: {"state": "not_started", "raw_result_observed": False} for role in contracts.ROLES
    }
    all_results_passed = False
    fatal: BaseException | None = None
    try:
        # Re-authenticate the exact plan tree and source snapshot immediately before build.
        plan = launch_api.load_launch_plan(plan_path, expected_plan_sha256)
        if plan.study_id != authorization.study_id or plan.plan_sha256 != authorization.plan_sha256:
            raise ValueError("authorized plan identity changed before app construction")
        payloads = _read_payloads(plan)
        app, functions = provider.build_app(plan)
        all_results_passed = await _run_in_app(
            app,
            functions,
            provider,
            plan,
            payloads,
            f"runtime-rule-study-{plan.study_id}",
            call_records,
            failures,
            app_state,
            idle,
        )
    except BaseException as error:
        failures.append(_safe_error("lifecycle", error))
        if not isinstance(error, Exception):
            fatal = error
    finally:
        app_state["app_id"] = app_state["app_id"] or _app_id(app)
        app_id = app_state["app_id"]
        if app_id is None:
            teardown = {
                "verified": False,
                "state": "unverified",
                "reason": "no app ID was observable after app startup",
            }
        else:
            try:
                teardown_raw = await provider.verify_teardown(app_id)
                teardown_encoded = receipts_api.encode_raw(teardown_raw)
                verified = (
                    isinstance(teardown_raw, Mapping)
                    and type(teardown_raw.get("verified")) is bool
                    and teardown_raw.get("verified") is True
                    and teardown_raw.get("app_id") == app_id
                )
                teardown = {"verified": verified, "evidence": teardown_encoded}
                if not verified:
                    failures.append(
                        _safe_error("teardown", RuntimeError("created app teardown is unverified"))
                    )
            except BaseException as error:
                teardown = {"verified": False, "error": _safe_error("teardown", error)}
                failures.append(_safe_error("teardown", error))
                if not isinstance(error, Exception):
                    fatal = fatal or error

        success = all_results_passed and teardown.get("verified") is True and not failures
        receipt_document: dict[str, object] = {
            "schema_version": 1,
            "kind": "runtime-rule-study-host-receipt",
            "study_id": authorization.study_id,
            "attempt_id": authorization.attempt_id,
            "plan_sha256": authorization.plan_sha256,
            "authorized_at_utc": authorization.authorized_at_utc,
            "recorded_at_utc": _utc_text(),
            "source_commit": plan.source_commit,
            "status": "passed" if success else "failed",
            "app_id": app_id,
            "all_results_passed": all_results_passed,
            "unchanged_idle": idle,
            "teardown": teardown,
            "calls": call_records,
            "failures": failures,
            "raw_value_encoding": {
                "valid_json": "preserved as ordinary JSON values",
                "malformed_integer": (
                    "values exceeding JSON decimal digit limits use signed hexadecimal tags"
                ),
                "malformed_string": (
                    "non-UTF-8-encodable strings use tagged Python ASCII literal representations"
                ),
                "malformed_mapping": (
                    "tagged entries preserve non-string or non-UTF-8-encodable string keys"
                ),
                "non_finite_float": "tagged with nan or signed infinity",
                "unknown_custom_value": (
                    "bounded sanitized type/repr summary, repr at most "
                    f"{receipts_api.RAW_SUMMARY_LIMIT} characters; "
                    "original object state is not preserved"
                ),
                "pickle": "not used",
            },
        }
        receipts_api.write_receipt(receipt_path, receipt_document)

    if fatal is not None:
        raise fatal
    return ExecutionReceipt(
        path=receipt_path,
        study_id=authorization.study_id,
        attempt_id=authorization.attempt_id,
        status=cast(str, receipt_document["status"]),
    )


def execute_launch(
    plan_path: str | Path,
    expected_plan_sha256: str,
    *,
    project_root: str | Path,
    attempts_dir: str | Path,
    receipt_path: str | Path,
    rates_path: str | Path,
    expected_rates_sha256: str,
) -> ExecutionReceipt:
    """Synchronously authorize, then run one bounded private async Modal lifecycle."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("execute_launch must be called without an active event loop")

    plan = launch_api.load_launch_plan(plan_path, expected_plan_sha256)
    attempt_path, checked_receipt_path = _validate_evidence_paths(
        plan_dir=Path(plan.plan_path).parent,
        attempts_dir=Path(attempts_dir),
        receipt_path=Path(receipt_path),
        study_id=plan.study_id,
    )
    del attempt_path  # Path existence and plan-tree checks precede marker creation.
    rates = _read_rates(Path(rates_path), expected_rates_sha256)
    provider = _provider_module()
    github_probe = provider.github_probe
    modal_probe = provider.make_modal_probe(rates, plan.study_id)
    authorization = launch_api.authorize_launch(
        plan_path,
        expected_plan_sha256,
        source_root=project_root,
        attempts_dir=attempts_dir,
        github_probe=github_probe,
        modal_probe=modal_probe,
    )
    return asyncio.run(
        _execute_authorized(
            initial_plan=plan,
            plan_path=Path(plan_path),
            expected_plan_sha256=expected_plan_sha256,
            receipt_path=checked_receipt_path,
            authorization=authorization,
            provider=provider,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="execute an authenticated saved plan"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--study-id")
    parser.add_argument("--source-commit")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plan-path", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--rates-path", type=Path)
    parser.add_argument("--expected-rates-sha256")
    parser.add_argument("--attempts-dir", type=Path)
    parser.add_argument("--receipt-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.execute:
            if not args.study_id or not args.source_commit or args.output_dir is None:
                raise ValueError("plan mode requires --study-id, --source-commit, and --output-dir")
            receipt = launch_api.create_launch_plan(
                args.project_root,
                args.study_id,
                args.source_commit,
                args.output_dir,
            )
            print(
                json.dumps(
                    {"plan_path": str(receipt.plan_path), "plan_sha256": receipt.plan_sha256}
                )
            )
            return 0

        required = {
            "--plan-path": args.plan_path,
            "--expected-plan-sha256": args.expected_plan_sha256,
            "--rates-path": args.rates_path,
            "--expected-rates-sha256": args.expected_rates_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"execute mode requires {', '.join(missing)}")
        research = args.project_root / ".research"
        attempts = args.attempts_dir or research / "runtime-rule-study-attempts"
        receipts = args.receipt_path
        if receipts is None:
            # The plan is authenticated below; load it once to derive a stable, exclusive path.
            plan = launch_api.load_launch_plan(args.plan_path, args.expected_plan_sha256)
            receipt_dir = research / "runtime-rule-study-host-receipts"
            receipt_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            receipts = receipt_dir / f"{plan.study_id}.host-receipt.json"
        attempts.mkdir(mode=0o700, parents=True, exist_ok=True)
        outcome = execute_launch(
            args.plan_path,
            args.expected_plan_sha256,
            project_root=args.project_root,
            attempts_dir=attempts,
            receipt_path=receipts,
            rates_path=args.rates_path,
            expected_rates_sha256=args.expected_rates_sha256,
        )
        print(json.dumps({"receipt_path": str(outcome.path), "status": outcome.status}))
        return 0 if outcome.status == "passed" else 1
    except Exception as error:
        print(f"runtime-rule-study: {sanitize_exception_message(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
