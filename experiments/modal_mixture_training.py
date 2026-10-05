"""Dry-run-first CLI and Modal orchestration for the controlled mixture experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from experiments import mixture_training_core as core
from experiments import mixture_training_data as data
from experiments import mixture_training_modal_host as modal_host
from experiments import mixture_training_runtime_helpers as helpers
from reflex_decisions import smoke

_LIMITS: dict[str, object] = {
    "gpu": "A10",
    "cpu_cores": 2,
    "memory_mib": 16384,
    "max_containers_per_function": 1,
    "min_containers": 0,
    "buffer_containers": 0,
    "scaledown_window_seconds": 2,
    "single_use_containers": True,
    "retries": 0,
    "startup_timeout_seconds": 300,
    "initialize_timeout_seconds": 900,
    "training_timeout_seconds": 3600,
}


def _sdk_version() -> str | None:
    try:
        return importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_error_message(error: BaseException) -> str:
    try:
        return smoke.sanitize_exception_message(error)
    except Exception:
        return "operation failed; details could not be sanitized"


def _new_receipt(experiment_id: str, *, profile: str, workspace: str) -> dict[str, object]:
    return {
        "schema_version": core.SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": "failed",
        "payloads": {},
        "results": {},
        "failure": None,
        "modal": {
            "profile": profile,
            "workspace": workspace,
            "sdk_version": _sdk_version(),
            "app_id": None,
            "limits": dict(_LIMITS),
        },
    }


def _record_failure(receipt: dict[str, object], stage: str, error: BaseException) -> None:
    if receipt.get("failure") is None:
        receipt["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": _safe_error_message(error),
        }
    receipt["status"] = "failed"


def _make_payloads(
    experiment_id: str, root: Path
) -> tuple[dict[str, object], tuple[Any, ...], tuple[Any, ...], dict[str, object]]:
    (
        real_records,
        balanced_records,
        synthetic_records,
        snli_training_records,
        pins,
    ) = core.load_natural_reasoning_data(root)
    real_train = tuple(
        record for record in real_records if record.dataset_id in data.REAL_TRAIN_DATASET_COUNTS
    )
    synthetic_train = tuple(
        record
        for record in synthetic_records
        if record.dataset_id in data.SYNTHETIC_TRAIN_DATASET_COUNTS
    )
    train_records = (*real_train, *synthetic_train, *snli_training_records)
    initialization_evaluation = data.build_synthetic_evaluation_presentations(synthetic_records)
    full_evaluation = data.build_evaluation_presentations(
        real_records, balanced_records, synthetic_records
    )
    init_payload = core.build_payload(
        experiment_id=experiment_id,
        run_id=f"{experiment_id}-init",
        nonce=str(uuid.uuid4()),
        phase="initialize",
        arm=None,
        pins=pins,
        train_records=(),
        evaluation_presentations=initialization_evaluation,
    )
    return init_payload, train_records, full_evaluation, pins


def _make_arm_payloads(
    experiment_id: str,
    train_records: tuple[Any, ...],
    evaluation_presentations: tuple[Any, ...],
    pins: dict[str, object],
    initialization: object,
) -> dict[str, dict[str, object]]:
    return {
        arm: core.build_payload(
            experiment_id=experiment_id,
            run_id=f"{experiment_id}{suffix}",
            nonce=str(uuid.uuid4()),
            phase="train",
            arm=arm,
            pins=pins,
            train_records=train_records,
            evaluation_presentations=evaluation_presentations,
            initialization=initialization,
        )
        for arm in data.ARM_NAMES
        for suffix in (data.RUN_SUFFIXES[arm],)
    }


def _make_failed_result(
    payload: dict[str, object], stage: str, error: BaseException
) -> dict[str, object]:
    return helpers.failed_worker_result(
        payload,
        schema_version=core.SCHEMA_VERSION,
        model_id=core.MODEL_ID,
        model_revision=core.MODEL_REVISION,
        stage=stage,
        error=error,
        sanitize=_safe_error_message,
    )


def _execute_lifecycle(
    *,
    init_payload: dict[str, object],
    invoke_init: Callable[[dict[str, object]], object],
    make_arm_payloads: Callable[[object], Mapping[str, dict[str, object]]],
    invoke_arm: Callable[[str, dict[str, object]], object],
    validate_result: Callable[[object, dict[str, object]], dict[str, object]],
    make_failed_result: Callable[[dict[str, object], str, BaseException], dict[str, object]]
    | None = None,
) -> dict[str, object]:
    """Validate initialization before submitting both independent training arms."""

    payloads: dict[str, dict[str, object]] = {"initialize": init_payload}
    results: dict[str, dict[str, object]] = {}
    try:
        init_result = validate_result(invoke_init(init_payload), init_payload)
    except Exception as exc:
        if make_failed_result is None:
            raise
        results["initialize"] = make_failed_result(init_payload, "initialization_worker", exc)
        return {"payloads": payloads, "results": results}
    results["initialize"] = init_result
    if init_result.get("status") != "passed":
        return {"payloads": payloads, "results": results}

    evidence = init_result.get("evidence")
    initialization = evidence.get("initialization") if isinstance(evidence, dict) else None
    arm_payloads = dict(make_arm_payloads(initialization))
    if set(arm_payloads) != set(data.ARM_NAMES):
        raise ValueError("training payloads must contain both approved arms")
    payloads.update(arm_payloads)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mixture-arm") as executor:
        futures = {
            arm: executor.submit(invoke_arm, arm, payload) for arm, payload in arm_payloads.items()
        }
        for arm, future in futures.items():
            try:
                results[arm] = validate_result(future.result(), arm_payloads[arm])
            except Exception as exc:
                if make_failed_result is not None:
                    results[arm] = make_failed_result(arm_payloads[arm], "worker_execution", exc)
                else:
                    results[arm] = {
                        "status": "failed",
                        "failure": {
                            "stage": "worker_execution",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
    return {"payloads": payloads, "results": results}


def _save_untrusted_response(output_path: Path, responses: dict[str, object]) -> Path | None:
    if not responses:
        return None
    safe_value: dict[str, object] = {}
    for key, value in responses.items():
        try:
            safe_value[key] = json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError, OverflowError, RecursionError):
            safe_value[key] = {
                "unserializable_type": type(value).__name__,
                "message": "response could not be represented as strict JSON",
            }
    destination = output_path.with_name(f"{output_path.name}.untrusted-{uuid.uuid4()}.json")
    helpers.write_json_exclusive(destination, safe_value)
    return destination


def _run(args: argparse.Namespace, reservation: smoke.ArtifactReservation) -> int:
    experiment_id = args.experiment_id
    receipt = _new_receipt(experiment_id, profile=args.profile, workspace=args.workspace)
    root = Path(args.root).expanduser().resolve()
    output_path = reservation.destination
    invalid_responses: dict[str, object] = {}
    try:
        init_payload, train_records, evaluations, pins = _make_payloads(experiment_id, root)
        receipt["payloads"] = {"initialize": init_payload}
        if args.profile != core.PROFILE or args.workspace != core.WORKSPACE:
            raise ValueError("only the pinned personal Modal profile and workspace are allowed")
        if receipt["modal"]["sdk_version"] != modal_host.EXPECTED_MODAL_VERSION:
            raise RuntimeError("Modal 1.6.1 is required")
        if not args.execute:
            receipt["status"] = "dry_run"
        else:
            response_lock = Lock()

            def make_arm_payloads(initialization: object) -> Mapping[str, dict[str, object]]:
                return _make_arm_payloads(
                    experiment_id, train_records, evaluations, pins, initialization
                )

            def validate(raw: object, payload: dict[str, object]) -> dict[str, object]:
                return core.validate_result(raw, payload)

            lifecycle = modal_host.launch_modal(
                init_payload=init_payload,
                make_arm_payloads=make_arm_payloads,
                validate_result=validate,
                make_failed_result=_make_failed_result,
                execute_lifecycle=_execute_lifecycle,
                sanitize=_safe_error_message,
                profile=args.profile,
                workspace=args.workspace,
                root=root,
                receipt=receipt,
                invalid_responses=invalid_responses,
                response_lock=response_lock,
            )
            receipt["payloads"] = lifecycle["payloads"]
            receipt["results"] = lifecycle["results"]
            statuses = [
                result.get("status")
                for result in receipt["results"].values()
                if isinstance(result, dict)
            ]
            receipt["status"] = (
                "passed"
                if set(receipt["results"]) == {"initialize", *data.ARM_NAMES}
                and statuses == ["passed", "passed", "passed"]
                else "failed"
            )
            if receipt["status"] == "failed":
                receipt["failure"] = {
                    "stage": "lifecycle",
                    "type": "MixtureRunFailed",
                    "message": "initialization or one or more training arms failed",
                }
    except Exception as exc:
        _record_failure(receipt, "preflight_or_execution", exc)
    try:
        raw_path = _save_untrusted_response(output_path, invalid_responses)
        if raw_path is not None:
            receipt["modal"]["untrusted_response_path"] = str(raw_path)
    except Exception as exc:
        if receipt.get("failure") is None:
            _record_failure(receipt, "untrusted_response_persistence", exc)
        else:
            receipt["modal"]["untrusted_response_persistence_error"] = {
                "type": type(exc).__name__,
                "message": _safe_error_message(exc),
            }
    smoke.write_json_artifact(reservation, receipt)
    print(
        json.dumps(
            {"status": receipt["status"], "artifact": str(output_path)},
            sort_keys=True,
        )
    )
    if not args.execute:
        return 0 if receipt["status"] == "dry_run" else 1
    return 0 if receipt["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--profile", default=core.PROFILE)
    parser.add_argument("--workspace", default=core.WORKSPACE)
    parser.add_argument(
        "--execute", action="store_true", help="explicitly launch paid remote training"
    )
    args = parser.parse_args(argv)
    try:
        with smoke.reserve_output(args.output) as reservation:
            return _run(args, reservation)
    except (OSError, ValueError) as exc:
        print(
            f"could not reserve or write the requested receipt: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
