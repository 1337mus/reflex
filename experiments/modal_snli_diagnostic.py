"""Plan or explicitly launch the bounded balanced-SNLI Modal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from experiments import modal_smoke
from experiments import snli_diagnostic_core as core
from experiments import snli_diagnostic_runtime as runtime
from reflex_decisions import smoke
from reflex_decisions.data import (
    DecisionRecord,
    SplitManifest,
    audit_splits,
    load_manifest,
    load_records,
)

PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
VOLUME_NAME = "reflex-rehearsal-artifacts"
VOLUME_ROOT = "/artifacts"
EXPECTED_MODAL_VERSION = "1.6.1"


def plan() -> dict[str, object]:
    """Describe fixed limits without importing Modal, Torch, or model adapters."""

    return {
        "mode": "plan-only",
        "models": list(core.MODEL_NAMES),
        "workers": list(core.WORKERS),
        "records": 192,
        "source_groups": 64,
        "presentations_per_model": 1152,
        "scored_forwards": 4608,
        "auxiliary_forwards": 3,
        "max_total_forwards": 4611,
        "modal": _function_options(),
        "profile": PROFILE,
        "workspace": WORKSPACE,
        "adapter_path": runtime.ADAPTER_PATH,
        "adapter_file_sha256": dict(runtime.ADAPTER_FILE_SHA256),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="explicitly launch paid inference")
    parser.add_argument("--records", help="prepared balanced-SNLI JSONL records")
    parser.add_argument("--manifest", help="prepared balanced-SNLI manifest")
    parser.add_argument("--recipe", help="prepared balanced-SNLI selection recipe")
    parser.add_argument("--output", help="new local receipt path; existing paths are refused")
    args = parser.parse_args(argv)
    if not args.launch:
        print(json.dumps(plan(), sort_keys=True, indent=2))
        return 0
    missing = [
        name for name in ("records", "manifest", "recipe", "output") if not getattr(args, name)
    ]
    if missing:
        parser.error("--launch requires --records, --manifest, --recipe, and --output")
    try:
        with smoke.reserve_output(args.output) as reservation:
            receipt = _launch(args)
            smoke.write_json_artifact(reservation, receipt)
        print(json.dumps({"status": receipt["status"], "artifact": str(reservation.destination)}))
        return 0 if receipt["status"] == "passed" else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "failure": {
                        "type": type(exc).__name__,
                        "message": smoke.sanitize_exception_message(exc),
                    },
                }
            ),
            file=sys.stderr,
        )
        return 1


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_panel(
    manifest: SplitManifest,
    records: tuple[DecisionRecord, ...],
    recipe: object,
    *,
    records_sha256: str,
    manifest_sha256: str,
) -> None:
    from reflex_decisions import snli_diagnostic

    if manifest.data_kind != "benchmark" or manifest.held_out_families:
        raise ValueError("balanced SNLI manifest must be a non-held-out benchmark")
    if len(manifest.datasets) != 1:
        raise ValueError("balanced SNLI manifest must contain exactly one dataset")
    dataset = manifest.datasets[0]
    if dataset.dataset_id != snli_diagnostic.DATASET_ID or dataset.split != "development":
        raise ValueError("balanced SNLI manifest identity or split is incorrect")
    if len(records) != 192 or len({record.record_id for record in records}) != 192:
        raise ValueError("balanced SNLI panel must contain exactly 192 unique records")
    audit_splits(manifest, records)
    groups: dict[str, list[DecisionRecord]] = {}
    for record in records:
        if record.dataset_id != dataset.dataset_id:
            raise ValueError("balanced SNLI record references an unknown dataset")
        if tuple(option.id for option in record.request.options) != snli_diagnostic.SNLI_LABELS:
            raise ValueError("balanced SNLI request must preserve its approved semantic order")
        if record.answer_id not in snli_diagnostic.SNLI_LABELS:
            raise ValueError("balanced SNLI answer is outside the three approved labels")
        groups.setdefault(record.source_group_id, []).append(record)
    if len(groups) != 64 or any(
        Counter(row.answer_id for row in members)
        != Counter({label: 1 for label in snli_diagnostic.SNLI_LABELS})
        for members in groups.values()
    ):
        raise ValueError("balanced SNLI panel must have one example per label in each of 64 groups")
    if not isinstance(recipe, dict) or recipe.get("dataset_id") != snli_diagnostic.DATASET_ID:
        raise ValueError("balanced SNLI selection recipe is malformed")
    outputs = recipe.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("records_sha256") != records_sha256:
        raise ValueError("selection recipe does not pin the supplied records file")
    if outputs.get("manifest_sha256") != manifest_sha256:
        raise ValueError("selection recipe does not pin the supplied manifest file")
    selection = recipe.get("selection")
    selected = selection.get("selected_groups") if isinstance(selection, dict) else None
    if not isinstance(selected, list) or len(selected) != 64:
        raise ValueError("selection recipe must identify exactly 64 source groups")
    record_by_id = {record.record_id: record for record in records}
    recipe_ids: set[str] = set()
    for group in selected:
        if not isinstance(group, dict) or not isinstance(group.get("source_group_id"), str):
            raise ValueError("selection recipe contains a malformed group")
        group_id = group["source_group_id"]
        members = group.get("members")
        if group_id in recipe_ids or not isinstance(members, list) or len(members) != 3:
            raise ValueError("selection recipe contains duplicate or incomplete groups")
        recipe_ids.add(group_id)
        member_labels: set[str] = set()
        for member in members:
            if not isinstance(member, dict):
                raise ValueError("selection recipe contains a malformed member")
            pair_id, label = member.get("pair_id"), member.get("label")
            if not isinstance(pair_id, str) or label not in snli_diagnostic.SNLI_LABELS:
                raise ValueError("selection recipe member identity is malformed")
            record_id = (
                f"{snli_diagnostic.DATASET_ID}-{hashlib.sha256(pair_id.encode()).hexdigest()}"
            )
            record = record_by_id.get(record_id)
            if record is None or record.source_group_id != group_id or record.answer_id != label:
                raise ValueError("selection recipe member does not match the supplied panel")
            member_labels.add(label)
        if member_labels != set(snli_diagnostic.SNLI_LABELS):
            raise ValueError("selection recipe group must contain all three semantic labels")
    if recipe_ids != set(groups):
        raise ValueError("selection recipe source groups differ from the supplied panel")


def _load_payload(args: argparse.Namespace, root: Path) -> dict[str, object]:
    records_path = root / args.records
    manifest_path = root / args.manifest
    recipe_path = root / args.recipe
    records_bytes, manifest_bytes, recipe_bytes = (
        records_path.read_bytes(),
        manifest_path.read_bytes(),
        recipe_path.read_bytes(),
    )
    try:
        recipe = json.loads(recipe_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("balanced SNLI selection recipe is not valid JSON") from exc
    records_sha256 = hashlib.sha256(records_bytes).hexdigest()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest, records = load_manifest(manifest_path), load_records(records_path)
    _validate_panel(
        manifest,
        records,
        recipe,
        records_sha256=records_sha256,
        manifest_sha256=manifest_sha256,
    )
    protocol_path = root / core.PROTOCOL_PATH
    protocol_sha256 = _sha256_file(protocol_path)
    verify_protocol = getattr(core, "verify_protocol", None)
    if callable(verify_protocol):
        verified_protocol_sha256 = verify_protocol(protocol_path)
        if verified_protocol_sha256 != protocol_sha256:
            raise ValueError("protocol changed while launch preflight was running")
    elif protocol_sha256 != core.EXPECTED_PROTOCOL_SHA256:
        raise ValueError("balanced SNLI protocol SHA-256 mismatch")
    pins = {
        "records_sha256": records_sha256,
        "manifest_sha256": manifest_sha256,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "protocol_sha256": protocol_sha256,
        "source_file_sha256": core.source_fingerprints(root),
    }
    run_id, nonce = str(uuid.uuid4()), str(uuid.uuid4())
    while nonce == run_id:
        nonce = str(uuid.uuid4())
    payload = core.build_payload(records, pins=pins, run_id=run_id, nonce=nonce)
    normalized = core.validate_payload(payload)
    if (
        _sha256_file(records_path) != records_sha256
        or _sha256_file(manifest_path) != manifest_sha256
    ):
        raise ValueError("balanced SNLI input files changed during launch preflight")
    if _sha256_file(recipe_path) != pins["recipe_sha256"]:
        raise ValueError("balanced SNLI selection recipe changed during launch preflight")
    return normalized


def _function_options() -> dict[str, object]:
    return {
        "gpu": "A10",
        "cpu": (2.0, 2.0),
        "memory": (16384, 16384),
        "max_containers": 3,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window": 2,
        "retries": 0,
        "single_use_containers": True,
        "serialized": False,
        "include_source": False,
        "startup_timeout": 300,
        "timeout": 1800,
    }


def _remote_file_path(relative: str) -> str:
    if relative.startswith("src/reflex_decisions/"):
        relative = relative.removeprefix("src/")
    return f"/root/{relative}"


def _build_modal_image(modal: Any, project_root: Path) -> Any:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .env(
            {
                "PYTHONPATH": "/root",
                "USE_HUB_KERNELS": "NO",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            }
        )
        .pip_install(
            *(f"{package}=={version}" for package, version in runtime.RUNTIME_VERSION_PINS.items()),
            extra_index_url="https://download.pytorch.org/whl/cu130",
        )
    )
    for relative in core.SOURCE_FINGERPRINT_PATHS:
        image = image.add_local_file(
            str(project_root / relative), remote_path=_remote_file_path(relative)
        )
    return image


def _collect_worker_calls(
    function: Any, payload: dict[str, object]
) -> tuple[dict[str, object], dict[str, Exception]]:
    calls: dict[str, Any] = {}
    results: dict[str, object] = {}
    failures: dict[str, Exception] = {}
    for worker in core.WORKERS:
        try:
            calls[worker] = function.spawn(payload, worker)
        except Exception as exc:
            failures[worker] = exc
    for worker in core.WORKERS:
        call = calls.get(worker)
        if call is None:
            continue
        try:
            results[worker] = call.get()
        except Exception as exc:
            failures[worker] = exc
    return results, failures


def _failed_model_result(
    payload: dict[str, object], model_name: str, stage: str, error: BaseException
) -> dict[str, object]:
    return runtime.failed_model_result(payload, model_name, stage, error)


def _collect_model_results(
    payload: dict[str, object], function: Any
) -> dict[str, dict[str, object]]:
    raw_results, failures = _collect_worker_calls(function, payload)
    models: dict[str, dict[str, object]] = {}
    model_to_worker = {
        model: worker for worker, names in runtime.WORKER_MODELS.items() for model in names
    }
    for model_name in core.MODEL_NAMES:
        worker = model_to_worker[model_name]
        raw_worker = raw_results.get(worker)
        raw_models = raw_worker.get("models") if isinstance(raw_worker, dict) else None
        if isinstance(raw_models, dict) and model_name in raw_models:
            try:
                models[model_name] = core.validate_model_result(
                    raw_models[model_name], payload, model_name
                )
                continue
            except Exception as exc:
                models[model_name] = _failed_model_result(
                    payload, model_name, "result_validation", exc
                )
                continue
        error = failures.get(worker, RuntimeError("worker returned no model result"))
        models[model_name] = _failed_model_result(payload, model_name, "worker_execution", error)
    return models


def _receipt(
    payload: dict[str, object],
    models: dict[str, dict[str, object]],
    *,
    modal_version: str | None,
    lifecycle_failure: BaseException | None = None,
) -> dict[str, object]:
    statuses = [models[name]["status"] for name in core.MODEL_NAMES]
    status = (
        "passed"
        if all(value == "passed" for value in statuses)
        else ("partial" if any(value == "passed" for value in statuses) else "failed")
    )
    failure = runtime._failure("modal_lifecycle", lifecycle_failure) if lifecycle_failure else None
    if failure is not None:
        status = "failed"
    receipt: dict[str, object] = {
        "schema_version": core.SCHEMA_VERSION,
        "status": status,
        "payload": payload,
        "models": models,
        "provenance": {
            "modal_profile": PROFILE,
            "workspace": WORKSPACE,
            "modal_sdk_version": modal_version,
            "limits": core.MODAL_LIMITS,
        },
    }
    if failure:
        receipt["failure"] = failure
    return core.validate_receipt(receipt, payload)


def _failed_preflight(error: BaseException, stage: str) -> dict[str, object]:
    return {
        "schema_version": core.SCHEMA_VERSION,
        "status": "failed",
        "payload": None,
        "models": {name: None for name in core.MODEL_NAMES},
        "provenance": {
            "modal_profile": PROFILE,
            "workspace": WORKSPACE,
            "modal_sdk_version": None,
            "limits": core.MODAL_LIMITS,
        },
        "failure": runtime._failure(stage, error),
    }


def _launch(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(__file__).resolve().parents[1]
    payload: dict[str, object] | None = None
    modal_version: str | None = None
    stage = "data_preflight"
    try:
        payload = _load_payload(args, project_root)
        stage = "credential_preflight"
        if any(os.environ.get(name, "") for name in modal_smoke.CREDENTIAL_OVERRIDES):
            raise RuntimeError("Modal credential environment overrides are not accepted")
        stage = "profile_verification"
        if not modal_smoke._verify_profile(PROFILE, WORKSPACE):
            raise RuntimeError("Modal token info did not match the fixed personal workspace")
        stage = "modal_version"
        modal_version = importlib.metadata.version("modal")
        if modal_version != EXPECTED_MODAL_VERSION:
            raise RuntimeError("Modal SDK 1.6.1 is required")
        stage = "modal_setup"
        modal = importlib.import_module("modal")
        volume = modal.Volume.from_name(VOLUME_NAME).with_mount_options(read_only=True)
        image = _build_modal_image(modal, project_root)
        app = modal.App("reflex-balanced-snli-diagnostic", image=image)
        run = app.function(**_function_options(), volumes={VOLUME_ROOT: volume})(
            runtime.remote_worker
        )
        stage = "worker_execution"
        lifecycle_failure: BaseException | None = None
        models: dict[str, dict[str, object]] | None = None
        try:
            with modal.enable_output(), app.run():
                models = _collect_model_results(payload, run)
        except Exception as exc:
            lifecycle_failure = exc
            models = models or {
                name: _failed_model_result(payload, name, "modal_lifecycle", exc)
                for name in core.MODEL_NAMES
            }
        return _receipt(
            payload, models, modal_version=modal_version, lifecycle_failure=lifecycle_failure
        )
    except Exception as exc:
        if payload is None:
            return _failed_preflight(exc, stage)
        models = {
            name: _failed_model_result(payload, name, stage, exc) for name in core.MODEL_NAMES
        }
        return _receipt(payload, models, modal_version=modal_version)


if __name__ == "__main__":
    sys.exit(main())
