"""Plan or explicitly launch the bounded Intern/Kev real-pilot comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from experiments import baseline_runner_core, modal_smoke
from experiments import real_pilot_baseline_core as core
from reflex_decisions import pilot_data, smoke
from reflex_decisions.schema import DecisionRequest

PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
RUNTIME_VERSION_PINS = {
    "torch": "2.14.1+cu130",
    "torchvision": "0.29.1+cu130",
    "transformers": "5.18.0",
    "peft": "0.21.0",
    "Pillow": "12.0.0",
    "pydantic": "2.13.5",
    "huggingface-hub": "1.33.0",
    "tokenizers": "0.23.2",
    "safetensors": "0.8.0",
}


def plan() -> dict[str, object]:
    """Describe the fixed reference run without importing Modal or Torch."""

    return {
        "mode": "plan-only",
        "models": ["intern", "kev"],
        "model_pins": core.MODEL_PINS,
        "presentations_per_model": core.EVALUATION_PRESENTATION_COUNT,
        "max_total_forwards": core.MODAL_LIMITS["max_total_forwards"],
        "modal": core.MODAL_LIMITS,
    }


def main(argv: list[str] | None = None) -> int:
    """Plan by default; paid remote scoring requires an explicit launch flag."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="explicitly launch both workers")
    parser.add_argument("--output", help="new JSON receipt destination; existing paths are refused")
    parser.add_argument("--records", default=pilot_data.DEFAULT_RECORDS_PATH)
    parser.add_argument("--manifest", default=pilot_data.DEFAULT_MANIFEST_PATH)
    parser.add_argument("--recipe", default=pilot_data.DEFAULT_RECIPE_PATH)
    args = parser.parse_args(argv)
    if not args.launch:
        print(json.dumps(plan(), sort_keys=True, indent=2))
        return 0
    if not args.output:
        parser.error("--launch requires --output for a fresh local receipt")
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


def _canonical_module_key(path: str | Path) -> str:
    resolved = Path(path).resolve()
    for remote_root, canonical_root in (
        (Path("/root/experiments"), "experiments"),
        (Path("/root/reflex_decisions"), "src/reflex_decisions"),
    ):
        try:
            relative = resolved.relative_to(remote_root)
        except ValueError:
            continue
        if relative.suffix == ".py":
            return f"{canonical_root}/{relative.as_posix()}"
    raise ValueError("remote reference source resolved outside approved package roots")


def _measure_remote_source_fingerprints() -> dict[str, str]:
    measured: dict[str, str] = {}
    for module_name, expected_path in core.SOURCE_FINGERPRINT_MODULES:
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str):
            raise ValueError(f"remote source module has no file: {module_name}")
        canonical = _canonical_module_key(module_path)
        if canonical != expected_path or canonical in measured:
            raise ValueError(f"remote source path did not match its pinned key: {module_name}")
        measured[canonical] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    for relative in (core.real_pilot_core.PROTOCOL_PATH, core.REFERENCE_PROTOCOL_PATH):
        path = Path("/root") / relative
        if not path.is_file():
            raise ValueError("a pinned reference protocol is missing from the worker image")
        measured[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return measured


def _verify_remote_source_fingerprints(
    expected: object, measured: dict[str, str] | None = None
) -> dict[str, str]:
    if (
        not isinstance(expected, dict)
        or set(expected) != set(core.SOURCE_FINGERPRINT_PATHS)
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for path, digest in expected.items()
        )
    ):
        raise ValueError("reference source fingerprint payload is malformed")
    actual = measured if measured is not None else _measure_remote_source_fingerprints()
    if set(actual) != set(core.SOURCE_FINGERPRINT_PATHS) or actual != expected:
        raise ValueError("remote reference source fingerprints differ from the pinned payload")
    return actual


def _load_scorer(model_name: str) -> tuple[Any, Any, dict[str, object]]:
    modules = {
        "intern": "experiments.baseline_intern",
        "kev": "experiments.baseline_kev",
    }
    if model_name not in modules:
        raise ValueError("reference model is outside the frozen allowlist")
    adapter = importlib.import_module(modules[model_name])
    scorer, provenance = adapter.load_scorer()
    return adapter, scorer, provenance


def _runtime_versions() -> dict[str, str]:
    versions = {package: importlib.metadata.version(package) for package in RUNTIME_VERSION_PINS}
    if versions != RUNTIME_VERSION_PINS:
        raise RuntimeError("remote model runtime versions differ from the frozen image")
    return versions


def _function_options() -> dict[str, object]:
    """Return the fixed no-retry, two-container remote worker limits."""

    return {
        "gpu": "A10",
        "cpu": (2.0, 2.0),
        "memory": (16384, 16384),
        "max_containers": 2,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window": 2,
        "retries": 0,
        "single_use_containers": True,
        "serialized": False,
        "include_source": False,
        "startup_timeout": 300,
        "timeout": 3600,
    }


def _collect_worker_calls(
    function: Any, payload: dict[str, object]
) -> tuple[dict[str, object], dict[str, Exception]]:
    """Spawn both one-shot jobs before waiting, and preserve independent failures."""

    calls: dict[str, Any] = {}
    results: dict[str, object] = {}
    failures: dict[str, Exception] = {}
    for model_name in ("intern", "kev"):
        try:
            calls[model_name] = function.spawn(payload, model_name)
        except Exception as exc:
            failures[model_name] = exc
    for model_name in ("intern", "kev"):
        call = calls.get(model_name)
        if call is None:
            continue
        try:
            results[model_name] = call.get()
        except Exception as exc:
            failures[model_name] = exc
    return results, failures


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
            "torch==2.14.1+cu130",
            "torchvision==0.29.1+cu130",
            "transformers==5.18.0",
            "peft==0.21.0",
            "pillow==12.0.0",
            "pydantic==2.13.5",
            "huggingface-hub==1.33.0",
            "tokenizers==0.23.2",
            "safetensors==0.8.0",
            extra_index_url="https://download.pytorch.org/whl/cu130",
        )
    )
    for relative in core.SOURCE_FINGERPRINT_PATHS:
        image = image.add_local_file(
            str(project_root / relative), remote_path=_remote_file_path(relative)
        )
    return image


def _persisted_model_state(
    value: object, *, expected_payload: dict[str, object], model_name: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("worker returned a non-object model receipt")
    if value.get("status") == "passed":
        state = core.validate_model_receipt(
            value, expected_payload=expected_payload, model_name=model_name
        )
    elif value.get("status") == "failed":
        state = core.validate_failed_model_receipt(
            value, expected_payload=expected_payload, model_name=model_name
        )
    else:
        raise ValueError("worker receipt has no terminal model status")
    return {
        key: item
        for key, item in state.items()
        if key not in {"schema_version", "run_id", "nonce", "pins"}
    }


def _failed_model_state(
    payload: dict[str, object], model_name: str, stage: str, error: BaseException
) -> dict[str, object]:
    receipt = _initial_model_receipt(payload, model_name)
    receipt["failure"] = {
        "stage": stage,
        "type": type(error).__name__,
        "message": smoke.sanitize_exception_message(error),
    }
    state = core.validate_failed_model_receipt(
        receipt, expected_payload=payload, model_name=model_name
    )
    return {
        key: item
        for key, item in state.items()
        if key not in {"schema_version", "run_id", "nonce", "pins"}
    }


def _collect_model_states(
    function: Any, payload: dict[str, object]
) -> dict[str, dict[str, object]]:
    results, failures = _collect_worker_calls(function, payload)
    states: dict[str, dict[str, object]] = {}
    for model_name in ("intern", "kev"):
        if model_name not in results:
            error = failures.get(model_name, RuntimeError("worker returned no receipt"))
            states[model_name] = _failed_model_state(payload, model_name, "worker_execution", error)
            continue
        try:
            states[model_name] = _persisted_model_state(
                results[model_name], expected_payload=payload, model_name=model_name
            )
        except Exception as exc:
            states[model_name] = _failed_model_state(payload, model_name, "receipt_validation", exc)
    return states


def _envelope(
    payload: dict[str, object],
    states: dict[str, dict[str, object]],
    *,
    modal_version: str | None,
    failure: dict[str, str] | None = None,
) -> dict[str, object]:
    model_statuses = {name: states[name].get("status") for name in ("intern", "kev")}
    status = (
        "passed"
        if all(value == "passed" for value in model_statuses.values())
        else "partial"
        if any(value == "passed" for value in model_statuses.values())
        else "failed"
    )
    if failure is not None:
        status = "failed"
    envelope: dict[str, object] = {
        "schema_version": core.SCHEMA_VERSION,
        "status": status,
        "payload": payload,
        "models": states,
        "provenance": {
            "profile": PROFILE,
            "workspace": WORKSPACE,
            "modal_sdk_version": modal_version,
            "hf_hub_disable_implicit_token": True,
        },
        "limits": core.MODAL_LIMITS,
    }
    if failure is not None:
        envelope["failure"] = failure
    return envelope


def _failed_preflight(error: BaseException, stage: str) -> dict[str, object]:
    return {
        "schema_version": core.SCHEMA_VERSION,
        "status": "failed",
        "payload": None,
        "models": {"intern": None, "kev": None},
        "provenance": {
            "profile": PROFILE,
            "workspace": WORKSPACE,
            "modal_sdk_version": None,
            "hf_hub_disable_implicit_token": True,
        },
        "limits": core.MODAL_LIMITS,
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": smoke.sanitize_exception_message(error),
        },
    }


def _launch(args: argparse.Namespace) -> dict[str, object]:
    """Verify the pinned local inputs, then run only the two approved one-shot jobs."""

    project_root = Path(__file__).resolve().parents[1]
    payload: dict[str, object] | None = None
    modal_version: str | None = None
    worker_states: dict[str, dict[str, object]] | None = None
    stage = "data_preflight"
    try:
        run_id, nonce = str(uuid.uuid4()), str(uuid.uuid4())
        while run_id == nonce:
            nonce = str(uuid.uuid4())
        records_path = project_root / args.records
        manifest_path = project_root / args.manifest
        recipe_path = project_root / args.recipe
        manifest, records, _recipe = pilot_data.verify_prepared_data(
            records_path, manifest_path, recipe_path
        )
        pilot_protocol = core.real_pilot_core.verify_protocol(
            project_root / core.real_pilot_core.PROTOCOL_PATH
        )
        reference_protocol = core.verify_reference_protocol(
            project_root / core.REFERENCE_PROTOCOL_PATH
        )
        source_hashes = core.source_fingerprints(project_root)
        payload = core.build_evaluation_payload(
            manifest,
            records,
            run_id=run_id,
            nonce=nonce,
            records_sha256=hashlib.sha256(records_path.read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            recipe_sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            protocol_sha256=pilot_protocol,
            reference_protocol_sha256=reference_protocol,
            source_file_sha256=source_hashes,
        )
        stage = "credential_preflight"
        if any(os.environ.get(name, "") for name in modal_smoke.CREDENTIAL_OVERRIDES):
            raise RuntimeError("Modal credential environment overrides are not accepted")
        stage = "profile_verification"
        if not modal_smoke._verify_profile(PROFILE, WORKSPACE):
            raise RuntimeError("Modal token info did not match the fixed personal workspace")
        stage = "modal_version"
        modal_version = importlib.metadata.version("modal")
        if modal_version != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")
        stage = "modal_setup"
        modal = importlib.import_module("modal")
        image = _build_modal_image(modal, project_root)
        app = modal.App("reflex-real-pilot-reference-baselines", image=image)
        score = app.function(**_function_options())(_remote_score)
        stage = "worker_execution"
        with modal.enable_output(), app.run():
            worker_states = _collect_model_states(score, payload)
        return _envelope(payload, worker_states, modal_version=modal_version)
    except Exception as exc:
        if payload is None:
            return _failed_preflight(exc, stage)
        failure = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": smoke.sanitize_exception_message(exc),
        }
        states = worker_states or {}
        for name in ("intern", "kev"):
            if name not in states:
                states[name] = _failed_model_state(payload, name, stage, exc)
        return _envelope(payload, states, modal_version=modal_version, failure=failure)


def _score_presentation(presentation: dict[str, object], scorer: Any) -> dict[str, object]:
    request = DecisionRequest.model_validate(presentation["request"])
    scored = scorer(request)
    option_ids = presentation["order_ids"]
    if not isinstance(option_ids, list):
        raise ValueError("reference order IDs are malformed")
    validated = baseline_runner_core.validate_presentation_result(
        {
            "record_id": presentation["record_id"],
            "request_hash": presentation["request_hash"],
            "permutation_index": presentation["order_index"],
            "option_ids": option_ids,
        },
        scored,
    )
    logits = validated["raw_logits"]
    winner = min(
        option_id
        for option_id, logit in zip(option_ids, logits, strict=True)
        if logit == max(logits)
    )
    return {
        "presentation_id": presentation["presentation_id"],
        "record_id": presentation["record_id"],
        "dataset_id": presentation["dataset_id"],
        "request_hash": presentation["request_hash"],
        "order_index": presentation["order_index"],
        "order_ids": list(option_ids),
        "candidate_logits": logits,
        "winner_option_id": winner,
        "input_tokens": validated["input_tokens"],
        "prompt_sha256": validated["prompt_sha256"],
    }


def _initial_model_receipt(payload: dict[str, object], model_name: str) -> dict[str, object]:
    model = core.MODEL_PINS[model_name]
    return {
        "schema_version": core.SCHEMA_VERSION,
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "pins": payload["pins"],
        "model_name": model_name,
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "status": "failed",
        "forward_counts": {"scored": 0, "auxiliary": None, "total": None},
        "prompt_hash_kind": core.PROMPT_HASH_KINDS[model_name],
        "provenance": {
            "measured_source_file_sha256": {},
            "scorer": {},
            "runtime_versions": {},
            "use_hub_kernels": None,
            "hf_hub_disable_implicit_token": None,
        },
        "presentations": [],
        "failure": {
            "stage": "not_started",
            "type": "NotStarted",
            "message": "reference scoring did not complete",
        },
    }


def _refresh_forward_counts(receipt: dict[str, object], model_name: str) -> None:
    presentations = receipt["presentations"]
    if not isinstance(presentations, list):
        raise ValueError("reference receipt presentations are malformed")
    provenance = receipt["provenance"]
    scorer = provenance.get("scorer") if isinstance(provenance, dict) else None
    auxiliary = (
        scorer.get("auxiliary_forward_count")
        if isinstance(scorer, dict) and type(scorer.get("auxiliary_forward_count")) is int
        else None
    )
    if auxiliary is not None and auxiliary not in {
        0,
        core.AUXILIARY_FORWARD_COUNTS[model_name],
    }:
        auxiliary = None
    scored = len(presentations)
    receipt["forward_counts"] = {
        "scored": scored,
        "auxiliary": auxiliary,
        "total": scored + auxiliary if auxiliary is not None else None,
    }


def _remote_score(payload: dict[str, object], model_name: str) -> dict[str, object]:
    """Run one source-verified model worker in a fresh Modal container."""

    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["USE_HUB_KERNELS"] = "NO"
    normalized = core.validate_remote_payload(payload)
    if model_name not in core.MODEL_PINS:
        raise ValueError("reference model is outside the frozen allowlist")
    receipt = _initial_model_receipt(normalized, model_name)
    receipt["provenance"]["use_hub_kernels"] = "NO"
    receipt["provenance"]["hf_hub_disable_implicit_token"] = True
    stage = "source_verification"
    try:
        measured = _measure_remote_source_fingerprints()
        receipt["provenance"]["measured_source_file_sha256"] = measured
        _verify_remote_source_fingerprints(normalized["pins"]["source_file_sha256"], measured)
        stage = "runtime_check"
        receipt["provenance"]["runtime_versions"] = _runtime_versions()
        stage = "model_loading"
        adapter, scorer, scorer_provenance = _load_scorer(model_name)
        core._validate_scorer_identity(model_name, scorer_provenance)
        receipt["provenance"]["scorer"] = scorer_provenance
        _refresh_forward_counts(receipt, model_name)
        del adapter
        stage = "inference"
        for presentation in normalized["evaluation_presentations"]:
            receipt["presentations"].append(_score_presentation(presentation, scorer))
            _refresh_forward_counts(receipt, model_name)
        stage = "receipt_validation"
        receipt["status"] = "passed"
        del receipt["failure"]
        core.validate_model_receipt(receipt, expected_payload=normalized, model_name=model_name)
    except Exception as exc:
        _refresh_forward_counts(receipt, model_name)
        receipt["status"] = "failed"
        receipt["failure"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": smoke.sanitize_exception_message(exc),
        }
        core.validate_failed_model_receipt(
            receipt, expected_payload=normalized, model_name=model_name
        )
    return receipt


if __name__ == "__main__":
    sys.exit(main())
