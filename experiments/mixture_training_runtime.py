"""Remote-only Qwen initialization and paired-training worker.

Importing this module is CPU-safe. Framework imports occur only after the payload,
protocol, and every uploaded source file have been verified inside ``remote_worker``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from experiments import mixture_training_core as core
from experiments import mixture_training_runtime_helpers as helpers
from experiments.mixture_training_contracts import ORIGINAL_PROTOCOL_PATH

VOLUME_ROOT = Path("/artifacts")
VOLUME_NAME = "reflex-rehearsal-artifacts"
_FORBIDDEN_KERNEL_PACKAGES = ("kernels", "fla", "causal_conv1d")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module_name(relative: str) -> tuple[str, str]:
    if relative == "experiments/__init__.py":
        return "experiments", "experiments/__init__.py"
    if relative.startswith("experiments/") and relative.endswith(".py"):
        return relative[:-3].replace("/", "."), relative
    if relative.startswith("src/reflex_decisions/") and relative.endswith(".py"):
        module_path = relative.removeprefix("src/")
        return module_path[:-3].replace("/", "."), relative
    raise ValueError(f"unsupported source fingerprint path: {relative}")


def _measure_remote_source_fingerprints() -> dict[str, str]:
    """Import allowlisted CPU modules and hash their actual uploaded ``__file__`` paths."""

    root = _project_root()
    measured: dict[str, str] = {}
    for relative in core.SOURCE_FINGERPRINT_PATHS:
        if relative in (core.PROTOCOL_PATH, ORIGINAL_PROTOCOL_PATH):
            path = root / relative
        else:
            module_name, uploaded_relative = _module_name(relative)
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                raise ValueError(f"source module has no file: {module_name}")
            path = Path(module_file).resolve(strict=True)
            expected_path = (root / uploaded_relative).resolve(strict=True)
            if path != expected_path:
                raise ValueError(
                    f"uploaded source module path differs from its allowlist: {relative}"
                )
        if not path.is_file():
            raise ValueError(f"allowlisted remote source is missing: {relative}")
        measured[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return measured


def _verify_remote_sources(payload: dict[str, object]) -> dict[str, str]:
    pins = payload["pins"]
    assert isinstance(pins, dict)
    expected = pins["source_file_sha256"]
    if not isinstance(expected, dict) or set(expected) != set(core.SOURCE_FINGERPRINT_PATHS):
        raise ValueError("payload source hashes do not match the explicit allowlist")
    measured = _measure_remote_source_fingerprints()
    if measured != expected:
        raise ValueError("remote source files differ from their pinned hashes")
    protocol_digest = core.verify_protocol(_project_root() / core.PROTOCOL_PATH)
    if protocol_digest != pins["protocol_sha256"]:
        raise ValueError("uploaded protocol hash differs from the payload pin")
    return measured


def _initial_result(payload: dict[str, object]) -> dict[str, object]:
    source_hashes = payload["pins"]["source_file_sha256"]
    counts = {
        "base_evaluation": 0,
        "training": 0,
        "final_evaluation": 0,
        "reload_parity": 0,
        "total": 0,
    }
    evidence: dict[str, object] = {
        "forward_counts": dict(counts),
        "input_token_counts": dict(counts),
        "outputs": [],
        "output_artifact": None,
        "initialization": payload["initialization"],
    }
    if payload["phase"] == "train":
        evidence.update(
            {
                "optimizer_updates_completed": 0,
                "training_step_losses": [],
                "adapter_inventory": None,
                "adapter_update": None,
                "base_gradients_none": None,
                "adapter_paths": {},
                "initialization_verified": False,
                "initial_tensor_sha256": None,
                "reload_parity": None,
            }
        )
    provenance: dict[str, object] = {
        "model_id": core.MODEL_ID,
        "model_revision": core.MODEL_REVISION,
        "versions": {},
        "source_file_sha256": source_hashes,
        "measured_source_file_sha256": {},
        "base_model": None,
        "cuda_device": None,
    }
    if payload["phase"] == "train":
        provenance["optimizer"] = {
            "name": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "max_gradient_norm": 1.0,
        }
    return helpers.build_worker_result(
        payload,
        schema_version=core.SCHEMA_VERSION,
        phase="preflight",
        status="failed",
        provenance=provenance,
        evidence=evidence,
        failure={
            "stage": "not_started",
            "type": "NotStarted",
            "message": "worker did not return a completion result",
        },
    )


def _runtime_packages() -> tuple[Any, Any, Any, Any, Any, dict[str, str]]:
    os.environ["USE_HUB_KERNELS"] = "NO"
    optional = {
        name: name in sys.modules or importlib.util.find_spec(name) is not None
        for name in _FORBIDDEN_KERNEL_PACKAGES
    }
    if any(optional.values()):
        raise RuntimeError("optional model-kernel packages must not be installed")
    versions = {name: importlib.metadata.version(name) for name in core.RUNTIME_VERSION_PINS}
    if versions != core.RUNTIME_VERSION_PINS:
        raise RuntimeError("remote package versions differ from the pinned runtime image")

    import torch
    import torch.nn.functional as functional
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return (
        torch,
        functional,
        transformers,
        (LoraConfig, PeftModel, get_peft_model),
        (AutoModelForCausalLM, AutoTokenizer),
        versions,
    )


def _write_progress(result: dict[str, object], run_dir: Path) -> None:
    helpers.write_json_atomic(run_dir / "progress.json", result, replace=True)
    training_helpers = importlib.import_module("experiments.modal_train_rehearsal")
    training_helpers._commit_volume()


def _safe_exception_message(error: BaseException) -> str:
    """Return a sanitized message without allowing the sanitizer to mask a failure."""

    try:
        from reflex_decisions.smoke import sanitize_exception_message

        return sanitize_exception_message(error)
    except Exception:
        return "worker failed; details could not be sanitized"


def _run_directory(run_id: str) -> Path:
    root = VOLUME_ROOT / "runs"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    run_dir.mkdir()
    return run_dir


def remote_worker(payload: dict[str, object]) -> dict[str, object]:
    """Validate, fingerprint, and run exactly one initialization or training phase."""

    normalized = core.validate_payload(payload)
    result = _initial_result(normalized)
    run_dir: Path | None = None
    stage = "source_verification"
    try:
        measured = _verify_remote_sources(normalized)
        result["provenance"]["measured_source_file_sha256"] = measured

        stage = "run_reservation"
        run_dir = _run_directory(str(normalized["run_id"]))
        result["phase"] = "run_reserved"
        _write_progress(result, run_dir)

        stage = "runtime_imports"
        torch, functional, _transformers, peft_items, model_items, versions = _runtime_packages()
        LoraConfig, PeftModel, get_peft_model = peft_items
        AutoModelForCausalLM, AutoTokenizer = model_items
        peft_module = importlib.import_module("peft")
        result["provenance"]["versions"] = versions
        execution = importlib.import_module("experiments.mixture_training_runtime_execution")

        stage = "base_model_load"
        if normalized["phase"] == "initialize":
            stage = "initialization"
            execution.run_initialization(
                normalized,
                result,
                run_dir,
                torch,
                AutoModelForCausalLM,
                AutoTokenizer,
                LoraConfig,
                get_peft_model,
                _write_progress,
            )
        else:
            stage = "training"
            execution.run_training(
                normalized,
                result,
                run_dir,
                torch,
                functional,
                AutoModelForCausalLM,
                AutoTokenizer,
                PeftModel,
                peft_module,
                _write_progress,
            )
        result["status"] = "passed"
        result["failure"] = None
        result["phase"] = str(normalized["phase"])
        _write_progress(result, run_dir)
    except Exception as exc:
        result["status"] = "failed"
        result["phase"] = stage
        result["failure"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": _safe_exception_message(exc),
        }
        if run_dir is not None:
            helpers.clean_partial_update(
                lambda: _write_progress(result, run_dir), original_error=exc
            )
    return result
