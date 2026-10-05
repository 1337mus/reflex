"""Remote-only inference worker for the pinned adapter transfer evaluation."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from experiments import adapter_transfer_contracts as contracts
from experiments import adapter_transfer_core as core
from experiments import mixture_training_contracts as mixture_contracts
from reflex_decisions import smoke

VOLUME_ROOT = Path("/artifacts")
VOLUME_NAME = "reflex-rehearsal-artifacts"
_FORBIDDEN_KERNEL_PACKAGES = ("kernels", "fla", "causal_conv1d")
_COUNT_KEYS = ("base_evaluation", "training", "final_evaluation", "reload_parity", "total")


def remote_worker(payload: dict[str, object]) -> dict[str, object]:
    """Validate and execute one adapter transfer evaluation on the remote GPU."""

    normalized = core.validate_payload(payload)
    result = _initial_result(normalized)
    stage = "source_verification"
    try:
        measured = _verify_remote_sources(normalized)
        result["provenance"]["measured_source_file_sha256"] = measured
        stage = "runtime_imports"
        runtime_items = _runtime_packages()
        result["provenance"]["versions"] = runtime_items[-1]
        stage = "inference"
        _run_evaluation(normalized, result, runtime_items)
        result["status"] = "passed"
        result["phase"] = "completed"
        result["failure"] = None
    except Exception as exc:
        failed = _build_failed_result(
            normalized,
            stage=stage,
            error=exc,
            provenance=result.get("provenance"),
            evidence=result.get("evidence"),
        )
        return core.validate_result(failed, normalized)
    try:
        return core.validate_result(result, normalized)
    except Exception as exc:
        failed = _build_failed_result(
            normalized,
            stage="result_validation",
            error=exc,
            provenance=result.get("provenance"),
            evidence=result.get("evidence"),
        )
        return core.validate_result(failed, normalized)


def _build_failed_result(
    payload: dict[str, object],
    *,
    stage: str,
    error: BaseException,
    provenance: object,
    evidence: object,
) -> dict[str, object]:
    try:
        return core.build_failed_result(
            payload,
            stage=stage,
            error=error,
            sanitize=smoke.sanitize_exception_message,
            provenance=provenance,
            evidence=evidence,
        )
    except Exception:
        initial = _initial_result(payload)
        if isinstance(evidence, dict):
            try:
                return core.build_failed_result(
                    payload,
                    stage=stage,
                    error=error,
                    sanitize=smoke.sanitize_exception_message,
                    provenance=initial["provenance"],
                    evidence=evidence,
                )
            except Exception:
                pass
        return core.build_failed_result(
            payload,
            stage=stage,
            error=error,
            sanitize=smoke.sanitize_exception_message,
            provenance=initial["provenance"],
            evidence=None,
        )


def _verify_remote_sources(payload: dict[str, object]) -> dict[str, str]:
    expected = payload["pins"]["source_file_sha256"]
    measured = core.source_fingerprints(Path(__file__).resolve().parents[1])
    if not isinstance(expected, dict) or measured != expected:
        raise ValueError("remote source files differ from their pinned hashes")
    return measured


def _runtime_packages() -> tuple[Any, ...]:
    os.environ["USE_HUB_KERNELS"] = "NO"
    optional = {
        name: name in sys.modules or importlib.util.find_spec(name) is not None
        for name in _FORBIDDEN_KERNEL_PACKAGES
    }
    if any(optional.values()):
        raise RuntimeError("optional model-kernel packages must not be installed")
    pins = mixture_contracts.RUNTIME_VERSION_PINS
    versions = {name: importlib.metadata.version(name) for name in pins}
    if versions != pins:
        raise RuntimeError("remote package versions differ from the pinned runtime image")

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return (
        torch,
        transformers,
        PeftModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        importlib.import_module("peft"),
        versions,
    )


def _initial_result(payload: dict[str, object]) -> dict[str, object]:
    counts = {key: 0 for key in _COUNT_KEYS}
    identity = {
        key: payload[key]
        for key in (
            "schema_version",
            "experiment_id",
            "run_id",
            "nonce",
            "selection_sha256",
            "payload_sha256",
        )
    }
    return {
        **identity,
        "status": "failed",
        "phase": "preflight",
        "provenance": {
            "model_id": contracts.MODEL_ID,
            "model_revision": contracts.MODEL_REVISION,
            "versions": {},
            "source_file_sha256": payload["pins"]["source_file_sha256"],
            "measured_source_file_sha256": {},
            "base_model": None,
            "cuda_device": None,
            "reload_tokenizer_file_sha256": None,
        },
        "evidence": {
            "forward_counts": dict(counts),
            "input_token_counts": dict(counts),
            "outputs": {"base": [], "adapter": []},
            "adapter_identity": None,
            "reload_parity": {"outputs": [], "max_candidate_logit_delta": None},
        },
        "failure": {
            "stage": "not_started",
            "type": "NotStarted",
            "message": "worker did not return a completion result",
        },
    }


def _run_evaluation(
    payload: dict[str, object], result: dict[str, object], runtime_items: tuple[Any, ...]
) -> None:
    torch, _transformers, PeftModel, AutoModelForCausalLM, AutoTokenizer, peft_module, _versions = (
        runtime_items
    )
    execution = importlib.import_module("experiments.mixture_training_runtime_execution")
    scoring = importlib.import_module("experiments.mixture_training_runtime_scoring")
    helpers = importlib.import_module("experiments.mixture_training_runtime_helpers")
    snapshot = payload["selection"]["snapshot"]
    saved_state, tensor_digest, files_sha256 = _load_saved_adapter_state(snapshot, torch)
    evidence = result["evidence"]
    evidence["adapter_identity"] = {
        "snapshot_path": snapshot["path"],
        "files_sha256": files_sha256,
        "tensor_sha256": tensor_digest,
        "reloaded_tensor_sha256": None,
        "dtype": "torch.float32",
    }
    outputs: dict[str, list[dict[str, object]]] = {"base": [], "adapter": []}
    evidence["outputs"] = outputs
    run_dir = Path("/tmp")

    def no_progress(*_args: object) -> None:
        return None

    def score(
        model: Any,
        tokenizer: Any,
        presentations: list[dict[str, object]],
        category: str,
        *,
        role: str,
        retain_outputs: bool = True,
    ) -> list[dict[str, object]]:
        phase_result = {**result, "evidence": {**evidence, "outputs": []}}
        try:
            rows = scoring._score_presentations(
                model,
                tokenizer,
                presentations,
                torch,
                phase_result,
                category,
                run_dir,
                write_progress=no_progress,
                retain_outputs=True,
            )
        except Exception:
            partial = phase_result["evidence"].get("outputs")
            partial_rows = partial if isinstance(partial, list) else []
            if category == "reload_parity":
                evidence["reload_parity"] = {
                    "outputs": partial_rows,
                    "max_candidate_logit_delta": None,
                }
            else:
                outputs[role] = partial_rows
            raise
        if category == "reload_parity":
            evidence["reload_parity"] = {
                "outputs": rows,
                "max_candidate_logit_delta": None,
            }
        else:
            outputs[role] = rows
        return rows

    base_model, base_tokenizer, base_provenance = execution._base_model(
        AutoModelForCausalLM, AutoTokenizer, torch
    )
    result["provenance"]["base_model"] = base_provenance
    result["provenance"]["cuda_device"] = torch.cuda.get_device_name(0)
    score(
        base_model,
        base_tokenizer,
        payload["presentations"],
        "base_evaluation",
        role="base",
    )
    del base_model, base_tokenizer
    _release_cuda(torch)

    adapter_base, adapter_tokenizer, adapter_base_provenance = execution._base_model(
        AutoModelForCausalLM, AutoTokenizer, torch
    )
    _same_tokenizer(base_provenance, adapter_base_provenance)
    adapted = PeftModel.from_pretrained(
        adapter_base,
        str(snapshot["path"]),
        is_trainable=False,
        autocast_adapter_dtype=True,
    ).eval()
    loaded_state = scoring._verify_saved_adapter(
        adapted, snapshot, tensor_digest, peft_module, torch
    )
    scoring._equal_states(saved_state, loaded_state, torch)
    adapter_outputs = score(
        adapted,
        adapter_tokenizer,
        payload["presentations"],
        "final_evaluation",
        role="adapter",
    )
    del adapted, adapter_base, adapter_tokenizer, loaded_state
    _release_cuda(torch)

    reload_base, reload_tokenizer, reload_provenance = execution._base_model(
        AutoModelForCausalLM, AutoTokenizer, torch
    )
    _same_tokenizer(base_provenance, reload_provenance)
    reloaded = PeftModel.from_pretrained(
        reload_base,
        str(snapshot["path"]),
        is_trainable=False,
        autocast_adapter_dtype=True,
    ).eval()
    reloaded_state = scoring._verify_saved_adapter(
        reloaded, snapshot, tensor_digest, peft_module, torch
    )
    scoring._equal_states(saved_state, reloaded_state, torch)
    reloaded_digest = helpers.tensor_state_sha256(reloaded_state, torch_module=torch)
    if reloaded_digest != tensor_digest:
        raise ValueError("fresh reload adapter tensor digest differs from the saved file")
    evidence["adapter_identity"]["reloaded_tensor_sha256"] = reloaded_digest
    result["provenance"]["reload_tokenizer_file_sha256"] = reload_provenance[
        "tokenizer_file_sha256"
    ]
    parity_outputs = score(
        reloaded,
        reload_tokenizer,
        payload["parity_presentations"],
        "reload_parity",
        role="adapter",
        retain_outputs=True,
    )
    max_delta = _candidate_logit_delta(adapter_outputs, parity_outputs)
    evidence["outputs"] = outputs
    evidence["reload_parity"] = {
        "outputs": parity_outputs,
        "max_candidate_logit_delta": max_delta,
    }
    del reloaded, reload_base, reload_tokenizer, reloaded_state, saved_state
    _release_cuda(torch)


def _load_saved_adapter_state(
    snapshot: dict[str, object], torch: Any
) -> tuple[dict[str, Any], str, dict[str, str]]:
    scoring = importlib.import_module("experiments.mixture_training_runtime_scoring")
    verified_files = scoring._snapshot_hashes(snapshot)
    snapshot_dir = Path(str(snapshot["path"]))
    safe_tensors = importlib.import_module("safetensors.torch")
    state = safe_tensors.load_file(str(snapshot_dir / "adapter_model.safetensors"), device="cpu")
    if not isinstance(state, dict) or not state:
        raise ValueError("verified safetensors adapter state is empty or malformed")
    helpers = importlib.import_module("experiments.mixture_training_runtime_helpers")
    digest = helpers.tensor_state_sha256(state, torch_module=torch)
    mixture_contracts.validate_sha256(digest, "saved adapter tensor SHA-256")
    return state, digest, verified_files


def _same_tokenizer(base_provenance: dict[str, object], other: dict[str, object]) -> None:
    base_hashes = base_provenance.get("tokenizer_file_sha256")
    other_hashes = other.get("tokenizer_file_sha256")
    if not isinstance(base_hashes, dict) or base_hashes != other_hashes:
        raise ValueError("fresh model load used different pinned tokenizer files")


def _candidate_logit_delta(
    primary: list[dict[str, object]], parity: list[dict[str, object]]
) -> float:
    by_id = {row["presentation_id"]: row for row in primary}
    deltas: list[float] = []
    for row in parity:
        expected = by_id.get(row["presentation_id"])
        if expected is None:
            raise ValueError("reload parity presentation has no primary adapter score")
        left, right = row["candidate_logits"], expected["candidate_logits"]
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            raise ValueError("reload parity candidate score dimensions differ")
        deltas.extend(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))
    return max(deltas, default=0.0)


def _release_cuda(torch: Any) -> None:
    gc.collect()
    cuda = getattr(torch, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
