"""Remote inference-only execution for the frozen fresh-evaluation payload."""
# ruff: noqa: E501

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from experiments import fresh_eval_core as core
from experiments import mixture_training_contracts as mixture_contracts
from reflex_decisions import smoke

_FORBIDDEN_KERNEL_PACKAGES = ("kernels", "fla", "causal_conv1d")
_COUNT_KEYS = ("base_evaluation", "training", "final_evaluation", "reload_parity", "total")


def _check_compilation(expected: dict[str, object], compiled: Any) -> None:
    """Reject remote tokenization drift before its logits become evidence."""

    actual_ids = list(compiled.input_ids)
    actual_hash = hashlib.sha256(
        json.dumps(actual_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if compiled.request_hash != expected["request_hash"]:
        raise ValueError("remote compiled request hash differs from the frozen identity")
    if compiled.prompt_hash != expected["prompt_sha256"]:
        raise ValueError("remote compiled prompt hash differs from the frozen identity")
    if len(actual_ids) != expected["input_tokens"] or actual_hash != expected["input_ids_sha256"]:
        raise ValueError("remote compiled input IDs differ from the frozen identity")
    if list(compiled.candidate_token_ids) != expected["candidate_token_ids"]:
        raise ValueError("remote compiled candidate IDs differ from the frozen identity")


def _exception_message(error: BaseException) -> str:
    return smoke.sanitize_exception_message(error).strip() or type(error).__name__


def _initial_result(payload: dict[str, object]) -> dict[str, object]:
    counts = {key: 0 for key in _COUNT_KEYS}
    return {
        **{
            key: payload[key]
            for key in (
                "schema_version",
                "experiment_id",
                "run_id",
                "nonce",
                "selection_sha256",
                "payload_sha256",
            )
        },
        "status": "failed",
        "phase": "preflight",
        "provenance": {
            "model_id": payload["pins"]["model_id"],
            "model_revision": payload["pins"]["model_revision"],
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
            "unknown_work": None,
        },
        "failure": {
            "stage": "not_started",
            "type": "NotStarted",
            "message": "worker did not complete",
        },
    }


def _verify_remote_sources(payload: dict[str, object]) -> dict[str, str]:
    measured = core.source_fingerprints(Path(__file__).resolve().parents[1])
    if measured != payload["pins"]["source_file_sha256"]:
        raise ValueError("remote source files differ from their pinned hashes")
    return measured


def _runtime_packages() -> tuple[Any, ...]:
    os.environ["USE_HUB_KERNELS"] = "NO"
    if any(
        name in sys.modules or importlib.util.find_spec(name) is not None
        for name in _FORBIDDEN_KERNEL_PACKAGES
    ):
        raise RuntimeError("optional model-kernel packages must not be installed")
    versions = {
        name: importlib.metadata.version(name) for name in mixture_contracts.RUNTIME_VERSION_PINS
    }
    if versions != mixture_contracts.RUNTIME_VERSION_PINS:
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


def _score_panel(
    model: Any,
    tokenizer: Any,
    presentations: list[dict[str, object]],
    compiled_requests: list[dict[str, object]],
    torch: Any,
    result: dict[str, object],
    category: str,
    retained_outputs: list[dict[str, object]],
) -> list[dict[str, object]]:
    from reflex_decisions.schema import DecisionRequest

    scoring = importlib.import_module("experiments.mixture_training_runtime_scoring")
    rows: list[dict[str, object]] = []
    for presentation, expected in zip(presentations, compiled_requests, strict=True):
        request = DecisionRequest.model_validate(presentation["request"])
        try:
            with torch.inference_mode():
                logits, compiled = scoring._score_forward(
                    model, request, tokenizer, torch, result, category
                )
        except (Exception, KeyboardInterrupt):
            result["evidence"]["unknown_work"] = {
                "possible_forwards": 1,
                "reason": "a model forward raised before its completion could be observed",
            }
            raise
        _check_compilation(expected, compiled)
        rows.append(scoring._scored_row(presentation, logits, compiled))
        retained_outputs[:] = rows
    return rows


def _same_tokenizer(left: dict[str, object], right: dict[str, object]) -> None:
    if left.get("tokenizer_file_sha256") != right.get("tokenizer_file_sha256"):
        raise ValueError("fresh model load used different pinned tokenizer files")


def _verify_tokenizer_pin(provenance: dict[str, object], payload: dict[str, object]) -> None:
    hashes = provenance.get("tokenizer_file_sha256")
    if (
        not isinstance(hashes, dict)
        or hashes.get("tokenizer.json") != payload["pins"]["tokenizer_file_sha256"]
    ):
        raise ValueError("remote tokenizer provenance differs from the frozen CPU tokenizer")


def _release_cuda(torch: Any) -> None:
    gc.collect()
    empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _candidate_logit_delta(
    primary: list[dict[str, object]], parity: list[dict[str, object]]
) -> float:
    by_id = {row["presentation_id"]: row for row in primary}
    deltas: list[float] = []
    for row in parity:
        expected = by_id.get(row["presentation_id"])
        if expected is None or row["winner_option_id"] != expected["winner_option_id"]:
            raise RuntimeError("fresh adapter reload differs from the primary semantic winner")
        deltas.extend(
            abs(float(a) - float(b))
            for a, b in zip(row["candidate_logits"], expected["candidate_logits"], strict=True)
        )
    return max(deltas, default=0.0)


def _run_evaluation(
    payload: dict[str, object], result: dict[str, object], packages: tuple[Any, ...]
) -> None:
    torch, _transformers, PeftModel, AutoModelForCausalLM, AutoTokenizer, peft_module, _versions = (
        packages
    )
    execution = importlib.import_module("experiments.mixture_training_runtime_execution")
    scoring = importlib.import_module("experiments.mixture_training_runtime_scoring")
    helpers = importlib.import_module("experiments.mixture_training_runtime_helpers")
    transfer_runtime = importlib.import_module("experiments.adapter_transfer_runtime")
    snapshot = payload["selection"]["snapshot"]
    saved_state, tensor_digest, files_sha256 = transfer_runtime._load_saved_adapter_state(
        snapshot, torch
    )
    if tensor_digest != payload["pins"]["adapter_tensor_sha256"]:
        raise ValueError("saved adapter tensor digest differs from the frozen selection")
    evidence = result["evidence"]
    evidence["adapter_identity"] = {
        "snapshot_path": snapshot["path"],
        "files_sha256": files_sha256,
        "tensor_sha256": tensor_digest,
        "reloaded_tensor_sha256": None,
        "dtype": "torch.float32",
    }
    compiled = payload["compiled_requests"]
    compiled_by_id = {row["presentation_id"]: row for row in compiled}
    base, tokenizer, provenance = execution._base_model(AutoModelForCausalLM, AutoTokenizer, torch)
    result["provenance"]["base_model"] = provenance
    _verify_tokenizer_pin(provenance, payload)
    result["provenance"]["cuda_device"] = torch.cuda.get_device_name(0)
    evidence["outputs"]["base"] = []
    _score_panel(
        base,
        tokenizer,
        payload["presentations"],
        compiled,
        torch,
        result,
        "base_evaluation",
        evidence["outputs"]["base"],
    )
    del base, tokenizer
    _release_cuda(torch)
    adapter_base, adapter_tokenizer, adapter_provenance = execution._base_model(
        AutoModelForCausalLM, AutoTokenizer, torch
    )
    _same_tokenizer(provenance, adapter_provenance)
    adapted = PeftModel.from_pretrained(
        adapter_base, str(snapshot["path"]), is_trainable=False, autocast_adapter_dtype=True
    ).eval()
    loaded_state = scoring._verify_saved_adapter(
        adapted, snapshot, tensor_digest, peft_module, torch
    )
    scoring._equal_states(saved_state, loaded_state, torch)
    evidence["outputs"]["adapter"] = []
    _score_panel(
        adapted,
        adapter_tokenizer,
        payload["presentations"],
        compiled,
        torch,
        result,
        "final_evaluation",
        evidence["outputs"]["adapter"],
    )
    adapter_rows = evidence["outputs"]["adapter"]
    del adapted, adapter_base, adapter_tokenizer, loaded_state
    _release_cuda(torch)
    reload_base, reload_tokenizer, reload_provenance = execution._base_model(
        AutoModelForCausalLM, AutoTokenizer, torch
    )
    _same_tokenizer(provenance, reload_provenance)
    reloaded = PeftModel.from_pretrained(
        reload_base, str(snapshot["path"]), is_trainable=False, autocast_adapter_dtype=True
    ).eval()
    reloaded_state = scoring._verify_saved_adapter(
        reloaded, snapshot, tensor_digest, peft_module, torch
    )
    scoring._equal_states(saved_state, reloaded_state, torch)
    reloaded_digest = helpers.tensor_state_sha256(reloaded_state, torch_module=torch)
    if reloaded_digest != tensor_digest:
        raise ValueError("fresh reload adapter tensor digest differs from the saved adapter")
    evidence["adapter_identity"]["reloaded_tensor_sha256"] = reloaded_digest
    result["provenance"]["reload_tokenizer_file_sha256"] = reload_provenance[
        "tokenizer_file_sha256"
    ]
    parity_compiled = [
        compiled_by_id[row["presentation_id"]] for row in payload["parity_presentations"]
    ]
    parity_rows: list[dict[str, object]] = []
    evidence["reload_parity"] = {"outputs": parity_rows, "max_candidate_logit_delta": None}
    _score_panel(
        reloaded,
        reload_tokenizer,
        payload["parity_presentations"],
        parity_compiled,
        torch,
        result,
        "reload_parity",
        parity_rows,
    )
    evidence["reload_parity"] = {
        "outputs": parity_rows,
        "max_candidate_logit_delta": _candidate_logit_delta(adapter_rows, parity_rows),
    }
    del reloaded, reload_base, reload_tokenizer, reloaded_state, saved_state
    _release_cuda(torch)


def remote_worker(payload: dict[str, object]) -> dict[str, object]:
    """Execute exactly the frozen base, adapter, and reload forwards once."""

    normalized = core.validate_payload(payload)
    result = _initial_result(normalized)
    stage = "source_verification"
    try:
        result["provenance"]["measured_source_file_sha256"] = _verify_remote_sources(normalized)
        stage = "runtime_imports"
        packages = _runtime_packages()
        result["provenance"]["versions"] = packages[-1]
        stage = "inference"
        _run_evaluation(normalized, result, packages)
        result.update({"status": "passed", "phase": "completed", "failure": None})
    except (Exception, KeyboardInterrupt) as exc:
        result.update(
            {
                "status": "failed",
                "phase": f"{stage}_failed",
                "failure": {
                    "stage": stage,
                    "type": type(exc).__name__,
                    "message": _exception_message(exc),
                },
            }
        )
    try:
        return core.validate_result(result, normalized)
    except (Exception, KeyboardInterrupt) as exc:
        result.update(
            {
                "status": "failed",
                "phase": "result_validation_failed",
                "failure": {
                    "stage": "result_validation",
                    "type": type(exc).__name__,
                    "message": _exception_message(exc),
                },
            }
        )
        return result
