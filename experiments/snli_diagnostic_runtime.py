"""Remote-only model loading, source checks, and scoring for the SNLI diagnostic."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
from typing import Any

from experiments import snli_diagnostic_core as core
from reflex_decisions import smoke
from reflex_decisions.schema import DecisionRequest

ADAPTER_PATH = "/artifacts/runs/real-pilot-2026-10-04-r2/adapter-update-252"
ADAPTER_FILE_SHA256 = {
    "adapter_config.json": "fa6fdf55295985c39b5cfd6f6dfb66ab3c941756429cd402f77e1ac455fdea8d",
    "adapter_model.safetensors": "315c23b1517c9590386d22afad5fac2927f97c31f42cdfa73b2c921494266fd8",
}
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
WORKER_MODELS = {"qwen": ("qwen_base", "qwen_final"), "intern": ("intern",), "kev": ("kev",)}


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_digest_map(expected: object, actual: dict[str, str]) -> dict[str, str]:
    if not isinstance(expected, dict) or set(expected) != set(core.SOURCE_FINGERPRINT_PATHS):
        raise ValueError("payload source fingerprint map does not match the explicit allowlist")
    if expected != actual:
        raise ValueError("remote source files differ from their pinned local fingerprints")
    return actual


def _measure_remote_source_fingerprints() -> dict[str, str]:
    root = Path("/root")
    measured: dict[str, str] = {}
    for relative in core.SOURCE_FINGERPRINT_PATHS:
        path = root / relative
        if relative.startswith("src/reflex_decisions/"):
            path = root / relative.removeprefix("src/")
        if not path.is_file():
            raise ValueError(f"allowlisted remote source is missing: {relative}")
        measured[relative] = _sha256_file(path)
    return measured


def _runtime_versions() -> dict[str, str]:
    actual = {name: importlib.metadata.version(name) for name in RUNTIME_VERSION_PINS}
    if actual != RUNTIME_VERSION_PINS:
        raise RuntimeError("remote model runtime versions differ from the frozen image")
    return actual


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": smoke.sanitize_exception_message(error),
    }


def verify_qwen_adapter_files(adapter_dir: str | Path) -> dict[str, object]:
    """Verify saved adapter bytes before handing the directory to PEFT."""

    actual = {
        filename: _sha256_file(Path(adapter_dir) / filename) for filename in ADAPTER_FILE_SHA256
    }
    if actual != ADAPTER_FILE_SHA256:
        raise ValueError("stored Qwen adapter file hashes differ from the frozen snapshot")
    return {"expected_file_sha256": dict(ADAPTER_FILE_SHA256), "actual_file_sha256": actual}


def _empty_model_result(
    payload: dict[str, object],
    model_name: str,
    *,
    source_hashes: dict[str, str] | None = None,
    versions: dict[str, str] | None = None,
) -> dict[str, object]:
    model_pin = core.MODEL_PINS[model_name]
    return {
        "model_name": model_name,
        "status": "failed",
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "pins": payload["pins"],
        "model_id": model_pin["model_id"],
        "model_revision": model_pin["model_revision"],
        "forward_counts": {"scored": 0, "auxiliary": None, "total": None},
        "provenance": {
            "measured_source_file_sha256": source_hashes or {},
            "runtime_versions": versions or {},
            "effective_dtype": None,
            "scorer": {},
            "base": None,
            "adapter": None,
        },
        "presentations": [],
        "failure": {
            "stage": "not_started",
            "type": "NotStarted",
            "message": "model evaluation did not complete",
        },
    }


def _fail_model(
    result: dict[str, object], stage: str, error: BaseException, *, auxiliary: int | None = None
) -> dict[str, object]:
    result["status"] = "failed"
    presentations = result.get("presentations")
    scored = len(presentations) if isinstance(presentations, list) else 0
    result["forward_counts"] = {
        "scored": scored,
        "auxiliary": auxiliary,
        # A failed call may have executed a forward before it raised or before its
        # output was validated. Keep the observed row prefix and known auxiliary work,
        # but do not infer an exact total from those two partial counts.
        "total": None,
    }
    result["failure"] = _failure(stage, error)
    return result


def _score_reference_presentations(
    presentations: list[dict[str, object]], scorer: Any, torch: Any, result: dict[str, object]
) -> None:
    from experiments import baseline_runner_core

    for index, presentation in enumerate(presentations, start=1):
        request = DecisionRequest.model_validate(presentation["request"])
        with torch.inference_mode():
            scored = scorer(request)
        validated = baseline_runner_core.validate_presentation_result(
            {
                "record_id": presentation["record_id"],
                "request_hash": presentation["request_hash"],
                "permutation_index": presentation["order_index"],
                "option_ids": presentation["order_ids"],
            },
            scored,
        )
        logits = validated["raw_logits"]
        order_ids = presentation["order_ids"]
        winner = min(
            option_id
            for option_id, logit in zip(order_ids, logits, strict=True)
            if logit == max(logits)
        )
        result["presentations"].append(
            {
                key: presentation[key]
                for key in (
                    "presentation_id",
                    "record_id",
                    "source_group_id",
                    "request_hash",
                    "order_index",
                    "order_ids",
                )
            }
            | {
                "candidate_logits": logits,
                "winner_option_id": winner,
                "input_tokens": validated["input_tokens"],
                "prompt_sha256": validated["prompt_sha256"],
            }
        )
        if index % 128 == 0:
            print(f"Scored {index}/1152 presentations for {result['model_name']}")


def _run_reference_model(
    payload: dict[str, object],
    model_name: str,
    source_hashes: dict[str, str],
    versions: dict[str, str],
) -> dict[str, object]:
    result = _empty_model_result(
        payload, model_name, source_hashes=source_hashes, versions=versions
    )
    stage = "model_loading"
    scorer_provenance: dict[str, object] = {}
    raw_provenance: dict[str, object] | None = None
    reference_core: Any | None = None

    def store_scorer_provenance(value: dict[str, object]) -> None:
        provenance = result["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError("model provenance container is malformed")
        scorer_runtime = value.get("runtime")
        provenance["scorer"] = value
        provenance["effective_dtype"] = (
            scorer_runtime.get("dtype")
            if isinstance(scorer_runtime, dict)
            else value.get("effective_dtype")
        )

    try:
        os.environ["USE_HUB_KERNELS"] = "NO"
        adapter_name = (
            "experiments.baseline_intern" if model_name == "intern" else "experiments.baseline_kev"
        )
        adapter = importlib.import_module(adapter_name)
        scorer, loaded_provenance = adapter.load_scorer()
        from experiments import real_pilot_baseline_core as reference_core

        if not isinstance(loaded_provenance, dict):
            raise ValueError("reference scorer provenance must be an object")
        raw_provenance = loaded_provenance
        scorer_provenance = reference_core._validate_scorer_identity(model_name, raw_provenance)
        store_scorer_provenance(scorer_provenance)
        import torch

        stage = "inference"
        _score_reference_presentations(payload["presentations"], scorer, torch, result)
        stage = "scorer_provenance"
        scorer_provenance = reference_core._validate_scorer_provenance(model_name, raw_provenance)
        store_scorer_provenance(scorer_provenance)
        auxiliary = scorer_provenance.get("auxiliary_forward_count")
        if type(auxiliary) is not int:
            raise ValueError("reference scorer did not report its auxiliary forward count")
        result["forward_counts"] = {
            "scored": len(result["presentations"]),
            "auxiliary": auxiliary,
            "total": len(result["presentations"]) + auxiliary,
        }
        result["status"] = "passed"
        del result["failure"]
        return core.validate_model_result(result, payload, model_name)
    except Exception as exc:
        if reference_core is not None and raw_provenance is not None:
            try:
                scorer_provenance = reference_core._validate_scorer_identity(
                    model_name, raw_provenance
                )
                store_scorer_provenance(scorer_provenance)
            except Exception:
                pass
        auxiliary = scorer_provenance.get("auxiliary_forward_count")
        return _fail_model(
            result, stage, exc, auxiliary=auxiliary if type(auxiliary) is int else None
        )


def _qwen_adapter_evidence(model: Any, torch: Any) -> dict[str, object]:
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    from experiments import modal_train_rehearsal

    adapter_dir = Path(ADAPTER_PATH)
    hashes = verify_qwen_adapter_files(adapter_dir)
    inventory = modal_train_rehearsal._validate_adapter_inventory(
        model, torch, require_trainable=False
    )
    modules, _parameters = modal_train_rehearsal._adapter_modules_and_parameters(model)
    for _name, module in modules:
        active = getattr(module, "active_adapters", getattr(module, "_active_adapter", None))
        active_values = [active] if isinstance(active, str) else list(active or ())
        disabled = getattr(module, "disable_adapters", getattr(module, "_disable_adapters", False))
        merged = getattr(module, "merged_adapters", ())
        if (
            active_values != ["default"]
            or disabled
            or merged
            or bool(getattr(module, "merged", False))
        ):
            raise RuntimeError("Qwen LoRA module is disabled, inactive, or merged")
    active = getattr(model, "active_adapters", getattr(model, "active_adapter", None))
    active_values = [active] if isinstance(active, str) else list(active or ())
    if active_values != ["default"]:
        raise RuntimeError("Qwen default LoRA adapter is not active")
    adapter_file = adapter_dir / "adapter_model.safetensors"
    saved = load_file(str(adapter_file), device="cpu")
    loaded = get_peft_model_state_dict(model)
    normalized = {name: tensor.detach().cpu() for name, tensor in loaded.items()}
    if set(saved) != set(normalized) or any(
        saved[name].shape != normalized[name].shape
        or not torch.equal(saved[name], normalized[name])
        for name in saved
    ):
        raise RuntimeError(
            "reloaded Qwen adapter differs from saved tensor keys, shapes, or values"
        )
    if any(parameter.requires_grad for _name, parameter in model.named_parameters()):
        raise RuntimeError("reloaded Qwen model must be fully frozen")
    return {
        **hashes,
        "tensor_keys_shapes_values_match": True,
        "default_adapter_active": True,
        "adapter_unmerged": True,
        "inventory": inventory,
        "all_adapter_parameters_frozen": True,
    }


def _compile_qwen_presentations(
    presentations: list[dict[str, object]], tokenizer: Any
) -> list[tuple[dict[str, object], DecisionRequest, Any]]:
    from reflex_decisions.rendering import compile_request

    return [
        (
            presentation,
            request := DecisionRequest.model_validate(presentation["request"]),
            compile_request(request, tokenizer, max_tokens=2048),
        )
        for presentation in presentations
    ]


def _score_qwen_presentations(
    model: Any,
    rows: list[tuple[dict[str, object], DecisionRequest, Any]],
    torch: Any,
    result: dict[str, object],
) -> None:
    from experiments.modal_real_pilot import _candidate_tensor_logits, _semantic_winner

    model.eval()
    counter = {"count": 0}
    with torch.inference_mode():
        for index, (presentation, request, compiled) in enumerate(rows, start=1):
            logits_tensor = _candidate_tensor_logits(model, request, compiled, torch, counter)
            logits = [float(value) for value in logits_tensor.detach().cpu().tolist()]
            order_ids = list(presentation["order_ids"])
            result["presentations"].append(
                {
                    key: presentation[key]
                    for key in (
                        "presentation_id",
                        "record_id",
                        "source_group_id",
                        "request_hash",
                        "order_index",
                        "order_ids",
                    )
                }
                | {
                    "candidate_logits": logits,
                    "winner_option_id": _semantic_winner(order_ids, logits),
                    "input_tokens": len(compiled.input_ids),
                    "prompt_sha256": compiled.prompt_hash,
                }
            )
            if index % 128 == 0:
                print(f"Scored {index}/1152 presentations for {result['model_name']}")
    if counter["count"] != len(rows):
        raise RuntimeError("Qwen scorer forward count differs from the presentation count")


def _run_qwen_models(
    payload: dict[str, object], source_hashes: dict[str, str], versions: dict[str, str]
) -> dict[str, dict[str, object]]:
    results = {
        name: _empty_model_result(payload, name, source_hashes=source_hashes, versions=versions)
        for name in WORKER_MODELS["qwen"]
    }
    try:
        os.environ["USE_HUB_KERNELS"] = "NO"
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from experiments import modal_train_rehearsal

        modal_train_rehearsal._assert_runtime(torch)
        model, tokenizer, base_provenance = modal_train_rehearsal._load_base(
            AutoTokenizer, AutoModelForCausalLM, torch
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("pinned Qwen base must be frozen before evaluation")
        rows = _compile_qwen_presentations(payload["presentations"], tokenizer)
        if len(rows) != 1152:
            raise ValueError("Qwen input did not compile all 1,152 presentations")
        base_result = results["qwen_base"]
        base_result["provenance"].update(
            {"base": base_provenance, "effective_dtype": base_provenance.get("effective_dtype")}
        )
        try:
            _score_qwen_presentations(model, rows, torch, base_result)
            scored = len(base_result["presentations"])
            base_result["forward_counts"] = {"scored": scored, "auxiliary": 0, "total": scored}
            base_result["status"] = "passed"
            del base_result["failure"]
            results["qwen_base"] = core.validate_model_result(base_result, payload, "qwen_base")
        except Exception as exc:
            _fail_model(base_result, "base_inference", exc, auxiliary=0)
        final_result = results["qwen_final"]
        final_result["provenance"].update(
            {"base": base_provenance, "effective_dtype": base_provenance.get("effective_dtype")}
        )
        try:
            adapter_dir = Path(ADAPTER_PATH)
            verify_qwen_adapter_files(adapter_dir)
            final_model = PeftModel.from_pretrained(
                model, str(adapter_dir), is_trainable=False, autocast_adapter_dtype=True
            ).eval()
            final_result["provenance"]["adapter"] = _qwen_adapter_evidence(final_model, torch)
            _score_qwen_presentations(final_model, rows, torch, final_result)
            scored = len(final_result["presentations"])
            final_result["forward_counts"] = {"scored": scored, "auxiliary": 0, "total": scored}
            final_result["status"] = "passed"
            del final_result["failure"]
            results["qwen_final"] = core.validate_model_result(final_result, payload, "qwen_final")
        except Exception as exc:
            _fail_model(final_result, "final_inference", exc, auxiliary=0)
    except Exception as exc:
        for result in results.values():
            failure = result.get("failure")
            if isinstance(failure, dict) and failure.get("stage") == "not_started":
                _fail_model(result, "qwen_setup", exc, auxiliary=0)
    return results


class _ProductionRuntime:
    def measure_sources(self) -> dict[str, str]:
        return _measure_remote_source_fingerprints()

    def runtime_versions(self) -> dict[str, str]:
        return _runtime_versions()

    def run_worker(
        self,
        payload: dict[str, object],
        worker: str,
        source_hashes: dict[str, str],
        versions: dict[str, str],
    ) -> dict[str, dict[str, object]]:
        if worker == "qwen":
            return _run_qwen_models(payload, source_hashes, versions)
        return {worker: _run_reference_model(payload, worker, source_hashes, versions)}


def failed_model_result(
    payload: dict[str, object], model_name: str, stage: str, error: BaseException
) -> dict[str, object]:
    return core.validate_model_result(
        _fail_model(
            _empty_model_result(payload, model_name),
            stage,
            error,
            auxiliary=0 if model_name.startswith("qwen_") else None,
        ),
        payload,
        model_name,
    )


def remote_worker(
    payload: dict[str, object], worker: str, runtime: Any | None = None
) -> dict[str, object]:
    """Verify payload, source, and runtime pins before importing any model runtime."""

    original = payload
    stage = "payload_validation"
    try:
        payload = core.validate_payload(payload)
        if worker not in WORKER_MODELS:
            raise ValueError("worker name is outside the frozen model allowlist")
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["USE_HUB_KERNELS"] = "NO"
        worker_runtime = runtime if runtime is not None else _ProductionRuntime()
        stage = "source_verification"
        source_hashes = _verify_digest_map(
            payload["pins"]["source_file_sha256"], worker_runtime.measure_sources()
        )
        stage = "runtime_check"
        versions = worker_runtime.runtime_versions()
        if versions != RUNTIME_VERSION_PINS:
            raise RuntimeError("worker runtime versions differ from the frozen Modal image")
        stage = "model_evaluation"
        models = worker_runtime.run_worker(payload, worker, source_hashes, versions)
        normalized: dict[str, dict[str, object]] = {}
        for model_name in WORKER_MODELS[worker]:
            try:
                normalized[model_name] = core.validate_model_result(
                    models[model_name], payload, model_name
                )
            except Exception as exc:
                normalized[model_name] = failed_model_result(
                    payload, model_name, "result_validation", exc
                )
        return {"worker": worker, "models": normalized}
    except Exception as exc:
        failure_models = {
            name: _fail_model(
                _empty_model_result(original, name),
                stage,
                exc,
                auxiliary=0 if name.startswith("qwen_") else None,
            )
            for name in WORKER_MODELS.get(worker, ())
            if isinstance(original, dict) and "run_id" in original and "pins" in original
        }
        return {"worker": worker, "models": failure_models, "failure": _failure(stage, exc)}
