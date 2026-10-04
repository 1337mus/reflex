"""Plan or explicitly launch one bounded Qwen inference smoke on Modal."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from typing import Any

from reflex_decisions import smoke

CREDENTIAL_OVERRIDES = (
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_OAUTH_REFRESH_TOKEN",
    "MODAL_OAUTH_CLIENT_ID",
    "MODAL_OAUTH_CLIENT_SECRET",
)


def _plan() -> dict[str, object]:
    return {
        "mode": "plan-only",
        "model": smoke.MODEL_ID,
        "model_revision": smoke.MODEL_REVISION,
        "python": "3.12",
        "modal": "1.6.1",
        "transformers": "5.18.0",
        "torch": "2.14.1+cu130",
        "gpu": "A10",
        "calls": 1,
        "max_input_tokens": smoke.MAX_INPUT_TOKENS,
        "parity_tolerances": {
            "candidate_logit_max_abs": smoke.LOGIT_TOLERANCE,
            "candidate_probability_max_abs": smoke.PROBABILITY_TOLERANCE,
        },
    }


def _failure(stage: str, error_type: str, message: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "failed",
        "failure": {"stage": stage, "type": error_type, "message": message},
        "provenance": {
            "model": smoke.MODEL_ID,
            "model_revision": smoke.MODEL_REVISION,
            "profile": os.environ.get("MODAL_PROFILE"),
        },
        "limits": _limits(),
        "checks": {},
    }


def _limits() -> dict[str, object]:
    return {
        "gpu": "A10",
        "cpu_physical_cores": [2.0, 2.0],
        "memory_mib": [8192, 8192],
        "max_containers": 1,
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window_seconds": 2,
        "retries": 0,
        "single_use_containers": True,
        "startup_timeout_seconds": 300,
        "input_timeout_seconds": 900,
        "dollar_ceiling": None,
    }


def _write_failure(
    reservation: smoke.ArtifactReservation, stage: str, kind: str, message: str
) -> int:
    smoke.write_json_artifact(reservation, _failure(stage, kind, message))
    print(json.dumps({"status": "failed", "artifact": str(reservation.destination)}))
    return 1


def _verify_profile(profile: str, workspace: str) -> bool:
    os.environ["MODAL_PROFILE"] = profile
    child_env = os.environ.copy()
    child_env.update({"MODAL_PROFILE": profile, "NO_COLOR": "1", "TERM": "dumb"})
    try:
        info = subprocess.run(
            [sys.executable, "-m", "modal", "token", "info"],
            capture_output=True,
            text=True,
            check=False,
            env=child_env,
        )
    except OSError:
        return False
    expected = re.fullmatch(r"([^()\s]+)(?: \(([^()\s]+)\))?", workspace)
    workspace_lines = [line for line in info.stdout.splitlines() if line.startswith("Workspace:")]
    if info.returncode != 0 or expected is None or len(workspace_lines) != 1:
        return False
    actual = re.fullmatch(r"Workspace: ([^()\s]+)(?: \(([^()\s]+)\))?", workspace_lines[0])
    return bool(
        actual
        and actual.group(1) == expected.group(1)
        and (expected.group(2) is None or actual.group(2) == expected.group(2))
    )


def _remote_inference() -> dict[str, object]:
    """Run all model imports, downloads, checks, and inference inside the container."""

    import os

    os.environ["USE_HUB_KERNELS"] = "NO"
    import hashlib
    import importlib.metadata
    import importlib.util
    import logging
    import math

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from reflex_decisions import smoke as remote_smoke

    stage = "initialization"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "provenance": {
            "model": remote_smoke.MODEL_ID,
            "model_revision": remote_smoke.MODEL_REVISION,
            "python": sys.version.split()[0],
            "versions": {},
        },
        "limits": _limits(),
        "checks": {},
        "requests": [],
    }
    try:
        stage = "runtime_metadata"
        package_names = (
            "torch",
            "transformers",
            "tokenizers",
            "safetensors",
            "huggingface-hub",
            "pydantic",
        )
        versions = {name: importlib.metadata.version(name) for name in package_names}
        result["provenance"]["versions"] = versions
        result["provenance"]["torch_cuda_runtime"] = torch.version.cuda
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            result["provenance"]["driver_version"] = driver.stdout.strip().splitlines()[0]
        except (OSError, IndexError, subprocess.SubprocessError):
            result["provenance"]["driver_version"] = None

        stage = "cuda_check"
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            result["failure"] = {
                "stage": stage,
                "type": "RuntimeError",
                "message": "CUDA BF16 is unavailable",
            }
            return result
        device = torch.cuda.get_device_properties(0)
        result["provenance"].update(
            {"device": device.name, "device_capability": list(torch.cuda.get_device_capability(0))}
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        stage = "tokenizer_load"
        tokenizer = AutoTokenizer.from_pretrained(
            remote_smoke.MODEL_ID,
            revision=remote_smoke.MODEL_REVISION,
            trust_remote_code=False,
            token=False,
        )
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        if pad_token_id is None:
            raise RuntimeError("tokenizer has no usable pad or EOS token")
        requests = remote_smoke.synthetic_requests()
        compiled = [remote_smoke.compile_smoke_request(request, tokenizer) for request in requests]

        stage = "model_load"
        model, load_info = AutoModelForCausalLM.from_pretrained(
            remote_smoke.MODEL_ID,
            revision=remote_smoke.MODEL_REVISION,
            dtype=torch.bfloat16,
            trust_remote_code=False,
            output_loading_info=True,
            attn_implementation="eager",
            use_kernels=False,
            token=False,
        )
        required_info = (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if any(key not in load_info for key in required_info):
            raise RuntimeError("Transformers loading diagnostics did not contain required fields")
        allowed_unexpected = ("model.visual.", "mtp.")
        unexpected = list(load_info["unexpected_keys"])
        unexplained = [key for key in unexpected if not key.startswith(allowed_unexpected)]
        loading = {
            "missing_keys": list(load_info["missing_keys"]),
            "unexpected_keys": unexpected,
            "mismatched_keys": list(load_info["mismatched_keys"]),
            "error_msgs": list(load_info["error_msgs"]),
            "allowed_unexpected_prefixes": list(allowed_unexpected),
            "unexplained_unexpected_keys": unexplained,
        }
        expected_model = isinstance(model, Qwen3_5ForCausalLM)
        expected_config = isinstance(model.config, Qwen3_5TextConfig)
        tied = (
            model.config.tie_word_embeddings
            and model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        )
        no_meta = all(parameter.device.type != "meta" for parameter in model.parameters())
        layers = len(model.model.layers)
        loading_ok = (
            not loading["missing_keys"]
            and not loading["mismatched_keys"]
            and not loading["error_msgs"]
            and not unexplained
        )
        result["checks"]["loading"] = {
            "passed": expected_model
            and expected_config
            and tied
            and no_meta
            and layers == 24
            and loading_ok,
            "model_class": type(model).__name__,
            "config_class": type(model.config).__name__,
            "layer_count": layers,
            "tied_embeddings": tied,
            "no_meta_parameters": no_meta,
            "diagnostics": loading,
        }
        if not result["checks"]["loading"]["passed"]:
            result["failure"] = {
                "stage": stage,
                "type": "LoadingCheckFailed",
                "message": "model or weight diagnostics failed",
            }
            return result
        model = model.to("cuda").eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        result["provenance"]["parameter_count"] = parameter_count

        stage = "kernel_path_check"
        modeling = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        optional = {
            name: importlib.util.find_spec(name) is not None
            for name in ("kernels", "fla", "causal_conv1d")
        }
        reference_functions = {
            name: callable(getattr(modeling, name, None))
            for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")
        }
        kernels_ok = (
            os.environ.get("USE_HUB_KERNELS") == "NO"
            and not any(optional.values())
            and all(reference_functions.values())
        )
        result["checks"]["kernels"] = {
            "passed": kernels_ok,
            "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
            "use_kernels_argument": False,
            "optional_packages_installed": optional,
            "reference_functions_available": reference_functions,
            "attention_implementation": "eager",
        }
        logging.getLogger("transformers").setLevel(logging.WARNING)

        stage = "inference"
        padded = remote_smoke.left_pad_inputs(
            [item.input_ids for item in compiled], pad_token_id=pad_token_id
        )
        device_inputs = {
            "input_ids": torch.tensor(padded.input_ids, dtype=torch.long, device="cuda"),
            "attention_mask": torch.tensor(padded.attention_mask, dtype=torch.long, device="cuda"),
            "position_ids": torch.tensor(padded.position_ids, dtype=torch.long, device="cuda"),
        }
        single_logits: list[list[float]] = []
        with torch.inference_mode():
            for item in compiled:
                one = remote_smoke.left_pad_inputs((item.input_ids,), pad_token_id=pad_token_id)
                output = model(
                    input_ids=torch.tensor(one.input_ids, dtype=torch.long, device="cuda"),
                    attention_mask=torch.tensor(
                        one.attention_mask, dtype=torch.long, device="cuda"
                    ),
                    position_ids=torch.tensor(one.position_ids, dtype=torch.long, device="cuda"),
                    use_cache=False,
                    logits_to_keep=1,
                )
                single_logits.append(
                    output.logits[0, -1]
                    .index_select(0, torch.tensor(item.candidate_token_ids, device="cuda"))
                    .float()
                    .cpu()
                    .tolist()
                )
            batch_output = model(**device_inputs, use_cache=False, logits_to_keep=1)
            batch_logits = [
                batch_output.logits[index, -1]
                .index_select(0, torch.tensor(item.candidate_token_ids, device="cuda"))
                .float()
                .cpu()
                .tolist()
                for index, item in enumerate(compiled)
            ]
        request_results = []
        all_parity_passed = True
        all_mappings_consistent = True

        def summarize_scores(request: Any, values: list[float]) -> dict[str, object]:
            try:
                return remote_smoke.score_candidates(
                    request,
                    values,
                    model_revision=f"{remote_smoke.MODEL_ID}@{remote_smoke.MODEL_REVISION}",
                ).model_dump(mode="json")
            except (TypeError, ValueError, OverflowError):
                return {"status": "failed", "scores": None, "answer_id": None}

        for index, (request, item) in enumerate(zip(requests, compiled, strict=True)):
            parity = remote_smoke.compare_candidate_logits(
                request, single_logits[index], batch_logits[index]
            )
            all_parity_passed = all_parity_passed and parity.passed
            mapping_consistent = tuple(
                option_id for _, option_id in item.symbol_to_option_id
            ) == tuple(option.id for option in request.options) and len(
                item.candidate_token_ids
            ) == len(request.options)
            all_mappings_consistent = all_mappings_consistent and mapping_consistent
            single_values = [
                value if math.isfinite(value) else None for value in single_logits[index]
            ]
            batch_values = [
                value if math.isfinite(value) else None for value in batch_logits[index]
            ]
            request_results.append(
                {
                    "request_hash": item.request_hash,
                    "schema_hash": item.schema_hash,
                    "prompt_hash": item.prompt_hash,
                    "input_ids_hash": hashlib.sha256(
                        json.dumps(item.input_ids).encode()
                    ).hexdigest(),
                    "input_length": len(item.input_ids),
                    "candidate_mapping": [
                        {"symbol": symbol, "option_id": option_id, "token_id": token_id}
                        for (symbol, option_id), token_id in zip(
                            item.symbol_to_option_id, item.candidate_token_ids, strict=True
                        )
                    ],
                    "mapping_consistent": mapping_consistent,
                    "single": {
                        "candidate_logits": single_values,
                        "scores": summarize_scores(request, single_logits[index]),
                    },
                    "batch": {
                        "candidate_logits": batch_values,
                        "scores": summarize_scores(request, batch_logits[index]),
                    },
                    "parity": {
                        "passed": parity.passed,
                        "logit_differences": parity.logit_differences,
                        "probability_differences": parity.probability_differences,
                        "max_abs_logit_difference": parity.max_abs_logit_difference,
                        "max_abs_probability_difference": parity.max_abs_probability_difference,
                        "top_id_agreement": parity.top_id_agreement,
                        "logit_tolerance": parity.logit_tolerance,
                        "probability_tolerance": parity.probability_tolerance,
                        "failure_reasons": parity.failure_reasons,
                    },
                }
            )
        original_ids = tuple(option_id for _, option_id in compiled[0].symbol_to_option_id)
        reversed_ids = tuple(option_id for _, option_id in compiled[2].symbol_to_option_id)
        result["requests"] = request_results
        result["checks"]["permutation_mapping"] = {
            "passed": all_mappings_consistent and reversed_ids == tuple(reversed(original_ids)),
            "original_option_ids": original_ids,
            "reversed_option_ids": reversed_ids,
            "answer_order_invariance_required": False,
        }
        result["checks"]["parity"] = {
            "passed": all_parity_passed,
            "logit_tolerance": remote_smoke.LOGIT_TOLERANCE,
            "probability_tolerance": remote_smoke.PROBABILITY_TOLERANCE,
        }
        result["provenance"]["max_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        result["status"] = (
            "passed"
            if all_parity_passed
            and result["checks"]["kernels"]["passed"]
            and result["checks"]["loading"]["passed"]
            and result["checks"]["permutation_mapping"]["passed"]
            else "failed"
        )
        if result["status"] != "passed":
            result["failure"] = {
                "stage": "parity",
                "type": "ParityCheckFailed",
                "message": "one or more parity or mapping checks failed",
            }
        return result
    except Exception as exc:
        result["failure"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": remote_smoke.sanitize_exception_message(exc),
        }
        return result


def _launch(profile: str, workspace: str, reservation: smoke.ArtifactReservation) -> int:
    if any(os.environ.get(name, "") for name in CREDENTIAL_OVERRIDES):
        return _write_failure(
            reservation,
            "environment",
            "CredentialOverride",
            "Modal credential environment overrides are not accepted",
        )
    if not _verify_profile(profile, workspace):
        return _write_failure(
            reservation,
            "auth",
            "WorkspaceMismatch",
            "Modal token info did not match the expected workspace",
        )
    try:
        modal_version = importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError:
        return _write_failure(
            reservation, "modal_sdk", "SDKNotInstalled", "Modal 1.6.1 is required"
        )
    if modal_version != "1.6.1":
        return _write_failure(
            reservation, "modal_sdk", "SDKVersionMismatch", "Modal 1.6.1 is required"
        )
    try:
        modal = importlib.import_module("modal")
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .env({"USE_HUB_KERNELS": "NO"})
            .pip_install(
                "torch==2.14.1+cu130",
                "transformers==5.18.0",
                "pydantic==2.13.5",
                extra_index_url="https://download.pytorch.org/whl/cu130",
            )
            .add_local_python_source("reflex_decisions", copy=True)
        )
        app = modal.App("reflex-qwen-first-inference-smoke", image=image)
        run_smoke = app.function(
            gpu="A10",
            cpu=(2.0, 2.0),
            memory=(8192, 8192),
            max_containers=1,
            min_containers=0,
            buffer_containers=0,
            scaledown_window=2,
            retries=0,
            single_use_containers=True,
            serialized=True,
            include_source=False,
            startup_timeout=300,
            timeout=900,
        )(_remote_inference)
        artifact_written = False
        with modal.enable_output(), app.run():
            result = run_smoke.remote()
            if not isinstance(result, dict):
                result = _failure(
                    "result_transfer",
                    "UnexpectedResult",
                    "remote smoke did not return a JSON object",
                )
            result.setdefault("provenance", {})["modal_sdk_version"] = modal_version
            smoke.write_json_artifact(reservation, result)
            artifact_written = True
        print(
            json.dumps(
                {"status": result.get("status", "failed"), "artifact": str(reservation.destination)}
            )
        )
        return 0 if result.get("status") == "passed" else 1
    except Exception as exc:
        if locals().get("artifact_written") or os.path.lexists(reservation.destination):
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "artifact": str(reservation.destination),
                        "stage": "app_exit",
                    }
                )
            )
            return 1
        return _write_failure(
            reservation, "modal_run", type(exc).__name__, "remote launch or result transfer failed"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch", action="store_true", help="explicitly launch one paid remote smoke"
    )
    parser.add_argument("--profile", help="named personal Modal profile")
    parser.add_argument("--workspace", help="expected Modal Workspace name")
    parser.add_argument("--output", help="new local JSON artifact path")
    args = parser.parse_args(argv)
    if not args.launch:
        print(json.dumps(_plan(), sort_keys=True, indent=2))
        return 0
    if (
        not args.profile
        or not args.profile.strip()
        or not args.workspace
        or not args.workspace.strip()
        or not args.output
    ):
        parser.error("--launch requires nonempty --profile, --workspace, and --output")
    try:
        with smoke.reserve_output(args.output) as reservation:
            return _launch(args.profile.strip(), args.workspace.strip(), reservation)
    except (OSError, ValueError) as exc:
        print(f"output preflight failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
