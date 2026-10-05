"""Raw candidate-logit adapter for the pinned Qwen 3.5 base checkpoint."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from reflex_decisions.rendering import compile_request

if TYPE_CHECKING:
    from reflex_decisions.schema import DecisionRequest

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
TEMPERATURE = 1.0
AUXILIARY_FORWARD_COUNT = 0
MAX_INPUT_TOKENS = 2048


def validate_loading_info(info: object) -> dict[str, list[object]]:
    required = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if not isinstance(info, dict) or any(key not in info for key in required):
        raise RuntimeError("Transformers returned incomplete loading diagnostics")
    diagnostics = {key: list(info[key]) for key in required}
    if any(diagnostics.values()):
        raise RuntimeError(f"checkpoint loading diagnostics are not clean: {diagnostics}")
    return diagnostics


def _optional_kernels() -> dict[str, bool]:
    return {
        name: importlib.util.find_spec(name) is not None
        for name in ("kernels", "fla", "causal_conv1d")
    }


def load_scorer() -> tuple[Callable[[DecisionRequest], dict[str, object]], dict[str, object]]:
    """Load the frozen text model on CUDA and return unscaled option logits."""

    if "transformers" in sys.modules and os.environ.get("USE_HUB_KERNELS") != "NO":
        raise RuntimeError("set USE_HUB_KERNELS=NO before importing Transformers")
    os.environ["USE_HUB_KERNELS"] = "NO"

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    optional_kernels = _optional_kernels()
    if any(optional_kernels.values()):
        raise RuntimeError("optional model-kernel packages must not be installed")

    checkpoint = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        token=False,
    )
    model, load_info = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        trust_remote_code=False,
        output_loading_info=True,
        attn_implementation="eager",
        use_kernels=False,
        use_safetensors=True,
        token=False,
    )
    diagnostics = validate_loading_info(load_info)

    expected_model = isinstance(model, Qwen3_5ForCausalLM)
    expected_config = isinstance(model.config, Qwen3_5TextConfig)
    tied = bool(
        model.config.tie_word_embeddings
        and model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    )
    no_meta = all(parameter.device.type != "meta" for parameter in model.parameters())
    layer_count = len(model.model.layers)
    if not (expected_model and expected_config and tied and no_meta and layer_count == 24):
        raise RuntimeError("Qwen text-model shape or checkpoint checks failed")

    model = model.to("cuda").eval()
    parameter_dtypes: dict[str, int] = {}
    parameter_count = 0
    for parameter in model.parameters():
        label = str(parameter.dtype)
        parameter_dtypes[label] = parameter_dtypes.get(label, 0) + parameter.numel()
        parameter_count += parameter.numel()

    def score(request: DecisionRequest) -> dict[str, object]:
        compiled = compile_request(request, tokenizer, max_tokens=MAX_INPUT_TOKENS)
        input_ids = torch.tensor([compiled.input_ids], dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        candidate_ids = torch.tensor(compiled.candidate_token_ids, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        raw_logits = output.logits[0, -1].index_select(0, candidate_ids).float().cpu().tolist()
        if len(raw_logits) != len(request.options) or any(
            not math.isfinite(value) for value in raw_logits
        ):
            raise RuntimeError("Qwen returned invalid candidate logits")
        return {
            "raw_logits": raw_logits,
            "input_tokens": len(compiled.input_ids),
            "prompt_sha256": compiled.prompt_hash,
        }

    provenance: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "temperature": TEMPERATURE,
        "auxiliary_forward_count": AUXILIARY_FORWARD_COUNT,
        "temperature_note": "raw logits; no checkpoint calibration applied",
        "model_class": type(model).__name__,
        "config_class": type(model.config).__name__,
        "layer_count": layer_count,
        "tied_embeddings": tied,
        "no_meta_parameters": no_meta,
        "parameter_count": parameter_count,
        "parameter_dtypes": parameter_dtypes,
        "effective_dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "attention_implementation": "eager",
        "use_kernels": False,
        "use_hub_kernels": os.environ["USE_HUB_KERNELS"],
        "optional_kernel_packages_present": optional_kernels,
        "load_diagnostics": diagnostics,
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        },
        "max_input_tokens": MAX_INPUT_TOKENS,
        "single_unpadded_request": True,
        "logits_to_keep": 1,
    }
    return score, provenance
