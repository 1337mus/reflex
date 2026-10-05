"""Adapter for the pinned official Intern-Decision inference implementation."""

import hashlib
import importlib.metadata
import importlib.util
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from reflex_decisions.schema import DecisionRequest

MODEL_ID = "internlm/Intern-Decision-0.8B"
REVISION = "85a0cc5a99d67ea8d56dfe98115689212867171d"
TEMPERATURE = 2.747760550703
AUXILIARY_FORWARD_COUNT = 1
MODULE_SHA256 = "62664f5bebb593e825370f82b2733edd13b16118b436d09db2f0c36405b9b4ea"
SOURCE_COMMIT = "3572c8a68b5df5dafe02d0e093989ba8ec0183bc"
MODULE_SOURCE = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/inference.py"


def verify_module_sha256(path: str | Path, expected: str = MODULE_SHA256) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"official inference source SHA-256 mismatch: {digest}")


def validate_loading_info(info: object) -> dict[str, list[object]]:
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if not isinstance(info, dict) or any(key not in info for key in keys):
        raise RuntimeError("Transformers returned incomplete loading diagnostics")
    diagnostics = {key: list(info[key]) for key in keys}
    failures = {key: values for key, values in diagnostics.items() if values}
    if failures:
        raise RuntimeError(f"checkpoint loading diagnostics are not clean: {failures}")
    return diagnostics


def checked_candidate_token_ids(
    tokenizer: Any, symbols: tuple[str, ...], option_count: int, vocab_size: int
) -> list[int]:
    if len(symbols) != option_count:
        raise ValueError("decision symbol count does not match option count")
    ids = []
    for symbol in symbols:
        encoded = tokenizer.encode(symbol, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError("each candidate symbol must encode to exactly one token")
        token_id = encoded[0]
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocab_size
        ):
            raise ValueError("candidate symbol token ID is outside the checkpoint vocabulary")
        ids.append(token_id)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate symbol token IDs must be distinct")
    return ids


def build_official_request(request: DecisionRequest) -> dict[str, object]:
    criteria = {
        option.id: option.label
        if option.description is None
        else f"{option.label}: {option.description}"
        for option in request.options
    }
    return {
        "state": request.context,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": request.question,
                "criteria": criteria,
            }
        },
    }


def _record_calibration_parity(
    provenance: dict[str, object],
    computed: Sequence[object],
    option_ids: Sequence[str],
    official: Mapping[str, object],
) -> None:
    if len(computed) != len(option_ids) or set(official) != set(option_ids):
        raise RuntimeError("temperature check disagrees with official predict()")
    for value, option_id in zip(computed, option_ids, strict=True):
        expected = official[option_id]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or isinstance(expected, bool)
            or not isinstance(expected, (int, float))
            or not math.isfinite(value)
            or not math.isfinite(expected)
            or abs(value - expected) > 1e-5
        ):
            raise RuntimeError("temperature check disagrees with official predict()")
    provenance["calibration_parity_passed"] = True
    provenance["auxiliary_forward_count"] = AUXILIARY_FORWARD_COUNT


def _import_reviewed_module(path: Path) -> Any:
    name = f"_intern_decision_{REVISION[:12]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create import spec for official inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_engine(module: Any, checkpoint: Path) -> tuple[Any, dict[str, list[object]]]:
    published_temperature = getattr(module, "DEFAULT_TEMPERATURE", None)
    if (
        isinstance(published_temperature, bool)
        or not isinstance(published_temperature, (int, float))
        or float(published_temperature) != TEMPERATURE
    ):
        raise RuntimeError("official inference module has an unexpected published temperature")
    model_class = module.Qwen3_5ForConditionalGeneration
    inherited = object()
    old = model_class.__dict__.get("from_pretrained", inherited)
    original, diagnostics = model_class.from_pretrained, {}

    def checked(*args: Any, **kwargs: Any) -> Any:
        kwargs.update(
            output_loading_info=True,
            use_safetensors=True,
            use_kernels=False,
            trust_remote_code=False,
        )
        model, info = original(*args, **kwargs)
        diagnostics.update(validate_loading_info(info))
        return model

    model_class.from_pretrained = staticmethod(checked)
    try:
        engine = module.DecisionEngine(
            checkpoint=str(checkpoint),
            temperature=TEMPERATURE,
            device="cuda",
            dtype="float32",
            attn_implementation="eager",
            max_length=2048,
        )
    finally:
        if old is inherited:
            delattr(model_class, "from_pretrained")
        else:
            model_class.from_pretrained = old
    return engine, diagnostics


def load_scorer() -> tuple[Callable[[DecisionRequest], dict[str, object]], dict[str, object]]:
    """Load official pinned code/weights and return raw candidate logits plus provenance."""
    if "transformers" in sys.modules and os.environ.get("USE_HUB_KERNELS") != "NO":
        raise RuntimeError("set USE_HUB_KERNELS=NO before importing Transformers")
    os.environ["USE_HUB_KERNELS"] = "NO"

    import torch
    import transformers
    from huggingface_hub import snapshot_download

    checkpoint = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION))
    source = checkpoint / "inference.py"
    verify_module_sha256(source)
    module = _import_reviewed_module(source)
    engine, diagnostics = _load_engine(module, checkpoint)
    model = engine.backend.model
    if any(
        importlib.util.find_spec(name) is not None for name in ("kernels", "fla", "causal_conv1d")
    ):
        raise RuntimeError("optional model-kernel packages must not be installed")
    dtypes: dict[str, int] = {}
    count = 0
    for parameter in model.parameters():
        dtype = str(parameter.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
        count += parameter.numel()
    parity_checked = False
    provenance: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "source_commit": SOURCE_COMMIT,
        "module_source": MODULE_SOURCE,
        "module_sha256": MODULE_SHA256,
        "temperature": TEMPERATURE,
        "auxiliary_forward_count": 0,
        "temperature_note": "Hub default; repository preset is XTuner-bound",
        "model_class": type(model).__name__,
        "config_class": type(model.config).__name__,
        "parameter_count": count,
        "parameter_dtypes": dtypes,
        "effective_dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "attention_implementation": getattr(model.config, "_attn_implementation", "eager"),
        "use_hub_kernels": os.environ["USE_HUB_KERNELS"],
        "kernel_packages_present": {
            name: importlib.util.find_spec(name) is not None
            for name in ("kernels", "fla", "causal_conv1d")
        },
        "load_diagnostics": diagnostics,
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        },
        "max_length": 2048,
        "option_ids_visible_in_prompt": True,
        "calibration_parity_passed": False,
    }

    def score(request: DecisionRequest) -> dict[str, object]:
        nonlocal parity_checked
        row = build_official_request(request)
        compiled, logits, token_count, _ = engine.backend.score(row)
        if compiled.fields != ("decision",):
            raise RuntimeError("official prompt compiler returned unexpected fields")
        vocab_size = model.get_input_embeddings().weight.shape[0]
        ids = checked_candidate_token_ids(
            engine.tokenizer, compiled.symbols["decision"], len(request.options), vocab_size
        )
        values = logits[0, ids].float().cpu().tolist()
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("official scorer returned non-finite candidate logits")
        prompt = engine.backend.tokenizer.apply_chat_template(
            compiled.messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            add_vision_id=True,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if not parity_checked:
            official = engine.predict(row)["answers"]["decision"]["probabilities"]
            probabilities = torch.softmax(torch.tensor(values) / TEMPERATURE, dim=0).tolist()
            _record_calibration_parity(
                provenance,
                probabilities,
                [option.id for option in request.options],
                official,
            )
            parity_checked = True
        return {
            "raw_logits": values,
            "input_tokens": int(token_count),
            "prompt_sha256": prompt_hash,
        }

    return score, provenance
