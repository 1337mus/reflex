"""CPU-safe request adapter for the pinned official Kev-0.8B checkpoint."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reflex_decisions.schema import DecisionRequest

MODEL_ID = "jaredpalmer/kev-0.8b"
REVISION = "9a45d25eb2ab761841196625383fa1dff0e56c1e"
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
BASE_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SOURCE_COMMIT = "45923b7a3460b6d36358e2e143455902c1eb856b"
SOURCE_SHA256 = {
    "model.py": "2634ffe7747d69473fb6596942c248e5df2586df3610bd1adb47e7e9acd99f96",
    "checkpoint.py": "f3edb4d159c0aa12d7006fc599fe6ebc3f2c288b3b69890c289f7dac451dd03d",
}
SAFE_CHECKPOINT_SHA256 = "22b5200862dfa63c723c7979097868eb8b0ebfb4f64570c21400506196705860"
MAX_INPUT_TOKENS = 2048
AUXILIARY_FORWARD_COUNT = 2


def _request_to_kev_record(request: DecisionRequest) -> dict[str, object]:
    """Convert Reflex text fields to Kev's native record while retaining candidate order."""

    options = [
        f"{option.label}: {option.description}" if option.description is not None else option.label
        for option in request.options
    ]
    return {
        "state": request.context,
        "questions": [
            {
                "instr": request.question,
                "options": options,
                "label": 0,
            }
        ],
    }


def _prompt_sha256(token_ids: Sequence[int]) -> str:
    """Hash the exact encoded token sequence with an unambiguous canonical representation."""

    if any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError("encoded token IDs must be nonnegative integers")
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _validate_encoded(encoded: Mapping[str, object]) -> list[int]:
    """Reject truncation and over-limit encodings before the official model sees them."""

    ids = encoded.get("ids")
    if not isinstance(ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in ids
    ):
        raise ValueError("Kev encoder must return nonnegative integer token IDs")
    if encoded.get("state_truncated") is True:
        raise ValueError("Kev encoder unexpectedly truncated state")
    if len(ids) > MAX_INPUT_TOKENS:
        raise ValueError(f"Kev encoding exceeds {MAX_INPUT_TOKENS}-token limit")
    return ids


def _fetch_official_sources(source_root: Path) -> None:
    """Fetch only the two reviewed modules from the pinned upstream source commit."""

    from urllib.request import Request, urlopen

    source_root.mkdir(parents=True, exist_ok=True)
    source_url = f"https://raw.githubusercontent.com/jaredpalmer/kev/{SOURCE_COMMIT}/kev"
    for filename in ("model.py", "checkpoint.py"):
        request = Request(
            f"{source_url}/{filename}",
            headers={"User-Agent": "reflex-decisions-baseline"},
        )
        with urlopen(request, timeout=30) as response:
            source = response.read()
        digest = hashlib.sha256(source).hexdigest()
        if digest != SOURCE_SHA256[filename]:
            raise RuntimeError(
                f"pinned Kev source hash mismatch for {filename}: expected "
                f"{SOURCE_SHA256[filename]}, got {digest}"
            )
        (source_root / f"checkpoint-kev_{filename}").write_bytes(source)


def _read_verified_source(source_root: Path, filename: str) -> bytes:
    path = source_root / f"checkpoint-kev_{filename}"
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != SOURCE_SHA256[filename]:
        raise RuntimeError(
            f"pinned Kev source hash mismatch for {filename}: expected "
            f"{SOURCE_SHA256[filename]}, got {digest}"
        )
    return source


def _install_official_modules(source_root: Path) -> tuple[Any, Any, str]:
    """Import the two provenance-pinned upstream modules under an isolated package name."""

    import sys
    import types

    import_source = {
        filename: _read_verified_source(source_root, filename)
        for filename in ("model.py", "checkpoint.py")
    }
    package_name = f"_reflex_kev_{REVISION[:8]}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_root)]
    sys.modules[package_name] = package

    model_module = types.ModuleType(f"{package_name}.model")
    model_module.__file__ = str(source_root / "checkpoint-kev_model.py")
    model_module.__package__ = package_name
    sys.modules[model_module.__name__] = model_module
    exec(
        compile(import_source["model.py"], model_module.__file__, "exec"),
        model_module.__dict__,
    )

    loader_bytes = import_source["checkpoint.py"]
    safe_load_anchor = b'torch.load(f"{run}/head.pt", map_location="cpu")'
    if loader_bytes.count(safe_load_anchor) != 1:
        raise RuntimeError("pinned Kev metadata loader no longer has one expected torch.load call")
    safe_loader_bytes = loader_bytes.replace(
        safe_load_anchor,
        b'torch.load(f"{run}/head.pt", map_location="cpu", weights_only=True)',
    )
    safe_loader_sha256 = hashlib.sha256(safe_loader_bytes).hexdigest()
    if safe_loader_sha256 != SAFE_CHECKPOINT_SHA256:
        raise RuntimeError("restricted-weight Kev loader source hash mismatch")
    loader_module = types.ModuleType(f"{package_name}.checkpoint")
    loader_module.__file__ = str(source_root / "checkpoint-kev_checkpoint.py")
    loader_module.__package__ = package_name
    sys.modules[loader_module.__name__] = loader_module
    exec(
        compile(safe_loader_bytes, loader_module.__file__, "exec"),
        loader_module.__dict__,
    )
    return loader_module, model_module, safe_loader_sha256


def _assert_tensor_state(
    expected: Mapping[str, Any], actual: Mapping[str, Any], label: str, torch: Any
) -> None:
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"{label} tensor keys differ; missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for key, expected_tensor in expected.items():
        actual_tensor = actual[key]
        if tuple(expected_tensor.shape) != tuple(actual_tensor.shape):
            raise RuntimeError(f"{label} tensor shape differs for {key}")
        if not torch.equal(expected_tensor.cpu(), actual_tensor.cpu()):
            raise RuntimeError(f"{label} tensor values differ for {key}")
        if not torch.isfinite(actual_tensor).all():
            raise RuntimeError(f"{label} contains a non-finite tensor: {key}")


def _validate_temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Kev checkpoint temperature must be a finite positive number")
    temperature = float(value)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Kev checkpoint temperature must be a finite positive number")
    return temperature


def _validate_base_loading_info(info: object) -> dict[str, list[object]]:
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if not isinstance(info, dict) or any(key not in info for key in keys):
        raise RuntimeError("Transformers returned incomplete base-model loading diagnostics")
    diagnostics = {key: list(info[key]) for key in keys}
    failures = {key: values for key, values in diagnostics.items() if values}
    if failures:
        raise RuntimeError(f"base-model loading diagnostics are not clean: {failures}")
    return diagnostics


def _load_official_checkpoint(
    checkpoint: Any, loader_module: Any, model_module: Any, torch: Any
) -> tuple[Any, Any, dict[str, list[object]]]:
    """Capture the base-model loader's public diagnostics and restore its method."""

    model_class = model_module.AutoModelForCausalLM
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
        diagnostics.update(_validate_base_loading_info(info))
        return model

    model_class.from_pretrained = staticmethod(checked)
    try:
        tokenizer, model = checkpoint.load(
            "cuda",
            loader_module.LoadOptions(
                dtype=torch.float32,
                merge=False,
                attn="eager",
                temperature=1.0,
                backend="torch",
                cuda_graphs=False,
            ),
        )
    finally:
        if old is inherited:
            delattr(model_class, "from_pretrained")
        else:
            model_class.from_pretrained = old
    if not diagnostics:
        raise RuntimeError("Kev base-model loading diagnostics were not captured")
    return tokenizer, model, diagnostics


def _assert_effective_fp32_cuda(model: Any, torch: Any) -> tuple[dict[str, int], int, str, str]:
    """Require every model parameter to be materialized as CUDA float32."""

    parameters = list(model.parameters())
    if not parameters:
        raise RuntimeError("Kev model has no parameters")
    buffers = list(model.buffers())
    if any(tensor.is_meta for tensor in (*parameters, *buffers)):
        raise RuntimeError("Kev model contains unmaterialized meta tensors")
    if any(parameter.device.type != "cuda" for parameter in parameters):
        raise RuntimeError("Kev model has parameters outside the CUDA device")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("Kev model does not match the expected effective float32 dtype")
    dtypes: dict[str, int] = {}
    parameter_count = 0
    for parameter in parameters:
        dtype = str(parameter.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + parameter.numel()
        parameter_count += parameter.numel()
    return dtypes, parameter_count, str(parameters[0].dtype), str(parameters[0].device)


def _disable_backbone_cache(model: Any) -> bool:
    """Override the checkpoint's Qwen cache default before any official forward call."""

    model.lm.config.use_cache = False
    if model.lm.config.use_cache is not False:
        raise RuntimeError("Kev backbone cache must be disabled for the reference path")
    return model.lm.config.use_cache


def load_scorer() -> tuple[Callable[[DecisionRequest], dict[str, object]], dict[str, object]]:
    """Load the pinned native Kev scorer on CUDA; heavy runtime imports happen only here."""

    import os
    import tempfile

    os.environ["USE_HUB_KERNELS"] = "NO"
    torch = importlib.import_module("torch")
    load_file = importlib.import_module("safetensors.torch").load_file

    if not torch.cuda.is_available():
        raise RuntimeError("the Kev baseline scorer requires the authorized remote CUDA runner")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    source_lifetime = tempfile.TemporaryDirectory(prefix="reflex-kev-source-")
    source_root = Path(source_lifetime.name)
    _fetch_official_sources(source_root)
    loader_module, model_module, safe_loader_sha256 = _install_official_modules(source_root)
    checkpoint = loader_module.Checkpoint(f"{MODEL_ID}@{REVISION}")
    if checkpoint.meta.base != BASE_MODEL_ID or checkpoint.meta.base_revision != BASE_REVISION:
        raise RuntimeError("Kev checkpoint metadata does not match the pinned Qwen base revision")
    temperature = _validate_temperature(checkpoint.meta.temperature)
    if checkpoint.meta.lora != 16 or checkpoint.meta.head_dim != 256:
        raise RuntimeError("Kev adapter/head metadata differs from the reviewed release")

    tokenizer, model, load_diagnostics = _load_official_checkpoint(
        checkpoint, loader_module, model_module, torch
    )

    adapter_path = checkpoint.file("adapter_model.safetensors")
    expected_adapter = load_file(str(adapter_path), device="cpu")
    get_peft_model_state_dict = importlib.import_module("peft").get_peft_model_state_dict

    loaded_adapter = get_peft_model_state_dict(model.lm, adapter_name="default")
    _assert_tensor_state(expected_adapter, loaded_adapter, "Kev LoRA adapter", torch)
    _assert_tensor_state(checkpoint.meta.head, model.head.state_dict(), "Kev pointer head", torch)

    # Match the official default loader path after inspecting every loaded PEFT tensor.
    model.lm = model.lm.merge_and_unload()
    model.eval()
    use_cache = _disable_backbone_cache(model)
    parameter_dtypes, parameter_count, effective_dtype, device = _assert_effective_fp32_cuda(
        model, torch
    )

    probe = {
        "state": "A synthetic calibration-path check.",
        "questions": [{"instr": "Choose one.", "options": ["first", "second"], "label": 0}],
    }
    probe_encoding = model.encode(tokenizer, probe, max_state=2048, max_branch=2048, strict=True)
    probe_ids = _validate_encoded(probe_encoding)
    with torch.inference_mode():
        model.head.temperature = 1.0
        probe_raw = model.forward(probe_encoding)[0].float()
        model.head.temperature = temperature
        probe_shipped = model.forward(probe_encoding)[0].float()
        expected_probabilities = torch.softmax(probe_raw / temperature, dim=-1)
        shipped_probabilities = torch.softmax(probe_shipped, dim=-1)
    model.head.temperature = 1.0
    if not torch.allclose(expected_probabilities, shipped_probabilities, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Kev raw-logit temperature path failed probability parity")

    def score(request: DecisionRequest) -> dict[str, object]:
        if not Path(source_lifetime.name).is_dir():
            raise RuntimeError("verified Kev source directory expired before scorer use")
        record = _request_to_kev_record(request)
        encoded = model.encode(tokenizer, record, max_state=2048, max_branch=2048, strict=True)
        input_ids = _validate_encoded(encoded)
        with torch.inference_mode():
            outputs = model.forward(encoded)
        if len(outputs) != 1 or len(outputs[0]) != len(request.options):
            raise RuntimeError("Kev returned a different number of option scores than requested")
        raw_logits = outputs[0].float().cpu().tolist()
        if any(not math.isfinite(value) for value in raw_logits):
            raise RuntimeError("Kev returned a non-finite option score")
        return {
            "raw_logits": raw_logits,
            "input_tokens": len(input_ids),
            "prompt_sha256": _prompt_sha256(input_ids),
        }

    provenance: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": dict(SOURCE_SHA256),
        "safe_checkpoint_module_sha256": safe_loader_sha256,
        "adapter_sha256": _file_sha256(adapter_path),
        "head_sha256": _file_sha256(checkpoint.file("head.pt")),
        "adapter_tensor_count": len(expected_adapter),
        "temperature": temperature,
        "auxiliary_forward_count": AUXILIARY_FORWARD_COUNT,
        "temperature_parity_max_abs": float(
            torch.max(torch.abs(expected_probabilities - shipped_probabilities)).item()
        ),
        "runtime": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "device": device,
            "dtype": effective_dtype,
            "parameter_dtypes": parameter_dtypes,
            "parameter_count": parameter_count,
            "effective_dtype": effective_dtype,
            "attention": "eager",
            "use_cache": use_cache,
            "use_hub_kernels": os.environ["USE_HUB_KERNELS"],
            "upstream_torch_range": ">=2.6,<2.9",
            "upstream_torch_range_satisfied": _torch_in_official_range(torch.__version__),
        },
        "load_diagnostics": load_diagnostics,
        "probe_input_tokens": len(probe_ids),
    }
    return score, provenance


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_in_official_range(version: str) -> bool:
    core = version.split("+", maxsplit=1)[0]
    try:
        major, minor = (int(part) for part in core.split(".")[:2])
    except (ValueError, TypeError):
        return False
    return (major, minor) >= (2, 6) and (major, minor) < (2, 9)
