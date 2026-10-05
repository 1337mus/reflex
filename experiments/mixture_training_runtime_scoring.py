"""Scoring and immutable adapter snapshot verification for the remote worker."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments import mixture_training_core as core
from experiments import mixture_training_runtime_helpers as helpers


def _file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _score_forward(
    model: Any,
    request: Any,
    tokenizer: Any,
    torch: Any,
    result: dict[str, object],
    category: str,
) -> tuple[Any, Any]:
    from reflex_decisions.rendering import compile_request

    compiled = compile_request(request, tokenizer, max_tokens=core.MAX_INPUT_TOKENS)
    input_ids = torch.tensor([compiled.input_ids], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
    )
    helpers.increment_completed_forward(result["evidence"], category, len(compiled.input_ids))
    candidate_ids = torch.tensor(compiled.candidate_token_ids, dtype=torch.long, device="cuda")
    candidate_logits = output.logits[0, -1].index_select(0, candidate_ids).float()
    if candidate_logits.numel() != len(request.options):
        raise RuntimeError("candidate-only output count did not match the request")
    if not bool(torch.isfinite(candidate_logits).all().item()):
        raise RuntimeError("candidate logits contained a non-finite value")
    return candidate_logits, compiled


def _scored_row(presentation: dict[str, object], logits: Any, compiled: Any) -> dict[str, object]:
    values = [float(value) for value in logits.detach().cpu().tolist()]
    order_ids = presentation["order_ids"]
    maximum = max(values)
    winner = min(
        option_id for option_id, score in zip(order_ids, values, strict=True) if score == maximum
    )
    identity = {
        key: presentation[key]
        for key in (
            "presentation_id",
            "record_id",
            "dataset_id",
            "source_group_id",
            "request_hash",
            "order_index",
            "order_ids",
        )
    }
    return identity | {
        "candidate_logits": values,
        "winner_option_id": winner,
        "input_tokens": len(compiled.input_ids),
        "prompt_sha256": compiled.prompt_hash,
    }


def _score_presentations(
    model: Any,
    tokenizer: Any,
    presentations: list[dict[str, object]],
    torch: Any,
    result: dict[str, object],
    category: str,
    run_dir: Path,
    *,
    write_progress: Callable[[dict[str, object], Path], None],
    retain_outputs: bool = True,
) -> list[dict[str, object]]:
    from reflex_decisions.schema import DecisionRequest

    outputs: list[dict[str, object]] = []
    for index, presentation in enumerate(presentations, start=1):
        request = DecisionRequest.model_validate(presentation["request"])
        with torch.inference_mode():
            logits, compiled = _score_forward(model, request, tokenizer, torch, result, category)
        outputs.append(_scored_row(presentation, logits, compiled))
        if retain_outputs:
            result["evidence"]["outputs"] = outputs
        if index % 128 == 0:
            result["phase"] = f"{category}_{index}"
            write_progress(result, run_dir)
    return outputs


def _save_outputs(
    result: dict[str, object],
    run_dir: Path,
    write_progress: Callable[[dict[str, object], Path], None],
) -> None:
    evidence = result["evidence"]
    path = run_dir / "outputs.json"
    digest = helpers.write_json_exclusive(path, evidence["outputs"])
    evidence["output_artifact"] = {
        "path": f"/artifacts/runs/{run_dir.name}/outputs.json",
        "sha256": digest,
    }
    write_progress(result, run_dir)


def _state_dict(model: Any, peft_module: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in peft_module.get_peft_model_state_dict(model).items()
    }


def _equal_states(left: dict[str, Any], right: dict[str, Any], torch: Any) -> None:
    if set(left) != set(right):
        raise RuntimeError("adapter tensor keys differ from the saved safetensors")
    for name in sorted(left):
        if left[name].shape != right[name].shape:
            raise RuntimeError("adapter tensor shapes differ from saved safetensors")
        if left[name].dtype != torch.float32 or right[name].dtype != torch.float32:
            raise RuntimeError("adapter tensor values are not FP32")
        if not torch.equal(left[name], right[name]):
            raise RuntimeError("adapter tensor values differ from saved safetensors")


def _snapshot_hashes(snapshot: dict[str, object]) -> dict[str, str]:
    directory = Path(str(snapshot["path"]))
    actual = {path.name: _file_hash(path) for path in sorted(directory.iterdir()) if path.is_file()}
    if actual != snapshot["files_sha256"]:
        raise ValueError("adapter snapshot file hashes differ from its saved receipt")
    return actual


def _verify_saved_adapter(
    model: Any,
    snapshot: dict[str, object],
    expected_tensor_sha256: str,
    peft_module: Any,
    torch: Any,
) -> dict[str, Any]:
    before = _snapshot_hashes(snapshot)
    safe_tensors = importlib.import_module("safetensors.torch")
    saved = safe_tensors.load_file(
        str(Path(str(snapshot["path"])) / "adapter_model.safetensors"), device="cpu"
    )
    actual = _state_dict(model, peft_module)
    _equal_states(actual, saved, torch)
    digest = helpers.tensor_state_sha256(actual, torch_module=torch)
    if digest != expected_tensor_sha256:
        raise ValueError("loaded adapter tensor digest differs from the initialization pin")
    if _snapshot_hashes(snapshot) != before:
        raise ValueError("adapter snapshot files changed while being verified")
    return actual
