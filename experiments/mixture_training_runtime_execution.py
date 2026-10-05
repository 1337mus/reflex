"""GPU initialization and paired training execution for the remote worker."""

from __future__ import annotations

import gc
import importlib
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments import mixture_training_core as core
from experiments import mixture_training_runtime_helpers as helpers
from experiments import mixture_training_runtime_scoring as scoring
from experiments.mixture_training_contracts import (
    LEARNING_RATE,
    MAX_GRADIENT_NORM,
    WEIGHT_DECAY,
)


def _base_model(
    AutoModelForCausalLM: Any, AutoTokenizer: Any, torch: Any
) -> tuple[Any, Any, dict[str, object]]:
    training_helpers = importlib.import_module("experiments.modal_train_rehearsal")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    model, tokenizer, provenance = training_helpers._load_base(
        AutoTokenizer, AutoModelForCausalLM, torch
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer has no usable pad or EOS token")
    return model, tokenizer, provenance


def run_initialization(
    payload: dict[str, object],
    result: dict[str, object],
    run_dir: Path,
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    LoraConfig: Any,
    get_peft_model: Any,
    write_progress: Callable[[dict[str, object], Path], None],
) -> None:
    from experiments import mixture_training_data as data

    training_helpers = importlib.import_module("experiments.modal_train_rehearsal")
    model, tokenizer, base_provenance = _base_model(AutoModelForCausalLM, AutoTokenizer, torch)
    result["provenance"]["base_model"] = base_provenance
    result["provenance"]["cuda_device"] = torch.cuda.get_device_name(0)
    presentations = payload["evaluation_presentations"]
    outputs = scoring._score_presentations(
        model,
        tokenizer,
        presentations,
        torch,
        result,
        "base_evaluation",
        run_dir,
        write_progress=write_progress,
    )
    scoring._save_outputs(result, run_dir, write_progress)

    model.requires_grad_(False)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(training_helpers.core.LORA_TARGET_MODULES),
        task_type="CAUSAL_LM",
    )
    random.seed(data.INIT_SEED)
    torch.manual_seed(data.INIT_SEED)
    torch.cuda.manual_seed_all(data.INIT_SEED)
    adapted = get_peft_model(model, lora_config)
    inventory = training_helpers._validate_adapter_inventory(adapted, torch, require_trainable=True)
    state = scoring._state_dict(adapted, importlib.import_module("peft"))
    digest = helpers.tensor_state_sha256(state, torch_module=torch)
    snapshot = training_helpers._save_adapter_snapshot(adapted, run_dir, 0)
    scoring._verify_saved_adapter(adapted, snapshot, digest, importlib.import_module("peft"), torch)
    result["evidence"]["adapter_inventory"] = inventory
    result["evidence"]["initialization"] = {
        "snapshot": snapshot,
        "tensor_sha256": digest,
    }
    result["phase"] = "initialize"
    if len(outputs) != data.EXPECTED_SYNTHETIC_PRESENTATIONS:
        raise RuntimeError("initialization did not score the exact synthetic evaluation panel")
    write_progress(result, run_dir)


def run_training(
    payload: dict[str, object],
    result: dict[str, object],
    run_dir: Path,
    torch: Any,
    functional: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    PeftModel: Any,
    peft_module: Any,
    write_progress: Callable[[dict[str, object], Path], None],
) -> None:
    from experiments import mixture_training_data as data

    training_helpers = importlib.import_module("experiments.modal_train_rehearsal")
    base_model, tokenizer, base_provenance = _base_model(AutoModelForCausalLM, AutoTokenizer, torch)
    result["provenance"]["base_model"] = base_provenance
    result["provenance"]["cuda_device"] = torch.cuda.get_device_name(0)
    descriptor = payload["initialization"]
    snapshot = descriptor["snapshot"]
    scoring._snapshot_hashes(snapshot)
    adapted = PeftModel.from_pretrained(
        base_model,
        str(snapshot["path"]),
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    inventory = training_helpers._validate_adapter_inventory(adapted, torch, require_trainable=True)
    initial_state = scoring._verify_saved_adapter(
        adapted,
        snapshot,
        descriptor["tensor_sha256"],
        peft_module,
        torch,
    )
    result["evidence"].update(
        {
            "adapter_inventory": inventory,
            "initialization_verified": True,
            "initial_tensor_sha256": descriptor["tensor_sha256"],
        }
    )
    trainable = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    examples = core.training_examples(payload)
    if len(examples) != data.TRAINING_PRESENTATIONS:
        raise RuntimeError("regenerated training schedule has the wrong presentation count")
    result["phase"] = "training"
    result["evidence"]["base_gradients_none"] = True
    result["evidence"]["training_step_losses"] = []
    result["evidence"]["adapter_paths"] = {}
    b_gradient_l1_total = 0.0
    adapted.train()

    for update in range(1, data.MAX_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        update_examples = examples[(update - 1) * 4 : update * 4]
        if len(update_examples) != data.MICROBATCHES_PER_UPDATE:
            raise RuntimeError("optimizer update did not receive exactly four examples")
        microbatch_losses: list[float] = []
        for example in update_examples:
            logits, _compiled = scoring._score_forward(
                adapted, example.request, tokenizer, torch, result, "training"
            )
            loss = functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([example.gold_index], dtype=torch.long, device="cuda"),
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("training loss was non-finite")
            microbatch_losses.append(float(loss.detach().item()))
            (loss / data.MICROBATCHES_PER_UPDATE).backward()

        b_gradients = [
            parameter.grad
            for name, parameter in adapted.named_parameters()
            if ".lora_B." in name and parameter.grad is not None
        ]
        gradients = [
            parameter.grad for parameter in adapted.parameters() if parameter.grad is not None
        ]
        if not b_gradients or any(
            not bool(torch.isfinite(gradient).all().item()) for gradient in gradients
        ):
            raise RuntimeError("LoRA-B gradients were missing or non-finite")
        b_gradient_l1 = sum(float(gradient.detach().abs().sum().item()) for gradient in b_gradients)
        if not math.isfinite(b_gradient_l1):
            raise RuntimeError("LoRA-B gradient evidence was non-finite")
        b_gradient_l1_total += b_gradient_l1
        if any(
            parameter.grad is not None
            for name, parameter in adapted.named_parameters()
            if ".lora_A." not in name and ".lora_B." not in name
        ):
            result["evidence"]["base_gradients_none"] = False
            raise RuntimeError("a frozen backbone parameter received a gradient")
        norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRADIENT_NORM)
        if not bool(torch.isfinite(norm).item()):
            raise RuntimeError("LoRA gradient norm was non-finite")
        optimizer.step()
        step_losses = result["evidence"]["training_step_losses"]
        step_losses.append(
            {
                "update": update,
                "mean_loss": sum(microbatch_losses) / len(microbatch_losses),
                "lo_ra_b_gradient_l1": b_gradient_l1,
                "gradient_norm": float(norm.item()),
            }
        )
        result["evidence"]["optimizer_updates_completed"] = update
        if update in (126, 252):
            adapted.eval()
            snapshot_result = training_helpers._save_adapter_snapshot(adapted, run_dir, update)
            result["evidence"]["adapter_paths"][str(update)] = snapshot_result
            write_progress(result, run_dir)
            if update == 126:
                adapted.train()
        elif update % 16 == 0:
            write_progress(result, run_dir)

    final_snapshot = result["evidence"]["adapter_paths"]["252"]
    final_state = scoring._state_dict(adapted, peft_module)
    update_evidence = training_helpers._adapter_state_changes(initial_state, final_state, torch)
    if update_evidence["changed_tensor_count"] == 0:
        raise RuntimeError("training did not change any LoRA adapter tensor")
    if not math.isfinite(b_gradient_l1_total) or b_gradient_l1_total <= 0.0:
        raise RuntimeError("training produced no nonzero LoRA-B gradient evidence")
    result["evidence"]["adapter_update"] = update_evidence
    scoring._snapshot_hashes(final_snapshot)
    presentations = payload["evaluation_presentations"]
    adapted.eval()
    final_outputs = scoring._score_presentations(
        adapted,
        tokenizer,
        presentations,
        torch,
        result,
        "final_evaluation",
        run_dir,
        write_progress=write_progress,
    )
    scoring._save_outputs(result, run_dir, write_progress)
    del optimizer, trainable, adapted, base_model, logits, loss, b_gradients, gradients
    gc.collect()
    torch.cuda.empty_cache()

    final_base, final_tokenizer, final_provenance = _base_model(
        AutoModelForCausalLM, AutoTokenizer, torch
    )
    if final_provenance.get("tokenizer_file_sha256") != base_provenance.get(
        "tokenizer_file_sha256"
    ):
        raise RuntimeError("fresh reload used different pinned tokenizer files")
    reloaded = PeftModel.from_pretrained(
        final_base,
        str(final_snapshot["path"]),
        is_trainable=False,
        autocast_adapter_dtype=True,
    ).eval()
    training_helpers._validate_adapter_inventory(reloaded, torch, require_trainable=False)
    reloaded_state = scoring._verify_saved_adapter(
        reloaded,
        final_snapshot,
        helpers.tensor_state_sha256(final_state, torch_module=torch),
        peft_module,
        torch,
    )
    scoring._equal_states(final_state, reloaded_state, torch)

    parity_presentations = sorted(presentations, key=lambda row: row["presentation_id"])[
        : core.RELOAD_PARITY_COUNT
    ]
    parity_outputs = scoring._score_presentations(
        reloaded,
        final_tokenizer,
        parity_presentations,
        torch,
        result,
        "reload_parity",
        run_dir,
        write_progress=write_progress,
        retain_outputs=False,
    )
    by_id = {row["presentation_id"]: row for row in final_outputs}
    max_difference = 0.0
    for row in parity_outputs:
        final_row = by_id[row["presentation_id"]]
        if row["winner_option_id"] != final_row["winner_option_id"]:
            raise RuntimeError("fresh reload winner differs from final evaluation")
        max_difference = max(
            max_difference,
            *(
                abs(left - right)
                for left, right in zip(
                    row["candidate_logits"], final_row["candidate_logits"], strict=True
                )
            ),
        )
    if max_difference > 1e-3:
        raise RuntimeError("fresh reload candidate logits differ by more than 1e-3")
    result["evidence"]["reload_parity"] = {
        "tensor_values_exact": True,
        "winner_match": True,
        "max_abs_difference": max_difference,
        "outputs": parity_outputs,
    }
    result["evidence"]["adapter_inventory"] = inventory
    result["evidence"]["base_gradients_none"] = True
    write_progress(result, run_dir)
    result["phase"] = "train"
