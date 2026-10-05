"""Remote continuation-training loop for the runtime-rule study."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_payloads as payload_api
from experiments import runtime_rule_study_progress as progress_api
from experiments import runtime_rule_study_provenance as provenance_api
from experiments import runtime_rule_study_training_evidence as training_evidence_api


def train_adapter(
    payload: object,
    result: dict[str, object],
    model: Any,
    prepared: Any,
    initial_state: Mapping[str, Any],
    run_dir: str | Path,
    torch: Any,
    peft_module: Any,
    ledger: Any,
    *,
    on_progress: Callable[[dict[str, object], Path], None] | None = None,
) -> dict[str, Any]:
    """Train one fixed continuation arm from its selected adapter state."""

    authenticated_payload = payload_api.validate_payload(payload)
    run_id = contracts.validate_safe_run_id(authenticated_payload["run_id"])
    role = authenticated_payload["role"]
    if role not in contracts.ROLES[1:]:
        raise ValueError("training role must be one of the fixed continuation arms")
    target_run_dir = Path("/artifacts/runs") / run_id
    if Path(run_dir) != target_run_dir:
        raise ValueError("run directory must exactly match this role's artifact path")
    result_evidence = result.get("evidence")
    if not isinstance(result_evidence, dict):
        raise ValueError("training result evidence must be an object")
    training_evidence = result_evidence.get("training")
    if not isinstance(training_evidence, dict):
        raise ValueError("training evidence must be an object")
    initial_training = training_evidence_api.initial_training_evidence()
    if contracts.canonical_json(training_evidence) != contracts.canonical_json(initial_training):
        raise ValueError("fresh training evidence is required; continuation does not resume")
    provenance_api.validate_provenance(
        result.get("provenance"),
        payload=authenticated_payload,
        passed=False,
        require_loaded=True,
    )
    selected_identity_value = provenance_api.validate_selected_adapter_identity(
        result_evidence.get("selected_adapter_identity"),
        payload=authenticated_payload,
        required=True,
    )
    if selected_identity_value is None:
        raise ValueError("selected adapter identity is required for training")
    selected_identity = cast(Mapping[str, object], selected_identity_value)
    specs_by_category = {
        "training": cast(list[dict[str, object]], authenticated_payload["training_specs"]),
        "final_evaluation": cast(
            list[dict[str, object]], authenticated_payload["evaluation_specs"]
        ),
        "reload_parity": cast(list[dict[str, object]], authenticated_payload["reload_specs"]),
    }
    progress = progress_api.validate_progress(
        ledger.snapshot(), role=role, specs_by_category=specs_by_category
    )
    counts = cast(Mapping[str, int], progress["completed_forward_counts"])
    if counts["total"] != 0 or progress["pending_forward"] is not None:
        raise ValueError("training requires a fresh zero-call ledger")
    examples = payload_api.training_examples(authenticated_payload)
    training_items = getattr(prepared, "training_items", None)
    prepared_examples = getattr(prepared, "training_examples", None)
    if len(examples) != contracts.TRAINING_UPDATES * contracts.MICROBATCHES_PER_UPDATE:
        raise ValueError("regenerated training schedule differs from the fixed 1344-example plan")
    if (
        not isinstance(training_items, tuple)
        or len(training_items) != len(examples)
        or not isinstance(prepared_examples, tuple)
        or prepared_examples != examples
    ):
        raise ValueError("prepared training items and examples do not match the fixed schedule")
    for index, (example, item, spec) in enumerate(
        zip(examples, training_items, specs_by_category["training"], strict=True)
    ):
        example = cast(Any, example)
        item = cast(Any, item)
        if (
            getattr(item, "category", None) != "training"
            or type(getattr(item, "index", None)) is not int
            or item.index != index
            or getattr(item, "request", None) != example.request
            or getattr(item, "expected_spec_json", None)
            != contracts.canonical_json(spec).decode("utf-8")
        ):
            raise ValueError(f"prepared training item {index} does not match its scheduled example")

    # Keep heavyweight runtime helpers behind the invocation boundary so this module
    # remains importable by CPU-only planning and validation tools.
    from experiments import mixture_training_runtime_helpers as runtime_helpers
    from experiments import mixture_training_runtime_scoring as adapter_scoring
    from experiments import modal_train_rehearsal as training_helpers
    from experiments import runtime_rule_study_runtime_scoring as runtime_scoring

    inventory = training_helpers._validate_adapter_inventory(model, torch, require_trainable=True)
    evidence = training_evidence
    evidence["adapter_inventory"] = inventory
    live_initial_state = adapter_scoring._state_dict(model, peft_module)
    selected_digest = cast(str, selected_identity["tensor_sha256"])
    live_initial_digest = runtime_helpers.tensor_state_sha256(
        live_initial_state, torch_module=torch
    )
    evidence["initial_tensor_sha256"] = live_initial_digest
    supplied_initial_digest = runtime_helpers.tensor_state_sha256(initial_state, torch_module=torch)
    if live_initial_digest != selected_digest or supplied_initial_digest != selected_digest:
        raise ValueError(
            "loaded and supplied initial adapter states must match the selected tensor pin"
        )
    if (
        training_helpers._adapter_state_changes(dict(initial_state), live_initial_state, torch)[
            "changed_tensor_count"
        ]
        != 0
    ):
        raise ValueError("supplied initial adapter state differs from the loaded selected adapter")
    adapter_scoring._verify_saved_adapter(
        model,
        cast(dict[str, object], selected_identity["snapshot"]),
        selected_digest,
        peft_module,
        torch,
    )
    evidence["initialization_verified"] = True

    random.seed(contracts.TRAINING_SEED)
    torch.manual_seed(contracts.TRAINING_SEED)
    torch.cuda.manual_seed_all(contracts.TRAINING_SEED)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("selected adapter has no trainable parameters")
    optimizer_settings = contracts.optimizer_settings()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=contracts.LEARNING_RATE,
        betas=contracts.ADAMW_BETAS,
        eps=contracts.ADAMW_EPSILON,
        weight_decay=contracts.WEIGHT_DECAY,
    )
    groups = optimizer.param_groups
    expected_parameter_ids = [id(parameter) for parameter in trainable]
    actual_parameter_ids = [
        id(parameter) for group in groups for parameter in group.get("params", ())
    ]
    settings_match = (
        len(groups) == 1
        and actual_parameter_ids == expected_parameter_ids
        and groups[0].get("lr") == contracts.LEARNING_RATE
        and tuple(groups[0].get("betas", ())) == contracts.ADAMW_BETAS
        and groups[0].get("eps") == contracts.ADAMW_EPSILON
        and groups[0].get("weight_decay") == contracts.WEIGHT_DECAY
    )
    state_entries = len(optimizer.state)
    if not settings_match or state_entries != 0:
        raise RuntimeError("fresh AdamW settings or initial state differ from the fixed contract")
    evidence["optimizer"] = optimizer_settings
    evidence["optimizer_initial_state_entries"] = state_entries
    evidence["base_gradients_none"] = None
    evidence["training_step_losses"] = []
    evidence["adapter_paths"] = {}
    result["phase"] = "training"
    model.train()

    training_losses = cast(list[dict[str, object]], evidence["training_step_losses"])
    adapter_paths = cast(dict[str, object], evidence["adapter_paths"])
    score_forward = runtime_scoring.score_forward
    total_b_gradient_l1 = 0.0
    total_gradient_norm = 0.0

    for update in range(1, contracts.TRAINING_UPDATES + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_start = (update - 1) * contracts.MICROBATCHES_PER_UPDATE
        batch = examples[batch_start : batch_start + contracts.MICROBATCHES_PER_UPDATE]
        if len(batch) != contracts.MICROBATCHES_PER_UPDATE:
            raise RuntimeError("optimizer update did not receive exactly four examples")

        loss_tensors: list[Any] = []
        for example, item in zip(batch, training_items[batch_start : batch_start + 4], strict=True):
            example = cast(Any, example)
            candidate_logits = score_forward(
                model, item, torch, ledger, cast(dict[str, object], result_evidence), "training"
            )
            target = torch.tensor([example.gold_index], dtype=torch.long, device="cuda")
            loss = torch.nn.functional.cross_entropy(candidate_logits.unsqueeze(0), target)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("training loss was non-finite")
            loss_tensors.append(loss.detach())
            (loss / contracts.MICROBATCHES_PER_UPDATE).backward()

        named_parameters = list(model.named_parameters())
        base_parameters = [
            parameter
            for name, parameter in named_parameters
            if ".lora_A." not in name and ".lora_B." not in name
        ]
        if any(parameter.grad is not None for parameter in base_parameters):
            evidence["base_gradients_none"] = False
            raise RuntimeError("a frozen base parameter received a gradient")
        if any(parameter.requires_grad for parameter in base_parameters):
            raise RuntimeError("a base model parameter became trainable")
        evidence["base_gradients_none"] = True

        b_gradients = [
            parameter.grad
            for name, parameter in named_parameters
            if ".lora_B." in name and parameter.grad is not None
        ]
        gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
        if not b_gradients:
            raise RuntimeError("LoRA-B gradients were missing")
        if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
            raise RuntimeError("adapter gradients contained a non-finite value")
        b_gradient_l1 = sum(float(gradient.detach().abs().sum().item()) for gradient in b_gradients)
        if not math.isfinite(b_gradient_l1):
            raise RuntimeError("LoRA-B gradient evidence was non-finite")

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, contracts.MAX_GRADIENT_NORM)
        if not bool(torch.isfinite(gradient_norm).item()):
            raise RuntimeError("adapter gradient norm was non-finite")
        gradient_norm_value = float(gradient_norm.item())
        if not math.isfinite(gradient_norm_value) or gradient_norm_value < 0.0:
            raise RuntimeError("adapter gradient norm evidence was invalid")

        optimizer.step()
        # A returned optimizer step is the update boundary. Preserve it before any
        # fallible scalar conversion, row append, snapshot, or progress callback.
        evidence["optimizer_updates_completed"] = update
        mean_loss = sum(float(loss.item()) for loss in loss_tensors) / len(loss_tensors)
        if not math.isfinite(mean_loss) or mean_loss < 0.0:
            raise RuntimeError("training mean loss evidence was invalid")
        training_losses.append(
            {
                "update": update,
                "mean_loss": mean_loss,
                "lo_ra_b_gradient_l1": b_gradient_l1,
                "gradient_norm": gradient_norm_value,
            }
        )
        total_b_gradient_l1 += b_gradient_l1
        total_gradient_norm += gradient_norm_value

        if update in (contracts.UNSCORED_SAVE_UPDATE, contracts.FINAL_SAVE_UPDATE):
            snapshot = training_helpers._save_adapter_snapshot(model, target_run_dir, update)
            adapter_paths[str(update)] = snapshot
            adapter_scoring._snapshot_hashes(snapshot)
            if on_progress is not None:
                on_progress(result, target_run_dir)
        elif update % 16 == 0 and on_progress is not None:
            on_progress(result, target_run_dir)

    final_snapshot = cast(dict[str, object], adapter_paths[str(contracts.FINAL_SAVE_UPDATE)])
    final_state = adapter_scoring._state_dict(model, peft_module)
    adapter_update = training_helpers._adapter_state_changes(
        dict(initial_state), final_state, torch
    )
    final_digest = runtime_helpers.tensor_state_sha256(final_state, torch_module=torch)
    evidence["adapter_update"] = adapter_update
    evidence["final_tensor_sha256"] = final_digest
    adapter_scoring._verify_saved_adapter(model, final_snapshot, final_digest, peft_module, torch)
    if cast(int, adapter_update["changed_tensor_count"]) == 0:
        raise RuntimeError("training did not change any LoRA adapter tensor")
    if not math.isfinite(total_b_gradient_l1) or total_b_gradient_l1 <= 0.0:
        raise RuntimeError("training produced no positive LoRA-B gradient evidence")
    if not math.isfinite(total_gradient_norm) or total_gradient_norm <= 0.0:
        raise RuntimeError("training produced no positive gradient norm evidence")
    return final_state
