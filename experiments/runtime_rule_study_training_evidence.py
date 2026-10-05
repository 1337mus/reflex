"""Validate CPU evidence for continuation training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from experiments import mixture_training_evidence as shared
from experiments import runtime_rule_study_contracts as contracts
from experiments.mixture_training_contracts import canonical_json, normalize_json_object

_TRAINING_UPDATES = contracts.TRAINING_UPDATES
_MICROBATCHES_PER_UPDATE = contracts.MICROBATCHES_PER_UPDATE
_MAX_TRAINING_FORWARDS = _TRAINING_UPDATES * _MICROBATCHES_PER_UPDATE
_SNAPSHOT_UPDATES = (contracts.UNSCORED_SAVE_UPDATE, contracts.FINAL_SAVE_UPDATE)
_EVIDENCE_FIELDS = frozenset(
    {
        "optimizer",
        "optimizer_initial_state_entries",
        "optimizer_updates_completed",
        "training_step_losses",
        "adapter_inventory",
        "base_gradients_none",
        "initialization_verified",
        "initial_tensor_sha256",
        "adapter_paths",
        "adapter_update",
        "final_tensor_sha256",
    }
)


def _json_object(value: object, label: str) -> dict[str, object]:
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            for key, item in current.items():
                if not isinstance(key, str):
                    raise ValueError(f"{label} object keys must be strings")
                pending.append(item)
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            pending.extend(current)
    return cast(dict[str, object], normalize_json_object(value, label))


def _validate_losses(
    value: object,
    updates: int,
    completed_forwards: int,
    *,
    passed: bool,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("training_step_losses must be an array")
    if len(value) > updates:
        raise ValueError("training_step_losses exceed completed optimizer updates")
    if passed:
        if updates != _TRAINING_UPDATES or len(value) != _TRAINING_UPDATES:
            raise ValueError(
                f"passed training evidence must contain all {_TRAINING_UPDATES} loss records"
            )
    elif len(value) != updates:
        terminal_gap = updates > 0 and len(value) == updates - 1
        if not terminal_gap:
            raise ValueError("failed training evidence may omit only its terminal loss record")
        if completed_forwards != _MICROBATCHES_PER_UPDATE * updates:
            raise ValueError("terminal loss gap requires exactly four forwards per update")

    losses: list[dict[str, object]] = []
    b_gradient_total = 0.0
    gradient_norm_total = 0.0
    for index, item in enumerate(value, start=1):
        row = _json_object(item, "training step loss")
        if set(row) != {"update", "mean_loss", "lo_ra_b_gradient_l1", "gradient_norm"}:
            raise ValueError("training step loss has an unexpected schema")
        if shared._integer(row["update"], "training loss update", minimum=1) != index:
            raise ValueError("training step losses must cover updates in order without gaps")
        shared._finite_number(row["mean_loss"], "training mean loss", minimum=0.0)
        b_gradient_total += shared._finite_number(
            row["lo_ra_b_gradient_l1"], "LoRA-B gradient L1", minimum=0.0
        )
        gradient_norm_total += shared._finite_number(
            row["gradient_norm"], "gradient norm", minimum=0.0
        )
        losses.append(row)

    if passed:
        if shared._finite_number(b_gradient_total, "summed LoRA-B gradient L1") <= 0.0:
            raise ValueError(
                "passed training evidence requires positive aggregate B-gradient evidence"
            )
        if shared._finite_number(gradient_norm_total, "summed gradient norm") <= 0.0:
            raise ValueError("passed training evidence requires positive aggregate gradient norms")
    return losses


def initial_training_evidence() -> dict[str, object]:
    return {
        "optimizer": None,
        "optimizer_initial_state_entries": None,
        "optimizer_updates_completed": 0,
        "training_step_losses": [],
        "adapter_inventory": None,
        "base_gradients_none": None,
        "initialization_verified": False,
        "initial_tensor_sha256": None,
        "adapter_paths": {},
        "adapter_update": None,
        "final_tensor_sha256": None,
    }


def validate_training_evidence(
    value: object,
    *,
    run_id: object,
    completed_training_forwards: object,
    passed: object,
) -> dict[str, object]:
    checked_run_id = contracts.validate_safe_run_id(run_id)
    if type(passed) is not bool:
        raise ValueError("passed must be a boolean")
    completed_forwards = shared._integer(completed_training_forwards, "completed training forwards")
    if completed_forwards > _MAX_TRAINING_FORWARDS:
        raise ValueError("completed training forwards exceed the fixed study count")

    evidence = _json_object(value, "training evidence")
    if set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("training evidence has an unexpected schema")

    optimizer = evidence["optimizer"]
    initial_state_entries = evidence["optimizer_initial_state_entries"]
    if (optimizer is None) != (initial_state_entries is None):
        raise ValueError("optimizer settings and initial state-entry evidence must appear together")
    if optimizer is not None:
        if not isinstance(optimizer, Mapping) or canonical_json(optimizer) != canonical_json(
            contracts.optimizer_settings()
        ):
            raise ValueError("optimizer settings differ from the fixed fresh AdamW contract")
        if shared._integer(initial_state_entries, "optimizer initial state entries") != 0:
            raise ValueError("optimizer must start with zero state entries")

    updates = shared._integer(evidence["optimizer_updates_completed"], "optimizer updates")
    if updates > _TRAINING_UPDATES:
        raise ValueError(f"optimizer updates exceed the fixed {_TRAINING_UPDATES}-update study")
    if passed and (updates != _TRAINING_UPDATES or completed_forwards != _MAX_TRAINING_FORWARDS):
        raise ValueError(
            f"passed training evidence must complete {_TRAINING_UPDATES} updates and "
            f"{_MAX_TRAINING_FORWARDS} forwards"
        )
    minimum_forwards = _MICROBATCHES_PER_UPDATE * updates
    maximum_forwards = minimum_forwards + _MICROBATCHES_PER_UPDATE
    if not passed and not minimum_forwards <= completed_forwards <= maximum_forwards:
        raise ValueError("completed training forwards are inconsistent with optimizer updates")

    inventory_value = evidence["adapter_inventory"]
    inventory: dict[str, object] | None = None
    if inventory_value is not None:
        inventory = shared._validate_inventory(inventory_value)

    initialization_verified = evidence["initialization_verified"]
    if type(initialization_verified) is not bool:
        raise ValueError("initialization_verified must be a boolean")
    initial_tensor = evidence["initial_tensor_sha256"]
    if initial_tensor is not None:
        initial_tensor = contracts.validate_sha256(initial_tensor, "initial tensor SHA-256")
    if initialization_verified and (
        initial_tensor != contracts.SELECTED_TENSOR_SHA256 or inventory is None
    ):
        raise ValueError("verified initialization requires the selected tensor and valid inventory")

    base_gradients_none = evidence["base_gradients_none"]
    if base_gradients_none is not None and type(base_gradients_none) is not bool:
        raise ValueError("base_gradients_none must be a boolean or null")
    if passed and base_gradients_none is not True:
        raise ValueError("passed training evidence must confirm base gradients remained absent")

    if completed_forwards > 0 and (
        not initialization_verified
        or inventory is None
        or optimizer is None
        or initial_state_entries is None
    ):
        raise ValueError(
            "training forwards require verified initialization, inventory, and fresh optimizer"
        )

    losses = _validate_losses(
        evidence["training_step_losses"],
        updates,
        completed_forwards,
        passed=passed,
    )
    evidence["training_step_losses"] = losses

    raw_paths = _json_object(evidence["adapter_paths"], "adapter_paths")
    snapshot_keys = {str(update) for update in _SNAPSHOT_UPDATES}
    if set(raw_paths) - snapshot_keys:
        raise ValueError("adapter_paths contain an update outside the fixed snapshot schedule")
    midpoint_key, final_key = (str(update) for update in _SNAPSHOT_UPDATES)
    if final_key in raw_paths and midpoint_key not in raw_paths:
        raise ValueError("final adapter snapshot requires the midpoint snapshot")
    if passed and set(raw_paths) != snapshot_keys:
        raise ValueError("passed training evidence requires both adapter snapshots")
    snapshots: dict[str, object] = {}
    for key in (midpoint_key, final_key):
        if key not in raw_paths:
            continue
        update = int(key)
        if update > updates:
            raise ValueError("adapter snapshot update exceeds completed optimizer updates")
        snapshots[key] = shared._validate_snapshot(raw_paths[key], checked_run_id, update)
    evidence["adapter_paths"] = snapshots

    raw_update = evidence["adapter_update"]
    adapter_update: dict[str, object] | None = None
    if raw_update is not None:
        if inventory is None:
            raise ValueError("adapter update evidence requires a valid adapter inventory")
        normalized_update = _json_object(raw_update, "adapter_update")
        no_change_failure = (
            not passed
            and set(normalized_update) == {"changed_tensor_count", "changed_tensor_names"}
            and type(normalized_update["changed_tensor_count"]) is int
            and normalized_update["changed_tensor_count"] == 0
            and normalized_update["changed_tensor_names"] == []
        )
        if no_change_failure:
            adapter_update = normalized_update
        else:
            adapter_update = shared._validate_adapter_update(normalized_update, inventory)
    if passed and adapter_update is None:
        raise ValueError("passed training evidence requires adapter update evidence")
    evidence["adapter_update"] = adapter_update

    final_tensor = evidence["final_tensor_sha256"]
    if final_tensor is not None:
        final_tensor = contracts.validate_sha256(final_tensor, "final tensor SHA-256")
    if passed and (final_tensor is None or final_tensor == initial_tensor):
        raise ValueError("passed training evidence requires a changed final tensor digest")

    if passed and (
        not initialization_verified
        or inventory is None
        or optimizer is None
        or initial_state_entries is None
        or adapter_update is None
        or not snapshots
    ):
        raise ValueError("passed training evidence is missing required training state")

    return evidence
