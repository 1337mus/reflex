"""CPU evidence validators shared by mixture result consumers."""

from __future__ import annotations

import math
from collections.abc import Mapping

from experiments.mixture_training_contracts import (
    LEARNING_RATE,
    MAX_GRADIENT_NORM,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_VERSION_PINS,
    normalize_json_object,
    validate_runtime_versions,
    validate_sha256,
)
from experiments.mixture_training_outputs import (
    _expected_output_identity,
    _validate_artifact,
    _validate_counts,
    _validate_output_rows,
)

_PROVENANCE_FIELDS = {
    "model_id",
    "model_revision",
    "versions",
    "source_file_sha256",
    "measured_source_file_sha256",
    "base_model",
    "optimizer",
    "cuda_device",
}
_BASE_PROVENANCE_FIELDS = _PROVENANCE_FIELDS - {"optimizer"}
_COMMON_EVIDENCE_FIELDS = {
    "forward_counts",
    "input_token_counts",
    "outputs",
    "output_artifact",
    "initialization",
}
_TRAIN_EVIDENCE_FIELDS = {
    "optimizer_updates_completed",
    "training_step_losses",
    "adapter_inventory",
    "adapter_update",
    "base_gradients_none",
    "adapter_paths",
    "initialization_verified",
    "initial_tensor_sha256",
    "reload_parity",
}
_BASE_VERSION_KEYS = set(RUNTIME_VERSION_PINS) - {"Pillow"}
_BASE_VERSION_PINS = {
    name: version for name, version in RUNTIME_VERSION_PINS.items() if name != "Pillow"
}
_INVENTORY_FIELDS = {
    "module_count",
    "module_names",
    "adapter_tensor_count",
    "adapter_parameter_count",
    "adapter_tensor_names",
    "adapter_tensor_shapes",
    "adapter_tensor_dtypes",
    "trainable_parameter_count",
    "base_parameters_frozen_bf16",
}
_EXPECTED_TRAINABLE_PARAMETERS = 2_015_232


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer greater than or equal to {minimum}")
    return value


def _finite_number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{label} is outside its finite range")
    return number


def _validate_provenance(
    value: object, payload: Mapping[str, object], *, passed: bool, train: bool
) -> dict[str, object]:
    provenance = normalize_json_object(value, "provenance")
    required_fields = _PROVENANCE_FIELDS if train else _BASE_PROVENANCE_FIELDS
    if passed and not required_fields.issubset(provenance):
        raise ValueError("passed result is missing required provenance")
    if train is False and "optimizer" in provenance:
        raise ValueError("initialization provenance must not claim optimizer training")
    if train and passed and "optimizer" not in provenance:
        raise ValueError("passed training provenance is missing optimizer settings")
    if "model_id" in provenance and provenance["model_id"] != MODEL_ID:
        raise ValueError("result model_id differs from the pinned base model")
    if "model_revision" in provenance and provenance["model_revision"] != MODEL_REVISION:
        raise ValueError("result model revision differs from the pinned base revision")
    versions = provenance.get("versions")
    if versions is not None:
        if passed:
            validate_runtime_versions(versions)
        elif not isinstance(versions, Mapping) or any(
            key not in RUNTIME_VERSION_PINS or RUNTIME_VERSION_PINS[key] != version
            for key, version in versions.items()
        ):
            raise ValueError("partial runtime version evidence differs from frozen pins")
    expected_sources = payload["pins"]["source_file_sha256"]
    for field in ("source_file_sha256", "measured_source_file_sha256"):
        fingerprints = provenance.get(field)
        if fingerprints is None:
            if passed:
                raise ValueError(f"passed result is missing {field}")
            continue
        if not isinstance(fingerprints, Mapping):
            raise ValueError(f"{field} must be an object")
        for path, digest in fingerprints.items():
            if path not in expected_sources:
                raise ValueError(f"{field} contains a source outside the allowlist")
            validate_sha256(digest, f"{field} {path}")
        if passed and dict(fingerprints) != expected_sources:
            raise ValueError(f"passed {field} differs from the payload source pins")
    base = provenance.get("base_model")
    if base is not None:
        base = normalize_json_object(base, "base_model provenance")
        if base.get("model_id") != MODEL_ID or base.get("model_revision") != MODEL_REVISION:
            raise ValueError("loaded base model evidence differs from the pinned model")
        base_versions = base.get("versions")
        if base_versions is not None and dict(base_versions) != _BASE_VERSION_PINS:
            raise ValueError("base model package versions differ from the eight-package pins")
        if passed and (
            not isinstance(base_versions, Mapping) or set(base_versions) != _BASE_VERSION_KEYS
        ):
            raise ValueError("passed base model evidence is missing its eight package versions")
        provenance["base_model"] = base
    elif passed:
        raise ValueError("passed result is missing actual base model loading evidence")
    cuda_device = provenance.get("cuda_device")
    if cuda_device is not None and (not isinstance(cuda_device, str) or not cuda_device.strip()):
        raise ValueError("cuda_device must be a nonempty string")
    if passed and (
        not isinstance(provenance.get("cuda_device"), str) or not provenance["cuda_device"].strip()
    ):
        raise ValueError("passed result is missing CUDA device evidence")
    optimizer = provenance.get("optimizer")
    if optimizer is not None:
        if not isinstance(optimizer, Mapping):
            raise ValueError("optimizer provenance must be an object")
        if optimizer.get("name") != "AdamW":
            raise ValueError("optimizer provenance must identify AdamW")
        if (
            optimizer.get("learning_rate") != LEARNING_RATE
            or optimizer.get("weight_decay") != 0.0
            or optimizer.get("max_gradient_norm") != MAX_GRADIENT_NORM
        ):
            raise ValueError("optimizer parameters differ from the frozen training contract")
    return provenance


def _validate_snapshot(value: object, run_id: str, update: int) -> dict[str, object]:
    snapshot = normalize_json_object(value, "adapter snapshot")
    if set(snapshot) != {"update", "path", "files_sha256"}:
        raise ValueError("adapter snapshot has an unexpected schema")
    if _integer(snapshot["update"], "adapter snapshot update") != update:
        raise ValueError("adapter snapshot update does not match its evidence slot")
    expected_path = f"/artifacts/runs/{run_id}/adapter-update-{update:03d}"
    if snapshot["path"] != expected_path:
        raise ValueError("adapter snapshot path is not canonical for this run")
    files = snapshot["files_sha256"]
    required = {"adapter_model.safetensors", "adapter_config.json"}
    if not isinstance(files, Mapping) or not required.issubset(files):
        raise ValueError("adapter snapshot is missing required saved files")
    if set(files) - (required | {"README.md"}):
        raise ValueError("adapter snapshot contains an unexpected file")
    for name, digest in files.items():
        validate_sha256(digest, f"snapshot {name} SHA-256")
    return snapshot


def _validate_inventory(value: object) -> dict[str, object]:
    inventory = normalize_json_object(value, "adapter_inventory")
    if set(inventory) != _INVENTORY_FIELDS:
        raise ValueError("adapter inventory has an unexpected schema")
    if _integer(inventory["module_count"], "adapter module count") != 60:
        raise ValueError("adapter inventory must contain 60 target modules")
    modules = inventory["module_names"]
    if (
        not isinstance(modules, list)
        or len(modules) != 60
        or any(not isinstance(name, str) or not name for name in modules)
        or len(set(modules)) != 60
    ):
        raise ValueError("adapter module names are incomplete or duplicated")
    if _integer(inventory["adapter_tensor_count"], "adapter tensor count") != 120:
        raise ValueError("adapter inventory must contain 120 tensors")
    names = inventory["adapter_tensor_names"]
    if (
        not isinstance(names, list)
        or len(names) != 120
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != 120
    ):
        raise ValueError("adapter tensor names are incomplete or duplicated")
    shapes = inventory["adapter_tensor_shapes"]
    dtypes = inventory["adapter_tensor_dtypes"]
    if not isinstance(shapes, Mapping) or set(shapes) != set(names):
        raise ValueError("adapter tensor shapes do not cover the exact tensor inventory")
    if not isinstance(dtypes, Mapping) or set(dtypes) != set(names):
        raise ValueError("adapter tensor dtypes do not cover the exact tensor inventory")
    parameter_count = 0
    for name in names:
        shape = shapes[name]
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(dim) is not int or dim < 1 for dim in shape)
        ):
            raise ValueError("adapter tensor shape is malformed")
        parameter_count += math.prod(shape)
        if dtypes[name] != "torch.float32":
            raise ValueError("adapter tensors must remain FP32")
    if (
        _integer(inventory["adapter_parameter_count"], "adapter parameter count")
        != _EXPECTED_TRAINABLE_PARAMETERS
        or parameter_count != _EXPECTED_TRAINABLE_PARAMETERS
        or _integer(inventory["trainable_parameter_count"], "trainable parameter count")
        != _EXPECTED_TRAINABLE_PARAMETERS
        or inventory["base_parameters_frozen_bf16"] is not True
    ):
        raise ValueError("adapter inventory parameter or frozen-base evidence is invalid")
    return inventory


def _validate_adapter_update(value: object, inventory: Mapping[str, object]) -> dict[str, object]:
    update = normalize_json_object(value, "adapter_update")
    if set(update) != {"changed_tensor_count", "changed_tensor_names"}:
        raise ValueError("adapter update evidence has an unexpected schema")
    count = _integer(update["changed_tensor_count"], "changed tensor count", minimum=1)
    names = update["changed_tensor_names"]
    inventory_names = inventory["adapter_tensor_names"]
    if not isinstance(inventory_names, list):
        raise ValueError("adapter inventory tensor names are malformed")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or count != len(names)
    ):
        raise ValueError("adapter update changed tensors do not match the inventory")
    inventory_name_set = set(inventory_names)
    canonical_names = []
    for name in names:
        if name in inventory_name_set:
            canonical_names.append(name)
            continue
        canonical_name = None
        for suffix in (".lora_A.weight", ".lora_B.weight"):
            if name.endswith(suffix):
                canonical_name = name[: -len(suffix)] + suffix.replace(".weight", ".default.weight")
                break
        if canonical_name not in inventory_name_set:
            raise ValueError("adapter update changed tensor is absent from the inventory")
        canonical_names.append(canonical_name)
    if len(set(canonical_names)) != len(canonical_names):
        raise ValueError("adapter update names collide after default-adapter normalization")
    return update


def _validate_training_losses(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != 252:
        raise ValueError("training evidence must include 252 per-update loss records")
    losses: list[dict[str, object]] = []
    b_gradient_total = 0.0
    gradient_norm_total = 0.0
    for update, item in enumerate(value, start=1):
        row = normalize_json_object(item, "training step loss")
        if set(row) != {"update", "mean_loss", "lo_ra_b_gradient_l1", "gradient_norm"}:
            raise ValueError("training step loss has an unexpected schema")
        if _integer(row["update"], "training loss update", minimum=1) != update:
            raise ValueError("training step losses must cover updates 1 through 252 in order")
        _finite_number(row["mean_loss"], "training mean loss", minimum=0.0)
        b_gradient_total += _finite_number(
            row["lo_ra_b_gradient_l1"], "LoRA-B gradient L1", minimum=0.0
        )
        gradient_norm_total += _finite_number(row["gradient_norm"], "gradient norm", minimum=0.0)
        losses.append(row)
    if b_gradient_total <= 0.0 or gradient_norm_total <= 0.0:
        raise ValueError("training run must contain nonzero aggregate gradient evidence")
    return losses


def _validate_training_evidence(
    evidence: dict[str, object], payload: Mapping[str, object], outputs: list[dict[str, object]]
) -> None:
    run_id = str(payload["run_id"])
    if _integer(evidence.get("optimizer_updates_completed"), "optimizer_updates_completed") != 252:
        raise ValueError("passed training result must complete exactly 252 optimizer updates")
    losses = _validate_training_losses(evidence.get("training_step_losses"))
    evidence["training_step_losses"] = losses
    inventory = _validate_inventory(evidence.get("adapter_inventory"))
    evidence["adapter_inventory"] = inventory
    paths_value = normalize_json_object(evidence.get("adapter_paths"), "adapter_paths")
    if set(paths_value) != {"126", "252"}:
        raise ValueError("training snapshots must be saved at updates 126 and 252")
    snapshots = {
        key: _validate_snapshot(paths_value[key], run_id, int(key)) for key in ("126", "252")
    }
    update = _validate_adapter_update(evidence.get("adapter_update"), inventory)
    evidence["adapter_paths"] = snapshots
    evidence["adapter_update"] = update
    if evidence.get("base_gradients_none") is not True:
        raise ValueError("training must prove frozen base-model gradients remained absent")
    if evidence.get("initialization_verified") is not True:
        raise ValueError("training must prove that the update-0 initialization was verified")
    initialization = normalize_json_object(
        evidence.get("initialization"), "initialization evidence"
    )
    expected_initialization = payload["initialization"]
    if initialization != expected_initialization:
        raise ValueError("training initialization evidence differs from its payload descriptor")
    initial_tensor_sha = validate_sha256(
        evidence.get("initial_tensor_sha256"), "initial_tensor_sha256"
    )
    if initial_tensor_sha != expected_initialization["tensor_sha256"]:
        raise ValueError("training loaded tensor digest differs from initialization")
    parity = normalize_json_object(evidence.get("reload_parity"), "reload_parity")
    if set(parity) != {"tensor_values_exact", "winner_match", "max_abs_difference", "outputs"}:
        raise ValueError("reload parity evidence has an unexpected schema")
    if parity["tensor_values_exact"] is not True or parity["winner_match"] is not True:
        raise ValueError("reload parity did not confirm exact tensors and matching winners")
    parity_outputs = parity["outputs"]
    expected_by_id = {row["presentation_id"]: row for row in outputs}
    expected_ids = sorted(expected_by_id)[:32]
    if not isinstance(parity_outputs, list) or len(parity_outputs) != 32:
        raise ValueError("reload parity must retain the actual 32 scored rows")
    parity_presentations = [
        _expected_output_identity(expected_by_id[presentation_id])
        | {
            "request": next(
                row["request"]
                for row in payload["evaluation_presentations"]
                if row["presentation_id"] == presentation_id
            )
        }
        for presentation_id in expected_ids
    ]
    normalized_parity = _validate_output_rows(
        parity_outputs,
        parity_presentations,
        label="reload parity outputs",
        require_complete=True,
    )
    output_by_id = {row["presentation_id"]: row for row in outputs}
    max_difference = 0.0
    for parity_row in normalized_parity:
        original = output_by_id[parity_row["presentation_id"]]
        if parity_row["winner_option_id"] != original["winner_option_id"]:
            raise ValueError("reload parity winner differs from final evaluation")
        max_difference = max(
            max_difference,
            *(
                abs(float(left) - float(right))
                for left, right in zip(
                    parity_row["candidate_logits"], original["candidate_logits"], strict=True
                )
            ),
        )
    reported_difference = _finite_number(
        parity["max_abs_difference"], "reload max absolute difference", minimum=0.0
    )
    if reported_difference > 1e-3 or abs(reported_difference - max_difference) > 1e-7:
        raise ValueError("reload parity logit difference is inaccurate or exceeds 1e-3")
    parity["outputs"] = normalized_parity
    evidence["reload_parity"] = parity


__all__ = [
    "_COMMON_EVIDENCE_FIELDS",
    "_TRAIN_EVIDENCE_FIELDS",
    "_validate_artifact",
    "_validate_adapter_update",
    "_validate_counts",
    "_validate_output_rows",
    "_validate_provenance",
    "_validate_training_losses",
    "_validate_training_evidence",
]
