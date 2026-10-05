"""Validate fixed CPU evidence for runtime-study provenance and selection identity."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from experiments import runtime_rule_study_contracts as contracts
from experiments.mixture_training_contracts import (
    normalize_json_object,
    validate_sha256,
)

_PROVENANCE_FIELDS = frozenset(
    {
        "model_id",
        "model_revision",
        "source_file_sha256",
        "measured_source_file_sha256",
        "runtime_versions",
        "base_model",
        "cuda_device",
        "reload_tokenizer_file_sha256",
    }
)
_BASE_MODEL_FIELDS = frozenset(
    {
        "model_id",
        "model_revision",
        "model_class",
        "config_class",
        "layer_count",
        "tied_embeddings",
        "no_meta_parameters",
        "load_diagnostics",
        "tokenizer_file_sha256",
        "effective_dtype",
        "device",
        "attention_implementation",
        "use_kernels",
        "use_hub_kernels",
        "versions",
    }
)
_DIAGNOSTIC_FIELDS = frozenset({"error_msgs", "mismatched_keys", "missing_keys", "unexpected_keys"})
_TOKENIZER_FILES = frozenset(
    {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
)
_BASE_VERSION_PINS = {
    name: version for name, version in contracts.RUNTIME_VERSION_PINS.items() if name != "Pillow"
}
_CUDA_INDEX_ONLY = re.compile(r"cuda(?::[0-9]+)?\Z", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _PayloadIdentity:
    model_id: str
    model_revision: str
    source_hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class _ProvenanceContext(_PayloadIdentity):
    role: str
    runtime_versions: dict[str, str]
    tokenizer_hashes: dict[str, str]


@dataclass(frozen=True, slots=True)
class _SelectedContext:
    selected_tensor_sha256: str
    selected_snapshot: dict[str, object]


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
            if any(not isinstance(key, str) for key in current):
                raise ValueError(f"{label} object keys must be strings")
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            pending.extend(current)
    return cast(dict[str, object], normalize_json_object(value, label))


def _required(mapping: Mapping[str, object], key: str, label: str) -> object:
    if key not in mapping:
        raise ValueError(f"{label} is missing required field {key}")
    return mapping[key]


def _nonblank_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _digest_map(
    value: object,
    label: str,
    *,
    allowed_keys: set[str] | frozenset[str] | None = None,
    exact_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, str]:
    raw = _json_object(value, label)
    keys = set(raw)
    if allowed_keys is not None and keys - set(allowed_keys):
        raise ValueError(f"{label} contains a key outside its allowlist")
    if exact_keys is not None and keys != set(exact_keys):
        raise ValueError(f"{label} must contain exactly its pinned keys")
    result: dict[str, str] = {}
    for key, digest in raw.items():
        if not key.strip():
            raise ValueError(f"{label} keys must be nonblank strings")
        result[key] = validate_sha256(digest, f"{label} {key} SHA-256")
    return {key: result[key] for key in sorted(result)}


def _version_map(
    value: object, label: str, *, allowed_keys: set[str] | frozenset[str]
) -> dict[str, str]:
    raw = _json_object(value, label)
    keys = set(raw)
    if keys - set(allowed_keys):
        raise ValueError(f"{label} contains a key outside its allowlist")
    result: dict[str, str] = {}
    for key, version in raw.items():
        if not key.strip() or not isinstance(version, str) or not version.strip():
            raise ValueError(f"{label} values must be nonblank strings")
        result[key] = version
    return {key: result[key] for key in sorted(result)}


def _selected_snapshot(value: object, label: str) -> dict[str, object]:
    snapshot = _json_object(value, label)
    if set(snapshot) != {"update", "path", "files_sha256"}:
        raise ValueError(f"{label} has an unexpected schema")
    if type(snapshot["update"]) is not int or snapshot["update"] < 1:
        raise ValueError(f"{label} update must be a positive integer")
    _nonblank_string(snapshot["path"], f"{label} path")
    files = _digest_map(
        snapshot["files_sha256"],
        f"{label} files_sha256",
        allowed_keys={"adapter_config.json", "adapter_model.safetensors", "README.md"},
    )
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(files):
        raise ValueError(f"{label} is missing a required adapter file hash")
    return {"update": snapshot["update"], "path": snapshot["path"], "files_sha256": files}


def _payload_study(payload: object) -> tuple[Mapping[str, object], dict[str, object]]:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be an object")
    payload_view = cast(Mapping[str, object], payload)
    study = _json_object(_required(payload_view, "study", "payload"), "payload study")
    return payload_view, study


def _payload_identity(payload: object) -> _PayloadIdentity:
    payload_view, study = _payload_study(payload)
    source_hashes = _digest_map(
        _required(payload_view, "source_file_sha256", "payload"), "payload source_file_sha256"
    )
    model_id = _nonblank_string(_required(study, "model_id", "payload study"), "payload model_id")
    model_revision = _nonblank_string(
        _required(study, "model_revision", "payload study"), "payload model_revision"
    )
    if model_id != contracts.MODEL_ID:
        raise ValueError("payload model_id differs from the fixed model contract")
    if model_revision != contracts.MODEL_REVISION:
        raise ValueError("payload model_revision differs from the fixed revision contract")
    return _PayloadIdentity(model_id, model_revision, source_hashes)


def _provenance_context(payload: object) -> _ProvenanceContext:
    payload_view, study = _payload_study(payload)
    identity = _payload_identity(payload)
    role = _nonblank_string(_required(payload_view, "role", "payload"), "payload role")
    if role not in contracts.ROLES:
        raise ValueError("payload role is unsupported")
    runtime_versions = _version_map(
        _required(study, "runtime_versions", "payload study"),
        "payload runtime_versions",
        allowed_keys=set(contracts.RUNTIME_VERSION_PINS),
    )
    if runtime_versions != contracts.RUNTIME_VERSION_PINS:
        raise ValueError("payload runtime_versions differ from fixed pins")
    tokenizer_hashes = _digest_map(
        _required(study, "tokenizer_file_sha256", "payload study"),
        "payload tokenizer_file_sha256",
        allowed_keys=_TOKENIZER_FILES,
        exact_keys=_TOKENIZER_FILES,
    )
    return _ProvenanceContext(
        model_id=identity.model_id,
        model_revision=identity.model_revision,
        source_hashes=identity.source_hashes,
        role=role,
        runtime_versions=runtime_versions,
        tokenizer_hashes=tokenizer_hashes,
    )


def _selected_context(payload: object) -> _SelectedContext:
    _, study = _payload_study(payload)
    selected_digest = validate_sha256(
        _required(study, "selected_tensor_sha256", "payload study"),
        "payload selected tensor SHA-256",
    )
    if selected_digest != contracts.SELECTED_TENSOR_SHA256:
        raise ValueError("payload selected tensor digest differs from the fixed pin")
    selection = _json_object(_required(study, "selection", "payload study"), "payload selection")
    selected_snapshot = _selected_snapshot(
        _required(selection, "snapshot", "payload study selection"),
        "payload selected snapshot",
    )
    return _SelectedContext(selected_digest, selected_snapshot)


def initial_provenance(payload: object) -> dict[str, object]:
    """Build initial provenance with payload identity and no invented measurements."""

    context = _payload_identity(payload)
    return {
        "model_id": context.model_id,
        "model_revision": context.model_revision,
        "source_file_sha256": dict(context.source_hashes),
        "measured_source_file_sha256": None,
        "runtime_versions": None,
        "base_model": None,
        "cuda_device": None,
        "reload_tokenizer_file_sha256": None,
    }


def _validate_base_model(value: object) -> dict[str, object]:
    base = _json_object(value, "base_model")
    if set(base) != _BASE_MODEL_FIELDS:
        raise ValueError("base_model has an unexpected schema")
    for field in (
        "model_id",
        "model_revision",
        "model_class",
        "config_class",
        "effective_dtype",
        "device",
        "attention_implementation",
    ):
        _nonblank_string(base[field], f"base_model {field}")
    if type(base["layer_count"]) is not int or base["layer_count"] < 1:
        raise ValueError("base_model layer_count must be an integer greater than zero")
    for field in ("tied_embeddings", "no_meta_parameters", "use_kernels"):
        if type(base[field]) is not bool:
            raise ValueError(f"base_model {field} must be a boolean")
    hub_kernels = base["use_hub_kernels"]
    if hub_kernels is not None and (not isinstance(hub_kernels, str) or not hub_kernels.strip()):
        raise ValueError("base_model use_hub_kernels must be a nonblank string or null")
    diagnostics = _json_object(base["load_diagnostics"], "base_model load_diagnostics")
    if set(diagnostics) != _DIAGNOSTIC_FIELDS:
        raise ValueError("base_model load_diagnostics has an unexpected schema")
    if any(not isinstance(diagnostics[name], list) for name in _DIAGNOSTIC_FIELDS):
        raise ValueError("base_model load_diagnostics values must be arrays")
    base["load_diagnostics"] = diagnostics
    base["tokenizer_file_sha256"] = _digest_map(
        base["tokenizer_file_sha256"],
        "base_model tokenizer_file_sha256",
        allowed_keys=_TOKENIZER_FILES,
    )
    base["versions"] = _version_map(
        base["versions"], "base_model versions", allowed_keys=set(_BASE_VERSION_PINS)
    )
    return base


def _require_fixed_loaded_base(base: Mapping[str, object], context: _ProvenanceContext) -> None:
    if (
        base["model_id"] != contracts.MODEL_ID
        or base["model_revision"] != contracts.MODEL_REVISION
        or base["model_class"] != "Qwen3_5ForCausalLM"
        or base["config_class"] != "Qwen3_5TextConfig"
        or base["layer_count"] != 24
        or base["effective_dtype"] != "torch.bfloat16"
        or base["device"] != "cuda:0"
        or base["attention_implementation"] != "eager"
        or base["use_kernels"] is not False
        or base["use_hub_kernels"] != "NO"
        or base["tied_embeddings"] is not True
        or base["no_meta_parameters"] is not True
    ):
        raise ValueError("loaded base model differs from the fixed identity")
    diagnostics = cast(Mapping[str, object], base["load_diagnostics"])
    if any(diagnostics[name] for name in _DIAGNOSTIC_FIELDS):
        raise ValueError("loaded base model has nonempty load diagnostics")
    if base["versions"] != _BASE_VERSION_PINS:
        raise ValueError("loaded base package versions differ from the fixed pins")
    if base["tokenizer_file_sha256"] != context.tokenizer_hashes:
        raise ValueError("loaded base tokenizer hashes differ from the payload pins")


def validate_provenance(
    value: object,
    *,
    payload: object,
    passed: bool,
    require_loaded: bool = False,
    require_reload: bool = False,
) -> dict[str, object]:
    """Validate exact provenance while preserving well-formed failure observations."""

    for label, flag in (
        ("passed", passed),
        ("require_loaded", require_loaded),
        ("require_reload", require_reload),
    ):
        if type(flag) is not bool:
            raise ValueError(f"{label} must be a boolean")
    context = _provenance_context(payload)
    if require_reload and context.role == contracts.ROLE_UNCHANGED:
        raise ValueError("require_reload is invalid for unchanged role")
    provenance = _json_object(value, "provenance")
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("provenance has an unexpected schema")
    if provenance["model_id"] != context.model_id:
        raise ValueError("provenance model_id differs from payload")
    if provenance["model_revision"] != context.model_revision:
        raise ValueError("provenance model_revision differs from payload")
    declared_sources = _digest_map(provenance["source_file_sha256"], "source_file_sha256")
    if declared_sources != context.source_hashes:
        raise ValueError("source_file_sha256 differs from payload")
    provenance["source_file_sha256"] = declared_sources

    measured_sources = provenance["measured_source_file_sha256"]
    if measured_sources is not None:
        measured_sources = _digest_map(
            measured_sources,
            "measured_source_file_sha256",
            allowed_keys=set(context.source_hashes),
        )
    provenance["measured_source_file_sha256"] = measured_sources

    runtime_versions = provenance["runtime_versions"]
    if runtime_versions is not None:
        runtime_versions = _version_map(
            runtime_versions,
            "runtime_versions",
            allowed_keys=set(context.runtime_versions),
        )
    provenance["runtime_versions"] = runtime_versions

    raw_base = provenance["base_model"]
    base = None if raw_base is None else _validate_base_model(raw_base)
    provenance["base_model"] = base

    cuda_device = provenance["cuda_device"]
    if cuda_device is not None:
        cuda_device = _nonblank_string(cuda_device, "cuda_device")
    provenance["cuda_device"] = cuda_device

    raw_reload_hashes = provenance["reload_tokenizer_file_sha256"]
    reload_hashes: dict[str, str] | None = None
    if context.role == contracts.ROLE_UNCHANGED:
        if raw_reload_hashes is not None:
            raise ValueError("unchanged role must not report reload tokenizer hashes")
    elif raw_reload_hashes is not None:
        reload_hashes = _digest_map(
            raw_reload_hashes,
            "reload_tokenizer_file_sha256",
            allowed_keys=_TOKENIZER_FILES,
            exact_keys=_TOKENIZER_FILES,
        )
    provenance["reload_tokenizer_file_sha256"] = reload_hashes

    complete = passed or require_loaded or require_reload
    reload_required = require_reload or (passed and context.role != contracts.ROLE_UNCHANGED)
    if complete:
        if measured_sources is None:
            raise ValueError("measured_source_file_sha256 is required when the model is loaded")
        if measured_sources != context.source_hashes:
            raise ValueError("measured_source_file_sha256 differs from payload")
        if runtime_versions is None:
            raise ValueError("runtime_versions is required when the model is loaded")
        if (
            runtime_versions != context.runtime_versions
            or runtime_versions != contracts.RUNTIME_VERSION_PINS
        ):
            raise ValueError("runtime_versions differ from payload or frozen pins")
        if base is None:
            raise ValueError("base_model is required when the model is loaded")
        _require_fixed_loaded_base(base, context)
        if cuda_device is None:
            raise ValueError("cuda_device is required when the model is loaded")
        if _CUDA_INDEX_ONLY.fullmatch(cuda_device.strip()):
            raise ValueError("cuda_device must be an observed GPU name, not a CUDA index")
    if reload_required:
        if base is None:
            raise ValueError("require_reload needs loaded base model evidence")
        if reload_hashes is None:
            raise ValueError("reload tokenizer hashes are required")
        if reload_hashes != context.tokenizer_hashes:
            raise ValueError("reload tokenizer hashes differ from payload pins")
    return provenance


def validate_selected_adapter_identity(
    value: object, *, payload: object, required: bool
) -> dict[str, object] | None:
    """Validate a selected adapter descriptor against its pinned payload selection."""

    if type(required) is not bool:
        raise ValueError("required must be a boolean")
    if value is None:
        if required:
            raise ValueError("selected adapter identity is required")
        return None
    context = _selected_context(payload)
    identity = _json_object(value, "selected adapter identity")
    if set(identity) != {"snapshot", "tensor_sha256", "adapter_dtype"}:
        raise ValueError("selected adapter identity has an unexpected schema")
    snapshot = _selected_snapshot(identity["snapshot"], "selected snapshot")
    if snapshot != context.selected_snapshot:
        raise ValueError("selected snapshot differs from the payload selection")
    tensor_digest = validate_sha256(identity["tensor_sha256"], "selected tensor SHA-256")
    if (
        tensor_digest != contracts.SELECTED_TENSOR_SHA256
        or tensor_digest != context.selected_tensor_sha256
    ):
        raise ValueError("selected tensor digest differs from the fixed payload pin")
    if identity["adapter_dtype"] != "torch.float32":
        raise ValueError("selected adapter dtype must be torch.float32")
    return {
        "snapshot": snapshot,
        "tensor_sha256": tensor_digest,
        "adapter_dtype": "torch.float32",
    }
