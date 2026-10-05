"""Tests for fixed runtime-study provenance and selected adapter identity."""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from experiments import runtime_rule_study_contracts as contracts
from experiments.runtime_rule_study_provenance import (
    initial_provenance,
    validate_provenance,
    validate_selected_adapter_identity,
)

_SOURCE_HASHES = {
    "experiments/example.py": "a" * 64,
    "docs/runtime-study.md": "b" * 64,
}
_TOKENIZER_HASHES = {
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "d" * 64,
    "vocab.json": "e" * 64,
    "merges.txt": "f" * 64,
}
_SNAPSHOT_PATH = "/artifacts/runs/selected-training-parent/adapter-update-378"


def _snapshot(*, include_readme: bool = False) -> dict[str, object]:
    files_sha256 = {
        "adapter_config.json": "1" * 64,
        "adapter_model.safetensors": "2" * 64,
    }
    if include_readme:
        files_sha256["README.md"] = "3" * 64
    return {"update": 378, "path": _SNAPSHOT_PATH, "files_sha256": files_sha256}


def _payload(role: str = "continued_practice") -> dict[str, object]:
    return {
        "role": role,
        "source_file_sha256": dict(_SOURCE_HASHES),
        "study": {
            "model_id": contracts.MODEL_ID,
            "model_revision": contracts.MODEL_REVISION,
            "runtime_versions": dict(contracts.RUNTIME_VERSION_PINS),
            "tokenizer_file_sha256": dict(_TOKENIZER_HASHES),
            "selected_tensor_sha256": contracts.SELECTED_TENSOR_SHA256,
            "selection": {"snapshot": _snapshot(include_readme=True)},
        },
    }


def _base_model(payload: dict[str, object]) -> dict[str, object]:
    study = cast(dict[str, object], payload["study"])
    versions = {
        name: version
        for name, version in contracts.RUNTIME_VERSION_PINS.items()
        if name != "Pillow"
    }
    return {
        "model_id": contracts.MODEL_ID,
        "model_revision": contracts.MODEL_REVISION,
        "model_class": "Qwen3_5ForCausalLM",
        "config_class": "Qwen3_5TextConfig",
        "layer_count": 24,
        "tied_embeddings": True,
        "no_meta_parameters": True,
        "load_diagnostics": {
            "error_msgs": [],
            "mismatched_keys": [],
            "missing_keys": [],
            "unexpected_keys": [],
        },
        "tokenizer_file_sha256": deepcopy(study["tokenizer_file_sha256"]),
        "effective_dtype": "torch.bfloat16",
        "device": "cuda:0",
        "attention_implementation": "eager",
        "use_kernels": False,
        "use_hub_kernels": "NO",
        "versions": versions,
    }


def _complete_provenance(payload: dict[str, object]) -> dict[str, object]:
    provenance = initial_provenance(payload)
    provenance.update(
        {
            "measured_source_file_sha256": deepcopy(_SOURCE_HASHES),
            "runtime_versions": dict(contracts.RUNTIME_VERSION_PINS),
            "base_model": _base_model(payload),
            "cuda_device": "NVIDIA A10",
        }
    )
    if payload["role"] != contracts.ROLE_UNCHANGED:
        provenance["reload_tokenizer_file_sha256"] = deepcopy(_TOKENIZER_HASHES)
    return provenance


def test_initial_provenance_declares_payload_identity_and_null_measurements() -> None:
    payload = _payload()

    first = initial_provenance(payload)
    second = initial_provenance(payload)

    assert first == {
        "model_id": contracts.MODEL_ID,
        "model_revision": contracts.MODEL_REVISION,
        "source_file_sha256": _SOURCE_HASHES,
        "measured_source_file_sha256": None,
        "runtime_versions": None,
        "base_model": None,
        "cuda_device": None,
        "reload_tokenizer_file_sha256": None,
    }
    assert first is not second
    assert first["source_file_sha256"] is not payload["source_file_sha256"]


def test_initial_provenance_requires_only_identity_fields_from_the_payload() -> None:
    payload = {
        "source_file_sha256": dict(_SOURCE_HASHES),
        "study": {
            "model_id": contracts.MODEL_ID,
            "model_revision": contracts.MODEL_REVISION,
        },
    }

    provenance = initial_provenance(payload)

    assert provenance["model_id"] == contracts.MODEL_ID
    assert provenance["source_file_sha256"] == _SOURCE_HASHES


def test_failed_preflight_accepts_empty_partial_measurements_and_detaches_them() -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    provenance["measured_source_file_sha256"] = {}
    provenance["runtime_versions"] = {}

    validated = validate_provenance(provenance, payload=payload, passed=False)

    assert validated == provenance
    assert validated is not provenance
    assert validated["measured_source_file_sha256"] is not provenance["measured_source_file_sha256"]


def test_failed_provenance_preserves_valid_adverse_observations() -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    base_model = _base_model(payload)
    base_model["model_class"] = "UnexpectedModelClass"
    base_model["effective_dtype"] = "torch.float32"
    base_model["load_diagnostics"] = {
        "error_msgs": ["synthetic load error"],
        "mismatched_keys": [],
        "missing_keys": ["synthetic.key"],
        "unexpected_keys": [],
    }
    provenance.update(
        {
            "measured_source_file_sha256": {
                "experiments/example.py": "9" * 64,
            },
            "runtime_versions": {"torch": "9.9.9"},
            "base_model": base_model,
            "cuda_device": "NVIDIA different GPU",
        }
    )

    validated = validate_provenance(provenance, payload=payload, passed=False)

    assert validated["measured_source_file_sha256"] == {
        "experiments/example.py": "9" * 64,
    }
    assert validated["runtime_versions"] == {"torch": "9.9.9"}
    assert validated["base_model"] == base_model
    assert validated["cuda_device"] == "NVIDIA different GPU"


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("model_id", "other/model", "model_id differs from payload"),
        ("model_revision", "other-revision", "model_revision differs from payload"),
        ("source_file_sha256", {"experiments/example.py": "a" * 64}, "source_file_sha256"),
        (
            "source_file_sha256",
            {**_SOURCE_HASHES, "experiments/new.py": "3" * 64},
            "source_file_sha256 differs from payload",
        ),
        (
            "source_file_sha256",
            {**_SOURCE_HASHES, "experiments/example.py": "9" * 64},
            "source_file_sha256 differs from payload",
        ),
        ("extra", "unexpected", "unexpected schema"),
    ],
)
def test_provenance_requires_exact_payload_identity_and_schema(
    field: str, replacement: object, message: str
) -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    provenance[field] = replacement

    with pytest.raises(ValueError, match=message):
        validate_provenance(provenance, payload=payload, passed=False)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "measured_source_file_sha256",
            {"unlisted.py": "a" * 64},
            "outside its allowlist",
        ),
        ("measured_source_file_sha256", {"experiments/example.py": "bad"}, "SHA-256"),
        ("runtime_versions", {"unknown-package": "1.0"}, "outside its allowlist"),
        ("runtime_versions", {"torch": "  "}, "nonblank strings"),
        ("runtime_versions", {"torch": 2}, "nonblank strings"),
    ],
)
def test_partial_measured_maps_reject_unknown_keys_and_malformed_values(
    field: str, replacement: object, message: str
) -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    provenance[field] = replacement

    with pytest.raises(ValueError, match=message):
        validate_provenance(provenance, payload=payload, passed=False)


def test_failed_provenance_requires_declared_source_hashes_to_equal_payload() -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    provenance["source_file_sha256"] = {**_SOURCE_HASHES, "extra.py": "3" * 64}

    with pytest.raises(ValueError, match="source_file_sha256 differs from payload"):
        validate_provenance(provenance, payload=payload, passed=False)


@pytest.mark.parametrize("role", ["continued_practice", "runtime_mix"])
def test_passed_training_requires_complete_matching_measured_provenance(role: str) -> None:
    payload = _payload(role)
    provenance = _complete_provenance(payload)

    validated = validate_provenance(provenance, payload=payload, passed=True)

    assert validated == provenance


def test_passed_unchanged_requires_loaded_provenance_but_no_reload_tokenizer_map() -> None:
    payload = _payload(contracts.ROLE_UNCHANGED)
    provenance = _complete_provenance(payload)
    assert provenance["reload_tokenizer_file_sha256"] is None

    assert validate_provenance(provenance, payload=payload, passed=True) == provenance


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "measured_source_file_sha256",
            {"experiments/example.py": "9" * 64},
            "measured_source_file_sha256 differs from payload",
        ),
        (
            "runtime_versions",
            {**contracts.RUNTIME_VERSION_PINS, "torch": "9.9.9"},
            "runtime_versions differ from payload or frozen pins",
        ),
        ("measured_source_file_sha256", None, "measured_source_file_sha256 is required"),
        ("runtime_versions", None, "runtime_versions is required"),
        ("base_model", None, "base_model is required"),
        ("cuda_device", None, "cuda_device is required"),
        ("cuda_index", None, "cuda_device must be an observed GPU name"),
    ],
)
def test_passed_provenance_rejects_incomplete_or_mismatching_measurements(
    field: str, replacement: object, message: str
) -> None:
    payload = _payload()
    provenance = _complete_provenance(payload)
    if field == "cuda_index":
        provenance["cuda_device"] = "cuda:0"
    else:
        provenance[field] = replacement

    with pytest.raises(ValueError, match=message):
        validate_provenance(provenance, payload=payload, passed=True)


def test_require_loaded_runs_complete_checks_even_for_a_failed_result() -> None:
    payload = _payload()
    provenance = _complete_provenance(payload)
    provenance["runtime_versions"] = {"torch": "9.9.9"}

    with pytest.raises(ValueError, match="runtime_versions differ from payload or frozen pins"):
        validate_provenance(provenance, payload=payload, passed=False, require_loaded=True)

    provenance = _complete_provenance(payload)
    validated = validate_provenance(provenance, payload=payload, passed=False, require_loaded=True)
    assert validated == provenance


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model_class", "loaded base model differs from the fixed identity"),
        ("model_id", "loaded base model differs from the fixed identity"),
        ("model_revision", "loaded base model differs from the fixed identity"),
        ("config_class", "loaded base model differs from the fixed identity"),
        ("layer_count", "loaded base model differs from the fixed identity"),
        ("effective_dtype", "loaded base model differs from the fixed identity"),
        ("device", "loaded base model differs from the fixed identity"),
        ("attention_implementation", "loaded base model differs from the fixed identity"),
        ("use_kernels", "loaded base model differs from the fixed identity"),
        ("use_hub_kernels", "loaded base model differs from the fixed identity"),
        ("tied_embeddings", "loaded base model differs from the fixed identity"),
        ("no_meta_parameters", "loaded base model differs from the fixed identity"),
        ("diagnostics", "loaded base model has nonempty load diagnostics"),
        ("tokenizer", "loaded base tokenizer hashes differ from the payload pins"),
        ("versions", "loaded base package versions differ from the fixed pins"),
    ],
)
def test_loaded_base_must_match_the_fixed_identity(mutation: str, message: str) -> None:
    payload = _payload()
    provenance = _complete_provenance(payload)
    base_model = cast(dict[str, object], provenance["base_model"])
    if mutation == "model_class":
        base_model["model_class"] = "OtherModel"
    elif mutation == "model_id":
        base_model["model_id"] = "other/model"
    elif mutation == "model_revision":
        base_model["model_revision"] = "other-revision"
    elif mutation == "config_class":
        base_model["config_class"] = "OtherConfig"
    elif mutation == "layer_count":
        base_model["layer_count"] = 23
    elif mutation == "effective_dtype":
        base_model["effective_dtype"] = "torch.float32"
    elif mutation == "device":
        base_model["device"] = "cpu"
    elif mutation == "attention_implementation":
        base_model["attention_implementation"] = "sdpa"
    elif mutation == "use_kernels":
        base_model["use_kernels"] = True
    elif mutation == "use_hub_kernels":
        base_model["use_hub_kernels"] = "YES"
    elif mutation == "tied_embeddings":
        base_model["tied_embeddings"] = False
    elif mutation == "no_meta_parameters":
        base_model["no_meta_parameters"] = False
    elif mutation == "diagnostics":
        diagnostics = cast(dict[str, object], base_model["load_diagnostics"])
        diagnostics["missing_keys"] = ["synthetic.missing"]
    elif mutation == "tokenizer":
        tokenizers = cast(dict[str, str], base_model["tokenizer_file_sha256"])
        tokenizers["vocab.json"] = "9" * 64
    elif mutation == "versions":
        versions = cast(dict[str, str], base_model["versions"])
        versions["torch"] = "9.9.9"

    with pytest.raises(ValueError, match=message):
        validate_provenance(provenance, payload=payload, passed=True)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("layer_count", True, "base_model layer_count must be an integer"),
        ("tied_embeddings", 1, "base_model tied_embeddings must be a boolean"),
        ("use_kernels", 0, "base_model use_kernels must be a boolean"),
        ("use_hub_kernels", False, "base_model use_hub_kernels must be a nonblank string or null"),
        ("versions", {"torch": ""}, "base_model versions values must be nonblank strings"),
        (
            "load_diagnostics",
            {"error_msgs": {}, "mismatched_keys": [], "missing_keys": [], "unexpected_keys": []},
            "load_diagnostics values must be arrays",
        ),
        ("extra", "not allowed", "base_model has an unexpected schema"),
    ],
)
def test_failed_base_model_still_requires_a_complete_typed_schema(
    field: str, replacement: object, message: str
) -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    base_model = _base_model(payload)
    base_model[field] = replacement
    provenance["base_model"] = base_model

    with pytest.raises(ValueError, match=message):
        validate_provenance(provenance, payload=payload, passed=False)


def test_require_reload_needs_a_loaded_base_and_pinned_tokenizer_map() -> None:
    payload = _payload()
    provenance = _complete_provenance(payload)
    provenance["base_model"] = None

    with pytest.raises(ValueError, match="base_model is required when the model is loaded"):
        validate_provenance(provenance, payload=payload, passed=False, require_reload=True)

    provenance = _complete_provenance(payload)
    provenance["reload_tokenizer_file_sha256"] = {**_TOKENIZER_HASHES, "vocab.json": "9" * 64}
    with pytest.raises(ValueError, match="reload tokenizer hashes differ from payload pins"):
        validate_provenance(provenance, payload=payload, passed=False, require_reload=True)

    provenance = _complete_provenance(payload)
    assert (
        validate_provenance(provenance, payload=payload, passed=False, require_reload=True)
        == provenance
    )


def test_require_reload_alone_requires_complete_loaded_provenance() -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    provenance["base_model"] = _base_model(payload)
    provenance["reload_tokenizer_file_sha256"] = deepcopy(_TOKENIZER_HASHES)

    with pytest.raises(ValueError, match="measured_source_file_sha256 is required"):
        validate_provenance(provenance, payload=payload, passed=False, require_reload=True)


def test_require_reload_alone_rejects_wrong_loaded_base_identity() -> None:
    payload = _payload()
    provenance = _complete_provenance(payload)
    base_model = cast(dict[str, object], provenance["base_model"])
    base_model["model_class"] = "UnexpectedModelClass"

    with pytest.raises(ValueError, match="loaded base model differs from the fixed identity"):
        validate_provenance(provenance, payload=payload, passed=False, require_reload=True)


def test_failed_reload_may_preserve_mismatching_full_tokenizer_hashes() -> None:
    payload = _payload()
    provenance = _complete_provenance(payload)
    provenance["reload_tokenizer_file_sha256"] = {**_TOKENIZER_HASHES, "vocab.json": "9" * 64}

    validated = validate_provenance(provenance, payload=payload, passed=False)

    assert validated["reload_tokenizer_file_sha256"] == provenance["reload_tokenizer_file_sha256"]


@pytest.mark.parametrize("role", [contracts.ROLE_UNCHANGED, "continued_practice"])
def test_reload_tokenizer_hashes_must_be_null_for_unchanged_and_complete_when_present(
    role: str,
) -> None:
    payload = _payload(role)
    provenance = _complete_provenance(payload)
    if role == contracts.ROLE_UNCHANGED:
        provenance["reload_tokenizer_file_sha256"] = dict(_TOKENIZER_HASHES)
        with pytest.raises(
            ValueError, match="unchanged role must not report reload tokenizer hashes"
        ):
            validate_provenance(provenance, payload=payload, passed=False)
    else:
        provenance["reload_tokenizer_file_sha256"] = {"tokenizer.json": "c" * 64}
        with pytest.raises(ValueError, match="must contain exactly its pinned keys"):
            validate_provenance(provenance, payload=payload, passed=False)


def test_require_reload_is_invalid_for_unchanged_role() -> None:
    payload = _payload(contracts.ROLE_UNCHANGED)
    provenance = _complete_provenance(payload)

    with pytest.raises(ValueError, match="require_reload is invalid for unchanged role"):
        validate_provenance(provenance, payload=payload, passed=False, require_reload=True)


@pytest.mark.parametrize(
    ("flag", "value"),
    [("passed", 1), ("require_loaded", 1), ("require_reload", 1)],
)
def test_provenance_flags_reject_boolean_numeric_aliases(flag: str, value: object) -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    arguments: dict[str, object] = {"payload": payload, "passed": False}
    arguments[flag] = value

    with pytest.raises(ValueError, match=f"{flag} must be a boolean"):
        validate_provenance(provenance, **arguments)  # type: ignore[arg-type]


def test_provenance_rejects_non_json_keys_and_nonfinite_nested_diagnostics() -> None:
    payload = _payload()
    provenance = initial_provenance(payload)
    provenance["runtime_versions"] = {1: "1.0"}

    with pytest.raises(ValueError, match="object keys must be strings"):
        validate_provenance(provenance, payload=payload, passed=False)

    provenance = initial_provenance(payload)
    base_model = _base_model(payload)
    diagnostics = cast(dict[str, object], base_model["load_diagnostics"])
    diagnostics["error_msgs"] = [float("nan")]
    provenance["base_model"] = base_model
    with pytest.raises(ValueError, match="strict JSON"):
        validate_provenance(provenance, payload=payload, passed=False)


def test_selected_adapter_identity_allows_null_only_when_optional() -> None:
    payload = _payload()

    assert validate_selected_adapter_identity(None, payload=payload, required=False) is None
    with pytest.raises(ValueError, match="selected adapter identity is required"):
        validate_selected_adapter_identity(None, payload=payload, required=True)


def test_selected_adapter_identity_matches_payload_snapshot_and_keeps_readme() -> None:
    payload = _payload()
    identity: dict[str, object] = {
        "snapshot": _snapshot(include_readme=True),
        "tensor_sha256": contracts.SELECTED_TENSOR_SHA256,
        "adapter_dtype": "torch.float32",
    }

    validated = validate_selected_adapter_identity(identity, payload=payload, required=True)

    assert validated == identity
    assert validated is not identity
    assert validated is not None
    snapshot = validated["snapshot"]
    assert snapshot is not identity["snapshot"]
    assert cast(dict[str, object], snapshot)["path"] == _SNAPSHOT_PATH
    assert "README.md" in cast(dict[str, object], cast(dict[str, object], snapshot)["files_sha256"])
    files = cast(dict[str, object], snapshot)["files_sha256"]
    assert files is not cast(dict[str, object], identity["snapshot"])["files_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("arm_path", "selected snapshot differs from the payload selection"),
        ("update", "selected snapshot differs from the payload selection"),
        ("file_hash", "selected snapshot differs from the payload selection"),
        ("tensor", "selected tensor digest differs from the fixed payload pin"),
        ("dtype", "selected adapter dtype must be torch.float32"),
        ("boolean_update", "selected snapshot update must be a positive integer"),
        ("extra", "selected adapter identity has an unexpected schema"),
    ],
)
def test_selected_adapter_identity_rejects_identity_drift(mutation: str, message: str) -> None:
    payload = _payload()
    identity: dict[str, object] = {
        "snapshot": _snapshot(include_readme=True),
        "tensor_sha256": contracts.SELECTED_TENSOR_SHA256,
        "adapter_dtype": "torch.float32",
    }
    if mutation == "arm_path":
        snapshot = cast(dict[str, object], identity["snapshot"])
        snapshot["path"] = "/artifacts/runs/current-role-arm/adapter-update-378"
    elif mutation == "update":
        snapshot = cast(dict[str, object], identity["snapshot"])
        snapshot["update"] = 377
    elif mutation == "boolean_update":
        snapshot = cast(dict[str, object], identity["snapshot"])
        snapshot["update"] = True
    elif mutation == "file_hash":
        snapshot = cast(dict[str, object], identity["snapshot"])
        files = cast(dict[str, str], snapshot["files_sha256"])
        files["adapter_model.safetensors"] = "9" * 64
    elif mutation == "tensor":
        identity["tensor_sha256"] = "9" * 64
    elif mutation == "dtype":
        identity["adapter_dtype"] = "torch.bfloat16"
    elif mutation == "extra":
        identity["unexpected"] = True

    with pytest.raises(ValueError, match=message):
        validate_selected_adapter_identity(identity, payload=payload, required=True)


@pytest.mark.parametrize(
    ("payload_mutation", "message"),
    [
        ("missing_role", "payload is missing required field role"),
        ("missing_source", "payload is missing required field source_file_sha256"),
        ("missing_study", "payload is missing required field study"),
        (
            "missing_selection_snapshot",
            "payload study selection is missing required field snapshot",
        ),
        ("wrong_selected_digest", "payload selected tensor digest differs from the fixed pin"),
    ],
)
def test_payload_context_requires_the_fields_used_by_this_validator(
    payload_mutation: str, message: str
) -> None:
    payload = _payload()
    if payload_mutation == "missing_role":
        payload.pop("role")
    elif payload_mutation == "missing_source":
        payload.pop("source_file_sha256")
    elif payload_mutation == "missing_study":
        payload.pop("study")
    elif payload_mutation == "missing_selection_snapshot":
        study = cast(dict[str, object], payload["study"])
        study["selection"] = {}
    elif payload_mutation == "wrong_selected_digest":
        study = cast(dict[str, object], payload["study"])
        study["selected_tensor_sha256"] = "9" * 64

    with pytest.raises(ValueError, match=message):
        if payload_mutation in {"missing_role", "missing_source", "missing_study"}:
            provenance = initial_provenance(payload)
            validate_provenance(provenance, payload=payload, passed=False)
        else:
            identity = {
                "snapshot": _snapshot(include_readme=True),
                "tensor_sha256": contracts.SELECTED_TENSOR_SHA256,
                "adapter_dtype": "torch.float32",
            }
            validate_selected_adapter_identity(identity, payload=payload, required=False)


def test_selected_identity_required_flag_rejects_integer_alias() -> None:
    with pytest.raises(ValueError, match="required must be a boolean"):
        validate_selected_adapter_identity(None, payload=_payload(), required=1)  # type: ignore[arg-type]
