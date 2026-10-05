"""Private metadata validation for runtime-rule study worker payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from experiments import (
    adapter_transfer_contracts,
    mixture_training_contracts,
    runtime_rule_study_cache,
    runtime_rule_study_contracts,
    runtime_rule_study_inputs,
    runtime_rule_study_tokens,
)
from experiments.mixture_training_contracts import (
    canonical_json,
    normalize_json_object,
    validate_sha256,
)


def _validate_source_path(path: object) -> str:
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or any(character in path for character in "*?[]")
        or any(part in ("", ".", "..") or part.startswith(".") for part in path.split("/"))
    ):
        raise ValueError("source path is unsafe")
    if path in {"pyproject.toml", "uv.lock"}:
        return path
    if path.startswith("experiments/") and path.endswith(".py"):
        return path
    if path.startswith("src/reflex_decisions/") and path.endswith(".py"):
        return path
    if path.startswith("docs/") and path.endswith("protocol.md"):
        return path
    raise ValueError("source path is outside the fixed upload file classes")


def validate_source_file_sha256(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(not isinstance(path, str) for path in value):
        raise ValueError("source_file_sha256 must be a path-to-digest object")
    source_map = normalize_json_object(value, "source_file_sha256")
    for path, digest in source_map.items():
        _validate_source_path(path)
        validate_sha256(digest, f"source SHA-256 for {path}")
    if not _MINIMUM_SOURCE_PATHS.issubset(source_map):
        raise ValueError("source_file_sha256 omits required historical or study source paths")
    return {path: source_map[path] for path in sorted(source_map)}


_STUDY_FIELDS = {
    "bundle_sha256",
    "protocol_sha256",
    "model_id",
    "model_revision",
    "selection",
    "selected_tensor_sha256",
    "tokenizer_file_sha256",
    "runtime_versions",
    "input_file_sha256",
    "retention_cache",
    "compilation_sha256",
    "schedule_sha256",
}
_RETENTION_CACHE_FIELDS = {
    "receipt_file_sha256",
    "outputs_sha256",
    "cache_sha256",
    "source_file_sha256",
}
_SOURCE_PATHS = set(adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS)
_STUDY_PROTOCOL_PATH = runtime_rule_study_inputs._STUDY_PROTOCOL_PATH
_STUDY_PROTOCOL_SHA256 = runtime_rule_study_inputs._STUDY_PROTOCOL_SHA256
_TRANSFER_SUMMARY_PATH = runtime_rule_study_inputs._TRANSFER_SUMMARY_PATH
_TRANSFER_SUMMARY_SHA256 = runtime_rule_study_inputs._TRANSFER_SUMMARY_SHA256
_SELECTION_PATH = adapter_transfer_contracts.SELECTION_PATH
_SELECTION_SHA256 = adapter_transfer_contracts.EXPECTED_SELECTION_SHA256
_NATURAL_RECEIPT_PATH = runtime_rule_study_cache.NATURAL_RECEIPT_PATH
_NATURAL_RECEIPT_SHA256 = runtime_rule_study_cache.NATURAL_RECEIPT_SHA256
_TRANSFER_RECEIPT_PATH = runtime_rule_study_cache.TRANSFER_RECEIPT_PATH
_TRANSFER_RECEIPT_SHA256 = runtime_rule_study_cache.TRANSFER_RECEIPT_SHA256
_EXPECTED_INPUTS = {
    **mixture_training_contracts.DATA_FILE_SHA256,
    **runtime_rule_study_inputs._NEW_FILE_SHA256,
    **adapter_transfer_contracts.PANEL_FILE_SHA256,
    _SELECTION_PATH: _SELECTION_SHA256,
    _STUDY_PROTOCOL_PATH: _STUDY_PROTOCOL_SHA256,
    _TRANSFER_SUMMARY_PATH: _TRANSFER_SUMMARY_SHA256,
}
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
}
_EXPECTED_RECEIPTS = {
    _NATURAL_RECEIPT_PATH: _NATURAL_RECEIPT_SHA256,
    _TRANSFER_RECEIPT_PATH: _TRANSFER_RECEIPT_SHA256,
}
_STUDY_SOURCE_PATHS = {
    "experiments/runtime_rule_study_data.py",
    "experiments/runtime_rule_study_inputs.py",
    "experiments/runtime_rule_study_cache.py",
    "experiments/runtime_rule_study_tokens.py",
    "experiments/runtime_rule_study_contracts.py",
    "experiments/runtime_rule_study_bundle.py",
    "experiments/runtime_rule_study_statistics.py",
    "experiments/runtime_rule_study_payloads.py",
    "experiments/runtime_rule_study_payload_rows.py",
    "experiments/runtime_rule_study_payload_metadata.py",
    "docs/runtime-rule-study-protocol.md",
    "pyproject.toml",
    "uv.lock",
}
_MINIMUM_SOURCE_PATHS = _SOURCE_PATHS | _STUDY_SOURCE_PATHS


def _string_hash_map(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a path-to-digest object")
    normalized = normalize_json_object(value, label)
    result = {}
    for key, digest in normalized.items():
        result[key] = validate_sha256(digest, f"{label} {key}")
    return {key: result[key] for key in sorted(result)}


def _fixed_input_hashes() -> dict[str, str]:
    fixed = {
        path: validate_sha256(digest, f"fixed input SHA-256 for {path}")
        for path, digest in _EXPECTED_INPUTS.items()
    }
    # Explicit cardinalities prevent dict merges from hiding a duplicate path.
    if len(_SOURCE_PATHS) != 60 or len(fixed) != 27 or len(_SOURCE_PATHS | set(fixed)) != 87:
        raise ValueError("fixed input hash allowlists do not form the exact 87-key map")
    return fixed


def _validate_input_hashes(value: object) -> dict[str, str]:
    hashes = _string_hash_map(value, "input_file_sha256")
    expected = _fixed_input_hashes()
    if set(hashes) != _SOURCE_PATHS | set(expected):
        raise ValueError("input_file_sha256 differs from the exact 87-key fixed allowlist")
    for path, digest in expected.items():
        if hashes[path] != digest:
            raise ValueError(f"input_file_sha256 differs from the fixed pin for {path}")
    return hashes


def _validate_tokenizer_hashes(value: object) -> dict[str, str]:
    hashes = _string_hash_map(value, "tokenizer_file_sha256")
    if set(hashes) != _TOKENIZER_FILES:
        raise ValueError("tokenizer_file_sha256 must contain the exact four tokenizer files")
    for name in hashes:
        if (
            not name
            or name.startswith(".")
            or "/" in name
            or "\\" in name
            or ".." in name
            or any(character in name for character in "*?[]")
        ):
            raise ValueError("tokenizer filename is unsafe")
    if hashes.get("tokenizer.json") != runtime_rule_study_tokens.TOKENIZER_SHA256:
        raise ValueError("tokenizer.json differs from the frozen CPU tokenizer pin")
    return hashes


def _validate_retention_cache(value: object) -> dict[str, object]:
    cache = normalize_json_object(value, "retention_cache")
    if set(cache) != _RETENTION_CACHE_FIELDS:
        raise ValueError("retention_cache has an unexpected schema")
    receipts = normalize_json_object(cache["receipt_file_sha256"], "receipt_file_sha256")
    if canonical_json(receipts) != canonical_json(_EXPECTED_RECEIPTS):
        raise ValueError("retention receipt hashes differ from the frozen receipt pins")
    outputs_sha256 = validate_sha256(cache["outputs_sha256"], "cached outputs SHA-256")
    cache_sha256 = validate_sha256(cache["cache_sha256"], "retention cache SHA-256")
    source_hashes = _string_hash_map(cache["source_file_sha256"], "retention source_file_sha256")
    if set(source_hashes) != _SOURCE_PATHS:
        raise ValueError("retention source map differs from the exact historical 60 paths")
    return {
        "receipt_file_sha256": {path: receipts[path] for path in sorted(receipts)},
        "outputs_sha256": outputs_sha256,
        "cache_sha256": cache_sha256,
        "source_file_sha256": source_hashes,
    }


def _validate_schedule_hashes(value: object) -> dict[str, str]:
    hashes = _string_hash_map(value, "schedule_sha256")
    if set(hashes) != set(runtime_rule_study_contracts.ROLES[1:]):
        raise ValueError("schedule_sha256 must contain exactly both training-arm hashes")
    return hashes


def _validate_runtime_versions(value: object) -> dict[str, str]:
    actual = normalize_json_object(value, "runtime_versions")
    expected = runtime_rule_study_contracts.RUNTIME_VERSION_PINS
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("runtime_versions differ from the fixed nine package pins")
    return {name: actual[name] for name in sorted(actual)}


def _validated_study(study: object, source_file_sha256: object) -> dict[str, object]:
    result = normalize_json_object(study, "study")
    if set(result) != _STUDY_FIELDS:
        raise ValueError("study metadata has an unexpected schema")
    result["bundle_sha256"] = validate_sha256(result["bundle_sha256"], "bundle SHA-256")
    if result["protocol_sha256"] != runtime_rule_study_contracts.PROTOCOL_SHA256:
        raise ValueError("study protocol SHA-256 differs from the frozen protocol")
    if (
        result["model_id"] != runtime_rule_study_contracts.MODEL_ID
        or result["model_revision"] != runtime_rule_study_contracts.MODEL_REVISION
    ):
        raise ValueError("study model identity differs from the frozen model")
    selection = adapter_transfer_contracts.validate_selection(result["selection"])
    if canonical_json(selection) != canonical_json(result["selection"]):
        raise ValueError("study selection is not in canonical validated form")
    result["selection"] = selection
    result["selected_tensor_sha256"] = validate_sha256(
        result["selected_tensor_sha256"], "selected tensor SHA-256"
    )
    if result["selected_tensor_sha256"] != runtime_rule_study_contracts.SELECTED_TENSOR_SHA256:
        raise ValueError("selected tensor SHA-256 differs from the frozen adapter pin")
    tokenizers = _validate_tokenizer_hashes(result["tokenizer_file_sha256"])
    versions = _validate_runtime_versions(result["runtime_versions"])
    inputs = _validate_input_hashes(result["input_file_sha256"])
    cache = _validate_retention_cache(result["retention_cache"])
    if inputs[_STUDY_PROTOCOL_PATH] != result["protocol_sha256"]:
        raise ValueError("input protocol hash differs from study protocol identity")
    if inputs[_SELECTION_PATH] != _SELECTION_SHA256:
        raise ValueError("input selection file hash differs from the frozen selection pin")
    if inputs[_TRANSFER_SUMMARY_PATH] != _TRANSFER_SUMMARY_SHA256:
        raise ValueError("input adapter summary hash differs from its frozen pin")
    source_hashes = validate_source_file_sha256(source_file_sha256)
    cached_source_hashes = cast(dict[str, str], cache["source_file_sha256"])
    for path in _SOURCE_PATHS:
        if inputs[path] != cached_source_hashes[path]:
            raise ValueError(f"input and retention source hashes disagree for {path}")
        if source_hashes[path] != cached_source_hashes[path]:
            raise ValueError(f"upload and retention source hashes disagree for {path}")
    protocol_path = _STUDY_PROTOCOL_PATH
    if source_hashes[protocol_path] != runtime_rule_study_contracts.PROTOCOL_SHA256:
        raise ValueError("uploaded study protocol hash differs from its frozen pin")
    result["compilation_sha256"] = validate_sha256(
        result["compilation_sha256"], "compilation SHA-256"
    )
    schedule_hashes = _validate_schedule_hashes(result["schedule_sha256"])
    result["tokenizer_file_sha256"] = tokenizers
    result["runtime_versions"] = versions
    result["input_file_sha256"] = inputs
    result["retention_cache"] = cache
    result["schedule_sha256"] = schedule_hashes
    return result


def build_study_metadata(
    bundle: object, bundle_sha256: object, source_file_sha256: object
) -> dict[str, object]:
    envelope = normalize_json_object(bundle, "authenticated bundle")
    expected_digest = validate_sha256(bundle_sha256, "trusted bundle SHA-256")
    if envelope.get("bundle_sha256") != expected_digest:
        raise ValueError("authenticated bundle digest differs from the trusted digest")
    compilation = normalize_json_object(envelope.get("compilation"), "compilation")
    schedule_hashes = _validate_schedule_hashes(compilation.get("schedule_sha256"))
    result: dict[str, object] = {
        "bundle_sha256": expected_digest,
        "protocol_sha256": envelope.get("protocol_sha256"),
        "model_id": envelope.get("model_id"),
        "model_revision": envelope.get("model_revision"),
        "selection": envelope.get("selection"),
        "selected_tensor_sha256": envelope.get("selected_tensor_sha256"),
        "tokenizer_file_sha256": envelope.get("tokenizer_file_sha256"),
        "runtime_versions": envelope.get("runtime_versions"),
        "input_file_sha256": envelope.get("input_file_sha256"),
        "retention_cache": envelope.get("retention_cache"),
        "compilation_sha256": compilation.get("compilation_sha256"),
        "schedule_sha256": schedule_hashes,
    }
    return _validated_study(result, source_file_sha256)


def validate_study_metadata(study: object, source_file_sha256: object) -> dict[str, object]:
    return _validated_study(study, source_file_sha256)
