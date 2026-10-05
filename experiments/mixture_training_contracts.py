"""Shared CPU-only pins and strict JSON helpers for the controlled mixture run."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments import training_rehearsal_core

SCHEMA_VERSION = 2
HISTORICAL_PROTOCOL_SHA256 = {
    "docs/mixture-training-protocol.md": (
        "06d3014a45ebeb19500ba2ab002f0595c42b44e561fbb9efc329af0d02a1030c"
    ),
    "docs/mixture-training-seed2-protocol.md": (
        "2cf62fa31f3e52a41ebd632e4883fc6c2f082194a7fa9750b455eb78735c12c8"
    ),
}
PROTOCOL_PATH = "docs/natural-reasoning-protocol.md"
EXPECTED_PROTOCOL_SHA256 = "872cade19b7383722af28503a91c458d3fbe04838c8963e7da9644f1302379ec"
MODEL_ID = training_rehearsal_core.MODEL_ID
MODEL_REVISION = training_rehearsal_core.MODEL_REVISION
PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
MAX_INPUT_TOKENS = 2048
RELOAD_PARITY_COUNT = 32
MAX_TOTAL_FORWARDS = 12073
MAX_TOTAL_INPUT_TOKENS = MAX_TOTAL_FORWARDS * MAX_INPUT_TOKENS
LEARNING_RATE = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
WEIGHT_DECAY = 0.0
MAX_GRADIENT_NORM = 1.0

RUNTIME_VERSION_PINS = {
    "torch": "2.14.1+cu130",
    "torchvision": "0.29.1+cu130",
    "transformers": "5.18.0",
    "peft": "0.21.0",
    "Pillow": "12.0.0",
    "pydantic": "2.13.5",
    "huggingface-hub": "1.33.0",
    "tokenizers": "0.23.2",
    "safetensors": "0.8.0",
}

DATA_FILE_SHA256 = dict(
    [
        (
            "data/processed/real-pilot-v1.jsonl",
            "8d803048df38b1120d2a1e91f404a480fba6097f6f300c0df9b1ec2512cff7ba",
        ),
        (
            "data/pilots/real-pilot-v1-manifest.json",
            "baf4c270f5eb6005fbb8ce8c08daa7e01a7762a128d0d2384532f8e3eb9a2cef",
        ),
        (
            "data/pilots/real-pilot-v1-recipe.json",
            "2315f8c39efa95f2cc018e8d5a7281ba37f304f66161aa737a6ab9389bdded4f",
        ),
        (
            "data/processed/snli-balanced-v1/records.jsonl",
            "a1fa41d19b381e227ebca258b1561e7e39184f6f58a325aaa5a2b1ff61ddd98c",
        ),
        (
            "data/processed/snli-balanced-v1/manifest.json",
            "4ecab09c6f013c274b38a81c6ef226226fee6520745e01457f461c5e5dc628bd",
        ),
        (
            "data/processed/snli-balanced-v1/recipe.json",
            "e11fc94b99376f4860caf3b9afbb22026e6015ee7b4bfeb42344a7a0cc0bbd58",
        ),
        (
            "data/processed/synthetic-seed-v1-r2/records.jsonl",
            "30b0e07b89977b1345d403936ab666b06f822e13053ffe71cf11fb0243321e5c",
        ),
        (
            "data/processed/synthetic-seed-v1-r2/manifest.json",
            "b4e8655c38eb4fb0ac95ceaf7062aa0c43f981fe9b8309f568943f4ff6c59a11",
        ),
        (
            "data/processed/synthetic-seed-v1-r2/recipe.json",
            "31a830aa56f9c28d983eea8b8942ee9f9b9db2b5199d8b2e71d42a654a337433",
        ),
        (
            "data/processed/synthetic-seed-v1-r2/provenance.jsonl",
            "89c9b874f8de31bc3a61315cf4bd068147165195afb7f23472eede9ef31e45c5",
        ),
        (
            "data/processed/synthetic-seed-v1-r2/audit.json",
            "19f6b90426517bc50734a48f26e35d22ebc0aef794cac73e0adb29db9df666d1",
        ),
        (
            "data/raw/snli_1.0.zip",
            "afb3d70a5af5d8de0d9d81e2637e0fb8c22d1235c2749d83125ca43dab0dbd3e",
        ),
        (
            "data/processed/snli-training-v1/records.jsonl",
            "976a0c06f679b9bd989871580b893d32ed593b0912c191ccba56237f7407f2fc",
        ),
        (
            "data/processed/snli-training-v1/manifest.json",
            "4be0e00a9588fb7d9077b41c67227ad634737c74910680ad9c24d572c62618b7",
        ),
        (
            "data/processed/snli-training-v1/recipe.json",
            "aced7f92bdef65d97fad9fafb2d7f20abef107229bc8e5dea3594969dbd5de0d",
        ),
    ]
)

SNLI_SOURCE_FILE_SHA256 = {
    "src/reflex_decisions/snli_training_source.py": (
        "a4be159c6c799bf5215ccf5a337a25e34722be92b3c21a7ba6d9a4b8fb01148a"
    ),
    "src/reflex_decisions/snli_training_data.py": (
        "3106b9ec77bfeb6bbedd2e2768bbc01c4503a6b6f877611b8e9a4c1602b1ae4c"
    ),
    "experiments/prepare_snli_training.py": (
        "4e2e407aeb049a7f9433f32ae0ce831e9cdf6ce8b1387d5878721deac95db3bb"
    ),
}

# The analysis entrypoint is part of the signed source bundle once it is added.
SOURCE_FINGERPRINT_PATHS = (
    "experiments/__init__.py",
    "experiments/mixture_training_contracts.py",
    "experiments/mixture_training_core.py",
    "experiments/mixture_training_data.py",
    "experiments/mixture_training_evidence.py",
    "experiments/mixture_training_evaluations.py",
    "experiments/mixture_training_inputs.py",
    "experiments/mixture_training_outputs.py",
    "experiments/mixture_training_results.py",
    "experiments/mixture_training_runtime.py",
    "experiments/mixture_training_runtime_execution.py",
    "experiments/mixture_training_runtime_helpers.py",
    "experiments/mixture_training_runtime_scoring.py",
    "experiments/mixture_training_modal_host.py",
    "experiments/modal_mixture_training.py",
    "experiments/modal_train_rehearsal.py",
    "experiments/training_rehearsal_core.py",
    "experiments/real_pilot_core.py",
    "experiments/baseline_qwen.py",
    "experiments/modal_smoke.py",
    "experiments/analyze_real_pilot.py",
    "experiments/analyze_snli_diagnostic.py",
    "experiments/snli_diagnostic_core.py",
    "experiments/snli_diagnostic_results.py",
    "experiments/baseline_intern.py",
    "experiments/baseline_kev.py",
    "experiments/baseline_runner_core.py",
    "experiments/real_pilot_baseline_core.py",
    "experiments/real_pilot_contracts.py",
    "experiments/mixture_training_baselines.py",
    "experiments/mixture_training_analysis.py",
    "experiments/mixture_training_calibration.py",
    "experiments/mixture_training_statistics.py",
    "experiments/analyze_mixture_training.py",
    "src/reflex_decisions/__init__.py",
    "src/reflex_decisions/data.py",
    "src/reflex_decisions/broader_data.py",
    "src/reflex_decisions/calibration.py",
    "src/reflex_decisions/evaluation.py",
    "src/reflex_decisions/pilot_data.py",
    "src/reflex_decisions/pilot_data_sources.py",
    "src/reflex_decisions/rendering.py",
    "src/reflex_decisions/schema.py",
    "src/reflex_decisions/scoring.py",
    "src/reflex_decisions/smoke.py",
    "src/reflex_decisions/snli_diagnostic.py",
    "src/reflex_decisions/snli_training_data.py",
    "src/reflex_decisions/snli_training_source.py",
    "src/reflex_decisions/synthetic_data.py",
    "experiments/prepare_snli_training.py",
    PROTOCOL_PATH,
)

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json(value: object) -> bytes:
    """Serialize strict canonical UTF-8 JSON."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("value is not strict JSON") from exc
    return text.encode("utf-8")


def json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(value: bytes | str) -> Any:
    """Decode artifact JSON while rejecting duplicate keys and NaN/Infinity."""

    try:
        source = value.decode("utf-8") if isinstance(value, bytes) else value
        if not isinstance(source, str):
            raise TypeError("JSON input must be bytes or text")
        return json.loads(
            source,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("artifact is not strict JSON") from exc


def normalize_json_object(value: object, label: str) -> dict[str, Any]:
    """Return a built-in dict after enforcing the shared JSON serialization rules."""

    if isinstance(value, (bytes, str)):
        normalized = strict_json_loads(value)
    else:
        normalized = strict_json_loads(canonical_json(value))
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUID")
    return value


def source_fingerprints(root: str | Path | None = None) -> dict[str, str]:
    """Hash allowlisted local Python files and the frozen active protocol."""

    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        canonical_root = project_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("project root is unavailable") from exc
    hashes: dict[str, str] = {}
    for relative in SOURCE_FINGERPRINT_PATHS:
        path = project_root / relative
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(canonical_root)
            if not resolved.is_file():
                raise ValueError(f"source fingerprint path is not a file: {relative}")
            hashes[relative] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"required source fingerprint file is unavailable: {relative}"
            ) from exc
    for relative, expected in SNLI_SOURCE_FILE_SHA256.items():
        if hashes.get(relative) != expected:
            raise ValueError(f"SNLI candidate source SHA-256 mismatch: {relative}")
    return hashes


def verify_protocol(path: str | Path = PROTOCOL_PATH) -> str:
    """Verify and return the frozen natural-reasoning protocol digest."""

    protocol_path = Path(path)
    try:
        digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("mixture training protocol is unavailable") from exc
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("mixture training protocol SHA-256 mismatch")
    return digest


def validate_safe_run_id(value: object, label: str = "run_id") -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a safe run ID")
    try:
        return training_rehearsal_core.validate_run_id(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a safe run ID") from exc


def validate_pins(value: object) -> dict[str, object]:
    pins = normalize_json_object(value, "pins")
    if set(pins) != {"data_file_sha256", "protocol_sha256", "source_file_sha256"}:
        raise ValueError("pins have an unexpected schema")
    data_hashes = pins["data_file_sha256"]
    source_hashes = pins["source_file_sha256"]
    if not isinstance(data_hashes, dict) or set(data_hashes) != set(DATA_FILE_SHA256):
        raise ValueError("data file fingerprints do not match the exact allowlist")
    if data_hashes != DATA_FILE_SHA256:
        raise ValueError("data file fingerprints differ from frozen bytes")
    protocol_hash = validate_sha256(pins["protocol_sha256"], "protocol_sha256")
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("protocol fingerprint differs from the frozen protocol")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_FINGERPRINT_PATHS):
        raise ValueError("source fingerprints do not match the exact allowlist")
    for path, digest in source_hashes.items():
        validate_sha256(digest, f"source fingerprint for {path}")
    for path, expected in SNLI_SOURCE_FILE_SHA256.items():
        if source_hashes.get(path) != expected:
            raise ValueError(f"SNLI candidate source fingerprint differs: {path}")
    if source_hashes.get(PROTOCOL_PATH) != protocol_hash:
        raise ValueError("protocol and source fingerprints do not match")
    return pins


def make_pins(root: str | Path | None = None) -> dict[str, object]:
    return {
        "data_file_sha256": dict(DATA_FILE_SHA256),
        "protocol_sha256": verify_protocol(
            Path(root or Path(__file__).resolve().parents[1]) / PROTOCOL_PATH
        ),
        "source_file_sha256": source_fingerprints(root),
    }


def validate_runtime_versions(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or dict(value) != RUNTIME_VERSION_PINS:
        raise ValueError("runtime package versions differ from the frozen nine-package pins")
    return dict(RUNTIME_VERSION_PINS)
