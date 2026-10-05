"""Pinned constants and selection checks for the adapter-transfer evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from experiments.mixture_training_contracts import (
    SOURCE_FINGERPRINT_PATHS as MIXTURE_SOURCE_FINGERPRINT_PATHS,
)
from experiments.mixture_training_contracts import (
    json_sha256,
    strict_json_loads,
    validate_sha256,
)

SCHEMA_VERSION = 1
EXPERIMENT_ID = "adapter-transfer-v1"
SELECTION_PATH = "data/evaluations/adapter-transfer-v1-selection.json"
EXPECTED_SELECTION_SHA256: str | None = (
    "5b274b9cf947b497dee5f2c05547d7fe7d4cae2dcd546a8245180406885569e5"
)
EXPECTED_CANONICAL_SELECTION_SHA256 = (
    "9570ebef89d3a70b6aca19cd4c4b2fec6c80a395c4e3b288701dd8f1e83b3a07"
)

PROTOCOL_PATH = "docs/adapter-transfer-protocol.md"
EXPECTED_PROTOCOL_SHA256 = "d2cf0191da9ee236dc3fe438fedd0461a08618357d69aecb8aa870e7dbae93f1"

COPA_RECORDS_PATH = "data/processed/copa-dev-pilot-v1.jsonl"
COPA_MANIFEST_PATH = "data/baselines/copa-dev-manifest.json"
BROADER_RECORDS_PATH = "data/processed/broader-dev-pilot-v1.jsonl"
BROADER_MANIFEST_PATH = "data/baselines/broader-dev-manifest.json"
BROADER_RECIPE_PATH = "data/baselines/broader-dev-recipe.json"

PANEL_FILE_SHA256 = {
    COPA_RECORDS_PATH: "a8daefa5a7300cc6243f3af1208bbd2887d41c306416bb37bf95263748657c5b",
    COPA_MANIFEST_PATH: "4e60e82ea1622d4e069b4244aa596db64cafa03bdb9ec85d3c4d06d2c40e7635",
    BROADER_RECORDS_PATH: "5d09ac1c1df2b7c1849e675179cb57306040f473c0ebad60527f54d186ac6845",
    BROADER_MANIFEST_PATH: "8dbddaec2ceac622ed998cd0c6051472c11bf38d776a895c08ff94ad5968b085",
    BROADER_RECIPE_PATH: "6450653a445cc12b21304c267cdf0c8976009a01c6982fb07751b349cd81f68b",
}
BOOLQ_PANEL_SHA256 = "a68fe0f08059eb1d811340aff4d4fe99a25068d3b106e881d9eaf5cd8abbed0d"

MAX_INPUT_TOKENS = 2048
BASE_PRESENTATION_COUNT = 128
ADAPTER_PRESENTATION_COUNT = 128
PARITY_PRESENTATION_COUNT = 16
MAX_TOTAL_FORWARDS = 272
MAX_TOTAL_INPUT_TOKENS = 557_056
MAX_LOGIT_DELTA = 0.001

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"

_TRANSFER_SOURCE_FINGERPRINT_PATHS = (
    "experiments/adapter_transfer_contracts.py",
    "experiments/adapter_transfer_data.py",
    "experiments/adapter_transfer_core.py",
    "experiments/adapter_transfer_results.py",
    "experiments/adapter_transfer_runtime.py",
    "experiments/modal_adapter_transfer.py",
    "experiments/analyze_adapter_transfer.py",
    "experiments/baseline_runner_core.py",
    "src/reflex_decisions/baseline_data.py",
    "src/reflex_decisions/broader_data.py",
)
SOURCE_FINGERPRINT_PATHS = tuple(
    dict.fromkeys(
        (*MIXTURE_SOURCE_FINGERPRINT_PATHS, *_TRANSFER_SOURCE_FINGERPRINT_PATHS, PROTOCOL_PATH)
    )
)

_SELECTION_FIELDS = {
    "selection_id",
    "training_run_id",
    "arm",
    "update",
    "training_receipt_sha256",
    "analysis_sha256",
    "agreement_sha256",
    "storage_verification_sha256",
    "snapshot",
}
_SNAPSHOT_FIELDS = {"update", "path", "files_sha256"}
_SNAPSHOT_FILE_NAMES = {"adapter_config.json", "adapter_model.safetensors"}
_SELECTED_CANDIDATE = ("natural-reasoning-2026-10-04-r1", "snli_mix", 378)
_SELECTED_SNAPSHOT_PATH = "/artifacts/runs/natural-reasoning-2026-10-04-r1-snli/adapter-update-378"


def require_selection_pin() -> str:
    """Fail closed until root pins the exact reviewed selection artifact."""

    if EXPECTED_SELECTION_SHA256 is None:
        raise ValueError("adapter-transfer selection SHA-256 has not been pinned")
    return validate_sha256(EXPECTED_SELECTION_SHA256, "selection SHA-256")


def load_selection(root: str | Path) -> tuple[dict[str, object], str]:
    """Read and hash the reviewed selection artifact inside the project root."""

    expected = require_selection_pin()
    project_root = Path(root)
    try:
        canonical_root = project_root.resolve(strict=True)
        path = project_root / SELECTION_PATH
        resolved = path.resolve(strict=True)
        resolved.relative_to(canonical_root)
        if not resolved.is_file():
            raise ValueError("selection path is not a file")
        raw = resolved.read_bytes()
    except (OSError, ValueError) as exc:
        raise ValueError("pinned adapter-transfer selection is unavailable") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError("selection SHA-256 mismatch")
    return validate_selection(strict_json_loads(raw)), digest


def validate_selection(value: object) -> dict[str, object]:
    """Validate the strict, pre-scoring adapter selection record."""

    from experiments.mixture_training_contracts import normalize_json_object

    selection = normalize_json_object(value, "selection")
    if set(selection) != _SELECTION_FIELDS:
        raise ValueError("selection has an unexpected schema")
    if selection["selection_id"] != "adapter-transfer-v1":
        raise ValueError("selection ID is unsupported")
    training_run_id, arm, update = (
        selection["training_run_id"],
        selection["arm"],
        selection["update"],
    )
    if (training_run_id, arm, update) != _SELECTED_CANDIDATE:
        raise ValueError("selection does not name the pinned final SNLI adapter candidate")
    if type(update) is not int:
        raise ValueError("selection update must be an integer")
    for name in (
        "training_receipt_sha256",
        "analysis_sha256",
        "agreement_sha256",
        "storage_verification_sha256",
    ):
        validate_sha256(selection[name], f"selection {name}")
    snapshot = normalize_json_object(selection["snapshot"], "selection snapshot")
    if set(snapshot) != _SNAPSHOT_FIELDS:
        raise ValueError("selection snapshot has an unexpected schema")
    if type(snapshot["update"]) is not int or snapshot["update"] != update:
        raise ValueError("selection snapshot update differs from the selected final update")
    path = snapshot["path"]
    expected_suffix = f"/adapter-update-{update:03d}"
    if (
        not isinstance(path, str)
        or not path.startswith("/artifacts/runs/")
        or ".." in path.split("/")
        or not path.endswith(expected_suffix)
    ):
        raise ValueError("selection snapshot path is not a safe final checkpoint path")
    if path != _SELECTED_SNAPSHOT_PATH:
        raise ValueError("selection snapshot path differs from the pinned final SNLI checkpoint")
    files = snapshot["files_sha256"]
    if not isinstance(files, Mapping) or not _SNAPSHOT_FILE_NAMES.issubset(files):
        raise ValueError("selection snapshot must pin both adapter files")
    if set(files) - (_SNAPSHOT_FILE_NAMES | {"README.md"}):
        raise ValueError("selection snapshot contains an unexpected saved file")
    normalized_files = {
        name: validate_sha256(files[name], f"selection snapshot {name} SHA-256")
        for name in sorted(files)
    }
    normalized = {
        **selection,
        "snapshot": {
            "update": update,
            "path": path,
            "files_sha256": normalized_files,
        },
    }
    if json_sha256(normalized) != EXPECTED_CANONICAL_SELECTION_SHA256:
        raise ValueError("canonical selection SHA-256 does not match its reviewed pin")
    return normalized
