"""Fixed runtime settings and accounting for the runtime-rule study."""

from __future__ import annotations

from collections.abc import Mapping

from experiments import (
    adapter_transfer_contracts,
    mixture_training_contracts,
    runtime_rule_study_inputs,
)
from experiments import (
    runtime_rule_study_data as study,
)
from experiments.mixture_training_contracts import canonical_json as canonical_json
from experiments.mixture_training_contracts import json_sha256 as json_sha256
from experiments.mixture_training_contracts import normalize_json_object as normalize_json_object
from experiments.mixture_training_contracts import strict_json_loads as strict_json_loads
from experiments.mixture_training_contracts import (
    validate_runtime_versions as validate_runtime_versions,
)
from experiments.mixture_training_contracts import validate_safe_run_id as validate_safe_run_id
from experiments.mixture_training_contracts import validate_sha256 as validate_sha256
from experiments.mixture_training_contracts import validate_uuid as validate_uuid

SCHEMA_VERSION = 1
EXPERIMENT_ID = "runtime-rules-v1"
ROLE_UNCHANGED = "unchanged"
ROLES = (ROLE_UNCHANGED, *study.ARM_NAMES)

PROTOCOL_PATH = runtime_rule_study_inputs._STUDY_PROTOCOL_PATH
PROTOCOL_SHA256 = runtime_rule_study_inputs._STUDY_PROTOCOL_SHA256
MODEL_ID = adapter_transfer_contracts.MODEL_ID
MODEL_REVISION = adapter_transfer_contracts.MODEL_REVISION
PROFILE = mixture_training_contracts.PROFILE
WORKSPACE = mixture_training_contracts.WORKSPACE
RUNTIME_VERSION_PINS = dict(mixture_training_contracts.RUNTIME_VERSION_PINS)
SELECTED_TENSOR_SHA256 = "b9ade96b9f6077934985a4b261a6f7400a1e210003b02094b492f36c8a844324"
TRAINING_SEED = 20261009
TRAINING_UPDATES = 336
MICROBATCHES_PER_UPDATE = 4
UNSCORED_SAVE_UPDATE = 168
FINAL_SAVE_UPDATE = 336
BASE_DTYPE = "bfloat16"
LORA_DTYPE = "float32"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LEARNING_RATE = 5e-5
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPSILON = 1e-8
WEIGHT_DECAY = 0.0
MAX_GRADIENT_NORM = 1.0
MAX_INPUT_TOKENS = 2_048
RELOAD_PARITY_COUNT = 32
MAX_CANDIDATE_SCORE_DIFFERENCE = 1e-3
MAX_TOTAL_FORWARDS = 10_808
MAX_TOTAL_INPUT_TOKENS = MAX_TOTAL_FORWARDS * MAX_INPUT_TOKENS
MODAL_VOLUME = "reflex-rehearsal-artifacts"
_COUNT_CATEGORIES = (
    "training_continued_practice",
    "training_runtime_mix",
    "evaluation_both_arms",
    "unchanged_adapter",
    "reload_both_arms",
)
_EXPECTED_FORWARD_CATEGORY_COUNTS = {
    "training_continued_practice": study.TRAINING_PRESENTATIONS,
    "training_runtime_mix": study.TRAINING_PRESENTATIONS,
    "evaluation_both_arms": 2 * (3_782 + study.NEW_PANEL_PRESENTATIONS),
    "unchanged_adapter": study.NEW_PANEL_PRESENTATIONS,
    "reload_both_arms": 2 * study.RELOAD_PER_FAMILY * 2,
}
_EXPECTED_FORWARD_COUNTS = {
    **_EXPECTED_FORWARD_CATEGORY_COUNTS,
    "total": sum(_EXPECTED_FORWARD_CATEGORY_COUNTS.values()),
}


def _validate_role(role: object) -> str:
    if not isinstance(role, str) or role not in ROLES:
        raise ValueError("role must be unchanged, continued_practice, or runtime_mix")
    return role


def _strict_integer_map(
    value: object, expected: Mapping[str, int], *, label: str, exact_values: bool
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{label} categories do not match the fixed study plan")
    normalized: dict[str, int] = {}
    for name, limit in expected.items():
        count = value[name]
        if type(count) is not int:
            raise ValueError(f"{label} values must be strict integers")
        if count < 1 and not exact_values:
            raise ValueError(f"{label} totals must be positive")
        if exact_values and count != limit:
            raise ValueError(f"{label} differ from the fixed study plan")
        normalized[name] = count
    return normalized


def expected_forward_counts(role: object) -> dict[str, int]:
    """Return the fixed forward counts for one worker role."""

    checked_role = _validate_role(role)
    if checked_role == ROLE_UNCHANGED:
        return {"training": 0, "final_evaluation": 164, "reload_parity": 0, "total": 164}
    return {"training": 1_344, "final_evaluation": 3_946, "reload_parity": 32, "total": 5_322}


def expected_token_counts(role: object, compilation: object) -> dict[str, int]:
    """Account one worker's input tokens from an approved aggregate compilation."""

    checked_role = _validate_role(role)
    compiled = normalize_json_object(compilation, "compilation")
    digest = validate_sha256(compiled.pop("compilation_sha256", None), "compilation SHA-256")
    if digest != json_sha256(compiled):
        raise ValueError("compilation SHA-256 does not match its canonical content")

    training = compiled.get("training")
    if not isinstance(training, Mapping) or set(training) != set(study.ARM_NAMES):
        raise ValueError("compilation training roles do not match the two fixed arms")
    counts = _strict_integer_map(
        compiled.get("counts"), _EXPECTED_FORWARD_COUNTS, label="forward count", exact_values=True
    )
    token_counts = _strict_integer_map(
        compiled.get("input_token_counts"),
        {name: 1 for name in (*_COUNT_CATEGORIES, "total")},
        label="input token count",
        exact_values=False,
    )
    maximum = compiled.get("max_input_tokens")
    if type(maximum) is not int or not 1 <= maximum <= MAX_INPUT_TOKENS:
        raise ValueError("maximum input token count must be within the fixed input limit")

    for name in _COUNT_CATEGORIES:
        count, tokens = counts[name], token_counts[name]
        if not count <= tokens <= count * MAX_INPUT_TOKENS:
            raise ValueError(f"{name} token total is outside the per-forward input bounds")
    if token_counts["total"] != sum(token_counts[name] for name in _COUNT_CATEGORIES):
        raise ValueError("total input tokens do not equal the category sum")
    if not counts["total"] <= token_counts["total"] <= MAX_TOTAL_INPUT_TOKENS:
        raise ValueError("total input tokens are outside the fixed study budget")

    if token_counts["evaluation_both_arms"] % 2 or token_counts["reload_both_arms"] % 2:
        raise ValueError("paired evaluation and reload token totals must divide evenly by arm")

    if checked_role == ROLE_UNCHANGED:
        return {
            "training": 0,
            "final_evaluation": token_counts["unchanged_adapter"],
            "reload_parity": 0,
            "total": token_counts["unchanged_adapter"],
        }

    training_tokens = token_counts[f"training_{checked_role}"]
    final_tokens = token_counts["evaluation_both_arms"] // 2
    reload_per_arm = token_counts["reload_both_arms"] // 2
    return {
        "training": training_tokens,
        "final_evaluation": final_tokens,
        "reload_parity": reload_per_arm,
        "total": training_tokens + final_tokens + reload_per_arm,
    }


def optimizer_settings() -> dict[str, object]:
    """Return a fresh JSON-safe AdamW configuration for the two training arms."""

    return {
        "name": "AdamW",
        "learning_rate": LEARNING_RATE,
        "betas": list(ADAMW_BETAS),
        "epsilon": ADAMW_EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "max_gradient_norm": MAX_GRADIENT_NORM,
    }


def worker_settings(role: object) -> dict[str, object]:
    """Return a fresh JSON-safe fixed worker contract, without launching compute."""

    checked_role = _validate_role(role)
    return {
        "gpu": "A10",
        "timeout_seconds": 900 if checked_role == ROLE_UNCHANGED else 3_600,
        "startup_timeout_seconds": 300,
        "cpu": 2,
        "memory_mib": 16_384,
        "min_containers": 0,
        "buffer_containers": 0,
        "retries": 0,
        "single_use": True,
        "scaledown_window_seconds": 2,
    }
