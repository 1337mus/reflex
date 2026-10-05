from __future__ import annotations

from copy import deepcopy

import pytest

from experiments import (
    mixture_training_contracts,
    runtime_rule_study_data,
    runtime_rule_study_inputs,
)
from experiments import runtime_rule_study_contracts as contracts

_FORWARD_COUNTS = {
    "training_continued_practice": 1_344,
    "training_runtime_mix": 1_344,
    "evaluation_both_arms": 7_892,
    "unchanged_adapter": 164,
    "reload_both_arms": 64,
    "total": 10_808,
}
_TOKEN_COUNTS = {
    "training_continued_practice": 225_924,
    "training_runtime_mix": 349_296,
    "evaluation_both_arms": 1_843_160,
    "unchanged_adapter": 92_958,
    "reload_both_arms": 32_872,
    "total": 2_544_210,
}


def _seal(compilation: dict[str, object]) -> dict[str, object]:
    compilation["compilation_sha256"] = mixture_training_contracts.json_sha256(compilation)
    return compilation


def _compilation() -> dict[str, object]:
    return _seal(
        {
            "training": {arm: [] for arm in runtime_rule_study_data.ARM_NAMES},
            "counts": dict(_FORWARD_COUNTS),
            "input_token_counts": dict(_TOKEN_COUNTS),
            "max_input_tokens": 695,
        }
    )


def _changed(mutate) -> dict[str, object]:
    compilation = deepcopy(_compilation())
    del compilation["compilation_sha256"]
    mutate(compilation)
    return _seal(compilation)


def test_fixed_protocol_training_optimizer_and_worker_budget() -> None:
    assert contracts.SCHEMA_VERSION == 1
    assert contracts.EXPERIMENT_ID == "runtime-rules-v1"
    assert contracts.ROLES == ("unchanged", *runtime_rule_study_data.ARM_NAMES)
    assert (contracts.PROTOCOL_PATH, contracts.PROTOCOL_SHA256) == (
        runtime_rule_study_inputs._STUDY_PROTOCOL_PATH,
        runtime_rule_study_inputs._STUDY_PROTOCOL_SHA256,
    )
    assert contracts.MODEL_ID == "Qwen/Qwen3.5-0.8B-Base"
    assert contracts.MODEL_REVISION == "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
    assert contracts.PROFILE == "reflex-personal"
    assert contracts.WORKSPACE == "rajath-61258"
    assert contracts.RUNTIME_VERSION_PINS == mixture_training_contracts.RUNTIME_VERSION_PINS
    assert contracts.SELECTED_TENSOR_SHA256 == (
        "b9ade96b9f6077934985a4b261a6f7400a1e210003b02094b492f36c8a844324"
    )
    assert (
        contracts.TRAINING_SEED,
        contracts.TRAINING_UPDATES,
        contracts.MICROBATCHES_PER_UPDATE,
        contracts.UNSCORED_SAVE_UPDATE,
        contracts.FINAL_SAVE_UPDATE,
    ) == (20261009, 336, 4, 168, 336)
    assert (
        contracts.BASE_DTYPE,
        contracts.LORA_DTYPE,
        contracts.LORA_RANK,
        contracts.LORA_ALPHA,
        contracts.LORA_DROPOUT,
    ) == ("bfloat16", "float32", 8, 16, 0.0)

    assert contracts.MAX_INPUT_TOKENS == 2_048
    assert contracts.MAX_TOTAL_FORWARDS == sum(
        contracts.expected_forward_counts(role)["total"] for role in contracts.ROLES
    )
    assert contracts.MAX_TOTAL_INPUT_TOKENS == contracts.MAX_TOTAL_FORWARDS * 2_048

    for role in contracts.ROLES:
        worker = contracts.worker_settings(role)
        assert worker == {
            "gpu": "A10",
            "timeout_seconds": 900 if role == "unchanged" else 3_600,
            "startup_timeout_seconds": 300,
            "cpu": 2,
            "memory_mib": 16_384,
            "min_containers": 0,
            "buffer_containers": 0,
            "retries": 0,
            "single_use": True,
            "scaledown_window_seconds": 2,
        }


def test_optimizer_recipe_is_fresh_adamw_with_the_fixed_study_rate() -> None:
    optimizer = contracts.optimizer_settings()
    assert optimizer == {
        "name": "AdamW",
        "learning_rate": 5e-5,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
    }
    optimizer["betas"][0] = 0.0
    assert contracts.optimizer_settings()["betas"] == [0.9, 0.999]


def test_per_worker_forward_and_token_totals_match_approved_aggregate_metadata() -> None:
    compilation = _compilation()
    assert contracts.expected_forward_counts("unchanged") == {
        "training": 0,
        "final_evaluation": 164,
        "reload_parity": 0,
        "total": 164,
    }
    expected_trained = {
        "training": 1_344,
        "final_evaluation": 3_946,
        "reload_parity": 32,
        "total": 5_322,
    }
    assert contracts.expected_forward_counts("continued_practice") == expected_trained
    assert contracts.expected_forward_counts("runtime_mix") == expected_trained
    assert contracts.expected_token_counts("unchanged", compilation) == {
        "training": 0,
        "final_evaluation": 92_958,
        "reload_parity": 0,
        "total": 92_958,
    }
    assert contracts.expected_token_counts("continued_practice", compilation) == {
        "training": 225_924,
        "final_evaluation": 921_580,
        "reload_parity": 16_436,
        "total": 1_163_940,
    }
    assert contracts.expected_token_counts("runtime_mix", compilation) == {
        "training": 349_296,
        "final_evaluation": 921_580,
        "reload_parity": 16_436,
        "total": 1_287_312,
    }
    assert (
        sum(contracts.expected_token_counts(role, compilation)["total"] for role in contracts.ROLES)
        == _TOKEN_COUNTS["total"]
    )


@pytest.mark.parametrize("role", [None, "", "runtime", "unknown", True])
def test_all_public_settings_apis_reject_unknown_roles(role: object) -> None:
    with pytest.raises(ValueError):
        contracts.expected_forward_counts(role)
    with pytest.raises(ValueError):
        contracts.expected_token_counts(role, _compilation())
    with pytest.raises(ValueError):
        contracts.worker_settings(role)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["training"].pop("runtime_mix"),
        lambda value: value["training"].__setitem__("other", []),
    ],
    ids=["missing-training-arm", "extra-training-arm"],
)
def test_token_accounting_rejects_missing_or_extra_compilation_roles(mutate) -> None:
    with pytest.raises(ValueError):
        contracts.expected_token_counts("continued_practice", _changed(mutate))


def test_token_accounting_rejects_bad_canonical_digest() -> None:
    compilation = _compilation()
    compilation["compilation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="compilation SHA-256"):
        contracts.expected_token_counts("runtime_mix", compilation)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["counts"].pop("training_runtime_mix"),
        lambda value: value["counts"].__setitem__("unexpected", 1),
        lambda value: value["counts"].__setitem__("training_runtime_mix", True),
        lambda value: value["counts"].__setitem__("training_runtime_mix", 1_344.0),
        lambda value: value["input_token_counts"].pop("training_runtime_mix"),
        lambda value: value["input_token_counts"].__setitem__("unexpected", 10),
        lambda value: value["input_token_counts"].__setitem__("training_runtime_mix", True),
        lambda value: value["input_token_counts"].__setitem__("training_runtime_mix", 1.0),
        lambda value: value["input_token_counts"].__setitem__("training_runtime_mix", 0),
        lambda value: value["input_token_counts"].__setitem__("total", 1),
    ],
    ids=[
        "missing-count-category",
        "extra-count-category",
        "boolean-forward-count",
        "float-forward-count",
        "missing-token-category",
        "extra-token-category",
        "boolean-token-total",
        "float-token-total",
        "zero-token-total",
        "token-sum-mismatch",
    ],
)
def test_token_accounting_rejects_noncanonical_categories_and_counts(mutate) -> None:
    with pytest.raises(ValueError):
        contracts.expected_token_counts("runtime_mix", _changed(mutate))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["input_token_counts"].__setitem__("training_continued_practice", 1_343),
        lambda value: value["input_token_counts"].__setitem__(
            "training_runtime_mix", 1_344 * 2_048 + 1
        ),
        lambda value: value["input_token_counts"].__setitem__(
            "evaluation_both_arms", value["input_token_counts"]["evaluation_both_arms"] + 1
        ),
        lambda value: value["input_token_counts"].__setitem__(
            "reload_both_arms", value["input_token_counts"]["reload_both_arms"] + 1
        ),
    ],
    ids=[
        "below-one-token-per-forward",
        "above-token-budget",
        "odd-evaluation-half",
        "odd-reload-half",
    ],
)
def test_token_accounting_rejects_out_of_bounds_and_odd_doubled_totals(mutate) -> None:
    def mutate_and_reconcile_total(value: dict[str, object]) -> None:
        mutate(value)
        categories = value["input_token_counts"]
        categories["total"] = sum(categories[key] for key in categories if key != "total")

    with pytest.raises(ValueError):
        contracts.expected_token_counts("continued_practice", _changed(mutate_and_reconcile_total))


@pytest.mark.parametrize("category", ["evaluation_both_arms", "reload_both_arms"])
def test_unchanged_worker_rejects_odd_paired_token_totals(category: str) -> None:
    def make_odd(compilation: dict[str, object]) -> None:
        token_counts = compilation["input_token_counts"]
        token_counts[category] += 1
        token_counts["total"] += 1

    with pytest.raises(
        ValueError,
        match="paired evaluation and reload token totals must divide evenly by arm",
    ):
        contracts.expected_token_counts("unchanged", _changed(make_odd))


def test_returned_settings_and_accounting_objects_are_isolated() -> None:
    optimizer = contracts.optimizer_settings()
    optimizer["betas"][0] = 0.0
    worker = contracts.worker_settings("runtime_mix")
    worker["retries"] = 10
    forward_counts = contracts.expected_forward_counts("runtime_mix")
    forward_counts["total"] = -1
    token_counts = contracts.expected_token_counts("runtime_mix", _compilation())
    token_counts["total"] = -1

    assert contracts.optimizer_settings()["betas"] == [0.9, 0.999]
    assert contracts.worker_settings("runtime_mix")["retries"] == 0
    assert contracts.expected_forward_counts("runtime_mix")["total"] == 5_322
    assert contracts.expected_token_counts("runtime_mix", _compilation())["total"] == 1_287_312
