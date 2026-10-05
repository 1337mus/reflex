"""Tests for fixed continuation-training evidence validation."""

from __future__ import annotations

import pytest

from experiments import runtime_rule_study_contracts as contracts
from experiments.runtime_rule_study_training_evidence import (
    initial_training_evidence,
    validate_training_evidence,
)

_RUN_ID = "training-evidence-run"


def _inventory() -> dict[str, object]:
    module_names = [f"adapter.module_{index:02d}" for index in range(60)]
    tensor_names = [
        f"{module_name}.lora_{factor}.default.weight"
        for module_name in module_names
        for factor in ("A", "B")
    ]
    shapes = {name: [8, 2_099 if index < 96 else 2_100] for index, name in enumerate(tensor_names)}
    return {
        "module_count": 60,
        "module_names": module_names,
        "adapter_tensor_count": 120,
        "adapter_parameter_count": 2_015_232,
        "adapter_tensor_names": tensor_names,
        "adapter_tensor_shapes": shapes,
        "adapter_tensor_dtypes": {name: "torch.float32" for name in tensor_names},
        "trainable_parameter_count": 2_015_232,
        "base_parameters_frozen_bf16": True,
    }


def _snapshot(update: int, *, include_readme: bool = False) -> dict[str, object]:
    files_sha256 = {
        "adapter_config.json": "a" * 64,
        "adapter_model.safetensors": "b" * 64,
    }
    if include_readme:
        files_sha256["README.md"] = "c" * 64
    return {
        "update": update,
        "path": f"/artifacts/runs/{_RUN_ID}/adapter-update-{update:03d}",
        "files_sha256": files_sha256,
    }


def _complete_training_evidence() -> dict[str, object]:
    inventory = _inventory()
    tensor_names = inventory["adapter_tensor_names"]
    assert isinstance(tensor_names, list)
    return {
        "optimizer": contracts.optimizer_settings(),
        "optimizer_initial_state_entries": 0,
        "optimizer_updates_completed": 336,
        "training_step_losses": [
            {
                "update": update,
                "mean_loss": 0.25,
                "lo_ra_b_gradient_l1": 0.5,
                "gradient_norm": 1.25,
            }
            for update in range(1, 337)
        ],
        "adapter_inventory": inventory,
        "base_gradients_none": True,
        "initialization_verified": True,
        "initial_tensor_sha256": contracts.SELECTED_TENSOR_SHA256,
        "adapter_paths": {
            "168": _snapshot(168, include_readme=True),
            "336": _snapshot(336),
        },
        "adapter_update": {
            "changed_tensor_count": 1,
            "changed_tensor_names": [tensor_names[0]],
        },
        "final_tensor_sha256": "d" * 64,
    }


def _losses(count: int) -> list[dict[str, object]]:
    return [
        {
            "update": update,
            "mean_loss": 0.25,
            "lo_ra_b_gradient_l1": 0.5,
            "gradient_norm": 1.25,
        }
        for update in range(1, count + 1)
    ]


def _failed_evidence(completed_forwards: int, updates: int = 0) -> dict[str, object]:
    evidence = initial_training_evidence()
    evidence["optimizer_updates_completed"] = updates
    evidence["training_step_losses"] = _losses(updates)
    if completed_forwards > 0:
        evidence.update(
            {
                "optimizer": contracts.optimizer_settings(),
                "optimizer_initial_state_entries": 0,
                "adapter_inventory": _inventory(),
                "base_gradients_none": False,
                "initialization_verified": True,
                "initial_tensor_sha256": contracts.SELECTED_TENSOR_SHA256,
            }
        )
    return evidence


def test_initial_training_evidence_has_exact_fresh_empty_schema() -> None:
    expected: dict[str, object] = {
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

    first = initial_training_evidence()
    second = initial_training_evidence()

    assert first == expected
    assert second == expected
    assert first is not second
    first_losses = first["training_step_losses"]
    first_paths = first["adapter_paths"]
    assert isinstance(first_losses, list)
    assert isinstance(first_paths, dict)
    first_losses.append({"update": 1})
    first_paths["168"] = {}
    assert second == expected


def test_zero_start_failure_preserves_detached_initial_evidence() -> None:
    evidence = initial_training_evidence()

    validated = validate_training_evidence(
        evidence,
        run_id="training-evidence-run",
        completed_training_forwards=0,
        passed=False,
    )

    assert validated == evidence
    assert validated is not evidence
    losses = validated["training_step_losses"]
    paths = validated["adapter_paths"]
    assert isinstance(losses, list)
    assert isinstance(paths, dict)
    losses.append({"update": 1})
    paths["168"] = {}
    assert evidence == initial_training_evidence()


def test_complete_training_pass_is_valid_and_detached() -> None:
    evidence = _complete_training_evidence()

    validated = validate_training_evidence(
        evidence,
        run_id=_RUN_ID,
        completed_training_forwards=1_344,
        passed=True,
    )

    assert validated == evidence
    assert validated is not evidence
    validated_optimizer = validated["optimizer"]
    validated_losses = validated["training_step_losses"]
    validated_inventory = validated["adapter_inventory"]
    validated_paths = validated["adapter_paths"]
    assert isinstance(validated_optimizer, dict)
    assert isinstance(validated_losses, list)
    assert isinstance(validated_inventory, dict)
    assert isinstance(validated_paths, dict)
    validated_optimizer["name"] = "mutated"
    validated_losses[0]["mean_loss"] = 99.0
    validated_inventory["module_names"].pop()
    validated_paths["168"]["files_sha256"]["README.md"] = "0" * 64
    assert evidence == _complete_training_evidence()


@pytest.mark.parametrize("completed_forwards", range(5))
def test_failed_training_accepts_each_prefix_of_first_microbatch_window(
    completed_forwards: int,
) -> None:
    evidence = _failed_evidence(completed_forwards)

    validated = validate_training_evidence(
        evidence,
        run_id=_RUN_ID,
        completed_training_forwards=completed_forwards,
        passed=False,
    )

    assert validated == evidence


@pytest.mark.parametrize("completed_forwards", range(4, 9))
def test_failed_training_accepts_next_microbatch_window_after_one_update(
    completed_forwards: int,
) -> None:
    evidence = _failed_evidence(completed_forwards, updates=1)

    validated = validate_training_evidence(
        evidence,
        run_id=_RUN_ID,
        completed_training_forwards=completed_forwards,
        passed=False,
    )

    assert validated["optimizer_updates_completed"] == 1
    assert validated["training_step_losses"] == _losses(1)


def test_failed_preflight_may_keep_a_valid_partial_initial_tensor_digest() -> None:
    evidence = initial_training_evidence()
    evidence["initial_tensor_sha256"] = "e" * 64

    validated = validate_training_evidence(
        evidence,
        run_id=_RUN_ID,
        completed_training_forwards=0,
        passed=False,
    )

    assert validated["initial_tensor_sha256"] == "e" * 64


@pytest.mark.parametrize(
    ("completed_forwards", "updates", "message"),
    [
        (3, 1, "inconsistent with optimizer updates"),
        (9, 1, "inconsistent with optimizer updates"),
        (2, 3, "inconsistent with optimizer updates"),
        (1_344, 337, "exceed the fixed"),
    ],
)
def test_failed_training_rejects_forward_and_update_count_drift(
    completed_forwards: int,
    updates: int,
    message: str,
) -> None:
    evidence = _failed_evidence(completed_forwards, updates)

    with pytest.raises(ValueError, match=message):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=completed_forwards,
            passed=False,
        )


def test_any_completed_training_forward_requires_selected_initialization() -> None:
    evidence = initial_training_evidence()

    with pytest.raises(ValueError, match="require verified initialization"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=1,
            passed=False,
        )


def test_failed_training_preserves_base_gradient_and_no_change_evidence() -> None:
    evidence = _failed_evidence(completed_forwards=4, updates=1)
    evidence["adapter_update"] = {"changed_tensor_count": 0, "changed_tensor_names": []}

    validated = validate_training_evidence(
        evidence,
        run_id=_RUN_ID,
        completed_training_forwards=4,
        passed=False,
    )

    assert validated["base_gradients_none"] is False
    assert validated["adapter_update"] == {
        "changed_tensor_count": 0,
        "changed_tensor_names": [],
    }


def test_passed_training_requires_absent_base_gradients() -> None:
    evidence = _complete_training_evidence()
    evidence["base_gradients_none"] = False

    with pytest.raises(ValueError, match="confirm base gradients remained absent"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=1_344,
            passed=True,
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("passed", 1, "passed must be a boolean"),
        ("completed_training_forwards", True, "completed training forwards must be an integer"),
        ("completed_training_forwards", 1_345, "exceed the fixed study count"),
        ("run_id", "../unsafe", "run_id must be a safe run ID"),
    ],
)
def test_rejects_argument_type_aliases_and_invalid_bounds(
    argument: str,
    value: object,
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "run_id": _RUN_ID,
        "completed_training_forwards": 0,
        "passed": False,
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        validate_training_evidence(
            initial_training_evidence(),
            run_id=arguments["run_id"],
            completed_training_forwards=arguments["completed_training_forwards"],
            passed=arguments["passed"],
        )


def test_rejects_extra_top_level_fields_and_non_string_nested_keys() -> None:
    extra_field = initial_training_evidence()
    extra_field["unreviewed"] = True
    with pytest.raises(ValueError, match="unexpected schema"):
        validate_training_evidence(
            extra_field,
            run_id=_RUN_ID,
            completed_training_forwards=0,
            passed=False,
        )

    numeric_key = initial_training_evidence()
    numeric_key["adapter_paths"] = {168: _snapshot(168)}
    with pytest.raises(ValueError, match="object keys must be strings"):
        validate_training_evidence(
            numeric_key,
            run_id=_RUN_ID,
            completed_training_forwards=0,
            passed=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_learning_rate", "differ from the fixed fresh AdamW contract"),
        ("nonzero_state", "must start with zero state entries"),
        ("boolean_state", "optimizer initial state entries must be an integer"),
        ("missing_state", "must appear together"),
        ("missing_optimizer", "must appear together"),
    ],
)
def test_requires_exact_optimizer_settings_and_fresh_state(
    mutation: str,
    message: str,
) -> None:
    evidence = _failed_evidence(completed_forwards=4, updates=1)
    if mutation == "wrong_learning_rate":
        optimizer = contracts.optimizer_settings()
        optimizer["learning_rate"] = 1e-4
        evidence["optimizer"] = optimizer
    elif mutation == "nonzero_state":
        evidence["optimizer_initial_state_entries"] = 1
    elif mutation == "boolean_state":
        evidence["optimizer_initial_state_entries"] = False
    elif mutation == "missing_state":
        evidence["optimizer_initial_state_entries"] = None
    else:
        evidence["optimizer"] = None

    with pytest.raises(ValueError, match=message):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=4,
            passed=False,
        )


def test_verified_initialization_requires_exact_selected_tensor_identity() -> None:
    evidence = _failed_evidence(completed_forwards=1)
    evidence["initial_tensor_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="selected tensor"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=1,
            passed=False,
        )


def test_initialization_marker_rejects_boolean_integer_alias() -> None:
    evidence = initial_training_evidence()
    evidence["initialization_verified"] = 1

    with pytest.raises(ValueError, match="initialization_verified must be a boolean"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=0,
            passed=False,
        )


@pytest.mark.parametrize("loss_count", [0, 1])
def test_failed_training_allows_only_a_complete_loss_prefix_or_one_terminal_gap(
    loss_count: int,
) -> None:
    evidence = _failed_evidence(completed_forwards=4, updates=1)
    evidence["training_step_losses"] = _losses(loss_count)

    validated = validate_training_evidence(
        evidence,
        run_id=_RUN_ID,
        completed_training_forwards=4,
        passed=False,
    )

    assert validated["training_step_losses"] == _losses(loss_count)


def test_terminal_loss_gap_cannot_follow_more_forwards_after_optimizer_return() -> None:
    evidence = _failed_evidence(completed_forwards=5, updates=1)
    evidence["training_step_losses"] = []

    with pytest.raises(
        ValueError, match="terminal loss gap requires exactly four forwards per update"
    ):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=5,
            passed=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("order", "cover updates in order"),
        ("extra_field", "unexpected schema"),
        ("boolean_update", "training loss update must be an integer"),
        ("boolean_loss", "training mean loss must be a finite number"),
        ("negative_loss", "outside its finite range"),
        ("nonfinite_gradient", "not strict JSON"),
    ],
)
def test_rejects_malformed_loss_rows_and_numeric_aliases(
    mutation: str,
    message: str,
) -> None:
    evidence = _failed_evidence(completed_forwards=8, updates=2)
    losses = evidence["training_step_losses"]
    assert isinstance(losses, list)
    first = losses[0]
    assert isinstance(first, dict)
    if mutation == "order":
        losses.reverse()
    elif mutation == "extra_field":
        first["unexpected"] = True
    elif mutation == "boolean_update":
        first["update"] = True
    elif mutation == "boolean_loss":
        first["mean_loss"] = True
    elif mutation == "negative_loss":
        first["mean_loss"] = -0.1
    else:
        first["gradient_norm"] = float("inf")

    with pytest.raises(ValueError, match=message):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=8,
            passed=False,
        )


def test_failed_training_rejects_more_than_one_missing_terminal_loss() -> None:
    evidence = _failed_evidence(completed_forwards=12, updates=3)
    evidence["training_step_losses"] = _losses(1)

    with pytest.raises(ValueError, match="omit only its terminal loss record"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=12,
            passed=False,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("lo_ra_b_gradient_l1", "positive aggregate B-gradient evidence"),
        ("gradient_norm", "positive aggregate gradient norms"),
    ],
)
def test_passed_training_requires_positive_finite_aggregate_gradients(
    field: str,
    message: str,
) -> None:
    evidence = _complete_training_evidence()
    losses = evidence["training_step_losses"]
    assert isinstance(losses, list)
    for row in losses:
        assert isinstance(row, dict)
        row[field] = 0.0

    with pytest.raises(ValueError, match=message):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=1_344,
            passed=True,
        )


def test_failed_evidence_does_not_infer_that_an_equal_final_digest_changed() -> None:
    evidence = _failed_evidence(completed_forwards=4, updates=1)
    evidence["final_tensor_sha256"] = contracts.SELECTED_TENSOR_SHA256

    validated = validate_training_evidence(
        evidence,
        run_id=_RUN_ID,
        completed_training_forwards=4,
        passed=False,
    )

    assert validated["final_tensor_sha256"] == contracts.SELECTED_TENSOR_SHA256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("path", "not canonical for this run"),
        ("update", "does not match its evidence slot"),
        ("source_digest", "must be a lowercase SHA-256 digest"),
    ],
)
def test_failed_snapshot_uses_exact_run_path_update_and_valid_file_digests(
    mutation: str,
    message: str,
) -> None:
    evidence = _failed_evidence(completed_forwards=672, updates=168)
    snapshot = _snapshot(168, include_readme=True)
    if mutation == "path":
        snapshot["path"] = "/artifacts/runs/another-run/adapter-update-168"
    elif mutation == "update":
        snapshot["update"] = 167
    else:
        files = snapshot["files_sha256"]
        assert isinstance(files, dict)
        files["README.md"] = "not-a-digest"
    evidence["adapter_paths"] = {"168": snapshot}

    with pytest.raises(ValueError, match=message):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=672,
            passed=False,
        )


def test_final_snapshot_requires_midpoint_and_cannot_exceed_completed_updates() -> None:
    missing_midpoint = _failed_evidence(completed_forwards=1_344, updates=336)
    missing_midpoint["adapter_paths"] = {"336": _snapshot(336)}
    with pytest.raises(ValueError, match="requires the midpoint snapshot"):
        validate_training_evidence(
            missing_midpoint,
            run_id=_RUN_ID,
            completed_training_forwards=1_344,
            passed=False,
        )

    not_yet_completed = _failed_evidence(completed_forwards=668, updates=167)
    not_yet_completed["adapter_paths"] = {"168": _snapshot(168)}
    with pytest.raises(ValueError, match="exceeds completed optimizer updates"):
        validate_training_evidence(
            not_yet_completed,
            run_id=_RUN_ID,
            completed_training_forwards=668,
            passed=False,
        )


def test_adapter_update_requires_inventory_and_known_changed_tensors() -> None:
    no_inventory = initial_training_evidence()
    no_inventory["adapter_update"] = {
        "changed_tensor_count": 1,
        "changed_tensor_names": ["unverified.tensor"],
    }
    with pytest.raises(ValueError, match="requires a valid adapter inventory"):
        validate_training_evidence(
            no_inventory,
            run_id=_RUN_ID,
            completed_training_forwards=0,
            passed=False,
        )

    unknown_tensor = _failed_evidence(completed_forwards=4, updates=1)
    unknown_tensor["adapter_update"] = {
        "changed_tensor_count": 1,
        "changed_tensor_names": ["unverified.tensor"],
    }
    with pytest.raises(ValueError, match="absent from the inventory"):
        validate_training_evidence(
            unknown_tensor,
            run_id=_RUN_ID,
            completed_training_forwards=4,
            passed=False,
        )


@pytest.mark.parametrize("count", [False, 0.0])
def test_no_change_adapter_evidence_requires_strict_failed_integer_zero(count: object) -> None:
    evidence = _failed_evidence(completed_forwards=4, updates=1)
    evidence["adapter_update"] = {"changed_tensor_count": count, "changed_tensor_names": []}

    with pytest.raises(ValueError, match="changed tensor count must be an integer"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=4,
            passed=False,
        )


def test_passed_training_rejects_no_change_and_equal_final_tensor() -> None:
    evidence = _complete_training_evidence()
    evidence["adapter_update"] = {"changed_tensor_count": 0, "changed_tensor_names": []}
    with pytest.raises(ValueError, match="changed tensor count must be an integer"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=1_344,
            passed=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("optimizer_updates_completed", True, "optimizer updates must be an integer"),
        ("base_gradients_none", 1, "base_gradients_none must be a boolean or null"),
        ("initial_tensor_sha256", "not-a-digest", "must be a lowercase SHA-256 digest"),
        ("final_tensor_sha256", "not-a-digest", "must be a lowercase SHA-256 digest"),
    ],
)
def test_rejects_boolean_count_aliases_and_malformed_tensor_digests(
    field: str,
    value: object,
    message: str,
) -> None:
    evidence = initial_training_evidence()
    evidence[field] = value

    with pytest.raises(ValueError, match=message):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=0,
            passed=False,
        )


@pytest.mark.parametrize("field", ["lo_ra_b_gradient_l1", "gradient_norm"])
def test_passed_training_rejects_nonfinite_aggregate_gradient_evidence(field: str) -> None:
    evidence = _complete_training_evidence()
    losses = evidence["training_step_losses"]
    assert isinstance(losses, list)
    for row in losses:
        assert isinstance(row, dict)
        row[field] = 1e308

    with pytest.raises(ValueError, match="outside its finite range"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=1_344,
            passed=True,
        )


def test_passed_training_requires_all_losses_and_a_real_adapter_change() -> None:
    evidence = _complete_training_evidence()
    evidence["training_step_losses"] = _losses(335)
    with pytest.raises(ValueError, match="all 336 loss records"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=1_344,
            passed=True,
        )


def test_adapter_paths_reject_unknown_snapshot_slots() -> None:
    unexpected_slot = _failed_evidence(completed_forwards=672, updates=168)
    unexpected_slot["adapter_paths"] = {"000": _snapshot(0)}
    with pytest.raises(ValueError, match="outside the fixed snapshot schedule"):
        validate_training_evidence(
            unexpected_slot,
            run_id=_RUN_ID,
            completed_training_forwards=672,
            passed=False,
        )


def test_failed_inventory_is_validated_even_before_initialization_is_verified() -> None:
    evidence = initial_training_evidence()
    evidence["adapter_inventory"] = {"module_count": 60}

    with pytest.raises(ValueError, match="adapter inventory has an unexpected schema"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=0,
            passed=False,
        )

    evidence = _complete_training_evidence()
    evidence["final_tensor_sha256"] = contracts.SELECTED_TENSOR_SHA256
    with pytest.raises(ValueError, match="changed final tensor digest"):
        validate_training_evidence(
            evidence,
            run_id=_RUN_ID,
            completed_training_forwards=1_344,
            passed=True,
        )
