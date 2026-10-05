"""Integration checks for authenticated runtime-rule result envelopes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import pytest

from experiments import (
    adapter_transfer_contracts,
    mixture_training_contracts,
)
from experiments import (
    runtime_rule_study_contracts as contracts,
)
from experiments import (
    runtime_rule_study_progress as progress_api,
)
from experiments import (
    runtime_rule_study_provenance as provenance_api,
)
from experiments import (
    runtime_rule_study_results as results_api,
)
from tests import test_runtime_rule_study_payloads as payload_builders
from tests import test_runtime_rule_study_provenance as provenance_builders
from tests import test_runtime_rule_study_training_evidence as training_builders


@pytest.fixture(scope="module")
def toy_bundle_fixture() -> dict[str, Any]:
    return payload_builders._payload_fixture()


@pytest.fixture(autouse=True)
def _toy_selection_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_hash = mixture_training_contracts.json_sha256(payload_builders._fixture_selection())
    monkeypatch.setattr(
        adapter_transfer_contracts, "EXPECTED_CANONICAL_SELECTION_SHA256", fixture_hash
    )


def _payload(fixture: dict[str, Any], role: str) -> dict[str, object]:
    return payload_builders._build_payload(fixture, role)


def _progress(
    payload: dict[str, object],
    completed: dict[str, int] | None = None,
    *,
    pending_category: str | None = None,
) -> dict[str, object]:
    counts = completed or {}
    specs: dict[str, list[dict[str, object]]] = {
        "training": cast(list[dict[str, object]], payload["training_specs"]),
        "final_evaluation": cast(list[dict[str, object]], payload["evaluation_specs"]),
        "reload_parity": cast(list[dict[str, object]], payload["reload_specs"]),
    }
    ledger = progress_api.ForwardLedger(cast(str, payload["role"]), specs)
    for category, rows in specs.items():
        for spec in rows[: counts.get(category, 0)]:
            ledger.begin(category, spec)
            ledger.complete()
        if category == pending_category:
            ledger.begin(category, rows[counts.get(category, 0)])
    return ledger.snapshot()


def _identity(payload: dict[str, object]) -> dict[str, object]:
    study = cast(dict[str, object], payload["study"])
    selection = cast(dict[str, object], study["selection"])
    return {
        "snapshot": deepcopy(selection["snapshot"]),
        "tensor_sha256": contracts.SELECTED_TENSOR_SHA256,
        "adapter_dtype": "torch.float32",
    }


def _loaded_provenance(
    payload: dict[str, object], *, require_reload: bool = False
) -> dict[str, object]:
    provenance = provenance_api.initial_provenance(payload)
    source_hashes = cast(dict[str, str], payload["source_file_sha256"])
    study = cast(dict[str, object], payload["study"])
    provenance.update(
        {
            "measured_source_file_sha256": deepcopy(source_hashes),
            "runtime_versions": dict(contracts.RUNTIME_VERSION_PINS),
            "base_model": provenance_builders._base_model(payload),
            "cuda_device": "NVIDIA A10",
        }
    )
    if require_reload:
        provenance["reload_tokenizer_file_sha256"] = deepcopy(
            cast(dict[str, object], study["tokenizer_file_sha256"])
        )
    return provenance


def _output_rows(
    presentations: list[dict[str, object]], specs: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for presentation, spec in zip(presentations, specs, strict=True):
        order_ids = cast(list[str], presentation["order_ids"])
        rows.append(
            {
                **{
                    field: deepcopy(presentation[field])
                    for field in (
                        "presentation_id",
                        "record_id",
                        "dataset_id",
                        "source_group_id",
                        "request_hash",
                        "order_index",
                        "order_ids",
                    )
                },
                "candidate_logits": [0.0 for _ in order_ids],
                "winner_option_id": min(order_ids),
                "input_tokens": spec["input_tokens"],
                "prompt_sha256": spec["prompt_sha256"],
            }
        )
    return rows


def _completed_training(payload: dict[str, object]) -> dict[str, object]:
    training = training_builders._complete_training_evidence()
    paths = cast(dict[str, dict[str, object]], training["adapter_paths"])
    for update, snapshot in paths.items():
        snapshot["path"] = f"/artifacts/runs/{payload['run_id']}/adapter-update-{int(update):03d}"
    return training


def _initialized_training() -> dict[str, object]:
    training = training_builders._failed_evidence(0)
    training.update(
        {
            "optimizer": contracts.optimizer_settings(),
            "optimizer_initial_state_entries": 0,
            "adapter_inventory": training_builders._inventory(),
            "base_gradients_none": True,
            "initialization_verified": True,
            "initial_tensor_sha256": contracts.SELECTED_TENSOR_SHA256,
        }
    )
    return training


def _failed_template(payload: dict[str, object], *, phase: str = "preflight") -> dict[str, object]:
    result = results_api.initial_result(payload)
    result.update(
        {
            "status": "failed",
            "phase": phase,
            "failure": {"stage": phase, "type": "SyntheticFailure", "message": "toy failure"},
        }
    )
    return result


def _authenticated(result: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
    return results_api.validate_result(
        result,
        payload=payload,
        expected_payload_sha256=payload["payload_sha256"],
    )


def _passed_unchanged(payload: dict[str, object]) -> dict[str, object]:
    result = results_api.initial_result(payload)
    result.update({"status": "passed", "phase": "completed", "failure": None})
    result["provenance"] = _loaded_provenance(payload)
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(_progress(payload, {"final_evaluation": 164}))
    evidence["selected_adapter_identity"] = _identity(payload)
    evidence["outputs"] = _output_rows(
        cast(list[dict[str, object]], payload["evaluation"]),
        cast(list[dict[str, object]], payload["evaluation_specs"]),
    )
    return result


def _passed_trained(payload: dict[str, object]) -> dict[str, object]:
    result = results_api.initial_result(payload)
    result.update({"status": "passed", "phase": "completed", "failure": None})
    result["provenance"] = _loaded_provenance(payload, require_reload=True)
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(
        _progress(
            payload,
            {"training": 1_344, "final_evaluation": 3_946, "reload_parity": 32},
        )
    )
    evidence["selected_adapter_identity"] = _identity(payload)
    evidence["training"] = _completed_training(payload)
    evaluation = _output_rows(
        cast(list[dict[str, object]], payload["evaluation"]),
        cast(list[dict[str, object]], payload["evaluation_specs"]),
    )
    evidence["outputs"] = evaluation
    evaluation_by_id = {row["presentation_id"]: row for row in evaluation}
    reload_presentations = cast(list[dict[str, object]], payload["reload"])
    reload_specs = cast(list[dict[str, object]], payload["reload_specs"])
    reload_outputs: list[dict[str, object]] = []
    for presentation, spec in zip(reload_presentations, reload_specs, strict=True):
        final = evaluation_by_id[presentation["presentation_id"]]
        reload_outputs.append(
            {
                **{
                    field: deepcopy(presentation[field])
                    for field in (
                        "presentation_id",
                        "record_id",
                        "dataset_id",
                        "source_group_id",
                        "request_hash",
                        "order_index",
                        "order_ids",
                    )
                },
                "candidate_logits": deepcopy(final["candidate_logits"]),
                "winner_option_id": final["winner_option_id"],
                "input_tokens": spec["input_tokens"],
                "prompt_sha256": spec["prompt_sha256"],
            }
        )
    evidence["reload"] = {
        "outputs": reload_outputs,
        "tensor_equal": True,
        "tensor_sha256": "d" * 64,
        "winner_match_count": 32,
        "max_candidate_logit_difference": 0.0,
    }
    return result


def test_initial_result_is_a_mutable_unreturned_template(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "unchanged")

    initial = results_api.initial_result(payload)

    assert initial["status"] == "running"
    assert initial["phase"] == "preflight"
    assert initial["failure"] is None
    initial_evidence = cast(dict[str, object], initial["evidence"])
    initial_counts = cast(dict[str, object], initial_evidence["completed_forward_counts"])
    assert initial_counts["total"] == 0
    with pytest.raises(ValueError, match="status must be passed or failed"):
        _authenticated(initial, payload)


def test_unchanged_pass_authenticates_full_payload_and_detaches(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "unchanged")
    raw = _passed_unchanged(payload)

    checked = _authenticated(raw, payload)

    checked_evidence = cast(dict[str, object], checked["evidence"])
    checked_outputs = cast(list[dict[str, object]], checked_evidence["outputs"])
    checked_logits = cast(list[float], checked_outputs[0]["candidate_logits"])
    checked_logits[0] = 91.0
    raw_evidence = cast(dict[str, object], raw["evidence"])
    raw_outputs = cast(list[dict[str, object]], raw_evidence["outputs"])
    raw_logits = cast(list[float], raw_outputs[0]["candidate_logits"])
    assert raw_logits[0] == 0.0
    assert checked["phase"] == "completed"


def test_trained_pass_requires_successful_training_evaluation_and_reload(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")

    checked = _authenticated(_passed_trained(payload), payload)

    checked_evidence = cast(dict[str, object], checked["evidence"])
    assert checked_evidence["completed_forward_counts"] == {
        "training": 1_344,
        "final_evaluation": 3_946,
        "reload_parity": 32,
        "total": 5_322,
    }
    reload_evidence = cast(dict[str, object], checked_evidence["reload"])
    assert reload_evidence["max_candidate_logit_difference"] == 0.0


def test_failure_before_load_preserves_initial_evidence(toy_bundle_fixture: dict[str, Any]) -> None:
    payload = _payload(toy_bundle_fixture, "continued_practice")

    checked = _authenticated(_failed_template(payload), payload)

    provenance = cast(dict[str, object], checked["provenance"])
    evidence = cast(dict[str, object], checked["evidence"])
    training = cast(dict[str, object], evidence["training"])
    reload_evidence = cast(dict[str, object], evidence["reload"])
    assert provenance["base_model"] is None
    assert training["optimizer_updates_completed"] == 0
    assert reload_evidence["outputs"] == []


@pytest.mark.parametrize(
    ("role", "category", "phase", "completed"),
    [
        ("runtime_mix", "training", "training", {}),
        ("runtime_mix", "final_evaluation", "final_evaluation", {"training": 1_344}),
        (
            "runtime_mix",
            "reload_parity",
            "reload",
            {"training": 1_344, "final_evaluation": 3_946},
        ),
    ],
)
def test_pending_model_calls_require_their_full_prerequisites(
    toy_bundle_fixture: dict[str, Any],
    role: str,
    category: str,
    phase: str,
    completed: dict[str, int],
) -> None:
    payload = _payload(toy_bundle_fixture, role)
    result = _failed_template(payload, phase=phase)
    result["provenance"] = _loaded_provenance(payload, require_reload=category == "reload_parity")
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(_progress(payload, completed, pending_category=category))
    evidence["selected_adapter_identity"] = _identity(payload)
    if category == "training":
        evidence["training"] = _initialized_training()
    elif category in {"final_evaluation", "reload_parity"}:
        evidence["training"] = _completed_training(payload)
    if category == "reload_parity":
        evidence["outputs"] = _output_rows(
            cast(list[dict[str, object]], payload["evaluation"]),
            cast(list[dict[str, object]], payload["evaluation_specs"]),
        )

    checked = _authenticated(result, payload)

    evidence = cast(dict[str, object], checked["evidence"])
    pending = cast(dict[str, object], evidence["pending_forward"])
    assert pending["category"] == category


def test_failed_final_evaluation_can_preserve_only_a_terminal_scored_row_gap(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "unchanged")
    result = _failed_template(payload, phase="final_evaluation")
    result["provenance"] = _loaded_provenance(payload)
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(_progress(payload, {"final_evaluation": 2}))
    evidence["selected_adapter_identity"] = _identity(payload)
    evaluations = cast(list[dict[str, object]], payload["evaluation"])
    evaluation_specs = cast(list[dict[str, object]], payload["evaluation_specs"])
    evidence["outputs"] = _output_rows(
        evaluations[:1],
        evaluation_specs[:1],
    )

    checked = _authenticated(result, payload)

    checked_evidence = cast(dict[str, object], checked["evidence"])
    checked_outputs = cast(list[object], checked_evidence["outputs"])
    assert len(checked_outputs) == 1
    evidence["completed_forward_counts"] = _progress(
        payload, {"final_evaluation": 2}, pending_category="final_evaluation"
    )["completed_forward_counts"]
    evidence["completed_input_token_counts"] = _progress(
        payload, {"final_evaluation": 2}, pending_category="final_evaluation"
    )["completed_input_token_counts"]
    evidence["pending_forward"] = _progress(
        payload, {"final_evaluation": 2}, pending_category="final_evaluation"
    )["pending_forward"]
    with pytest.raises(ValueError, match="terminal evaluation output gap"):
        _authenticated(result, payload)


def test_initial_reload_evidence_allows_partial_evaluation_but_observations_do_not(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")
    result = _failed_template(payload, phase="final_evaluation")
    result["provenance"] = _loaded_provenance(payload)
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(_progress(payload, {"training": 1_344, "final_evaluation": 1}))
    evidence["selected_adapter_identity"] = _identity(payload)
    evidence["training"] = _completed_training(payload)
    evaluations = cast(list[dict[str, object]], payload["evaluation"])
    evaluation_specs = cast(list[dict[str, object]], payload["evaluation_specs"])
    evidence["outputs"] = _output_rows(evaluations[:1], evaluation_specs[:1])

    checked = _authenticated(result, payload)

    checked_evidence = cast(dict[str, object], checked["evidence"])
    assert len(cast(list[object], checked_evidence["outputs"])) == 1
    result["provenance"] = _loaded_provenance(payload, require_reload=True)
    reload_evidence = cast(dict[str, object], evidence["reload"])
    reload_presentations = cast(list[dict[str, object]], payload["reload"])
    reload_specs = cast(list[dict[str, object]], payload["reload_specs"])
    evidence["reload"] = {
        **reload_evidence,
        "outputs": _output_rows(
            reload_presentations[:1],
            reload_specs[:1],
        ),
    }

    with pytest.raises(ValueError, match="reload work requires complete final outputs"):
        _authenticated(result, payload)


def test_training_loss_gap_must_stop_before_any_pending_or_later_call(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")
    result = _failed_template(payload, phase="training")
    result["provenance"] = _loaded_provenance(payload)
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(_progress(payload, {"training": 4}))
    evidence["selected_adapter_identity"] = _identity(payload)
    training = training_builders._failed_evidence(4, updates=1)
    training["training_step_losses"] = []
    evidence["training"] = training

    assert _authenticated(result, payload)["status"] == "failed"
    evidence.update(_progress(payload, {"training": 4}, pending_category="training"))
    with pytest.raises(ValueError, match="terminal training loss gap"):
        _authenticated(result, payload)


def test_pending_first_training_call_requires_verified_initialization_and_optimizer(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")
    result = _failed_template(payload, phase="training")
    result["provenance"] = _loaded_provenance(payload)
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(_progress(payload, pending_category="training"))
    evidence["selected_adapter_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="pending training forward requires"):
        _authenticated(result, payload)


def test_next_training_batch_cannot_start_before_optimizer_update_is_recorded(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")
    result = _failed_template(payload, phase="training")
    result["provenance"] = _loaded_provenance(payload)
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(_progress(payload, {"training": 4}, pending_category="training"))
    evidence["selected_adapter_identity"] = _identity(payload)
    evidence["training"] = training_builders._failed_evidence(4, updates=0)

    with pytest.raises(ValueError, match="pending batch boundary"):
        _authenticated(result, payload)


def test_midpoint_snapshot_is_required_before_the_next_update(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")
    result = _failed_template(payload, phase="training")
    result["provenance"] = _loaded_provenance(payload)
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(
        _progress(
            payload, {"training": contracts.UNSCORED_SAVE_UPDATE * 4}, pending_category="training"
        )
    )
    evidence["selected_adapter_identity"] = _identity(payload)
    evidence["training"] = training_builders._failed_evidence(
        contracts.UNSCORED_SAVE_UPDATE * 4, updates=contracts.UNSCORED_SAVE_UPDATE
    )

    with pytest.raises(ValueError, match="midpoint adapter snapshot"):
        _authenticated(result, payload)


def test_advanced_phase_without_loaded_prerequisites_is_rejected(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")
    result = _failed_template(payload, phase="training")

    with pytest.raises(ValueError, match="required when the model is loaded"):
        _authenticated(result, payload)


def test_payload_digest_and_result_identity_cannot_drift(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "unchanged")
    result = _failed_template(payload)
    result["run_id"] = "different-run"

    with pytest.raises(ValueError, match="result run_id differs"):
        _authenticated(result, payload)

    result["run_id"] = payload["run_id"]
    result["source_commit"] = "0" * 40
    with pytest.raises(ValueError, match="result source_commit differs"):
        _authenticated(result, payload)
    with pytest.raises(ValueError, match="trusted expected digest"):
        results_api.validate_result(result, payload=payload, expected_payload_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda result: result.__setitem__("schema_version", True), "schema_version"),
        (
            lambda result: result["evidence"]["completed_forward_counts"].__setitem__(
                "total", False
            ),
            "strict nonnegative integers",
        ),
        (
            lambda result: result["failure"].__setitem__("message", "x" * 2_001),
            "failure message",
        ),
    ],
)
def test_numeric_aliases_and_oversized_failure_text_are_rejected(
    toy_bundle_fixture: dict[str, Any], mutator: Any, message: str
) -> None:
    payload = _payload(toy_bundle_fixture, "unchanged")
    result = _failed_template(payload)
    mutator(result)

    with pytest.raises(ValueError, match=message):
        _authenticated(result, payload)


def test_reload_failure_keeps_adverse_tensor_winner_and_logit_observations(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")
    result = _passed_trained(payload)
    result.update(
        {
            "status": "failed",
            "phase": "reload",
            "failure": {"stage": "reload", "type": "ParityError", "message": "mismatch"},
        }
    )
    evidence = cast(dict[str, object], result["evidence"])
    evidence.update(
        _progress(
            payload,
            {"training": 1_344, "final_evaluation": 3_946, "reload_parity": 1},
        )
    )
    reload_presentations = cast(list[dict[str, object]], payload["reload"])
    reload_specs = cast(list[dict[str, object]], payload["reload_specs"])
    adverse = _output_rows(reload_presentations[:1], reload_specs[:1])[0]
    order_ids = cast(list[str], adverse["order_ids"])
    adverse["candidate_logits"] = [
        1.0 if option_id == max(order_ids) else 0.0 for option_id in order_ids
    ]
    adverse["winner_option_id"] = max(order_ids)
    evidence["reload"] = {
        "outputs": [adverse],
        "tensor_equal": False,
        "tensor_sha256": "f" * 64,
        "winner_match_count": 0,
        "max_candidate_logit_difference": 1.0,
    }

    checked = _authenticated(result, payload)

    checked_evidence = cast(dict[str, object], checked["evidence"])
    checked_reload = cast(dict[str, object], checked_evidence["reload"])
    assert checked_reload["tensor_equal"] is False
    assert checked_reload["tensor_sha256"] == "f" * 64
    assert checked_reload["winner_match_count"] == 0
    assert checked_reload["max_candidate_logit_difference"] == 1.0


def test_nested_non_string_keys_are_rejected_without_mutating_result(
    toy_bundle_fixture: dict[str, Any],
) -> None:
    payload = _payload(toy_bundle_fixture, "runtime_mix")
    result = _passed_trained(payload)
    evidence = cast(dict[str, object], result["evidence"])
    training = cast(dict[str, object], evidence["training"])
    adapter_paths = cast(dict[str, object], training["adapter_paths"])
    training["adapter_paths"] = {int(key): value for key, value in adapter_paths.items()}
    original = deepcopy(result)

    with pytest.raises(ValueError, match="string keys"):
        _authenticated(result, payload)

    assert result == original
