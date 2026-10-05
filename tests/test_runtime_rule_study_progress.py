from __future__ import annotations

from copy import deepcopy

import pytest

from experiments import mixture_training_contracts
from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_progress as progress

_CATEGORIES = ("training", "final_evaluation", "reload_parity")


def _specs(role: str) -> dict[str, list[dict[str, object]]]:
    counts = contracts.expected_forward_counts(role)
    return {
        category: [
            {
                "input_tokens": 1 + (index * 7 + category_index * 3) % 31,
                "fixture_identity": f"{category}:{index}",
                "request_json_sha256": mixture_training_contracts.json_sha256(
                    {"category": category, "index": index}
                ),
            }
            for index in range(counts[category])
        ]
        for category_index, category in enumerate(_CATEGORIES)
    }


def _finished_unchanged() -> tuple[progress.ForwardLedger, dict[str, list[dict[str, object]]]]:
    specs = _specs("unchanged")
    ledger = progress.ForwardLedger("unchanged", specs)
    for spec in specs["final_evaluation"]:
        ledger.begin("final_evaluation", spec)
        ledger.complete()
    return ledger, specs


def _reseal_totals(value: dict[str, object], category: str, count: int, tokens: int) -> None:
    forwards = value["completed_forward_counts"]
    input_tokens = value["completed_input_token_counts"]
    forwards[category] = count
    forwards["total"] = sum(forwards[name] for name in _CATEGORIES)
    input_tokens[category] = tokens
    input_tokens["total"] = sum(input_tokens[name] for name in _CATEGORIES)


def test_initial_ledger_has_empty_exact_accounting_and_detaches_its_plan() -> None:
    specs = _specs("unchanged")
    first_spec = deepcopy(specs["final_evaluation"][0])
    ledger = progress.ForwardLedger("unchanged", specs)
    specs["final_evaluation"][0]["fixture_identity"] = "caller-mutated"

    assert ledger.snapshot() == {
        "completed_forward_counts": {
            "training": 0,
            "final_evaluation": 0,
            "reload_parity": 0,
            "total": 0,
        },
        "completed_input_token_counts": {
            "training": 0,
            "final_evaluation": 0,
            "reload_parity": 0,
            "total": 0,
        },
        "pending_forward": None,
    }
    ledger.begin("final_evaluation", first_spec)
    before = ledger.snapshot()
    detached = ledger.snapshot()
    detached["pending_forward"]["input_tokens"] = 2_048
    detached["completed_forward_counts"]["total"] = 99
    assert ledger.snapshot() == before


def test_unchanged_role_finishes_164_variable_length_evaluations() -> None:
    ledger, specs = _finished_unchanged()
    expected_tokens = sum(spec["input_tokens"] for spec in specs["final_evaluation"])

    ledger.require_complete()
    assert ledger.snapshot() == {
        "completed_forward_counts": {
            "training": 0,
            "final_evaluation": 164,
            "reload_parity": 0,
            "total": 164,
        },
        "completed_input_token_counts": {
            "training": 0,
            "final_evaluation": expected_tokens,
            "reload_parity": 0,
            "total": expected_tokens,
        },
        "pending_forward": None,
    }
    with pytest.raises(ValueError, match="no incomplete planned forward"):
        ledger.begin("final_evaluation", specs["final_evaluation"][-1])
    assert ledger.snapshot()["completed_forward_counts"]["total"] == 164


def test_trained_role_executes_the_full_plan_in_fixed_category_order() -> None:
    specs = _specs("runtime_mix")
    ledger = progress.ForwardLedger("runtime_mix", specs)
    for category in _CATEGORIES:
        for spec in specs[category]:
            ledger.begin(category, spec)
            ledger.complete()

    ledger.require_complete()
    snapshot = ledger.snapshot()
    assert snapshot["completed_forward_counts"] == {
        "training": 1_344,
        "final_evaluation": 3_946,
        "reload_parity": 32,
        "total": 5_322,
    }
    assert snapshot["completed_input_token_counts"] == {
        category: sum(spec["input_tokens"] for spec in specs[category]) for category in _CATEGORIES
    } | {"total": sum(spec["input_tokens"] for rows in specs.values() for spec in rows)}
    assert (
        progress.validate_progress(
            snapshot, role="runtime_mix", specs_by_category=specs, require_complete=True
        )
        == snapshot
    )


def test_begin_enforces_next_category_and_next_exact_spec() -> None:
    specs = _specs("continued_practice")
    ledger = progress.ForwardLedger("continued_practice", specs)
    with pytest.raises(ValueError, match="next category is training"):
        ledger.begin("final_evaluation", specs["final_evaluation"][0])
    with pytest.raises(ValueError, match="unknown progress category"):
        ledger.begin("not-a-category", {})
    with pytest.raises(ValueError, match="next planned spec"):
        ledger.begin("training", {**specs["training"][0], "input_tokens": 22})

    ledger.begin("training", specs["training"][0])
    with pytest.raises(ValueError, match="pending forward must be completed"):
        ledger.begin("final_evaluation", specs["final_evaluation"][0])
    ledger.complete()
    with pytest.raises(ValueError, match="next category is training"):
        ledger.begin("final_evaluation", specs["final_evaluation"][0])
    ledger.begin("training", specs["training"][1])


@pytest.mark.parametrize("field", ["fixture_identity", "request_json_sha256", "input_tokens"])
def test_begin_rejects_changed_spec_even_with_recomputed_identity(field: str) -> None:
    specs = _specs("unchanged")
    ledger = progress.ForwardLedger("unchanged", specs)
    changed = deepcopy(specs["final_evaluation"][0])
    if field == "request_json_sha256":
        changed[field] = mixture_training_contracts.json_sha256({"different": "valid digest"})
    elif field == "input_tokens":
        changed[field] = changed[field] + 1
    else:
        changed[field] = "different but well-formed identity"

    with pytest.raises(ValueError, match="next planned spec"):
        ledger.begin("final_evaluation", changed)
    assert ledger.snapshot()["completed_forward_counts"]["total"] == 0
    assert ledger.snapshot()["pending_forward"] is None


def test_pending_attempt_cannot_be_replaced_or_completed_twice() -> None:
    specs = _specs("unchanged")
    ledger = progress.ForwardLedger("unchanged", specs)
    ledger.begin("final_evaluation", specs["final_evaluation"][0])
    pending_snapshot = ledger.snapshot()
    with pytest.raises(ValueError, match="pending forward must be completed"):
        ledger.begin("final_evaluation", specs["final_evaluation"][1])
    assert ledger.snapshot() == pending_snapshot

    ledger.complete()
    completed_snapshot = ledger.snapshot()
    with pytest.raises(ValueError, match="no pending forward"):
        ledger.complete()
    assert ledger.snapshot() == completed_snapshot


def test_started_but_uncompleted_attempt_is_unknown_and_cannot_pass_completion() -> None:
    specs = _specs("unchanged")
    ledger = progress.ForwardLedger("unchanged", specs)
    ledger.begin("final_evaluation", specs["final_evaluation"][0])

    with pytest.raises(ValueError, match="pending forward remains unresolved"):
        ledger.require_complete()
    snapshot = ledger.snapshot()
    assert snapshot["completed_forward_counts"]["total"] == 0
    assert snapshot["pending_forward"]["index"] == 0
    with pytest.raises(ValueError, match="pending forward must be completed"):
        ledger.begin("final_evaluation", specs["final_evaluation"][0])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["completed_forward_counts"].__setitem__("final_evaluation", True),
        lambda value: value["completed_forward_counts"].__setitem__("final_evaluation", 1.0),
        lambda value: value["completed_input_token_counts"].__setitem__("final_evaluation", False),
        lambda value: value["completed_input_token_counts"].__setitem__("final_evaluation", 1.0),
        lambda value: value["completed_forward_counts"].__setitem__("unexpected", 0),
        lambda value: value.__setitem__("unexpected", None),
    ],
    ids=[
        "bool-count",
        "float-count",
        "bool-token-total",
        "float-token-total",
        "extra-count",
        "extra-field",
    ],
)
def test_validate_progress_rejects_aliases_and_extra_fields(mutate) -> None:
    ledger, specs = _finished_unchanged()
    invalid = deepcopy(ledger.snapshot())
    mutate(invalid)
    with pytest.raises(ValueError):
        progress.validate_progress(invalid, role="unchanged", specs_by_category=specs)


def test_validate_progress_rejects_completed_prefix_token_drift() -> None:
    specs = _specs("unchanged")
    ledger = progress.ForwardLedger("unchanged", specs)
    ledger.begin("final_evaluation", specs["final_evaluation"][0])
    ledger.complete()
    invalid = ledger.snapshot()
    category_tokens = invalid["completed_input_token_counts"]["final_evaluation"]
    _reseal_totals(invalid, "final_evaluation", 1, category_tokens + 1)

    with pytest.raises(ValueError, match="do not equal the planned completed prefix"):
        progress.validate_progress(invalid, role="unchanged", specs_by_category=specs)


def test_validate_progress_rejects_later_category_before_earlier_plan_finishes() -> None:
    specs = _specs("continued_practice")
    ledger = progress.ForwardLedger("continued_practice", specs)
    invalid = ledger.snapshot()
    later_spec = specs["final_evaluation"][0]
    _reseal_totals(invalid, "final_evaluation", 1, later_spec["input_tokens"])

    with pytest.raises(ValueError, match="category execution order"):
        progress.validate_progress(invalid, role="continued_practice", specs_by_category=specs)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda pending: pending.__setitem__("category", "reload_parity"),
        lambda pending: pending.__setitem__("index", True),
        lambda pending: pending.__setitem__("index", 1.0),
        lambda pending: pending.__setitem__("spec_sha256", "0" * 64),
        lambda pending: pending.__setitem__("input_tokens", True),
        lambda pending: pending.__setitem__("input_tokens", 2.0),
        lambda pending: pending.__setitem__("extra", 0),
    ],
    ids=[
        "wrong-category",
        "bool-index",
        "float-index",
        "fake-hash",
        "bool-tokens",
        "float-tokens",
        "extra-pending-key",
    ],
)
def test_validate_progress_rejects_fabricated_pending_metadata(mutate) -> None:
    specs = _specs("unchanged")
    ledger = progress.ForwardLedger("unchanged", specs)
    ledger.begin("final_evaluation", specs["final_evaluation"][0])
    invalid = ledger.snapshot()
    mutate(invalid["pending_forward"])

    with pytest.raises(ValueError):
        progress.validate_progress(invalid, role="unchanged", specs_by_category=specs)


def test_validate_progress_returns_a_detached_copy_of_valid_partial_evidence() -> None:
    specs = _specs("unchanged")
    ledger = progress.ForwardLedger("unchanged", specs)
    ledger.begin("final_evaluation", specs["final_evaluation"][0])
    evidence = ledger.snapshot()

    validated = progress.validate_progress(evidence, role="unchanged", specs_by_category=specs)
    validated["pending_forward"]["category"] = "training"
    validated["completed_input_token_counts"]["total"] = 99
    assert ledger.snapshot() == evidence
    assert evidence["pending_forward"]["category"] == "final_evaluation"


def test_validate_progress_requires_full_plan_when_requested() -> None:
    specs = _specs("unchanged")
    ledger = progress.ForwardLedger("unchanged", specs)
    with pytest.raises(ValueError, match="does not complete the planned role budget"):
        progress.validate_progress(
            ledger.snapshot(), role="unchanged", specs_by_category=specs, require_complete=True
        )

    complete, specs = _finished_unchanged()
    assert (
        progress.validate_progress(
            complete.snapshot(), role="unchanged", specs_by_category=specs, require_complete=True
        )
        == complete.snapshot()
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda specs: specs.pop("reload_parity"),
        lambda specs: specs.__setitem__("unknown", []),
        lambda specs: specs["final_evaluation"].pop(),
        lambda specs: specs["final_evaluation"].__setitem__(0, {"input_tokens": True}),
        lambda specs: specs["final_evaluation"].__setitem__(0, {"input_tokens": 1.0}),
        lambda specs: specs["final_evaluation"].__setitem__(0, {"input_tokens": 0}),
        lambda specs: specs["final_evaluation"].__setitem__(0, {"input_tokens": 2_049}),
        lambda specs: specs["final_evaluation"].__setitem__(0, {"input_tokens": 1, "bad": {1, 2}}),
    ],
    ids=[
        "missing-category",
        "extra-category",
        "wrong-list-length",
        "bool-token-plan",
        "float-token-plan",
        "zero-token-plan",
        "over-limit-token-plan",
        "non-json-plan",
    ],
)
def test_plan_construction_rejects_invalid_role_budget_or_specs(mutate) -> None:
    specs = _specs("unchanged")
    mutate(specs)
    with pytest.raises((TypeError, ValueError)):
        progress.ForwardLedger("unchanged", specs)
