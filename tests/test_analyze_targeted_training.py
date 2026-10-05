from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import pytest

from experiments import analyze_targeted_training as analysis
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _presentation() -> dict[str, object]:
    return {
        "presentation_id": "p1",
        "record_id": "r1",
        "dataset_id": "d1",
        "source_group_id": "g1",
        "request_hash": "h1",
        "order_index": 0,
        "order_ids": ["a", "b"],
    }


def _score() -> dict[str, object]:
    return _presentation() | {
        "winner_option_id": "a",
        "candidate_logits": [1.0, 0.0],
        "input_tokens": 9,
        "prompt_sha256": "prompt",
    }


def _compiled() -> tuple[dict[str, object], ...]:
    return (
        {
            "presentation_id": "p1",
            "request_hash": "h1",
            "input_tokens": 9,
            "prompt_sha256": "prompt",
        },
    )


def _lifecycle() -> dict[str, object]:
    return {
        "status": "passed",
        "app_id": "ap-targeted",
        "calls": {
            "unchanged": "fc-reference",
            "control": "fc-control",
            "treatment": "fc-treatment",
        },
        "teardown": {
            "app_id": "ap-targeted",
            "verified": True,
            "stop_requested": False,
            "observations": [{"attempt": 1, "state": "APP_STATE_STOPPED", "n_tasks": 0}],
            "errors": [],
        },
    }


def test_validate_outputs_rejects_missing_compiled_prompt_identity() -> None:
    row = _score()
    del row["prompt_sha256"]
    with pytest.raises(ValueError, match="identity"):
        analysis.validate_outputs((row,), (_presentation(),), _compiled())


def test_validate_outputs_rejects_duplicate_presentation_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        analysis.validate_outputs((_score(), _score()), (_presentation(),), _compiled())


def test_validate_outputs_rejects_incompatible_immutable_identity() -> None:
    row = _score() | {"record_id": "wrong"}
    with pytest.raises(ValueError, match="incompatible"):
        analysis.validate_outputs((row,), (_presentation(),), _compiled())


def test_validate_outputs_rejects_wrong_compiled_token_count() -> None:
    compiled = (
        {
            "presentation_id": "p1",
            "request_hash": "h1",
            "input_tokens": 10,
            "prompt_sha256": "prompt",
        },
    )

    with pytest.raises(ValueError, match="compiled|token"):
        analysis.validate_outputs((_score(),), (_presentation(),), compiled)


def test_receipt_rejects_missing_payload_binding_before_gold() -> None:
    receipt = {
        "status": "passed",
        "roles": {
            role: {"status": "passed", "execution": {"unknown": {}}} for role in analysis.ROLES
        },
    }

    with pytest.raises(ValueError, match="payload|schema"):
        analysis.validate_complete_receipt(receipt)


def test_receipt_rejects_wrong_role_payload_digest() -> None:
    plan = {
        "experiment_id": "targeted-reasoning-v1",
        "run_id": "run",
        "roles": [],
        "role_payload_sha256": {role: "0" * 64 for role in analysis.ROLES},
    }
    plan_sha = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payloads = {
        role: {
            "experiment_id": "targeted-reasoning-v1",
            "run_id": "run",
            "role": role,
            "payload_sha256": "0" * 64,
        }
        for role in analysis.ROLES
    }
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": "targeted-reasoning-v1",
        "run_id": "run",
        "plan": plan | {"plan_sha256": plan_sha},
        "role_payloads": payloads,
        "roles": {role: {} for role in analysis.ROLES},
        "lifecycle": _lifecycle(),
    }

    with pytest.raises(ValueError, match="payload identity or digest"):
        analysis.validate_complete_receipt(
            receipt, expected_plan_sha256=plan_sha, plan=receipt["plan"]
        )


def test_receipt_requires_plan_to_bind_all_role_payload_digests() -> None:
    plan = {"experiment_id": "targeted-reasoning-v1", "run_id": "run", "roles": []}
    sha = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": "targeted-reasoning-v1",
        "run_id": "run",
        "plan": plan | {"plan_sha256": sha},
        "role_payloads": {role: {} for role in analysis.ROLES},
        "roles": {role: {} for role in analysis.ROLES},
        "lifecycle": _lifecycle(),
    }

    with pytest.raises(ValueError, match="plan.*payload"):
        analysis.validate_complete_receipt(receipt, expected_plan_sha256=sha, plan=receipt["plan"])


def _host_receipt() -> dict[str, object]:
    payloads = {}
    roles = {}
    for role in analysis.ROLES:
        body = {"experiment_id": "targeted-reasoning-v1", "run_id": "run", "role": role}
        digest = analysis._canonical_digest(body)
        payloads[role] = body | {"payload_sha256": digest}
        raw = body | {"status": "passed", "payload_sha256": digest}
        roles[role] = {"raw_worker_result": raw, "validated_result": raw}
    plan_body = {
        "experiment_id": "targeted-reasoning-v1",
        "run_id": "run",
        "role_payload_sha256": {role: payloads[role]["payload_sha256"] for role in analysis.ROLES},
    }
    plan = plan_body | {"plan_sha256": analysis._canonical_digest(plan_body)}
    return {
        "schema_version": 1,
        "status": "passed",
        "experiment_id": "targeted-reasoning-v1",
        "run_id": "run",
        "plan": plan,
        "role_payloads": payloads,
        "roles": roles,
        "lifecycle": _lifecycle(),
    }


@pytest.mark.parametrize("mutation", ("missing", "failed", "wrong_app", "unobserved", "calls"))
def test_receipt_rejects_bad_teardown_before_gold(mutation: str) -> None:
    receipt = _host_receipt()
    lifecycle = receipt["lifecycle"]
    if mutation == "missing":
        del receipt["lifecycle"]
    elif mutation == "failed":
        lifecycle["teardown"]["verified"] = False
    elif mutation == "wrong_app":
        lifecycle["teardown"]["app_id"] = "ap-other"
    elif mutation == "unobserved":
        lifecycle["teardown"]["observations"] = [{"state": "APP_STATE_STOPPING", "n_tasks": 0}]
    else:
        lifecycle["calls"]["control"] = lifecycle["calls"]["treatment"]

    class NoGold:
        @property
        def evaluation_records(self):
            raise AssertionError("gold opened before lifecycle validation")

    with pytest.raises(ValueError, match="lifecycle|teardown|FunctionCall"):
        analysis.analyze_receipt(
            receipt,
            NoGold(),
            expected_plan_sha256=receipt["plan"]["plan_sha256"],
            plan=receipt["plan"],
        )


def test_receipt_accepts_completed_host_lifecycle_shape(monkeypatch) -> None:
    receipt = _host_receipt()
    monkeypatch.setattr(analysis.core, "validate_completed_result", lambda _payload, raw: raw)
    accepted = analysis.validate_complete_receipt(
        receipt, expected_plan_sha256=receipt["plan"]["plan_sha256"], plan=receipt["plan"]
    )
    assert accepted["lifecycle"]["teardown"]["observations"][0]["n_tasks"] == 0


def test_summarize_weights_questions_before_groups() -> None:
    records = []
    panel = []
    rows = []
    compiled = []
    for record_id, group_id, orders, winner in (
        ("r1", "g1", 1, "a"),
        ("r2", "g1", 3, "b"),
        ("r3", "g2", 1, "a"),
    ):
        request = DecisionRequest(
            context=record_id,
            question="Choose",
            options=(
                Option(id="a", label="A"),
                Option(id="b", label="B"),
                Option(id="c", label="C"),
            ),
        )
        records.append(
            DecisionRecord(
                record_id=record_id,
                dataset_id="toy",
                source_group_id=group_id,
                request=request,
                answer_id="a",
            )
        )
        for order in range(orders):
            ids = ["a", "b", "c"]
            ids = ids[order:] + ids[:order]
            ordered_request = request.model_copy(
                update={
                    "options": tuple(
                        next(option for option in request.options if option.id == item)
                        for item in ids
                    )
                }
            )
            identity = {
                "presentation_id": f"{record_id}-{order}",
                "record_id": record_id,
                "dataset_id": "toy",
                "source_group_id": group_id,
                "request_hash": request.request_hash,
                "order_index": order,
                "order_ids": ids,
                "request": ordered_request.model_dump(mode="json"),
            }
            panel.append(identity)
            score = [1.0 if value == winner else 0.0 for value in ids]
            rows.append(
                {key: identity[key] for key in analysis.IDENTITY_FIELDS}
                | {
                    "candidate_logits": score,
                    "winner_option_id": winner,
                    "input_tokens": 9,
                    "prompt_sha256": "prompt",
                }
            )
            compiled.append(
                {
                    "presentation_id": identity["presentation_id"],
                    "request_hash": request.request_hash,
                    "input_tokens": 9,
                    "prompt_sha256": "prompt",
                }
            )
    outputs = {role: rows for role in analysis.ROLES}
    report = analysis.summarize(
        outputs, records, panel, {role: compiled for role in analysis.ROLES}
    )

    assert report["datasets"]["toy"]["roles"]["unchanged"]["group_mean_accuracy"] == 0.75


def _passing_learning_report() -> dict[str, object]:
    reserved = ("targeted-hans-reserved-v1", "targeted-winogrande-reserved-v1")
    floors = (
        "dbpedia14-pilot-v1-development",
        "sms-pilot-v1-development",
        "snli-balanced-v1-development",
        "synthetic-atomic-fact-inference-v1-development",
        "synthetic-numeric-selection-v1-development",
        "boolq-dev-pilot-v1",
        "copa-dev-pilot-v1",
        "arc-challenge-dev-v1",
    )
    monitoring = ("hans-eval-v1", "winogrande-dev-v1")
    datasets = {}
    for task in (*reserved, *floors, *monitoring):
        difference = "1/20" if task in reserved else "-1/20"
        datasets[task] = {
            "differences": {
                "treatment-control": {
                    "group_mean_difference_fraction": difference,
                    "bootstrap_95": {"lower": 0.001, "upper": 0.2},
                },
                "treatment-unchanged": {
                    "group_mean_difference_fraction": difference,
                    "bootstrap_95": {"lower": 0.001, "upper": 0.2},
                },
            }
        }
    return {
        "datasets": datasets,
        "hans_classes": {
            answer: {"treatment_unchanged_fraction": "-1/20"}
            for answer in ("entailment", "non-entailment")
        },
    }


def test_learning_accepts_exact_five_point_boundaries_across_24_checks() -> None:
    result = analysis.evaluate_learning(_passing_learning_report())

    assert result["passed"] is True
    assert len(result["checks"]) == 24
    assert result["failed_check_names"] == []


def test_one_reserved_target_cannot_cancel_another_and_lower_bound_is_strict() -> None:
    report = _passing_learning_report()
    hans = report["datasets"]["targeted-hans-reserved-v1"]["differences"]
    wino = report["datasets"]["targeted-winogrande-reserved-v1"]["differences"]
    hans["treatment-control"]["group_mean_difference_fraction"] = "1/5"
    wino["treatment-control"]["group_mean_difference_fraction"] = "49/1000"
    hans["treatment-control"]["bootstrap_95"]["lower"] = 0.0

    result = analysis.evaluate_learning(report)

    assert result["passed"] is False
    assert result["failed_check_names"] == [
        "targeted-hans-reserved-v1.paired_lower_positive",
        "targeted-winogrande-reserved-v1.treatment_control_gain",
    ]


def test_class_and_retention_failures_are_independent() -> None:
    report = _passing_learning_report()
    report["hans_classes"]["non-entailment"]["treatment_unchanged_fraction"] = "-51/1000"
    report["datasets"]["boolq-dev-pilot-v1"]["differences"]["treatment-unchanged"][
        "group_mean_difference_fraction"
    ] = "-51/1000"

    result = analysis.evaluate_learning(report)

    assert result["failed_check_names"] == [
        "targeted-hans-reserved-v1.non-entailment.class_floor",
        "boolq-dev-pilot-v1.treatment_unchanged_floor",
    ]


def test_bootstrap_draws_share_deterministic_group_indices_and_type7() -> None:
    import random

    rng = random.Random(20261015)
    draws = analysis._draws(("a", "b"), rng)
    assert len(draws) == 2000
    assert draws[:2] == (("a", "a"), ("b", "b"))
    assert analysis._type7([Fraction(0), Fraction(1), Fraction(2)], Fraction(1, 40)) == 0.05


@pytest.mark.parametrize(
    "change",
    [
        lambda row: row.pop("source_group_id"),
        lambda row: row.update(winner_option_id="b"),
        lambda row: row.update(candidate_logits=[1.0]),
        lambda row: row.update(candidate_logits=[float("nan"), 0.0]),
        lambda row: row.update(candidate_logits=[True, 0.0]),
        lambda row: row.update(candidate_logits=[10**1000, 0.0]),
        lambda row: row.update(input_tokens=9.0),
        lambda row: row.update(correct=True),
    ],
)
def test_score_rows_reject_malformed_identity_winner_logits_or_labels(change) -> None:
    row = _score()
    change(row)

    with pytest.raises(ValueError):
        analysis.validate_outputs((row,), (_presentation(),), _compiled())


def test_score_tie_uses_minimum_semantic_option_id() -> None:
    row = _score()
    row["candidate_logits"] = [1.0, 1.0]

    assert (
        analysis.validate_outputs((row,), (_presentation(),), _compiled())["p1"]["winner_option_id"]
        == "a"
    )


def test_host_gold_requires_exact_membership_and_request_hash() -> None:
    request = DecisionRequest(
        context="toy",
        question="Choose",
        options=(Option(id="a", label="A"), Option(id="b", label="B")),
    )
    record = DecisionRecord(
        record_id="r1", dataset_id="d1", source_group_id="g1", request=request, answer_id="a"
    )
    extra = DecisionRecord(
        record_id="unused", dataset_id="d1", source_group_id="g1", request=request, answer_id="a"
    )
    panel = (
        _presentation()
        | {"request_hash": request.request_hash, "request": request.model_dump(mode="json")},
    )
    rows = (_score() | {"request_hash": request.request_hash},)
    compiled = (_compiled()[0] | {"request_hash": request.request_hash},)
    outputs = {role: rows for role in analysis.ROLES}
    compilations = {role: compiled for role in analysis.ROLES}

    with pytest.raises(ValueError, match="membership"):
        analysis.summarize(outputs, (record, extra), panel, compilations)
    with pytest.raises(ValueError, match="request hash"):
        analysis.summarize(
            outputs,
            (record,),
            (_presentation() | {"request": request.model_dump(mode="json")},),
            compilations,
        )


def test_partial_receipt_fails_before_host_gold_is_read() -> None:
    class GoldSpy:
        @property
        def evaluation_records(self) -> object:
            raise AssertionError("gold was opened before execution passed")

    with pytest.raises(ValueError, match="passed"):
        analysis.analyze_receipt(
            {"status": "failed"}, GoldSpy(), expected_plan_sha256="0" * 64, plan={}
        )


def test_cli_rejects_failed_receipt_before_loading_pinned_inputs(tmp_path, monkeypatch) -> None:
    from experiments import targeted_training_data

    receipt_path = tmp_path / "receipt.json"
    plan_path = tmp_path / "plan.json"
    receipt_path.write_text(json.dumps({"status": "failed"}))
    plan_path.write_text("{}")
    loaded = False

    def forbidden_load(_root: object) -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError("inputs opened before execution barrier")

    monkeypatch.setattr(targeted_training_data, "load_inputs", forbidden_load, raising=False)
    with pytest.raises(ValueError, match="passed"):
        analysis.main(
            [
                "--receipt",
                str(receipt_path),
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                "0" * 64,
                "--root",
                str(tmp_path),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
    assert loaded is False
