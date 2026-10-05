from __future__ import annotations

import pytest


def test_pending_ledger_preserves_completed_work_and_marks_failed_forward_unknown() -> None:
    from experiments import targeted_training_runtime as runtime

    ledger = runtime.PendingForwardLedger("control")
    ledger.complete("training", 12)
    ledger.fail_pending("final_evaluation", "p-1")

    assert ledger.completed == {"training": 12, "final_evaluation": 0, "reload_parity": 0}
    assert ledger.attempted_unknown == {"training": 0, "final_evaluation": 1, "reload_parity": 0}
    assert ledger.not_started == {"training": 4788, "final_evaluation": 6181, "reload_parity": 32}
    assert ledger.failure == {"category": "final_evaluation", "presentation_id": "p-1"}


def test_pending_ledger_counts_only_completed_forward_tokens() -> None:
    from experiments import targeted_training_runtime as runtime

    ledger = runtime.PendingForwardLedger("unchanged")
    ledger.complete("final_evaluation", input_tokens=11)
    ledger.fail_pending("final_evaluation", "second")

    assert ledger.receipt()["input_token_counts"] == {
        "training": 0,
        "final_evaluation": 11,
        "reload_parity": 0,
        "total": 11,
    }


def test_validate_optimizer_configuration_requires_empty_fresh_adamw_state() -> None:
    from experiments import targeted_training_runtime as runtime

    with pytest.raises(ValueError, match="empty"):
        runtime.validate_optimizer_configuration(
            {"lr": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0, "state": {"x": 1}}
        )


def test_scored_forward_preserves_pending_work_when_model_call_throws() -> None:
    from experiments import targeted_training_runtime as runtime
    from reflex_decisions.schema import DecisionRequest

    class Tokenizer:
        def encode(self, text, *, add_special_tokens=False):
            values = list(range(len(text)))
            if text[-1:] in {"A", "B"}:
                values[-1] = ord(text[-1])
            return values

    request = DecisionRequest.model_validate(
        {
            "context": "context",
            "question": "question",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        }
    )
    ledger = runtime.PendingForwardLedger("unchanged")
    with pytest.raises(RuntimeError, match="boom"):
        runtime.run_scored_forward(
            ledger,
            "final_evaluation",
            {
                "presentation_id": "p-1",
                "request_hash": request.request_hash,
                "order_ids": ["a", "b"],
                "request": request.model_dump(mode="json"),
            },
            Tokenizer(),
            lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    assert ledger.completed["final_evaluation"] == 0
    assert ledger.attempted_unknown["final_evaluation"] == 1
    assert ledger.not_started["final_evaluation"] == 6181
    assert ledger.failure == {"category": "final_evaluation", "presentation_id": "p-1"}


def test_checked_score_blocks_mismatched_compilation_and_preserves_pending_failure() -> None:
    from experiments import targeted_training_runtime as runtime
    from reflex_decisions.schema import DecisionRequest

    class Tokenizer:
        def encode(self, text, *, add_special_tokens=False):
            values = list(range(len(text)))
            if text[-1:] in {"A", "B"}:
                values[-1] = ord(text[-1])
            return values

    request = DecisionRequest.model_validate(
        {
            "context": "c",
            "question": "q",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        }
    )
    row = {
        "presentation_id": "p",
        "request_hash": request.request_hash,
        "order_ids": ["a", "b"],
        "request": request.model_dump(mode="json"),
    }
    expected = runtime.compile_presentations((row,), Tokenizer())[0]
    ledger = runtime.PendingForwardLedger("unchanged")
    calls = []

    class Scoring:
        def _score_forward(self, *_args):
            calls.append(1)
            raise RuntimeError("model failed")

    with pytest.raises(RuntimeError):
        runtime._checked_score(
            Scoring(),
            object(),
            row,
            expected,
            Tokenizer(),
            object(),
            {"evidence": {}},
            ledger,
            "final_evaluation",
        )
    assert calls == [1]
    assert ledger.attempted_unknown["final_evaluation"] == 1
    with pytest.raises(ValueError, match="compilation"):
        runtime._checked_score(
            Scoring(),
            object(),
            row,
            {**expected, "prompt_sha256": "bad"},
            Tokenizer(),
            object(),
            {"evidence": {}},
            runtime.PendingForwardLedger("unchanged"),
            "final_evaluation",
        )
    assert calls == [1]


def test_training_schedule_rejects_legal_but_incorrect_gold_index() -> None:
    from experiments import targeted_training_runtime as runtime
    from reflex_decisions.schema import DecisionRequest

    request = DecisionRequest.model_validate(
        {
            "context": "toy",
            "question": "Choose",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        }
    )
    rows = []
    for index in range(4800):
        update, micro = index // 4 + 1, index % 4
        rows.append(
            {
                "presentation_id": f"targeted-reasoning-v1:control:update-{update}:micro-{micro}",
                "record_id": f"r-{index}",
                "dataset_id": "toy",
                "source_group_id": "g",
                "request_hash": request.request_hash,
                "order_index": 0,
                "order_ids": ["a", "b"],
                "request": request.model_dump(mode="json"),
                "gold_option_id": "a",
                "gold_index": 0,
                "update": update,
                "microbatch_index": micro,
            }
        )
    rows[0]["gold_index"] = 1

    with pytest.raises(ValueError, match="gold"):
        runtime.validate_training_schedule("control", rows)


def test_worker_rejects_bad_payload_before_model_import(monkeypatch) -> None:
    from experiments import targeted_training_runtime as runtime

    imported = []
    real_import = runtime.importlib.import_module

    def forbidden_import(name, *args, **kwargs):
        if name in {"torch", "peft", "transformers"}:
            imported.append(name)
            raise AssertionError("model dependency imported before immutable payload validation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(runtime.importlib, "import_module", forbidden_import)
    payload = {
        "role": "unchanged",
        "training": (),
        "evaluation_presentations": [{} for _ in range(6182)],
        "compiled": {},
        "payload_sha256": "bad",
    }

    with pytest.raises(ValueError, match="payload digest|schema"):
        runtime.run_worker(payload)
    assert imported == []
