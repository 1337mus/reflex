from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext

import pytest

from reflex_decisions.schema import DecisionRequest, Option


def test_check_compilation_rejects_a_remote_candidate_identity_change() -> None:
    from experiments import fresh_eval_runtime as runtime

    expected = {
        "presentation_id": "fresh-eval-v1:record:order-0",
        "request_hash": "a" * 64,
        "order_ids": ["a", "b"],
        "prompt_sha256": "b" * 64,
        "input_tokens": 3,
        "candidate_token_ids": [1, 2],
        "input_ids_sha256": hashlib.sha256(
            json.dumps([4, 5, 6], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }

    with pytest.raises(ValueError, match="candidate"):
        runtime._check_compilation(
            expected,
            type(
                "Compiled",
                (),
                {
                    "request_hash": "a" * 64,
                    "prompt_hash": "b" * 64,
                    "input_ids": (4, 5, 6),
                    "candidate_token_ids": (1, 3),
                },
            )(),
        )


def test_reload_delta_rejects_a_changed_semantic_winner() -> None:
    from experiments import fresh_eval_runtime as runtime

    primary = [
        {
            "presentation_id": "fresh-eval-v1:record:order-0",
            "winner_option_id": "a",
            "candidate_logits": [1.0, 0.0],
        }
    ]
    reloaded = [
        {
            "presentation_id": "fresh-eval-v1:record:order-0",
            "winner_option_id": "b",
            "candidate_logits": [0.0, 1.0],
        }
    ]

    with pytest.raises(RuntimeError, match="semantic winner"):
        runtime._candidate_logit_delta(primary, reloaded)


def test_failed_forward_marks_one_inflight_call_unknown(monkeypatch) -> None:
    from experiments import fresh_eval_runtime as runtime

    request = DecisionRequest(
        context="Context.",
        question="Question?",
        options=(Option(id="a", label="A"), Option(id="b", label="B")),
    )
    presentation = {"request": request.model_dump(mode="json")}
    scoring = type(
        "Scoring",
        (),
        {
            "_score_forward": staticmethod(
                lambda *_args: (_ for _ in ()).throw(RuntimeError("lost"))
            )
        },
    )
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _name: scoring)
    result = {"evidence": {"unknown_work": None}}
    retained: list[dict[str, object]] = []

    with pytest.raises(RuntimeError, match="lost"):
        runtime._score_panel(
            object(),
            object(),
            [presentation],
            [{}],
            type("Torch", (), {"inference_mode": staticmethod(nullcontext)})(),
            result,
            "base_evaluation",
            retained,
        )

    assert result["evidence"]["unknown_work"] == {
        "possible_forwards": 1,
        "reason": "a model forward raised before its completion could be observed",
    }


@pytest.mark.parametrize("error", [RuntimeError("second forward lost"), KeyboardInterrupt()])
def test_second_forward_failure_retains_the_first_scored_row_and_count(monkeypatch, error) -> None:
    from experiments import fresh_eval_runtime as runtime

    request = DecisionRequest(
        context="Context.",
        question="Question?",
        options=(Option(id="a", label="A"), Option(id="b", label="B")),
    )
    calls = 0

    def score_forward(*_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error
        result["evidence"]["forward_counts"]["base_evaluation"] += 1
        return object(), object()

    scoring = type(
        "Scoring",
        (),
        {
            "_score_forward": staticmethod(score_forward),
            "_scored_row": staticmethod(lambda *_args: {"presentation_id": "first"}),
        },
    )
    monkeypatch.setattr(runtime.importlib, "import_module", lambda _name: scoring)
    monkeypatch.setattr(runtime, "_check_compilation", lambda *_args: None)
    result = {
        "evidence": {
            "unknown_work": None,
            "forward_counts": {"base_evaluation": 0},
        }
    }
    retained: list[dict[str, object]] = []
    presentation = {"request": request.model_dump(mode="json")}

    with pytest.raises(type(error)):
        runtime._score_panel(
            object(),
            object(),
            [presentation, presentation],
            [{}, {}],
            type("Torch", (), {"inference_mode": staticmethod(nullcontext)})(),
            result,
            "base_evaluation",
            retained,
        )

    assert retained == [{"presentation_id": "first"}]
    assert result["evidence"]["forward_counts"]["base_evaluation"] == 1
    assert result["evidence"]["unknown_work"]["possible_forwards"] == 1


def test_runtime_exception_message_falls_back_for_blank_keyboard_interrupt() -> None:
    from experiments import fresh_eval_runtime as runtime

    assert runtime._exception_message(KeyboardInterrupt()) == "KeyboardInterrupt"
