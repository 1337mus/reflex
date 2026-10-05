from __future__ import annotations

import sys
from types import ModuleType

import pytest

from reflex_decisions.schema import DecisionRequest, Option


def _presentations() -> list[dict[str, object]]:
    request = DecisionRequest(
        context="A premise.",
        question="Does the hypothesis follow?",
        options=(Option(id="entails", label="Entails"), Option(id="other", label="Other")),
    )
    reverse = request.model_copy(update={"options": tuple(reversed(request.options))})
    return [
        {
            "presentation_id": "fresh-eval-v1:hans-001:order-0",
            "record_id": "hans-001",
            "dataset_id": "hans-eval-v1",
            "source_group_id": "hans-001",
            "request_hash": request.request_hash,
            "order_index": 0,
            "order_ids": ["entails", "other"],
            "request": request.model_dump(mode="json"),
        },
        {
            "presentation_id": "fresh-eval-v1:hans-001:order-1",
            "record_id": "hans-001",
            "dataset_id": "hans-eval-v1",
            "source_group_id": "hans-001",
            "request_hash": reverse.request_hash,
            "order_index": 1,
            "order_ids": ["other", "entails"],
            "request": reverse.model_dump(mode="json"),
        },
    ]


def test_load_presentations_delegates_to_the_public_data_module(monkeypatch) -> None:
    from experiments import fresh_eval_core as core

    module = ModuleType("reflex_decisions.fresh_eval_data")
    calls = []
    module.build_presentations = lambda records: calls.append(records) or [{"presentation_id": "p"}]
    monkeypatch.setitem(sys.modules, "reflex_decisions.fresh_eval_data", module)

    assert core.build_presentations(("record",)) == [{"presentation_id": "p"}]
    assert calls == [("record",)]


def test_validate_presentations_rejects_a_leaked_gold_field() -> None:
    from experiments import fresh_eval_core as core

    rows = _presentations()
    rows[0]["answer_id"] = "entails"

    with pytest.raises(ValueError, match="schema"):
        core.validate_presentations(rows, expected_counts={"hans-eval-v1": 1})


@pytest.mark.parametrize("option_count", [3, 4, 5])
def test_validate_presentations_accepts_a_multichoice_left_rotation(option_count: int) -> None:
    from experiments import fresh_eval_core as core

    request = DecisionRequest(
        context="A science question.",
        question="Choose one.",
        options=tuple(
            Option(id=chr(97 + index), label=chr(65 + index)) for index in range(option_count)
        ),
    )
    rotated = request.model_copy(update={"options": request.options[1:] + request.options[:1]})
    rows = [
        {
            "presentation_id": "fresh-eval-v1:arc-001:order-0",
            "record_id": "arc-001",
            "dataset_id": "arc-challenge-dev-v1",
            "source_group_id": "arc-001",
            "request_hash": request.request_hash,
            "order_index": 0,
            "order_ids": [chr(97 + index) for index in range(option_count)],
            "request": request.model_dump(mode="json"),
        },
        {
            "presentation_id": "fresh-eval-v1:arc-001:order-1",
            "record_id": "arc-001",
            "dataset_id": "arc-challenge-dev-v1",
            "source_group_id": "arc-001",
            "request_hash": rotated.request_hash,
            "order_index": 1,
            "order_ids": [chr(97 + index) for index in range(1, option_count)] + ["a"],
            "request": rotated.model_dump(mode="json"),
        },
    ]

    assert core.validate_presentations(rows, expected_counts={"arc-challenge-dev-v1": 1}) == rows


def test_compile_requests_binds_the_exact_input_and_candidate_ids() -> None:
    from experiments import fresh_eval_core as core

    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return [ord(character) for character in text]

    compiled = core.compile_requests(
        _presentations(), Tokenizer(), expected_counts={"hans-eval-v1": 1}
    )

    assert compiled[0]["presentation_id"] == "fresh-eval-v1:hans-001:order-0"
    assert compiled[0]["candidate_token_ids"] == [65, 66]
    assert compiled[0]["input_tokens"] > 0
    assert len(compiled[0]["input_ids_sha256"]) == 64


def test_validate_compiled_requests_rejects_an_altered_identity() -> None:
    from experiments import fresh_eval_core as core

    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return [ord(character) for character in text]

    presentations = _presentations()
    compiled = core.compile_requests(
        presentations, Tokenizer(), expected_counts={"hans-eval-v1": 1}
    )
    compiled[0]["order_ids"] = ["other", "entails"]

    with pytest.raises(ValueError, match="identity"):
        core.validate_compiled_requests(
            compiled, presentations, expected_counts={"hans-eval-v1": 1}
        )


def test_validate_gpu_pins_rejects_host_only_label_metadata() -> None:
    from experiments import fresh_eval_core as core

    pins = {
        "data_file_sha256": {"source.tsv": "a" * 64},
        "dataset_counts": {
            "hans-eval-v1": 300,
            "winogrande-dev-v1": 200,
            "arc-challenge-dev-v1": 200,
        },
        "panel_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "answer_ids": ["entails"],
    }

    with pytest.raises(ValueError, match="schema"):
        core.validate_gpu_pins(pins)
