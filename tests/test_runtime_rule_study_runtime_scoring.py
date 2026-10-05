"""Synthetic checks for runtime-rule request preparation and model scoring."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import cast

import pytest

from experiments import (
    mixture_training_contracts,
    runtime_rule_study_payloads,
    runtime_rule_study_progress,
    runtime_rule_study_tokens,
)
from experiments import runtime_rule_study_runtime_scoring as scoring_api
from experiments.training_rehearsal_core import TrainingExample
from reflex_decisions.schema import DecisionRequest, Option


class CharacterTokenizer:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        self.encode_calls += 1
        return [ord(character) for character in text]


class FakeTensor:
    def __init__(self, values: object, *, requires_grad: bool = False) -> None:
        self.values = values
        self.requires_grad = requires_grad

    def index_select(self, _dimension: int, indices: FakeTensor) -> FakeTensor:
        assert isinstance(self.values, list)
        assert isinstance(indices.values, list)
        return FakeTensor(
            [self.values[index] for index in indices.values],
            requires_grad=self.requires_grad,
        )

    def float(self) -> FakeTensor:
        return self

    def numel(self) -> int:
        assert isinstance(self.values, list)
        return len(self.values)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[object]:
        assert isinstance(self.values, list)
        return [float(value) for value in self.values]


class FakeLogits:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, index: tuple[int, int]) -> FakeTensor:
        assert index == (0, -1)
        return FakeTensor(self.values, requires_grad=True)


class FakeFinite:
    def __init__(self, tensor: FakeTensor) -> None:
        self.tensor = tensor

    def all(self) -> FakeFinite:
        return self

    def item(self) -> bool:
        assert isinstance(self.tensor.values, list)
        return all(math.isfinite(float(value)) for value in self.tensor.values)


class FakeTorch:
    long = object()

    def __init__(self) -> None:
        self.inference_depth = 0
        self.tensor_calls: list[tuple[object, object, object]] = []

    def tensor(self, values: object, *, dtype: object, device: object) -> FakeTensor:
        self.tensor_calls.append((values, dtype, device))
        return FakeTensor(values)

    def ones_like(self, tensor: FakeTensor) -> FakeTensor:
        assert isinstance(tensor.values, list)
        rows = cast(list[list[object]], tensor.values)
        return FakeTensor([[1 for _ in rows[0]]])

    def isfinite(self, tensor: FakeTensor) -> FakeFinite:
        return FakeFinite(tensor)

    @contextmanager
    def inference_mode(self) -> Iterator[None]:
        self.inference_depth += 1
        try:
            yield
        finally:
            self.inference_depth -= 1


def _request(name: str, *, reverse: bool = False) -> DecisionRequest:
    options = (
        Option(id=f"{name}-first", label="First"),
        Option(id=f"{name}-second", label="Second"),
    )
    return DecisionRequest(
        context=f"Self-authored context for {name}.",
        question=f"Choose an option for {name}.",
        options=tuple(reversed(options)) if reverse else options,
    )


def _presentation(request: DecisionRequest, presentation_id: str) -> dict[str, object]:
    return {
        "presentation_id": presentation_id,
        "record_id": f"record-{presentation_id}",
        "dataset_id": "self-authored-fixture",
        "source_group_id": f"group-{presentation_id}",
        "request_hash": request.request_hash,
        "order_index": 0,
        "order_ids": [option.id for option in request.options],
        "request": request.model_dump(mode="json"),
    }


def _spec(
    request: DecisionRequest,
    tokenizer: CharacterTokenizer,
    *,
    record_id: str,
    presentation_id: str | None = None,
) -> dict[str, object]:
    result = runtime_rule_study_tokens.compile_spec(request, tokenizer)
    result["record_id"] = record_id
    if presentation_id is not None:
        result["presentation_id"] = presentation_id
    return result


def _install_toy_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    examples: tuple[TrainingExample, ...],
) -> None:
    monkeypatch.setattr(runtime_rule_study_payloads, "validate_payload", lambda value: payload)
    monkeypatch.setattr(runtime_rule_study_payloads, "training_examples", lambda value: examples)


def _ledger_for_training(
    item: scoring_api.PreparedItem,
) -> runtime_rule_study_progress.ForwardLedger:
    first_training_spec = json.loads(item.expected_spec_json)
    return runtime_rule_study_progress.ForwardLedger(
        "runtime_mix",
        {
            "training": [
                first_training_spec,
                *({"input_tokens": 1} for _ in range(1_343)),
            ],
            "final_evaluation": [{"input_tokens": 1} for _ in range(3_946)],
            "reload_parity": [{"input_tokens": 1} for _ in range(32)],
        },
    )


def _ledger_for_evaluation(
    items: tuple[scoring_api.PreparedItem, ...],
) -> runtime_rule_study_progress.ForwardLedger:
    planned = [json.loads(item.expected_spec_json) for item in items]
    planned.extend({"input_tokens": 1} for _ in range(164 - len(planned)))
    return runtime_rule_study_progress.ForwardLedger(
        "unchanged",
        {"training": [], "final_evaluation": planned, "reload_parity": []},
    )


def _ledger_for_reload(
    prepared: scoring_api.PreparedRequests,
) -> runtime_rule_study_progress.ForwardLedger:
    training_specs = [json.loads(item.expected_spec_json) for item in prepared.training_items]
    training_specs.extend({"input_tokens": 1} for _ in range(1_344 - len(training_specs)))
    evaluation_specs = [json.loads(item.expected_spec_json) for item in prepared.evaluation_items]
    evaluation_specs.extend({"input_tokens": 1} for _ in range(3_946 - len(evaluation_specs)))
    reload_specs = [json.loads(item.expected_spec_json) for item in prepared.reload_items]
    reload_specs.extend({"input_tokens": 1} for _ in range(32 - len(reload_specs)))
    ledger = runtime_rule_study_progress.ForwardLedger(
        "runtime_mix",
        {
            "training": training_specs,
            "final_evaluation": evaluation_specs,
            "reload_parity": reload_specs,
        },
    )
    for category, specs in (
        ("training", training_specs),
        ("final_evaluation", evaluation_specs),
    ):
        for spec in specs:
            ledger.begin(category, spec)
            ledger.complete()
    return ledger


def _output_row(
    presentation: dict[str, object],
    item: scoring_api.PreparedItem,
    candidate_logits: list[float],
) -> dict[str, object]:
    order_ids = cast(list[str], presentation["order_ids"])
    maximum = max(candidate_logits)
    return {
        **{
            key: presentation[key]
            for key in (
                "presentation_id",
                "record_id",
                "dataset_id",
                "source_group_id",
                "request_hash",
                "order_index",
                "order_ids",
            )
        },
        "candidate_logits": candidate_logits,
        "winner_option_id": min(
            option_id
            for option_id, value in zip(order_ids, candidate_logits, strict=True)
            if value == maximum
        ),
        "input_tokens": len(item.compiled.input_ids),
        "prompt_sha256": item.compiled.prompt_hash,
    }


def _toy_payload() -> tuple[dict[str, object], tuple[TrainingExample, ...]]:
    tokenizer = CharacterTokenizer()
    training_request = _request("training")
    evaluation_requests = (
        _request("evaluation", reverse=True),
        _request("evaluation-second"),
    )
    reload_requests = evaluation_requests
    training = TrainingExample(
        record_id="training-record",
        request=training_request,
        gold_option_id=training_request.options[0].id,
        gold_index=0,
    )
    evaluation = [
        _presentation(request, f"evaluation-row-{index}")
        for index, request in enumerate(evaluation_requests)
    ]
    reload = evaluation
    payload: dict[str, object] = {
        "role": "runtime_mix",
        "training_specs": [_spec(training_request, tokenizer, record_id=training.record_id)],
        "evaluation": evaluation,
        "evaluation_specs": [
            _spec(
                request,
                tokenizer,
                record_id=cast(str, presentation["record_id"]),
                presentation_id=cast(str, presentation["presentation_id"]),
            )
            for presentation, request in zip(evaluation, evaluation_requests, strict=True)
        ],
        "reload": reload,
        "reload_specs": [
            _spec(
                request,
                tokenizer,
                record_id=cast(str, presentation["record_id"]),
                presentation_id=cast(str, presentation["presentation_id"]),
            )
            for presentation, request in zip(reload, reload_requests, strict=True)
        ],
    }
    return payload, (training,)


def test_prepare_requests_compiles_every_category_and_detaches_exact_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    tokenizer = CharacterTokenizer()

    prepared = scoring_api.prepare_requests(payload, tokenizer)

    assert [item.category for item in prepared.training_items] == ["training"]
    assert [item.index for item in prepared.training_items] == [0]
    assert [item.category for item in prepared.evaluation_items] == [
        "final_evaluation",
        "final_evaluation",
    ]
    assert [item.index for item in prepared.evaluation_items] == [0, 1]
    assert [item.category for item in prepared.reload_items] == ["reload_parity", "reload_parity"]
    assert prepared.training_examples == examples
    assert prepared.training_items[0].request is examples[0].request
    evaluation_rows = cast(list[dict[str, object]], payload["evaluation"])
    training_specs = cast(list[dict[str, object]], payload["training_specs"])
    evaluation_specs = cast(list[dict[str, object]], payload["evaluation_specs"])
    reload_specs = cast(list[dict[str, object]], payload["reload_specs"])
    assert (
        prepared.evaluation_items[0].request.model_dump(mode="json")
        == evaluation_rows[0]["request"]
    )
    assert tokenizer.encode_calls == 15
    pairs = [(prepared.training_items[0], training_specs[0])]
    pairs.extend(zip(prepared.evaluation_items, evaluation_specs, strict=True))
    pairs.extend(zip(prepared.reload_items, reload_specs, strict=True))
    for item, spec in pairs:
        assert json.loads(item.expected_spec_json) == spec
        assert item.expected_spec_json == mixture_training_contracts.canonical_json(spec).decode(
            "utf-8"
        )


def test_unchanged_role_has_no_training_or_reload_items(monkeypatch: pytest.MonkeyPatch) -> None:
    payload, _examples = _toy_payload()
    unchanged_payload = dict(payload)
    unchanged_payload.update(
        {
            "role": "unchanged",
            "training_specs": [],
            "reload": [],
            "reload_specs": [],
        }
    )
    _install_toy_payload(monkeypatch, unchanged_payload, ())

    prepared = scoring_api.prepare_requests(unchanged_payload, CharacterTokenizer())

    assert prepared.training_items == ()
    assert prepared.training_examples == ()
    assert prepared.reload_items == ()
    assert len(prepared.evaluation_items) == 2


def test_invalid_payload_is_rejected_before_tokenization() -> None:
    tokenizer = CharacterTokenizer()

    with pytest.raises(ValueError, match="payload has an unexpected top-level schema"):
        scoring_api.prepare_requests({}, tokenizer)

    assert tokenizer.encode_calls == 0


def test_score_forward_keeps_autograd_and_completes_ledger_before_reading_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    item = scoring_api.prepare_requests(payload, CharacterTokenizer()).training_items[0]
    ledger = _ledger_for_training(item)
    evidence: dict[str, object] = {}
    torch = FakeTorch()
    vocabulary = [0.0] * 128
    vocabulary[item.compiled.candidate_token_ids[0]] = 1.25
    vocabulary[item.compiled.candidate_token_ids[1]] = -0.5

    class Model:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            assert evidence["pending_forward"] == ledger.snapshot()["pending_forward"]
            counts = cast(dict[str, object], evidence["completed_forward_counts"])
            assert counts["training"] == 0
            assert kwargs["use_cache"] is False
            assert kwargs["logits_to_keep"] == 1
            assert torch.inference_depth == 0
            return type("Output", (), {"logits": FakeLogits(vocabulary)})()

    model = Model()

    candidate_logits = scoring_api.score_forward(model, item, torch, ledger, evidence, "training")

    assert candidate_logits.values == [1.25, -0.5]
    assert candidate_logits.requires_grad is True
    input_ids = cast(FakeTensor, model.calls[0]["input_ids"])
    attention_mask = cast(FakeTensor, model.calls[0]["attention_mask"])
    assert input_ids.values == [list(item.compiled.input_ids)]
    assert attention_mask.values == [[1] * len(item.compiled.input_ids)]
    completed = cast(dict[str, object], evidence["completed_forward_counts"])
    assert completed["training"] == 1
    assert evidence["pending_forward"] is None


def test_model_exception_retains_the_pending_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    item = scoring_api.prepare_requests(payload, CharacterTokenizer()).training_items[0]
    ledger = _ledger_for_training(item)
    evidence: dict[str, object] = {}

    class FailingModel:
        def __call__(self, **_kwargs: object) -> object:
            raise OSError("synthetic model failure")

    with pytest.raises(OSError, match="synthetic model failure"):
        scoring_api.score_forward(FailingModel(), item, FakeTorch(), ledger, evidence, "training")

    assert evidence == ledger.snapshot()
    counts = cast(dict[str, object], evidence["completed_forward_counts"])
    pending = cast(dict[str, object], evidence["pending_forward"])
    assert counts["training"] == 0
    assert pending["category"] == "training"


def test_unreadable_returned_output_is_completed_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    item = scoring_api.prepare_requests(payload, CharacterTokenizer()).training_items[0]
    ledger = _ledger_for_training(item)
    evidence: dict[str, object] = {}

    class UnreadableModel:
        def __call__(self, **_kwargs: object) -> object:
            return object()

    with pytest.raises(RuntimeError, match="readable final-token logits"):
        scoring_api.score_forward(
            UnreadableModel(), item, FakeTorch(), ledger, evidence, "training"
        )

    assert evidence == ledger.snapshot()
    counts = cast(dict[str, object], evidence["completed_forward_counts"])
    assert counts["training"] == 1
    assert evidence["pending_forward"] is None


def test_nonfinite_candidate_logits_fail_after_completed_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    item = scoring_api.prepare_requests(payload, CharacterTokenizer()).training_items[0]
    ledger = _ledger_for_training(item)
    evidence: dict[str, object] = {}
    vocabulary = [0.0] * 128
    vocabulary[item.compiled.candidate_token_ids[0]] = float("nan")

    class NonfiniteModel:
        def __call__(self, **_kwargs: object) -> object:
            return type("Output", (), {"logits": FakeLogits(vocabulary)})()

    with pytest.raises(RuntimeError, match="non-finite value"):
        scoring_api.score_forward(NonfiniteModel(), item, FakeTorch(), ledger, evidence, "training")

    counts = cast(dict[str, object], evidence["completed_forward_counts"])
    assert counts["training"] == 1
    assert evidence["pending_forward"] is None


def test_wrong_order_and_exhausted_ledgers_make_no_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    prepared = scoring_api.prepare_requests(payload, CharacterTokenizer())
    item = prepared.evaluation_items[0]

    class NeverCalledModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **_kwargs: object) -> object:
            self.calls += 1
            raise AssertionError("preflight must stop before model work")

    model = NeverCalledModel()
    with pytest.raises(ValueError, match="next category is training"):
        scoring_api.score_forward(
            model,
            item,
            FakeTorch(),
            _ledger_for_training(prepared.training_items[0]),
            {},
            "final_evaluation",
        )

    evaluation_specs = [json.loads(row.expected_spec_json) for row in prepared.evaluation_items]
    evaluation_specs.extend({"input_tokens": 1} for _ in range(164 - len(evaluation_specs)))
    exhausted = runtime_rule_study_progress.ForwardLedger(
        "unchanged",
        {"training": [], "final_evaluation": evaluation_specs, "reload_parity": []},
    )
    for spec in evaluation_specs:
        exhausted.begin("final_evaluation", spec)
        exhausted.complete()
    with pytest.raises(ValueError, match="prepared item index differs"):
        scoring_api.score_forward(model, item, FakeTorch(), exhausted, {}, "final_evaluation")

    assert model.calls == 0


def test_score_presentations_appends_rows_and_uses_canonical_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    prepared = scoring_api.prepare_requests(payload, CharacterTokenizer())
    items = prepared.evaluation_items
    presentations = cast(list[dict[str, object]], payload["evaluation"])
    ledger = _ledger_for_evaluation(items)
    evidence: dict[str, object] = {"outputs": []}
    torch = FakeTorch()

    class TieModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **_kwargs: object) -> object:
            self.calls += 1
            assert torch.inference_depth == 1
            return type("Output", (), {"logits": FakeLogits([0.0] * 65 + [4.0, 4.0])})()

    model = TieModel()
    progress: list[tuple[int, int]] = []

    scoring_api.score_presentations(
        model,
        items,
        presentations,
        torch,
        ledger,
        evidence,
        "final_evaluation",
        on_progress=lambda completed, total: progress.append((completed, total)),
    )

    outputs = cast(list[dict[str, object]], evidence["outputs"])
    assert model.calls == 2
    assert [row["winner_option_id"] for row in outputs] == [
        min(cast(list[str], row["order_ids"])) for row in presentations
    ]
    assert [row["candidate_logits"] for row in outputs] == [[4.0, 4.0], [4.0, 4.0]]
    assert progress == [(1, 2), (2, 2)]
    counts = cast(dict[str, object], evidence["completed_forward_counts"])
    assert counts["final_evaluation"] == 2
    assert torch.inference_depth == 0


def test_late_panel_identity_mismatch_prevents_all_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    prepared = scoring_api.prepare_requests(payload, CharacterTokenizer())
    first, second = prepared.evaluation_items
    presentations = cast(list[dict[str, object]], payload["evaluation"])
    changed_spec = json.loads(second.expected_spec_json)
    changed_spec["presentation_id"] = "unexpected-late-row"
    tampered_second = replace(
        second,
        expected_spec_json=mixture_training_contracts.canonical_json(changed_spec).decode("utf-8"),
    )
    items = (first, tampered_second)
    ledger = _ledger_for_evaluation(items)
    evidence: dict[str, object] = {"outputs": []}

    class NeverCalledModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **_kwargs: object) -> object:
            self.calls += 1
            raise AssertionError("all rows must pass preflight before model work")

    model = NeverCalledModel()
    with pytest.raises(ValueError, match="differs from its prepared request"):
        scoring_api.score_presentations(
            model,
            items,
            presentations,
            FakeTorch(),
            ledger,
            evidence,
            "final_evaluation",
        )

    assert model.calls == 0
    assert evidence["outputs"] == []
    snapshot = ledger.snapshot()
    counts = cast(dict[str, object], snapshot["completed_forward_counts"])
    assert counts["final_evaluation"] == 0


def test_reload_stops_on_first_parity_failure_and_preserves_adverse_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    prepared = scoring_api.prepare_requests(payload, CharacterTokenizer())
    presentations = cast(list[dict[str, object]], payload["reload"])
    final_outputs = [
        _output_row(presentation, item, [0.0, 1.0])
        for presentation, item in zip(presentations, prepared.evaluation_items, strict=True)
    ]
    evidence: dict[str, object] = {
        "outputs": final_outputs,
        "reload": {
            "outputs": [],
            "tensor_equal": None,
            "tensor_sha256": None,
            "winner_match_count": None,
            "max_candidate_logit_difference": None,
        },
    }
    ledger = _ledger_for_reload(prepared)
    torch = FakeTorch()

    class ParityFailureModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **_kwargs: object) -> object:
            self.calls += 1
            assert torch.inference_depth == 1
            return type("Output", (), {"logits": FakeLogits([0.0] * 65 + [2.0, 0.0])})()

    model = ParityFailureModel()
    progress: list[tuple[int, int]] = []

    def on_progress(completed: int, total: int) -> None:
        progress.append((completed, total))
        reload_evidence = cast(dict[str, object], evidence["reload"])
        reload_rows = cast(list[object], reload_evidence["outputs"])
        assert len(reload_rows) == completed
        assert reload_evidence["winner_match_count"] == 0
        assert reload_evidence["max_candidate_logit_difference"] == 2.0

    with pytest.raises(ValueError, match="reload winner differs at"):
        scoring_api.score_presentations(
            model,
            prepared.reload_items,
            presentations,
            torch,
            ledger,
            evidence,
            "reload_parity",
            on_progress=on_progress,
        )

    reload_evidence = cast(dict[str, object], evidence["reload"])
    reload_rows = cast(list[dict[str, object]], reload_evidence["outputs"])
    first_order = cast(list[str], presentations[0]["order_ids"])
    assert model.calls == 1
    assert len(reload_rows) == 1
    assert reload_rows[0]["winner_option_id"] == first_order[0]
    assert reload_evidence["winner_match_count"] == 0
    assert reload_evidence["max_candidate_logit_difference"] == 2.0
    assert progress == [(1, 2)]
    snapshot = ledger.snapshot()
    counts = cast(dict[str, object], snapshot["completed_forward_counts"])
    assert counts["reload_parity"] == 1


def test_evaluation_failure_preserves_prior_output_prefix_and_pending_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, examples = _toy_payload()
    _install_toy_payload(monkeypatch, payload, examples)
    prepared = scoring_api.prepare_requests(payload, CharacterTokenizer())
    presentations = cast(list[dict[str, object]], payload["evaluation"])
    ledger = _ledger_for_evaluation(prepared.evaluation_items)
    evidence: dict[str, object] = {"outputs": []}
    torch = FakeTorch()

    class SecondCallFails:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, **_kwargs: object) -> object:
            self.calls += 1
            if self.calls == 2:
                raise OSError("synthetic second-row failure")
            return type("Output", (), {"logits": FakeLogits([0.0] * 65 + [1.0, 1.0])})()

    model = SecondCallFails()
    progress: list[tuple[int, int]] = []
    with pytest.raises(OSError, match="synthetic second-row failure"):
        scoring_api.score_presentations(
            model,
            prepared.evaluation_items,
            presentations,
            torch,
            ledger,
            evidence,
            "final_evaluation",
            on_progress=lambda completed, total: progress.append((completed, total)),
        )

    assert model.calls == 2
    outputs = cast(list[dict[str, object]], evidence["outputs"])
    assert len(outputs) == 1
    assert progress == [(1, 2)]
    assert evidence == ledger.snapshot() | {"outputs": evidence["outputs"]}
    counts = cast(dict[str, object], evidence["completed_forward_counts"])
    pending = cast(dict[str, object], evidence["pending_forward"])
    assert counts["final_evaluation"] == 1
    assert pending["index"] == 1
