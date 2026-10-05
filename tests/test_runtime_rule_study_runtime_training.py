from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from experiments import mixture_training_runtime_helpers as runtime_helpers
from experiments import mixture_training_runtime_scoring as adapter_scoring
from experiments import modal_train_rehearsal as training_helpers
from experiments import runtime_rule_study_data as study_data
from experiments import runtime_rule_study_payloads as payload_api
from experiments import runtime_rule_study_progress as progress_api
from experiments import runtime_rule_study_provenance as provenance_api
from experiments import runtime_rule_study_runtime_scoring as runtime_scoring
from experiments import runtime_rule_study_runtime_training as runtime_training
from experiments.mixture_training_contracts import canonical_json
from experiments.runtime_rule_study_contracts import SELECTED_TENSOR_SHA256
from experiments.runtime_rule_study_training_evidence import (
    initial_training_evidence,
    validate_training_evidence,
)
from reflex_decisions.schema import DecisionRequest, Option


def _minimal_payload() -> dict[str, object]:
    spec = {"input_tokens": 1, "input_ids_sha256": "a" * 64}
    return {
        "role": "continued_practice",
        "run_id": "study-control",
        "training_specs": [dict(spec) for _ in range(1_344)],
        "evaluation_specs": [dict(spec) for _ in range(3_946)],
        "reload_specs": [dict(spec) for _ in range(32)],
    }


def _initial_result() -> dict[str, object]:
    return {
        "phase": "preflight",
        "provenance": {},
        "evidence": {
            "selected_adapter_identity": {},
            "training": initial_training_evidence(),
        },
    }


def test_train_adapter_rejects_a_noncanonical_run_directory_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    payload = {"role": "continued_practice", "run_id": "study-control"}
    monkeypatch.setattr(payload_api, "validate_payload", lambda _value: payload)
    result = {
        "phase": "preflight",
        "provenance": {},
        "evidence": {
            "selected_adapter_identity": {},
            "training": initial_training_evidence(),
        },
    }

    class FakeModel:
        changes = 0

        def train(self) -> None:
            self.changes += 1

    model = FakeModel()
    optimizer_calls: list[tuple[object, ...]] = []
    torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=lambda *args, **kwargs: optimizer_calls.append((args, kwargs)))
    )

    with pytest.raises(ValueError, match="run directory"):
        runtime_training.train_adapter(
            payload,
            result,
            model,
            object(),
            {},
            tmp_path / "study-control",
            torch,
            object(),
            object(),
        )

    assert model.changes == 0
    assert optimizer_calls == []


def test_train_adapter_rejects_prior_training_evidence_instead_of_resuming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"role": "continued_practice", "run_id": "study-control"}
    monkeypatch.setattr(payload_api, "validate_payload", lambda _value: payload)
    monkeypatch.setattr(
        provenance_api,
        "validate_provenance",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        provenance_api,
        "validate_selected_adapter_identity",
        lambda *_args, **_kwargs: {"tensor_sha256": "a" * 64, "snapshot": {}},
    )
    training = initial_training_evidence()
    training["optimizer_updates_completed"] = 1
    result = {
        "phase": "training",
        "provenance": {},
        "evidence": {"selected_adapter_identity": {}, "training": training},
    }
    optimizer_calls: list[tuple[object, ...]] = []
    torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=lambda *args, **kwargs: optimizer_calls.append((args, kwargs)))
    )

    with pytest.raises(ValueError, match="fresh training evidence"):
        runtime_training.train_adapter(
            payload,
            result,
            object(),
            object(),
            {},
            "/artifacts/runs/study-control",
            torch,
            object(),
            object(),
        )

    assert optimizer_calls == []


def test_train_adapter_requires_the_fresh_zero_call_ledger_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _minimal_payload()
    monkeypatch.setattr(payload_api, "validate_payload", lambda _value: payload)
    monkeypatch.setattr(provenance_api, "validate_provenance", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        provenance_api,
        "validate_selected_adapter_identity",
        lambda *_args, **_kwargs: {"tensor_sha256": "a" * 64, "snapshot": {}},
    )
    specs = {
        "training": payload["training_specs"],
        "final_evaluation": payload["evaluation_specs"],
        "reload_parity": payload["reload_specs"],
    }
    ledger = progress_api.ForwardLedger("continued_practice", specs)
    ledger.begin("training", payload["training_specs"][0])
    ledger.complete()
    optimizer_calls: list[tuple[object, ...]] = []
    torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=lambda *args, **kwargs: optimizer_calls.append((args, kwargs)))
    )

    with pytest.raises(ValueError, match="fresh zero-call ledger"):
        runtime_training.train_adapter(
            payload,
            _initial_result(),
            object(),
            object(),
            {},
            "/artifacts/runs/study-control",
            torch,
            object(),
            ledger,
        )

    assert optimizer_calls == []


def test_train_adapter_rejects_prepared_training_items_out_of_schedule_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _minimal_payload()
    options = (Option(id="a", label="A"), Option(id="b", label="B"))
    requests = tuple(
        DecisionRequest(
            context=f"Self-authored schedule request {index}.",
            question="Which option applies?",
            options=options,
        )
        for index in range(2)
    )
    examples = tuple(
        study_data.TrainingExample(
            record_id=f"record-{index % 2}",
            request=requests[index % 2],
            gold_option_id=options[index % 2].id,
            gold_index=index % 2,
        )
        for index in range(1_344)
    )
    training_specs = [
        {"input_tokens": 1, "input_ids_sha256": "a" * 64, "record_id": example.record_id}
        for example in examples
    ]
    payload["training_specs"] = training_specs
    items = [
        SimpleNamespace(
            category="training",
            index=index,
            request=example.request,
            expected_spec_json=canonical_json(training_specs[index]).decode("utf-8"),
        )
        for index, example in enumerate(examples)
    ]
    items[0].request = requests[1]
    prepared = SimpleNamespace(training_items=tuple(items), training_examples=examples)
    monkeypatch.setattr(payload_api, "validate_payload", lambda _value: payload)
    monkeypatch.setattr(payload_api, "training_examples", lambda _value: examples)
    monkeypatch.setattr(provenance_api, "validate_provenance", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        provenance_api,
        "validate_selected_adapter_identity",
        lambda *_args, **_kwargs: {"tensor_sha256": "a" * 64, "snapshot": {}},
    )
    ledger = progress_api.ForwardLedger(
        "continued_practice",
        {
            "training": training_specs,
            "final_evaluation": payload["evaluation_specs"],
            "reload_parity": payload["reload_specs"],
        },
    )
    optimizer_calls: list[tuple[object, ...]] = []
    torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=lambda *args, **kwargs: optimizer_calls.append((args, kwargs)))
    )

    with pytest.raises(ValueError, match="prepared training item"):
        runtime_training.train_adapter(
            payload,
            _initial_result(),
            object(),
            prepared,
            {},
            "/artifacts/runs/study-control",
            torch,
            object(),
            ledger,
        )

    assert optimizer_calls == []
    assert ledger.snapshot()["completed_forward_counts"]["total"] == 0


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value

    def detach(self):
        return self

    def abs(self):
        return _FakeScalar(abs(self.value))

    def sum(self):
        return self


class _FakeParameter:
    def __init__(self, name: str, *, requires_grad: bool, value: float = 0.0) -> None:
        self.name = name
        self.requires_grad = requires_grad
        self.value = value
        self.grad: _FakeScalar | None = None


class _FakeState:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeDetachedLoss:
    def __init__(self, model: _FakeModel, value: float) -> None:
        self.model = model
        self.value = value

    def item(self) -> float:
        if self.model.fail_loss_row_after_step and self.model.optimizer.step_calls:
            raise RuntimeError("injected loss-row conversion failure")
        return self.value


class _FakeLoss:
    def __init__(self, model: _FakeModel, parameter: _FakeParameter, value: float) -> None:
        self.model = model
        self.parameter = parameter
        self.value = value
        self.scale = 1.0

    def item(self) -> float:
        return self.value

    def detach(self):
        return _FakeDetachedLoss(self.model, self.value)

    def __truediv__(self, denominator: int):
        assert denominator == 4
        scaled = _FakeLoss(self.model, self.parameter, self.value)
        scaled.scale = self.scale / denominator
        return scaled

    def backward(self) -> None:
        contribution = math.inf if self.model.bad_gradient else self.scale
        existing = 0.0 if self.parameter.grad is None else self.parameter.grad.value
        self.parameter.grad = _FakeScalar(existing + contribution)


class _FakeLogits:
    def __init__(self, model: _FakeModel, parameter: _FakeParameter) -> None:
        self.model = model
        self.parameter = parameter

    def unsqueeze(self, _dimension: int):
        return self


class _FakeOptimizer:
    def __init__(self, model: _FakeModel, parameters, kwargs) -> None:
        self.model = model
        self.params = list(parameters)
        self.kwargs = kwargs
        self.param_groups = [
            {
                "params": self.params,
                "lr": kwargs["lr"],
                "betas": kwargs["betas"],
                "eps": kwargs["eps"],
                "weight_decay": kwargs["weight_decay"],
            }
        ]
        self.state: dict[object, object] = {}
        self.step_calls = 0
        self.step_gradient_values: list[float] = []

    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none is True
        for parameter in self.params:
            parameter.grad = None

    def step(self) -> None:
        self.step_calls += 1
        gradient = self.params[0].grad
        assert gradient is not None
        self.step_gradient_values.append(gradient.value)
        if not self.model.no_change:
            self.params[0].value -= self.kwargs["lr"] * gradient.value


class _FakeModel:
    def __init__(
        self,
        *,
        no_change: bool = False,
        fail_loss_row_after_step: bool = False,
        bad_gradient: bool = False,
    ) -> None:
        self.adapter = _FakeParameter("layer.lora_B.weight", requires_grad=True)
        self.base = _FakeParameter("embed.weight", requires_grad=False)
        self.no_change = no_change
        self.fail_loss_row_after_step = fail_loss_row_after_step
        self.bad_gradient = bad_gradient
        self.optimizer = SimpleNamespace(step_calls=0)
        self.training = False

    def parameters(self):
        return iter((self.adapter, self.base))

    def named_parameters(self):
        return iter(((self.adapter.name, self.adapter), (self.base.name, self.base)))

    def train(self) -> None:
        self.training = True


def _valid_inventory() -> dict[str, object]:
    names = [
        f"layer.{index}.lora_{matrix}.default.weight"
        for index in range(60)
        for matrix in ("A", "B")
    ]
    shapes = {name: [1] for name in names}
    shapes[names[0]] = [2_015_113]
    return {
        "module_count": 60,
        "module_names": [f"layer.{index}" for index in range(60)],
        "adapter_tensor_count": 120,
        "adapter_parameter_count": 2_015_232,
        "adapter_tensor_names": names,
        "adapter_tensor_shapes": shapes,
        "adapter_tensor_dtypes": {name: "torch.float32" for name in names},
        "trainable_parameter_count": 2_015_232,
        "base_parameters_frozen_bf16": True,
    }


def _fake_training_setup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    no_change: bool = False,
    fail_save_update: int | None = None,
    fail_hash_update: int | None = None,
    fail_loss_row_after_step: bool = False,
    fail_forward_at: int | None = None,
    bad_gradient: bool = False,
    inject_base_gradient: bool = False,
):
    payload = _minimal_payload()
    options = (Option(id="a", label="A"), Option(id="b", label="B"))
    request = DecisionRequest(
        context="Self-authored schedule request.",
        question="Which option applies?",
        options=options,
    )
    examples = tuple(
        SimpleNamespace(
            record_id="record",
            request=request,
            gold_option_id=options[index % 2].id,
            gold_index=index % 2,
        )
        for index in range(1_344)
    )
    training_specs = [
        {"input_tokens": 1, "input_ids_sha256": "a" * 64, "record_id": "record"} for _ in examples
    ]
    payload["training_specs"] = training_specs
    items = tuple(
        SimpleNamespace(
            category="training",
            index=index,
            request=request,
            compiled=object(),
            expected_spec_json=canonical_json(training_specs[index]).decode("utf-8"),
        )
        for index in range(len(examples))
    )
    prepared = SimpleNamespace(training_items=items, training_examples=examples)
    monkeypatch.setattr(payload_api, "validate_payload", lambda _value: payload)
    monkeypatch.setattr(payload_api, "training_examples", lambda _value: examples)
    monkeypatch.setattr(provenance_api, "validate_provenance", lambda *_args, **_kwargs: {})
    selected_snapshot = {
        "update": 378,
        "path": "/artifacts/sources/adapter-update-378",
        "files_sha256": {
            "adapter_model.safetensors": "a" * 64,
            "adapter_config.json": "b" * 64,
        },
    }
    selected_identity = {
        "snapshot": selected_snapshot,
        "tensor_sha256": SELECTED_TENSOR_SHA256,
        "adapter_dtype": "torch.float32",
    }
    monkeypatch.setattr(
        provenance_api,
        "validate_selected_adapter_identity",
        lambda *_args, **_kwargs: selected_identity,
    )
    specs = {
        "training": training_specs,
        "final_evaluation": payload["evaluation_specs"],
        "reload_parity": payload["reload_specs"],
    }
    ledger = progress_api.ForwardLedger("continued_practice", specs)
    model = _FakeModel(
        no_change=no_change,
        fail_loss_row_after_step=fail_loss_row_after_step,
        bad_gradient=bad_gradient,
    )
    optimizer_instances: list[_FakeOptimizer] = []

    def make_optimizer(parameters, **kwargs):
        optimizer = _FakeOptimizer(model, parameters, kwargs)
        optimizer_instances.append(optimizer)
        model.optimizer = optimizer
        return optimizer

    seed_calls: list[tuple[str, int]] = []
    torch = SimpleNamespace(
        long=object(),
        optim=SimpleNamespace(AdamW=make_optimizer),
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                cross_entropy=lambda logits, target: _FakeLoss(
                    logits.model, logits.parameter, 2.0 + target[0]
                ),
            ),
            utils=SimpleNamespace(
                clip_grad_norm_=lambda parameters, _limit: _FakeScalar(
                    math.sqrt(
                        sum(parameter.grad.value**2 for parameter in parameters if parameter.grad)
                    )
                ),
            ),
        ),
        cuda=SimpleNamespace(manual_seed_all=lambda seed: seed_calls.append(("cuda", seed))),
        manual_seed=lambda seed: seed_calls.append(("torch", seed)),
        tensor=lambda values, **_kwargs: values,
        isfinite=lambda scalar: SimpleNamespace(
            all=lambda: SimpleNamespace(item=lambda: math.isfinite(scalar.value)),
            item=lambda: math.isfinite(scalar.value),
        ),
    )
    monkeypatch.setattr(
        training_helpers,
        "_validate_adapter_inventory",
        lambda *_args, **_kwargs: _valid_inventory(),
    )
    monkeypatch.setattr(
        training_helpers,
        "_adapter_state_changes",
        lambda before, after, _torch: {
            "changed_tensor_count": int(
                before["layer.lora_B.weight"].value != after["layer.lora_B.weight"].value
            ),
            "changed_tensor_names": (
                ["layer.0.lora_B.default.weight"]
                if before["layer.lora_B.weight"].value != after["layer.lora_B.weight"].value
                else []
            ),
        },
    )
    saved_updates: list[int] = []

    def save_snapshot(_model, run_dir, update):
        saved_updates.append(update)
        if update == fail_save_update:
            raise OSError("injected snapshot save failure")
        return {
            "update": update,
            "path": str(run_dir / f"adapter-update-{update:03d}"),
            "files_sha256": {
                "adapter_model.safetensors": "a" * 64,
                "adapter_config.json": "b" * 64,
            },
        }

    monkeypatch.setattr(training_helpers, "_save_adapter_snapshot", save_snapshot)
    monkeypatch.setattr(
        adapter_scoring,
        "_state_dict",
        lambda _model, _peft: {"layer.lora_B.weight": _FakeState(model.adapter.value)},
    )
    verified_snapshots: list[tuple[int, str]] = []

    def verify_saved_adapter(_model, snapshot, digest, _peft, _torch):
        verified_snapshots.append((snapshot["update"], digest))
        return {}

    monkeypatch.setattr(adapter_scoring, "_verify_saved_adapter", verify_saved_adapter)
    hash_checks: list[int] = []

    def verify_snapshot_hashes(snapshot):
        hash_checks.append(snapshot["update"])
        if snapshot["update"] == fail_hash_update:
            raise OSError("injected snapshot hash verification failure")
        return snapshot["files_sha256"]

    monkeypatch.setattr(adapter_scoring, "_snapshot_hashes", verify_snapshot_hashes)
    forward_indices: list[int] = []

    def score_forward(_model, item, _torch, active_ledger, evidence, category):
        forward_indices.append(item.index)
        spec = json.loads(item.expected_spec_json)
        active_ledger.begin(category, spec)
        evidence.update(active_ledger.snapshot())
        if item.index == fail_forward_at:
            raise RuntimeError("injected model-forward failure")
        active_ledger.complete()
        evidence.update(active_ledger.snapshot())
        if inject_base_gradient and item.index == 3:
            model.base.grad = _FakeScalar(1.0)
        return _FakeLogits(model, model.adapter)

    monkeypatch.setattr(runtime_scoring, "score_forward", score_forward)
    monkeypatch.setattr(
        runtime_helpers,
        "tensor_state_sha256",
        lambda state, *, torch_module: (
            SELECTED_TENSOR_SHA256 if state["layer.lora_B.weight"].value == 0.0 else "c" * 64
        ),
    )
    state = {"layer.lora_B.weight": _FakeState(0.0)}
    result = _initial_result()
    return SimpleNamespace(
        payload=payload,
        examples=examples,
        prepared=prepared,
        model=model,
        torch=torch,
        ledger=ledger,
        initial_state=state,
        result=result,
        optimizer_instances=optimizer_instances,
        seed_calls=seed_calls,
        saved_updates=saved_updates,
        fail_hash_update=fail_hash_update,
        forward_indices=forward_indices,
        selected_snapshot=selected_snapshot,
        verified_snapshots=verified_snapshots,
        hash_checks=hash_checks,
    )


def _train_fake_runtime(runtime, *, on_progress=None):
    return runtime_training.train_adapter(
        runtime.payload,
        runtime.result,
        runtime.model,
        runtime.prepared,
        runtime.initial_state,
        "/artifacts/runs/study-control",
        runtime.torch,
        object(),
        runtime.ledger,
        on_progress=on_progress,
    )


def test_fixed_loop_scales_microbatch_gradients_and_validates_final_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_training_setup(monkeypatch)
    progress_updates: list[int] = []

    final_state = _train_fake_runtime(
        runtime,
        on_progress=lambda current, _run_dir: progress_updates.append(
            current["evidence"]["training"]["optimizer_updates_completed"]
        ),
    )

    optimizer = runtime.optimizer_instances[0]
    assert optimizer.kwargs == {
        "lr": 5e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
    }
    assert optimizer.state == {}
    assert optimizer.step_gradient_values == [1.0] * 336
    assert runtime.seed_calls == [("torch", 20261009), ("cuda", 20261009)]
    assert runtime.saved_updates == [168, 336]
    assert runtime.hash_checks == [168, 336]
    assert runtime.verified_snapshots == [
        (378, SELECTED_TENSOR_SHA256),
        (336, "c" * 64),
    ]
    assert progress_updates == [*range(16, 169, 16), 168, *range(176, 337, 16)]
    assert runtime.result["evidence"]["completed_forward_counts"]["training"] == 1_344
    assert runtime.result["evidence"]["pending_forward"] is None
    assert final_state["layer.lora_B.weight"].value == pytest.approx(-5e-5 * 336)
    evidence = runtime.result["evidence"]["training"]
    validated = validate_training_evidence(
        evidence,
        run_id="study-control",
        completed_training_forwards=runtime.ledger.snapshot()["completed_forward_counts"][
            "training"
        ],
        passed=True,
    )
    assert validated["optimizer_updates_completed"] == 336
    assert len(validated["training_step_losses"]) == 336
    assert validated["adapter_paths"].keys() == {"168", "336"}


def test_returned_optimizer_step_is_recorded_before_loss_row_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_training_setup(monkeypatch, fail_loss_row_after_step=True)

    with pytest.raises(RuntimeError, match="loss-row conversion"):
        _train_fake_runtime(runtime)

    evidence = runtime.result["evidence"]["training"]
    assert evidence["optimizer_updates_completed"] == 1
    assert evidence["training_step_losses"] == []
    assert runtime.optimizer_instances[0].step_calls == 1
    assert runtime.ledger.snapshot()["completed_forward_counts"]["training"] == 4


def test_midpoint_snapshot_failure_prevents_the_next_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_training_setup(monkeypatch, fail_save_update=168)

    with pytest.raises(OSError, match="snapshot save failure"):
        _train_fake_runtime(runtime)

    evidence = runtime.result["evidence"]["training"]
    assert evidence["optimizer_updates_completed"] == 168
    assert len(evidence["training_step_losses"]) == 168
    assert runtime.saved_updates == [168]
    assert len(runtime.forward_indices) == 168 * 4


def test_hash_failure_preserves_returned_midpoint_snapshot_and_stops_before_update_169(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_training_setup(monkeypatch, fail_hash_update=168)

    with pytest.raises(OSError, match="snapshot hash verification failure"):
        _train_fake_runtime(runtime)

    evidence = runtime.result["evidence"]["training"]
    assert evidence["optimizer_updates_completed"] == 168
    assert evidence["adapter_paths"]["168"] == {
        "update": 168,
        "path": "/artifacts/runs/study-control/adapter-update-168",
        "files_sha256": {
            "adapter_model.safetensors": "a" * 64,
            "adapter_config.json": "b" * 64,
        },
    }
    assert runtime.saved_updates == [168]
    assert len(runtime.forward_indices) == 168 * 4


def test_failed_microbatch_preserves_pending_forward_without_completing_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_training_setup(monkeypatch, fail_forward_at=2)

    with pytest.raises(RuntimeError, match="model-forward failure"):
        _train_fake_runtime(runtime)

    progress = runtime.ledger.snapshot()
    assert progress["completed_forward_counts"]["training"] == 2
    assert progress["pending_forward"]["index"] == 2
    assert runtime.result["evidence"]["pending_forward"]["index"] == 2
    assert runtime.result["evidence"]["training"]["optimizer_updates_completed"] == 0
    assert runtime.optimizer_instances[0].step_calls == 0


def test_base_gradient_is_recorded_before_training_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_training_setup(monkeypatch, inject_base_gradient=True)

    with pytest.raises(RuntimeError, match="base parameter received a gradient"):
        _train_fake_runtime(runtime)

    assert runtime.result["evidence"]["training"]["base_gradients_none"] is False
    assert runtime.optimizer_instances[0].step_calls == 0


def test_no_change_evidence_is_retained_before_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_training_setup(monkeypatch, no_change=True)

    with pytest.raises(RuntimeError, match="did not change"):
        _train_fake_runtime(runtime)

    evidence = runtime.result["evidence"]["training"]
    assert evidence["adapter_update"] == {
        "changed_tensor_count": 0,
        "changed_tensor_names": [],
    }
    assert evidence["final_tensor_sha256"] == SELECTED_TENSOR_SHA256
    validate_training_evidence(
        evidence,
        run_id="study-control",
        completed_training_forwards=1_344,
        passed=False,
    )


def test_nonfinite_adapter_gradient_stops_before_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fake_training_setup(monkeypatch, bad_gradient=True)

    with pytest.raises(RuntimeError, match="non-finite value"):
        _train_fake_runtime(runtime)

    assert runtime.ledger.snapshot()["completed_forward_counts"]["training"] == 4
    assert runtime.optimizer_instances[0].step_calls == 0
    assert runtime.result["evidence"]["training"]["optimizer_updates_completed"] == 0
