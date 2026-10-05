from __future__ import annotations

import copy

import pytest


@pytest.fixture(scope="module")
def valid_unchanged_payload() -> dict[str, object]:
    from pathlib import Path
    from types import SimpleNamespace

    from experiments import adapter_transfer_contracts
    from experiments import analyze_targeted_training as analysis
    from experiments import targeted_training_core as core
    from reflex_decisions.schema import DecisionRequest

    class Tokenizer:
        def encode(self, text, *, add_special_tokens=False):
            values = list(range(len(text)))
            if text[-1:] in {"A", "B"}:
                values[-1] = ord(text[-1])
            return values

    root = Path(__file__).resolve().parents[1]
    selection, _ = adapter_transfer_contracts.load_selection(root)
    request = DecisionRequest.model_validate(
        {
            "context": "toy",
            "question": "Choose",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        }
    )
    panel = []
    index = 0
    for dataset_id, count in analysis.TASK_COUNTS.items():
        for _ in range(count):
            panel.append(
                {
                    "presentation_id": f"p-{index}",
                    "record_id": f"r-{index}",
                    "dataset_id": dataset_id,
                    "source_group_id": f"g-{index}",
                    "request_hash": request.request_hash,
                    "order_index": 0,
                    "order_ids": ["a", "b"],
                    "request": request.model_dump(mode="json"),
                }
            )
            index += 1
    inputs = SimpleNamespace(
        presentations=tuple(panel),
        schedules={},
        selection=selection,
        strata=analysis.TASK_STRATA,
        file_sha256={"toy": "0" * 64},
        audit={"toy": 1},
    )
    return core.build_role_payload(
        inputs,
        "unchanged",
        Tokenizer(),
        run_id="toy-run",
        source_pins=core.targeted_source_manifest(root),
    )


def test_unchanged_role_payload_has_exact_counts_and_token_accounting(
    valid_unchanged_payload,
) -> None:
    from experiments import targeted_training_core as core

    checked = core.validate_role_payload(valid_unchanged_payload)

    assert checked["forward_caps"]["total"] == 6182
    assert checked["training"] == ()
    assert checked["reload_presentation_ids"] == ()
    assert checked["compiled_input_token_counts"]["total"] > 0


@pytest.fixture(scope="module")
def valid_unchanged_result(valid_unchanged_payload) -> dict[str, object]:
    from experiments import targeted_training_core as core
    from experiments.mixture_training_contracts import RUNTIME_VERSION_PINS
    from experiments.targeted_training_contracts import (
        MODEL_ID,
        MODEL_REVISION,
        SELECTED_ADAPTER_TENSOR_SHA256,
        TOKENIZER_FILE_SHA256,
    )

    payload = valid_unchanged_payload
    outputs = []
    for row, compiled in zip(
        payload["evaluation_presentations"], payload["compiled"]["final_evaluation"], strict=True
    ):
        outputs.append(
            {
                key: row[key]
                for key in (
                    "presentation_id",
                    "record_id",
                    "dataset_id",
                    "source_group_id",
                    "request_hash",
                    "order_index",
                    "order_ids",
                )
            }
            | {
                "candidate_logits": [1.0, 0.0],
                "winner_option_id": "a",
                "input_tokens": compiled["input_tokens"],
                "prompt_sha256": compiled["prompt_sha256"],
            }
        )
    return {
        "status": "passed",
        "experiment_id": core.EXPERIMENT_ID,
        "run_id": payload["run_id"],
        "role": "unchanged",
        "payload_sha256": payload["payload_sha256"],
        "outputs": outputs,
        "compiled": payload["compiled"],
        "execution": {
            "completed": payload["forward_caps"],
            "attempted_unknown": {"training": 0, "final_evaluation": 0, "reload_parity": 0},
            "not_started": {"training": 0, "final_evaluation": 0, "reload_parity": 0},
            "input_token_counts": payload["compiled_input_token_counts"],
        },
        "provenance": {
            "source_pins_sha256": core.canonical_sha256(payload["source_pins"]),
            "base_model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "tokenizer_file_sha256": TOKENIZER_FILE_SHA256,
                "effective_dtype": "torch.bfloat16",
                "device": "cuda:0",
                "layer_count": 24,
                "tied_embeddings": True,
                "no_meta_parameters": True,
                "attention_implementation": "eager",
                "use_kernels": False,
                "use_hub_kernels": "NO",
                "versions": {
                    name: version
                    for name, version in RUNTIME_VERSION_PINS.items()
                    if name != "Pillow"
                },
            },
            "adapter_tensor_sha256": SELECTED_ADAPTER_TENSOR_SHA256,
            "adapter_inventory": {
                "module_count": 60,
                "adapter_tensor_count": 120,
                "adapter_parameter_count": 2015232,
                "trainable_parameter_count": 0,
                "base_parameters_frozen_bf16": True,
                "adapter_tensor_names": [f"tensor-{i}" for i in range(120)],
                "adapter_tensor_dtypes": {f"tensor-{i}": "torch.float32" for i in range(120)},
            },
        },
        "evidence": {
            "optimizer": None,
            "saved_adapter": None,
            "reload_parity": None,
            "adapter_update": None,
            "training_step_losses": [],
        },
    }


def test_completed_result_rejects_wrong_compiled_score_prompt(
    valid_unchanged_payload, valid_unchanged_result
) -> None:
    from experiments import targeted_training_core as core

    assert (
        core.validate_completed_result(valid_unchanged_payload, valid_unchanged_result)["status"]
        == "passed"
    )
    result = copy.deepcopy(valid_unchanged_result)
    result["outputs"][0]["prompt_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="prompt"):
        core.validate_completed_result(valid_unchanged_payload, result)


@pytest.fixture(scope="module")
def valid_control_pair(valid_unchanged_payload, valid_unchanged_result):
    from experiments import analyze_targeted_training as analysis
    from experiments import targeted_training_core as core
    from experiments.targeted_training_contracts import (
        BETAS,
        EPSILON,
        LEARNING_RATE,
        MAX_GRADIENT_NORM,
        TRAINING_SEED,
        WEIGHT_DECAY,
    )

    class Tokenizer:
        def encode(self, text, *, add_special_tokens=False):
            values = list(range(len(text)))
            if text[-1:] in {"A", "B"}:
                values[-1] = ord(text[-1])
            return values

    payload = copy.deepcopy(valid_unchanged_payload)
    payload["role"] = "control"
    request_row = payload["evaluation_presentations"][0]
    training = []
    for index in range(4800):
        update, micro = index // 4 + 1, index % 4
        training.append(
            {
                key: request_row[key]
                for key in (
                    "record_id",
                    "dataset_id",
                    "source_group_id",
                    "request_hash",
                    "order_index",
                    "order_ids",
                    "request",
                )
            }
            | {
                "presentation_id": f"targeted-reasoning-v1:control:update-{update}:micro-{micro}",
                "gold_option_id": "a",
                "gold_index": 0,
                "update": update,
                "microbatch_index": micro,
            }
        )
    payload["training"] = tuple(training)
    payload["compiled"]["training"] = core.compile_presentations(tuple(training), Tokenizer())
    payload["forward_caps"] = core.expected_forward_counts("control")
    retention = [
        row["presentation_id"]
        for row in sorted(
            payload["evaluation_presentations"],
            key=lambda row: (
                row["dataset_id"],
                row["record_id"],
                row["order_index"],
                row["presentation_id"],
            ),
        )
        if row["dataset_id"] in analysis.RETENTION_COUNTS
    ][:32]
    payload["reload_presentation_ids"] = tuple(retention)
    compiled_by_id = {
        row["presentation_id"]: row for row in payload["compiled"]["final_evaluation"]
    }
    counts = {
        "training": sum(row["input_tokens"] for row in payload["compiled"]["training"]),
        "final_evaluation": valid_unchanged_payload["compiled_input_token_counts"][
            "final_evaluation"
        ],
        "reload_parity": sum(
            compiled_by_id[identifier]["input_tokens"] for identifier in retention
        ),
    }
    counts["total"] = sum(counts.values())
    payload["compiled_input_token_counts"] = counts
    payload["compiled_input_tokens"] = counts["total"]
    payload["payload_sha256"] = core.canonical_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )
    result = copy.deepcopy(valid_unchanged_result)
    result["role"] = "control"
    result["payload_sha256"] = payload["payload_sha256"]
    result["compiled"] = payload["compiled"]
    result["execution"]["completed"] = payload["forward_caps"]
    result["execution"]["input_token_counts"] = counts
    result["provenance"]["adapter_inventory"]["trainable_parameter_count"] = 2015232
    final_by_id = {row["presentation_id"]: row for row in result["outputs"]}
    result["evidence"] = {
        "optimizer": {
            "lr": LEARNING_RATE,
            "betas": list(BETAS),
            "eps": EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "max_gradient_norm": MAX_GRADIENT_NORM,
            "training_seed": TRAINING_SEED,
            "initial_state_empty": True,
            "scheduler": None,
            "updates": 1200,
            "microbatches_per_update": 4,
            "base_gradients_absent": True,
        },
        "training_step_losses": [{"update": update, "mean_loss": 1.0} for update in range(1, 1201)],
        "adapter_update": {"changed_tensor_count": 1, "changed_tensor_names": ["tensor-0"]},
        "saved_adapter": {
            "update": 1200,
            "path": "/artifacts/runs/toy-run-control/adapter-update-1200",
            "files_sha256": {
                "adapter_model.safetensors": "a" * 64,
                "adapter_config.json": "b" * 64,
            },
            "files_size_bytes": {"adapter_model.safetensors": 1, "adapter_config.json": 1},
        },
        "reload_parity": {
            "tensor_values_exact": True,
            "winner_match": True,
            "max_candidate_logit_delta": 0.0,
            "presentation_ids": list(retention),
            "pre_save_tensor_sha256": "c" * 64,
            "reloaded_tensor_sha256": "c" * 64,
            "reload_outputs": [copy.deepcopy(final_by_id[identifier]) for identifier in retention],
        },
    }
    return payload, result


def test_expected_forward_counts_match_the_frozen_three_role_budget() -> None:
    from experiments import targeted_training_core as core

    assert core.expected_forward_counts(core.ROLE_UNCHANGED) == {
        "training": 0,
        "final_evaluation": 6182,
        "reload_parity": 0,
        "total": 6182,
    }
    assert core.expected_forward_counts(core.ROLE_CONTROL) == {
        "training": 4800,
        "final_evaluation": 6182,
        "reload_parity": 32,
        "total": 11014,
    }
    assert core.expected_forward_counts(core.ROLE_TREATMENT) == {
        "training": 4800,
        "final_evaluation": 6182,
        "reload_parity": 32,
        "total": 11014,
    }
    assert core.MAX_TOTAL_FORWARDS == 28210


def test_validate_completed_counts_rejects_a_role_that_exceeds_its_cap() -> None:
    from experiments import targeted_training_core as core

    counts = core.expected_forward_counts(core.ROLE_CONTROL)
    counts["training"] += 1
    counts["total"] += 1

    import pytest

    with pytest.raises(ValueError, match="training"):
        core.validate_completed_counts(core.ROLE_CONTROL, counts)


def test_load_source_maps_preserves_separate_fresh_and_historical_allowlists() -> None:
    from pathlib import Path

    from experiments import targeted_training_core as core

    maps = core.load_source_maps(Path(__file__).resolve().parents[1])

    assert set(maps) == {"fresh", "historical"}
    assert len(maps["fresh"]) == 71
    assert len(maps["historical"]) == 60


def test_compile_presentations_rejects_overlong_prompts_without_truncation() -> None:
    from experiments import targeted_training_core as core

    class Tokenizer:
        def encode(self, text, *, add_special_tokens=False):
            return list(range(len(text)))

    row = {
        "presentation_id": "p-1",
        "request_hash": "not-used",
        "order_ids": ["a", "b"],
        "request": {
            "context": "x" * 3_000,
            "question": "q",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        },
    }

    import pytest

    with pytest.raises(ValueError, match="2048"):
        core.compile_presentations((row,), Tokenizer())


def test_validate_completed_result_rejects_trained_reload_score_drift(valid_control_pair) -> None:
    from experiments import targeted_training_core as core

    payload, valid_result = valid_control_pair
    assert core.validate_completed_result(payload, valid_result)["status"] == "passed"
    result = copy.deepcopy(valid_result)
    result["evidence"]["reload_parity"]["reload_outputs"][0]["candidate_logits"][0] += 1.1e-6

    with pytest.raises(ValueError, match="reload"):
        core.validate_completed_result(payload, result)


def test_completed_result_rejects_missing_actual_token_ledger(
    valid_unchanged_payload, valid_unchanged_result
) -> None:
    from experiments import targeted_training_core as core

    result = copy.deepcopy(valid_unchanged_result)
    del result["execution"]["input_token_counts"]

    with pytest.raises(ValueError, match="ledger|token"):
        core.validate_completed_result(valid_unchanged_payload, result)


def test_real_peft_inventory_names_bind_to_saved_state_change_names(valid_control_pair) -> None:
    from experiments import targeted_training_core as core

    payload, original = valid_control_pair
    result = copy.deepcopy(original)
    inventory = result["provenance"]["adapter_inventory"]
    names = [
        f"base_model.model.layers.{i}.self_attn.q_proj.lora_A.default.weight" for i in range(120)
    ]
    inventory["adapter_tensor_names"] = names
    inventory["adapter_tensor_dtypes"] = {name: "torch.float32" for name in names}
    result["evidence"]["adapter_update"]["changed_tensor_names"] = [
        names[0].replace(".lora_A.default.weight", ".lora_A.weight")
    ]
    assert core.validate_completed_result(payload, result)["status"] == "passed"
    result["evidence"]["adapter_update"]["changed_tensor_names"] = ["unrelated.weight"]
    with pytest.raises(ValueError, match="change proof"):
        core.validate_completed_result(payload, result)


def test_completed_result_rejects_runtime_package_drift(
    valid_unchanged_payload, valid_unchanged_result
) -> None:
    from experiments import targeted_training_core as core

    result = copy.deepcopy(valid_unchanged_result)
    result["provenance"]["base_model"]["versions"]["torch"] = "2.14.1"
    with pytest.raises(ValueError, match="package versions"):
        core.validate_completed_result(valid_unchanged_payload, result)
    result["provenance"]["base_model"]["versions"]["torch"] = "2.14.1+cu130"
    result["provenance"]["base_model"]["use_hub_kernels"] = "YES"
    with pytest.raises(ValueError, match="runtime"):
        core.validate_completed_result(valid_unchanged_payload, result)


def test_run_worker_retains_earlier_outputs_and_pending_failure(
    valid_unchanged_payload, valid_unchanged_result, monkeypatch, tmp_path
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace

    from experiments import targeted_training_runtime as runtime

    class Model:
        def eval(self):
            return self

    model = Model()
    inventory = valid_unchanged_result["provenance"]["adapter_inventory"]
    provenance = valid_unchanged_result["provenance"]["base_model"]
    peft = SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model)
    )
    scoring = SimpleNamespace(
        _verify_saved_adapter=lambda *_args: {},
        _scored_row=lambda row, _logits, compiled: {
            **{
                key: row[key]
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
            "candidate_logits": [1.0, 0.0],
            "winner_option_id": "a",
            "input_tokens": compiled["input_tokens"],
            "prompt_sha256": compiled["prompt_sha256"],
        },
    )
    torch = SimpleNamespace(
        manual_seed=lambda *_args: None,
        cuda=SimpleNamespace(manual_seed_all=lambda *_args: None),
        inference_mode=nullcontext,
    )
    deps = {
        "torch": torch,
        "functional": object(),
        "peft": peft,
        "AutoModelForCausalLM": object(),
        "AutoTokenizer": object(),
        "execution": SimpleNamespace(_base_model=lambda *_args: (model, object(), provenance)),
        "scoring": scoring,
        "rehearsal": SimpleNamespace(_validate_adapter_inventory=lambda *_args, **_kw: inventory),
        "helpers": object(),
    }
    seen = 0

    def forward(_scoring, _model, row, expected, _tokenizer, _torch, _progress, ledger, category):
        nonlocal seen
        seen += 1
        ledger.begin(category, row["presentation_id"])
        if seen == 3:
            ledger.fail_pending(category, row["presentation_id"])
            raise RuntimeError("synthetic forward failure")
        ledger.complete(category, input_tokens=expected["input_tokens"])
        return None, expected

    monkeypatch.setattr(runtime, "_checked_score", forward)
    result = runtime.run_worker(
        valid_unchanged_payload,
        source_root=tmp_path,
        artifact_root=tmp_path,
        preimport_verifier=lambda *_args: None,
        dependency_loader=lambda: deps,
    )
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "final_evaluation"
    assert len(result["outputs"]) == 2
    assert result["execution"]["completed"]["final_evaluation"] == 2
    assert result["execution"]["attempted_unknown"]["final_evaluation"] == 1
    assert result["provenance"]["adapter_inventory"] == inventory


def test_trained_worker_persists_failure_before_first_progress_checkpoint(
    valid_control_pair, monkeypatch, tmp_path
) -> None:
    import json
    from types import SimpleNamespace

    from experiments import mixture_training_runtime_helpers as helpers
    from experiments import targeted_training_runtime as runtime

    payload, valid_result = valid_control_pair

    class Model:
        def parameters(self):
            return []

        def train(self):
            return self

    class Optimizer:
        state = {}

        def zero_grad(self, **_kwargs):
            pass

    model = Model()
    torch = SimpleNamespace(
        manual_seed=lambda *_args: None,
        cuda=SimpleNamespace(manual_seed_all=lambda *_args: None),
        optim=SimpleNamespace(AdamW=lambda *_args, **_kwargs: Optimizer()),
    )
    deps = {
        "torch": torch,
        "functional": object(),
        "peft": SimpleNamespace(
            PeftModel=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model)
        ),
        "AutoModelForCausalLM": object(),
        "AutoTokenizer": object(),
        "execution": SimpleNamespace(
            _base_model=lambda *_args: (model, object(), valid_result["provenance"]["base_model"])
        ),
        "scoring": SimpleNamespace(_verify_saved_adapter=lambda *_args: {}),
        "rehearsal": SimpleNamespace(
            _validate_adapter_inventory=lambda *_args, **_kw: valid_result["provenance"][
                "adapter_inventory"
            ]
        ),
        "helpers": helpers,
    }

    def fail_forward(
        _scoring, _model, row, _expected, _tokenizer, _torch, _progress, ledger, category
    ):
        ledger.begin(category, row["presentation_id"])
        ledger.fail_pending(category, row["presentation_id"])
        raise RuntimeError("synthetic first-forward failure")

    monkeypatch.setattr(runtime, "_checked_score", fail_forward)
    result = runtime.run_worker(
        payload,
        source_root=tmp_path,
        artifact_root=tmp_path,
        preimport_verifier=lambda *_args: None,
        dependency_loader=lambda: deps,
    )
    path = tmp_path / "runs/toy-run-control/failure-receipt.json"
    persisted = json.loads(path.read_text())
    assert persisted["failure"]["stage"] == "training"
    assert persisted["execution"]["attempted_unknown"]["training"] == 1
    assert result["failure"] == persisted["failure"]


def test_role_payload_rejects_missing_digest_and_run_binding() -> None:
    from experiments import targeted_training_core as core

    with pytest.raises(ValueError, match="digest|run"):
        core.validate_role_payload({"experiment_id": core.EXPERIMENT_ID, "role": "unchanged"})
