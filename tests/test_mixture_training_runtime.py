from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

from experiments import mixture_training_contracts as contracts
from experiments import mixture_training_runtime_execution as runtime_execution

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_party_module_path(module_name: str) -> str | None:
    if module_name == "experiments" or module_name.startswith("experiments."):
        stem = PROJECT_ROOT / Path(*module_name.split("."))
    elif module_name == "reflex_decisions" or module_name.startswith("reflex_decisions."):
        stem = PROJECT_ROOT / "src" / Path(*module_name.split("."))
    else:
        return None

    for candidate in (Path(str(stem) + ".py"), stem / "__init__.py"):
        if candidate.is_file():
            return str(candidate.relative_to(PROJECT_ROOT))
    return None


def _source_module_name(relative_path: str) -> str | None:
    if relative_path.startswith("experiments/"):
        module = relative_path.removesuffix(".py").replace("/", ".")
    elif relative_path.startswith("src/reflex_decisions/"):
        module = relative_path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    else:
        return None
    return module.removesuffix(".__init__")


def _is_first_party_module(module_name: str) -> bool:
    return module_name in {"experiments", "reflex_decisions"} or module_name.startswith(
        ("experiments.", "reflex_decisions.")
    )


def test_runtime_training_constants_come_from_contracts() -> None:
    assert runtime_execution.LEARNING_RATE == contracts.LEARNING_RATE
    assert runtime_execution.WEIGHT_DECAY == contracts.WEIGHT_DECAY
    assert runtime_execution.MAX_GRADIENT_NORM == contracts.MAX_GRADIENT_NORM


def test_runtime_payload_partition_and_arm_schedules_include_only_approved_snli_arm() -> None:
    from experiments import mixture_training_core as core
    from experiments import mixture_training_data as data
    from reflex_decisions.data import DecisionRecord
    from reflex_decisions.schema import DecisionRequest, Option

    options = (Option(id="entails", label="Entails"), Option(id="neutral", label="Neutral"))

    def records(dataset_id: str, count: int, prefix: str) -> list[dict[str, object]]:
        return [
            DecisionRecord(
                record_id=f"{prefix}-{index}",
                dataset_id=dataset_id,
                source_group_id=f"{prefix}-group-{index}",
                request=DecisionRequest(
                    context=f"Runtime fixture context {prefix} {index}.",
                    question="Which option follows?",
                    options=options,
                ),
                answer_id=options[index % len(options)].id,
            ).model_dump(mode="json")
            for index in range(count)
        ]

    rows = []
    for dataset_id, count in data.REAL_TRAIN_DATASET_COUNTS.items():
        rows.extend(records(dataset_id, count, dataset_id))
    for dataset_id, count in data.SYNTHETIC_TRAIN_DATASET_COUNTS.items():
        rows.extend(records(dataset_id, count, dataset_id))
    for dataset_id, count in data.SNLI_TRAIN_DATASET_COUNTS.items():
        rows.extend(records(dataset_id, count, dataset_id))

    _, parsed = core._validate_training_rows(rows)
    real = tuple(row for row in parsed if row.dataset_id in data.REAL_TRAIN_DATASET_COUNTS)
    synthetic = tuple(
        row for row in parsed if row.dataset_id in data.SYNTHETIC_TRAIN_DATASET_COUNTS
    )
    snli = tuple(row for row in parsed if row.dataset_id in data.SNLI_TRAIN_DATASET_COUNTS)
    schedules = data.build_training_schedules(real, synthetic, snli)
    by_id = {row.record_id: row.dataset_id for row in parsed}

    assert len(parsed) == data.UNIQUE_TRAINING_RECORD_COUNT == 1504
    assert set(schedules) == {"synthetic_repeat", "snli_mix"}
    assert len(schedules["synthetic_repeat"]) == len(schedules["snli_mix"]) == 1512
    assert not any(
        by_id[example.record_id] in data.SNLI_TRAIN_DATASET_COUNTS
        for example in schedules["synthetic_repeat"]
    )
    assert any(
        by_id[example.record_id] in data.SNLI_TRAIN_DATASET_COUNTS
        for example in schedules["snli_mix"]
    )


def test_runtime_training_producer_runs_all_updates_and_result_consumer_accepts_receipt(
    monkeypatch, tmp_path
) -> None:
    import hashlib
    from types import SimpleNamespace
    from uuid import uuid4

    from experiments import mixture_training_core as core
    from experiments import mixture_training_data as data
    from experiments import mixture_training_runtime as runtime
    from experiments import mixture_training_runtime_execution as execution
    from experiments import mixture_training_runtime_helpers as helpers
    from experiments import mixture_training_runtime_scoring as scoring
    from experiments import real_pilot_core
    from experiments.mixture_training_contracts import (
        DATA_FILE_SHA256,
        EXPECTED_PROTOCOL_SHA256,
        MODEL_ID,
        MODEL_REVISION,
        RUNTIME_VERSION_PINS,
        canonical_json,
    )
    from reflex_decisions import snli_diagnostic, synthetic_data
    from reflex_decisions.broader_data import SNLI_LABELS
    from reflex_decisions.data import DecisionRecord
    from reflex_decisions.rendering import render_prompt
    from reflex_decisions.schema import DecisionRequest, Option

    def records(dataset_id: str, count: int, prefix: str, option_count: int) -> tuple:
        options = tuple(
            Option(id=f"option-{index}", label=f"Option {index}") for index in range(option_count)
        )
        return tuple(
            DecisionRecord(
                record_id=f"{prefix}-{index}",
                dataset_id=dataset_id,
                source_group_id=f"{prefix}-group-{index}",
                request=DecisionRequest(
                    context=f"Runtime integration context {prefix} {index}.",
                    question="Which option follows?",
                    options=options,
                ),
                answer_id=options[index % len(options)].id,
            )
            for index in range(count)
        )

    training_records = tuple(
        record
        for dataset_id, count in data.REAL_TRAIN_DATASET_COUNTS.items()
        for record in records(dataset_id, count, f"train-{dataset_id}", 2)
    )
    training_records += tuple(
        record
        for dataset_id, count in data.SYNTHETIC_TRAIN_DATASET_COUNTS.items()
        for record in records(dataset_id, count, f"train-{dataset_id}", 3)
    )
    training_records += tuple(
        record
        for dataset_id, count in data.SNLI_TRAIN_DATASET_COUNTS.items()
        for record in records(dataset_id, count, f"train-{dataset_id}", 3)
    )

    real_evaluation_records = tuple(
        record
        for dataset_id, count in real_pilot_core.DATASET_RECORD_COUNTS.items()
        for record in records(
            dataset_id,
            count,
            f"evaluation-{dataset_id}",
            14
            if dataset_id.startswith("dbpedia14-")
            else 3
            if dataset_id.startswith("snli-")
            else 2,
        )
    )
    balanced_options = tuple(Option(id=label, label=label) for label in SNLI_LABELS)
    balanced_records = tuple(
        DecisionRecord(
            record_id=f"balanced-evaluation-{group}-{label}",
            dataset_id=snli_diagnostic.DATASET_ID,
            source_group_id=f"balanced-evaluation-group-{group}",
            request=DecisionRequest(
                context=f"Balanced evaluation context {group} {label}.",
                question="Which relation follows?",
                options=balanced_options,
            ),
            answer_id=label,
        )
        for group in range(snli_diagnostic.SELECTED_GROUP_COUNT)
        for label in SNLI_LABELS
    )
    synthetic_records = synthetic_data.build_candidate().records
    presentations = list(
        data.build_evaluation_presentations(
            real_evaluation_records, balanced_records, synthetic_records
        )
    )
    experiment_id = "natural-reasoning-integration"
    run_id = experiment_id + data.RUN_SUFFIXES["synthetic_repeat"]
    initialization = {
        "snapshot": {
            "update": 0,
            "path": f"/artifacts/runs/{experiment_id}-init/adapter-update-000",
            "files_sha256": {
                "adapter_config.json": "a" * 64,
                "adapter_model.safetensors": "b" * 64,
            },
        },
        "tensor_sha256": "c" * 64,
    }
    pins = {
        "data_file_sha256": dict(DATA_FILE_SHA256),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": core.source_fingerprints(),
    }
    payload = core.build_payload(
        experiment_id=experiment_id,
        run_id=run_id,
        nonce=str(uuid4()),
        phase="train",
        arm="synthetic_repeat",
        pins=pins,
        train_records=training_records,
        evaluation_presentations=presentations,
        initialization=initialization,
    )

    module_names = [f"layer-{index:02d}" for index in range(60)]
    tensor_names = [
        f"{module}.lora_{part}.default.weight" for module in module_names for part in ("A", "B")
    ]
    tensor_shapes = {name: [1] for name in tensor_names}
    tensor_shapes[tensor_names[-1]] = [2_015_113]
    inventory = {
        "module_count": 60,
        "module_names": module_names,
        "adapter_tensor_count": 120,
        "adapter_parameter_count": 2_015_232,
        "adapter_tensor_names": tensor_names,
        "adapter_tensor_shapes": tensor_shapes,
        "adapter_tensor_dtypes": {name: "torch.float32" for name in tensor_names},
        "trainable_parameter_count": 2_015_232,
        "base_parameters_frozen_bf16": True,
    }

    class Flag:
        def all(self) -> Flag:
            return self

        def item(self) -> bool:
            return True

    class Scalar:
        def __init__(self, value: float = 0.5) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class Gradient:
        def all(self) -> Flag:
            return Flag()

        def detach(self) -> Gradient:
            return self

        def abs(self) -> Gradient:
            return self

        def sum(self) -> Scalar:
            return Scalar(1.0)

    class Parameter:
        def __init__(self, trainable: bool) -> None:
            self.requires_grad = trainable
            self.grad = None

    class FakeModel:
        def __init__(self) -> None:
            self.adapter = Parameter(True)
            self.backbone = Parameter(False)

        def parameters(self):
            return (self.adapter, self.backbone)

        def named_parameters(self):
            return (
                ("layer.lora_B.weight", self.adapter),
                ("layer.base.weight", self.backbone),
            )

        def train(self):
            return self

        def eval(self):
            return self

    class FakeLoss:
        def __init__(self, parameter: Parameter) -> None:
            self.parameter = parameter

        def detach(self) -> FakeLoss:
            return self

        def item(self) -> float:
            return 0.5

        def __truediv__(self, _denominator: int) -> FakeLoss:
            return self

        def backward(self) -> None:
            self.parameter.grad = Gradient()

    class FakeLogits:
        def __init__(self, parameter: Parameter) -> None:
            self.parameter = parameter

        def unsqueeze(self, _dimension: int) -> FakeLogits:
            return self

    class FakeOptimizer:
        def __init__(self, parameters, **_kwargs) -> None:
            self.parameters = tuple(parameters)

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none
            for parameter in self.parameters:
                parameter.grad = None

        def step(self) -> None:
            pass

    adapted = FakeModel()
    reloaded = FakeModel()

    class FakePeft:
        calls = 0

        @classmethod
        def from_pretrained(cls, *_args, is_trainable: bool, **_kwargs):
            cls.calls += 1
            return adapted if is_trainable else reloaded

    base_provenance = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "versions": {
            name: version for name, version in RUNTIME_VERSION_PINS.items() if name != "Pillow"
        },
        "tokenizer_file_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        execution,
        "_base_model",
        lambda *_args: (FakeModel(), SimpleNamespace(pad_token_id=0), base_provenance),
    )
    fake_training_helpers = SimpleNamespace(
        _validate_adapter_inventory=lambda *_args, **_kwargs: inventory,
        _save_adapter_snapshot=lambda _model, _run_dir, update: {
            "update": update,
            "path": f"/artifacts/runs/{run_id}/adapter-update-{update:03d}",
            "files_sha256": {
                "adapter_config.json": "a" * 64,
                "adapter_model.safetensors": "b" * 64,
            },
        },
        _adapter_state_changes=lambda *_args: {
            "changed_tensor_count": 1,
            "changed_tensor_names": ["layer-00.lora_B.weight"],
        },
    )
    monkeypatch.setattr(
        execution.importlib,
        "import_module",
        lambda name: fake_training_helpers if name == "experiments.modal_train_rehearsal" else None,
    )
    monkeypatch.setattr(scoring, "_snapshot_hashes", lambda snapshot: snapshot["files_sha256"])
    monkeypatch.setattr(scoring, "_verify_saved_adapter", lambda *_args: {"state": "initial"})
    monkeypatch.setattr(scoring, "_state_dict", lambda *_args: {"state": "trained"})
    monkeypatch.setattr(scoring, "_equal_states", lambda *_args: None)
    monkeypatch.setattr(helpers, "tensor_state_sha256", lambda *_args, **_kwargs: "c" * 64)
    torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=FakeOptimizer),
        tensor=lambda *_args, **_kwargs: object(),
        long=object(),
        float32="torch.float32",
        isfinite=lambda _value: Flag(),
        nn=SimpleNamespace(
            utils=SimpleNamespace(clip_grad_norm_=lambda *_args, **_kwargs: Scalar())
        ),
        cuda=SimpleNamespace(
            get_device_name=lambda _index: "CPU fake runtime", empty_cache=lambda: None
        ),
    )

    def score_forward(_model, _request, _tokenizer, _torch, result, category):
        helpers.increment_completed_forward(result["evidence"], category, 1)
        return FakeLogits(adapted.adapter), None

    monkeypatch.setattr(scoring, "_score_forward", score_forward)

    def score_presentations(
        _model,
        _tokenizer,
        rows,
        _torch,
        result,
        category,
        _run_dir,
        *,
        write_progress,
        retain_outputs=True,
    ):
        outputs = []
        for row in rows:
            helpers.increment_completed_forward(result["evidence"], category, 1)
            request = DecisionRequest.model_validate(row["request"])
            logits = [float(index) for index in range(len(row["order_ids"]))]
            output = {
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
            output.update(
                {
                    "candidate_logits": logits,
                    "winner_option_id": row["order_ids"][-1],
                    "input_tokens": 1,
                    "prompt_sha256": hashlib.sha256(
                        render_prompt(request).encode("utf-8")
                    ).hexdigest(),
                }
            )
            outputs.append(output)
        if retain_outputs:
            result["evidence"]["outputs"] = outputs
        return outputs

    monkeypatch.setattr(scoring, "_score_presentations", score_presentations)

    def save_outputs(result, run_dir, write_progress):
        outputs = result["evidence"]["outputs"]
        digest = hashlib.sha256(canonical_json(outputs) + b"\n").hexdigest()
        result["evidence"]["output_artifact"] = {
            "path": f"/artifacts/runs/{run_id}/outputs.json",
            "sha256": digest,
        }

    monkeypatch.setattr(scoring, "_save_outputs", save_outputs)
    result = runtime._initial_result(payload)
    result["provenance"]["versions"] = dict(RUNTIME_VERSION_PINS)
    result["provenance"]["measured_source_file_sha256"] = pins["source_file_sha256"]
    result["provenance"]["cuda_device"] = "CPU fake runtime"
    result["provenance"]["optimizer"] = {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "max_gradient_norm": 1.0,
    }

    execution.run_training(
        payload,
        result,
        tmp_path / run_id,
        torch,
        SimpleNamespace(cross_entropy=lambda logits, _target: FakeLoss(adapted.adapter)),
        object(),
        object(),
        FakePeft,
        object(),
        lambda *_args: None,
    )
    result["phase"] = "train"
    result["status"] = "passed"
    result["failure"] = None

    normalized = core.validate_result(result, payload)

    assert normalized["status"] == "passed"
    assert normalized["evidence"]["optimizer_updates_completed"] == 378
    assert len(normalized["evidence"]["training_step_losses"]) == 378
    assert tuple(normalized["evidence"]["adapter_paths"]) == ("189", "378")
    assert normalized["evidence"]["forward_counts"]["training"] == 1512


def test_importing_runtime_on_cpu_does_not_import_model_or_modal_packages() -> None:
    script = """
import sys
from experiments import mixture_training_runtime
forbidden = ("torch", "transformers", "peft", "modal")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_fingerprint_allowlist_covers_local_import_closure() -> None:
    allowlisted_paths = set(contracts.SOURCE_FINGERPRINT_PATHS)
    missing: set[tuple[str, str, str]] = set()

    def require_module(importer: str, module_name: str) -> None:
        source_path = _first_party_module_path(module_name)
        if source_path is None:
            missing.add((importer, module_name, "<unresolved first-party module>"))
        elif source_path not in allowlisted_paths:
            missing.add((importer, module_name, source_path))

    for relative in sorted(allowlisted_paths):
        if not relative.endswith(".py"):
            continue
        source = PROJECT_ROOT / relative
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        module = _source_module_name(relative)
        if module is None:
            continue
        package = module if relative.endswith("/__init__.py") else module.rpartition(".")[0]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_first_party_module(alias.name):
                        require_module(relative, alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = package.split(".") if package else []
                    prefix = package_parts[: len(package_parts) - node.level + 1]
                    base = ".".join(prefix + (node.module.split(".") if node.module else []))
                else:
                    base = node.module or ""
                if _is_first_party_module(base):
                    require_module(relative, base)
                    base_path = _first_party_module_path(base)
                    if base_path is not None and base_path.endswith("/__init__.py"):
                        for alias in node.names:
                            if alias.name != "*":
                                candidate = f"{base}.{alias.name}"
                                if _first_party_module_path(candidate):
                                    require_module(relative, candidate)
            elif isinstance(node, ast.Call):
                is_dynamic_import = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"import_module", "__import__"}
                ) or (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                if (
                    is_dynamic_import
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and _is_first_party_module(node.args[0].value)
                ):
                    require_module(relative, node.args[0].value)

    assert not missing, (
        f"local imports are absent from the source fingerprint allowlist: {sorted(missing)}"
    )


def test_remote_source_fingerprints_import_an_isolated_uploaded_bundle(tmp_path) -> None:
    bundle_root = tmp_path / "uploaded-source"
    for relative in contracts.SOURCE_FINGERPRINT_PATHS:
        source = PROJECT_ROOT / relative
        destination = bundle_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    script = """
import sys
from pathlib import Path
from experiments import mixture_training_runtime

assert mixture_training_runtime._project_root() == Path.cwd()
measured = mixture_training_runtime._measure_remote_source_fingerprints()
assert set(measured) == set(mixture_training_runtime.core.SOURCE_FINGERPRINT_PATHS)
forbidden = ("torch", "torchvision", "transformers", "peft", "modal", "safetensors")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden
)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(bundle_root), str(bundle_root / "src")))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=bundle_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
