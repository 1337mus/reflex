"""CPU checks for the diagnostic runtime's cross-module result contracts."""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from math import prod
from types import ModuleType

from experiments import modal_snli_diagnostic as runner
from experiments import snli_diagnostic_core as diagnostic_core
from experiments import snli_diagnostic_results as results
from experiments import snli_diagnostic_runtime as runtime
from experiments import training_rehearsal_core


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, int],
        dtype: str,
        *,
        value: int = 0,
        requires_grad: bool = False,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.value = value
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return prod(self.shape)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self


class _FakeLoraModule:
    def __init__(self) -> None:
        self.lora_A = {"default": object()}
        self.lora_B = {"default": object()}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged_adapters: tuple[str, ...] = ()
        self.merged = False


class _FakeQwenModel:
    active_adapters = ["default"]

    def __init__(self, torch: object) -> None:
        self.modules = [(f"model.layer_{index}", _FakeLoraModule()) for index in range(60)]
        self.adapter_state: dict[str, _FakeTensor] = {}
        for module_index in range(60):
            for adapter_name in ("lora_A", "lora_B"):
                name = f"model.layer_{module_index}.{adapter_name}.default.weight"
                size = 1
                if len(self.adapter_state) == 119:
                    size = training_rehearsal_core.EXPECTED_TRAINABLE_PARAMETERS - 119
                self.adapter_state[name] = _FakeTensor(
                    (1, size), torch.float32, value=len(self.adapter_state)
                )
        self.base_parameter = _FakeTensor((1, 1), torch.bfloat16)

    def named_modules(self):
        return iter([("", self), *self.modules])

    def named_parameters(self):
        return iter([*self.adapter_state.items(), ("model.embed.weight", self.base_parameter)])


class _FakeTorch:
    float32 = "torch.float32"
    bfloat16 = "torch.bfloat16"

    @staticmethod
    def equal(left: _FakeTensor, right: _FakeTensor) -> bool:
        return left.shape == right.shape and left.value == right.value


def test_qwen_adapter_evidence_satisfies_the_frozen_result_validator(tmp_path, monkeypatch) -> None:
    torch = _FakeTorch()
    model = _FakeQwenModel(torch)
    peft = ModuleType("peft")
    peft.get_peft_model_state_dict = lambda _model: model.adapter_state
    safetensors = ModuleType("safetensors")
    safetensors_torch = ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda _path, device: model.adapter_state
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setattr(runtime, "ADAPTER_PATH", str(tmp_path))
    monkeypatch.setattr(
        runtime,
        "verify_qwen_adapter_files",
        lambda _directory: {
            "expected_file_sha256": dict(runtime.ADAPTER_FILE_SHA256),
            "actual_file_sha256": dict(runtime.ADAPTER_FILE_SHA256),
        },
    )

    evidence = runtime._qwen_adapter_evidence(model, torch)

    validated = results._validate_qwen_adapter(evidence)
    assert validated["inventory"]["base_parameters_frozen_bf16"] is True


def _balanced_payload() -> dict[str, object]:
    from reflex_decisions.data import DecisionRecord
    from reflex_decisions.schema import DecisionRequest, Option
    from reflex_decisions.snli_diagnostic import DATASET_ID, SNLI_LABELS

    options = tuple(Option(id=label, label=label.title()) for label in SNLI_LABELS)
    records = []
    for group_index in range(64):
        group_id = hashlib.sha256(f"sdk-group-{group_index}".encode()).hexdigest()
        for label in SNLI_LABELS:
            pair_id = hashlib.sha256(f"sdk-pair-{group_index}-{label}".encode()).hexdigest()
            records.append(
                DecisionRecord(
                    record_id=f"{DATASET_ID}-{hashlib.sha256(pair_id.encode()).hexdigest()}",
                    dataset_id=DATASET_ID,
                    source_group_id=group_id,
                    request=DecisionRequest(
                        context=f"Premise for SDK group {group_index}.",
                        question=f"Does the premise support {label}?",
                        options=options,
                    ),
                    answer_id=label,
                )
            )
    source_hashes = {path: "d" * 64 for path in diagnostic_core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[diagnostic_core.PROTOCOL_PATH] = diagnostic_core.EXPECTED_PROTOCOL_SHA256
    pins = {
        "records_sha256": diagnostic_core.EXPECTED_RECORDS_SHA256,
        "manifest_sha256": diagnostic_core.EXPECTED_MANIFEST_SHA256,
        "recipe_sha256": diagnostic_core.EXPECTED_RECIPE_SHA256,
        "protocol_sha256": diagnostic_core.EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": source_hashes,
    }
    return diagnostic_core.build_payload(
        tuple(records),
        pins=pins,
        run_id=str(uuid.UUID(int=31)),
        nonce=str(uuid.UUID(int=32)),
    )


def test_modal_sdk_mismatch_is_saved_as_a_failed_receipt_without_starting_modal(
    tmp_path, monkeypatch, capsys
) -> None:
    payload = _balanced_payload()
    monkeypatch.setattr(runner, "_load_payload", lambda _args, _root: payload)
    monkeypatch.setattr(runner.modal_smoke, "_verify_profile", lambda *_args: True)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda _name: "1.6.0")
    for name in runner.modal_smoke.CREDENTIAL_OVERRIDES:
        monkeypatch.delenv(name, raising=False)

    real_import_module = runner.importlib.import_module
    modal_imports: list[str] = []

    def guarded_import_module(name: str, package: str | None = None):
        if name in {"modal", "torch"}:
            modal_imports.append(name)
            raise AssertionError(f"unexpected runtime import: {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(runner.importlib, "import_module", guarded_import_module)
    output = tmp_path / "failed-receipt.json"

    result = runner.main(
        [
            "--launch",
            "--records",
            str(tmp_path / "records.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--recipe",
            str(tmp_path / "recipe.json"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert result == 1
    assert output.is_file(), captured.err
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["provenance"]["modal_sdk_version"] == "1.6.0"
    assert {model["failure"]["stage"] for model in saved["models"].values()} == {"modal_version"}
    assert {model["failure"]["message"] for model in saved["models"].values()} == {
        "Modal SDK 1.6.1 is required"
    }
    assert modal_imports == []
    assert json.loads(captured.out)["artifact"] == str(output)
