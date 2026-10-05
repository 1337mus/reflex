"""Remote worker lifecycle tests that never load a model or Torch."""

from __future__ import annotations

import hashlib
import importlib
import shutil
from pathlib import Path

import pytest

from experiments import adapter_transfer_runtime as runtime
from experiments import mixture_training_runtime_execution as execution
from experiments import mixture_training_runtime_helpers as helpers
from experiments import mixture_training_runtime_scoring as scoring
from tests.test_adapter_transfer_core import _output, _payload


def test_source_mismatch_stops_before_runtime_imports_and_returns_failure(
    monkeypatch,
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": "adapter-transfer-v1",
        "run_id": "adapter-transfer-2026-10-05-r1",
        "nonce": "c8b5fb54-591a-4541-aeca-6c1b2b5e5c89",
        "selection_sha256": "a" * 64,
        "pins": {"source_file_sha256": {}},
        "payload_sha256": "b" * 64,
    }
    events: list[str] = []
    failed = {
        **{key: payload[key] for key in payload},
        "status": "failed",
        "phase": "source_verification_failed",
        "provenance": {},
        "evidence": {},
        "failure": {"stage": "source_verification", "type": "ValueError", "message": "changed"},
    }

    monkeypatch.setattr(
        runtime.core,
        "validate_payload",
        lambda value: events.append("payload") or payload,
    )

    def reject_source(_payload) -> dict[str, str]:
        events.append("sources")
        raise ValueError("uploaded source fingerprint mismatch")

    monkeypatch.setattr(runtime, "_verify_remote_sources", reject_source)

    def forbidden_runtime_imports():
        events.append("gpu_imports")
        pytest.fail("GPU packages were imported before source verification")

    monkeypatch.setattr(runtime, "_runtime_packages", forbidden_runtime_imports)

    def build_failed_result(_payload, *, stage, error, sanitize, provenance=None, evidence=None):
        assert stage == "source_verification"
        assert isinstance(error, ValueError)
        assert sanitize(error) == "uploaded source fingerprint mismatch"
        return failed

    monkeypatch.setattr(runtime.core, "build_failed_result", build_failed_result)
    monkeypatch.setattr(
        runtime.core,
        "validate_result",
        lambda result, checked_payload: (
            result
            if result is failed and checked_payload is payload
            else pytest.fail("failure result was not validated against the payload")
        ),
    )

    result = runtime.remote_worker(payload)

    assert events == ["payload", "sources"]
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "source_verification"


def test_source_verification_uses_the_preserved_remote_src_layout(tmp_path, monkeypatch) -> None:
    remote_root = tmp_path / "root"
    project_root = Path(__file__).resolve().parents[1]
    expected = runtime.core.source_fingerprints(project_root)
    for relative in runtime.contracts.SOURCE_FINGERPRINT_PATHS:
        destination = remote_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            project_root / relative,
            destination,
        )
    monkeypatch.setattr(
        runtime,
        "__file__",
        str(remote_root / "experiments" / "adapter_transfer_runtime.py"),
    )
    payload = {"pins": {"source_file_sha256": expected}}

    assert runtime._verify_remote_sources(payload) == expected

    shutil.move(remote_root / "src" / "reflex_decisions", remote_root / "reflex_decisions")
    with pytest.raises(ValueError, match="source is unavailable: src/reflex_decisions/"):
        runtime._verify_remote_sources(payload)


def test_worker_preserves_partial_evidence_when_bad_provenance_blocks_failure_receipt(
    monkeypatch,
) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": "adapter-transfer-v1",
        "run_id": "adapter-transfer-2026-10-05-r1",
        "nonce": "c8b5fb54-591a-4541-aeca-6c1b2b5e5c89",
        "selection_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "pins": {"source_file_sha256": {}},
    }
    partial_evidence = {"worker_progress": "score rows retained"}
    failure_result = {"status": "failed", "failure": {"stage": "result_validation"}}
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(runtime.core, "validate_payload", lambda value: payload)
    monkeypatch.setattr(runtime, "_verify_remote_sources", lambda _payload: {})
    monkeypatch.setattr(runtime, "_runtime_packages", lambda: (object(),))

    def finish_with_bad_provenance(_payload, result, _runtime_items) -> None:
        result["provenance"]["model_id"] = "untrusted"
        result["evidence"] = partial_evidence

    monkeypatch.setattr(runtime, "_run_evaluation", finish_with_bad_provenance)

    def build_failed_result(_payload, *, stage, error, sanitize, provenance=None, evidence=None):
        calls.append((provenance, evidence))
        assert stage == "result_validation"
        if provenance["model_id"] == "untrusted":
            raise ValueError("invalid worker provenance")
        assert evidence is partial_evidence
        return failure_result

    monkeypatch.setattr(runtime.core, "build_failed_result", build_failed_result)

    def validate_result(result, _payload):
        if result.get("phase") == "completed":
            raise ValueError("worker result provenance does not match the pinned model")
        return result

    monkeypatch.setattr(runtime.core, "validate_result", validate_result)

    result = runtime.remote_worker(payload)

    assert result is failure_result
    assert len(calls) == 2
    assert calls[0][1] is partial_evidence
    assert calls[1][0]["model_id"] == runtime.contracts.MODEL_ID
    assert calls[1][1] is partial_evidence


def test_failed_result_fallback_marks_counts_unknown_when_partial_evidence_is_invalid(
    monkeypatch,
) -> None:
    payload = _payload(monkeypatch)
    presentation = payload["presentations"][0]
    output = _output(presentation)
    result = runtime._initial_result(payload)
    evidence = result["evidence"]
    evidence["forward_counts"].update(base_evaluation=1, total=1)
    evidence["input_token_counts"].update(base_evaluation=5, total=5)
    evidence["outputs"]["base"] = [output]
    evidence["invalid_partial_marker"] = "forces schema rejection"

    failed = runtime._build_failed_result(
        payload,
        stage="base_evaluation",
        error=RuntimeError("base scoring stopped"),
        provenance=result["provenance"],
        evidence=evidence,
    )

    assert failed["status"] == "failed"
    assert failed["evidence"]["outputs"]["base"] == []
    assert failed["evidence"]["forward_counts"] == {
        "base_evaluation": None,
        "training": 0,
        "final_evaluation": None,
        "reload_parity": None,
        "total": None,
    }


def test_saved_tensor_decode_follows_exact_snapshot_file_verification(
    tmp_path, monkeypatch
) -> None:
    adapter_dir = tmp_path / "adapter-update-378"
    adapter_dir.mkdir()
    files = {
        "adapter_config.json": b'{"r":8}',
        "adapter_model.safetensors": b"verified tensor bytes",
    }
    for name, contents in files.items():
        (adapter_dir / name).write_bytes(contents)
    expected = {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()}
    snapshot = {"path": str(adapter_dir), "files_sha256": expected}
    events: list[str] = []
    fake_state = {"lora.weight": object()}

    class FakeScoring:
        @staticmethod
        def _snapshot_hashes(received) -> dict[str, str]:
            assert received is snapshot
            events.append("file_hashes")
            return expected

    class FakeSafeTensors:
        @staticmethod
        def load_file(path: str, *, device: str):
            events.append("decode")
            assert path == str(adapter_dir / "adapter_model.safetensors")
            assert device == "cpu"
            return fake_state

    class FakeHelpers:
        @staticmethod
        def tensor_state_sha256(state, *, torch_module):
            events.append("tensor_digest")
            assert state is fake_state
            assert torch_module == "torch-module"
            return "d" * 64

    modules = {
        "experiments.mixture_training_runtime_scoring": FakeScoring,
        "safetensors.torch": FakeSafeTensors,
        "experiments.mixture_training_runtime_helpers": FakeHelpers,
    }
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        return modules[name] if name in modules else original_import_module(name, package)

    monkeypatch.setattr(runtime.importlib, "import_module", fake_import_module)

    state, digest, actual_files = runtime._load_saved_adapter_state(snapshot, "torch-module")

    assert events == ["file_hashes", "decode", "tensor_digest"]
    assert state is fake_state
    assert digest == "d" * 64
    assert actual_files == expected


def test_evaluation_reuses_pinned_scorer_and_checks_saved_state_before_adapter_load(
    monkeypatch,
) -> None:
    events: list[str] = []
    files = {
        "adapter_config.json": "a" * 64,
        "adapter_model.safetensors": "b" * 64,
    }
    snapshot = {
        "update": 378,
        "path": "/artifacts/runs/natural-reasoning-2026-10-04-r1-snli/adapter-update-378",
        "files_sha256": files,
    }
    presentation = {
        "presentation_id": "p0",
        "record_id": "r0",
        "dataset_id": "copa-dev-pilot-v1",
        "source_group_id": "g0",
        "request_hash": "c" * 64,
        "order_index": 0,
        "order_ids": ["cause", "effect"],
    }
    payload = {
        "schema_version": 1,
        "experiment_id": "adapter-transfer-v1",
        "run_id": "adapter-transfer-2026-10-05-r1",
        "nonce": "c8b5fb54-591a-4541-aeca-6c1b2b5e5c89",
        "selection_sha256": "d" * 64,
        "payload_sha256": "e" * 64,
        "pins": {"source_file_sha256": {}},
        "selection": {"snapshot": snapshot},
        "presentations": [presentation],
        "parity_presentations": [presentation],
    }
    result = runtime._initial_result(payload)
    state = {"lora.weight": "saved-state"}
    model_count = 0

    class FakeCuda:
        @staticmethod
        def get_device_name(_index: int) -> str:
            return "Fake A10"

    class FakeTorch:
        cuda = FakeCuda()

    class FakePeft:
        @staticmethod
        def from_pretrained(model, path: str, *, is_trainable: bool, autocast_adapter_dtype: bool):
            events.append("adapter_load")
            assert path == snapshot["path"]
            assert is_trainable is False
            assert autocast_adapter_dtype is True
            return FakeAdapter(model)

    class FakeAdapter:
        def __init__(self, base) -> None:
            self.base = base

        def eval(self):
            return self

    class FakeAutoModel:
        pass

    class FakeTokenizer:
        pass

    class FakeBase:
        def __init__(self, name: str) -> None:
            self.name = name

    def load_base(*_args):
        nonlocal model_count
        model_count += 1
        name = f"base-{model_count}"
        events.append(f"base_load:{name}")
        return (
            FakeBase(name),
            object(),
            {
                "model_id": "Qwen/Qwen3.5-0.8B-Base",
                "model_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
                "effective_dtype": "torch.bfloat16",
                "tokenizer_file_sha256": {"tokenizer.json": "f" * 64},
            },
        )

    def load_state(received, torch):
        events.append("snapshot_verified")
        assert received is snapshot
        assert torch is FakeTorch
        return state, "1" * 64, files

    def verify_adapter(model, received, digest, _peft, _torch):
        events.append("loaded_state_verified")
        assert received is snapshot
        assert digest == "1" * 64
        return state

    def score(
        _model,
        _tokenizer,
        rows,
        _torch,
        evidence_result,
        category,
        _run_dir,
        *,
        write_progress,
        retain_outputs=True,
    ):
        events.append(f"score:{category}")
        if category == "base_evaluation":
            assert evidence_result["evidence"]["adapter_identity"] == {
                "snapshot_path": snapshot["path"],
                "files_sha256": files,
                "tensor_sha256": "1" * 64,
                "reloaded_tensor_sha256": None,
                "dtype": "torch.float32",
            }
        outputs = []
        for row in rows:
            helpers.increment_completed_forward(evidence_result["evidence"], category, 5)
            outputs.append(
                {
                    **row,
                    "candidate_logits": [1.0, 0.0],
                    "winner_option_id": "cause",
                    "input_tokens": 5,
                    "prompt_sha256": "9" * 64,
                }
            )
        if retain_outputs:
            evidence_result["evidence"]["outputs"] = outputs
        return outputs

    monkeypatch.setattr(execution, "_base_model", load_base)
    monkeypatch.setattr(runtime, "_load_saved_adapter_state", load_state)
    monkeypatch.setattr(scoring, "_verify_saved_adapter", verify_adapter)
    monkeypatch.setattr(
        scoring, "_equal_states", lambda left, right, _torch: events.append("states_equal")
    )
    monkeypatch.setattr(helpers, "tensor_state_sha256", lambda *_args, **_kwargs: "1" * 64)
    monkeypatch.setattr(scoring, "_score_presentations", score)

    runtime_items = (
        FakeTorch,
        object(),
        FakePeft,
        FakeAutoModel,
        FakeTokenizer,
        object(),
        {"torch": "2.14.1+cu130"},
    )
    runtime._run_evaluation(payload, result, runtime_items)

    assert events.index("snapshot_verified") < events.index("adapter_load")
    assert events.count("base_load:base-1") == 1
    assert events.count("adapter_load") == 2
    assert events.count("loaded_state_verified") == 2
    assert events[-1] == "score:reload_parity"
    assert result["provenance"]["cuda_device"] == "Fake A10"
    assert result["provenance"]["reload_tokenizer_file_sha256"] == {"tokenizer.json": "f" * 64}
    assert result["evidence"]["adapter_identity"] == {
        "snapshot_path": snapshot["path"],
        "files_sha256": files,
        "tensor_sha256": "1" * 64,
        "reloaded_tensor_sha256": "1" * 64,
        "dtype": "torch.float32",
    }
    assert result["evidence"]["forward_counts"] == {
        "base_evaluation": 1,
        "training": 0,
        "final_evaluation": 1,
        "reload_parity": 1,
        "total": 3,
    }
    assert len(result["evidence"]["outputs"]["base"]) == 1
    assert len(result["evidence"]["outputs"]["adapter"]) == 1
    assert len(result["evidence"]["reload_parity"]["outputs"]) == 1
    assert result["evidence"]["reload_parity"]["max_candidate_logit_delta"] == 0.0
