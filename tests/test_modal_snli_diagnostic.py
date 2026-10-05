"""Fake-runtime tests for the bounded Modal SNLI diagnostic."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import uuid
from contextlib import contextmanager, nullcontext

import pytest

from experiments import modal_snli_diagnostic as runner
from experiments import snli_diagnostic_core as core
from experiments import snli_diagnostic_runtime as runtime
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _payload() -> dict[str, object]:
    from reflex_decisions.snli_diagnostic import DATASET_ID, SNLI_LABELS

    records: list[DecisionRecord] = []
    options = tuple(Option(id=label, label=label.title()) for label in SNLI_LABELS)
    for group_index in range(64):
        group_id = hashlib.sha256(f"group-{group_index}".encode()).hexdigest()
        for label in SNLI_LABELS:
            pair_id = hashlib.sha256(f"pair-{group_index}-{label}".encode()).hexdigest()
            record_id = f"{DATASET_ID}-{hashlib.sha256(pair_id.encode()).hexdigest()}"
            records.append(
                DecisionRecord(
                    record_id=record_id,
                    dataset_id=DATASET_ID,
                    source_group_id=group_id,
                    request=DecisionRequest(
                        context=f"Premise {pair_id}",
                        question="Which relation follows?",
                        options=options,
                    ),
                    answer_id=label,
                )
            )
    source_hashes = {path: "d" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    source_hashes[core.PROTOCOL_PATH] = core.EXPECTED_PROTOCOL_SHA256
    pins = {
        "records_sha256": core.EXPECTED_RECORDS_SHA256,
        "manifest_sha256": core.EXPECTED_MANIFEST_SHA256,
        "recipe_sha256": core.EXPECTED_RECIPE_SHA256,
        "protocol_sha256": core.EXPECTED_PROTOCOL_SHA256,
        "source_file_sha256": source_hashes,
    }
    return core.build_payload(
        tuple(records),
        pins=pins,
        run_id=str(uuid.UUID(int=1)),
        nonce=str(uuid.UUID(int=2)),
    )


def test_plan_only_does_not_import_modal_or_torch(monkeypatch, capsys) -> None:
    original_import_module = importlib.import_module

    def guarded_import(name: str, package: str | None = None):
        if name in {"modal", "torch"}:
            raise AssertionError(f"plan-only mode imported {name}")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)

    assert runner.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan-only"
    assert plan["models"] == ["qwen_base", "qwen_final", "intern", "kev"]
    assert plan["max_total_forwards"] == 4611


def test_payload_is_label_free_and_contains_all_six_orders() -> None:
    payload = core.validate_payload(_payload())

    assert len(payload["presentations"]) == 1152
    assert set(payload["presentations"][0]) == {
        "presentation_id",
        "record_id",
        "source_group_id",
        "request_hash",
        "order_index",
        "order_ids",
        "request",
    }
    by_record: dict[str, list[dict[str, object]]] = {}
    for presentation in payload["presentations"]:
        by_record.setdefault(presentation["record_id"], []).append(presentation)
        assert "answer_id" not in presentation
        assert "gold_label" not in presentation
        assert "label" not in presentation
    assert len(by_record) == 192
    assert all({row["order_index"] for row in rows} == set(range(6)) for rows in by_record.values())


def test_source_or_adapter_pin_mismatch_fails_exactly(tmp_path, monkeypatch) -> None:
    hashes = {path: "e" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}
    assert runtime._verify_digest_map(hashes, hashes) == hashes
    bad_hashes = dict(hashes)
    bad_hashes[core.SOURCE_FINGERPRINT_PATHS[0]] = "f" * 64
    with pytest.raises(ValueError, match="differ from their pinned"):
        runtime._verify_digest_map(hashes, bad_hashes)

    assert runtime.ADAPTER_FILE_SHA256 == {
        "adapter_config.json": "fa6fdf55295985c39b5cfd6f6dfb66ab3c941756429cd402f77e1ac455fdea8d",
        "adapter_model.safetensors": (
            "315c23b1517c9590386d22afad5fac2927f97c31f42cdfa73b2c921494266fd8"
        ),
    }
    assert core.MODEL_PINS["qwen_base"] == {
        "model_id": "Qwen/Qwen3.5-0.8B-Base",
        "model_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    }

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    contents = {"adapter_config.json": b"config", "adapter_model.safetensors": b"weights"}
    for name, value in contents.items():
        (adapter_dir / name).write_bytes(value)
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    monkeypatch.setattr(runtime, "ADAPTER_FILE_SHA256", expected)
    assert runtime.verify_qwen_adapter_files(adapter_dir) == {
        "expected_file_sha256": expected,
        "actual_file_sha256": expected,
    }
    monkeypatch.setattr(
        runtime, "ADAPTER_FILE_SHA256", {**expected, "adapter_config.json": "0" * 64}
    )
    with pytest.raises(ValueError, match="file hashes"):
        runtime.verify_qwen_adapter_files(adapter_dir)


def test_remote_source_rejection_happens_before_model_loading(monkeypatch) -> None:
    payload = _payload()

    class FakeRuntime:
        called = False

        def measure_sources(self) -> dict[str, str]:
            return {path: "0" * 64 for path in core.SOURCE_FINGERPRINT_PATHS}

        def runtime_versions(self) -> dict[str, str]:
            raise AssertionError("runtime checks must follow source verification")

        def run_worker(self, *_args: object) -> dict[str, dict[str, object]]:
            self.called = True
            raise AssertionError("model loading must not start for a source mismatch")

    fake = FakeRuntime()
    result = runtime.remote_worker(payload, "intern", fake)

    assert result["models"]["intern"]["status"] == "failed"
    assert result["failure"]["stage"] == "source_verification"
    assert fake.called is False


def test_scoring_runs_inside_inference_mode() -> None:
    inside = False

    @contextmanager
    def inference_mode():
        nonlocal inside
        inside = True
        try:
            yield
        finally:
            inside = False

    class FakeTorch:
        pass

    FakeTorch.inference_mode = staticmethod(inference_mode)

    request = DecisionRequest(
        context="A premise.",
        question="Choose a relation.",
        options=(
            Option(id="entailment", label="Entailment"),
            Option(id="neutral", label="Neutral"),
            Option(id="contradiction", label="Contradiction"),
        ),
    )
    presentation = {
        "presentation_id": "presentation-0",
        "record_id": "record-0",
        "source_group_id": "a" * 64,
        "request_hash": request.request_hash,
        "order_index": 0,
        "order_ids": [option.id for option in request.options],
        "request": request.model_dump(mode="json"),
    }
    result: dict[str, object] = {"model_name": "intern", "presentations": []}

    def scorer(_request: DecisionRequest) -> dict[str, object]:
        assert inside is True
        return {"raw_logits": [1.0, 0.0, -1.0], "input_tokens": 12, "prompt_sha256": "a" * 64}

    runtime._score_reference_presentations([presentation], scorer, FakeTorch, result)
    assert inside is False
    assert result["presentations"][0]["winner_option_id"] == "entailment"
    assert result["presentations"][0]["source_group_id"] == "a" * 64


def test_all_workers_spawn_before_get_and_sibling_results_survive_failure(monkeypatch) -> None:
    payload = _payload()
    monkeypatch.setattr(core, "validate_model_result", lambda result, *_args: result)
    events: list[str] = []

    class Call:
        def __init__(self, worker: str) -> None:
            self.worker = worker

        def get(self) -> dict[str, object]:
            events.append(f"get:{self.worker}")
            if self.worker == "intern":
                raise TimeoutError("Intern worker failed")
            names = runtime.WORKER_MODELS[self.worker]
            return {
                "worker": self.worker,
                "models": {name: {"model_name": name, "status": "passed"} for name in names},
            }

    class Function:
        def spawn(self, _payload: object, worker: str) -> Call:
            events.append(f"spawn:{worker}")
            return Call(worker)

    models = runner._collect_model_results(payload, Function())

    assert events[:3] == ["spawn:qwen", "spawn:intern", "spawn:kev"]
    assert events[3:] == ["get:qwen", "get:intern", "get:kev"]
    assert models["qwen_base"]["status"] == "passed"
    assert models["qwen_final"]["status"] == "passed"
    assert models["kev"]["status"] == "passed"
    assert models["intern"]["status"] == "failed"


def test_modal_lifecycle_error_forces_failed_top_level_status(monkeypatch) -> None:
    payload = _payload()
    models = {name: {"model_name": name, "status": "passed"} for name in core.MODEL_NAMES}
    monkeypatch.setattr(core, "validate_receipt", lambda receipt, _payload: receipt)

    receipt = runner._receipt(
        payload,
        models,
        modal_version="1.6.1",
        lifecycle_failure=RuntimeError("app teardown failed"),
    )

    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "modal_lifecycle"
    assert all(model["status"] == "passed" for model in receipt["models"].values())


def test_profile_preflight_error_is_not_mislabeled_as_modal_lifecycle(monkeypatch) -> None:
    import argparse

    payload = _payload()
    monkeypatch.setattr(runner, "_load_payload", lambda *_args: payload)
    monkeypatch.setattr(runner.modal_smoke, "_verify_profile", lambda *_args: False)
    for name in runner.modal_smoke.CREDENTIAL_OVERRIDES:
        monkeypatch.setenv(name, "")

    receipt = runner._launch(argparse.Namespace(records="", manifest="", recipe=""))

    assert receipt["status"] == "failed"
    assert "failure" not in receipt
    assert all(
        model["failure"]["stage"] == "profile_verification" for model in receipt["models"].values()
    )


def test_intern_provenance_refreshes_after_first_scored_request(monkeypatch) -> None:
    payload = _payload()
    raw_provenance: dict[str, object] = {
        "model_id": "intern-model",
        "model_revision": "intern-revision",
        "source_commit": "source-commit",
        "module_sha256": "a" * 64,
        "effective_dtype": "torch.float32",
        "auxiliary_forward_count": 0,
        "calibration_parity_passed": False,
    }

    def scorer(_request: DecisionRequest) -> dict[str, object]:
        raw_provenance["auxiliary_forward_count"] = 1
        raw_provenance["calibration_parity_passed"] = True
        return {"raw_logits": [1.0, 0.0, -1.0], "input_tokens": 12, "prompt_sha256": "b" * 64}

    class FakeAdapter:
        @staticmethod
        def load_scorer() -> tuple[object, dict[str, object]]:
            return scorer, raw_provenance

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return nullcontext()

    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "experiments.baseline_intern":
            return FakeAdapter
        return original_import_module(name, package)

    def normalized_identity(_model_name: str, value: object) -> dict[str, object]:
        return json.loads(json.dumps(value))

    def normalized_terminal(_model_name: str, value: object) -> dict[str, object]:
        assert value["calibration_parity_passed"] is True
        assert value["auxiliary_forward_count"] == 1
        return json.loads(json.dumps(value))

    monkeypatch.setattr(runtime.importlib, "import_module", fake_import_module)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setattr(runtime.core, "validate_model_result", lambda result, *_args: result)
    from experiments import real_pilot_baseline_core

    monkeypatch.setattr(real_pilot_baseline_core, "_validate_scorer_identity", normalized_identity)
    monkeypatch.setattr(
        real_pilot_baseline_core, "_validate_scorer_provenance", normalized_terminal
    )

    result = runtime._run_reference_model(
        payload,
        "intern",
        dict(payload["pins"]["source_file_sha256"]),
        dict(runtime.RUNTIME_VERSION_PINS),
    )

    assert result["status"] == "passed"
    assert result["provenance"]["scorer"]["calibration_parity_passed"] is True
    assert result["provenance"]["scorer"]["auxiliary_forward_count"] == 1
    assert result["forward_counts"] == {"scored": 1152, "auxiliary": 1, "total": 1153}


def test_failed_model_total_forward_count_is_unknown() -> None:
    result = {
        "presentations": [{}, {}],
        "status": "passed",
    }

    failed = runtime._fail_model(
        result,
        "inference",
        RuntimeError("scorer failed after one request"),
        auxiliary=1,
    )

    assert failed["forward_counts"] == {"scored": 2, "auxiliary": 1, "total": None}
