"""Synthetic end-to-end checks for authenticated study runtime execution."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from experiments import adapter_transfer_contracts, mixture_training_contracts
from experiments import mixture_training_runtime_helpers as runtime_helpers
from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_progress as progress_api
from experiments import runtime_rule_study_results as results_api
from experiments import runtime_rule_study_runtime as runtime_api
from experiments import runtime_rule_study_runtime_scoring as scoring_api
from experiments import runtime_rule_study_sources as sources_api
from reflex_decisions.smoke import sanitize_exception_message
from tests import (
    test_runtime_rule_study_payloads as payload_builders,
)
from tests import (
    test_runtime_rule_study_provenance as provenance_builders,
)
from tests import (
    test_runtime_rule_study_results as result_builders,
)
from tests import (
    test_runtime_rule_study_training_evidence as training_builders,
)


@pytest.fixture(scope="module")
def _exact_source_payload_fixture() -> dict[str, Any]:
    fixture = deepcopy(payload_builders._payload_fixture())
    original = cast(dict[str, str], fixture["source_file_sha256"])
    fixture["source_file_sha256"] = {
        path: original.get(path, hashlib.sha256(path.encode("utf-8")).hexdigest())
        for path in sources_api.SOURCE_PATHS
    }
    return fixture


@pytest.fixture(autouse=True)
def _pin_toy_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    selection_hash = mixture_training_contracts.json_sha256(payload_builders._fixture_selection())
    monkeypatch.setattr(
        adapter_transfer_contracts, "EXPECTED_CANONICAL_SELECTION_SHA256", selection_hash
    )


def _payload(fixture: dict[str, Any], role: str) -> dict[str, object]:
    payload = payload_builders._build_payload(deepcopy(fixture), role)
    assert set(cast(dict[str, str], payload["source_file_sha256"])) == set(sources_api.SOURCE_PATHS)
    return payload


def test_failure_message_uses_the_shared_secret_sanitizer() -> None:
    failure = runtime_api._failure(
        "base_model_load",
        RuntimeError(
            "request failed at https://user:password@example.invalid/model?token=secret "
            "with Authorization: Bearer abc123"
        ),
    )

    assert failure["message"] == sanitize_exception_message(
        RuntimeError(
            "request failed at https://user:password@example.invalid/model?token=secret "
            "with Authorization: Bearer abc123"
        )
    )
    assert "password" not in failure["message"]
    assert "token=secret" not in failure["message"]
    assert "abc123" not in failure["message"]


def test_failure_message_uses_a_bounded_generic_fallback_when_sanitizing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_api,
        "sanitize_exception_message",
        lambda _error: (_ for _ in ()).throw(RuntimeError("sanitizer unavailable")),
    )

    failure = runtime_api._failure("terminal_progress_persistence", OSError("private detail"))

    assert failure == {
        "stage": "terminal_progress_persistence",
        "type": "OSError",
        "message": "Worker failed during terminal_progress_persistence (OSError).",
    }


class _FakeBase:
    def __init__(self, events: list[str], index: int) -> None:
        self.events = events
        self.index = index

    def __del__(self) -> None:
        self.events.append(f"base_released_{self.index}")


class _FakeAdapter:
    def __init__(self, base: _FakeBase, state: dict[str, object]) -> None:
        self.base = base
        self.state = state
        self.in_eval = False

    def eval(self) -> None:
        self.in_eval = True


class _FakeCuda:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "NVIDIA A10"

    def empty_cache(self) -> None:
        self.events.append("cuda_cache_cleared")


class _FakeTorch:
    def __init__(self, events: list[str]) -> None:
        self.cuda = _FakeCuda(events)


def _install_worker_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, object],
    *,
    fail: str | None = None,
    fail_terminal_write: bool = False,
) -> tuple[list[str], dict[str, object]]:
    events: list[str] = []
    state_by_kind: dict[str, dict[str, object]] = {
        "selected": {"kind": "selected"},
        "final": {"kind": "final"},
    }
    expected_sources = cast(dict[str, str], payload["source_file_sha256"])
    specs = {
        "training": cast(list[dict[str, object]], payload["training_specs"]),
        "final_evaluation": cast(list[dict[str, object]], payload["evaluation_specs"]),
        "reload_parity": cast(list[dict[str, object]], payload["reload_specs"]),
    }
    persisted: dict[str, object] = {"write_count": 0}
    fake_torch = _FakeTorch(events)

    monkeypatch.setattr(sources_api, "measure_sources", lambda _root: dict(expected_sources))
    monkeypatch.setattr(
        runtime_api,
        "_forbidden_kernel_packages",
        lambda: dict.fromkeys(runtime_api._FORBIDDEN_KERNEL_PACKAGES, False),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: contracts.RUNTIME_VERSION_PINS[name],
    )
    base_loads = 0

    def load_base(
        _model_class: object, _tokenizer_class: object, _torch: object
    ) -> tuple[object, object, dict[str, object]]:
        nonlocal base_loads
        base_loads += 1
        events.append(f"base_load_{base_loads}")
        if fail == "base_load":
            raise OSError("synthetic base load failure")
        return (
            _FakeBase(events, base_loads),
            object(),
            provenance_builders._base_model(payload),
        )

    monkeypatch.setattr(runtime_api, "_load_base", load_base)

    def from_pretrained(
        base: _FakeBase,
        snapshot_path: str,
        *,
        is_trainable: bool,
        autocast_adapter_dtype: bool,
    ) -> _FakeAdapter:
        assert autocast_adapter_dtype is True
        snapshot_update = 378 if "update-378" in snapshot_path else 336
        events.append(f"adapter_load_{snapshot_update}")
        if fail == "adapter_load" and is_trainable:
            raise OSError("synthetic adapter load failure")
        state = state_by_kind["selected" if is_trainable else "final"]
        return _FakeAdapter(base, state)

    fake_peft_model = SimpleNamespace(from_pretrained=from_pretrained)
    runtime_modules = runtime_api._RuntimeModules(
        torch=fake_torch,
        peft=SimpleNamespace(PeftModel=fake_peft_model),
        model_class=object,
        tokenizer_class=object,
    )
    monkeypatch.setattr(runtime_api, "_load_runtime_modules", lambda: runtime_modules)

    helper_module = ModuleType("experiments.modal_train_rehearsal")
    helper_module._validate_adapter_inventory = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    adapter_module = ModuleType("experiments.mixture_training_runtime_scoring")

    def snapshot_hashes(snapshot: dict[str, object]) -> dict[str, str]:
        events.append(f"snapshot_{snapshot['update']}")
        if fail == "snapshot" and snapshot["update"] == 378:
            raise OSError("synthetic snapshot verification failure")
        return cast(dict[str, str], snapshot["files_sha256"])

    def verify_saved_adapter(
        _model: _FakeAdapter,
        snapshot: dict[str, object],
        _expected_digest: str,
        _peft: object,
        _torch: object,
    ) -> dict[str, object]:
        events.append(f"verify_{snapshot['update']}")
        return state_by_kind["selected" if snapshot["update"] == 378 else "final"]

    adapter_module._snapshot_hashes = snapshot_hashes  # type: ignore[attr-defined]
    adapter_module._verify_saved_adapter = verify_saved_adapter  # type: ignore[attr-defined]
    adapter_module._state_dict = lambda model, _peft: model.state  # type: ignore[attr-defined]
    training_module = ModuleType("experiments.runtime_rule_study_runtime_training")

    def train_adapter(
        _payload: object,
        result: dict[str, object],
        _model: _FakeAdapter,
        _prepared: object,
        _initial_state: dict[str, object],
        run_dir: Path,
        _torch: object,
        _peft: object,
        ledger: progress_api.ForwardLedger,
        *,
        on_progress: object,
    ) -> dict[str, object]:
        events.append("training")
        if fail == "train":
            raise RuntimeError("synthetic training failure")
        result["phase"] = "training"
        evidence = cast(dict[str, object], result["evidence"])
        training = training_builders._complete_training_evidence()
        paths = cast(dict[str, dict[str, object]], training["adapter_paths"])
        for update, snapshot in paths.items():
            snapshot["path"] = (
                f"/artifacts/runs/{payload['run_id']}/adapter-update-{int(update):03d}"
            )
        evidence["training"] = training
        for spec in specs["training"]:
            ledger.begin("training", spec)
            ledger.complete()
        evidence.update(ledger.snapshot())
        assert callable(on_progress)
        cast(Any, on_progress)(result, run_dir)
        return state_by_kind["final"]

    training_module.train_adapter = train_adapter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "experiments.modal_train_rehearsal", helper_module)
    monkeypatch.setitem(sys.modules, "experiments.mixture_training_runtime_scoring", adapter_module)
    monkeypatch.setitem(
        sys.modules, "experiments.runtime_rule_study_runtime_training", training_module
    )

    scoring_module = scoring_api

    def prepare_requests(_payload: object, _tokenizer: object) -> object:
        events.append("compile")
        if fail == "compile":
            raise ValueError("synthetic request compilation failure")
        return SimpleNamespace(
            training_items=tuple(range(len(specs["training"]))),
            evaluation_items=tuple(range(len(specs["final_evaluation"]))),
            reload_items=tuple(range(len(specs["reload_parity"]))),
        )

    monkeypatch.setattr(scoring_module, "prepare_requests", prepare_requests)

    def score_presentations(
        model: object,
        items: tuple[object, ...],
        presentations: object,
        _torch: object,
        ledger: progress_api.ForwardLedger,
        evidence: dict[str, object],
        category: str,
        *,
        on_progress: object = None,
    ) -> None:
        events.append(f"score_{category}")
        if category == "final_evaluation":
            assert isinstance(model, _FakeAdapter) and model.in_eval
        presentations = cast(list[dict[str, object]], presentations)
        category_specs = specs[category]
        assert len(items) == len(presentations) == len(category_specs)
        if fail == "pending_forward" and category == "final_evaluation":
            ledger.begin("final_evaluation", category_specs[0])
            evidence.update(ledger.snapshot())
            raise OSError("synthetic forward failure")
        if fail == "extraction" and category == "final_evaluation":
            ledger.begin("final_evaluation", category_specs[0])
            ledger.complete()
            evidence.update(ledger.snapshot())
            raise RuntimeError("synthetic output extraction failure")
        for spec in category_specs:
            ledger.begin(category, spec)
            ledger.complete()
        evidence.update(ledger.snapshot())
        rows = result_builders._output_rows(presentations, category_specs)
        if category == "final_evaluation":
            cast(list[dict[str, object]], evidence["outputs"]).extend(rows)
        else:
            reload_evidence = cast(dict[str, object], evidence["reload"])
            if fail == "parity":
                rows = rows[:1]
            cast(list[dict[str, object]], reload_evidence["outputs"]).extend(rows)
            reload_evidence["tensor_equal"] = True
            reload_evidence["tensor_sha256"] = "d" * 64
            reload_evidence["winner_match_count"] = len(rows)
            reload_evidence["max_candidate_logit_difference"] = 0.0
            if fail == "parity":
                raise RuntimeError("synthetic parity failure")
        assert on_progress is None or callable(on_progress)

    monkeypatch.setattr(scoring_module, "score_presentations", score_presentations)

    monkeypatch.setattr(
        runtime_helpers,
        "tensor_state_sha256",
        lambda state, *, torch_module: (
            contracts.SELECTED_TENSOR_SHA256 if state["kind"] == "selected" else "d" * 64
        ),
    )
    monkeypatch.setattr(runtime_api, "_states_equal", lambda *_args: True)
    monkeypatch.setattr(runtime_api, "ARTIFACT_RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(runtime_api, "_commit_artifact_volume", lambda: None)
    real_write_progress = runtime_api._write_progress

    def write_progress(run_dir: Path, result: dict[str, object]) -> None:
        persisted["write_count"] = cast(int, persisted["write_count"]) + 1
        count = cast(int, persisted["write_count"])
        if fail_terminal_write and count == 4:
            raise OSError("synthetic terminal persistence failure")
        real_write_progress(run_dir, result)

    monkeypatch.setattr(runtime_api, "_write_progress", write_progress)
    return events, persisted


def _validate_returned(payload: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    return results_api.validate_result(
        result,
        payload=payload,
        expected_payload_sha256=payload["payload_sha256"],
    )


def test_unchanged_worker_returns_a_validator_accepted_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _exact_source_payload_fixture: dict[str, Any],
) -> None:
    payload = _payload(_exact_source_payload_fixture, "unchanged")
    events, _persisted = _install_worker_fakes(monkeypatch, tmp_path, payload)
    result = runtime_api.remote_worker(payload, payload["payload_sha256"])

    checked = _validate_returned(payload, result)

    assert checked["status"] == "passed"
    assert checked["phase"] == "completed"
    assert "training" not in events
    assert "score_reload_parity" not in events
    assert not (tmp_path / "runs" / cast(str, payload["run_id"])).exists()


def test_trained_worker_persists_and_validates_the_exact_update_336_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _exact_source_payload_fixture: dict[str, Any],
) -> None:
    payload = _payload(_exact_source_payload_fixture, "runtime_mix")
    events, _persisted = _install_worker_fakes(monkeypatch, tmp_path, payload)

    result = runtime_api.remote_worker(payload, payload["payload_sha256"])

    run_dir = tmp_path / "runs" / cast(str, payload["run_id"])
    persisted = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    checked = _validate_returned(payload, result)
    checked_persisted = _validate_returned(payload, persisted)
    training = cast(dict[str, object], cast(dict[str, object], checked["evidence"])["training"])
    paths = cast(dict[str, dict[str, object]], training["adapter_paths"])
    final_snapshot = paths["336"]

    assert checked["status"] == "passed"
    assert checked["phase"] == "completed"
    assert checked_persisted["status"] == "passed"
    assert checked_persisted["phase"] == "completed"
    assert final_snapshot["update"] == 336
    assert final_snapshot["path"] == f"/artifacts/runs/{payload['run_id']}/adapter-update-336"
    assert "snapshot_336" in events
    assert events.index("base_released_1") < events.index("base_load_2")


def test_source_mismatch_fails_before_runtime_import_or_model_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _exact_source_payload_fixture: dict[str, Any],
) -> None:
    payload = _payload(_exact_source_payload_fixture, "unchanged")
    events, _persisted = _install_worker_fakes(monkeypatch, tmp_path, payload)
    monkeypatch.setattr(
        sources_api,
        "measure_sources",
        lambda _root: {
            **cast(dict[str, str], payload["source_file_sha256"]),
            "experiments/runtime_rule_study_runtime.py": "0" * 64,
        },
    )

    result = runtime_api.remote_worker(payload, payload["payload_sha256"])
    checked = _validate_returned(payload, result)

    assert checked["status"] == "failed"
    assert cast(dict[str, object], checked["failure"])["stage"] == "source_preflight"
    assert not any(event.startswith("base_load") for event in events)
    assert "compile" not in events


def test_compile_failure_stops_before_training_or_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _exact_source_payload_fixture: dict[str, Any],
) -> None:
    payload = _payload(_exact_source_payload_fixture, "runtime_mix")
    events, _persisted = _install_worker_fakes(monkeypatch, tmp_path, payload, fail="compile")

    result = runtime_api.remote_worker(payload, payload["payload_sha256"])
    checked = _validate_returned(payload, result)

    assert checked["status"] == "failed"
    assert cast(dict[str, object], checked["failure"])["stage"] == "request_compilation"
    assert "base_load_1" in events
    assert not any(
        event in {"training", "score_final_evaluation", "score_reload_parity"} for event in events
    )


@pytest.mark.parametrize(
    ("failure_mode", "expected_pending", "expected_completed"),
    [("pending_forward", True, 0), ("extraction", False, 1)],
)
def test_worker_preserves_observed_forward_prefix_and_pending_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _exact_source_payload_fixture: dict[str, Any],
    failure_mode: str,
    expected_pending: bool,
    expected_completed: int,
) -> None:
    payload = _payload(_exact_source_payload_fixture, "runtime_mix")
    events, _persisted = _install_worker_fakes(monkeypatch, tmp_path, payload, fail=failure_mode)

    result = runtime_api.remote_worker(payload, payload["payload_sha256"])
    checked = _validate_returned(payload, result)
    evidence = cast(dict[str, object], checked["evidence"])
    counts = cast(dict[str, int], evidence["completed_forward_counts"])

    assert checked["status"] == "failed"
    assert checked["phase"] == "final_evaluation"
    assert counts["training"] == 1_344
    assert counts["final_evaluation"] == expected_completed
    pending = evidence["pending_forward"]
    assert (pending is not None) is expected_pending
    assert "score_final_evaluation" in events


def test_terminal_progress_write_failure_persists_a_valid_failed_finalize_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _exact_source_payload_fixture: dict[str, Any],
) -> None:
    payload = _payload(_exact_source_payload_fixture, "runtime_mix")
    _events, persisted_state = _install_worker_fakes(
        monkeypatch,
        tmp_path,
        payload,
        fail_terminal_write=True,
    )

    result = runtime_api.remote_worker(payload, payload["payload_sha256"])

    run_dir = tmp_path / "runs" / cast(str, payload["run_id"])
    persisted = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    checked = _validate_returned(payload, result)
    checked_persisted = _validate_returned(payload, persisted)
    evidence = cast(dict[str, object], checked_persisted["evidence"])
    counts = cast(dict[str, int], evidence["completed_forward_counts"])

    assert persisted_state["write_count"] == 5
    assert checked["status"] == "failed"
    assert checked["phase"] == "finalize"
    assert checked_persisted["status"] == "failed"
    assert checked_persisted["phase"] == "finalize"
    assert counts == {
        "training": 1_344,
        "final_evaluation": 3_946,
        "reload_parity": 32,
        "total": 5_322,
    }
    assert len(cast(list[object], evidence["outputs"])) == 3_946
    assert len(cast(list[object], cast(dict[str, object], evidence["reload"])["outputs"])) == 32
    assert cast(dict[str, object], checked_persisted["failure"])["stage"] == (
        "terminal_progress_persistence"
    )
