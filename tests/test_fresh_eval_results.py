from __future__ import annotations

import pytest


def test_validate_unknown_work_requires_positive_uncertainty_for_lost_execution() -> None:
    from experiments import fresh_eval_results as results

    with pytest.raises(ValueError, match="unknown"):
        results.validate_unknown_work(
            {"possible_forwards": 0, "reason": "worker result was lost"},
            failure_stage="modal_lifecycle",
            passed=False,
        )


def test_validate_provenance_rejects_unpinned_runtime_versions() -> None:
    from experiments import fresh_eval_results as results

    payload = {
        "pins": {
            "model_id": "Qwen/Qwen3.5-0.8B-Base",
            "model_revision": "revision",
            "source_file_sha256": {"source.py": "a" * 64},
            "tokenizer_file_sha256": "b" * 64,
        }
    }
    provenance = {
        "model_id": "Qwen/Qwen3.5-0.8B-Base",
        "model_revision": "revision",
        "versions": {},
        "source_file_sha256": {"source.py": "a" * 64},
        "measured_source_file_sha256": {"source.py": "a" * 64},
        "base_model": {},
        "cuda_device": "NVIDIA A10",
        "reload_tokenizer_file_sha256": {},
    }

    with pytest.raises(ValueError, match="runtime versions"):
        results._validate_provenance(provenance, payload, passed=True)


def test_core_exposes_the_result_contract(monkeypatch) -> None:
    from experiments import fresh_eval_core as core
    from experiments import fresh_eval_results as results

    sentinel = {"status": "passed"}
    monkeypatch.setattr(results, "validate_result", lambda value, payload, root=None: sentinel)

    assert core.validate_result({}, {}) is sentinel
