from __future__ import annotations

from pathlib import Path

import pytest

from experiments import mixture_training_inputs as inputs
from experiments.mixture_training_contracts import DATA_FILE_SHA256
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option

ROOT = Path(__file__).resolve().parents[1]


def _record(record_id: str, dataset_id: str) -> DecisionRecord:
    return DecisionRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        source_group_id=f"{record_id}-group",
        request=DecisionRequest(
            context=f"Fixture {record_id}.",
            question="Which option is supported?",
            options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
        ),
        answer_id="yes",
    )


def test_natural_reasoning_loader_returns_all_four_bundles_and_pins(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = (_record("real-train", next(iter(inputs.mixture_data.REAL_TRAIN_DATASET_COUNTS))),)
    balanced = (_record("balanced-dev", "snli-balanced-v1-development"),)
    synthetic = (
        _record("synthetic-train", next(iter(inputs.mixture_data.SYNTHETIC_TRAIN_DATASET_COUNTS))),
    )
    snli_training = (_record("snli-train", "snli-training-v1"),)
    pins = {"test-pin": "natural-reasoning"}
    schedule_bundles = {}

    monkeypatch.setattr(inputs, "verify_protocol", lambda _path: "protocol")
    monkeypatch.setattr(inputs, "_verify_data_file_pins", lambda _root: None)
    monkeypatch.setattr(
        inputs.pilot_data,
        "verify_prepared_data",
        lambda *_args: ("manifest", real, "recipe"),
    )
    monkeypatch.setattr(inputs.real_pilot_core, "validate_pilot_data", lambda *_args: None)
    monkeypatch.setattr(inputs, "_verify_balanced_candidate", lambda _root: balanced)
    monkeypatch.setattr(inputs, "_verify_synthetic_candidate", lambda _root: synthetic)
    monkeypatch.setattr(inputs, "_verify_snli_training_candidate", lambda _root: snli_training)
    monkeypatch.setattr(
        inputs.mixture_data,
        "build_training_schedule_audit",
        lambda real_rows, synthetic_rows, snli_rows: schedule_bundles.update(
            real=tuple(real_rows), synthetic=tuple(synthetic_rows), snli=tuple(snli_rows)
        ),
    )
    monkeypatch.setattr(inputs.mixture_data, "build_evaluation_presentations", lambda *_args: ())
    monkeypatch.setattr(inputs.mixture_data, "_validate_disjoint_pools", lambda *_args: None)
    monkeypatch.setattr(inputs.mixture_data, "build_evaluation_panel_audit", lambda *_args: None)
    monkeypatch.setattr(inputs, "make_pins", lambda _root: pins)

    returned = inputs.load_natural_reasoning_data(tmp_path)

    assert returned == (real, balanced, synthetic, snli_training, pins)
    assert schedule_bundles == {"real": real, "synthetic": synthetic, "snli": snli_training}


def test_obsolete_four_value_loader_is_removed() -> None:
    assert not hasattr(inputs, "load_local_data")


def test_missing_snli_candidate_data_pin_fails_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    unpinned = dict(DATA_FILE_SHA256)
    del unpinned["data/processed/snli-training-v1/records.jsonl"]
    monkeypatch.setattr(inputs, "DATA_FILE_SHA256", unpinned)

    with pytest.raises(ValueError, match="allowlist differs from the frozen pins"):
        inputs._verify_data_file_pins(ROOT)


def test_wrong_snli_candidate_data_pin_fails_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    altered = dict(DATA_FILE_SHA256)
    altered["data/processed/snli-training-v1/records.jsonl"] = "0" * 64
    monkeypatch.setattr(inputs, "DATA_FILE_SHA256", altered)

    with pytest.raises(ValueError, match="allowlist differs from the frozen pins"):
        inputs._verify_data_file_pins(ROOT)
