from __future__ import annotations

import hashlib
import json

import pytest

from experiments import adapter_transfer_data as data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _records() -> tuple[DecisionRecord, ...]:
    rows: list[DecisionRecord] = []
    for dataset, prefix in (("boolq-dev-pilot-v1", "boolq"), ("copa-dev-pilot-v1", "copa")):
        for index in reversed(range(32)):
            rows.append(
                DecisionRecord(
                    record_id=f"{dataset}-{index:02d}",
                    dataset_id=dataset,
                    source_group_id=f"{prefix}-group-{index:02d}",
                    request=DecisionRequest(
                        context=f"Context {prefix} {index}.",
                        question=f"Question {prefix} {index}?",
                        options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
                    ),
                    answer_id="yes",
                )
            )
    return tuple(rows)


def _record(*, record_id: str, group_id: str, context: str) -> DecisionRecord:
    return DecisionRecord(
        record_id=record_id,
        dataset_id="fixture",
        source_group_id=group_id,
        request=DecisionRequest(
            context=context,
            question="Choose one?",
            options=(Option(id="a", label="A"), Option(id="b", label="B")),
        ),
        answer_id="a",
    )


def _selection() -> dict[str, object]:
    return {
        "selection_id": "adapter-transfer-v1",
        "training_run_id": "natural-reasoning-2026-10-04-r1",
        "arm": "snli_mix",
        "update": 378,
        "training_receipt_sha256": "1" * 64,
        "analysis_sha256": "2" * 64,
        "agreement_sha256": "3" * 64,
        "storage_verification_sha256": "4" * 64,
        "snapshot": {
            "update": 378,
            "path": "/artifacts/runs/natural-reasoning-2026-10-04-r1-snli/adapter-update-378",
            "files_sha256": {
                "adapter_config.json": "5" * 64,
                "adapter_model.safetensors": "6" * 64,
            },
        },
    }


def test_builds_exact_two_order_panels_and_first_four_group_parity() -> None:
    records = _records()

    presentations = data.build_presentations(records)
    parity = data.parity_presentations(presentations)

    assert len(presentations) == 128
    assert len(parity) == 16
    for start in range(0, len(presentations), 2):
        original, reversed_order = presentations[start : start + 2]
        assert original["dataset_id"] == reversed_order["dataset_id"]
        assert original["record_id"] == reversed_order["record_id"]
        assert original["order_index"] == 0
        assert reversed_order["order_index"] == 1
        assert original["order_ids"] == list(reversed(reversed_order["order_ids"]))
        assert "answer_id" not in original["request"]
        assert original["source_group_id"] != next(
            row.source_group_id for row in records if row.record_id == original["record_id"]
        )
    assert [row["record_id"] for row in presentations[::2]][:32] == sorted(
        row.record_id for row in records if row.dataset_id == "boolq-dev-pilot-v1"
    )
    assert [row["record_id"] for row in presentations[::2]][32:] == sorted(
        row.record_id for row in records if row.dataset_id == "copa-dev-pilot-v1"
    )
    assert {row["record_id"] for row in parity if row["dataset_id"] == "copa-dev-pilot-v1"} == {
        f"copa-dev-pilot-v1-{index:02d}" for index in range(4)
    }
    assert {row["record_id"] for row in parity if row["dataset_id"] == "boolq-dev-pilot-v1"} == {
        f"boolq-dev-pilot-v1-{index:02d}" for index in range(4)
    }


@pytest.mark.parametrize(
    ("eval_record", "train_record", "message"),
    [
        (
            _record(record_id="same", group_id="eval-group", context="Eval"),
            _record(record_id="same", group_id="train-group", context="Train"),
            "record IDs",
        ),
        (
            _record(record_id="eval", group_id="same-group", context="Eval"),
            _record(record_id="train", group_id="same-group", context="Train"),
            "source groups",
        ),
        (
            _record(record_id="eval", group_id="eval-group", context="Same"),
            _record(record_id="train", group_id="train-group", context="Same"),
            "semantic requests",
        ),
    ],
)
def test_rejects_each_training_overlap_identity(
    eval_record: DecisionRecord, train_record: DecisionRecord, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        data.validate_training_disjointness((eval_record,), (train_record,))


def test_selection_loading_fails_closed_until_exact_hash_is_pinned(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import adapter_transfer_contracts as contracts

    monkeypatch.setattr(contracts, "EXPECTED_SELECTION_SHA256", None)
    with pytest.raises(ValueError, match="has not been pinned"):
        data.load_inputs(tmp_path)


def test_selection_bytes_must_match_the_pinned_digest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import adapter_transfer_contracts as contracts

    selection_path = tmp_path / contracts.SELECTION_PATH
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text("{}\n")
    monkeypatch.setattr(contracts, "EXPECTED_SELECTION_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="selection SHA-256 mismatch"):
        data.load_inputs(tmp_path)


def test_load_inputs_verifies_panels_selection_sources_and_train_disjointness(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments import adapter_transfer_contracts as contracts
    from experiments import adapter_transfer_core as core
    from experiments import mixture_training_data, mixture_training_inputs
    from reflex_decisions import baseline_data, broader_data

    records = _records()
    copa = tuple(row for row in records if row.dataset_id == "copa-dev-pilot-v1")
    boolq = tuple(row for row in records if row.dataset_id == "boolq-dev-pilot-v1")
    snli = tuple(
        _record(
            record_id=f"snli-dev-pilot-v1-{index:02d}",
            group_id=f"snli-group-{index:02d}",
            context=f"SNLI {index}",
        ).model_copy(update={"dataset_id": "snli-dev-pilot-v1"})
        for index in range(32)
    )
    broader_records = (*boolq, *snli)
    source_bytes = {path: f"fixture:{path}".encode() for path in contracts.PANEL_FILE_SHA256}
    expected_files = {path: hashlib.sha256(raw).hexdigest() for path, raw in source_bytes.items()}
    selection = _selection()
    selection_raw = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    selection_path = tmp_path / contracts.SELECTION_PATH
    selection_path.parent.mkdir(parents=True)
    selection_path.write_bytes(selection_raw)
    protocol_path = tmp_path / contracts.PROTOCOL_PATH
    protocol_path.parent.mkdir(parents=True)
    protocol_raw = b"fixture protocol"
    protocol_path.write_bytes(protocol_raw)
    for relative, raw in source_bytes.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    train_rows = (
        _record(record_id="train-real", group_id="train-real-group", context="Real train"),
        _record(
            record_id="train-synthetic", group_id="train-synthetic-group", context="Synthetic train"
        ),
        _record(record_id="train-snli", group_id="train-snli-group", context="SNLI train"),
    )
    real_id = next(iter(mixture_training_data.REAL_TRAIN_DATASET_COUNTS))
    synthetic_id = next(iter(mixture_training_data.SYNTHETIC_TRAIN_DATASET_COUNTS))
    training = tuple(
        row.model_copy(update={"dataset_id": dataset_id})
        for row, dataset_id in zip(train_rows[:2], (real_id, synthetic_id), strict=True)
    ) + (train_rows[2],)
    training_pins = {"data_file_sha256": {"train.jsonl": "7" * 64}, "protocol_sha256": "8" * 64}

    monkeypatch.setattr(
        contracts, "EXPECTED_SELECTION_SHA256", hashlib.sha256(selection_raw).hexdigest()
    )
    from experiments.mixture_training_contracts import json_sha256

    monkeypatch.setattr(
        contracts,
        "EXPECTED_CANONICAL_SELECTION_SHA256",
        json_sha256(selection),
        raising=False,
    )
    monkeypatch.setattr(contracts, "PANEL_FILE_SHA256", expected_files)
    projection = broader_data.serialize_records(boolq)
    monkeypatch.setattr(contracts, "BOOLQ_PANEL_SHA256", hashlib.sha256(projection).hexdigest())
    monkeypatch.setattr(
        contracts,
        "EXPECTED_PROTOCOL_SHA256",
        hashlib.sha256(protocol_raw).hexdigest(),
    )
    monkeypatch.setattr(baseline_data, "verify_prepared_data", lambda *_args: (None, copa))
    monkeypatch.setattr(
        broader_data, "verify_prepared_data", lambda *_args: (None, broader_records)
    )
    monkeypatch.setattr(
        mixture_training_inputs,
        "load_natural_reasoning_data",
        lambda _root: (training[:1], (), training[1:2], training[2:], training_pins),
    )
    monkeypatch.setattr(data, "validate_training_pins", lambda pins: pins)
    source_hashes = {
        path: f"{index:064x}" for index, path in enumerate(contracts.SOURCE_FINGERPRINT_PATHS, 1)
    }
    monkeypatch.setattr(core, "source_fingerprints", lambda _root: source_hashes)

    loaded_records, presentations, loaded_selection, pins = data.load_inputs(tmp_path)

    assert loaded_records == tuple(
        row
        for dataset_id in ("boolq-dev-pilot-v1", "copa-dev-pilot-v1")
        for row in sorted(
            (record for record in records if record.dataset_id == dataset_id),
            key=lambda item: item.record_id,
        )
    )
    assert len(presentations) == 128
    assert loaded_selection == selection
    assert pins["selection_file_sha256"] == hashlib.sha256(selection_raw).hexdigest()
    assert pins["panel_sha256"] == {
        "copa": expected_files[contracts.COPA_RECORDS_PATH],
        "boolq": hashlib.sha256(projection).hexdigest(),
    }
