"""Tests for the pinned CPU inputs to the runtime-rule study."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import IO, Any

import pytest

from experiments import (
    adapter_transfer_contracts,
    adapter_transfer_data,
    mixture_training_contracts,
    mixture_training_data,
    mixture_training_inputs,
    real_pilot_core,
    runtime_rule_study_data,
    runtime_rule_study_inputs,
)
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option

ROOT = Path(__file__).resolve().parents[1]


def _record(
    dataset_id: str,
    index: int,
    *,
    option_count: int,
    group_id: str | None = None,
) -> DecisionRecord:
    options = tuple(
        Option(id=f"option-{option}", label=f"Option {option}") for option in range(option_count)
    )
    return DecisionRecord(
        record_id=f"{dataset_id}-record-{index:03d}",
        dataset_id=dataset_id,
        source_group_id=group_id or f"{dataset_id}-group-{index:03d}",
        request=DecisionRequest(
            context=f"Self-authored fixture context for {dataset_id} {index}.",
            question=f"Which choice applies to fixture {dataset_id} {index}?",
            options=options,
        ),
        answer_id=options[index % option_count].id,
    )


def _retention_records() -> tuple[
    tuple[DecisionRecord, ...], tuple[DecisionRecord, ...], tuple[DecisionRecord, ...]
]:
    real_options = {
        "dbpedia14-pilot-v1": 14,
        "sms-pilot-v1": 2,
        "snli-pilot-v1": 3,
    }
    real = tuple(
        _record(
            dataset_id,
            index,
            option_count=real_options[dataset_id.split("-pilot")[0] + "-pilot-v1"],
        )
        for dataset_id, count in real_pilot_core.DATASET_RECORD_COUNTS.items()
        for index in range(count)
    )

    balanced = tuple(
        _record(
            "snli-balanced-v1-development",
            group * 3 + label,
            option_count=3,
            group_id=f"balanced-source-{group:03d}",
        )
        for group in range(64)
        for label in range(3)
    )

    synthetic_rows: list[DecisionRecord] = []
    for dataset_id, count in mixture_training_data.SYNTHETIC_DATASET_COUNTS.items():
        if dataset_id == "synthetic-atomic-fact-inference-v1-development":
            menu_sizes = (3,) * count
        elif dataset_id == "synthetic-numeric-selection-v1-development":
            menu_sizes = (4,) * 9 + (8,) * 41
        else:
            menu_sizes = (2,) * count
        synthetic_rows.extend(
            _record(dataset_id, index, option_count=menu_sizes[index]) for index in range(count)
        )
    return real, balanced, tuple(synthetic_rows)


def _transfer_records() -> tuple[DecisionRecord, ...]:
    return tuple(
        _record(dataset_id, index, option_count=2)
        for dataset_id in ("boolq-dev-pilot-v1", "copa-dev-pilot-v1")
        for index in range(32)
    )


def _copy_temp_root(root: Path) -> None:
    paths = {
        *adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS,
        mixture_training_contracts.PROTOCOL_PATH,
        "docs/runtime-rule-study-protocol.md",
        "docs/verification/adapter-transfer-summary.json",
    }
    for relative in sorted(paths):
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _stub_natural_loader(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real, balanced, synthetic = _retention_records()
    snli_train = tuple(_record("snli-training-v1", index, option_count=3) for index in range(500))
    pins = mixture_training_contracts.make_pins(root)
    monkeypatch.setattr(
        mixture_training_inputs,
        "load_natural_reasoning_data",
        lambda _root: (real, balanced, synthetic, snli_train, pins),
    )


def _stub_transfer_loader(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = _transfer_records()
    presentations = tuple(adapter_transfer_data.build_presentations(records))
    summary = json.loads((root / "docs/verification/adapter-transfer-summary.json").read_text())
    historical_sources = summary["pins"]["candidate_source_file_sha256"]
    natural_pins = mixture_training_contracts.make_pins(root)
    selection = json.loads(
        (ROOT / adapter_transfer_contracts.SELECTION_PATH).read_text(encoding="utf-8")
    )
    transfer_pins = {
        "data_file_sha256": dict(adapter_transfer_contracts.PANEL_FILE_SHA256),
        "training_data_file_sha256": natural_pins["data_file_sha256"],
        "source_file_sha256": historical_sources,
        "selection_file_sha256": adapter_transfer_contracts.EXPECTED_SELECTION_SHA256,
        "protocol_sha256": adapter_transfer_contracts.EXPECTED_PROTOCOL_SHA256,
        "training_protocol_sha256": mixture_training_contracts.EXPECTED_PROTOCOL_SHA256,
    }
    monkeypatch.setattr(
        adapter_transfer_data,
        "load_inputs",
        lambda _root: (records, list(presentations), selection, transfer_pins),
    )


def _new_pool(family: str, split: str) -> tuple[DecisionRecord, ...]:
    dataset_id = f"{family}-data-v1-{split}"
    if split == "train":
        menu_sizes: tuple[int, ...] = (4,) * 19 + (6,) * 18 + (8,) * 19
    else:
        menu_sizes = (4,) * 5 + (6,) * 5 + (8,) * 4
    return tuple(
        _record(dataset_id, index, option_count=menu_size)
        for index, menu_size in enumerate(menu_sizes)
    )


def _write_new_inputs(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pools = {
        "data/processed/routing-data-v1/train.jsonl": _new_pool("routing", "train"),
        "data/processed/routing-data-v1/development.jsonl": _new_pool("routing", "development"),
        "data/processed/tool-data-v1/train.jsonl": _new_pool("tool", "train"),
        "data/processed/tool-data-v1/development.jsonl": _new_pool("tool", "development"),
    }
    pins = dict(runtime_rule_study_inputs._NEW_FILE_SHA256)
    for relative, records in pools.items():
        content = b"".join(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for record in records
        )
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        pins[relative] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(runtime_rule_study_inputs, "_NEW_FILE_SHA256", pins)


def _valid_temp_root(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _copy_temp_root(root)
    assert not (root / "data/processed").exists()
    assert all(not (root / path).exists() for path in runtime_rule_study_inputs._NEW_FILE_SHA256)
    _stub_natural_loader(root, monkeypatch)
    _stub_transfer_loader(root, monkeypatch)
    _write_new_inputs(root, monkeypatch)
    for family in ("routing", "tool"):
        for split in ("calibration", "test"):
            sentinel = root / f"data/processed/{family}-data-v1/{split}.jsonl"
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("sealed sentinel; must remain unread\n", encoding="utf-8")
    return root


def _replace_new_file_hash(monkeypatch: pytest.MonkeyPatch, relative: str, content: bytes) -> None:
    pins = dict(runtime_rule_study_inputs._NEW_FILE_SHA256)
    pins[relative] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(runtime_rule_study_inputs, "_NEW_FILE_SHA256", pins)


def _rewrite_jsonl(path: Path, change_first: dict[str, object]) -> bytes:
    rows = path.read_bytes().splitlines()
    value = json.loads(rows[0])
    value.update(change_first)
    rows[0] = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    content = b"\n".join(rows) + b"\n"
    path.write_bytes(content)
    return content


def test_retention_panel_is_exact_and_request_only() -> None:
    real, balanced, synthetic = _retention_records()
    transfer = _transfer_records()

    presentations = runtime_rule_study_inputs.build_retention_presentations(
        real, balanced, synthetic, transfer
    )

    expected_counts = {
        "dbpedia14-pilot-v1-development": 1568,
        "sms-pilot-v1-development": 120,
        "snli-balanced-v1-development": 1152,
        "synthetic-atomic-fact-inference-v1-development": 450,
        "synthetic-numeric-selection-v1-development": 364,
        "boolq-dev-pilot-v1": 64,
        "copa-dev-pilot-v1": 64,
    }
    assert len(presentations) == 3782
    assert Counter(row["dataset_id"] for row in presentations) == Counter(expected_counts)
    assert len({row["presentation_id"] for row in presentations}) == len(presentations)
    assert all(set(row) == runtime_rule_study_data.PRESENTATION_FIELDS for row in presentations)
    for row in presentations:
        assert isinstance(row["request"], dict)
        assert "answer_id" not in row["request"]

    old_panel = mixture_training_data.build_evaluation_presentations(real, balanced, synthetic)
    expected_old = tuple(
        row
        for row in old_panel
        if row["dataset_id"]
        in {
            "dbpedia14-pilot-v1-development",
            "sms-pilot-v1-development",
            "snli-balanced-v1-development",
            "synthetic-atomic-fact-inference-v1-development",
            "synthetic-numeric-selection-v1-development",
        }
    )
    expected_transfer = tuple(adapter_transfer_data.build_presentations(transfer))
    assert presentations == (*expected_old, *expected_transfer)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("wrong-old-dataset", "real pilot bundle dataset IDs"),
        ("wrong-old-count", "real pilot bundle dataset IDs"),
        ("wrong-transfer-dataset", "adapter-transfer panel"),
        ("wrong-transfer-count", "adapter-transfer panel"),
    ],
)
def test_retention_panel_rejects_wrong_dataset_or_count(invalid: str, message: str) -> None:
    real, balanced, synthetic = _retention_records()
    transfer = _transfer_records()
    if invalid == "wrong-old-dataset":
        real = (*real[:-1], real[-1].model_copy(update={"dataset_id": "unexpected-dataset"}))
    elif invalid == "wrong-old-count":
        real = real[:-1]
    elif invalid == "wrong-transfer-dataset":
        transfer = (
            *transfer[:-1],
            transfer[-1].model_copy(update={"dataset_id": "unexpected-dataset"}),
        )
    else:
        transfer = transfer[:-1]

    with pytest.raises(ValueError, match=message):
        runtime_rule_study_inputs.build_retention_presentations(real, balanced, synthetic, transfer)


def test_cross_dataset_evaluation_source_groups_must_be_disjoint() -> None:
    first = _record("old-development", 1, option_count=2, group_id="shared-eval-group")
    second = _record("new-development", 2, option_count=2, group_id="shared-eval-group")

    runtime_rule_study_inputs._validate_cross_bundle_disjointness((), (first,), ())
    with pytest.raises(ValueError, match="source groups overlap"):
        runtime_rule_study_inputs._validate_cross_bundle_disjointness((), (first, second), ())


def test_cross_dataset_evaluation_semantic_requests_must_be_disjoint() -> None:
    first = _record("old-development", 1, option_count=2)
    second = _record("new-development", 2, option_count=2).model_copy(
        update={"request": first.request, "answer_id": first.answer_id}
    )

    runtime_rule_study_inputs._validate_cross_bundle_disjointness((), (first,), ())
    with pytest.raises(ValueError, match="semantic requests overlap"):
        runtime_rule_study_inputs._validate_cross_bundle_disjointness((), (first, second), ())


def test_loader_reads_exact_pinned_new_files_and_ignores_sealed_sentinels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _valid_temp_root(tmp_path / "study-root", monkeypatch)
    opened_new: list[str] = []
    original_open = Path.open

    def track_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        if isinstance(mode, str) and "r" in mode:
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                pass
            else:
                if relative.startswith("data/processed/routing-data-v1/") or relative.startswith(
                    "data/processed/tool-data-v1/"
                ):
                    opened_new.append(relative)
        return original_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", track_open)
    loaded = runtime_rule_study_inputs.load_study_inputs(root)

    assert opened_new == list(runtime_rule_study_inputs._NEW_FILE_SHA256)
    assert len(loaded.training_pools) == 5
    assert len(loaded.development_pools) == 2
    assert len(loaded.evaluation_records) == 525
    assert len(loaded.retention_presentations) == 3782
    assert len(loaded.new_presentations) == 164
    assert loaded.selection["selection_id"] == "adapter-transfer-v1"
    assert loaded.file_sha256["docs/verification/adapter-transfer-summary.json"] == (
        runtime_rule_study_inputs._TRANSFER_SUMMARY_SHA256
    )
    assert set(runtime_rule_study_inputs._NEW_FILE_SHA256).issubset(loaded.file_sha256)


def test_loader_rejects_altered_pinned_bytes_after_positive_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _valid_temp_root(tmp_path / "study-root", monkeypatch)
    runtime_rule_study_inputs.load_study_inputs(root)
    relative = next(iter(runtime_rule_study_inputs._NEW_FILE_SHA256))
    path = root / relative
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match=f"SHA-256 mismatch: {relative}"):
        runtime_rule_study_inputs.load_study_inputs(root)


def test_loader_rejects_outside_root_symlink_after_positive_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _valid_temp_root(tmp_path / "study-root", monkeypatch)
    runtime_rule_study_inputs.load_study_inputs(root)
    relative = next(iter(runtime_rule_study_inputs._NEW_FILE_SHA256))
    pinned_path = root / relative
    outside = tmp_path / "outside-training.jsonl"
    outside.write_bytes(pinned_path.read_bytes())
    pinned_path.unlink()
    pinned_path.symlink_to(outside)

    with pytest.raises(ValueError, match="unavailable inside the project root"):
        runtime_rule_study_inputs.load_study_inputs(root)


@pytest.mark.parametrize(
    "content",
    [b"not-json\n", b'{"record_id":"first","record_id":"second"}\n', b'{"score":NaN}\n'],
)
def test_loader_rejects_malformed_records_after_positive_control(
    content: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _valid_temp_root(tmp_path / "study-root", monkeypatch)
    runtime_rule_study_inputs.load_study_inputs(root)
    relative = "data/processed/routing-data-v1/train.jsonl"
    (root / relative).write_bytes(content)
    _replace_new_file_hash(monkeypatch, relative, content)

    with pytest.raises(ValueError, match="malformed record"):
        runtime_rule_study_inputs.load_study_inputs(root)


def test_loader_rejects_wrong_split_after_positive_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _valid_temp_root(tmp_path / "study-root", monkeypatch)
    runtime_rule_study_inputs.load_study_inputs(root)
    relative = "data/processed/routing-data-v1/train.jsonl"
    path = root / relative
    content = _rewrite_jsonl(path, {"dataset_id": "routing-data-v1-development"})
    _replace_new_file_hash(monkeypatch, relative, content)

    with pytest.raises(ValueError, match="routing training data must contain exactly 56 records"):
        runtime_rule_study_inputs.load_study_inputs(root)


@pytest.mark.parametrize(
    ("overlap", "message"),
    [
        ("record-id", "record IDs must be unique"),
        ("raw-group", "source groups overlap"),
        ("semantic", "semantic requests overlap"),
        ("transfer-alias", "aliased transfer source groups overlap"),
    ],
)
def test_loader_rejects_training_evaluation_overlap_after_positive_control(
    overlap: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _valid_temp_root(tmp_path / "study-root", monkeypatch)
    baseline = runtime_rule_study_inputs.load_study_inputs(root)
    relative = "data/processed/routing-data-v1/train.jsonl"
    path = root / relative
    old_eval = next(
        record
        for record in _retention_records()[0]
        if record.dataset_id == "dbpedia14-pilot-v1-development"
    )
    if overlap == "record-id":
        changed = _rewrite_jsonl(path, {"record_id": old_eval.record_id})
        pins = {relative: changed}
    elif overlap == "raw-group":
        changed = _rewrite_jsonl(path, {"source_group_id": old_eval.source_group_id})
        pins = {relative: changed}
    elif overlap == "semantic":
        train_rows = path.read_bytes().splitlines()
        training_request = json.loads(train_rows[0])["request"]
        dev_relative = "data/processed/routing-data-v1/development.jsonl"
        dev_path = root / dev_relative
        development_rows = dev_path.read_bytes().splitlines()
        training_menu_size = len(training_request["options"])
        development_index = next(
            index
            for index, row in enumerate(development_rows)
            if len(json.loads(row)["request"]["options"]) == training_menu_size
        )
        development_first = json.loads(development_rows[development_index])
        development_first["request"] = training_request
        development_first["answer_id"] = training_request["options"][0]["id"]
        development_rows[development_index] = json.dumps(
            development_first, ensure_ascii=False, separators=(",", ":")
        ).encode()
        changed = b"\n".join(development_rows) + b"\n"
        dev_path.write_bytes(changed)
        pins = {dev_relative: changed}
    else:
        aliased_group = next(
            str(row["source_group_id"])
            for row in baseline.retention_presentations
            if row["dataset_id"] == "boolq-dev-pilot-v1"
        )
        changed = _rewrite_jsonl(path, {"source_group_id": aliased_group})
        pins = {relative: changed}
    for changed_path, content in pins.items():
        _replace_new_file_hash(monkeypatch, changed_path, content)

    with pytest.raises(ValueError, match=message):
        runtime_rule_study_inputs.load_study_inputs(root)


def test_loader_rejects_cross_dataset_evaluation_group_overlap_after_positive_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _valid_temp_root(tmp_path / "study-root", monkeypatch)
    runtime_rule_study_inputs.load_study_inputs(root)
    relative = "data/processed/routing-data-v1/development.jsonl"
    path = root / relative
    old_group = next(
        record.source_group_id
        for record in _retention_records()[0]
        if record.dataset_id == "dbpedia14-pilot-v1-development"
    )
    content = _rewrite_jsonl(path, {"source_group_id": old_group})
    _replace_new_file_hash(monkeypatch, relative, content)

    with pytest.raises(ValueError, match="source groups overlap across data pools"):
        runtime_rule_study_inputs.load_study_inputs(root)
