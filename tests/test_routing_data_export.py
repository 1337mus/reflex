"""Tests for exclusive routing fixture exports."""

import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path

import pytest

from reflex_decisions.data import load_records
from reflex_decisions.routing_data import RoutingGenerationConfig, generate_routing_data
from reflex_decisions.routing_data_spec import SPLITS, VERSION


def test_routing_export_module_is_available() -> None:
    assert find_spec("reflex_decisions.routing_data_export") is not None


def test_exporter_exposes_the_candidate_writer() -> None:
    module = importlib.import_module("reflex_decisions.routing_data_export")
    assert hasattr(module, "write_routing_candidate")


def _candidate():
    return generate_routing_data(RoutingGenerationConfig(731, (2, 1, 1, 2)))


def test_export_separates_test_rows_and_records_hashes_and_counts(tmp_path: Path) -> None:
    module = importlib.import_module("reflex_decisions.routing_data_export")
    candidate = _candidate()
    output = tmp_path / "first"

    receipt = module.write_routing_candidate(candidate, output)

    expected_paths = {
        "manifest.json",
        "recipe.json",
        "audit.json",
        "train.jsonl",
        "development.jsonl",
        "calibration.jsonl",
        "sealed/test.jsonl",
        "receipt.json",
    }
    actual_paths = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    assert actual_paths == expected_paths
    expected_counts = {"train": 2, "development": 1, "calibration": 1, "test": 2}
    assert receipt["counts"] == expected_counts
    receipt_files = receipt["files"]
    assert isinstance(receipt_files, dict)
    assert "receipt.json" not in receipt_files
    for relative, entry in receipt_files.items():
        content = (output / relative).read_bytes()
        assert entry["path"] == relative
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
    assert {
        relative: entry["records"]
        for relative, entry in receipt_files.items()
        if relative.endswith(".jsonl")
    } == {
        "train.jsonl": 2,
        "development.jsonl": 1,
        "calibration.jsonl": 1,
        "sealed/test.jsonl": 2,
    }
    assert json.loads((output / "receipt.json").read_text()) == receipt
    sealed_records = load_records(output / "sealed/test.jsonl")
    expected_test_ids = [row.record.record_id for row in candidate.rows if row.split == "test"]
    assert [record.record_id for record in sealed_records] == expected_test_ids
    test_rows = tuple(row for row in candidate.rows if row.split == "test")
    other_files = tuple(
        path for path in output.rglob("*") if path.is_file() and path.name != "test.jsonl"
    )
    test_content_stays_sealed = all(
        row.record.record_id.encode() not in path.read_bytes()
        and row.record.request.context.encode() not in path.read_bytes()
        and row.record.request.question.encode() not in path.read_bytes()
        for row in test_rows
        for path in other_files
    )
    assert test_content_stays_sealed

    recipe = json.loads((output / "recipe.json").read_text())
    assert recipe["version"] == VERSION
    assert recipe["seed"] == 731
    assert recipe["counts"] == expected_counts
    assert recipe["template_ids"] == {
        "train": "routing-v1",
        "development": "routing-development-v1",
        "calibration": "routing-calibration-v1",
        "test": "routing-sealed-v1",
    }
    source_paths = (
        "src/reflex_decisions/routing_data_spec.py",
        "src/reflex_decisions/routing_data_audit.py",
        "src/reflex_decisions/routing_data.py",
        "src/reflex_decisions/routing_data_export.py",
        "experiments/prepare_routing_data.py",
        "src/reflex_decisions/routing_rules.py",
        "src/reflex_decisions/routing_text.py",
        "src/reflex_decisions/routing_text_parser.py",
        "src/reflex_decisions/schema.py",
        "src/reflex_decisions/rendering.py",
        "src/reflex_decisions/data.py",
    )
    project_root = Path(__file__).resolve().parents[1]
    assert recipe["source_hashes"] == {
        relative: hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
        for relative in source_paths
    }
    assert set(receipt["counts"]) == set(SPLITS)


def test_export_bytes_repeat_identically_in_fresh_directories(tmp_path: Path) -> None:
    module = importlib.import_module("reflex_decisions.routing_data_export")
    candidate = _candidate()
    left = tmp_path / "left"
    right = tmp_path / "right"

    module.write_routing_candidate(candidate, left)
    module.write_routing_candidate(candidate, right)

    left_paths = {path.relative_to(left).as_posix() for path in left.rglob("*") if path.is_file()}
    right_paths = {
        path.relative_to(right).as_posix() for path in right.rglob("*") if path.is_file()
    }
    assert left_paths == right_paths
    same_bytes = all(
        (left / relative).read_bytes() == (right / relative).read_bytes() for relative in left_paths
    )
    assert same_bytes


def test_export_refuses_existing_directory_without_touching_sentinel(tmp_path: Path) -> None:
    module = importlib.import_module("reflex_decisions.routing_data_export")
    target = tmp_path / "occupied"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module.write_routing_candidate(_candidate(), target)

    assert list(path.name for path in target.iterdir()) == ["sentinel"]
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("answer", "answer does not match"),
        ("group", "source group ID"),
        ("seed", "generation recipe"),
        ("data_kind", "generation recipe"),
        ("license", "generation recipe"),
    ],
)
def test_tampered_candidate_is_rejected_before_output_creation(
    tmp_path: Path, mutation: str, message: str
) -> None:
    module = importlib.import_module("reflex_decisions.routing_data_export")
    candidate = _candidate()
    row = candidate.rows[0]
    if mutation == "answer":
        wrong_answer = next(
            option.id for option in row.record.request.options if option.id != row.record.answer_id
        )
        record = row.record.model_copy(update={"answer_id": wrong_answer})
        broken = replace(candidate, rows=(replace(row, record=record), *candidate.rows[1:]))
    elif mutation == "group":
        record = row.record.model_copy(update={"source_group_id": "tampered-group"})
        broken = replace(candidate, rows=(replace(row, record=record), *candidate.rows[1:]))
    elif mutation == "seed":
        broken = replace(
            candidate, config=replace(candidate.config, seed=candidate.config.seed + 1)
        )
    elif mutation == "data_kind":
        broken_manifest = candidate.manifest.model_copy(update={"data_kind": "benchmark"})
        broken = replace(candidate, manifest=broken_manifest)
    else:
        datasets = list(candidate.manifest.datasets)
        datasets[0] = datasets[0].model_copy(update={"license": "MIT"})
        broken_manifest = candidate.manifest.model_copy(update={"datasets": tuple(datasets)})
        broken = replace(candidate, manifest=broken_manifest)
    target = tmp_path / "rejected"

    with pytest.raises(ValueError, match=message):
        module.write_routing_candidate(broken, target)
    assert not target.exists()


def test_export_rejects_split_count_mismatch_before_output_creation(tmp_path: Path) -> None:
    module = importlib.import_module("reflex_decisions.routing_data_export")
    candidate = _candidate()
    broken = replace(candidate, rows=candidate.rows[:-1])
    target = tmp_path / "wrong-count"

    with pytest.raises(ValueError, match="split counts"):
        module.write_routing_candidate(broken, target)
    assert not target.exists()


def test_export_preserves_partial_outputs_after_a_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("reflex_decisions.routing_data_export")
    target = tmp_path / "partial"
    original_open = Path.open

    def injected_open(path: Path, *args: object, **kwargs: object):
        if path == target / "development.jsonl":
            raise OSError("injected output failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", injected_open)
    with pytest.raises(OSError, match="injected output failure"):
        module.write_routing_candidate(_candidate(), target)

    assert (target / "manifest.json").is_file()
    assert (target / "recipe.json").is_file()
    assert (target / "train.jsonl").is_file()
    assert not (target / "development.jsonl").exists()


def test_cli_prints_only_the_aggregate_receipt(tmp_path: Path) -> None:
    target = tmp_path / "cli-output"
    project_root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.prepare_routing_data",
            "--seed",
            "928",
            "--output",
            str(target),
            "--counts",
            "1",
            "1",
            "1",
            "1",
        ],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert process.returncode == 0
    assert process.stderr == ""
    printed = json.loads(process.stdout)
    assert printed["counts"] == {split: 1 for split in SPLITS}
    assert "files" in printed and "rows" not in printed
