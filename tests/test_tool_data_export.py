"""Tests for exclusive tool-choice fixture exports."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from reflex_decisions.data import load_records
from reflex_decisions.rendering import render_prompt
from reflex_decisions.tool_data import (
    ToolCandidate,
    ToolGenerationConfig,
    generate_tool_data,
)
from reflex_decisions.tool_data_export import write_tool_candidate
from reflex_decisions.tool_data_spec import SPLITS
from reflex_decisions.tool_rules import solve
from reflex_decisions.tool_text_parser import parse_tool_prompt


def _candidate() -> ToolCandidate:
    config = ToolGenerationConfig(731, (1, 1, 1, 1))
    return generate_tool_data(config)


def test_export_writes_split_pools_with_hashes_and_counts(tmp_path: Path) -> None:
    candidate = _candidate()
    output = tmp_path / "tool-data"

    receipt = write_tool_candidate(candidate, output)

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
    assert receipt["counts"] == {split: 1 for split in SPLITS}
    receipt_files = receipt["files"]
    assert isinstance(receipt_files, dict)
    assert "receipt.json" not in receipt_files
    for relative, entry in receipt_files.items():
        content = (output / relative).read_bytes()
        assert entry["path"] == relative
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
    assert json.loads((output / "receipt.json").read_text()) == receipt
    for split in SPLITS:
        saved_path = output / ("sealed/test.jsonl" if split == "test" else f"{split}.jsonl")
        for record in load_records(saved_path):
            parsed = parse_tool_prompt(render_prompt(record.request))
            assert record.answer_id == solve(parsed.scenario)
    test_rows = tuple(row for row in candidate.rows if row.split == "test")
    other_files = tuple(
        path
        for path in output.rglob("*")
        if path.is_file() and path != output / "sealed/test.jsonl"
    )
    for row in test_rows:
        for path in other_files:
            content = path.read_bytes()
            assert row.record.record_id.encode() not in content
            assert row.record.request.context.encode() not in content
            assert row.record.request.question.encode() not in content
    recipe = json.loads((output / "recipe.json").read_text())
    assert recipe["source_hashes"] == {
        relative: hashlib.sha256(
            (Path(__file__).resolve().parents[1] / relative).read_bytes()
        ).hexdigest()
        for relative in (
            "src/reflex_decisions/tool_data_spec.py",
            "src/reflex_decisions/tool_data_identity.py",
            "src/reflex_decisions/tool_data_audit.py",
            "src/reflex_decisions/tool_data.py",
            "src/reflex_decisions/tool_data_export.py",
            "experiments/prepare_tool_data.py",
            "src/reflex_decisions/tool_rules.py",
            "src/reflex_decisions/tool_text.py",
            "src/reflex_decisions/tool_text_parser.py",
            "src/reflex_decisions/schema.py",
            "src/reflex_decisions/rendering.py",
            "src/reflex_decisions/data.py",
        )
    }


def test_export_refuses_existing_directory_without_changing_it(tmp_path: Path) -> None:
    target = tmp_path / "occupied"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_tool_candidate(_candidate(), target)

    assert list(path.name for path in target.iterdir()) == ["keep.txt"]
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_export_reaudits_candidate_before_creating_output(tmp_path: Path) -> None:
    candidate = _candidate()
    row = candidate.rows[0]
    wrong_answer = next(
        option.id for option in row.record.request.options if option.id != row.record.answer_id
    )
    changed_record = row.record.model_copy(update={"answer_id": wrong_answer})
    changed_rows = (replace(row, record=changed_record), *candidate.rows[1:])
    target = tmp_path / "rejected"

    with pytest.raises(ValueError, match="record answer does not match the tool solver"):
        write_tool_candidate(replace(candidate, rows=changed_rows), target)

    assert not target.exists()


@pytest.mark.parametrize("mutation", ["seed", "data_kind", "license"])
def test_export_rejects_false_recipe_or_manifest_metadata(tmp_path: Path, mutation: str) -> None:
    candidate = _candidate()
    if mutation == "seed":
        broken = replace(candidate, config=replace(candidate.config, seed=732))
    else:
        manifest = candidate.manifest
        if mutation == "data_kind":
            manifest = manifest.model_copy(update={"data_kind": "benchmark"})
        else:
            datasets = list(manifest.datasets)
            datasets[0] = datasets[0].model_copy(update={"license": "MIT"})
            manifest = manifest.model_copy(update={"datasets": tuple(datasets)})
        broken = replace(candidate, manifest=manifest)
    target = tmp_path / f"rejected-{mutation}"

    with pytest.raises(ValueError, match="candidate does not match its generation recipe/config"):
        write_tool_candidate(broken, target)

    assert not target.exists()


def test_export_preserves_partial_evidence_after_a_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _candidate()
    target = tmp_path / "partial"
    original_open = Path.open

    def injected_open(path: Path, *args: object, **kwargs: object):
        if path == target / "development.jsonl":
            raise OSError("injected export failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", injected_open)
    with pytest.raises(OSError, match="injected export failure"):
        write_tool_candidate(candidate, target)

    assert (target / "manifest.json").is_file()
    assert (target / "recipe.json").is_file()
    assert (target / "train.jsonl").is_file()
    assert not (target / "development.jsonl").exists()
