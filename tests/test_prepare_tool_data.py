"""Tests for the bounded tool-choice preparation command."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import experiments.prepare_tool_data as cli
from experiments.prepare_tool_data import main


def test_cli_rejects_invalid_seed_before_creating_output(tmp_path) -> None:
    output = tmp_path / "never-created"

    with pytest.raises(SystemExit) as exit_info:
        main(["--seed", "-1", "--output", str(output)])

    assert exit_info.value.code == 2
    assert not output.exists()


def test_cli_exports_toy_candidate_and_prints_only_receipt(tmp_path: Path) -> None:
    output = tmp_path / "cli-data"
    project_root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.prepare_tool_data",
            "--seed",
            "928",
            "--output",
            str(output),
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
    receipt = json.loads(process.stdout)
    assert receipt["counts"] == {
        split: 1 for split in ("train", "development", "calibration", "test")
    }
    assert "files" in receipt and "rows" not in receipt


def test_cli_sanitizes_generation_or_export_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "generate_tool_data", lambda _config: "candidate")

    def fail_with_sensitive_detail(_candidate: object, _output: Path) -> dict[str, object]:
        raise ValueError("synthetic query leaked in detail")

    monkeypatch.setattr(cli, "write_tool_candidate", fail_with_sensitive_detail)
    status = main(["--seed", "13", "--output", str(tmp_path / "never-created")])

    assert status == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "tool data preparation failed\n"
    assert "synthetic query" not in captured.err
