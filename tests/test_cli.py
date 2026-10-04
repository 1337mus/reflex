import json
import os
import subprocess
import sys
from pathlib import Path

from reflex_decisions import cli

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "data" / "fixtures"


def test_validate_command_reports_fixture_summary(capsys) -> None:
    status = cli.main(
        [
            "validate",
            "--records",
            str(FIXTURES / "records.jsonl"),
            "--manifest",
            str(FIXTURES / "manifest.json"),
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "9 records" in captured.out
    assert "fixture" in captured.out


def test_subprocess_evaluate_writes_json_report(tmp_path) -> None:
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "reflex_decisions",
            "evaluate",
            "--records",
            str(FIXTURES / "records.jsonl"),
            "--manifest",
            str(FIXTURES / "manifest.json"),
            "--predictions",
            str(FIXTURES / "predictions.jsonl"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data_kind"] == "fixture"
    assert report["evaluation_split"] == "test"
    assert report["overall"]["record_count"] == 9
    assert "Evaluated 9 fixture records" in completed.stdout


def test_evaluate_refuses_input_alias_and_hardlink(tmp_path, capsys) -> None:
    records = FIXTURES / "records.jsonl"
    manifest = FIXTURES / "manifest.json"
    predictions = FIXTURES / "predictions.jsonl"
    original = records.read_bytes()

    alias_status = cli.main(
        [
            "evaluate",
            "--records",
            str(records),
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--output",
            str(records),
        ]
    )
    assert alias_status == 2
    assert records.read_bytes() == original
    assert "must not overwrite an input" in capsys.readouterr().err

    hardlink = tmp_path / "predictions-hardlink.jsonl"
    os.link(predictions, hardlink)
    hardlink_contents = hardlink.read_bytes()
    hardlink_status = cli.main(
        [
            "evaluate",
            "--records",
            str(records),
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--output",
            str(hardlink),
        ]
    )
    assert hardlink_status == 2
    assert hardlink.read_bytes() == hardlink_contents
    assert predictions.read_bytes() == hardlink_contents
    assert "must not overwrite an input" in capsys.readouterr().err


def test_malformed_input_fails_without_creating_report(tmp_path, capsys) -> None:
    malformed = tmp_path / "bad-records.jsonl"
    malformed.write_text('{"record_id":"private payload"\n', encoding="utf-8")
    output = tmp_path / "must-not-exist.json"

    status = cli.main(
        [
            "evaluate",
            "--records",
            str(malformed),
            "--manifest",
            str(FIXTURES / "manifest.json"),
            "--predictions",
            str(FIXTURES / "predictions.jsonl"),
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert not output.exists()
    error = capsys.readouterr().err
    assert "malformed JSON" in error
    assert "private payload" not in error
