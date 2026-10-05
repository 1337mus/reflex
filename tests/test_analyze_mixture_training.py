from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import analyze_mixture_training as cli
from experiments import mixture_training_analysis as analysis
from experiments import mixture_training_baselines as baselines
from experiments import mixture_training_core as core

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _inputs(root: Path, *, receipt: str = '{"experiment_id":"cli-check"}') -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "receipt.json"
    receipt_path.write_text(receipt, encoding="utf-8")
    for relative, value in (
        (baselines.SAVED_REAL_RECEIPT, {"kind": "real"}),
        (baselines.SAVED_BALANCED_RECEIPT, {"kind": "balanced"}),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    return receipt_path, root / "analysis.json"


def _arguments(root: Path, receipt: Path, output: Path) -> list[str]:
    return ["--root", str(root), "--receipt", str(receipt), "--output", str(output)]


def test_cli_reserves_output_loads_inputs_and_reports_gate_separately(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "project"
    receipt_path, output = _inputs(root)
    records = (("real-record",), ("balanced-record",), ("synthetic-record",), {"pin": "x"})
    events: list[str] = []

    def load_local_data(given_root):
        assert Path(given_root) == root.resolve()
        assert (output.parent / f".{output.name}.reflex-smoke.lock").is_file()
        events.append("load")
        return records

    def analyze(
        receipt,
        real_records,
        balanced_records,
        synthetic_records,
        real_receipt,
        balanced_receipt,
        *,
        root,
    ):
        assert receipt == {"experiment_id": "cli-check"}
        assert (real_records, balanced_records, synthetic_records) == records[:3]
        assert real_receipt == {"kind": "real"}
        assert balanced_receipt == {"kind": "balanced"}
        assert Path(root) == tmp_path / "project"
        assert (output.parent / f".{output.name}.reflex-smoke.lock").is_file()
        events.append("analyze")
        return {"status": "passed", "engineering_gate": {"overall_passed": False}}

    monkeypatch.setattr(core, "load_local_data", load_local_data)
    monkeypatch.setattr(analysis, "analyze_experiment", analyze)

    status = cli.main(_arguments(root, receipt_path, output))

    saved = json.loads(output.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert status == 0
    assert events == ["load", "analyze"]
    assert saved["status"] == "passed"
    assert saved["overall_gate_passed"] is False
    assert stdout == {
        "artifact": str(output.resolve()),
        "overall_gate_passed": False,
        "status": "passed",
    }


def test_cli_persists_failed_pair_without_evaluating_gates(tmp_path, monkeypatch):
    root = tmp_path / "project"
    receipt_path, output = _inputs(root)
    monkeypatch.setattr(core, "load_local_data", lambda _root: ((), (), (), {}))
    monkeypatch.setattr(
        analysis,
        "analyze_experiment",
        lambda *_args, **_kwargs: {
            "status": "failed",
            "engineering_gate": {"evaluated": False, "overall_passed": False},
        },
    )

    status = cli.main(_arguments(root, receipt_path, output))

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert status == 1
    assert saved["status"] == "failed"
    assert saved["engineering_gate"]["evaluated"] is False
    assert saved["overall_gate_passed"] is False


def test_cli_rejects_existing_output_before_analysis(tmp_path, monkeypatch, capsys):
    root = tmp_path / "project"
    receipt_path, output = _inputs(root)
    output.write_text("keep existing evidence\n", encoding="utf-8")
    monkeypatch.setattr(core, "load_local_data", lambda _root: pytest.fail("analysis started"))

    status = cli.main(_arguments(root, receipt_path, output))

    assert status != 0
    assert output.read_text(encoding="utf-8") == "keep existing evidence\n"
    assert "analysis failed" in capsys.readouterr().err


def test_cli_rejects_duplicate_json_keys_without_starting_analysis(tmp_path, monkeypatch):
    root = tmp_path / "project"
    receipt_path, output = _inputs(
        root, receipt='{"experiment_id":"first","experiment_id":"second"}'
    )
    monkeypatch.setattr(core, "load_local_data", lambda _root: pytest.fail("analysis started"))

    status = cli.main(_arguments(root, receipt_path, output))

    assert status != 0
    assert not output.exists()


def test_cli_imports_without_model_or_modal_packages() -> None:
    script = """
import sys
from experiments import analyze_mixture_training as cli
try:
    cli.main(["--help"])
except SystemExit as error:
    assert error.code == 0
forbidden = ("torch", "transformers", "peft", "modal")
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
