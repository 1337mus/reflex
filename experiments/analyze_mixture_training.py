"""Validate and analyze a saved controlled mixture-training receipt."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from experiments import mixture_training_analysis as analysis
from experiments import mixture_training_baselines as baselines
from experiments import mixture_training_core as core
from experiments.mixture_training_contracts import strict_json_loads
from reflex_decisions import smoke

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(root: Path, value: Path) -> Path:
    expanded = value.expanduser()
    return (expanded if expanded.is_absolute() else root / expanded).resolve()


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = strict_json_loads(path.read_bytes())
    except (OSError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is unavailable or not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _overall_gate_passed(result: Mapping[str, object]) -> bool:
    gate = result.get("engineering_gate")
    if not isinstance(gate, Mapping) or type(gate.get("overall_passed")) is not bool:
        raise ValueError("analysis engineering-gate summary is malformed")
    return gate["overall_passed"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True, help="saved mixture-run receipt")
    parser.add_argument("--output", type=Path, required=True, help="new analysis JSON path")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository data root")
    parser.add_argument("--real-receipt", type=Path, default=Path(baselines.SAVED_REAL_RECEIPT))
    parser.add_argument(
        "--balanced-receipt", type=Path, default=Path(baselines.SAVED_BALANCED_RECEIPT)
    )
    args = parser.parse_args(argv)

    try:
        root = args.root.expanduser().resolve(strict=True)
        receipt_path = _resolve(root, args.receipt)
        output_path = _resolve(root, args.output)
        real_receipt_path = _resolve(root, args.real_receipt)
        balanced_receipt_path = _resolve(root, args.balanced_receipt)
        with smoke.reserve_output(output_path) as reservation:
            receipt = _read_json(receipt_path, "mixture receipt")
            real_receipt = _read_json(real_receipt_path, "saved real-pilot receipt")
            balanced_receipt = _read_json(balanced_receipt_path, "saved balanced-SNLI receipt")
            real_records, balanced_records, synthetic_records, _pins = core.load_local_data(root)
            result = analysis.analyze_experiment(
                receipt,
                real_records,
                balanced_records,
                synthetic_records,
                real_receipt,
                balanced_receipt,
                root=root,
            )
            status = result.get("status")
            if status not in {"passed", "failed"}:
                raise ValueError("analysis result status is malformed")
            saved = dict(result)
            saved["overall_gate_passed"] = _overall_gate_passed(result)
            smoke.write_json_artifact(reservation, saved)
    except (OSError, ValueError) as exc:
        message = smoke.sanitize_exception_message(exc)
        print(f"analysis failed: {message}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": status,
                "artifact": str(output_path),
                "overall_gate_passed": saved["overall_gate_passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
