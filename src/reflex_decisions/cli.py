"""Offline command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .data import audit_splits, load_manifest, load_records
from .evaluation import evaluate, load_predictions, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reflex", description="Validate and evaluate offline decisions."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate decision records and split lineage")
    validate.add_argument("--records", required=True, type=Path)
    validate.add_argument("--manifest", required=True, type=Path)

    run = commands.add_parser("evaluate", help="score saved candidate logits against gold records")
    run.add_argument("--records", required=True, type=Path)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--predictions", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--calibration-id")
    run.add_argument("--min-confidence", type=float)
    return parser


def _ensure_output_is_separate(output: Path, inputs: Sequence[Path]) -> None:
    output_resolved = output.resolve()
    for source in inputs:
        source_resolved = source.resolve()
        if output_resolved == source_resolved:
            raise ValueError("report output must not overwrite an input file")
        if output.exists() and source.exists():
            try:
                if os.path.samefile(output, source):
                    raise ValueError("report output must not overwrite an input file")
            except OSError as exc:
                raise ValueError("could not verify report output is separate from inputs") from exc


def _write_atomic(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _run_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    records = load_records(args.records)
    summary = audit_splits(manifest, records)
    split_summary = ", ".join(
        f"{split}={count}" for split, count in summary.split_counts if count > 0
    )
    print(
        f"Validated {summary.record_count} records across {summary.dataset_count} datasets "
        f"({summary.data_kind}; {split_summary})."
    )
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    _ensure_output_is_separate(args.output, (args.records, args.manifest, args.predictions))
    manifest = load_manifest(args.manifest)
    records = load_records(args.records)
    predictions = load_predictions(args.predictions)
    report = evaluate(
        records,
        predictions,
        manifest,
        records_sha256=sha256_file(args.records),
        predictions_sha256=sha256_file(args.predictions),
        manifest_sha256=sha256_file(args.manifest),
        temperature=args.temperature,
        calibration_id=args.calibration_id,
        min_confidence=args.min_confidence,
    )
    content = json.dumps(
        report.model_dump(mode="json"), indent=2, ensure_ascii=False, allow_nan=False
    )
    _write_atomic(args.output, content)
    print(
        f"Evaluated {report.overall.record_count} {report.data_kind} records with "
        f"{report.model_revision}: accuracy={report.overall.micro_accuracy:.3f}, "
        f"coverage={report.overall.coverage:.3f}; report written to {args.output}."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "evaluate":
            return _run_evaluate(args)
        parser.error("unknown command")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
