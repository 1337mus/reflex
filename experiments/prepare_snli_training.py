"""Prepare a pinned private SNLI training candidate; write only with --write."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

from reflex_decisions.snli_training_data import (
    DEFAULT_ARCHIVE_PATH,
    DEFAULT_OUTPUT_DIR,
    SnliTrainingCandidate,
    build_snli_training_candidate,
    write_snli_training_candidate,
)

DEFAULT_EVIDENCE_PATH = Path(".context/snli-training-build-evidence.json")


def _summary(
    candidate: SnliTrainingCandidate, output_dir: Path, *, written: bool
) -> dict[str, object]:
    return {
        "record_count": len(candidate.records),
        "label_counts": candidate.recipe["counts"]["selected"]["label_counts"],
        "counts": candidate.recipe["counts"],
        "output_hashes": {
            **{
                name: candidate.recipe["outputs"][f"{name}_sha256"]
                for name in ("records", "manifest")
            },
            "recipe": hashlib.sha256(candidate.recipe_bytes).hexdigest(),
        },
        "output_dir": str(output_dir) if written else None,
        "written": written,
    }


def _write_evidence(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write records, manifest, and recipe into a new exclusive output directory",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    candidate = build_snli_training_candidate(args.archive)
    written = False
    if args.write:
        destination = write_snli_training_candidate(candidate, args.output_dir)
        elapsed_seconds = round(time.perf_counter() - started, 3)
        evidence = {
            "archive_path": str(args.archive.resolve()),
            "output_dir": str(destination.resolve()),
            "elapsed_seconds": elapsed_seconds,
            "output_hashes": _summary(candidate, destination, written=True)["output_hashes"],
            "counts": candidate.recipe["counts"],
        }
        _write_evidence(args.evidence, evidence)
        written = True
    print(
        json.dumps(_summary(candidate, args.output_dir, written=written), indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
