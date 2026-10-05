"""Prepare the balanced SNLI diagnostic with a plan-only default."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from reflex_decisions.snli_diagnostic import (
    DEFAULT_OUTPUT_DIR,
    build_candidate,
    write_candidate_files,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write", action="store_true", help="write to a new output directory")
    args = parser.parse_args(argv)

    if args.write and (args.output_dir.exists() or args.output_dir.is_symlink()):
        raise FileExistsError(f"output directory must not already exist: {args.output_dir}")
    candidate = build_candidate(args.repo_root)
    if args.write:
        write_candidate_files(candidate, args.output_dir)
    print(
        json.dumps(
            {
                "mode": "write" if args.write else "plan_only",
                "dataset_id": candidate.bundle.manifest.datasets[0].dataset_id,
                "record_count": len(candidate.bundle.records),
                "records_sha256": hashlib.sha256(candidate.records_bytes).hexdigest(),
                "manifest_sha256": hashlib.sha256(candidate.manifest_bytes).hexdigest(),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
