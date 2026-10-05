"""Prepare the frozen private fresh-evaluation data bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    args = parser.parse_args(argv)
    try:
        from reflex_decisions.fresh_eval_data import prepare_panel

        root = args.root.expanduser().resolve(strict=True)
        records, metadata = prepare_panel(root)
    except (OSError, ValueError) as exc:
        print(f"preparation failed: {type(exc).__name__}", file=sys.stderr)
        return 2
    summary = {
        "status": "prepared",
        "record_count": len(records),
        "counts": metadata["counts"],
        "gpu_pins": metadata["gpu_pins"],
        "manifest_sha256": metadata["manifest_sha256"],
        "recipe_sha256": metadata["recipe_sha256"],
        "record_metadata_sha256": metadata["record_metadata_sha256"],
        "selection_digest_sha256": metadata["selection"]["selection_digest_sha256"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
