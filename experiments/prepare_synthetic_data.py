"""Prepare solver-labeled synthetic data without model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reflex_decisions.synthetic_data import (
    DEFAULT_SCENARIO_COUNTS,
    DEFAULT_SEED,
    MAX_SCENARIOS_PER_FAMILY_SPLIT,
    SPLITS,
    GenerationConfig,
    build_candidate,
    write_candidate,
)

DEFAULT_OUTPUT = Path("data/processed/synthetic-seed-v1")


def _integer(value: str, minimum: int, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed < minimum or (maximum is not None and parsed > maximum):
        limit = f"between {minimum} and {maximum}" if maximum else f"at least {minimum}"
        raise argparse.ArgumentTypeError(f"value must be {limit}")
    return parsed


def _counts(args: argparse.Namespace, family: str) -> tuple[int, int, int]:
    return tuple(getattr(args, f"{family}_{split}_scenarios") for split in SPLITS)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=lambda value: _integer(value, 0), default=DEFAULT_SEED)
    for family in ("fact", "numeric"):
        for split, default in zip(SPLITS, DEFAULT_SCENARIO_COUNTS, strict=True):
            parser.add_argument(
                f"--{family}-{split}-scenarios",
                dest=f"{family}_{split}_scenarios",
                type=lambda value: _integer(value, 1, MAX_SCENARIOS_PER_FAMILY_SPLIT),
                default=default,
            )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true", help="write to a fresh directory")
    args = parser.parse_args(argv)
    try:
        config = GenerationConfig(args.seed, _counts(args, "fact"), _counts(args, "numeric"))
        candidate = build_candidate(config)
        output = (
            {
                "mode": "written",
                "output_dir": str(args.output_dir),
                "report": write_candidate(candidate, args.output_dir),
            }
            if args.write
            else {
                "mode": "plan-only",
                "output_dir": str(args.output_dir),
                "seed": config.seed,
                "recipe": candidate.recipe,
                "audit": candidate.audit,
            }
        )
    except (OSError, ValueError) as exc:
        print(f"synthetic preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
