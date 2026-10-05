"""Prepare a solver-labelled support-routing candidate without model inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from reflex_decisions.routing_data import RoutingGenerationConfig, generate_routing_data
from reflex_decisions.routing_data_export import write_routing_candidate


def _bounded_integer(value: str, field: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{field} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{field} is outside its allowed range")
    return parsed


def _seed(value: str) -> int:
    return _bounded_integer(value, "seed", 0, 2**64 - 1)


def _count(value: str) -> int:
    return _bounded_integer(value, "count", 1, 112)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=_seed, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=_count, nargs=4, default=(56, 14, 14, 28))
    args = parser.parse_args(argv)
    try:
        candidate = generate_routing_data(RoutingGenerationConfig(args.seed, tuple(args.counts)))
        receipt = write_routing_candidate(candidate, args.output)
    except Exception:
        print("routing data preparation failed", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
