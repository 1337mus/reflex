"""Prepare the pinned BoolQ and SNLI development pilot from local source files."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

from reflex_decisions import broader_data

DEFAULT_BOOLQ_SOURCE = Path("data/raw/boolq-validation.parquet")
DEFAULT_BOOLQ_DEV = Path("data/raw/boolq-dev.jsonl")
DEFAULT_SNLI_ARCHIVE = Path("data/raw/snli_1.0.zip")
DEFAULT_SNLI_DEV = Path("data/raw/snli_1.0_dev.jsonl")
DEFAULT_MANIFEST = Path("data/baselines/broader-dev-manifest.json")
DEFAULT_RECIPE = broader_data.DEFAULT_RECIPE_PATH
DEFAULT_OUTPUT = Path("data/processed/broader-dev-pilot-v1.jsonl")


def prepare_local_data(
    *,
    boolq_source_path: str | Path = DEFAULT_BOOLQ_SOURCE,
    boolq_dev_path: str | Path = DEFAULT_BOOLQ_DEV,
    snli_archive_path: str | Path = DEFAULT_SNLI_ARCHIVE,
    snli_dev_path: str | Path = DEFAULT_SNLI_DEV,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    recipe_path: str | Path = DEFAULT_RECIPE,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> str:
    """Verify source pins, reproduce selected records, and write once."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prepared data: {output}")

    boolq_source = Path(boolq_source_path).read_bytes()
    boolq_dev = Path(boolq_dev_path).read_bytes()
    snli_archive = Path(snli_archive_path).read_bytes()
    snli_dev = Path(snli_dev_path).read_bytes()
    broader_data.validate_source_pins(boolq_source, boolq_dev, snli_archive, snli_dev)
    records = broader_data.build_pilot_records(boolq_dev, snli_dev)
    payload = broader_data.serialize_records(records)
    records_digest = hashlib.sha256(payload).hexdigest()
    if records_digest != broader_data.EXPECTED_RECORDS_SHA256:
        raise ValueError("generated broader-data records do not match the approved recipe hash")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".broader-pilot-", dir=output.parent) as temp_dir:
        candidate_path = Path(temp_dir) / "records.jsonl"
        candidate_path.write_bytes(payload)
        broader_data.verify_prepared_data(candidate_path, manifest_path, recipe_path)

    broader_data.write_records_exclusive(output, payload)
    broader_data.verify_prepared_data(output, manifest_path, recipe_path)
    return records_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boolq-source", type=Path, default=DEFAULT_BOOLQ_SOURCE)
    parser.add_argument("--boolq-dev", type=Path, default=DEFAULT_BOOLQ_DEV)
    parser.add_argument("--snli-archive", type=Path, default=DEFAULT_SNLI_ARCHIVE)
    parser.add_argument("--snli-dev", type=Path, default=DEFAULT_SNLI_DEV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        records_digest = prepare_local_data(
            boolq_source_path=args.boolq_source,
            boolq_dev_path=args.boolq_dev,
            snli_archive_path=args.snli_archive,
            snli_dev_path=args.snli_dev,
            manifest_path=args.manifest,
            recipe_path=args.recipe,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared 64 BoolQ/SNLI development records; SHA-256 {records_digest}")


if __name__ == "__main__":
    main()
