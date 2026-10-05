"""Prepare the approved local COPA development pilot without network access."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

from reflex_decisions import baseline_data

DEFAULT_ARCHIVE = Path("data/raw/COPA-resources.tgz")
DEFAULT_DEV_XML = Path("data/raw/copa-dev.xml")
DEFAULT_MANIFEST = Path("data/baselines/copa-dev-manifest.json")
DEFAULT_OUTPUT = Path("data/processed/copa-dev-pilot-v1.jsonl")


def prepare_local_data(
    *,
    archive_path: str | Path = DEFAULT_ARCHIVE,
    dev_xml_path: str | Path = DEFAULT_DEV_XML,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
    source_revision: str = baseline_data.SOURCE_REVISION,
) -> str:
    """Verify pinned local inputs, audit the sample, and write it once."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prepared data: {output}")

    archive_bytes = Path(archive_path).read_bytes()
    xml_bytes = Path(dev_xml_path).read_bytes()
    baseline_data.validate_source_pins(archive_bytes, xml_bytes, source_revision=source_revision)
    records = baseline_data.build_pilot_records(xml_bytes)
    payload = baseline_data.serialize_records(records)
    records_digest = hashlib.sha256(payload).hexdigest()
    if records_digest != baseline_data.EXPECTED_RECORDS_SHA256:
        raise ValueError("generated COPA records do not match the approved recipe hash")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".copa-pilot-", dir=output.parent) as temp_dir:
        candidate_path = Path(temp_dir) / "records.jsonl"
        candidate_path.write_bytes(payload)
        baseline_data.verify_prepared_data(candidate_path, manifest_path)

    baseline_data.write_records_exclusive(output, payload)
    baseline_data.verify_prepared_data(output, manifest_path)
    return records_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--dev-xml", type=Path, default=DEFAULT_DEV_XML)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-revision", default=baseline_data.SOURCE_REVISION)
    args = parser.parse_args()
    try:
        records_digest = prepare_local_data(
            archive_path=args.archive,
            dev_xml_path=args.dev_xml,
            manifest_path=args.manifest,
            output_path=args.output,
            source_revision=args.source_revision,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared 32 COPA development records; SHA-256 {records_digest}")


if __name__ == "__main__":
    main()
