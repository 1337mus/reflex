"""Preparation-only readers for the pinned real pilot source files."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .pilot_data import DbpediaRow, SmsRow

if TYPE_CHECKING:
    from .broader_data import SnliItem


DBPEDIA_ROWS = 560000
SMS_ROWS = 5574
SMS_LABEL_COUNTS = {"ham": 4827, "spam": 747}
SNLI_ROWS = 10000
DUCKDB_VERSION = "1.5.6"


def parse_sms_collection(payload: bytes) -> tuple[SmsRow, ...]:
    """Parse the local UCI tab-separated SMS member with zero-based native row IDs."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SMS source is not valid UTF-8") from exc
    rows: list[SmsRow] = []
    for row_id, line in enumerate(text.splitlines()):
        label, separator, message = line.partition("\t")
        if not separator or label not in {"ham", "spam"} or not message.strip():
            raise ValueError("SMS source row has an invalid label or message")
        rows.append(SmsRow(row_id, label, message))
    if not rows:
        raise ValueError("SMS source is empty")
    return tuple(rows)


def verify_source_file(path: str | Path, expected_sha256: str, label: str) -> None:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"{label} source file is unavailable") from exc
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"{label} source SHA-256 mismatch")


def read_dbpedia_rows(path: str | Path, expected_sha256: str) -> Iterator[DbpediaRow]:
    """Read the pinned Parquet train file with DuckDB's stable file row numbers."""

    verify_source_file(path, expected_sha256, "DBpedia train")
    try:
        duckdb = importlib.import_module("duckdb")
    except ImportError as exc:
        raise ValueError("DBpedia preparation requires the approved DuckDB runtime") from exc
    if duckdb.__version__ != DUCKDB_VERSION:
        raise ValueError("DBpedia preparation DuckDB version does not match the approved reader")

    def rows() -> Iterator[DbpediaRow]:
        row_count = 0
        label_counts: Counter[int] = Counter()
        connection = duckdb.connect(database=":memory:")
        try:
            cursor = connection.execute(
                "SELECT file_row_number, title, content, label "
                "FROM read_parquet(?, file_row_number=true)",
                [str(path)],
            )
            while batch := cursor.fetchmany(4096):
                for source_row_id, title, content, label in batch:
                    if not isinstance(title, str) or not isinstance(content, str):
                        raise ValueError("DBpedia source row has invalid text fields")
                    row = DbpediaRow(int(source_row_id), title, content, int(label))
                    row_count += 1
                    label_counts[row.label] += 1
                    yield row
            if row_count != DBPEDIA_ROWS or label_counts != Counter({i: 40000 for i in range(14)}):
                raise ValueError("DBpedia source row or class counts do not match the audit")
        finally:
            connection.close()

    return rows()


def read_sms_rows(
    text_path: str | Path,
    archive_path: str | Path,
    *,
    text_sha256: str,
    archive_sha256: str,
) -> tuple[SmsRow, ...]:
    """Verify local UCI source receipts and parse the extracted member."""

    verify_source_file(archive_path, archive_sha256, "SMS archive")
    verify_source_file(text_path, text_sha256, "SMS extracted member")
    try:
        rows = parse_sms_collection(Path(text_path).read_bytes())
    except OSError as exc:
        raise ValueError("SMS extracted source file is unavailable") from exc
    if len(rows) != SMS_ROWS or Counter(row.label for row in rows) != Counter(SMS_LABEL_COUNTS):
        raise ValueError("SMS source row or label counts do not match the audit")
    return rows


def parse_snli_native_row_ids(
    payload: bytes,
    *,
    expected_rows: int | None = None,
) -> tuple[dict[str, int], int, int]:
    """Map labeled pairIDs to their first native row and count omitted unlabeled rows."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("SNLI development source is not valid UTF-8") from exc
    source_row_ids: dict[str, int] = {}
    unlabeled_count = 0
    row_count = 0
    for row_id, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("SNLI development source contains malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError("SNLI development source rows must be JSON objects")
        row_count += 1
        if row.get("gold_label") == "-":
            unlabeled_count += 1
            continue
        pair_id = row.get("pairID")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError("SNLI development source row has a blank pairID")
        pair_id = pair_id.strip()
        source_row_ids[pair_id] = min(row_id, source_row_ids.get(pair_id, row_id))
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError("SNLI development row count does not match the source receipt")
    return source_row_ids, unlabeled_count, row_count


def parse_snli_items(payload: bytes) -> tuple[SnliItem, ...]:
    """Use the established broader-data parser for SNLI groups and normalization."""

    from .broader_data import parse_snli_dev_jsonl

    return parse_snli_dev_jsonl(payload)
