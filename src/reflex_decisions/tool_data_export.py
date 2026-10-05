"""Exclusive on-disk export for synthetic tool-choice candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from .data import DecisionRecord, audit_splits
from .tool_data import ToolCandidate
from .tool_data_audit import audit_tool_data
from .tool_data_spec import SPLITS, TEMPLATE_IDS, VERSION

_PINNED_SOURCES = (
    "src/reflex_decisions/tool_data_spec.py",
    "src/reflex_decisions/tool_data_identity.py",
    "src/reflex_decisions/tool_data_audit.py",
    "src/reflex_decisions/tool_data.py",
    "src/reflex_decisions/tool_data_export.py",
    "experiments/prepare_tool_data.py",
    "src/reflex_decisions/tool_rules.py",
    "src/reflex_decisions/tool_text.py",
    "src/reflex_decisions/tool_text_parser.py",
    "src/reflex_decisions/schema.py",
    "src/reflex_decisions/rendering.py",
    "src/reflex_decisions/data.py",
)


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    return (text + "\n").encode("utf-8")


def _jsonl(records: Iterable[DecisionRecord]) -> bytes:
    lines = (
        json.dumps(
            record.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in _PINNED_SOURCES
    }


def write_tool_candidate(candidate: ToolCandidate, output_dir: Path) -> dict[str, object]:
    """Re-audit and write one complete candidate to a new directory exclusively."""
    if not isinstance(candidate, ToolCandidate):
        raise TypeError("candidate must be a ToolCandidate")
    expected = dict(zip(SPLITS, candidate.config.counts, strict=True))
    actual = {split: sum(row.split == split for row in candidate.rows) for split in SPLITS}
    if actual != expected:
        raise ValueError("candidate split counts do not match its configuration")
    audit = audit_tool_data(candidate.rows, candidate.manifest)
    audit_splits(candidate.manifest, tuple(row.record for row in candidate.rows))
    if candidate != _fresh_candidate(candidate):
        raise ValueError("candidate does not match its generation recipe/config")

    counts = {split: actual[split] for split in SPLITS}
    recipe = {
        "version": VERSION,
        "seed": candidate.config.seed,
        "counts": counts,
        "template_ids": {split: TEMPLATE_IDS[split] for split in SPLITS},
        "source_hashes": _source_hashes(),
    }
    records_by_split = {
        split: tuple(row.record for row in candidate.rows if row.split == split) for split in SPLITS
    }
    files = {
        "manifest.json": _json_bytes(candidate.manifest.model_dump(mode="json"), pretty=True),
        "recipe.json": _json_bytes(recipe, pretty=True),
        "audit.json": _json_bytes(audit, pretty=True),
        "train.jsonl": _jsonl(records_by_split["train"]),
        "development.jsonl": _jsonl(records_by_split["development"]),
        "calibration.jsonl": _jsonl(records_by_split["calibration"]),
        "sealed/test.jsonl": _jsonl(records_by_split["test"]),
    }
    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    (target / "sealed").mkdir()
    for relative, content in files.items():
        with (target / relative).open("xb") as output:
            output.write(content)

    receipt_files: dict[str, dict[str, object]] = {
        relative: {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}
        for relative, content in files.items()
    }
    for split in SPLITS:
        relative = "sealed/test.jsonl" if split == "test" else f"{split}.jsonl"
        receipt_files[relative]["records"] = counts[split]
    receipt: dict[str, object] = {
        "version": VERSION,
        "counts": counts,
        "files": receipt_files,
    }
    with (target / "receipt.json").open("xb") as output:
        output.write(_json_bytes(receipt, pretty=True))
    return receipt


def _fresh_candidate(candidate: ToolCandidate) -> ToolCandidate:
    from .tool_data import generate_tool_data

    return generate_tool_data(candidate.config)
