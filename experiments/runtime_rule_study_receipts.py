"""Encode raw remote observations and install immutable host receipts."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from stat import S_ISDIR
from typing import cast

from experiments import runtime_rule_study_contracts as contracts
from reflex_decisions.smoke import sanitize_exception_message

RAW_SUMMARY_LIMIT = 512
_RAW_DEPTH_LIMIT = 64
_RAW_TAG = "$runtime_rule_study_raw"


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"[:128]


def _unknown_value(value: object, reason: str) -> dict[str, object]:
    try:
        representation = repr(value)
    except Exception:
        representation = "<repr unavailable>"
    try:
        sanitized = sanitize_exception_message(RuntimeError(representation))
    except Exception:
        sanitized = "<sanitized repr unavailable>"
    return {
        _RAW_TAG: {
            "kind": "summary",
            "type": _type_name(value),
            "reason": reason[:128],
            "repr": sanitized[:RAW_SUMMARY_LIMIT],
            "preservation_limit": "unknown value stored as a bounded sanitized summary",
        }
    }


def _encodes_as_utf8(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def encode_raw(value: object, active: set[int] | None = None, depth: int = 0) -> object:
    """Encode malformed remote values without pickle or lossy JSON coercion."""

    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        if _encodes_as_utf8(value):
            return value
        return {
            _RAW_TAG: {
                "kind": "string",
                "encoding": "python_ascii_repr",
                "value": ascii(value),
            }
        }
    if type(value) is int:
        try:
            str(value)
        except ValueError:
            return {_RAW_TAG: {"kind": "integer", "encoding": "hex", "value": hex(value)}}
        return value
    if type(value) is float:
        number = value
        if math.isfinite(number):
            return number
        label = "nan" if math.isnan(number) else ("infinity" if number > 0 else "-infinity")
        return {_RAW_TAG: {"kind": "float", "value": label}}
    if depth >= _RAW_DEPTH_LIMIT:
        return _unknown_value(value, "maximum encoding depth reached")

    seen = set() if active is None else active
    identity = id(value)
    if isinstance(value, Mapping) or type(value) is list:
        if identity in seen:
            return {_RAW_TAG: {"kind": "cycle", "type": _type_name(value)}}
        seen.add(identity)
        try:
            if isinstance(value, Mapping):
                entries = list(value.items())
                if all(type(key) is str and _encodes_as_utf8(key) for key, _item in entries):
                    return {
                        cast(str, key): encode_raw(item, seen, depth + 1) for key, item in entries
                    }
                return {
                    _RAW_TAG: {
                        "kind": "mapping",
                        "entries": [
                            [encode_raw(key, seen, depth + 1), encode_raw(item, seen, depth + 1)]
                            for key, item in entries
                        ],
                    }
                }
            return [encode_raw(item, seen, depth + 1) for item in cast(list[object], value)]
        except Exception:
            return _unknown_value(value, "container could not be traversed")
        finally:
            seen.remove(identity)
    return _unknown_value(value, "value is not JSON-compatible")


def write_receipt(path: Path, document: Mapping[str, object]) -> None:
    """Atomically install a private receipt without replacing prior evidence."""

    parent = path.parent
    parent_stat = parent.lstat()
    if parent.is_symlink() or not parent.is_dir() or not S_ISDIR(parent_stat.st_mode):
        raise ValueError("receipt parent must be a real directory")
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    data = contracts.canonical_json(encode_raw(document))
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("receipt write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
