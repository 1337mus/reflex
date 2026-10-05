from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments import mixture_training_contracts as contracts


def test_validate_pins_rejects_tampered_incorporated_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = "protocols/original.md"
    seed2_path = "protocols/seed2.md"
    original_sha256 = "c" * 64
    seed2_sha256 = "a" * 64
    monkeypatch.setattr(contracts, "PROTOCOL_PATH", seed2_path)
    monkeypatch.setattr(contracts, "EXPECTED_PROTOCOL_SHA256", seed2_sha256)
    monkeypatch.setattr(contracts, "ORIGINAL_PROTOCOL_PATH", original_path)
    monkeypatch.setattr(contracts, "ORIGINAL_PROTOCOL_SHA256", original_sha256)
    monkeypatch.setattr(contracts, "SOURCE_FINGERPRINT_PATHS", (seed2_path, original_path))
    monkeypatch.setattr(contracts, "DATA_FILE_SHA256", {"fixture/records.jsonl": "b" * 64})
    pins = {
        "data_file_sha256": {"fixture/records.jsonl": "b" * 64},
        "protocol_sha256": seed2_sha256,
        "source_file_sha256": {
            seed2_path: seed2_sha256,
            original_path: "0" * 64,
        },
    }

    with pytest.raises(ValueError, match="incorporated original protocol"):
        contracts.validate_pins(pins)


def test_verify_protocol_rejects_tampered_incorporated_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    seed2_path = docs / "seed2.md"
    original_path = docs / "original.md"
    seed2_path.write_text("candidate protocol", encoding="utf-8")
    original_path.write_text("tampered original", encoding="utf-8")
    monkeypatch.setattr(contracts, "ORIGINAL_PROTOCOL_PATH", "docs/original.md")
    monkeypatch.setattr(
        contracts,
        "ORIGINAL_PROTOCOL_SHA256",
        hashlib.sha256(b"expected original").hexdigest(),
    )
    monkeypatch.setattr(
        contracts,
        "EXPECTED_PROTOCOL_SHA256",
        hashlib.sha256(b"candidate protocol").hexdigest(),
    )

    with pytest.raises(ValueError, match="incorporated original protocol"):
        contracts.verify_protocol(seed2_path)

    original_path.write_text("expected original", encoding="utf-8")
    assert (
        contracts.verify_protocol(seed2_path) == hashlib.sha256(b"candidate protocol").hexdigest()
    )
