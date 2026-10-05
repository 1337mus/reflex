from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments import mixture_training_contracts as contracts


def test_natural_reasoning_protocol_is_the_only_runtime_protocol_pin() -> None:
    assert contracts.SCHEMA_VERSION == 2
    assert contracts.PROTOCOL_PATH == "docs/natural-reasoning-protocol.md"
    assert contracts.HISTORICAL_PROTOCOL_SHA256 == {
        "docs/mixture-training-protocol.md": (
            "06d3014a45ebeb19500ba2ab002f0595c42b44e561fbb9efc329af0d02a1030c"
        ),
        "docs/mixture-training-seed2-protocol.md": (
            "2cf62fa31f3e52a41ebd632e4883fc6c2f082194a7fa9750b455eb78735c12c8"
        ),
    }
    assert "docs/mixture-training-protocol.md" not in contracts.SOURCE_FINGERPRINT_PATHS
    assert "docs/mixture-training-seed2-protocol.md" not in contracts.SOURCE_FINGERPRINT_PATHS
    assert "data/processed/snli-training-v1/records.jsonl" in contracts.DATA_FILE_SHA256
    assert "src/reflex_decisions/snli_training_data.py" in contracts.SOURCE_FINGERPRINT_PATHS
    assert "src/reflex_decisions/snli_training_source.py" in contracts.SOURCE_FINGERPRINT_PATHS
    assert contracts.MAX_TOTAL_FORWARDS == 12073
    assert contracts.MAX_TOTAL_INPUT_TOKENS == 24725504


def test_verify_protocol_rejects_changed_natural_reasoning_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = tmp_path / "natural-reasoning-protocol.md"
    protocol.write_text("frozen candidate protocol", encoding="utf-8")
    expected = hashlib.sha256(b"frozen candidate protocol").hexdigest()
    monkeypatch.setattr(contracts, "EXPECTED_PROTOCOL_SHA256", expected)
    monkeypatch.setattr(contracts, "PROTOCOL_PATH", str(protocol))

    assert contracts.verify_protocol(protocol) == expected
    protocol.write_text("tampered protocol", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol SHA-256 mismatch"):
        contracts.verify_protocol(protocol)


def test_validate_pins_rejects_missing_or_changed_snli_candidate_pins() -> None:
    pins = contracts.make_pins()
    data_hashes = dict(pins["data_file_sha256"])
    data_hashes.pop("data/processed/snli-training-v1/records.jsonl")
    with pytest.raises(ValueError, match="exact allowlist"):
        contracts.validate_pins({**pins, "data_file_sha256": data_hashes})

    data_hashes = dict(pins["data_file_sha256"])
    data_hashes["data/processed/snli-training-v1/records.jsonl"] = "0" * 64
    with pytest.raises(ValueError, match="differ from frozen bytes"):
        contracts.validate_pins({**pins, "data_file_sha256": data_hashes})

    source_hashes = dict(pins["source_file_sha256"])
    source_hashes["src/reflex_decisions/snli_training_data.py"] = "0" * 64
    with pytest.raises(ValueError, match="SNLI candidate source fingerprint differs"):
        contracts.validate_pins({**pins, "source_file_sha256": source_hashes})
