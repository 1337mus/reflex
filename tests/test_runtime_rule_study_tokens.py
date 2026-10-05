from __future__ import annotations

import builtins
import hashlib
import importlib.metadata
import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from experiments import (
    runtime_rule_study_data,
    runtime_rule_study_inputs,
    runtime_rule_study_tokens,
)
from experiments.runtime_rule_study_inputs import StudyInputs
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]


def _study_inputs() -> StudyInputs:
    from test_runtime_rule_study_data import _development_pools, _training_pools
    from test_runtime_rule_study_inputs import _retention_records, _transfer_records

    real, balanced, synthetic = _retention_records()
    transfer = _transfer_records()
    retention = runtime_rule_study_inputs.build_retention_presentations(
        real, balanced, synthetic, transfer
    )
    training_pools = _training_pools()
    development_pools = _development_pools()
    new_presentations = runtime_rule_study_data.build_new_evaluation_presentations(
        *development_pools
    )
    records_by_id = {
        record.record_id: record
        for record in (
            *real,
            *balanced,
            *synthetic,
            *transfer,
            *development_pools[0],
            *development_pools[1],
        )
    }
    panel_ids = dict.fromkeys(str(row["record_id"]) for row in (*retention, *new_presentations))
    evaluation_records = tuple(records_by_id[record_id] for record_id in panel_ids)
    assert len(evaluation_records) == 525
    return StudyInputs(
        training_pools=training_pools,
        development_pools=development_pools,
        evaluation_records=evaluation_records,
        retention_presentations=retention,
        new_presentations=new_presentations,
        selection={},
        file_sha256={},
    )


def _request(option_ids: tuple[str, str]) -> DecisionRequest:
    options = {
        "red": Option(id="red", label="Red", description="Warm"),
        "blue": Option(id="blue", label="Blue", description="Cool"),
    }
    return DecisionRequest(
        context="Choose a color.",
        question="Which one?",
        options=tuple(options[option_id] for option_id in option_ids),
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_compile_spec_binds_ordered_request_and_compiled_prompt() -> None:
    tokenizer = CharacterTokenizer()
    red_first = _request(("red", "blue"))
    blue_first = _request(("blue", "red"))

    red_spec = runtime_rule_study_tokens.compile_spec(red_first, tokenizer)
    blue_spec = runtime_rule_study_tokens.compile_spec(blue_first, tokenizer)

    assert set(red_spec) == {
        "request_json_sha256",
        "request_hash",
        "schema_hash",
        "prompt_sha256",
        "input_tokens",
        "input_ids_sha256",
        "candidate_token_ids",
        "symbol_to_option_id",
    }
    request_json = json.dumps(
        red_first.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    prompt = render_prompt(red_first)
    input_ids = [ord(character) for character in prompt]

    assert red_spec == {
        "request_json_sha256": _sha256(request_json),
        "request_hash": red_first.request_hash,
        "schema_hash": red_first.schema_hash,
        "prompt_sha256": _sha256(prompt.encode("utf-8")),
        "input_tokens": len(input_ids),
        "input_ids_sha256": _sha256(
            json.dumps(input_ids, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ),
        "candidate_token_ids": [ord("A"), ord("B")],
        "symbol_to_option_id": {"A": "red", "B": "blue"},
    }
    assert red_spec["request_hash"] == blue_spec["request_hash"]
    assert red_spec["request_json_sha256"] != blue_spec["request_json_sha256"]
    assert red_spec["prompt_sha256"] != blue_spec["prompt_sha256"]
    assert red_spec["input_ids_sha256"] != blue_spec["input_ids_sha256"]
    assert red_spec["symbol_to_option_id"] != blue_spec["symbol_to_option_id"]


def test_compile_spec_rejects_prompt_over_2048_tokens() -> None:
    request = DecisionRequest(
        context="x" * 2_100,
        question="Pick one.",
        options=(
            Option(id="first", label="First"),
            Option(id="second", label="Second"),
        ),
    )

    with pytest.raises(ValueError, match="rendered prompt .* limit is 2048"):
        runtime_rule_study_tokens.compile_spec(request, CharacterTokenizer())


def test_compile_spec_rejects_retokenized_candidate_boundary() -> None:
    request = _request(("red", "blue"))
    assert runtime_rule_study_tokens.compile_spec(request, CharacterTokenizer())

    class RetokenizingTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            token_ids = [ord(character) for character in text]
            if text.endswith("A"):
                return [*token_ids[:-1], 999, token_ids[-1]]
            return token_ids

    with pytest.raises(ValueError, match="candidate symbol 'A' retokenizes the prompt boundary"):
        runtime_rule_study_tokens.compile_spec(request, RetokenizingTokenizer())


def test_load_cpu_tokenizer_wraps_backend_token_ids(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / ".cache" / "qwen-tokenizer.json"
    cache_path.parent.mkdir()
    raw = b'{"mock":true}'
    cache_path.write_bytes(raw)
    monkeypatch.setattr(runtime_rule_study_tokens, "TOKENIZER_SHA256", _sha256(raw), raising=False)
    version_calls = []
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: version_calls.append(package) or "0.23.2",
    )
    backend_inputs = []

    class BackendTokenizer:
        @classmethod
        def from_str(cls, contents: str):
            backend_inputs.append(contents)
            return cls()

        def encode(self, text: str, *, add_special_tokens: bool = False):
            assert text == "probe"
            assert add_special_tokens is False
            return SimpleNamespace(ids=(13, 21))

    backend = ModuleType("tokenizers")
    backend.Tokenizer = BackendTokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", backend)

    tokenizer = runtime_rule_study_tokens.load_cpu_tokenizer(tmp_path)

    assert version_calls == ["tokenizers"]
    assert backend_inputs == [raw.decode("utf-8")]
    assert tokenizer.encode("probe", add_special_tokens=False) == [13, 21]


def test_load_cpu_tokenizer_rejects_bad_pin_before_backend_import(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / ".cache" / "qwen-tokenizer.json"
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"wrong tokenizer bytes")
    monkeypatch.delitem(sys.modules, "tokenizers", raising=False)
    backend_imports = []
    real_import = builtins.__import__

    def watch_import(name, *args, **kwargs):
        if name == "tokenizers":
            backend_imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", watch_import)

    with pytest.raises(ValueError, match="cached tokenizer JSON SHA-256"):
        runtime_rule_study_tokens.load_cpu_tokenizer(tmp_path)

    assert backend_imports == []


def test_load_cpu_tokenizer_rejects_wrong_package_version_before_backend_import(
    tmp_path, monkeypatch
) -> None:
    cache_path = tmp_path / ".cache" / "qwen-tokenizer.json"
    cache_path.parent.mkdir()
    raw = b'{"mock":true}'
    cache_path.write_bytes(raw)
    monkeypatch.setattr(runtime_rule_study_tokens, "TOKENIZER_SHA256", _sha256(raw))
    version_calls = []
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: version_calls.append(package) or "0.23.3",
    )
    monkeypatch.delitem(sys.modules, "tokenizers", raising=False)
    backend_imports = []
    real_import = builtins.__import__

    def watch_import(name, *args, **kwargs):
        if name == "tokenizers":
            backend_imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", watch_import)

    with pytest.raises(ValueError, match="tokenizers package must be version 0.23.2"):
        runtime_rule_study_tokens.load_cpu_tokenizer(tmp_path)

    assert version_calls == ["tokenizers"]
    assert backend_imports == []


def test_load_cpu_tokenizer_rejects_missing_cached_file(tmp_path) -> None:
    with pytest.raises(ValueError, match="pinned tokenizer JSON is missing"):
        runtime_rule_study_tokens.load_cpu_tokenizer(tmp_path)


def test_load_cpu_tokenizer_rejects_symlink_outside_root(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    cache_path = root / ".cache" / "qwen-tokenizer.json"
    cache_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    raw = b'{"mock":true}'
    outside.write_bytes(raw)
    cache_path.symlink_to(outside)
    monkeypatch.setattr(runtime_rule_study_tokens, "TOKENIZER_SHA256", _sha256(raw))
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda _package: pytest.fail("version lookup happened before root containment"),
    )

    with pytest.raises(ValueError, match="resolves outside the supplied root"):
        runtime_rule_study_tokens.load_cpu_tokenizer(root)


def test_compile_study_inputs_returns_fixed_specs_counts_and_digest() -> None:
    inputs = _study_inputs()
    tokenizer = CharacterTokenizer()

    result = runtime_rule_study_tokens.compile_study_inputs(inputs, tokenizer)

    expected_counts = {
        "training_continued_practice": 1_344,
        "training_runtime_mix": 1_344,
        "evaluation_both_arms": 7_892,
        "unchanged_adapter": 164,
        "reload_both_arms": 64,
        "total": 10_808,
    }
    assert set(result) == {
        "training",
        "evaluation",
        "unchanged",
        "reload",
        "schedule_sha256",
        "counts",
        "input_token_counts",
        "max_input_tokens",
        "compilation_sha256",
    }
    training = result["training"]
    assert set(training) == {"continued_practice", "runtime_mix"}
    assert {arm: len(specs) for arm, specs in training.items()} == {
        "continued_practice": 1_344,
        "runtime_mix": 1_344,
    }
    assert len(result["evaluation"]) == 3_946
    assert len(result["unchanged"]) == 164
    assert len(result["reload"]) == 32
    assert result["counts"] == expected_counts
    spec_fields = {
        "request_json_sha256",
        "request_hash",
        "schema_hash",
        "prompt_sha256",
        "input_tokens",
        "input_ids_sha256",
        "candidate_token_ids",
        "symbol_to_option_id",
    }
    assert all(
        set(spec) == spec_fields | {"record_id"} for specs in training.values() for spec in specs
    )
    assert all(
        set(spec) == spec_fields | {"record_id", "presentation_id"} for spec in result["evaluation"]
    )
    assert result["evaluation"][-164:] == result["unchanged"]
    assert [spec["presentation_id"] for spec in result["evaluation"]] == [
        row["presentation_id"]
        for row in (*inputs.retention_presentations, *inputs.new_presentations)
    ]

    schedules = runtime_rule_study_data.build_training_schedules(*inputs.training_pools)
    schedule_audit = runtime_rule_study_data.audit_training_schedules(
        schedules, *inputs.training_pools
    )
    assert result["schedule_sha256"] == schedule_audit["schedule_sha256"]
    assert {
        arm: [spec["record_id"] for spec in specs] for arm, specs in result["training"].items()
    } == {arm: [example.record_id for example in examples] for arm, examples in schedules.items()}
    reload_rows = runtime_rule_study_data.reload_presentations(
        inputs.new_presentations, *inputs.development_pools
    )
    assert [spec["presentation_id"] for spec in result["reload"]] == [
        row["presentation_id"] for row in reload_rows
    ]

    expected_tokens = {
        "training_continued_practice": sum(
            spec["input_tokens"] for spec in training["continued_practice"]
        ),
        "training_runtime_mix": sum(spec["input_tokens"] for spec in training["runtime_mix"]),
        "evaluation_both_arms": 2 * sum(spec["input_tokens"] for spec in result["evaluation"]),
        "unchanged_adapter": sum(spec["input_tokens"] for spec in result["unchanged"]),
        "reload_both_arms": 2 * sum(spec["input_tokens"] for spec in result["reload"]),
    }
    expected_tokens["total"] = sum(expected_tokens.values())
    assert result["input_token_counts"] == expected_tokens
    all_specs = (
        [spec for specs in training.values() for spec in specs]
        + result["evaluation"]
        + result["unchanged"]
        + result["reload"]
    )
    assert result["max_input_tokens"] == max(spec["input_tokens"] for spec in all_specs)
    assert result["max_input_tokens"] <= 2_048
    payload = {key: value for key, value in result.items() if key != "compilation_sha256"}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert result["compilation_sha256"] == _sha256(canonical)
    assert runtime_rule_study_tokens.compile_study_inputs(inputs, tokenizer) == result


def test_compile_study_inputs_rejects_malformed_retention_rows() -> None:
    inputs = _study_inputs()
    rows = inputs.retention_presentations
    duplicate = dict(rows[1])
    duplicate["presentation_id"] = rows[0]["presentation_id"]
    answer_field = dict(rows[0])
    answer_field["answer"] = "gold"
    boolean_index = dict(rows[0])
    boolean_index["order_index"] = True
    float_index = dict(rows[0])
    float_index["order_index"] = float(rows[0]["order_index"])
    cases = (
        (rows[:-1], "exactly 3782 presentations"),
        ((rows[0], duplicate, *rows[2:]), "presentation IDs must be unique"),
        ((answer_field, *rows[1:]), "fields do not match the request-only schema"),
        ((boolean_index, *rows[1:]), "order_index must be an integer"),
        ((float_index, *rows[1:]), "order_index must be an integer"),
    )

    for malformed_rows, reason in cases:
        malformed_inputs = replace(inputs, retention_presentations=malformed_rows)
        with pytest.raises(ValueError, match=reason):
            runtime_rule_study_tokens.compile_study_inputs(malformed_inputs, CharacterTokenizer())
