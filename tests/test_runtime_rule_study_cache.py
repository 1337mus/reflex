"""Tests for strict saved-output reuse in the runtime-rule study."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from experiments import (
    adapter_transfer_contracts,
    adapter_transfer_results,
    mixture_training_contracts,
    mixture_training_data,
    mixture_training_results,
    runtime_rule_study_tokens,
)
from experiments import (
    runtime_rule_study_cache as cache,
)
from experiments.mixture_training_contracts import json_sha256
from experiments.runtime_rule_study_data import _digest
from experiments.runtime_rule_study_inputs import StudyInputs
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option

ROOT = Path(__file__).resolve().parents[1]


def _presentation(
    index: int,
    reverse: bool = False,
    *,
    dataset_id: str = "fixture-development",
    prefix: str = "presentation",
) -> dict[str, object]:
    options: tuple[Option, ...] = (
        Option(id="option-z", label="Zed"),
        Option(id="option-a", label="Able"),
    )
    if reverse:
        options = tuple(reversed(options))
    request = DecisionRequest(
        context="Fixture context.", question="Which option is correct?", options=options
    )
    return {
        "presentation_id": f"{prefix}-{index}",
        "record_id": f"{prefix}-record-{index}",
        "dataset_id": dataset_id,
        "source_group_id": f"group-{index}",
        "request_hash": request.request_hash,
        "order_index": int(reverse),
        "order_ids": [option.id for option in options],
        "request": request.model_dump(mode="json"),
    }


def _saved_output(presentation: dict[str, object], index: int) -> dict[str, object]:
    request = DecisionRequest.model_validate(presentation["request"])
    return {
        **{
            key: presentation[key]
            for key in (
                "presentation_id",
                "record_id",
                "dataset_id",
                "source_group_id",
                "request_hash",
                "order_index",
                "order_ids",
            )
        },
        "candidate_logits": [1.0, 1.0],
        "winner_option_id": "option-a",
        "input_tokens": 20 + index % 20,
        "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
    }


def _compiled_row(presentation: dict[str, object], index: int) -> dict[str, object]:
    request = DecisionRequest.model_validate(presentation["request"])
    order_ids = presentation["order_ids"]
    assert isinstance(order_ids, list)
    return {
        "request_json_sha256": _digest(request.model_dump(mode="json")),
        "request_hash": request.request_hash,
        "schema_hash": "a" * 64,
        "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
        "input_tokens": 20 + index % 20,
        "input_ids_sha256": "b" * 64,
        "candidate_token_ids": [101, 102],
        "symbol_to_option_id": {"A": order_ids[0], "B": order_ids[1]},
        "record_id": presentation["record_id"],
        "presentation_id": presentation["presentation_id"],
    }


def _study_panel() -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    counts = (
        ("dbpedia14-pilot-v1-development", 1_568),
        ("sms-pilot-v1-development", 120),
        ("snli-balanced-v1-development", 1_152),
        ("synthetic-atomic-fact-inference-v1-development", 450),
        ("synthetic-numeric-selection-v1-development", 364),
        ("boolq-dev-pilot-v1", 64),
        ("copa-dev-pilot-v1", 64),
    )
    retention: list[dict[str, object]] = []
    for dataset_id, count in counts:
        start = len(retention)
        retention.extend(
            _presentation(
                start + index,
                reverse=index % 2 == 1,
                dataset_id=dataset_id,
                prefix=dataset_id,
            )
            for index in range(count)
        )
    new = tuple(
        _presentation(
            index,
            reverse=index % 2 == 1,
            dataset_id=(
                "routing-data-v1-development" if family == "routing" else "tool-data-v1-development"
            ),
            prefix=f"new-{family}",
        )
        for family in ("routing", "tool")
        for index in range(82)
    )
    return tuple(retention), new


def _cache_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root.mkdir(parents=True)
    selection_path = root / adapter_transfer_contracts.SELECTION_PATH
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    original_selection, _ = adapter_transfer_contracts.load_selection(ROOT)
    selected_arm = str(original_selection["arm"])
    selected_training_run_id = str(original_selection["training_run_id"])
    selected_update = int(original_selection["update"])

    natural_receipt_path = root / cache.NATURAL_RECEIPT_PATH
    natural_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    natural_payload: dict[str, object] = {"pins": {}}
    natural_wrapper = {
        "payloads": {"snli_mix": natural_payload},
        "results": {"snli_mix": {"fixture": "natural"}},
    }
    natural_bytes = json.dumps(natural_wrapper, sort_keys=True, separators=(",", ":")).encode()
    natural_receipt_path.write_bytes(natural_bytes)
    natural_sha256 = hashlib.sha256(natural_bytes).hexdigest()

    selection = deepcopy(original_selection)
    selection["training_receipt_sha256"] = natural_sha256
    selection_bytes = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    selection_path.write_bytes(selection_bytes)
    monkeypatch.setattr(adapter_transfer_contracts, "EXPECTED_SELECTION_SHA256", selection_sha256)
    monkeypatch.setattr(
        adapter_transfer_contracts,
        "EXPECTED_CANONICAL_SELECTION_SHA256",
        json_sha256(selection),
    )
    normalized_selection, loaded_selection_sha256 = adapter_transfer_contracts.load_selection(root)
    assert loaded_selection_sha256 == selection_sha256

    transfer_receipt_path = root / cache.TRANSFER_RECEIPT_PATH
    transfer_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    transfer_payload: dict[str, object] = {"selection": normalized_selection, "pins": {}}
    transfer_wrapper = {
        "payload": transfer_payload,
        "result": {"fixture": "transfer"},
    }
    transfer_bytes = json.dumps(transfer_wrapper, sort_keys=True, separators=(",", ":")).encode()
    transfer_receipt_path.write_bytes(transfer_bytes)
    transfer_sha256 = hashlib.sha256(transfer_bytes).hexdigest()
    monkeypatch.setattr(cache, "NATURAL_RECEIPT_SHA256", natural_sha256)
    monkeypatch.setattr(cache, "TRANSFER_RECEIPT_SHA256", transfer_sha256)

    historical_sources = {
        relative: hashlib.sha256(relative.encode()).hexdigest()
        for relative in adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS
    }
    natural_sources = {
        relative: historical_sources[relative]
        for relative in mixture_training_contracts.SOURCE_FINGERPRINT_PATHS
    }
    natural_payload["pins"] = {"source_file_sha256": natural_sources}
    transfer_payload["pins"] = {"source_file_sha256": historical_sources}
    # The wrappers are written after their nested payload pins are complete.
    natural_bytes = json.dumps(natural_wrapper, sort_keys=True, separators=(",", ":")).encode()
    natural_receipt_path.write_bytes(natural_bytes)
    natural_sha256 = hashlib.sha256(natural_bytes).hexdigest()
    selection["training_receipt_sha256"] = natural_sha256
    selection_bytes = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    selection_path.write_bytes(selection_bytes)
    monkeypatch.setattr(adapter_transfer_contracts, "EXPECTED_SELECTION_SHA256", selection_sha256)
    monkeypatch.setattr(
        adapter_transfer_contracts,
        "EXPECTED_CANONICAL_SELECTION_SHA256",
        json_sha256(selection),
    )
    normalized_selection, loaded_selection_sha256 = adapter_transfer_contracts.load_selection(root)
    transfer_payload["selection"] = normalized_selection
    transfer_bytes = json.dumps(transfer_wrapper, sort_keys=True, separators=(",", ":")).encode()
    transfer_receipt_path.write_bytes(transfer_bytes)
    transfer_sha256 = hashlib.sha256(transfer_bytes).hexdigest()
    monkeypatch.setattr(cache, "NATURAL_RECEIPT_SHA256", natural_sha256)
    monkeypatch.setattr(cache, "TRANSFER_RECEIPT_SHA256", transfer_sha256)

    retention, new = _study_panel()
    old_count = 3_654
    old_rows = [_saved_output(row, index) for index, row in enumerate(retention[:old_count])]
    transfer_rows = [
        _saved_output(row, old_count + index) for index, row in enumerate(retention[old_count:])
    ]
    base_rows = deepcopy(transfer_rows)
    for row in base_rows:
        order_ids = row["order_ids"]
        assert isinstance(order_ids, list)
        row["candidate_logits"] = [1.0, 0.0]
        row["winner_option_id"] = order_ids[0]

    versions = dict(mixture_training_contracts.RUNTIME_VERSION_PINS)
    base_versions = {key: value for key, value in versions.items() if key != "Pillow"}
    tokenizer_files = {
        "tokenizer.json": runtime_rule_study_tokens.TOKENIZER_SHA256,
        "tokenizer_config.json": "c" * 64,
    }
    base_model: dict[str, object] = {
        "model_id": mixture_training_contracts.MODEL_ID,
        "model_revision": mixture_training_contracts.MODEL_REVISION,
        "model_class": "Qwen3_5ForCausalLM",
        "config_class": "Qwen3_5TextConfig",
        "layer_count": 24,
        "tied_embeddings": True,
        "no_meta_parameters": True,
        "load_diagnostics": {},
        "tokenizer_file_sha256": tokenizer_files,
        "effective_dtype": "torch.bfloat16",
        "device": "cuda:0",
        "attention_implementation": "eager",
        "use_kernels": False,
        "use_hub_kernels": "NO",
        "versions": base_versions,
    }

    def provenance(source_map: dict[str, str]) -> dict[str, object]:
        return {
            "model_id": mixture_training_contracts.MODEL_ID,
            "model_revision": mixture_training_contracts.MODEL_REVISION,
            "versions": versions,
            "source_file_sha256": source_map,
            "measured_source_file_sha256": source_map,
            "base_model": deepcopy(base_model),
            "reload_tokenizer_file_sha256": deepcopy(tokenizer_files),
        }

    natural_result: dict[str, object] = {
        "status": "passed",
        "phase": "train",
        "run_id": (f"{selected_training_run_id}{mixture_training_data.RUN_SUFFIXES[selected_arm]}"),
        "arm": selected_arm,
        "provenance": provenance(natural_sources),
        "evidence": {
            "optimizer_updates_completed": selected_update,
            "adapter_paths": {str(selected_update): deepcopy(normalized_selection["snapshot"])},
            "outputs": old_rows,
        },
    }
    transfer_result: dict[str, object] = {
        "status": "passed",
        "phase": "completed",
        "experiment_id": adapter_transfer_contracts.EXPERIMENT_ID,
        "run_id": "adapter-transfer-2026-10-05-r1",
        "provenance": provenance(historical_sources),
        "evidence": {
            "adapter_identity": {
                "snapshot_path": normalized_selection["snapshot"]["path"],
                "files_sha256": deepcopy(normalized_selection["snapshot"]["files_sha256"]),
                "tensor_sha256": cache.SELECTED_TENSOR_SHA256,
                "reloaded_tensor_sha256": cache.SELECTED_TENSOR_SHA256,
                "dtype": "torch.float32",
            },
            "outputs": {"adapter": transfer_rows, "base": base_rows},
        },
    }

    def validate_natural(result: object, payload: object) -> dict[str, object]:
        assert result == {"fixture": "natural"}
        assert payload == natural_payload
        return deepcopy(natural_result)

    def validate_transfer(
        result: object, payload: object, *, root: str | Path
    ) -> dict[str, object]:
        assert result == {"fixture": "transfer"}
        assert payload == transfer_payload
        assert Path(root) == root_path.resolve()
        return deepcopy(transfer_result)

    root_path = root
    monkeypatch.setattr(mixture_training_results, "validate_result", validate_natural)
    monkeypatch.setattr(adapter_transfer_results, "validate_result", validate_transfer)
    file_sha256 = dict(historical_sources)
    file_sha256[adapter_transfer_contracts.SELECTION_PATH] = loaded_selection_sha256
    inputs = StudyInputs(
        training_pools=(),
        development_pools=((), ()),
        evaluation_records=(),
        retention_presentations=retention,
        new_presentations=new,
        selection=normalized_selection,
        file_sha256=file_sha256,
    )
    evaluation = [_compiled_row(row, index) for index, row in enumerate((*retention, *new))]
    forward_counts = dict(runtime_rule_study_tokens.EXPECTED_COUNTS)
    evaluation_tokens = 2 * sum(20 + index % 20 for index in range(len(evaluation)))
    unchanged_tokens = sum(20 + index % 20 for index in range(len(retention), len(evaluation)))
    input_token_counts = {
        "training_continued_practice": 1_344 * 32,
        "training_runtime_mix": 1_344 * 32,
        "evaluation_both_arms": evaluation_tokens,
        "unchanged_adapter": unchanged_tokens,
        "reload_both_arms": 64 * 25,
    }
    input_token_counts["total"] = sum(input_token_counts.values())
    unsigned_compilation: dict[str, object] = {
        "training": {"continued_practice": [], "runtime_mix": []},
        "evaluation": evaluation,
        "unchanged": [],
        "reload": [],
        "schedule_sha256": {"continued_practice": "d" * 64, "runtime_mix": "e" * 64},
        "counts": forward_counts,
        "input_token_counts": input_token_counts,
        "max_input_tokens": 39,
    }
    compilation = {
        **unsigned_compilation,
        "compilation_sha256": _digest(unsigned_compilation),
    }
    return {
        "root": root,
        "inputs": inputs,
        "compilation": compilation,
        "natural_result": natural_result,
        "transfer_result": transfer_result,
        "natural_payload": natural_payload,
        "transfer_payload": transfer_payload,
        "retention": retention,
        "new": new,
        "old_rows": old_rows,
        "transfer_rows": transfer_rows,
        "base_rows": base_rows,
        "source_map": historical_sources,
    }


def _rehash_compilation(compilation: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in compilation.items() if key != "compilation_sha256"}
    compilation["compilation_sha256"] = _digest(unsigned)


def test_join_preserves_exact_identity_prompt_order_tokens_and_tie_winner() -> None:
    presentations = (_presentation(0), _presentation(1, reverse=True))
    assert presentations[0]["request_hash"] == presentations[1]["request_hash"]
    outputs = [_saved_output(row, index) for index, row in enumerate(presentations)]
    compiled = [_compiled_row(row, index) for index, row in enumerate(presentations)]

    joined = cache._join_retention_rows(presentations, outputs, compiled)

    assert tuple(row["presentation_id"] for row in joined) == (
        "presentation-0",
        "presentation-1",
    )
    assert tuple(row["winner_option_id"] for row in joined) == ("option-a", "option-a")
    assert tuple(row["order_ids"] for row in joined) == (
        ["option-z", "option-a"],
        ["option-a", "option-z"],
    )


def test_join_rejects_boolean_presentation_order_index() -> None:
    presentation = _presentation(1, reverse=True)
    output = _saved_output(presentation, 0)
    output["order_index"] = True
    compiled = _compiled_row(presentation, 0)

    with pytest.raises(ValueError, match="output order_index must be an integer"):
        cache._join_retention_rows((presentation,), [output], [compiled])


def test_join_rejects_reordered_options_even_when_request_hash_is_unchanged() -> None:
    presentation = _presentation(0)
    reordered = _presentation(0, reverse=True)
    assert presentation["request_hash"] == reordered["request_hash"]
    output = _saved_output(presentation, 0)
    compiled = _compiled_row(reordered, 0)

    with pytest.raises(ValueError, match="compiled ordered request differs"):
        cache._join_retention_rows((presentation,), [output], [compiled])


def test_join_rejects_prompt_and_token_count_mismatches() -> None:
    presentation = _presentation(0)
    output = _saved_output(presentation, 0)
    compiled = _compiled_row(presentation, 0)
    compiled["prompt_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="compiled prompt hash differs"):
        cache._join_retention_rows((presentation,), [output], [compiled])

    compiled = _compiled_row(presentation, 0)
    compiled["input_tokens"] = 21
    with pytest.raises(ValueError, match="token count differs"):
        cache._join_retention_rows((presentation,), [output], [compiled])


def test_join_rejects_matching_but_incorrect_compiled_and_cached_prompt_hashes() -> None:
    presentation = _presentation(0)
    output = _saved_output(presentation, 0)
    compiled = _compiled_row(presentation, 0)
    output["prompt_sha256"] = "c" * 64
    compiled["prompt_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="canonical rendered request"):
        cache._join_retention_rows((presentation,), [output], [compiled])


@pytest.mark.parametrize("bad_index", [True, 1.0])
def test_join_rejects_non_integer_output_and_compiled_indices(bad_index: object) -> None:
    presentation = _presentation(1, reverse=True)
    output = _saved_output(presentation, 0)
    output["order_index"] = bad_index
    compiled = _compiled_row(presentation, 0)
    with pytest.raises(ValueError, match="output order_index must be an integer"):
        cache._join_retention_rows((presentation,), [output], [compiled])

    output = _saved_output(presentation, 0)
    compiled["order_index"] = bad_index
    with pytest.raises(ValueError, match="compiled retention order_index"):
        cache._join_retention_rows((presentation,), [output], [compiled])


@pytest.mark.parametrize("missing", [False, True])
def test_join_rejects_duplicate_or_missing_cached_rows(missing: bool) -> None:
    presentations = (_presentation(0), _presentation(1, reverse=True))
    outputs = [_saved_output(row, index) for index, row in enumerate(presentations)]
    compiled = [_compiled_row(row, index) for index, row in enumerate(presentations)]
    if missing:
        outputs.pop()
    else:
        outputs[1] = dict(outputs[0])

    with pytest.raises(ValueError, match="cached outputs|identities and order"):
        cache._join_retention_rows(presentations, outputs, compiled)


def test_join_rejects_a_winner_that_violates_the_lexicographic_tie_rule() -> None:
    presentation = _presentation(0)
    output = _saved_output(presentation, 0)
    output["winner_option_id"] = "option-z"

    with pytest.raises(ValueError, match="lexicographic tie rule"):
        cache._join_retention_rows((presentation,), [output], [_compiled_row(presentation, 0)])


def test_loads_complete_adapter_retention_cache_from_pinned_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)

    loaded = cache.load_cached_retention(fixture["root"], fixture["inputs"], fixture["compilation"])

    assert len(loaded.outputs) == 3_782
    expected = (*fixture["old_rows"], *fixture["transfer_rows"])
    assert loaded.outputs == tuple(expected)
    assert loaded.outputs != (*fixture["old_rows"], *fixture["base_rows"])
    assert all("request" not in row for row in loaded.outputs)
    assert loaded.tensor_sha256 == cache.SELECTED_TENSOR_SHA256
    assert loaded.tokenizer_file_sha256["tokenizer.json"] == (
        runtime_rule_study_tokens.TOKENIZER_SHA256
    )
    assert loaded.source_file_sha256 == fixture["source_map"]
    assert loaded.compilation_sha256 == fixture["compilation"]["compilation_sha256"]
    serialized = cache.serialize_cached_retention(loaded)
    assert isinstance(serialized["outputs"], list)
    assert serialized["cache_sha256"] == loaded.cache_sha256


def test_loader_rejects_wrong_selected_snapshot_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    identity = fixture["transfer_result"]["evidence"]["adapter_identity"]
    identity["snapshot_path"] = "/artifacts/runs/wrong/adapter-update-378"

    with pytest.raises(ValueError, match="transfer adapter identity"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], fixture["compilation"])


def test_loader_rejects_parent_training_run_id_in_arm_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    selection = fixture["inputs"].selection
    fixture["natural_result"]["run_id"] = selection["training_run_id"]

    with pytest.raises(ValueError, match="passed SNLI update-378 execution"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], fixture["compilation"])


@pytest.mark.parametrize("mismatch", ["tokenizer", "runtime", "source"])
def test_loader_rejects_wrong_historical_identity(
    mismatch: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    natural = fixture["natural_result"]
    transfer = fixture["transfer_result"]
    expected_message = {
        "tokenizer": "tokenizer.json differs",
        "runtime": "runtime package versions differ",
        "source": "differs from verified StudyInputs source pins",
    }[mismatch]
    if mismatch == "tokenizer":
        for result in (natural, transfer):
            tokenizer_files = result["provenance"]["base_model"]["tokenizer_file_sha256"]
            tokenizer_files["tokenizer.json"] = "f" * 64
        transfer["provenance"]["reload_tokenizer_file_sha256"]["tokenizer.json"] = "f" * 64
    elif mismatch == "runtime":
        transfer["provenance"]["versions"]["Pillow"] = "0.0.0"
    else:
        path = next(iter(fixture["source_map"]))
        fixture["inputs"].file_sha256[path] = "f" * 64

    with pytest.raises(ValueError, match=expected_message):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], fixture["compilation"])


def test_loader_checks_receipt_hash_before_strict_json_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    receipt = Path(fixture["root"]) / cache.NATURAL_RECEIPT_PATH
    receipt.write_bytes(b"not strict JSON")

    with pytest.raises(ValueError, match="pinned receipt SHA-256 mismatch"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], fixture["compilation"])


def test_loader_rejects_receipt_symlink_that_resolves_outside_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    receipt = Path(fixture["root"]) / cache.NATURAL_RECEIPT_PATH
    outside = tmp_path / "outside-receipt.json"
    outside.write_bytes(receipt.read_bytes())
    receipt.unlink()
    receipt.symlink_to(outside)

    with pytest.raises(ValueError, match="pinned receipt is unavailable inside"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], fixture["compilation"])


def test_loader_rejects_compilation_mutation_and_changed_evaluation_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    compilation = deepcopy(fixture["compilation"])
    compilation["evaluation"][0]["input_tokens"] += 1
    with pytest.raises(ValueError, match="compilation SHA-256 does not match"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], compilation)

    compilation = deepcopy(fixture["compilation"])
    evaluation = compilation["evaluation"]
    evaluation[0], evaluation[1] = evaluation[1], evaluation[0]
    _rehash_compilation(compilation)
    with pytest.raises(ValueError, match="compiled evaluation identities differ"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], compilation)

    compilation = deepcopy(fixture["compilation"])
    compilation["evaluation"].pop()
    _rehash_compilation(compilation)
    with pytest.raises(ValueError, match="exact frozen panel"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], compilation)


@pytest.mark.parametrize(
    ("field", "category", "bad_value", "message"),
    [
        ("counts", "training_continued_practice", 1_344.0, "strict integers"),
        ("input_token_counts", "reload_both_arms", 1_600.0, "strict integers"),
        ("input_token_counts", "total", 1, "do not equal the category sum"),
    ],
)
def test_loader_rejects_malformed_aggregate_count_maps(
    field: str,
    category: str,
    bad_value: object,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    compilation = deepcopy(fixture["compilation"])
    count_map = compilation[field]
    assert isinstance(count_map, dict)
    if field == "input_token_counts" and category == "total":
        count_map[category] += 1
    else:
        count_map[category] = bad_value
    _rehash_compilation(compilation)

    with pytest.raises(ValueError, match=message):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], compilation)


@pytest.mark.parametrize("bad_count", [True, 20.0])
def test_loader_rejects_boolean_or_float_compilation_token_counts(
    bad_count: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    compilation = deepcopy(fixture["compilation"])
    compilation["evaluation"][0]["input_tokens"] = bad_count
    _rehash_compilation(compilation)

    with pytest.raises(ValueError, match="token count must be a strict positive integer"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], compilation)


def test_loader_rejects_wrong_prompt_and_cached_token_count_after_valid_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    compilation = deepcopy(fixture["compilation"])
    compilation["evaluation"][0]["prompt_sha256"] = "f" * 64
    _rehash_compilation(compilation)
    with pytest.raises(ValueError, match="compiled prompt hash differs"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], compilation)

    compilation = deepcopy(fixture["compilation"])
    compilation["evaluation"][0]["input_tokens"] += 1
    _rehash_compilation(compilation)
    with pytest.raises(ValueError, match="token count differs"):
        cache.load_cached_retention(fixture["root"], fixture["inputs"], compilation)


def test_serializer_detects_mutated_rows_at_the_next_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cache_fixture(tmp_path / "cache-root", monkeypatch)
    loaded = cache.load_cached_retention(fixture["root"], fixture["inputs"], fixture["compilation"])
    loaded.outputs[0]["winner_option_id"] = "mutated-after-cache"

    with pytest.raises(ValueError, match="output digest no longer matches"):
        cache.serialize_cached_retention(loaded)
