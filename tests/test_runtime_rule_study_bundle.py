"""Tests for authenticated, label-safe runtime-rule study bundles."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from experiments import runtime_rule_study_bundle as bundle
from experiments import (
    runtime_rule_study_cache as cache,
)
from experiments import (
    runtime_rule_study_contracts,
    runtime_rule_study_tokens,
)
from experiments import (
    runtime_rule_study_data as study,
)
from experiments.mixture_training_contracts import json_sha256
from experiments.runtime_rule_study_inputs import StudyInputs
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest, Option


def _record(
    record_id: str,
    dataset_id: str,
    source_group_id: str,
    menu_size: int = 2,
) -> DecisionRecord:
    options = tuple(
        Option(id=f"{record_id}-option-{index}", label=f"Option {index}")
        for index in range(menu_size)
    )
    request = DecisionRequest(
        context=f"Context for {record_id}",
        question=f"Question for {record_id}?",
        options=options,
    )
    return DecisionRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        source_group_id=source_group_id,
        request=request,
        answer_id=options[0].id,
    )


def _new_development(family: str) -> tuple[DecisionRecord, ...]:
    dataset_id = f"{family}-data-v1-development"
    menu_sizes = (4,) * 5 + (6,) * 5 + (8,) * 4
    return tuple(
        _record(f"{family}-{index:02d}", dataset_id, f"{family}-group-{index:02d}", menu_size)
        for index, menu_size in enumerate(menu_sizes)
    )


def _inputs() -> StudyInputs:
    routing = _new_development("routing")
    tool = _new_development("tool")
    retention_record = _record(
        "retention-record", "dbpedia14-pilot-v1-development", "retention-group"
    )
    request = retention_record.request
    retention = {
        "presentation_id": "retention-presentation",
        "record_id": retention_record.record_id,
        "dataset_id": retention_record.dataset_id,
        "source_group_id": retention_record.source_group_id,
        "request_hash": request.request_hash,
        "order_index": 0,
        "order_ids": [option.id for option in request.options],
        "request": request.model_dump(mode="json"),
    }
    training = tuple(
        (
            _record(
                f"training-{index}",
                f"training-dataset-{index}",
                f"training-group-{index}",
            ),
        )
        for index in range(5)
    )
    return StudyInputs(
        training_pools=training,
        development_pools=(routing, tool),
        evaluation_records=(retention_record, *routing, *tool),
        retention_presentations=(retention,),
        new_presentations=study.build_new_evaluation_presentations(routing, tool),
        selection={
            "selection_id": "fixture-selection",
            "snapshot": {"path": "/fixture/adapter", "files_sha256": {"adapter": "b" * 64}},
        },
        file_sha256={"fixture-inputs.jsonl": "a" * 64},
    )


def _compiled_row(presentation: dict[str, object]) -> dict[str, object]:
    request = DecisionRequest.model_validate(presentation["request"])
    order_ids = presentation["order_ids"]
    assert isinstance(order_ids, list)
    return {
        "request_json_sha256": json_sha256(request.model_dump(mode="json")),
        "request_hash": request.request_hash,
        "schema_hash": request.schema_hash,
        "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
        "input_tokens": 20,
        "input_ids_sha256": "f" * 64,
        "candidate_token_ids": [101, 102],
        "symbol_to_option_id": {"A": order_ids[0], "B": order_ids[1]},
        "record_id": presentation["record_id"],
        "presentation_id": presentation["presentation_id"],
    }


def _compilation(inputs: StudyInputs) -> dict[str, object]:
    evaluation_rows = [
        _compiled_row(row) for row in (*inputs.retention_presentations, *inputs.new_presentations)
    ]
    unchanged = evaluation_rows[-len(inputs.new_presentations) :]
    reload_rows = study.reload_presentations(inputs.new_presentations, *inputs.development_pools)
    reload = [_compiled_row(row) for row in reload_rows]
    counts = dict(runtime_rule_study_tokens.EXPECTED_COUNTS)
    input_tokens = {
        "training_continued_practice": 1_344 * 20,
        "training_runtime_mix": 1_344 * 20,
        "evaluation_both_arms": 2 * 20 * len(evaluation_rows),
        "unchanged_adapter": 20 * len(unchanged),
        "reload_both_arms": 2 * 20 * len(reload),
    }
    input_tokens["total"] = sum(input_tokens.values())
    unsigned: dict[str, object] = {
        "training": {"continued_practice": [], "runtime_mix": []},
        "evaluation": evaluation_rows,
        "unchanged": unchanged,
        "reload": reload,
        "schedule_sha256": {"continued_practice": "d" * 64, "runtime_mix": "e" * 64},
        "counts": counts,
        "input_token_counts": input_tokens,
        "max_input_tokens": 20,
    }
    return {**unsigned, "compilation_sha256": json_sha256(unsigned)}


def _cache(inputs: StudyInputs, compilation: dict[str, object]) -> cache.CachedRetention:
    order_ids = inputs.retention_presentations[0]["order_ids"]
    assert isinstance(order_ids, list)
    assert order_ids and isinstance(order_ids[0], str)
    output = {
        "presentation_id": inputs.retention_presentations[0]["presentation_id"],
        "record_id": inputs.retention_presentations[0]["record_id"],
        "candidate_logits": [1.0, 0.0],
        "winner_option_id": order_ids[0],
        "prompt_sha256": "1" * 64,
        "input_tokens": 20,
    }
    partial = cache.CachedRetention(
        outputs=(output,),
        receipt_file_sha256={"receipt.json": "2" * 64},
        outputs_sha256=study._digest([output]),
        selection=deepcopy(inputs.selection),
        tensor_sha256="3" * 64,
        tokenizer_file_sha256={"tokenizer.json": "4" * 64},
        runtime_versions=dict(runtime_rule_study_contracts.RUNTIME_VERSION_PINS),
        base_model={"model_id": runtime_rule_study_contracts.MODEL_ID},
        source_file_sha256={"experiments/source.py": "5" * 64},
        compilation_sha256=str(compilation["compilation_sha256"]),
        cache_sha256="",
    )
    return replace(partial, cache_sha256=study._digest(cache._cache_payload(partial)))


@pytest.fixture
def authenticated_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    supplied_inputs = _inputs()
    fresh_inputs = deepcopy(supplied_inputs)
    fresh_compilation = _compilation(fresh_inputs)
    fresh_cache = _cache(fresh_inputs, fresh_compilation)
    tokenizer = object()
    events: dict[str, list[object]] = {
        "input_roots": [],
        "tokenizer_roots": [],
        "compiler_inputs": [],
        "compiler_tokenizers": [],
        "cache_inputs": [],
        "cache_compilations": [],
    }

    def load_inputs(root: str | Path) -> StudyInputs:
        events["input_roots"].append(root)
        return fresh_inputs

    def load_tokenizer(root: str | Path) -> object:
        events["tokenizer_roots"].append(root)
        return tokenizer

    def compile_inputs(inputs: StudyInputs, current_tokenizer: object) -> dict[str, object]:
        events["compiler_inputs"].append(inputs)
        events["compiler_tokenizers"].append(current_tokenizer)
        return deepcopy(fresh_compilation)

    def load_cache(
        root: str | Path, inputs: StudyInputs, compilation: object
    ) -> cache.CachedRetention:
        events["cache_inputs"].append(inputs)
        events["cache_compilations"].append(compilation)
        return fresh_cache

    monkeypatch.setattr(bundle, "load_study_inputs", load_inputs)
    monkeypatch.setattr(bundle, "load_cpu_tokenizer", load_tokenizer)
    monkeypatch.setattr(bundle, "compile_study_inputs", compile_inputs)
    monkeypatch.setattr(bundle, "load_cached_retention", load_cache)
    return {
        "root": tmp_path,
        "inputs": supplied_inputs,
        "fresh_inputs": fresh_inputs,
        "compilation": fresh_compilation,
        "cache": fresh_cache,
        "tokenizer": tokenizer,
        "events": events,
    }


def _rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    rows = [row for row in value if isinstance(row, dict)]
    assert len(rows) == len(value)
    return rows


def test_builds_label_safe_detached_bundle_after_fresh_reauthentication(
    authenticated_fixture: dict[str, Any],
) -> None:
    fixture = authenticated_fixture

    result = bundle.build_study_bundle(
        fixture["root"], fixture["inputs"], fixture["compilation"], fixture["cache"]
    )

    assert set(result) == {
        "schema_version",
        "experiment_id",
        "protocol_sha256",
        "model_id",
        "model_revision",
        "selection",
        "selected_tensor_sha256",
        "tokenizer_file_sha256",
        "runtime_versions",
        "input_file_sha256",
        "retention_cache",
        "training_pools",
        "evaluation",
        "unchanged",
        "reload",
        "compilation",
        "bundle_sha256",
    }
    assert result["schema_version"] == 1
    training_pools = result["training_pools"]
    assert isinstance(training_pools, list)
    assert training_pools
    first_training_pool = _rows(training_pools[0])
    assert first_training_pool[0]["answer_id"] == "training-0-option-0"
    panels = {name: _rows(result[name]) for name in ("evaluation", "unchanged", "reload")}
    assert len(panels["evaluation"]) == 165
    assert len(panels["unchanged"]) == 164
    assert len(panels["reload"]) == 32
    for rows in panels.values():
        assert all("answer_id" not in row for row in rows)
        assert all("candidate_logits" not in row and "winner_option_id" not in row for row in rows)
    compilation = result["compilation"]
    assert isinstance(compilation, dict)
    for panel in ("evaluation", "unchanged", "reload"):
        assert all("answer_id" not in row for row in _rows(compilation[panel]))
    retention_cache = result["retention_cache"]
    assert isinstance(retention_cache, dict)
    assert set(retention_cache) == {
        "receipt_file_sha256",
        "outputs_sha256",
        "cache_sha256",
        "source_file_sha256",
    }
    assert result["bundle_sha256"] == json_sha256(
        {key: value for key, value in result.items() if key != "bundle_sha256"}
    )

    verified = bundle.verify_bundle_digest(result, result["bundle_sha256"])
    verified["selection"]["snapshot"]["path"] = "changed"
    selection = result["selection"]
    assert isinstance(selection, dict)
    snapshot = selection["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["path"] == "/fixture/adapter"
    first_training_pool[0]["answer_id"] = "detached-answer"
    assert fixture["inputs"].training_pools[0][0].answer_id == "training-0-option-0"
    assert fixture["fresh_inputs"].training_pools[0][0].answer_id == "training-0-option-0"

    events = fixture["events"]
    assert events["input_roots"] == [fixture["root"]]
    assert events["tokenizer_roots"] == [fixture["root"]]
    assert events["compiler_inputs"] == [fixture["fresh_inputs"]]
    assert events["compiler_tokenizers"] == [fixture["tokenizer"]]
    assert events["cache_inputs"] == [fixture["fresh_inputs"]]
    assert events["cache_compilations"] == [fixture["compilation"]]


@pytest.mark.parametrize(
    "mutation", ["answer", "source_group", "option_order", "file_map", "selection"]
)
def test_rejects_every_mutated_supplied_input_field(
    mutation: str,
    authenticated_fixture: dict[str, Any],
) -> None:
    fixture = authenticated_fixture
    inputs = fixture["inputs"]
    if mutation == "answer":
        pools = list(inputs.training_pools)
        first_pool = list(pools[0])
        record = first_pool[0]
        first_pool[0] = record.model_copy(update={"answer_id": record.request.options[1].id})
        pools[0] = tuple(first_pool)
        supplied = replace(inputs, training_pools=tuple(pools))
    else:
        supplied = inputs
        if mutation == "source_group":
            supplied.new_presentations[0]["source_group_id"] = "changed-group"
        elif mutation == "option_order":
            row = supplied.new_presentations[0]
            row["order_ids"].reverse()
            request = row["request"]
            assert isinstance(request, dict)
            request["options"].reverse()
        elif mutation == "file_map":
            supplied.file_sha256["fixture-inputs.jsonl"] = "f" * 64
        else:
            supplied.selection["selection_id"] = "changed-selection"

    with pytest.raises(ValueError, match="differ from freshly authenticated inputs"):
        bundle.build_study_bundle(
            fixture["root"], supplied, fixture["compilation"], fixture["cache"]
        )


def test_rejects_compilation_mutation_even_after_recomputed_digest(
    authenticated_fixture: dict[str, Any],
) -> None:
    fixture = authenticated_fixture
    supplied = deepcopy(fixture["compilation"])
    supplied["evaluation"][0]["input_tokens"] += 1
    unsigned = {key: value for key, value in supplied.items() if key != "compilation_sha256"}
    supplied["compilation_sha256"] = json_sha256(unsigned)

    with pytest.raises(ValueError, match="differs from a fresh CPU compilation"):
        bundle.build_study_bundle(fixture["root"], fixture["inputs"], supplied, fixture["cache"])


def _rehash_cache(value: cache.CachedRetention) -> cache.CachedRetention:
    updated = replace(
        value,
        outputs_sha256=study._digest([dict(row) for row in value.outputs]),
        cache_sha256="",
    )
    return replace(updated, cache_sha256=study._digest(cache._cache_payload(updated)))


@pytest.mark.parametrize("rehash", [False, True])
def test_rejects_nested_cached_output_mutation_with_stale_or_recomputed_digests(
    rehash: bool,
    authenticated_fixture: dict[str, Any],
) -> None:
    fixture = authenticated_fixture
    supplied_cache = deepcopy(fixture["cache"])
    supplied_cache.outputs[0]["candidate_logits"][0] = 0.25
    if rehash:
        supplied_cache = _rehash_cache(supplied_cache)

    message = (
        "differs from freshly authenticated outputs"
        if rehash
        else "output digest no longer matches"
    )
    with pytest.raises(ValueError, match=message):
        bundle.build_study_bundle(
            fixture["root"], fixture["inputs"], fixture["compilation"], supplied_cache
        )


def _reseal_bundle(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "bundle_sha256"}
    value["bundle_sha256"] = json_sha256(unsigned)
    return value


def test_digest_verifier_rejects_boolean_version_and_unexpected_schema(
    authenticated_fixture: dict[str, Any],
) -> None:
    fixture = authenticated_fixture
    valid = bundle.build_study_bundle(
        fixture["root"], fixture["inputs"], fixture["compilation"], fixture["cache"]
    )
    boolean_version = deepcopy(valid)
    boolean_version["schema_version"] = True
    _reseal_bundle(boolean_version)
    with pytest.raises(ValueError, match="strict integer 1"):
        bundle.verify_bundle_digest(boolean_version, boolean_version["bundle_sha256"])

    extra_field = deepcopy(valid)
    extra_field["unreviewed"] = "value"
    _reseal_bundle(extra_field)
    with pytest.raises(ValueError, match="unexpected top-level schema"):
        bundle.verify_bundle_digest(extra_field, extra_field["bundle_sha256"])


def test_digest_verifier_requires_expected_digest_and_fixed_identity(
    authenticated_fixture: dict[str, Any],
) -> None:
    fixture = authenticated_fixture
    valid = bundle.build_study_bundle(
        fixture["root"], fixture["inputs"], fixture["compilation"], fixture["cache"]
    )
    with pytest.raises(ValueError, match="trusted expected digest"):
        bundle.verify_bundle_digest(valid, "0" * 64)

    changed_identity = deepcopy(valid)
    changed_identity["model_revision"] = "other-revision"
    _reseal_bundle(changed_identity)
    with pytest.raises(ValueError, match="model or protocol identity"):
        bundle.verify_bundle_digest(changed_identity, changed_identity["bundle_sha256"])
