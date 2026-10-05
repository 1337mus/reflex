"""Authenticate CPU inputs and serialize a label-safe runtime-rule study bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_data as study
from experiments.runtime_rule_study_cache import (
    CachedRetention,
    load_cached_retention,
    serialize_cached_retention,
)
from experiments.runtime_rule_study_inputs import StudyInputs, load_study_inputs
from experiments.runtime_rule_study_tokens import compile_study_inputs, load_cpu_tokenizer
from reflex_decisions.data import DecisionRecord

_BUNDLE_FIELDS = frozenset(
    {
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
)
_RETENTION_CACHE_FIELDS = (
    "receipt_file_sha256",
    "outputs_sha256",
    "cache_sha256",
    "source_file_sha256",
)
_REQUEST_FIELDS = {"context", "question", "options"}
_SHA_FIELDS = {"bundle_sha256"}


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    decoded = contracts.strict_json_loads(contracts.canonical_json(value))
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _record_json(record: DecisionRecord) -> dict[str, object]:
    if not isinstance(record, DecisionRecord):
        raise TypeError("study input pools must contain DecisionRecord values")
    return _json_object(record.model_dump(mode="json"), "DecisionRecord")


def _inputs_json(inputs: StudyInputs) -> dict[str, object]:
    if not isinstance(inputs, StudyInputs):
        raise TypeError("inputs must be a validated StudyInputs value")
    return {
        "training_pools": [
            [_record_json(record) for record in pool] for pool in inputs.training_pools
        ],
        "development_pools": [
            [_record_json(record) for record in pool] for pool in inputs.development_pools
        ],
        "evaluation_records": [_record_json(record) for record in inputs.evaluation_records],
        "retention_presentations": [dict(row) for row in inputs.retention_presentations],
        "new_presentations": [dict(row) for row in inputs.new_presentations],
        "selection": inputs.selection,
        "file_sha256": inputs.file_sha256,
    }


def _request_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    request_rows: list[dict[str, object]] = []
    for row in rows:
        if set(row) != study.PRESENTATION_FIELDS:
            raise ValueError("evaluation presentation is not the exact request-only schema")
        if type(row.get("order_index")) is not int:
            raise ValueError("evaluation presentation order_index must be a strict integer")
        request = row.get("request")
        if not isinstance(request, Mapping) or set(request) != _REQUEST_FIELDS:
            raise ValueError("evaluation request is not label-free")
        request_rows.append({key: row[key] for key in study.PRESENTATION_FIELDS})
    return request_rows


def _fresh_cache_json(
    root: str | Path, inputs: StudyInputs, compilation: object
) -> dict[str, object]:
    fresh_cache = load_cached_retention(root, inputs, compilation)
    return _json_object(serialize_cached_retention(fresh_cache), "serialized retention cache")


def build_study_bundle(
    root: str | Path, inputs: StudyInputs, compilation: object, cache: CachedRetention
) -> dict[str, object]:
    """Re-authenticate all inputs, then return a detached label-safe CPU bundle."""

    if not isinstance(inputs, StudyInputs):
        raise TypeError("inputs must be a validated StudyInputs value")
    fresh_inputs = load_study_inputs(root)
    if not isinstance(fresh_inputs, StudyInputs):
        raise TypeError("study input loader did not return StudyInputs")
    if contracts.canonical_json(_inputs_json(inputs)) != contracts.canonical_json(
        _inputs_json(fresh_inputs)
    ):
        raise ValueError("supplied StudyInputs differ from freshly authenticated inputs")

    tokenizer = load_cpu_tokenizer(root)
    fresh_compilation = compile_study_inputs(fresh_inputs, tokenizer)
    if contracts.canonical_json(compilation) != contracts.canonical_json(fresh_compilation):
        raise ValueError("supplied compilation differs from a fresh CPU compilation")

    fresh_cache_json = _fresh_cache_json(root, fresh_inputs, fresh_compilation)
    supplied_cache_json = serialize_cached_retention(cache)
    if contracts.canonical_json(supplied_cache_json) != contracts.canonical_json(fresh_cache_json):
        raise ValueError("supplied retention cache differs from freshly authenticated outputs")

    evaluation = _request_rows(
        (*fresh_inputs.retention_presentations, *fresh_inputs.new_presentations)
    )
    unchanged = _request_rows(fresh_inputs.new_presentations)
    reload_rows = study.reload_presentations(
        fresh_inputs.new_presentations, *fresh_inputs.development_pools
    )
    reload = _request_rows(reload_rows)
    retention_cache = {field: fresh_cache_json[field] for field in _RETENTION_CACHE_FIELDS}
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": contracts.EXPERIMENT_ID,
        "protocol_sha256": contracts.PROTOCOL_SHA256,
        "model_id": contracts.MODEL_ID,
        "model_revision": contracts.MODEL_REVISION,
        "selection": fresh_inputs.selection,
        "selected_tensor_sha256": fresh_cache_json["tensor_sha256"],
        "tokenizer_file_sha256": fresh_cache_json["tokenizer_file_sha256"],
        "runtime_versions": fresh_cache_json["runtime_versions"],
        "input_file_sha256": fresh_inputs.file_sha256,
        "retention_cache": retention_cache,
        "training_pools": [
            [_record_json(record) for record in pool] for pool in fresh_inputs.training_pools
        ],
        "evaluation": evaluation,
        "unchanged": unchanged,
        "reload": reload,
        "compilation": fresh_compilation,
    }
    digest = contracts.json_sha256(unsigned)
    return verify_bundle_digest({**unsigned, "bundle_sha256": digest}, digest)


def verify_bundle_digest(value: object, expected_sha256: object) -> dict[str, Any]:
    """Verify a trusted bundle digest and fixed envelope, not its underlying data."""

    bundle = _json_object(value, "bundle")
    if set(bundle) != _BUNDLE_FIELDS:
        raise ValueError("bundle has an unexpected top-level schema")
    if type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 1:
        raise ValueError("bundle schema_version must be strict integer 1")
    if (
        bundle.get("experiment_id") != contracts.EXPERIMENT_ID
        or bundle.get("protocol_sha256") != contracts.PROTOCOL_SHA256
        or bundle.get("model_id") != contracts.MODEL_ID
        or bundle.get("model_revision") != contracts.MODEL_REVISION
    ):
        raise ValueError("bundle model or protocol identity differs from the frozen study")
    if bundle.get("runtime_versions") != contracts.RUNTIME_VERSION_PINS:
        raise ValueError("bundle runtime_versions differ from the frozen study pins")
    expected = contracts.validate_sha256(expected_sha256, "trusted expected bundle SHA-256")
    recorded = contracts.validate_sha256(bundle.get("bundle_sha256"), "bundle SHA-256")
    unsigned = {key: item for key, item in bundle.items() if key not in _SHA_FIELDS}
    actual = contracts.json_sha256(unsigned)
    if recorded != actual:
        raise ValueError("bundle SHA-256 does not match its canonical fields")
    if actual != expected:
        raise ValueError("bundle SHA-256 differs from the trusted expected digest")
    return bundle
