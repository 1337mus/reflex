"""CPU tokenization and prompt accounting for the fixed runtime-rule study."""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from experiments import runtime_rule_study_data as study
from experiments.runtime_rule_study_data import _digest
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import Tokenizer, compile_request
from reflex_decisions.schema import DecisionRequest

if TYPE_CHECKING:
    from experiments.runtime_rule_study_inputs import StudyInputs

TOKENIZER_SHA256 = "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"
TOKENIZER_VERSION = "0.23.2"
REQUEST_FIELDS = {"context", "question", "options"}
RETENTION_COUNTS = {
    "dbpedia14-pilot-v1-development": 1_568,
    "sms-pilot-v1-development": 120,
    "snli-balanced-v1-development": 1_152,
    "synthetic-atomic-fact-inference-v1-development": 450,
    "synthetic-numeric-selection-v1-development": 364,
    "boolq-dev-pilot-v1": 64,
    "copa-dev-pilot-v1": 64,
}
EXPECTED_COUNTS = {
    "training_continued_practice": 1_344,
    "training_runtime_mix": 1_344,
    "evaluation_both_arms": 7_892,
    "unchanged_adapter": 164,
    "reload_both_arms": 64,
    "total": 10_808,
}


class _BackendTokenizerAdapter:
    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(self._backend.encode(text, add_special_tokens=add_special_tokens).ids)


def load_cpu_tokenizer(root: str | Path) -> _BackendTokenizerAdapter:
    """Load only the pinned tokenizer JSON already cached beneath ``root``."""

    try:
        canonical_root = Path(root).resolve(strict=True)
        tokenizer_path = (canonical_root / ".cache/qwen-tokenizer.json").resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("pinned tokenizer JSON is missing or its path cannot be resolved") from exc
    if not canonical_root.is_dir():
        raise ValueError("tokenizer root must be a directory")
    try:
        tokenizer_path.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError("pinned tokenizer JSON resolves outside the supplied root") from exc
    if not tokenizer_path.is_file():
        raise ValueError("pinned tokenizer JSON must be a regular file")
    if hashlib.sha256(raw := tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("cached tokenizer JSON SHA-256 does not match the pinned file")
    try:
        version = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"tokenizers package must be version {TOKENIZER_VERSION}") from exc
    if version != TOKENIZER_VERSION:
        raise ValueError(f"tokenizers package must be version {TOKENIZER_VERSION}")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ValueError("the pinned tokenizers package could not be imported") from exc
    return _BackendTokenizerAdapter(Tokenizer.from_str(raw.decode("utf-8")))


def compile_spec(request: DecisionRequest, tokenizer: Tokenizer) -> dict[str, object]:
    """Compile one ordered request into JSON-safe prompt and token identities."""

    compiled = compile_request(request, tokenizer, max_tokens=2048)
    return {
        "request_json_sha256": _digest(request.model_dump(mode="json")),
        "request_hash": compiled.request_hash,
        "schema_hash": compiled.schema_hash,
        "prompt_sha256": compiled.prompt_hash,
        "input_tokens": len(compiled.input_ids),
        "input_ids_sha256": _digest(list(compiled.input_ids)),
        "candidate_token_ids": list(compiled.candidate_token_ids),
        "symbol_to_option_id": dict(compiled.symbol_to_option_id),
    }


def _evaluation_record_index(records: Sequence[DecisionRecord]) -> dict[str, DecisionRecord]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("evaluation records must be a sequence")
    by_id: dict[str, DecisionRecord] = {}
    for record in records:
        if not isinstance(record, DecisionRecord):
            raise ValueError("evaluation records must contain DecisionRecord values")
        if record.record_id in by_id:
            raise ValueError("evaluation records contain duplicate record IDs")
        by_id[record.record_id] = record
    return by_id


def _token_total(specs: Sequence[Mapping[str, object]]) -> int:
    return sum(cast(int, spec["input_tokens"]) for spec in specs)


def _presentation_request(
    row: object, records_by_id: Mapping[str, DecisionRecord], *, label: str
) -> tuple[Mapping[str, object], DecisionRequest]:
    if not isinstance(row, Mapping) or set(row) != study.PRESENTATION_FIELDS:
        raise ValueError(f"{label} fields do not match the request-only schema")
    if type(row["order_index"]) is not int:
        raise ValueError(f"{label} order_index must be an integer")
    fields = ("presentation_id", "record_id", "dataset_id", "source_group_id", "request_hash")
    if any(not isinstance(row[field], str) for field in fields):
        raise ValueError(f"{label} identity fields must be strings")
    order_ids = row["order_ids"]
    if not isinstance(order_ids, list) or any(not isinstance(value, str) for value in order_ids):
        raise ValueError(f"{label} order_ids must be a list of strings")
    request_value = row["request"]
    if not isinstance(request_value, Mapping) or set(request_value) != REQUEST_FIELDS:
        raise ValueError(f"{label} request must contain only context, question, and options")
    try:
        request = DecisionRequest.model_validate(request_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} request is invalid") from exc
    record = records_by_id.get(cast(str, row["record_id"]))
    if record is None or row["dataset_id"] != record.dataset_id:
        raise ValueError(f"{label} does not join to its exact evaluation record")
    option_by_id = {option.id: option for option in record.request.options}
    if len(order_ids) != len(option_by_id) or set(order_ids) != set(option_by_id):
        raise ValueError(f"{label} order_ids do not match its source options")
    expected = record.request.model_copy(
        update={"options": tuple(option_by_id[option_id] for option_id in order_ids)}
    )
    if (
        row["request_hash"] != request.request_hash
        or request != expected
        or order_ids != [option.id for option in request.options]
    ):
        raise ValueError(f"{label} request, hash, or option order differs from its source")
    return row, request


def _validate_retention(
    rows: Sequence[Mapping[str, object]], records_by_id: Mapping[str, DecisionRecord]
) -> tuple[tuple[Mapping[str, object], DecisionRequest], ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("retention presentations must be a sequence")
    if len(rows) != 3_782:
        raise ValueError("retention panel must contain exactly 3782 presentations")
    identifiers: set[str] = set()
    dataset_counts: Counter[str] = Counter()
    validated = []
    for row in rows:
        valid_row, request = _presentation_request(row, records_by_id, label="retention")
        presentation_id = cast(str, valid_row["presentation_id"])
        if presentation_id in identifiers:
            raise ValueError("retention presentation IDs must be unique")
        identifiers.add(presentation_id)
        dataset_counts[cast(str, valid_row["dataset_id"])] += 1
        validated.append((valid_row, request))
    if dataset_counts != Counter(RETENTION_COUNTS):
        raise ValueError("retention panel task counts do not match the fixed seven-task panel")
    return tuple(validated)


def compile_study_inputs(inputs: StudyInputs, tokenizer: Tokenizer) -> dict[str, object]:
    """Regenerate the fixed CPU schedules and compile every planned request."""

    from experiments.runtime_rule_study_inputs import StudyInputs

    if not isinstance(inputs, StudyInputs):
        raise TypeError("inputs must be a validated StudyInputs value")
    records_by_id = _evaluation_record_index(inputs.evaluation_records)
    retention = _validate_retention(inputs.retention_presentations, records_by_id)
    if len(inputs.training_pools) != 5 or len(inputs.development_pools) != 2:
        raise ValueError("StudyInputs training or development pool count is invalid")
    development_routing, development_tool = inputs.development_pools
    new_rows = study.build_new_evaluation_presentations(development_routing, development_tool)
    study.audit_new_evaluation_presentations(
        inputs.new_presentations, development_routing, development_tool
    )
    new = tuple(
        _presentation_request(row, records_by_id, label="new development presentation")
        for row in new_rows
    )
    all_rows = (*retention, *new)
    if {cast(str, row["record_id"]) for row, _request in all_rows} != set(records_by_id):
        raise ValueError("evaluation records do not match final panel membership")

    pools = inputs.training_pools
    schedules = study.build_training_schedules(pools[0], pools[1], pools[2], pools[3], pools[4])
    schedule_audit = study.audit_training_schedules(
        schedules, pools[0], pools[1], pools[2], pools[3], pools[4]
    )
    training = {
        arm: [
            {**compile_spec(example.request, tokenizer), "record_id": example.record_id}
            for example in schedules[arm]
        ]
        for arm in study.ARM_NAMES
    }
    evaluation = [
        {
            **compile_spec(request, tokenizer),
            "record_id": cast(str, row["record_id"]),
            "presentation_id": cast(str, row["presentation_id"]),
        }
        for row, request in all_rows
    ]
    unchanged = [dict(spec) for spec in evaluation[-len(new) :]]
    reload_rows = study.reload_presentations(new_rows, development_routing, development_tool)
    reload = [
        {
            **compile_spec(DecisionRequest.model_validate(row["request"]), tokenizer),
            "record_id": cast(str, row["record_id"]),
            "presentation_id": cast(str, row["presentation_id"]),
        }
        for row in reload_rows
    ]

    groups = (
        ("training_continued_practice", training[study.CONTINUED_PRACTICE], 1),
        ("training_runtime_mix", training[study.RUNTIME_MIX], 1),
        ("evaluation_both_arms", evaluation, 2),
        ("unchanged_adapter", unchanged, 1),
        ("reload_both_arms", reload, 2),
    )
    counts = {name: multiplier * len(specs) for name, specs, multiplier in groups}
    counts["total"] = sum(counts.values())
    if counts != EXPECTED_COUNTS:
        raise ValueError("compiled forward counts differ from the fixed study plan")
    token_counts = {name: multiplier * _token_total(specs) for name, specs, multiplier in groups}
    token_counts["total"] = sum(token_counts.values())
    all_specs = [
        spec for group in (*training.values(), evaluation, unchanged, reload) for spec in group
    ]
    maximum = max(cast(int, spec["input_tokens"]) for spec in all_specs)
    result: dict[str, object] = dict(
        training=training,
        evaluation=evaluation,
        unchanged=unchanged,
        reload=reload,
        schedule_sha256=schedule_audit["schedule_sha256"],
        counts=counts,
        input_token_counts=token_counts,
        max_input_tokens=maximum,
    )
    result["compilation_sha256"] = _digest(result)
    return result
