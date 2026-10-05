"""Strict role-scoped request and training row validation for study payloads."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import permutations
from typing import cast

from experiments import (
    mixture_training_contracts,
)
from experiments import (
    runtime_rule_study_contracts as contracts,
)
from experiments import (
    runtime_rule_study_data as study,
)
from experiments import (
    runtime_rule_study_tokens as tokens,
)
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest

_REQUEST_FIELDS = {"context", "question", "options"}
_COMPILED_FIELDS = frozenset(
    "request_json_sha256 request_hash schema_hash prompt_sha256 input_tokens "
    "input_ids_sha256 candidate_token_ids symbol_to_option_id".split()
)
_RETENTION_SHAPES: dict[str, Counter[int]] = {
    "dbpedia14-pilot-v1-development": Counter({14: 56}),
    "sms-pilot-v1-development": Counter({2: 60}),
    "snli-balanced-v1-development": Counter({3: 192}),
    "synthetic-atomic-fact-inference-v1-development": Counter({3: 75}),
    "synthetic-numeric-selection-v1-development": Counter({2: 14, 4: 12, 8: 12, 16: 12}),
    "boolq-dev-pilot-v1": Counter({2: 32}),
    "copa-dev-pilot-v1": Counter({2: 32}),
}


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _records(value: object, *, label: str) -> tuple[DecisionRecord, ...]:
    rows = _sequence(value, label)
    records: list[DecisionRecord] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} contains a record that is not an object")
        try:
            record = DecisionRecord.model_validate(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains an invalid decision record") from exc
        if record.model_dump(mode="json") != row:
            raise ValueError(f"{label} record is not in canonical JSON form")
        records.append(record)
    if len({record.record_id for record in records}) != len(records):
        raise ValueError(f"{label} contains duplicate record IDs")
    return tuple(records)


def validate_training_pools(value: object) -> tuple[tuple[DecisionRecord, ...], ...]:
    pools_value = _sequence(value, "training_pools")
    if len(pools_value) != 5:
        raise ValueError("training_pools must contain the exact five fixed pools")
    pools = tuple(
        _records(rows, label=f"training pool {index}") for index, rows in enumerate(pools_value)
    )
    return pools


def _parse_request(value: object, *, label: str) -> DecisionRequest:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
        raise ValueError(f"{label} must contain only context, question, and options")
    try:
        request = DecisionRequest.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if request.model_dump(mode="json") != value:
        raise ValueError(f"{label} is not in canonical JSON form")
    return request


def _normalize_presentation(
    value: object, *, label: str
) -> tuple[dict[str, object], DecisionRequest]:
    if not isinstance(value, Mapping) or set(value) != study.PRESENTATION_FIELDS:
        raise ValueError(f"{label} fields do not match the exact request-only schema")
    row = dict(value)
    string_fields = (
        "presentation_id",
        "record_id",
        "dataset_id",
        "source_group_id",
        "request_hash",
    )
    if any(not isinstance(row[field], str) or not row[field] for field in string_fields):
        raise ValueError(f"{label} identity fields must be non-empty strings")
    if type(row["order_index"]) is not int or row["order_index"] < 0:
        raise ValueError(f"{label} order_index must be a non-negative strict integer")
    order_ids = row["order_ids"]
    if not isinstance(order_ids, list) or any(not isinstance(item, str) for item in order_ids):
        raise ValueError(f"{label} order_ids must be an array of strings")
    request = _parse_request(row["request"], label=f"{label} request")
    if row["request_hash"] != request.request_hash:
        raise ValueError(f"{label} request hash differs from the request")
    if order_ids != [option.id for option in request.options]:
        raise ValueError(f"{label} option order differs from its request")
    return row, request


def _validate_panel_rows(
    value: object, *, label: str, complete_orders: bool = True
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    values = _sequence(value, label)
    rows: list[dict[str, object]] = []
    groups: dict[str, list[dict[str, object]]] = {}
    seen_presentations: set[str] = set()
    closed_records: set[str] = set()
    previous_record: str | None = None
    for raw in values:
        row, request = _normalize_presentation(raw, label=label)
        presentation_id = cast(str, row["presentation_id"])
        record_id = cast(str, row["record_id"])
        if presentation_id in seen_presentations:
            raise ValueError(f"{label} contains duplicate presentation IDs")
        seen_presentations.add(presentation_id)
        if record_id != previous_record:
            if record_id in closed_records:
                raise ValueError(f"{label} record presentations are not contiguous")
            if previous_record is not None:
                closed_records.add(previous_record)
            previous_record = record_id
        group = groups.setdefault(record_id, [])
        if group:
            first, first_request = _normalize_presentation(group[0], label=label)
            if (
                row["dataset_id"] != first["dataset_id"]
                or row["source_group_id"] != first["source_group_id"]
                or row["request_hash"] != first["request_hash"]
            ):
                raise ValueError(f"{label} record identity changes between presentations")
            base_options = {option.id: option for option in first_request.options}
            current_options = {option.id: option for option in request.options}
            if current_options != base_options:
                raise ValueError(f"{label} record option meanings change between presentations")
        group.append(row)
        rows.append(row)
    for group in groups.values():
        indices = [cast(int, row["order_index"]) for row in group]
        if complete_orders and indices != list(range(len(group))):
            raise ValueError(f"{label} record order indices are not complete and ordered")
        if not complete_orders and indices != sorted(set(indices)):
            raise ValueError(f"{label} record order indices are not unique and ordered")
    return rows, groups


def _validate_retention(value: object) -> list[dict[str, object]]:
    rows, groups = _validate_panel_rows(value, label="retention evaluation")
    if len(rows) != 3_782:
        raise ValueError("retention evaluation must contain exactly 3782 presentations")
    dataset_counts = Counter(cast(str, row["dataset_id"]) for row in rows)
    if dataset_counts != Counter(tokens.RETENTION_COUNTS):
        raise ValueError("retention task counts differ from the fixed seven-task panel")
    dataset_order = [
        cast(str, row["dataset_id"])
        for index, row in enumerate(rows)
        if index == 0 or row["dataset_id"] != rows[index - 1]["dataset_id"]
    ]
    if dataset_order != list(tokens.RETENTION_COUNTS):
        raise ValueError("retention task order or grouping differs from the fixed panel order")
    observed_shapes: dict[str, Counter[int]] = {
        dataset: Counter() for dataset in tokens.RETENTION_COUNTS
    }
    for group in groups.values():
        dataset_id = cast(str, group[0]["dataset_id"])
        option_count = len(cast(list[object], group[0]["order_ids"]))
        observed_shapes[dataset_id][option_count] += 1
        base_request = _parse_request(group[0]["request"], label="retention base request")
        options = base_request.options
        if dataset_id.startswith("dbpedia14-"):
            reverse = tuple(reversed(options))
            expected_orders = tuple(
                dict.fromkeys(
                    (
                        *[options[index:] + options[:index] for index in range(option_count)],
                        *[reverse[index:] + reverse[:index] for index in range(option_count)],
                    )
                )
            )
        elif dataset_id == "synthetic-numeric-selection-v1-development":
            expected_orders = tuple(
                options[index:] + options[:index] for index in range(option_count)
            )
        else:
            expected_orders = tuple(permutations(options))
        expected_order_ids = tuple(
            tuple(option.id for option in order) for order in expected_orders
        )
        actual_orders = tuple(tuple(cast(list[str], row["order_ids"])) for row in group)
        if actual_orders != expected_order_ids:
            raise ValueError(f"{dataset_id} option order differs from the fixed presentation rule")
    if observed_shapes != _RETENTION_SHAPES:
        raise ValueError("retention record and menu counts differ from the fixed task plan")
    return rows


def _new_presentation_id(row: Mapping[str, object]) -> str:
    return mixture_training_contracts.json_sha256(
        {
            "namespace": "runtime-rules-v1",
            "family": "routing"
            if row["dataset_id"] == study.NEW_DEVELOPMENT_DATASET_IDS["routing"]
            else "tool",
            "record_id": row["record_id"],
            "order_index": row["order_index"],
            "request_hash": row["request_hash"],
            "order_ids": row["order_ids"],
        }
    )


def _validate_new_panel(value: object) -> list[dict[str, object]]:
    rows, groups = _validate_panel_rows(value, label="new development evaluation")
    if len(rows) != study.NEW_PANEL_PRESENTATIONS:
        raise ValueError("new development evaluation must contain exactly 164 presentations")
    expected_order = list(study.NEW_DEVELOPMENT_DATASET_IDS.values())
    dataset_order = [
        cast(str, row["dataset_id"])
        for index, row in enumerate(rows)
        if index == 0 or row["dataset_id"] != rows[index - 1]["dataset_id"]
    ]
    if dataset_order != expected_order:
        raise ValueError("new development task order differs from routing then tool")
    if Counter(cast(str, row["dataset_id"]) for row in rows) != Counter(
        {dataset: study.NEW_PRESENTATIONS_PER_FAMILY for dataset in expected_order}
    ):
        raise ValueError("new development family counts must be exactly 82 each")
    expected_shapes = Counter({4: 5, 6: 5, 8: 4})
    family_records: dict[str, list[str]] = {dataset: [] for dataset in expected_order}
    family_groups: dict[str, set[str]] = {dataset: set() for dataset in expected_order}
    for record_id, group in groups.items():
        dataset_id = cast(str, group[0]["dataset_id"])
        family_records[dataset_id].append(record_id)
        family_groups[dataset_id].add(cast(str, group[0]["source_group_id"]))
        menu_size = len(cast(list[object], group[0]["order_ids"]))
        if len(group) != menu_size or [row["order_index"] for row in group] != list(
            range(menu_size)
        ):
            raise ValueError("new development records must contain every cyclic menu rotation")
        base_request = _parse_request(group[0]["request"], label="new development base request")
        base_options = base_request.options
        for row in group:
            offset = cast(int, row["order_index"])
            request = _parse_request(row["request"], label="new development request")
            if request.options != base_options[offset:] + base_options[:offset]:
                raise ValueError("new development option order is not the fixed cyclic rotation")
            if row["presentation_id"] != _new_presentation_id(row):
                raise ValueError("new development presentation ID differs from its fixed identity")
    for dataset_id in expected_order:
        if family_records[dataset_id] != sorted(family_records[dataset_id]):
            raise ValueError("new development records are not in fixed record-ID order")
        if len(family_records[dataset_id]) != 14 or len(family_groups[dataset_id]) != 14:
            raise ValueError("each new development family must contain exactly 14 records/groups")
        shape = Counter(
            len(cast(list[object], groups[record_id][0]["order_ids"]))
            for record_id in family_records[dataset_id]
        )
        if shape != expected_shapes:
            raise ValueError("new development menu counts differ from the fixed 4/6/8 plan")
    return rows


def _validate_spec(
    value: object, request: DecisionRequest, *, row: Mapping[str, object], label: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    normalized = dict(value)
    expected_fields = set(_COMPILED_FIELDS) | {"record_id"}
    if "presentation_id" in row:
        expected_fields.add("presentation_id")
    if set(normalized) != expected_fields:
        raise ValueError(f"{label} has an unexpected compiled identity schema")
    identity_fields = {
        "request_json_sha256": mixture_training_contracts.json_sha256(
            request.model_dump(mode="json")
        ),
        "request_hash": request.request_hash,
        "schema_hash": request.schema_hash,
        "prompt_sha256": hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest(),
        "record_id": row["record_id"],
    }
    if "presentation_id" in expected_fields:
        identity_fields["presentation_id"] = row["presentation_id"]
    if any(normalized.get(key) != expected for key, expected in identity_fields.items()):
        raise ValueError(f"{label} request or presentation identity differs from its source row")
    input_tokens = normalized.get("input_tokens")
    if type(input_tokens) is not int or not 1 <= input_tokens <= contracts.MAX_INPUT_TOKENS:
        raise ValueError(f"{label} input_tokens must be a strict positive count within the limit")
    mixture_training_contracts.validate_sha256(
        normalized.get("input_ids_sha256"), f"{label} input IDs SHA-256"
    )
    candidate_ids = normalized.get("candidate_token_ids")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) != len(request.options)
        or any(type(token_id) is not int or token_id < 0 for token_id in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError(f"{label} candidate token IDs do not match its options")
    symbols = normalized.get("symbol_to_option_id")
    expected_symbols = {
        chr(ord("A") + index): option.id for index, option in enumerate(request.options)
    }
    if symbols != expected_symbols:
        raise ValueError(f"{label} candidate symbols do not match its ordered options")
    return normalized


def _validate_specs(
    value: object,
    rows: Sequence[Mapping[str, object]],
    requests: Sequence[DecisionRequest],
    *,
    label: str,
) -> list[dict[str, object]]:
    specs = _sequence(value, label)
    if len(specs) != len(rows) or len(requests) != len(rows):
        raise ValueError(f"{label} count differs from its request rows")
    return [
        _validate_spec(spec, request, row=row, label=label)
        for spec, request, row in zip(specs, requests, rows, strict=True)
    ]


def validate_role_rows(
    value: Mapping[str, object], role: str, schedule_hashes: Mapping[str, str]
) -> dict[str, object]:
    """Validate a payload's role-specific rows and return normalized JSON values."""

    if role == contracts.ROLE_UNCHANGED:
        if value.get("training_pools") != [] or value.get("training_specs") != []:
            raise ValueError("unchanged payload must not receive training data")
        if value.get("reload") != [] or value.get("reload_specs") != []:
            raise ValueError("unchanged payload must not receive reload parity data")
        if value.get("optimizer") is not None:
            raise ValueError("unchanged payload must not receive an optimizer")
        evaluation = _validate_new_panel(value.get("evaluation"))
        requests = [
            _parse_request(row["request"], label="evaluation request") for row in evaluation
        ]
        specs = _validate_specs(
            value.get("evaluation_specs"), evaluation, requests, label="unchanged evaluation specs"
        )
        if len(specs) != 164:
            raise ValueError("unchanged payload must contain exactly 164 evaluation specs")
        return {
            "training_pools": [],
            "training_specs": [],
            "evaluation": evaluation,
            "evaluation_specs": specs,
            "reload": [],
            "reload_specs": [],
        }

    pools = validate_training_pools(value.get("training_pools"))
    schedules = study.build_training_schedules(*pools)
    audit = study.audit_training_schedules(schedules, *pools)
    if audit["schedule_sha256"] != dict(schedule_hashes):
        raise ValueError("regenerated schedules differ from both fixed schedule hashes")
    examples = schedules[role]
    training_rows = _sequence(value.get("training_specs"), "training_specs")
    if len(training_rows) != len(examples):
        raise ValueError("training specs do not contain exactly one row per scheduled request")
    normalized_training: list[dict[str, object]] = []
    for spec, example in zip(training_rows, examples, strict=True):
        normalized_training.append(
            _validate_spec(
                spec,
                example.request,
                row={"record_id": example.record_id},
                label="training spec",
            )
        )

    evaluation_value = _sequence(value.get("evaluation"), "evaluation")
    retention = _validate_retention(evaluation_value[:3_782])
    new = _validate_new_panel(evaluation_value[3_782:])
    retention_presentation_ids = {cast(str, row["presentation_id"]) for row in retention}
    new_presentation_ids = {cast(str, row["presentation_id"]) for row in new}
    if retention_presentation_ids & new_presentation_ids:
        raise ValueError("evaluation presentation IDs must be unique across the full panel")
    retention_record_ids = {cast(str, row["record_id"]) for row in retention}
    new_record_ids = {cast(str, row["record_id"]) for row in new}
    if retention_record_ids & new_record_ids:
        raise ValueError("record IDs overlap between retention and new panels")
    evaluation = [*retention, *new]
    requests = [_parse_request(row["request"], label="evaluation request") for row in evaluation]
    evaluation_specs = _validate_specs(
        value.get("evaluation_specs"), evaluation, requests, label="training evaluation specs"
    )
    reload_value, _ = _validate_panel_rows(
        value.get("reload"), label="reload parity", complete_orders=False
    )
    expected_reload: list[dict[str, object]] = []
    for dataset_id in study.NEW_DEVELOPMENT_DATASET_IDS.values():
        family_rows = sorted(
            (row for row in new if row["dataset_id"] == dataset_id),
            key=lambda row: (row["record_id"], row["order_index"]),
        )
        expected_reload.extend(family_rows[: study.RELOAD_PER_FAMILY])
    if reload_value != expected_reload or len(reload_value) != 32:
        raise ValueError("reload parity rows differ from the fixed first-16-per-family selection")
    reload_requests = [
        _parse_request(row["request"], label="reload evaluation request") for row in expected_reload
    ]
    reload_specs = _validate_specs(
        value.get("reload_specs"), expected_reload, reload_requests, label="reload specs"
    )
    evaluation_specs_by_id = {cast(str, spec["presentation_id"]): spec for spec in evaluation_specs}
    expected_reload_specs = [
        evaluation_specs_by_id[cast(str, row["presentation_id"])] for row in expected_reload
    ]
    if reload_specs != expected_reload_specs:
        raise ValueError("reload specs do not match the selected evaluation specs")
    if mixture_training_contracts.canonical_json(value.get("optimizer")) != (
        mixture_training_contracts.canonical_json(contracts.optimizer_settings())
    ):
        raise ValueError("training payload optimizer differs from the frozen AdamW settings")
    return {
        "training_pools": [[record.model_dump(mode="json") for record in pool] for pool in pools],
        "training_specs": normalized_training,
        "evaluation": evaluation,
        "evaluation_specs": evaluation_specs,
        "reload": reload_value,
        "reload_specs": reload_specs,
    }


def selected_training_examples(value: Mapping[str, object], role: str) -> tuple[object, ...]:
    """Regenerate only the selected training arm from a validated payload."""

    if role == contracts.ROLE_UNCHANGED:
        return ()
    pools = validate_training_pools(value.get("training_pools"))
    schedules = study.build_training_schedules(*pools)
    return tuple(schedules[role])
