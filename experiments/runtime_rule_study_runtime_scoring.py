"""Preflight request compilation and identity-bound runtime scoring."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from experiments import mixture_training_contracts as json_contracts
from experiments import mixture_training_runtime_scoring as row_api
from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_payloads as payload_api
from experiments import runtime_rule_study_progress as progress_api
from experiments.training_rehearsal_core import TrainingExample
from reflex_decisions.rendering import CompiledRequest, Tokenizer, compile_request
from reflex_decisions.schema import DecisionRequest

Category = Literal["training", "final_evaluation", "reload_parity"]


@dataclass(frozen=True, slots=True)
class PreparedItem:
    """One request bound to its exact compiler output and planned ledger identity."""

    category: Category
    index: int
    request: DecisionRequest
    compiled: CompiledRequest
    expected_spec_json: str


@dataclass(frozen=True, slots=True)
class PreparedRequests:
    """Immutable ordered requests and regenerated examples for one worker role."""

    training_items: tuple[PreparedItem, ...]
    evaluation_items: tuple[PreparedItem, ...]
    reload_items: tuple[PreparedItem, ...]
    training_examples: tuple[TrainingExample, ...]


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], json_contracts.normalize_json_object(value, label))


def _rows(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _compiled_spec(request: DecisionRequest, compiled: CompiledRequest) -> dict[str, object]:
    return {
        "request_json_sha256": json_contracts.json_sha256(request.model_dump(mode="json")),
        "request_hash": compiled.request_hash,
        "schema_hash": compiled.schema_hash,
        "prompt_sha256": compiled.prompt_hash,
        "input_tokens": len(compiled.input_ids),
        "input_ids_sha256": json_contracts.json_sha256(list(compiled.input_ids)),
        "candidate_token_ids": list(compiled.candidate_token_ids),
        "symbol_to_option_id": dict(compiled.symbol_to_option_id),
    }


def _compile_item(
    category: Category,
    index: int,
    request: DecisionRequest,
    expected_spec: object,
    tokenizer: Tokenizer,
    *,
    identity: Mapping[str, object],
) -> PreparedItem:
    expected = _object(expected_spec, f"{category} spec {index}")
    compiled = compile_request(request, tokenizer, max_tokens=contracts.MAX_INPUT_TOKENS)
    actual = {**_compiled_spec(request, compiled), **dict(identity)}
    if json_contracts.canonical_json(actual) != json_contracts.canonical_json(expected):
        raise ValueError(f"{category} request {index} differs from its complete compiled spec")
    return PreparedItem(
        category=category,
        index=index,
        request=request,
        compiled=compiled,
        expected_spec_json=json_contracts.canonical_json(expected).decode("utf-8"),
    )


def _presentation_request(value: object, label: str) -> tuple[dict[str, object], DecisionRequest]:
    presentation = _object(value, label)
    try:
        request = DecisionRequest.model_validate(presentation.get("request"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} request is invalid") from exc
    if json_contracts.canonical_json(
        request.model_dump(mode="json")
    ) != json_contracts.canonical_json(presentation.get("request")):
        raise ValueError(f"{label} request is not in canonical model form")
    if presentation.get("request_hash") != request.request_hash or presentation.get(
        "order_ids"
    ) != [option.id for option in request.options]:
        raise ValueError(f"{label} request identity or option order differs")
    return presentation, request


def prepare_requests(payload: object, tokenizer: Tokenizer) -> PreparedRequests:
    """Validate the role payload, regenerate examples, and compile every planned request."""

    normalized = payload_api.validate_payload(payload)
    role = normalized["role"]
    if not isinstance(role, str):
        raise ValueError("validated payload role must be a string")
    generated_examples = payload_api.training_examples(normalized)
    training_specs = _rows(normalized["training_specs"], "training_specs")
    if len(generated_examples) != len(training_specs):
        raise ValueError("generated training examples and specs differ in count")

    training_examples: list[TrainingExample] = []
    training_items: list[PreparedItem] = []
    for index, (raw_example, raw_spec) in enumerate(
        zip(generated_examples, training_specs, strict=True)
    ):
        if not isinstance(raw_example, TrainingExample):
            raise ValueError(f"training example {index} has an invalid type")
        example = raw_example
        request = example.request
        record_id = example.record_id
        if not isinstance(request, DecisionRequest) or not isinstance(record_id, str):
            raise ValueError(f"training example {index} has invalid request identity")
        training_examples.append(example)
        training_items.append(
            _compile_item(
                "training",
                index,
                request,
                raw_spec,
                tokenizer,
                identity={"record_id": record_id},
            )
        )

    prepared_panels: dict[str, tuple[PreparedItem, ...]] = {}
    for category, presentation_field, spec_field in (
        ("final_evaluation", "evaluation", "evaluation_specs"),
        ("reload_parity", "reload", "reload_specs"),
    ):
        presentations = _rows(normalized[presentation_field], presentation_field)
        specs = _rows(normalized[spec_field], spec_field)
        if len(presentations) != len(specs):
            raise ValueError(f"{presentation_field} presentations and specs differ in count")
        items: list[PreparedItem] = []
        for index, (raw_presentation, raw_spec) in enumerate(
            zip(presentations, specs, strict=True)
        ):
            presentation, request = _presentation_request(
                raw_presentation, f"{presentation_field} presentation {index}"
            )
            panel_record_id = presentation.get("record_id")
            presentation_id = presentation.get("presentation_id")
            if not isinstance(panel_record_id, str) or not isinstance(presentation_id, str):
                raise ValueError(f"{presentation_field} presentation {index} has invalid identity")
            items.append(
                _compile_item(
                    cast(Category, category),
                    index,
                    request,
                    raw_spec,
                    tokenizer,
                    identity={"record_id": panel_record_id, "presentation_id": presentation_id},
                )
            )
        prepared_panels[category] = tuple(items)

    return PreparedRequests(
        training_items=tuple(training_items),
        evaluation_items=prepared_panels["final_evaluation"],
        reload_items=prepared_panels["reload_parity"],
        training_examples=tuple(training_examples),
    )


def _prepared_spec(item: PreparedItem) -> dict[str, object]:
    try:
        value = json.loads(item.expected_spec_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared item expected spec is invalid JSON") from exc
    spec = _object(value, "prepared item expected spec")
    if json_contracts.canonical_json(spec).decode("utf-8") != item.expected_spec_json:
        raise ValueError("prepared item expected spec is not canonical JSON")
    return spec


def _validate_item(item: PreparedItem, category: Category) -> dict[str, object]:
    if not isinstance(item, PreparedItem):
        raise TypeError("item must be a PreparedItem")
    if category not in {"training", "final_evaluation", "reload_parity"}:
        raise ValueError("category is not an approved runtime scoring category")
    if item.category != category:
        raise ValueError("prepared item category differs from the scoring category")
    if type(item.index) is not int or item.index < 0:
        raise ValueError("prepared item index must be a strict nonnegative integer")
    if not isinstance(item.request, DecisionRequest):
        raise ValueError("prepared item request must be a DecisionRequest")

    expected = _prepared_spec(item)
    identity_fields = ("record_id",) if category == "training" else ("record_id", "presentation_id")
    actual = _compiled_spec(item.request, item.compiled)
    if not set(identity_fields) <= set(expected):
        raise ValueError("prepared item spec is missing its payload identity")
    actual.update({field: expected[field] for field in identity_fields})
    if json_contracts.canonical_json(actual) != json_contracts.canonical_json(expected):
        raise ValueError("prepared item request or compiled IDs differ from its expected spec")
    if len(item.compiled.input_ids) < 1 or len(item.compiled.candidate_token_ids) != len(
        item.request.options
    ):
        raise ValueError("prepared item compiled token counts do not match its request")
    return expected


def _publish_progress(evidence: dict[str, object], ledger: progress_api.ForwardLedger) -> None:
    evidence.update(ledger.snapshot())


def score_forward(
    model: Any,
    item: PreparedItem,
    torch: Any,
    ledger: progress_api.ForwardLedger,
    evidence: dict[str, object],
    category: Category,
) -> Any:
    """Score one prepared request while preserving its model-call accounting."""

    expected_spec = _validate_item(item, category)
    snapshot = ledger.snapshot()
    completed = snapshot.get("completed_forward_counts")
    pending = snapshot.get("pending_forward")
    if not isinstance(completed, Mapping) or completed.get(category) != item.index:
        raise ValueError("prepared item index differs from the next ledger identity")
    if pending is not None:
        raise ValueError("ledger already has a pending forward")

    input_ids = torch.tensor([list(item.compiled.input_ids)], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    candidate_ids = torch.tensor(
        list(item.compiled.candidate_token_ids), dtype=torch.long, device="cuda"
    )

    ledger.begin(category, expected_spec)
    _publish_progress(evidence, ledger)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
    )
    ledger.complete()
    _publish_progress(evidence, ledger)

    try:
        candidate_logits = output.logits[0, -1].index_select(0, candidate_ids).float()
        count = candidate_logits.numel()
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("model output did not expose readable final-token logits") from exc
    if type(count) is not int or count != len(item.request.options):
        raise RuntimeError("candidate-only output count did not match the request")
    if not bool(torch.isfinite(candidate_logits).all().item()):
        raise RuntimeError("candidate logits contained a non-finite value")
    return candidate_logits


def score_presentations(
    model: Any,
    items: tuple[PreparedItem, ...],
    presentations: object,
    torch: Any,
    ledger: progress_api.ForwardLedger,
    evidence: dict[str, object],
    category: Category,
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Score an ordered evaluation or reload panel and retain each completed row."""

    if category not in {"final_evaluation", "reload_parity"}:
        raise ValueError("presentation scoring is limited to evaluation and reload categories")
    if not isinstance(presentations, Sequence) or isinstance(presentations, (str, bytes)):
        raise ValueError("presentations must be an ordered sequence")
    if len(items) != len(presentations):
        raise ValueError("prepared items and presentations differ in count")

    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, (item, raw_presentation) in enumerate(zip(items, presentations, strict=True)):
        if item.index != index:
            raise ValueError("prepared presentation items must be the exact ordered panel prefix")
        spec = _validate_item(item, category)
        presentation, request = _presentation_request(
            raw_presentation, f"{category} presentation {index}"
        )
        presentation_id = presentation.get("presentation_id")
        record_id = presentation.get("record_id")
        if (
            not isinstance(presentation_id, str)
            or not isinstance(record_id, str)
            or presentation_id in seen_ids
        ):
            raise ValueError(f"{category} presentation {index} has duplicate or invalid identity")
        seen_ids.add(presentation_id)
        expected_identity = {
            "presentation_id": presentation_id,
            "record_id": record_id,
        }
        order_index = presentation.get("order_index")
        if (
            request != item.request
            or any(spec.get(field) != value for field, value in expected_identity.items())
            or spec.get("request_hash") != request.request_hash
            or type(order_index) is not int
            or order_index < 0
        ):
            raise ValueError(f"{category} presentation {index} differs from its prepared request")
        normalized.append(presentation)

    snapshot = ledger.snapshot()
    counts = snapshot.get("completed_forward_counts")
    if not isinstance(counts, Mapping) or snapshot.get("pending_forward") is not None:
        raise ValueError("ledger is not ready for presentation scoring")
    completed_before = counts.get(category)
    if type(completed_before) is not int or completed_before != 0:
        raise ValueError("presentation scoring requires a fresh category prefix")

    reload_outputs: list[object] | None = None
    reload_evidence: dict[str, object] | None = None
    final_by_id: dict[str, dict[str, object]] = {}
    if category == "final_evaluation":
        target = evidence.get("outputs")
        if not isinstance(target, list) or target:
            raise ValueError("final evaluation output list must be an empty live list")
    else:
        raw_reload = evidence.get("reload")
        if not isinstance(raw_reload, Mapping):
            raise ValueError("reload evidence must be a live object")
        reload_evidence = cast(dict[str, object], raw_reload)
        raw_reload_outputs = reload_evidence.get("outputs")
        if not isinstance(raw_reload_outputs, list) or raw_reload_outputs:
            raise ValueError("reload output list must be an empty live list")
        reload_outputs = raw_reload_outputs
        if (
            reload_evidence.get("winner_match_count") is not None
            or reload_evidence.get("max_candidate_logit_difference") is not None
        ):
            raise ValueError("reload summaries must be empty before scoring starts")
        raw_final_outputs = evidence.get("outputs")
        if not isinstance(raw_final_outputs, list):
            raise ValueError("final evaluation outputs are required before reload scoring")
        for index, raw_final in enumerate(raw_final_outputs):
            final_row = _object(raw_final, f"final output {index}")
            presentation_id = final_row.get("presentation_id")
            if not isinstance(presentation_id, str) or presentation_id in final_by_id:
                raise ValueError("final evaluation outputs have missing or duplicate identities")
            final_by_id[presentation_id] = final_row
        for index, (presentation, item) in enumerate(zip(normalized, items, strict=True)):
            final = final_by_id.get(cast(str, presentation["presentation_id"]))
            if final is None:
                raise ValueError("final evaluation is missing a planned reload counterpart")
            final_logits = final.get("candidate_logits")
            final_order = cast(list[object], presentation["order_ids"])
            spec = _prepared_spec(item)
            if (
                not isinstance(final_logits, list)
                or len(final_logits) != len(final_order)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in final_logits
                )
                or final.get("winner_option_id") not in final_order
                or any(
                    final.get(field) != presentation.get(field)
                    for field in (
                        "presentation_id",
                        "record_id",
                        "dataset_id",
                        "source_group_id",
                        "request_hash",
                        "order_index",
                        "order_ids",
                    )
                )
                or final.get("input_tokens") != spec.get("input_tokens")
                or final.get("prompt_sha256") != spec.get("prompt_sha256")
            ):
                raise ValueError(f"final evaluation counterpart {index} differs from reload plan")

    for item, presentation in zip(items, normalized, strict=True):
        with torch.inference_mode():
            logits = score_forward(model, item, torch, ledger, evidence, category)
        row = row_api._scored_row(presentation, logits, item.compiled)
        if category == "final_evaluation":
            cast(list[dict[str, object]], evidence["outputs"]).append(row)
        else:
            assert reload_outputs is not None
            assert reload_evidence is not None
            reload_outputs.append(row)
            final = final_by_id[cast(str, presentation["presentation_id"])]
            matched = row["winner_option_id"] == final["winner_option_id"]
            maximum_difference = max(
                abs(float(actual) - float(expected))
                for actual, expected in zip(
                    cast(list[float], row["candidate_logits"]),
                    cast(list[float], final["candidate_logits"]),
                    strict=True,
                )
            )
            prior_matches = reload_evidence.get("winner_match_count")
            prior_matches = 0 if prior_matches is None else prior_matches
            if type(prior_matches) is not int:
                raise ValueError("reload winner summary must be a strict integer")
            reload_evidence["winner_match_count"] = prior_matches + int(matched)
            prior_difference = reload_evidence.get("max_candidate_logit_difference")
            prior_difference = 0.0 if prior_difference is None else prior_difference
            if isinstance(prior_difference, bool) or not isinstance(prior_difference, (int, float)):
                raise ValueError("reload logit summary must be a finite number")
            reload_evidence["max_candidate_logit_difference"] = max(
                float(prior_difference), maximum_difference
            )

        if on_progress is not None:
            on_progress(item.index + 1, len(items))
        if category == "reload_parity":
            assert reload_evidence is not None
            if not matched:
                raise ValueError(
                    f"reload winner differs at presentation {presentation['presentation_id']}"
                )
            if maximum_difference > contracts.MAX_CANDIDATE_SCORE_DIFFERENCE:
                raise ValueError(
                    "reload candidate logit difference exceeds the fixed tolerance at "
                    f"presentation {presentation['presentation_id']}"
                )
