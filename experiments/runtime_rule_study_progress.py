"""Pure CPU accounting for planned runtime-rule model forwards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from experiments import mixture_training_contracts as json_contracts
from experiments import runtime_rule_study_contracts as contracts

_CATEGORIES = ("training", "final_evaluation", "reload_parity")
_TOTAL = "total"
_COUNT_FIELDS = frozenset((*_CATEGORIES, _TOTAL))
_PROGRESS_FIELDS = frozenset(
    {"completed_forward_counts", "completed_input_token_counts", "pending_forward"}
)
_PENDING_FIELDS = frozenset({"category", "index", "spec_sha256", "input_tokens"})


@dataclass(frozen=True, slots=True)
class _RolePlan:
    role: str
    specs: dict[str, tuple[dict[str, object], ...]]
    forward_caps: dict[str, int]
    token_caps: dict[str, int]

    @property
    def total_forward_cap(self) -> int:
        return sum(self.forward_caps.values())

    @property
    def total_token_cap(self) -> int:
        return sum(self.token_caps.values())


def _copy_json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    try:
        normalized = json_contracts.normalize_json_object(value, label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a strict JSON object") from exc
    return cast(dict[str, object], normalized)


def _input_tokens(spec: Mapping[str, object], label: str) -> int:
    value = spec.get("input_tokens")
    if type(value) is not int or not 1 <= value <= contracts.MAX_INPUT_TOKENS:
        raise ValueError(f"{label} input_tokens must be a strict integer from 1 to 2048")
    return value


def _build_plan(role: object, specs_by_category: object) -> _RolePlan:
    forward_counts = contracts.expected_forward_counts(role)
    if not isinstance(specs_by_category, Mapping) or set(specs_by_category) != set(_CATEGORIES):
        raise ValueError("spec categories must exactly match the fixed execution order")
    if any(not isinstance(specs_by_category[name], list) for name in _CATEGORIES):
        raise ValueError("each progress category must contain an ordered list of specs")
    copied = _copy_json_object(specs_by_category, "specs_by_category")

    plan: dict[str, tuple[dict[str, object], ...]] = {}
    forward_caps: dict[str, int] = {}
    token_caps: dict[str, int] = {}
    for category in _CATEGORIES:
        raw_specs = copied.get(category)
        if not isinstance(raw_specs, list):
            raise ValueError("each progress category must contain an ordered list of specs")
        if len(raw_specs) != forward_counts[category]:
            raise ValueError(f"{category} spec count does not match the role forward budget")
        category_specs: list[dict[str, object]] = []
        for index, raw_spec in enumerate(raw_specs):
            if not isinstance(raw_spec, dict):
                raise ValueError(f"{category} spec {index} must be a JSON object")
            spec = cast(dict[str, object], raw_spec)
            _input_tokens(spec, f"{category} spec {index}")
            category_specs.append(spec)
        plan[category] = tuple(category_specs)
        forward_caps[category] = len(category_specs)
        token_caps[category] = sum(_input_tokens(spec, category) for spec in category_specs)
    return _RolePlan(
        role=cast(str, role),
        specs=plan,
        forward_caps=forward_caps,
        token_caps=token_caps,
    )


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in (*_CATEGORIES, _TOTAL)}


def _next_category(plan: _RolePlan, completed: Mapping[str, int]) -> str | None:
    for category in _CATEGORIES:
        if completed[category] < plan.forward_caps[category]:
            return category
    return None


def _strict_count_map(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _COUNT_FIELDS:
        raise ValueError(f"{label} fields do not match the exact progress schema")
    result: dict[str, int] = {}
    for field in (*_CATEGORIES, _TOTAL):
        count = value[field]
        if type(count) is not int or count < 0:
            raise ValueError(f"{label} values must be strict nonnegative integers")
        result[field] = count
    return result


def _validate_completed_prefix(
    plan: _RolePlan,
    forward_counts: Mapping[str, int],
    token_counts: Mapping[str, int],
) -> None:
    earlier_incomplete = False
    for category in _CATEGORIES:
        count = forward_counts[category]
        if count > plan.forward_caps[category]:
            raise ValueError(f"{category} completed count exceeds its planned cap")
        expected_tokens = sum(
            _input_tokens(spec, category) for spec in plan.specs[category][:count]
        )
        if token_counts[category] != expected_tokens:
            raise ValueError(f"{category} input tokens do not equal the planned completed prefix")
        if token_counts[category] > plan.token_caps[category]:
            raise ValueError(f"{category} input tokens exceed its planned cap")
        if earlier_incomplete and count > 0:
            raise ValueError("completed progress violates category execution order")
        if count < plan.forward_caps[category]:
            earlier_incomplete = True

    if forward_counts[_TOTAL] != sum(forward_counts[name] for name in _CATEGORIES):
        raise ValueError("total completed forwards do not equal the category sum")
    if token_counts[_TOTAL] != sum(token_counts[name] for name in _CATEGORIES):
        raise ValueError("total completed input tokens do not equal the category sum")
    if forward_counts[_TOTAL] > plan.total_forward_cap:
        raise ValueError("total completed forwards exceed the role plan")
    if token_counts[_TOTAL] > plan.total_token_cap:
        raise ValueError("total completed input tokens exceed the role plan")


def _validate_pending(
    plan: _RolePlan,
    forward_counts: Mapping[str, int],
    token_counts: Mapping[str, int],
    value: object,
) -> dict[str, object]:
    pending = _copy_json_object(value, "pending_forward")
    if set(pending) != _PENDING_FIELDS:
        raise ValueError("pending forward fields do not match the exact schema")
    category = pending["category"]
    if not isinstance(category, str) or category not in _CATEGORIES:
        raise ValueError("pending forward category is invalid")
    next_category = _next_category(plan, forward_counts)
    if category != next_category:
        raise ValueError("pending forward is not the next planned category")

    index = pending["index"]
    if type(index) is not int or index < 0 or index != forward_counts[category]:
        raise ValueError("pending forward index is not the next planned index")
    spec = plan.specs[category][index]
    spec_digest = json_contracts.validate_sha256(pending["spec_sha256"], "pending spec_sha256")
    if spec_digest != json_contracts.json_sha256(spec):
        raise ValueError("pending forward hash does not match the next planned spec")
    tokens = pending["input_tokens"]
    if type(tokens) is not int or not 1 <= tokens <= contracts.MAX_INPUT_TOKENS:
        raise ValueError("pending forward input_tokens must be a strict positive integer")
    expected_tokens = _input_tokens(spec, category)
    if tokens != expected_tokens:
        raise ValueError("pending forward token count differs from the next planned spec")

    if forward_counts[category] + 1 > plan.forward_caps[category]:
        raise ValueError("pending forward exceeds its category count cap")
    if token_counts[category] + tokens > plan.token_caps[category]:
        raise ValueError("pending forward exceeds its category token cap")
    if forward_counts[_TOTAL] + 1 > plan.total_forward_cap:
        raise ValueError("pending forward exceeds the role count cap")
    if token_counts[_TOTAL] + tokens > plan.total_token_cap:
        raise ValueError("pending forward exceeds the role token cap")
    return pending


def validate_progress(
    value: object,
    *,
    role: object,
    specs_by_category: object,
    require_complete: bool = False,
) -> dict[str, object]:
    """Validate and detach exact partial or complete forward-accounting evidence."""

    plan = _build_plan(role, specs_by_category)
    progress = _copy_json_object(value, "progress")
    if set(progress) != _PROGRESS_FIELDS:
        raise ValueError("progress fields do not match the exact schema")
    forward_counts = _strict_count_map(progress["completed_forward_counts"], "forward counts")
    token_counts = _strict_count_map(progress["completed_input_token_counts"], "input token counts")
    _validate_completed_prefix(plan, forward_counts, token_counts)

    pending_value = progress["pending_forward"]
    if pending_value is None:
        pending = None
    else:
        pending = _validate_pending(plan, forward_counts, token_counts, pending_value)
    progress["pending_forward"] = pending

    if require_complete:
        if pending is not None:
            raise ValueError("pending forward remains unresolved")
        if (
            any(forward_counts[name] != plan.forward_caps[name] for name in _CATEGORIES)
            or forward_counts[_TOTAL] != plan.total_forward_cap
            or any(token_counts[name] != plan.token_caps[name] for name in _CATEGORIES)
            or token_counts[_TOTAL] != plan.total_token_cap
        ):
            raise ValueError("progress does not complete the planned role budget")
    return progress


class ForwardLedger:
    """Track attempts and completed forwards against one detached fixed plan."""

    __slots__ = ("_plan", "_completed_forwards", "_completed_tokens", "_pending")

    def __init__(self, role: object, specs_by_category: object) -> None:
        self._plan = _build_plan(role, specs_by_category)
        self._completed_forwards = _empty_counts()
        self._completed_tokens = _empty_counts()
        self._pending: dict[str, object] | None = None

    def begin(self, category: str, spec: dict[str, object]) -> None:
        """Mark the exact next planned spec as an unresolved model attempt."""

        if self._pending is not None:
            raise ValueError("pending forward must be completed before beginning another")
        if not isinstance(category, str) or category not in _CATEGORIES:
            raise ValueError("unknown progress category")
        next_category = _next_category(self._plan, self._completed_forwards)
        if next_category is None:
            raise ValueError("no incomplete planned forward remains")
        if category != next_category:
            raise ValueError(f"next category is {next_category}")
        index = self._completed_forwards[category]
        planned_spec = self._plan.specs[category][index]
        candidate = _copy_json_object(spec, "spec")
        if json_contracts.canonical_json(candidate) != json_contracts.canonical_json(planned_spec):
            raise ValueError("spec does not match the next planned spec")
        tokens = _input_tokens(planned_spec, category)
        if self._completed_forwards[category] + 1 > self._plan.forward_caps[category]:
            raise ValueError("forward count exceeds its category cap")
        if self._completed_tokens[category] + tokens > self._plan.token_caps[category]:
            raise ValueError("input tokens exceed their category cap")
        if self._completed_forwards[_TOTAL] + 1 > self._plan.total_forward_cap:
            raise ValueError("forward count exceeds the role cap")
        if self._completed_tokens[_TOTAL] + tokens > self._plan.total_token_cap:
            raise ValueError("input tokens exceed the role cap")
        self._pending = {
            "category": category,
            "index": index,
            "spec_sha256": json_contracts.json_sha256(planned_spec),
            "input_tokens": tokens,
        }

    def complete(self) -> None:
        """Record model work immediately after the model call returns."""

        if self._pending is None:
            raise ValueError("no pending forward to complete")
        category = cast(str, self._pending["category"])
        tokens = cast(int, self._pending["input_tokens"])
        new_category_count = self._completed_forwards[category] + 1
        new_total_count = self._completed_forwards[_TOTAL] + 1
        new_category_tokens = self._completed_tokens[category] + tokens
        new_total_tokens = self._completed_tokens[_TOTAL] + tokens
        if new_category_count > self._plan.forward_caps[category]:
            raise ValueError("completed forward exceeds its category cap")
        if new_category_tokens > self._plan.token_caps[category]:
            raise ValueError("completed input tokens exceed their category cap")
        if new_total_count > self._plan.total_forward_cap:
            raise ValueError("completed forward exceeds the role cap")
        if new_total_tokens > self._plan.total_token_cap:
            raise ValueError("completed input tokens exceed the role cap")

        self._completed_forwards[category] = new_category_count
        self._completed_forwards[_TOTAL] = new_total_count
        self._completed_tokens[category] = new_category_tokens
        self._completed_tokens[_TOTAL] = new_total_tokens
        self._pending = None

    def snapshot(self) -> dict[str, object]:
        """Return detached JSON evidence for the ledger's current state."""

        return json_contracts.normalize_json_object(
            {
                "completed_forward_counts": dict(self._completed_forwards),
                "completed_input_token_counts": dict(self._completed_tokens),
                "pending_forward": None if self._pending is None else dict(self._pending),
            },
            "progress",
        )

    def require_complete(self) -> None:
        """Require an exact completed budget and no unresolved model attempt."""

        if self._pending is not None:
            raise ValueError("pending forward remains unresolved")
        if (
            any(
                self._completed_forwards[name] != self._plan.forward_caps[name]
                for name in _CATEGORIES
            )
            or self._completed_forwards[_TOTAL] != self._plan.total_forward_cap
            or any(
                self._completed_tokens[name] != self._plan.token_caps[name] for name in _CATEGORIES
            )
            or self._completed_tokens[_TOTAL] != self._plan.total_token_cap
        ):
            raise ValueError("progress does not complete the planned role budget")
