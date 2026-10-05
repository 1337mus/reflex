"""Strict host-side validation and analysis for targeted matched training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path

from experiments import targeted_training_core as core
from experiments.targeted_training_core import ROLE_CONTROL, ROLE_TREATMENT, ROLE_UNCHANGED, ROLES

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20261015
DIFFERENCES = (
    (ROLE_TREATMENT, ROLE_CONTROL),
    (ROLE_TREATMENT, ROLE_UNCHANGED),
    (ROLE_CONTROL, ROLE_UNCHANGED),
)
IDENTITY_FIELDS = (
    "presentation_id",
    "record_id",
    "dataset_id",
    "source_group_id",
    "request_hash",
    "order_index",
    "order_ids",
)
SCORE_FIELDS = frozenset(
    (*IDENTITY_FIELDS, "candidate_logits", "winner_option_id", "input_tokens", "prompt_sha256")
)
RESERVED_COUNTS = {"targeted-hans-reserved-v1": 600, "targeted-winogrande-reserved-v1": 400}
MONITORING_COUNTS = {"hans-eval-v1": 600, "winogrande-dev-v1": 400, "arc-challenge-dev-v1": 400}
RETENTION_COUNTS = {
    "dbpedia14-pilot-v1-development": 1568,
    "sms-pilot-v1-development": 120,
    "snli-balanced-v1-development": 1152,
    "synthetic-atomic-fact-inference-v1-development": 450,
    "synthetic-numeric-selection-v1-development": 364,
    "boolq-dev-pilot-v1": 64,
    "copa-dev-pilot-v1": 64,
}
TASK_COUNTS = RESERVED_COUNTS | MONITORING_COUNTS | RETENTION_COUNTS
TASK_STRATA = (
    {task: "reserved" for task in RESERVED_COUNTS}
    | {task: "monitoring" for task in MONITORING_COUNTS}
    | {task: "retention" for task in RETENTION_COUNTS}
)


def _mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def _presentation_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    identifier = row.get("presentation_id")
    order_ids = row.get("order_ids")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("presentation ID is malformed")
    if not isinstance(order_ids, Sequence) or isinstance(order_ids, (str, bytes)):
        raise ValueError("presentation order IDs are malformed")
    if len(order_ids) < 2 or any(not isinstance(item, str) or not item for item in order_ids):
        raise ValueError("presentation order IDs are malformed")
    if len(set(order_ids)) != len(order_ids):
        raise ValueError("presentation order IDs are duplicated")
    return (identifier, tuple(order_ids))


def _finite_logit(value: object) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def validate_outputs(
    rows: object, presentations: object, compiled_rows: object
) -> dict[str, dict[str, object]]:
    """Strictly join label-free worker rows to the immutable scoring panel.

    The worker is deliberately unable to provide correctness: this function only
    validates immutable identities and finite candidate-only scores.  Gold labels
    are joined later, after the complete receipt has been accepted on the host.
    """

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("outputs must be a sequence")
    if not isinstance(presentations, Sequence) or isinstance(presentations, (str, bytes)):
        raise ValueError("presentations must be a sequence")
    if not isinstance(compiled_rows, Sequence) or isinstance(compiled_rows, (str, bytes)):
        raise ValueError("compiled scoring rows must be a sequence")
    expected: dict[str, Mapping[str, object]] = {}
    for value in presentations:
        presentation = _mapping(value, "presentation is malformed")
        presentation_id, _ = _presentation_identity(presentation)
        if presentation_id in expected:
            raise ValueError("duplicate presentation ID in panel")
        expected[presentation_id] = presentation
    compiled: dict[str, Mapping[str, object]] = {}
    for value in compiled_rows:
        item = _mapping(value, "compiled scoring row is malformed")
        identifier = item.get("presentation_id")
        tokens = item.get("input_tokens")
        prompt_hash = item.get("prompt_sha256")
        if not isinstance(identifier, str) or not identifier or identifier in compiled:
            raise ValueError("compiled scoring row identity is missing or duplicated")
        if type(tokens) is not int or not 0 < tokens <= core.MAX_INPUT_TOKENS:
            raise ValueError("compiled scoring token count is malformed")
        if not isinstance(prompt_hash, str) or not prompt_hash:
            raise ValueError("compiled scoring prompt identity is malformed")
        if item.get("request_hash") != expected.get(identifier, {}).get("request_hash"):
            raise ValueError("compiled scoring request identity differs from panel")
        compiled[identifier] = item
    if set(compiled) != set(expected):
        raise ValueError("compiled scoring row membership differs from panel")
    actual: dict[str, dict[str, object]] = {}
    for value in rows:
        row = _mapping(value, "output row is malformed")
        presentation_id = row.get("presentation_id")
        if not isinstance(presentation_id, str) or not presentation_id:
            raise ValueError("output presentation ID is malformed")
        if presentation_id in actual:
            raise ValueError("duplicate output presentation ID")
        if presentation_id not in expected:
            raise ValueError("output has an extra presentation ID")
        if set(row) != SCORE_FIELDS:
            raise ValueError("output score identity is incomplete or contains extra fields")
        presentation = expected[presentation_id]
        for key in IDENTITY_FIELDS:
            if key == "order_index" and type(row[key]) is not int:
                raise ValueError("output order index must be an integer")
            if row[key] != presentation.get(key):
                raise ValueError("output immutable presentation identity is incompatible")
        reference = compiled[presentation_id]
        if row["input_tokens"] != reference["input_tokens"] or type(row["input_tokens"]) is not int:
            raise ValueError("output token count differs from compiled scoring row")
        if row["prompt_sha256"] != reference["prompt_sha256"]:
            raise ValueError("output prompt differs from compiled scoring row")
        order_ids = presentation["order_ids"]
        winner = row.get("winner_option_id")
        logits = row.get("candidate_logits")
        if (
            not isinstance(winner, str)
            or winner not in order_ids
            or not isinstance(logits, Sequence)
            or isinstance(logits, (str, bytes))
            or len(logits) != len(order_ids)
            or any(not _finite_logit(score) for score in logits)
        ):
            raise ValueError("output winner or candidate logits are malformed")
        maximum = max(float(score) for score in logits)
        expected_winner = min(
            option_id
            for option_id, score in zip(order_ids, logits, strict=True)
            if score == maximum
        )
        if winner != expected_winner:
            raise ValueError("output winner does not match candidate logits")
        actual[presentation_id] = dict(row)
    if set(actual) != set(expected):
        raise ValueError("outputs are missing panel presentation IDs")
    return actual


def _canonical_digest(value: object) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("receipt binding is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def validate_complete_receipt(
    receipt: object, *, expected_plan_sha256: str | None = None, plan: object | None = None
) -> dict[str, object]:
    """Validate execution evidence before a caller can read any host-only gold."""
    outer = _mapping(receipt, "complete receipt must be an object")
    if (
        outer.get("status") != "passed"
        or type(outer.get("schema_version")) is not int
        or outer["schema_version"] != 1
    ):
        raise ValueError("only a passed schema-1 receipt can be analyzed")
    if outer.get("experiment_id") != core.EXPERIMENT_ID:
        raise ValueError("receipt experiment identity differs")
    run_id = outer.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("receipt run identity is missing")
    pinned_plan = _mapping(outer.get("plan"), "immutable plan is missing")
    stored_sha = pinned_plan.get("plan_sha256")
    if (
        not isinstance(expected_plan_sha256, str)
        or len(expected_plan_sha256) != 64
        or stored_sha != expected_plan_sha256
        or stored_sha
        != _canonical_digest(
            {key: value for key, value in pinned_plan.items() if key != "plan_sha256"}
        )
    ):
        raise ValueError("immutable plan digest differs from reviewed SHA")
    if plan is not None and pinned_plan != _mapping(plan, "explicit plan is malformed"):
        raise ValueError("receipt plan differs from the explicit immutable plan")
    if (
        pinned_plan.get("experiment_id") != core.EXPERIMENT_ID
        or pinned_plan.get("run_id") != run_id
    ):
        raise ValueError("plan experiment or run identity differs")
    lifecycle = _mapping(outer.get("lifecycle"), "host lifecycle is missing")
    app_id = lifecycle.get("app_id")
    if lifecycle.get("status") != "passed" or not isinstance(app_id, str) or not app_id:
        raise ValueError("host lifecycle did not pass with a valid app ID")
    calls = _mapping(lifecycle.get("calls"), "host FunctionCall identities are missing")
    if (
        set(calls) != set(ROLES)
        or any(not isinstance(identifier, str) or not identifier for identifier in calls.values())
        or len(set(calls.values())) != len(ROLES)
    ):
        raise ValueError("host FunctionCall identities are incomplete or duplicated")
    teardown = _mapping(lifecycle.get("teardown"), "host teardown evidence is missing")
    observations = teardown.get("observations")
    if (
        teardown.get("app_id") != app_id
        or teardown.get("verified") is not True
        or not isinstance(observations, list)
        or not observations
        or not any(
            isinstance(item, Mapping)
            and item.get("state") == "APP_STATE_STOPPED"
            and type(item.get("n_tasks")) is int
            and item["n_tasks"] == 0
            for item in observations
        )
    ):
        raise ValueError("host teardown does not prove the app stopped with zero tasks")
    payloads = _mapping(outer.get("role_payloads"), "receipt role payloads are missing")
    roles = _mapping(outer.get("roles"), "receipt roles are missing")
    if set(payloads) != set(ROLES) or set(roles) != set(ROLES):
        raise ValueError("receipt role or payload set differs from the fixed three roles")
    plan_payloads = _mapping(
        pinned_plan.get("role_payload_sha256"), "plan role payload digest map is missing"
    )
    if set(plan_payloads) != set(ROLES):
        raise ValueError("plan role payload digest map differs from fixed roles")
    for role in ROLES:
        payload = _mapping(payloads[role], "role payload is malformed")
        digest = payload.get("payload_sha256")
        if (
            payload.get("role") != role
            or payload.get("run_id") != run_id
            or payload.get("experiment_id") != core.EXPERIMENT_ID
            or not isinstance(digest, str)
            or digest
            != _canonical_digest(
                {key: value for key, value in payload.items() if key != "payload_sha256"}
            )
        ):
            raise ValueError("role payload identity or digest differs")
        if plan_payloads[role] != digest:
            raise ValueError("plan role payload digest differs from receipt")
        wrapper = _mapping(roles[role], "role receipt is malformed")
        raw = _mapping(wrapper.get("raw_worker_result"), "raw worker result is missing")
        validated = _mapping(wrapper.get("validated_result"), "validated worker result is missing")
        if (
            raw.get("status") != "passed"
            or raw.get("run_id") != run_id
            or raw.get("experiment_id") != core.EXPERIMENT_ID
            or raw.get("role") != role
            or raw.get("payload_sha256") != digest
        ):
            raise ValueError("worker result identity or status differs")
        accepted = core.validate_completed_result(dict(payload), dict(raw))
        if _canonical_digest(raw) != _canonical_digest(validated) or _canonical_digest(
            raw
        ) != _canonical_digest(accepted):
            raise ValueError("stored validated result differs from the raw completed result")
    return dict(outer)


def _type7(values: Sequence[Fraction], probability: Fraction) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] + weight * (ordered[upper] - ordered[lower]))


def _draws(group_ids: Sequence[str], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    if not group_ids:
        raise ValueError("a dataset needs at least one analysis group")
    return tuple(
        tuple(group_ids[rng.randrange(len(group_ids))] for _ in group_ids)
        for _ in range(BOOTSTRAP_REPLICATES)
    )


def _gold_record(value: object) -> tuple[str, str, str, object, str]:
    from reflex_decisions.data import DecisionRecord
    from reflex_decisions.schema import DecisionRequest

    if isinstance(value, DecisionRecord):
        record = value
    elif isinstance(value, Mapping):
        # The single host gold field is answer_id; aliases obscure malformed data.
        if "answer_id" not in value or "gold_option_id" in value:
            raise ValueError("evaluation record has no unambiguous answer_id")
        record = DecisionRecord.model_validate(value)
    else:
        raise ValueError("evaluation record is malformed")
    request = record.request
    if not isinstance(request, DecisionRequest):
        raise ValueError("evaluation request is malformed")
    return record.record_id, record.dataset_id, record.source_group_id, request, record.answer_id


def _gold_panel(
    evaluation_records: object, presentations: object
) -> tuple[list[Mapping[str, object]], dict[str, tuple[str, str, object, str]]]:
    from reflex_decisions.schema import DecisionRequest

    if not isinstance(evaluation_records, Sequence) or isinstance(evaluation_records, (str, bytes)):
        raise ValueError("evaluation records must be a sequence")
    if not isinstance(presentations, Sequence) or isinstance(presentations, (str, bytes)):
        raise ValueError("presentations must be a sequence")
    gold: dict[str, tuple[str, str, object, str]] = {}
    for value in evaluation_records:
        record_id, dataset_id, group_id, request, answer = _gold_record(value)
        if record_id in gold:
            raise ValueError("duplicate evaluation record ID")
        gold[record_id] = (dataset_id, group_id, request, answer)
    panel: list[Mapping[str, object]] = []
    seen_presentations: set[str] = set()
    seen_orders: set[tuple[str, int]] = set()
    used_records: set[str] = set()
    for value in presentations:
        row = _mapping(value, "presentation is malformed")
        presentation_id, order_ids = _presentation_identity(row)
        if presentation_id in seen_presentations:
            raise ValueError("duplicate panel presentation ID")
        seen_presentations.add(presentation_id)
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in gold:
            raise ValueError("presentation cannot join to exact host gold")
        dataset_id, group_id, request, _ = gold[record_id]
        if row.get("dataset_id") != dataset_id or row.get("source_group_id") != group_id:
            raise ValueError("presentation dataset or group differs from gold")
        if row.get("request_hash") != request.request_hash:
            raise ValueError("presentation request hash differs from gold")
        if set(order_ids) != {option.id for option in request.options}:
            raise ValueError("presentation option IDs differ from gold request")
        if "request" not in row:
            raise ValueError("presentation request is missing")
        panel_request = DecisionRequest.model_validate(row["request"])
        if (
            panel_request.request_hash != request.request_hash
            or tuple(option.id for option in panel_request.options) != order_ids
        ):
            raise ValueError("presentation request or order differs from gold")
        order_index = row.get("order_index")
        if (
            type(order_index) is not int
            or order_index < 0
            or (record_id, order_index) in seen_orders
        ):
            raise ValueError("presentation order index is missing or duplicated")
        seen_orders.add((record_id, order_index))
        used_records.add(record_id)
        panel.append(row)
    if not panel or used_records != set(gold):
        raise ValueError("host gold membership differs from the exact panel")
    if any((record_id, 0) not in seen_orders for record_id in gold):
        raise ValueError("panel lacks an original order for a question")
    return panel, gold


def summarize(
    outputs_by_role: Mapping[str, object],
    evaluation_records: object,
    presentations: object,
    compiled_by_role: Mapping[str, object],
    strata: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Compute host-gold metrics after execution evidence was independently accepted."""
    if set(outputs_by_role) != set(ROLES) or set(compiled_by_role) != set(ROLES):
        raise ValueError("all three role outputs and compilations are required")
    panel, gold = _gold_panel(evaluation_records, presentations)
    validated = {
        role: validate_outputs(outputs_by_role[role], panel, compiled_by_role[role])
        for role in ROLES
    }
    datasets = sorted({str(row["dataset_id"]) for row in panel})
    if strata is not None and set(strata) != set(datasets):
        raise ValueError("analysis strata differ from panel tasks")
    rng = random.Random(BOOTSTRAP_SEED)
    report: dict[str, object] = {}
    class_report: dict[str, object] = {}
    for dataset_id in datasets:
        subset = [row for row in panel if row["dataset_id"] == dataset_id]
        groups: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in subset:
            groups[str(row["source_group_id"])][str(row["record_id"])].append(row)
        group_ids = sorted(groups)
        draws = _draws(group_ids, rng)
        role_metrics: dict[str, object] = {}
        group_scores: dict[str, dict[str, Fraction]] = {}
        two_order_panel = all(
            len(question_rows) == 2
            for questions in groups.values()
            for question_rows in questions.values()
        )
        for role in ROLES:
            correct = 0
            original_correct = 0
            original_total = 0
            changes = 0
            per_group: dict[str, Fraction] = {}
            for group_id in group_ids:
                question_scores: list[Fraction] = []
                for record_id in sorted(groups[group_id]):
                    question_rows = groups[group_id][record_id]
                    hits = 0
                    winners: set[str] = set()
                    for row in question_rows:
                        winner = validated[role][str(row["presentation_id"])]["winner_option_id"]
                        hit = int(winner == gold[record_id][3])
                        hits += hit
                        correct += hit
                        winners.add(str(winner))
                        if row["order_index"] == 0:
                            original_correct += hit
                            original_total += 1
                    question_scores.append(Fraction(hits, len(question_rows)))
                    if two_order_panel and len(winners) > 1:
                        changes += 1
                per_group[group_id] = sum(question_scores, Fraction(0)) / len(question_scores)
            group_scores[role] = per_group
            mean = sum(per_group.values(), Fraction(0)) / len(per_group)
            metrics: dict[str, object] = {
                "correct": correct,
                "presentations": len(subset),
                "accuracy": correct / len(subset),
                "original_order_correct": original_correct,
                "original_order_total": original_total,
                "original_order_accuracy": original_correct / original_total,
                "questions": sum(len(questions) for questions in groups.values()),
                "groups": len(groups),
                "group_mean_accuracy": float(mean),
                "group_mean_fraction": str(mean),
            }
            if two_order_panel:
                metrics["order_change_count"] = changes
            role_metrics[role] = metrics
        differences: dict[str, object] = {}
        for left, right in DIFFERENCES:
            deltas = {
                group: group_scores[left][group] - group_scores[right][group] for group in group_ids
            }
            point = sum(deltas.values(), Fraction(0)) / len(group_ids)
            samples = [
                sum((deltas[group] for group in draw), Fraction(0)) / len(draw) for draw in draws
            ]
            differences[f"{left}-{right}"] = {
                "group_mean_difference": float(point),
                "group_mean_difference_fraction": str(point),
                "bootstrap_95": {
                    "lower": _type7(samples, Fraction(1, 40)),
                    "upper": _type7(samples, Fraction(39, 40)),
                },
            }
        report[dataset_id] = {
            "stratum": strata[dataset_id] if strata else None,
            "roles": role_metrics,
            "differences": differences,
        }
        if dataset_id == "targeted-hans-reserved-v1":
            for answer in ("entailment", "non-entailment"):
                class_rows = [row for row in subset if gold[str(row["record_id"])][3] == answer]
                if not class_rows:
                    raise ValueError("reserved HANS answer class is empty")
                class_metrics = {}
                for role in ROLES:
                    class_correct = sum(
                        validated[role][str(row["presentation_id"])]["winner_option_id"] == answer
                        for row in class_rows
                    )
                    class_metrics[role] = {
                        "correct": class_correct,
                        "presentations": len(class_rows),
                        "accuracy": class_correct / len(class_rows),
                        "accuracy_fraction": str(Fraction(class_correct, len(class_rows))),
                    }
                delta = Fraction(class_metrics[ROLE_TREATMENT]["accuracy_fraction"]) - Fraction(
                    class_metrics[ROLE_UNCHANGED]["accuracy_fraction"]
                )
                class_report[answer] = {
                    "roles": class_metrics,
                    "treatment_unchanged_difference": float(delta),
                    "treatment_unchanged_fraction": str(delta),
                }
    return {
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "method": "paired group bootstrap, type-7 percentiles",
        },
        "datasets": report,
        "hans_classes": class_report,
    }


def evaluate_learning(report: Mapping[str, object]) -> dict[str, object]:
    """Apply the fixed 24 independent learning checks."""
    datasets = _mapping(report.get("datasets"), "learning report datasets are malformed")
    classes = _mapping(report.get("hans_classes"), "reserved HANS class report is malformed")
    if set(datasets) != set(TASK_COUNTS) or set(classes) != {"entailment", "non-entailment"}:
        raise ValueError("learning report task or HANS class membership differs")
    checks: list[dict[str, object]] = []

    def difference(task: str, contrast: str) -> Fraction:
        item = _mapping(datasets[task], "task report is malformed")
        contrasts = _mapping(item.get("differences"), "task contrasts are malformed")
        value = _mapping(contrasts.get(contrast), "task contrast is malformed")
        rational = value.get("group_mean_difference_fraction")
        if not isinstance(rational, str):
            raise ValueError("exact group-mean difference is missing")
        return Fraction(rational)

    def add(
        name: str, value: Fraction | float, threshold: Fraction, *, strict: bool = False
    ) -> None:
        passed = value > threshold if strict else value >= threshold
        checks.append(
            {
                "name": name,
                "passed": passed,
                "observed": float(value),
                "threshold": float(threshold),
                "strict": strict,
            }
        )

    for task in sorted(RESERVED_COUNTS):
        gain = difference(task, "treatment-control")
        add(f"{task}.treatment_control_gain", gain, Fraction(1, 20))
        item = _mapping(datasets[task], "task report is malformed")
        contrasts = _mapping(item.get("differences"), "task contrasts are malformed")
        paired = _mapping(contrasts.get("treatment-control"), "paired contrast is malformed")
        interval = _mapping(paired.get("bootstrap_95"), "paired interval is malformed")
        lower = interval.get("lower")
        if type(lower) not in {int, float} or not math.isfinite(float(lower)):
            raise ValueError("paired lower bound is malformed")
        add(f"{task}.paired_lower_positive", float(lower), Fraction(0), strict=True)
        add(
            f"{task}.treatment_unchanged_gain",
            difference(task, "treatment-unchanged"),
            Fraction(1, 20),
        )
    for answer in ("entailment", "non-entailment"):
        item = _mapping(classes[answer], "HANS answer class is malformed")
        rational = item.get("treatment_unchanged_fraction")
        if not isinstance(rational, str):
            raise ValueError("exact HANS class difference is missing")
        add(f"targeted-hans-reserved-v1.{answer}.class_floor", Fraction(rational), Fraction(-1, 20))
    for task in sorted((*RETENTION_COUNTS, "arc-challenge-dev-v1")):
        for contrast in ("treatment-control", "treatment-unchanged"):
            add(
                f"{task}.{contrast.replace('-', '_')}_floor",
                difference(task, contrast),
                Fraction(-1, 20),
            )
    if len(checks) != 24:
        raise ValueError("learning check inventory differs from 24 fixed gates")
    failed = [str(item["name"]) for item in checks if not item["passed"]]
    return {
        "passed": not failed,
        "promotion_eligible": not failed,
        "checks": checks,
        "failed_check_names": failed,
    }


def analyze_receipt(
    receipt: object, inputs: object, *, expected_plan_sha256: str, plan: object
) -> dict[str, object]:
    """Validate a complete execution before opening host-only evaluation answers."""
    accepted = validate_complete_receipt(
        receipt, expected_plan_sha256=expected_plan_sha256, plan=plan
    )
    panel = getattr(inputs, "presentations", None)
    strata = getattr(inputs, "strata", None)
    if not isinstance(panel, (tuple, list)) or not isinstance(strata, dict):
        raise ValueError("host label-free panel or strata are malformed")
    if len(panel) != core.FINAL_EVALUATION_FORWARDS:
        raise ValueError("host panel count differs from fixed final evaluation")
    if set(strata) != set(TASK_COUNTS) or strata != TASK_STRATA:
        raise ValueError("host task strata differ from the fixed study")
    counts: Counter[str] = Counter()
    for item in panel:
        row = _mapping(item, "host presentation is malformed")
        if set(row) != set((*IDENTITY_FIELDS, "request")):
            raise ValueError("host presentation is not exactly label-free")
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str):
            raise ValueError("host presentation task is malformed")
        counts[dataset_id] += 1
    if counts != TASK_COUNTS:
        raise ValueError("host panel task counts differ from the fixed study")
    role_payloads = _mapping(accepted["role_payloads"], "role payloads are missing")
    role_receipts = _mapping(accepted["roles"], "role receipts are missing")
    outputs_by_role: dict[str, object] = {}
    compiled_by_role: dict[str, object] = {}
    for role in ROLES:
        payload = _mapping(role_payloads[role], "role payload is malformed")
        if _canonical_digest(payload.get("evaluation_presentations")) != _canonical_digest(panel):
            raise ValueError("role scoring panel differs from frozen host presentations")
        compilation = _mapping(payload.get("compiled"), "role compilation is missing")
        compiled = compilation.get("final_evaluation")
        role_wrapper = _mapping(role_receipts[role], "role receipt is malformed")
        result = _mapping(role_wrapper.get("validated_result"), "validated result is missing")
        if _canonical_digest(result.get("compiled")) != _canonical_digest(compilation):
            raise ValueError("worker compilation differs from frozen payload")
        outputs = result.get("outputs")
        validate_outputs(outputs, panel, compiled)
        outputs_by_role[role] = outputs
        compiled_by_role[role] = compiled
    # All execution, score identity, prompt, and token checks above precede this read.
    records = inputs.evaluation_records
    metrics = summarize(outputs_by_role, records, panel, compiled_by_role, strata)
    learning = evaluate_learning(metrics)
    return {
        "experiment_id": core.EXPERIMENT_ID,
        "run_id": accepted["run_id"],
        "plan_sha256": expected_plan_sha256,
        "execution_passed": True,
        "learning": learning,
        "metrics": metrics,
        "limitations": [
            "Per-task intervals are descriptive except reserved "
            "treatment-control lower-bound gates.",
            "Retention floors are operational point checks, not noninferiority claims.",
            "One training seed cannot establish seed robustness.",
            "Combined treatment cannot isolate either targeted dataset contribution.",
            "No unseen-template or pretraining-exposure claim is supported.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Analyze a complete host receipt using explicit reviewed inputs."""
    parser = argparse.ArgumentParser(description="Analyze one completed targeted study receipt")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(args.receipt.read_text())
        plan = json.loads(args.plan.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("explicit receipt or plan is unavailable or malformed") from exc
    # This must happen before the data loader can open host-only answer records.
    validate_complete_receipt(receipt, expected_plan_sha256=args.expected_plan_sha256, plan=plan)
    from experiments.targeted_training_data import load_inputs

    inputs = load_inputs(args.root)
    report = analyze_receipt(
        receipt, inputs, expected_plan_sha256=args.expected_plan_sha256, plan=plan
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    learning = _mapping(report["learning"], "learning result is malformed")
    failed = learning["failed_check_names"]
    print(
        f"Execution passed; learning {'passed' if learning['passed'] else 'failed'}; "
        f"{24 - len(failed)}/24 checks passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
