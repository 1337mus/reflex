"""CPU-only analysis for controlled mixture training receipts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import experiments.mixture_training_baselines as baselines
import experiments.mixture_training_core as core
import experiments.mixture_training_data as mixture_data
from experiments.mixture_training_contracts import (
    PROFILE,
    SCHEMA_VERSION,
    WORKSPACE,
    canonical_json,
    normalize_json_object,
    validate_safe_run_id,
)
from experiments.mixture_training_statistics import (
    _exact,
    _gate_report,
    _state_analysis,
    paired_group_delta,
    passes_minimum,
)
from reflex_decisions.data import DecisionRecord

__all__ = ["analyze_experiment", "paired_group_delta", "passes_minimum"]

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
_BASE_STATES = ("base", "real_only", "synthetic_mix")
_RESULT_NAMES = ("initialize", "real_only", "synthetic_mix")
_OUTPUT_IDENTITY = (
    "presentation_id",
    "record_id",
    "dataset_id",
    "source_group_id",
    "request_hash",
    "order_index",
    "order_ids",
)
_DATA_DISCLOSURE = (
    "Synthetic atomic-fact inference is related to SNLI reasoning. The balanced SNLI panel is "
    "development data, not a sealed test. Results describe one training seed and do not show "
    "training-seed uncertainty or broad generalization."
)


def _combine_base_outputs(
    reused: Mapping[str, Mapping[str, object]],
    initialized: Sequence[Mapping[str, object]],
    panel: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    outputs = {key: dict(value) for key, value in reused.items()}
    for row in initialized:
        presentation_id = row.get("presentation_id")
        if not isinstance(presentation_id, str) or presentation_id in outputs:
            raise ValueError("initialization outputs contain a duplicate presentation ID")
        outputs[presentation_id] = dict(row)
    expected = {row.get("presentation_id"): row for row in panel}
    if len(expected) != len(panel) or set(outputs) != set(expected):
        raise ValueError("combined base outputs do not cover the exact evaluation panel")
    for presentation_id, output in outputs.items():
        expected_row = expected[presentation_id]
        if any(output.get(key) != expected_row[key] for key in _OUTPUT_IDENTITY):
            raise ValueError("combined base output identity differs from the evaluation panel")
    if len(outputs) != mixture_data.EXPECTED_EVALUATION_PRESENTATIONS:
        raise ValueError("combined base output count differs from the frozen 4,023-row panel")
    return outputs


def _payloads_and_results(
    receipt: object,
    real_records: Sequence[DecisionRecord],
    balanced_records: Sequence[DecisionRecord],
    synthetic_records: Sequence[DecisionRecord],
    pins: Mapping[str, object],
    panel: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    envelope = normalize_json_object(receipt, "mixture receipt")
    if set(envelope) != {
        "schema_version",
        "experiment_id",
        "status",
        "payloads",
        "results",
        "failure",
        "modal",
    }:
        raise ValueError("mixture receipt has an unexpected schema")
    if type(envelope["schema_version"]) is not int or envelope["schema_version"] != SCHEMA_VERSION:
        raise ValueError("mixture receipt schema version is unsupported")
    experiment_id = validate_safe_run_id(envelope["experiment_id"], "experiment_id")
    if envelope["status"] not in {"passed", "failed"}:
        raise ValueError("mixture receipt status is invalid")
    payload_values = envelope["payloads"]
    result_values = envelope["results"]
    if not isinstance(payload_values, Mapping) or set(payload_values) - set(_RESULT_NAMES):
        raise ValueError("mixture receipt payload map is malformed")
    if not isinstance(result_values, Mapping) or set(result_values) - set(_RESULT_NAMES):
        raise ValueError("mixture receipt result map is malformed")
    if set(result_values) - set(payload_values):
        raise ValueError("mixture receipt has a result without its payload")

    evaluation_panel = tuple(dict(row) for row in panel)
    initialization_panel = mixture_data.build_synthetic_evaluation_presentations(synthetic_records)
    train_records = tuple(
        row for row in real_records if row.dataset_id in mixture_data.REAL_TRAIN_DATASET_COUNTS
    ) + tuple(
        row
        for row in synthetic_records
        if row.dataset_id in mixture_data.SYNTHETIC_TRAIN_DATASET_COUNTS
    )
    payloads: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    nonces: set[str] = set()
    for name in _RESULT_NAMES:
        raw_payload = payload_values.get(name)
        if raw_payload is None:
            continue
        saved = core.validate_payload(raw_payload)
        phase = "initialize" if name == "initialize" else "train"
        arm = None if name == "initialize" else name
        initialization = None
        if phase == "train":
            init = results.get("initialize")
            evidence = init.get("evidence") if init else None
            initialization = (
                evidence.get("initialization") if isinstance(evidence, Mapping) else None
            )
            if initialization is None:
                raise ValueError(
                    "training payload exists without a passed initialization descriptor"
                )
        expected = core.build_payload(
            experiment_id=experiment_id,
            run_id=cast(str, saved["run_id"]),
            nonce=cast(str, saved["nonce"]),
            phase=phase,
            arm=arm,
            pins=pins,
            train_records=() if phase == "initialize" else train_records,
            evaluation_presentations=initialization_panel
            if phase == "initialize"
            else evaluation_panel,
            initialization=cast(Mapping[str, object] | None, initialization),
        )
        if saved != expected:
            raise ValueError(f"saved {name} payload differs from the frozen local plan")
        nonce = cast(str, saved["nonce"])
        if nonce in nonces:
            raise ValueError("mixture payloads must use distinct nonces")
        nonces.add(nonce)
        payloads[name] = saved
        raw_result = result_values.get(name)
        if raw_result is not None:
            result = core.validate_result(raw_result, saved)
            results[name] = result

    if set(result_values) != set(results):
        raise ValueError("one or more mixture results could not be validated")
    if "initialize" in results and results["initialize"].get("status") == "failed":
        if set(payloads) != {"initialize"}:
            raise ValueError("training payloads cannot exist after failed initialization")
    failure = envelope["failure"]
    if failure is not None and (
        not isinstance(failure, Mapping)
        or set(failure) != {"stage", "type", "message"}
        or any(not isinstance(failure[key], str) or not failure[key].strip() for key in failure)
    ):
        raise ValueError("mixture receipt lifecycle failure is malformed")
    if envelope["status"] == "passed":
        if (
            failure is not None
            or set(payloads) != set(_RESULT_NAMES)
            or set(results) != set(_RESULT_NAMES)
        ):
            raise ValueError("passed mixture receipt is incomplete")
        if any(result.get("status") != "passed" for result in results.values()):
            raise ValueError("passed mixture receipt contains a failed result")
        modal = envelope["modal"]
        if not isinstance(modal, Mapping) or (
            modal.get("profile") != PROFILE
            or modal.get("workspace") != WORKSPACE
            or modal.get("sdk_version") != "1.6.1"
        ):
            raise ValueError("passed mixture receipt has invalid Modal identity")
    return envelope, payloads, results


def _partial_evidence(
    envelope: Mapping[str, object],
    payloads: Mapping[str, Mapping[str, object]],
    results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    states: dict[str, object] = {}
    for name in _RESULT_NAMES:
        result = results.get(name)
        if result is None:
            states[name] = {"status": "missing", "payload_present": name in payloads}
            continue
        evidence = result.get("evidence")
        evidence = evidence if isinstance(evidence, Mapping) else {}
        counts = evidence.get("forward_counts")
        tokens = evidence.get("input_token_counts")
        outputs = evidence.get("outputs")
        artifact = evidence.get("output_artifact")
        states[name] = {
            "status": result.get("status"),
            "phase": result.get("phase"),
            "forward_counts": counts,
            "input_token_counts": tokens,
            "completed_output_count": len(outputs) if isinstance(outputs, list) else None,
            "output_artifact": artifact,
            "failure": result.get("failure"),
        }
    return {
        "receipt_status": envelope.get("status"),
        "receipt_failure": envelope.get("failure"),
        "states": states,
    }


def _failed_analysis(
    envelope: Mapping[str, object],
    payloads: Mapping[str, Mapping[str, object]],
    results: Mapping[str, Mapping[str, object]],
    reasons: Sequence[str],
) -> dict[str, object]:
    return {
        "analysis_protocol": "reflex-mixture-training-analysis-v1",
        "status": "failed",
        "run_identity": {"experiment_id": envelope.get("experiment_id")},
        "failure": {"reasons": list(reasons)},
        "partial_evidence": _partial_evidence(envelope, payloads, results),
        "primary_transfer_contrast": {
            "status": "not_evaluated",
            "reason": "both arms and shared initialization did not pass",
        },
        "engineering_gate": {
            "evaluated": False,
            "components": {},
            "overall_passed": False,
        },
        "data_disclosure": _DATA_DISCLOSURE,
    }


def _result_outputs(result: Mapping[str, object], label: str) -> dict[str, dict[str, object]]:
    evidence = result.get("evidence")
    outputs = evidence.get("outputs") if isinstance(evidence, Mapping) else None
    if not isinstance(outputs, list):
        raise ValueError(f"passed {label} result has no scored outputs")
    mapped: dict[str, dict[str, object]] = {}
    for row in outputs:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} output row is malformed")
        presentation_id = row.get("presentation_id")
        if not isinstance(presentation_id, str) or presentation_id in mapped:
            raise ValueError(f"{label} outputs contain a duplicate presentation ID")
        mapped[presentation_id] = dict(row)
    return mapped


def _check_prompt_token_parity(
    base: Mapping[str, Mapping[str, object]], final: Mapping[str, Mapping[str, object]], label: str
) -> None:
    if set(base) != set(final):
        raise ValueError(f"{label} final rows do not match the reused base evaluation panel")
    for presentation_id in base:
        old, new = base[presentation_id], final[presentation_id]
        if any(old.get(key) != new.get(key) for key in _OUTPUT_IDENTITY):
            raise ValueError(f"{label} final row has changed identity or option order")
        if any(old.get(key) != new.get(key) for key in ("prompt_sha256", "input_tokens")):
            raise ValueError(f"{label} final row changed prompt or tokenizer behavior")


def _historical_reference_rates(
    reuse_metadata: Mapping[str, object],
    real_records: Sequence[DecisionRecord],
) -> dict[str, dict[str, Fraction]]:
    raw_groups = reuse_metadata.get("historical_reference_group_counts")
    if not isinstance(raw_groups, Mapping) or set(raw_groups) != set(
        baselines.HISTORICAL_ACCURACY_ANCHORS
    ):
        raise ValueError("validated historical reference group counts are missing")
    rates: dict[str, dict[str, Fraction]] = {}
    for dataset_id, (
        expected_correct,
        expected_total,
    ) in baselines.HISTORICAL_ACCURACY_ANCHORS.items():
        panel_groups = [row.source_group_id for row in real_records if row.dataset_id == dataset_id]
        if len(panel_groups) != expected_total or len(set(panel_groups)) != expected_total:
            raise ValueError(
                f"historical anchor panel must contain one record per source group: {dataset_id}"
            )
        groups = raw_groups[dataset_id]
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError(f"historical reference groups are malformed for {dataset_id}")
        if set(groups) != set(panel_groups):
            raise ValueError(
                "historical reference and local panel must contain the same "
                f"source groups: {dataset_id}"
            )
        rates[dataset_id] = {}
        correct_total = 0
        presentation_total = 0
        for group_id, raw_counts in groups.items():
            if (
                not isinstance(group_id, str)
                or not isinstance(raw_counts, Mapping)
                or type(raw_counts.get("correct")) is not int
                or type(raw_counts.get("total")) is not int
                or raw_counts["total"] < 1
                or raw_counts["total"] != 1
                or not 0 <= raw_counts["correct"] <= raw_counts["total"]
            ):
                raise ValueError(
                    f"historical reference group counts are malformed for {dataset_id}"
                )
            rates[dataset_id][group_id] = Fraction(raw_counts["correct"], raw_counts["total"])
            correct_total += raw_counts["correct"]
            presentation_total += raw_counts["total"]
        if (correct_total, presentation_total) != (expected_correct, expected_total):
            raise ValueError(
                f"historical reference accuracy differs from its frozen anchor: {dataset_id}"
            )
    return rates


def _artifact_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def analyze_experiment(
    receipt: object,
    real_records: Sequence[DecisionRecord],
    balanced_records: Sequence[DecisionRecord],
    synthetic_records: Sequence[DecisionRecord],
    real_receipt: object,
    balanced_receipt: object,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Validate a paired mixture receipt, then report frozen metrics and gates."""

    project_root = Path(root or DEFAULT_ROOT).resolve(strict=True)
    local_real, local_balanced, local_synthetic, pins = core.load_local_data(project_root)
    if (tuple(real_records), tuple(balanced_records), tuple(synthetic_records)) != (
        local_real,
        local_balanced,
        local_synthetic,
    ):
        raise ValueError("analysis records differ from the eleven pinned local data files")
    panel = mixture_data.build_evaluation_presentations(local_real, local_balanced, local_synthetic)
    envelope, payloads, results = _payloads_and_results(
        receipt,
        local_real,
        local_balanced,
        local_synthetic,
        pins,
        panel,
    )
    passed = (
        envelope["status"] == "passed"
        and set(results) == set(_RESULT_NAMES)
        and all(result.get("status") == "passed" for result in results.values())
    )
    if not passed:
        reasons = []
        if envelope["failure"] is not None:
            failure = cast(Mapping[str, object], envelope["failure"])
            reasons.append(
                f"lifecycle failure at {failure['stage']}: {failure['type']}: {failure['message']}"
            )
        for name in _RESULT_NAMES:
            result = results.get(name)
            if result is None:
                reasons.append(f"{name} result is missing")
            elif result.get("status") != "passed":
                failure = result.get("failure")
                reasons.append(f"{name} result failed: {failure}")
        return _failed_analysis(
            envelope, payloads, results, reasons or ["mixture pair did not pass"]
        )

    initialization = results["initialize"]
    init_evidence = cast(Mapping[str, object], initialization["evidence"])
    init_provenance = cast(Mapping[str, object], initialization["provenance"])
    init_identity = baselines._base_identity(init_provenance, "mixture initialization")
    init_versions = baselines._runtime_versions(init_provenance, "mixture initialization")
    for name in ("real_only", "synthetic_mix"):
        arm_provenance = cast(Mapping[str, object], results[name]["provenance"])
        arm_identity = baselines._base_identity(arm_provenance, f"{name} training arm")
        arm_versions = baselines._runtime_versions(arm_provenance, f"{name} training arm")
        if arm_identity != init_identity or arm_versions != init_versions:
            raise ValueError(
                "training arm base/tokenizer identity differs from shared initialization"
            )
    reused, reuse_metadata = baselines.validate_reused_base_outputs(
        root=project_root,
        panel=panel,
        real_records=local_real,
        balanced_records=local_balanced,
        new_base_provenance=cast(Mapping[str, object], initialization["provenance"]),
        source_hashes=cast(
            Mapping[str, str], cast(Mapping[str, object], pins)["source_file_sha256"]
        ),
        real_receipt=real_receipt,
        balanced_receipt=balanced_receipt,
    )
    base_outputs = _combine_base_outputs(
        reused,
        cast(list[Mapping[str, object]], init_evidence["outputs"]),
        panel,
    )
    outputs = {"base": base_outputs}
    artifact_hashes: dict[str, object] = {}
    for name in _RESULT_NAMES:
        result = results[name]
        evidence = cast(Mapping[str, object], result["evidence"])
        artifact_hashes[name] = evidence.get("output_artifact")
        if name != "initialize":
            final = _result_outputs(result, name)
            _check_prompt_token_parity(base_outputs, final, name)
            outputs[name] = final

    revision = cast(str, cast(Mapping[str, object], initialization["provenance"])["model_revision"])
    records = (*local_real, *local_balanced, *local_synthetic)
    state_metrics: dict[str, object] = {}
    selected_by_state: dict[str, dict[str, dict[str, Fraction]]] = {}
    original_by_state: dict[str, dict[str, dict[str, Fraction]]] = {}
    for name in _BASE_STATES:
        state_metrics[name], selected_by_state[name], original_by_state[name] = _state_analysis(
            name,
            outputs[name],
            panel,
            records,
            revision if name == "base" else revision + "+adapter",
        )
    dataset_ids = tuple(selected_by_state["base"])
    selected = {
        dataset_id: {name: selected_by_state[name][dataset_id] for name in _BASE_STATES}
        for dataset_id in dataset_ids
    }
    original = {
        dataset_id: {name: original_by_state[name][dataset_id] for name in _BASE_STATES}
        for dataset_id in dataset_ids
    }
    historical_reference = _historical_reference_rates(reuse_metadata, local_real)
    gate_report, bootstrap = _gate_report(selected, original, historical_reference)
    delta = paired_group_delta(
        selected["snli-balanced-v1-development"]["synthetic_mix"],
        selected["snli-balanced-v1-development"]["real_only"],
    )
    transfer = sum(delta.values(), Fraction()) / len(delta)
    total_forward_count = sum(
        int(
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], results[name]["evidence"])["forward_counts"],
            )["total"]
        )
        for name in _RESULT_NAMES
    )
    if total_forward_count != core.MAX_TOTAL_FORWARDS:
        raise ValueError("passed mixture pair does not account for exactly 11,065 forwards")
    result_pins = {name: payloads[name]["pins"] for name in _RESULT_NAMES}
    return {
        "analysis_protocol": "reflex-mixture-training-analysis-v1",
        "status": "passed",
        "run_identity": {"experiment_id": envelope["experiment_id"]},
        "artifacts": {
            "mixture_receipt_canonical_json_sha256": _artifact_hash(envelope),
            "source_receipts": reuse_metadata["source_receipt_sha256"],
            "worker_output_artifacts": artifact_hashes,
            "data_file_sha256": cast(Mapping[str, object], pins)["data_file_sha256"],
            "protocol_sha256": cast(Mapping[str, object], pins)["protocol_sha256"],
            "source_file_sha256": cast(Mapping[str, object], pins)["source_file_sha256"],
        },
        "base_reuse": reuse_metadata,
        "states": state_metrics,
        "primary_transfer_contrast": {
            "status": "evaluated",
            "dataset_id": "snli-balanced-v1-development",
            "direction": "synthetic_mix - real_only",
            "group_mean_delta": _exact(transfer),
            "paired_bootstrap_95": cast(
                Mapping[str, object],
                cast(Mapping[str, object], bootstrap["intervals"])[
                    "balanced_snli_mix_minus_real_only"
                ],
            ),
        },
        "paired_bootstrap": bootstrap,
        "engineering_gate": gate_report,
        "provenance": {
            "payload_pins_identical": result_pins["initialize"]
            == result_pins["real_only"]
            == result_pins["synthetic_mix"],
            "base_and_arm_initialization_identical": all(
                cast(Mapping[str, object], results[name]["evidence"])["initialization"]
                == init_evidence["initialization"]
                for name in ("real_only", "synthetic_mix")
            ),
            "base_and_arm_model_tokenizer_identity_identical": True,
            "evaluation_presentations": len(panel),
            "total_forward_count": total_forward_count,
        },
        "data_disclosure": _DATA_DISCLOSURE,
    }
