"""Analyze saved Intern/Kev references alongside the original Qwen pilot, using CPU only."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments import analyze_real_pilot as pilot_analysis
from experiments import real_pilot_baseline_core as baseline_core
from experiments import real_pilot_core as pilot_core
from reflex_decisions.data import DecisionRecord, SplitManifest
from reflex_decisions.smoke import reserve_output, write_json_artifact

DEFAULT_RECORDS = pilot_analysis.DEFAULT_RECORDS
DEFAULT_MANIFEST = pilot_analysis.DEFAULT_MANIFEST
DEFAULT_RECIPE = pilot_analysis.DEFAULT_RECIPE
DEFAULT_OUTPUT = Path("artifacts/real-pilot-baselines-analysis-v1.json")
_MODEL_NAMES = ("intern", "kev")


def _normalized_object(value: object, label: str) -> dict[str, object]:
    try:
        normalized = json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError(f"{label} must be strict JSON") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return normalized


def _identity(value: Mapping[str, object], label: str) -> tuple[str, str]:
    run_id, nonce = value.get("run_id"), value.get("nonce")
    if not isinstance(run_id, str) or not isinstance(nonce, str):
        raise ValueError(f"{label} has no run ID or nonce")
    return run_id, nonce


def _build_qwen_payload(
    receipt: Mapping[str, object],
    manifest: SplitManifest,
    records: Sequence[DecisionRecord],
    hashes: Mapping[str, str],
) -> dict[str, object]:
    run_id, nonce = _identity(receipt, "Qwen receipt")
    return pilot_core.build_remote_payload(
        manifest,
        records,
        run_id=run_id,
        nonce=nonce,
        records_sha256=hashes["records_sha256"],
        manifest_sha256=hashes["manifest_sha256"],
        recipe_sha256=hashes["recipe_sha256"],
        protocol_sha256=pilot_core.verify_protocol(),
        source_file_sha256=pilot_core.source_fingerprints(),
    )


def _build_reference_payload(
    receipt: Mapping[str, object],
    manifest: SplitManifest,
    records: Sequence[DecisionRecord],
    hashes: Mapping[str, str],
) -> dict[str, object]:
    persisted_payload = receipt.get("payload")
    if not isinstance(persisted_payload, Mapping):
        raise ValueError("reference receipt has no saved evaluation payload")
    run_id, nonce = _identity(persisted_payload, "reference payload")
    return baseline_core.build_evaluation_payload(
        manifest,
        records,
        run_id=run_id,
        nonce=nonce,
        records_sha256=hashes["records_sha256"],
        manifest_sha256=hashes["manifest_sha256"],
        recipe_sha256=hashes["recipe_sha256"],
        protocol_sha256=pilot_core.verify_protocol(),
        reference_protocol_sha256=baseline_core.verify_reference_protocol(),
        source_file_sha256=baseline_core.source_fingerprints(),
    )


def _validate_reference_envelope(
    value: object, expected_payload: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    receipt = _normalized_object(value, "reference receipt")
    required = {
        "schema_version",
        "status",
        "payload",
        "limits",
        "provenance",
        "models",
    }
    if set(receipt) not in (required, required | {"failure"}):
        raise ValueError("reference receipt has an unexpected schema")
    persisted_payload = _normalized_object(receipt["payload"], "saved reference payload")
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != baseline_core.SCHEMA_VERSION
        or persisted_payload != _normalized_object(expected_payload, "expected reference payload")
        or receipt["limits"] != baseline_core.MODAL_LIMITS
    ):
        raise ValueError("reference receipt identity, pins, or limits differ from the local plan")
    provenance = receipt["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "modal_sdk_version",
        "profile",
        "workspace",
        "hf_hub_disable_implicit_token",
    }:
        raise ValueError("reference receipt runtime provenance has an unexpected schema")
    if (
        not isinstance(provenance["modal_sdk_version"], str)
        or not provenance["modal_sdk_version"].strip()
        or provenance["profile"] != "reflex-personal"
        or provenance["workspace"] != "rajath-61258"
        or provenance["hf_hub_disable_implicit_token"] is not True
    ):
        raise ValueError("reference receipt runtime provenance is invalid")
    models = receipt["models"]
    if not isinstance(models, dict) or set(models) != set(_MODEL_NAMES):
        raise ValueError("reference receipt must retain both keyed model states")

    validated: dict[str, dict[str, object]] = {}
    statuses: set[str] = set()
    for model_name in _MODEL_NAMES:
        state = _normalized_object(models[model_name], f"{model_name} model state")
        statuses.add(str(state.get("status")))
        if state.get("status") not in {"passed", "failed"}:
            raise ValueError(f"{model_name} reference state has an invalid status")
        validated[model_name] = baseline_core.validate_persisted_model_state(
            state, expected_payload=expected_payload, model_name=model_name
        )
    if "failure" in receipt:
        failure = receipt["failure"]
        if (
            not isinstance(failure, dict)
            or set(failure) != {"stage", "type", "message"}
            or any(not isinstance(failure[key], str) or not failure[key].strip() for key in failure)
        ):
            raise ValueError("reference run failure details are malformed")
        expected_status = "failed"
    elif statuses == {"passed"}:
        expected_status = "passed"
    elif statuses == {"failed"}:
        expected_status = "failed"
    else:
        expected_status = "partial"
    if receipt["status"] != expected_status:
        raise ValueError("reference receipt status does not match its model and run states")
    return receipt, validated


def _outputs_by_id(state: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rows = state.get("presentations")
    if not isinstance(rows, list):
        raise ValueError("reference state has no presentation rows")
    outputs: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("presentation_id"), str):
            raise ValueError("reference presentation row is malformed")
        presentation_id = row["presentation_id"]
        if presentation_id in outputs:
            raise ValueError("reference presentation rows contain a duplicate ID")
        outputs[presentation_id] = row
    return outputs


def _analyze_reference_state(
    state: Mapping[str, object],
    records: Sequence[DecisionRecord],
    manifest: SplitManifest,
    panel_by_id: Mapping[str, Mapping[str, object]],
    panel_by_key: Mapping[tuple[str, int], Mapping[str, object]],
    *,
    records_sha256: str,
    manifest_sha256: str,
) -> tuple[dict[str, object], dict[str, dict[str, bool]]]:
    outputs = _outputs_by_id(state)
    model_revision = str(state["model_revision"])
    calibration_fit = pilot_analysis._calibration_fit(records, outputs, panel_by_key)
    temperature = float(calibration_fit["temperature"])
    evaluation_records = tuple(
        record for record in records if record.dataset_id not in pilot_core.TRAIN_DATASET_IDS
    )
    predictions = pilot_analysis._prediction_rows(
        evaluation_records, outputs, panel_by_key, model_revision
    )
    records_by_dataset: dict[str, tuple[DecisionRecord, ...]] = {
        dataset_id: tuple(record for record in records if record.dataset_id == dataset_id)
        for dataset_id in pilot_analysis._DEVELOPMENT_DATASETS
    }
    metrics: dict[str, dict[str, dict[str, object]]] = {
        "temperature_1": {},
        "fitted": {},
    }
    correctness: dict[str, dict[str, bool]] = {}
    order_analysis: dict[str, dict[str, object]] = {}
    for dataset_id, dataset_records in records_by_dataset.items():
        record_ids = {record.record_id for record in dataset_records}
        dataset_predictions = tuple(
            prediction for prediction in predictions if prediction.record_id in record_ids
        )
        metrics["temperature_1"][dataset_id] = pilot_analysis._evaluate_dataset(
            dataset_records,
            dataset_predictions,
            manifest,
            records_sha256=records_sha256,
            manifest_sha256=manifest_sha256,
            temperature=1.0,
            calibration_id=None,
        )
        metrics["fitted"][dataset_id] = pilot_analysis._evaluate_dataset(
            dataset_records,
            dataset_predictions,
            manifest,
            records_sha256=records_sha256,
            manifest_sha256=manifest_sha256,
            temperature=temperature,
            calibration_id=pilot_analysis.CALIBRATION_ID,
        )
        order_result, correct_by_record = pilot_analysis._semantic_order_analysis(
            dataset_records,
            outputs,
            panel_by_id,
            model_revision,
        )
        order_analysis[dataset_id] = order_result
        correctness[dataset_id] = correct_by_record
    for metric_set in metrics.values():
        metric_set["dbpedia-sms-macro"] = pilot_analysis._macro_metrics(metric_set)
    return (
        {
            "model_id": state["model_id"],
            "model_revision": model_revision,
            "status": "passed",
            "forward_counts": state["forward_counts"],
            "prompt_hash_kind": state["prompt_hash_kind"],
            "provenance": state["provenance"],
            "calibration_fit": calibration_fit,
            "metrics": metrics,
            "order_analysis": order_analysis,
            "tie_policy": (
                "semantic option ID breaks score ties; tied results automatically abstain"
            ),
            "winner_temperature_invariance": (
                "positive temperature scaling preserves non-tied semantic winners"
            ),
        },
        correctness,
    )


def _qwen_correctness(
    receipt: Mapping[str, object],
    payload: Mapping[str, object],
    records: Sequence[DecisionRecord],
) -> dict[str, dict[str, bool]]:
    output_maps = pilot_analysis._output_maps(receipt)
    panel_by_id, _panel_by_key = pilot_analysis._panel_maps(dict(payload))
    provenance = receipt.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("model_revision"), str):
        raise ValueError("Qwen final receipt has no pinned base model revision")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Qwen final receipt has no saved adapter provenance")
    final_model_revision = (
        f"{provenance['model_revision']}+adapter-sha256:"
        f"{pilot_analysis._final_adapter_sha256(evidence)}"
    )
    final_rows = output_maps["final"]
    results: dict[str, dict[str, bool]] = {}
    records_by_dataset: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        if record.dataset_id in pilot_analysis._DEVELOPMENT_DATASETS:
            records_by_dataset[record.dataset_id].append(record)
    for dataset_id in pilot_analysis._DEVELOPMENT_DATASETS:
        _order_result, correct = pilot_analysis._semantic_order_analysis(
            records_by_dataset[dataset_id],
            final_rows,
            panel_by_id,
            final_model_revision,
        )
        results[dataset_id] = correct
    return results


def _paired_comparison(
    records_by_dataset: Mapping[str, Sequence[DecisionRecord]],
    qwen_correct: Mapping[str, Mapping[str, bool]],
    reference_correct: Mapping[str, Mapping[str, bool]],
) -> dict[str, object]:
    rng = random.Random(pilot_analysis.BOOTSTRAP_SEED)
    paired_results = {
        dataset_id: pilot_analysis._paired_dataset_comparison(
            records_by_dataset[dataset_id],
            qwen_correct[dataset_id],
            reference_correct[dataset_id],
            rng,
        )
        for dataset_id in pilot_analysis._DEVELOPMENT_DATASETS
    }
    macro = pilot_analysis._macro_bootstrap_interval(
        records_by_dataset, qwen_correct, reference_correct, rng
    )
    dbpedia_sms = pilot_analysis._DEVELOPMENT_DATASETS[:2]
    per_dataset = {
        dataset_id: {
            "source_group_count": paired_results[dataset_id]["source_group_count"],
            "qwen_final_accuracy": paired_results[dataset_id]["base_accuracy"],
            "reference_accuracy": paired_results[dataset_id]["final_accuracy"],
            "accuracy_delta_reference_minus_qwen_final": paired_results[dataset_id][
                "accuracy_delta_final_minus_base"
            ],
            "qwen_correct_reference_wrong": paired_results[dataset_id]["base_correct_final_wrong"],
            "qwen_wrong_reference_correct": paired_results[dataset_id]["base_wrong_final_correct"],
            "both_correct": paired_results[dataset_id]["both_correct"],
            "both_wrong": paired_results[dataset_id]["both_wrong"],
            "paired_group_bootstrap_95": paired_results[dataset_id]["paired_group_bootstrap_95"],
        }
        for dataset_id in pilot_analysis._DEVELOPMENT_DATASETS
    }
    return {
        "direction": "reference_minus_qwen_final",
        "per_dataset": per_dataset,
        "dbpedia_sms_macro": {
            "dataset_ids": list(dbpedia_sms),
            "qwen_final_accuracy": math.fsum(
                float(per_dataset[name]["qwen_final_accuracy"]) for name in dbpedia_sms
            )
            / 2,
            "reference_accuracy": math.fsum(
                float(per_dataset[name]["reference_accuracy"]) for name in dbpedia_sms
            )
            / 2,
            "accuracy_delta_reference_minus_qwen_final": macro["accuracy_delta_final_minus_base"],
            "paired_group_bootstrap_95": macro,
        },
        "bootstrap": {
            "replicates": pilot_analysis.BOOTSTRAP_REPLICATES,
            "seed": pilot_analysis.BOOTSTRAP_SEED,
            "sampling_unit": "source group",
            "permutations_are_independent_units": False,
        },
    }


def analyze_receipts(
    qwen_receipt_value: object,
    reference_receipt_value: object,
    records: Sequence[DecisionRecord],
    manifest: SplitManifest,
    *,
    hashes: Mapping[str, str],
    qwen_analysis_value: object | None = None,
) -> dict[str, object]:
    """Validate all run pins and analyze complete states without loading model runtimes."""

    qwen_receipt = _normalized_object(qwen_receipt_value, "Qwen receipt")
    qwen_payload = _build_qwen_payload(qwen_receipt, manifest, records, hashes)
    qwen_receipt = pilot_core.validate_passed_receipt(qwen_receipt, expected_payload=qwen_payload)
    reference_receipt_value = _normalized_object(reference_receipt_value, "reference receipt")
    reference_payload = _build_reference_payload(reference_receipt_value, manifest, records, hashes)
    reference_receipt, reference_states = _validate_reference_envelope(
        reference_receipt_value, reference_payload
    )
    qwen_panel = qwen_payload["evaluation_presentations"]
    reference_panel = reference_payload["evaluation_presentations"]
    if qwen_panel != reference_panel:
        raise ValueError("Qwen and reference receipts do not share the exact frozen panel")

    # All run identities, source maps, protocol pins, and exact panel membership are checked
    # before this point performs any local scoring or joins labels.
    qwen_analysis = pilot_analysis.analyze_receipt(
        qwen_receipt,
        qwen_payload,
        records,
        manifest,
        records_sha256=hashes["records_sha256"],
        manifest_sha256=hashes["manifest_sha256"],
    )
    if qwen_analysis_value is not None:
        saved_analysis = _normalized_object(qwen_analysis_value, "saved Qwen analysis")
        if saved_analysis != qwen_analysis:
            raise ValueError("saved Qwen analysis does not match its validated receipt")
        qwen_analysis = saved_analysis

    panel_by_id, panel_by_key = pilot_analysis._panel_maps(reference_payload)
    qwen_correct = _qwen_correctness(qwen_receipt, qwen_payload, records)
    records_by_dataset: dict[str, tuple[DecisionRecord, ...]] = {
        dataset_id: tuple(record for record in records if record.dataset_id == dataset_id)
        for dataset_id in pilot_analysis._DEVELOPMENT_DATASETS
    }
    qwen_run_id, qwen_nonce = _identity(qwen_receipt, "Qwen receipt")
    ref_run_id, ref_nonce = _identity(reference_payload, "reference payload")
    result_models: dict[str, dict[str, object]] = {
        "qwen_final": {
            "status": "passed",
            "run_id": qwen_run_id,
            "nonce": qwen_nonce,
            "model": qwen_analysis["model"],
            "state": qwen_analysis["states"]["final"],
        }
    }
    comparisons: dict[str, object] = {}
    for model_name in _MODEL_NAMES:
        state = reference_states[model_name]
        if state["status"] != "passed":
            result_models[model_name] = {
                "status": "failed",
                "model_id": state["model_id"],
                "model_revision": state["model_revision"],
                "forward_counts": state["forward_counts"],
                "prompt_hash_kind": state["prompt_hash_kind"],
                "provenance": state["provenance"],
                "failure": state["failure"],
                "partial_presentation_count": len(state["presentations"]),
                "partial_presentations": state["presentations"],
            }
            continue
        model_report, correct = _analyze_reference_state(
            state,
            records,
            manifest,
            panel_by_id,
            panel_by_key,
            records_sha256=hashes["records_sha256"],
            manifest_sha256=hashes["manifest_sha256"],
        )
        result_models[model_name] = {"status": "passed", "state": model_report}
        comparisons[model_name] = _paired_comparison(records_by_dataset, qwen_correct, correct)

    return {
        "analysis_protocol": "reflex-real-pilot-baselines-analysis-v1",
        "reference_run_identity": {
            "run_id": ref_run_id,
            "nonce": ref_nonce,
            "pins": reference_payload["pins"],
        },
        "qwen_run_identity": {
            "run_id": qwen_run_id,
            "nonce": qwen_nonce,
            "pins": qwen_receipt["pins"],
        },
        "models": result_models,
        "paired_comparisons": comparisons,
        "interpretation": (
            "development comparison on matched requests; references are compared with the "
            "saved Qwen final adapter, not ranked as an overall winner"
        ),
        "data_disclosure": "SNLI is development-only; labels and corpus text remain local",
        **(
            {"reference_run_failure": reference_receipt["failure"]}
            if "failure" in reference_receipt
            else {}
        ),
    }


def analyze_run(
    qwen_receipt_path: str | Path,
    reference_receipt_path: str | Path,
    records_path: str | Path = DEFAULT_RECORDS,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    recipe_path: str | Path = DEFAULT_RECIPE,
    qwen_analysis_path: str | Path | None = None,
) -> dict[str, object]:
    manifest, records, _recipe, hashes = pilot_analysis._final_data_recipe(
        records_path, manifest_path, recipe_path
    )
    _qwen_raw, qwen_value = pilot_analysis._read_json(qwen_receipt_path, "Qwen receipt")
    _reference_raw, reference_value = pilot_analysis._read_json(
        reference_receipt_path, "reference receipt"
    )
    qwen_analysis_value = None
    if qwen_analysis_path is not None:
        _analysis_raw, qwen_analysis_value = pilot_analysis._read_json(
            qwen_analysis_path, "saved Qwen analysis"
        )
    return analyze_receipts(
        qwen_value,
        reference_value,
        records,
        manifest,
        hashes=hashes,
        qwen_analysis_value=qwen_analysis_value,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-receipt", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--qwen-analysis", type=Path)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = analyze_run(
            args.qwen_receipt,
            args.reference_receipt,
            args.records,
            args.manifest,
            args.recipe,
            args.qwen_analysis,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with reserve_output(args.output) as reservation:
            write_json_artifact(reservation, report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved real-pilot baseline analysis to {args.output}")


if __name__ == "__main__":
    main()
