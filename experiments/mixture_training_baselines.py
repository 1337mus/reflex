"""Validate and remap frozen base-model outputs for mixture analysis."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any, cast

import experiments.analyze_real_pilot as real_analysis
import experiments.analyze_snli_diagnostic as balanced_analysis
import experiments.mixture_training_core as mixture_core
import experiments.real_pilot_core as real_pilot_core
import experiments.snli_diagnostic_core as snli_diagnostic_core
from experiments.mixture_training_contracts import (
    RUNTIME_VERSION_PINS,
    SOURCE_FINGERPRINT_PATHS,
    canonical_json,
    strict_json_loads,
)
from reflex_decisions import pilot_data, snli_diagnostic
from reflex_decisions.data import DecisionRecord

SAVED_REAL_RECEIPT = "artifacts/real-pilot-2026-10-04-r2-receipt.json"
SAVED_BALANCED_RECEIPT = "artifacts/snli-balanced-2026-10-04-r1-receipt.json"
SAVED_RECEIPT_SHA256 = {
    SAVED_REAL_RECEIPT: "c40d8889f58b507640d81aa3b2b5e2c98f3bb03a6459a51b64da5a07fff145d7",
    SAVED_BALANCED_RECEIPT: "76be5d7870a8780e5bebb59adf4fb834c908042b6686a21fdc045965ab68eed7",
}
REUSED_BASE_COUNTS = {"real_r2_base": 1932, "balanced_r1_base": 1152}
HISTORICAL_ACCURACY_ANCHORS = {
    "dbpedia14-pilot-v1-development": (50, 56),
    "sms-pilot-v1-development": (58, 60),
    "snli-pilot-v1-development": (57, 128),
}


def _historical_accuracy_count_payload() -> dict[str, dict[str, int]]:
    return {
        dataset_id: {"correct": correct, "total": total}
        for dataset_id, (correct, total) in HISTORICAL_ACCURACY_ANCHORS.items()
    }


_BASE_IDENTITY_FIELDS = (
    "model_id",
    "model_revision",
    "model_class",
    "config_class",
    "layer_count",
    "tied_embeddings",
    "no_meta_parameters",
    "load_diagnostics",
    "tokenizer_file_sha256",
    "effective_dtype",
    "attention_implementation",
    "use_kernels",
    "use_hub_kernels",
    "versions",
)
_STABLE_SEMANTIC_SOURCES = (
    "src/reflex_decisions/rendering.py",
    "src/reflex_decisions/schema.py",
    "src/reflex_decisions/scoring.py",
)


def _read_saved_receipt(root: Path, relative: str, supplied: object) -> dict[str, Any]:
    path = root / relative
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"saved base receipt is unavailable: {relative}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SAVED_RECEIPT_SHA256[relative]:
        raise ValueError(f"saved base receipt bytes differ from their frozen pin: {relative}")
    saved = strict_json_loads(raw)
    try:
        provided = strict_json_loads(canonical_json(supplied))
    except ValueError as exc:
        raise ValueError(f"{relative} input must be strict JSON") from exc
    if saved != provided or not isinstance(saved, dict):
        raise ValueError(
            f"supplied saved base receipt differs from its pinned artifact: {relative}"
        )
    return saved


def _validated_real_base(
    root: Path, receipt_value: object
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_saved_receipt(root, SAVED_REAL_RECEIPT, receipt_value)
    records_path = root / real_analysis.DEFAULT_RECORDS
    manifest_path = root / real_analysis.DEFAULT_MANIFEST
    recipe_path = root / real_analysis.DEFAULT_RECIPE
    manifest, records, _recipe, hashes = real_analysis._final_data_recipe(
        records_path, manifest_path, recipe_path
    )
    protocol_sha = real_pilot_core.verify_protocol(root / real_pilot_core.PROTOCOL_PATH)
    source_hashes = real_pilot_core.source_fingerprints(root)
    if source_hashes.get(real_pilot_core.PROTOCOL_PATH) != protocol_sha:
        raise ValueError("saved real-pilot protocol and source pins differ")
    expected_payload = real_pilot_core.build_remote_payload(
        manifest,
        records,
        run_id=receipt.get("run_id"),
        nonce=receipt.get("nonce"),
        records_sha256=hashes["records_sha256"],
        manifest_sha256=hashes["manifest_sha256"],
        recipe_sha256=hashes["recipe_sha256"],
        protocol_sha256=protocol_sha,
        source_file_sha256=source_hashes,
    )
    validated = real_pilot_core.validate_passed_receipt(receipt, expected_payload=expected_payload)
    return validated, {
        "records": records,
        "source_sha256": hashlib.sha256((root / SAVED_REAL_RECEIPT).read_bytes()).hexdigest(),
    }


def _validated_balanced_base(
    root: Path, receipt_value: object
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_saved_receipt(root, SAVED_BALANCED_RECEIPT, receipt_value)
    records_path = root / "data/processed/snli-balanced-v1/records.jsonl"
    manifest_path = root / "data/processed/snli-balanced-v1/manifest.json"
    recipe_path = root / "data/processed/snli-balanced-v1/recipe.json"
    try:
        records_bytes, manifest_bytes, recipe_bytes = (
            path.read_bytes() for path in (records_path, manifest_path, recipe_path)
        )
    except OSError as exc:
        raise ValueError("saved balanced-SNLI data artifacts are unavailable") from exc
    candidate = snli_diagnostic.build_candidate(root)
    records = candidate.bundle.records
    manifest = candidate.bundle.manifest
    payload = snli_diagnostic_core.validate_payload(receipt.get("payload"))
    balanced_analysis._validate_local_artifacts(
        root=root,
        receipt_payload=payload,
        records=records,
        manifest=manifest,
        records_bytes=records_bytes,
        manifest_bytes=manifest_bytes,
        recipe_bytes=recipe_bytes,
    )
    validated, rebuilt_payload = balanced_analysis._validated_receipt(receipt, records)
    if rebuilt_payload != payload:
        raise ValueError("saved balanced-SNLI receipt payload differs from local records")
    return validated, {
        "records": records,
        "source_sha256": hashlib.sha256((root / SAVED_BALANCED_RECEIPT).read_bytes()).hexdigest(),
    }


def _base_identity(value: Mapping[str, object], label: str) -> dict[str, object]:
    nested_key = "base_model" if "base_model" in value else "base"
    if nested_key in value:
        raw = value.get(nested_key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} has malformed nested base provenance")
        base = dict(raw)
    else:
        base = dict(value)
    missing = set(_BASE_IDENTITY_FIELDS) - set(base)
    if missing:
        raise ValueError(f"{label} is missing base identity fields: {', '.join(sorted(missing))}")
    identity = {key: base[key] for key in _BASE_IDENTITY_FIELDS}
    versions = identity["versions"]
    expected_base_versions = {
        name: version for name, version in RUNTIME_VERSION_PINS.items() if name != "Pillow"
    }
    if isinstance(versions, Mapping) and "Pillow" in versions:
        versions = {key: version for key, version in versions.items() if key != "Pillow"}
        identity["versions"] = versions
    if versions != expected_base_versions:
        raise ValueError(f"{label} base package versions differ from the eight frozen pins")
    if (
        identity["model_id"] != mixture_core.MODEL_ID
        or identity["model_revision"] != mixture_core.MODEL_REVISION
        or identity["model_class"] != "Qwen3_5ForCausalLM"
        or identity["config_class"] != "Qwen3_5TextConfig"
        or identity["layer_count"] != 24
        or identity["tied_embeddings"] is not True
        or identity["no_meta_parameters"] is not True
        or identity["load_diagnostics"]
        != {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        or identity["effective_dtype"] != "torch.bfloat16"
        or identity["attention_implementation"] != "eager"
        or identity["use_kernels"] is not False
        or identity["use_hub_kernels"] != "NO"
        or not isinstance(identity["tokenizer_file_sha256"], Mapping)
        or not identity["tokenizer_file_sha256"]
    ):
        raise ValueError(f"{label} base runtime identity differs from the pinned model setup")
    return identity


def _runtime_versions(value: Mapping[str, object], label: str) -> dict[str, str]:
    versions = value.get("versions", value.get("runtime_versions"))
    if versions != RUNTIME_VERSION_PINS:
        raise ValueError(f"{label} full runtime versions differ from the nine frozen pins")
    return cast(dict[str, str], dict(versions))


def _check_stable_sources(
    provenance: Mapping[str, object], source_hashes: Mapping[str, str], label: str
) -> None:
    for map_name in ("source_file_sha256", "measured_source_file_sha256"):
        source_map = provenance.get(map_name)
        if source_map is None:
            continue
        if not isinstance(source_map, Mapping):
            raise ValueError(f"{label} {map_name} is malformed")
        for path in _STABLE_SEMANTIC_SOURCES:
            if source_map.get(path) != source_hashes.get(path):
                raise ValueError(f"{label} changed pinned scoring source: {path}")


def _old_base_provenance(validated: Mapping[str, Any], kind: str) -> Mapping[str, object]:
    if kind == "real_r2":
        provenance = validated.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("saved real-pilot receipt has no provenance")
        # R2 predates the separate full-runtime field and its versions entry pins the
        # eight-package base model environment. `_base_identity` checks those pins.
        return provenance
    state = validated.get("models", {}).get("qwen_base")
    provenance = state.get("provenance") if isinstance(state, Mapping) else None
    if not isinstance(provenance, Mapping):
        raise ValueError("saved balanced-SNLI Qwen base state has no provenance")
    _runtime_versions(provenance, "saved balanced-SNLI Qwen base")
    return provenance


def _presentation_key(row: Mapping[str, object]) -> tuple[str, str, tuple[str, ...]]:
    record_id, request_hash, order_ids = (
        row.get("record_id"),
        row.get("request_hash"),
        row.get("order_ids"),
    )
    if (
        not isinstance(record_id, str)
        or not isinstance(request_hash, str)
        or not isinstance(order_ids, list)
        or not all(isinstance(option, str) for option in order_ids)
    ):
        raise ValueError("saved base output has malformed semantic identity")
    return record_id, request_hash, tuple(order_ids)


def _map_old_outputs(
    outputs: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object]],
    *,
    source_name: str,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    expected_by_key: dict[tuple[str, str, tuple[str, ...]], Mapping[str, object]] = {}
    for row in expected_rows:
        key = _presentation_key(row)
        if key in expected_by_key:
            raise ValueError("local mixture panel contains duplicate semantic presentation IDs")
        expected_by_key[key] = row
    mapped: dict[str, dict[str, object]] = {}
    lineage: dict[str, dict[str, object]] = {}
    source_ids: set[str] = set()
    for output in outputs:
        key = _presentation_key(output)
        expected = expected_by_key.get(key)
        source_id = output.get("presentation_id")
        if expected is None or not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{source_name} output does not map to the approved mixture panel")
        if source_id in source_ids:
            raise ValueError(f"{source_name} base outputs contain duplicate source IDs")
        source_ids.add(source_id)
        for field in ("dataset_id", "order_index", "source_group_id"):
            if field in output and output[field] != expected.get(field):
                raise ValueError(f"{source_name} base output has mismatched {field}")
        current_id = expected.get("presentation_id")
        if not isinstance(current_id, str) or current_id in mapped:
            raise ValueError("local mixture panel has duplicate or malformed presentation IDs")
        required = ("candidate_logits", "winner_option_id", "input_tokens", "prompt_sha256")
        if any(field not in output for field in required):
            raise ValueError(f"{source_name} base output is missing scored fields")
        mapped[current_id] = {
            **{
                field: expected[field]
                for field in (
                    "presentation_id",
                    "record_id",
                    "dataset_id",
                    "source_group_id",
                    "request_hash",
                    "order_index",
                    "order_ids",
                )
            },
            **{field: output[field] for field in required},
        }
        lineage[current_id] = {
            "source_presentation_id": source_id,
            "source": source_name,
        }
    if len(mapped) != len(outputs):
        raise ValueError(f"{source_name} output mapping lost source rows")
    return mapped, lineage


def _select_real_reuse_subset(
    outputs: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Select the R2 rows retained by the mixture panel after validating the full R2 receipt."""

    expected_by_key = {_presentation_key(row): row for row in expected_rows}
    if len(expected_by_key) != len(expected_rows):
        raise ValueError("local mixture panel contains duplicate semantic presentation IDs")
    if len(outputs) != real_pilot_core.EVALUATION_PRESENTATION_COUNT:
        raise ValueError("validated R2 output count differs from its complete 2,572-row panel")

    selected: list[Mapping[str, object]] = []
    seen_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    seen_ids: set[str] = set()
    original_by_record = {
        cast(str, row["record_id"]): row
        for row in expected_rows
        if row.get("dataset_id") == pilot_data.SNLI_DATASET_ID
    }
    dropped_orders: dict[str, set[tuple[str, ...]]] = {}
    for output in outputs:
        key = _presentation_key(output)
        source_id = output.get("presentation_id")
        if not isinstance(source_id, str) or not source_id or source_id in seen_ids:
            raise ValueError("validated R2 outputs contain duplicate or malformed source IDs")
        if key in seen_keys:
            raise ValueError("validated R2 outputs contain duplicate semantic rows")
        seen_ids.add(source_id)
        seen_keys.add(key)
        if key in expected_by_key:
            selected.append(output)
            continue
        dataset_id = output.get("dataset_id")
        record_id = output.get("record_id")
        original = original_by_record.get(record_id) if isinstance(record_id, str) else None
        original_order = _presentation_key(original)[2] if original is not None else ()
        order_ids = key[2]
        allowed_orders = set(permutations(original_order)) - {original_order}
        if (
            dataset_id != pilot_data.SNLI_DATASET_ID
            or original is None
            or output.get("request_hash") != original.get("request_hash")
            or order_ids not in allowed_orders
        ):
            raise ValueError("R2 rows outside the mixture panel are not approved SNLI orders")
        dropped_orders.setdefault(record_id, set()).add(order_ids)

    expected_dropped = real_pilot_core.EVALUATION_PRESENTATION_COUNT - len(expected_rows)
    if (
        len(selected) != len(expected_rows)
        or {key for key in seen_keys if key in expected_by_key} != set(expected_by_key)
        or len(outputs) - len(selected) != expected_dropped
        or len(dropped_orders) != real_pilot_core.DATASET_RECORD_COUNTS[pilot_data.SNLI_DATASET_ID]
        or any(
            orders
            != set(permutations(_presentation_key(original_by_record[record_id])[2]))
            - {_presentation_key(original_by_record[record_id])[2]}
            for record_id, orders in dropped_orders.items()
        )
    ):
        raise ValueError("R2 reuse selection did not retain the exact mixture-compatible rows")
    return selected


def _group_accuracy_counts(
    outputs: Mapping[str, Mapping[str, object]],
    records: Sequence[DecisionRecord],
    dataset_ids: Sequence[str],
) -> dict[str, dict[str, dict[str, int]]]:
    record_by_id = {record.record_id: record for record in records}
    counts: dict[str, dict[str, dict[str, int]]] = {dataset_id: {} for dataset_id in dataset_ids}
    for output in outputs.values():
        dataset_id = output.get("dataset_id")
        if dataset_id not in counts or output.get("order_index") != 0:
            continue
        record_id = output.get("record_id")
        record = record_by_id.get(record_id) if isinstance(record_id, str) else None
        group_id = output.get("source_group_id")
        winner = output.get("winner_option_id")
        if record is None or not isinstance(group_id, str) or not isinstance(winner, str):
            raise ValueError("historical reference output has malformed original-order identity")
        group = counts[dataset_id].setdefault(group_id, {"correct": 0, "total": 0})
        group["correct"] += winner == record.answer_id
        group["total"] += 1
    return counts


def _aggregate_group_counts(
    grouped: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict[str, dict[str, int]]:
    return {
        dataset_id: {
            "correct": sum(group["correct"] for group in groups.values()),
            "total": sum(group["total"] for group in groups.values()),
        }
        for dataset_id, groups in grouped.items()
    }


def validate_reused_base_outputs(
    *,
    root: str | Path,
    panel: Sequence[Mapping[str, object]],
    real_records: Sequence[DecisionRecord],
    balanced_records: Sequence[DecisionRecord],
    new_base_provenance: Mapping[str, object],
    source_hashes: Mapping[str, str],
    real_receipt: object,
    balanced_receipt: object,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Validate frozen prior receipts and map their base logits onto the new panel."""

    project_root = Path(root).resolve(strict=True)
    current_source_hashes = dict(source_hashes)
    if set(current_source_hashes) != set(SOURCE_FINGERPRINT_PATHS):
        raise ValueError("current source hashes do not match the mixture allowlist")
    for relative in _STABLE_SEMANTIC_SOURCES:
        actual = hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
        if current_source_hashes.get(relative) != actual:
            raise ValueError(f"local scoring source hash mismatch: {relative}")

    real_state, real_meta = _validated_real_base(project_root, real_receipt)
    balanced_state, balanced_meta = _validated_balanced_base(project_root, balanced_receipt)
    if tuple(real_meta["records"]) != tuple(
        row for row in real_records if row.dataset_id in real_pilot_core.DATASET_RECORD_COUNTS
    ):
        by_id = {row.record_id: row for row in real_records}
        if any(by_id.get(row.record_id) != row for row in real_meta["records"]):
            raise ValueError("saved real-pilot source records differ from current local records")
    if tuple(balanced_meta["records"]) != tuple(balanced_records):
        by_id = {row.record_id: row for row in balanced_records}
        if any(by_id.get(row.record_id) != row for row in balanced_meta["records"]):
            raise ValueError("saved balanced-SNLI source records differ from current local records")

    new_identity = _base_identity(new_base_provenance, "mixture initialization")
    old_real_provenance = _old_base_provenance(real_state, "real_r2")
    balanced_model = cast(dict[str, Any], balanced_state["models"]["qwen_base"])
    old_balanced_provenance = _old_base_provenance(balanced_state, "balanced")
    identities = (
        _base_identity(old_real_provenance, "saved real-pilot base"),
        _base_identity(old_balanced_provenance, "saved balanced-SNLI base"),
    )
    if any(identity != new_identity for identity in identities):
        raise ValueError(
            "reused base model/tokenizer/runtime identity differs from new initialization"
        )
    for label, provenance in (
        ("saved real-pilot base", old_real_provenance),
        ("saved balanced-SNLI base", old_balanced_provenance),
    ):
        _check_stable_sources(provenance, current_source_hashes, label)
    _runtime_versions(new_base_provenance, "mixture initialization")

    panel_by_dataset: dict[str, list[Mapping[str, object]]] = {}
    for row in panel:
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str):
            raise ValueError("mixture panel has a malformed dataset ID")
        panel_by_dataset.setdefault(dataset_id, []).append(row)
    real_ids = set(real_pilot_core.DATASET_RECORD_COUNTS)
    real_rows = [row for row in panel if row.get("dataset_id") in real_ids]
    balanced_rows = panel_by_dataset.get(snli_diagnostic.DATASET_ID, [])
    if len(real_rows) != REUSED_BASE_COUNTS["real_r2_base"]:
        raise ValueError("mixture panel real rows do not match the saved 1,932-output source")
    if len(balanced_rows) != REUSED_BASE_COUNTS["balanced_r1_base"]:
        raise ValueError(
            "mixture panel balanced-SNLI rows do not match the saved 1,152-output source"
        )

    real_outputs = real_state.get("evidence", {}).get("base_outputs")
    balanced_outputs = balanced_model.get("presentations")
    if not isinstance(real_outputs, list) or not isinstance(balanced_outputs, list):
        raise ValueError("validated saved base receipt has no complete output rows")
    if len(balanced_outputs) != len(balanced_rows):
        raise ValueError("saved balanced-SNLI base output count differs from its reuse panel")
    selected_real_outputs = _select_real_reuse_subset(
        cast(list[Mapping[str, object]], real_outputs), real_rows
    )
    real_mapped, real_lineage = _map_old_outputs(
        selected_real_outputs, real_rows, source_name="real_r2_base"
    )
    real_final_outputs = real_state.get("evidence", {}).get("final_outputs")
    if not isinstance(real_final_outputs, list):
        raise ValueError("validated saved R2 receipt has no complete final-adapter outputs")
    selected_real_final_outputs = _select_real_reuse_subset(
        cast(list[Mapping[str, object]], real_final_outputs), real_rows
    )
    real_final_mapped, real_final_lineage = _map_old_outputs(
        selected_real_final_outputs, real_rows, source_name="real_r2_final"
    )
    if set(real_final_lineage) != set(real_lineage) or any(
        real_final_lineage[presentation_id]["source_presentation_id"]
        != lineage_row["source_presentation_id"]
        for presentation_id, lineage_row in real_lineage.items()
    ):
        raise ValueError("R2 base and final outputs do not share the same presentation lineage")
    balanced_mapped, balanced_lineage = _map_old_outputs(
        cast(list[Mapping[str, object]], balanced_outputs),
        balanced_rows,
        source_name="balanced_r1_base",
    )
    overlap = set(real_mapped) & set(balanced_mapped)
    if overlap:
        raise ValueError("saved base sources overlap on mixture presentation IDs")
    mapped = {**real_mapped, **balanced_mapped}
    lineage = {**real_lineage, **balanced_lineage}
    final_groups = _group_accuracy_counts(
        real_final_mapped,
        real_records,
        ("dbpedia14-pilot-v1-development", "sms-pilot-v1-development"),
    )
    base_snli_groups = _group_accuracy_counts(
        real_mapped, real_records, ("snli-pilot-v1-development",)
    )
    historical_groups = {**final_groups, **base_snli_groups}
    historical_counts = _aggregate_group_counts(historical_groups)
    if historical_counts != _historical_accuracy_count_payload():
        raise ValueError("validated R2 historical outputs differ from the frozen accuracy anchors")
    return mapped, {
        "source_receipt_sha256": {
            SAVED_REAL_RECEIPT: real_meta["source_sha256"],
            SAVED_BALANCED_RECEIPT: balanced_meta["source_sha256"],
        },
        "reused_rows": len(mapped),
        "by_source": {
            "real_r2_base": len(real_mapped),
            "balanced_r1_base": len(balanced_mapped),
        },
        "presentation_id_lineage": lineage,
        "source_presentation_ids_preserved_in_lineage": len(lineage),
        "historical_reference_group_counts": historical_groups,
        "historical_reference_accuracy_counts": historical_counts,
    }
