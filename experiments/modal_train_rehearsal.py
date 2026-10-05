"""Plan or explicitly launch one bounded synthetic Qwen LoRA rehearsal on Modal."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from experiments import modal_smoke
from experiments import training_rehearsal_core as core
from reflex_decisions import smoke
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import CompiledRequest, compile_request
from reflex_decisions.schema import DecisionRequest

VOLUME_NAME = "reflex-rehearsal-artifacts"
VOLUME_ROOT = Path("/artifacts")
PROFILE = "reflex-personal"
WORKSPACE = "rajath-61258"
TOKENIZER_FILE_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
)
EXPECTED_PACKAGES = {
    "torch": "2.14.1",
    "torchvision": "0.29.1",
    "transformers": "5.18.0",
    "peft": "0.21.0",
}


def _normalize_receipt(receipt: dict[str, object]) -> dict[str, object]:
    """Return strict built-in JSON while retaining a safe failure if serialization broke."""

    try:
        return json.loads(json.dumps(receipt, allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError):
        safe_receipt: dict[str, object] = {}
        for key, value in receipt.items():
            if key in {"status", "failure"}:
                continue
            try:
                safe_receipt[key] = json.loads(json.dumps(value, allow_nan=False))
            except (TypeError, ValueError, OverflowError, RecursionError):
                continue
        safe_receipt.setdefault("schema_version", 1)
        safe_receipt.setdefault("run_id", "unknown")
        safe_receipt.setdefault("provenance", {})
        safe_receipt.setdefault("evidence", {})
        safe_receipt["status"] = "failed"
        safe_receipt["failure"] = {
            "stage": "receipt_serialization",
            "type": "NonJsonReceipt",
            "message": "rehearsal receipt contained non-JSON-safe metadata",
        }
        return json.loads(json.dumps(safe_receipt, allow_nan=False))


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": smoke.sanitize_exception_message(error),
    }


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_fingerprints() -> dict[str, str]:
    return core.source_fingerprints()


def _canonical_source_key(path: str | Path) -> str:
    resolved = Path(path).resolve()
    for remote_root, canonical_root in (
        (Path("/root/experiments"), "experiments"),
        (Path("/root/reflex_decisions"), "src/reflex_decisions"),
    ):
        try:
            relative = resolved.relative_to(remote_root)
        except ValueError:
            continue
        if relative.suffix != ".py":
            break
        return f"{canonical_root}/{relative.as_posix()}"
    raise ValueError("remote source module resolved outside the packaged training sources")


def _measure_remote_source_fingerprints() -> dict[str, str]:
    measured: dict[str, str] = {}
    for module_name, expected_key in core.SOURCE_FINGERPRINT_MODULES:
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str):
            raise ValueError(f"remote source module has no file: {module_name}")
        key = _canonical_source_key(module_path)
        if key != expected_key or key in measured:
            raise ValueError(
                f"remote source module path did not match its canonical key: {module_name}"
            )
        measured[key] = _sha256_file(module_path)
    return measured


def _verify_remote_source_fingerprints(source_hashes: object) -> dict[str, str]:
    expected_keys = set(core.SOURCE_FINGERPRINT_PATHS)
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != expected_keys
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            for name, digest in source_hashes.items()
        )
    ):
        raise ValueError("source fingerprint payload is malformed")
    measured = _measure_remote_source_fingerprints()
    if set(measured) != expected_keys or measured != source_hashes:
        raise ValueError("remote source fingerprints did not match the caller-provided map")
    return measured


def _initial_receipt(
    run_id: str,
    *,
    attempt_id: str | None = None,
    profile: str | None = None,
    workspace: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "failed",
        "run_id": run_id,
        "attempt_id": attempt_id or str(uuid.uuid4()),
        "purpose": "synthetic training mechanics and fixture memorization only",
        "phase": "preflight",
        "failure": {
            "stage": "not_started",
            "type": "NotStarted",
            "message": "training rehearsal did not return a completion receipt",
        },
        "provenance": {
            "model_id": core.MODEL_ID,
            "model_revision": core.MODEL_REVISION,
            "seed": core.SEED,
            "records_sha256": core.EXPECTED_RECORDS_SHA256,
            "manifest_sha256": core.EXPECTED_MANIFEST_SHA256,
            "protocol_sha256": core.EXPECTED_PROTOCOL_SHA256,
            "profile": profile,
            "workspace": workspace,
        },
        "evidence": {
            "optimizer_updates_completed": 0,
            "forward_count": 0,
            "diagnostics": [],
            "adapter_paths": [],
        },
        "limits": core.plan()["modal"],
    }


def _validate_remote_receipt(
    receipt: object,
    expected: dict[str, object],
    *,
    require_full_provenance: bool = False,
) -> dict[str, object]:
    """Validate remote identity and provenance before adopting its JSON evidence."""

    try:
        normalized = json.loads(json.dumps(receipt, allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Modal returned a non-JSON training receipt") from exc
    if not isinstance(normalized, dict):
        raise ValueError("Modal returned a non-object training receipt")
    if (
        type(normalized.get("schema_version")) is not int
        or normalized["schema_version"] != 1
        or normalized.get("run_id") != expected.get("run_id")
        or normalized.get("attempt_id") != expected.get("attempt_id")
        or normalized.get("status") not in {"passed", "failed"}
    ):
        raise ValueError("Modal returned a receipt with mismatched identity or schema")

    expected_provenance = {
        "model_id": core.MODEL_ID,
        "model_revision": core.MODEL_REVISION,
        "seed": core.SEED,
        "records_sha256": core.EXPECTED_RECORDS_SHA256,
        "manifest_sha256": core.EXPECTED_MANIFEST_SHA256,
        "protocol_sha256": core.EXPECTED_PROTOCOL_SHA256,
    }
    caller_hashes = expected.get("source_file_sha256")
    if not isinstance(caller_hashes, dict) or not caller_hashes:
        raise ValueError("local source fingerprint map is missing")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for name, digest in caller_hashes.items()
    ):
        raise ValueError("local source fingerprint map is malformed")
    expected_provenance["source_file_sha256"] = caller_hashes

    provenance = normalized.get("provenance")
    if not isinstance(provenance, dict):
        if normalized["status"] == "failed" and not require_full_provenance:
            provenance = {}
        else:
            raise ValueError("Modal receipt provenance is missing")
    if normalized["status"] == "passed" or require_full_provenance:
        for name, value in expected_provenance.items():
            if provenance.get(name) != value:
                raise ValueError(f"Modal receipt provenance mismatch for {name}")
        if provenance.get("measured_source_file_sha256") != caller_hashes:
            raise ValueError("remote source attestation did not match the local source map")
    if normalized["status"] == "passed":
        _validate_passed_evidence(normalized, str(expected["run_id"]))
    else:
        for name, value in expected_provenance.items():
            if name in provenance and provenance[name] != value:
                raise ValueError(f"Modal failure receipt provenance mismatch for {name}")
        if (
            "measured_source_file_sha256" in provenance
            and provenance["measured_source_file_sha256"] != caller_hashes
        ):
            raise ValueError("Modal failure receipt source attestation did not match")
    return normalized


def _validate_passed_evidence(receipt: dict[str, object], run_id: str) -> None:
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("passed receipt evidence is missing")
    updates = evidence.get("optimizer_updates_completed")
    if type(updates) is not int or not core.DIAGNOSTIC_UPDATES[0] <= updates <= core.MAX_UPDATES:
        raise ValueError("passed receipt optimizer update count is invalid")
    diagnostics = evidence.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise ValueError("passed receipt has no accuracy diagnostic")
    final_diagnostic = diagnostics[-1]
    if not isinstance(final_diagnostic, dict) or final_diagnostic.get("update") != updates:
        raise ValueError("passed receipt final diagnostic does not match its update count")
    accuracy = final_diagnostic.get("accuracy")
    if (
        not isinstance(accuracy, (int, float))
        or isinstance(accuracy, bool)
        or not math.isfinite(float(accuracy))
        or float(accuracy) < core.EARLY_STOP_ACCURACY
        or float(accuracy) > 1.0
        or evidence.get("memorization_threshold_met") is not True
    ):
        raise ValueError("passed receipt does not prove the accuracy threshold")
    diagnostic_updates = [row.get("update") for row in diagnostics if isinstance(row, dict)]
    if diagnostic_updates != list(core.DIAGNOSTIC_UPDATES[: len(diagnostics)]):
        raise ValueError("passed receipt diagnostics do not match the pinned schedule")

    gradient = evidence.get("lo_ra_b_gradient_l1_total")
    if (
        not isinstance(gradient, (int, float))
        or isinstance(gradient, bool)
        or not math.isfinite(float(gradient))
        or float(gradient) <= 0.0
    ):
        raise ValueError("passed receipt does not prove a finite nonzero LoRA-B gradient")
    adapter_update = evidence.get("adapter_update")
    if not isinstance(adapter_update, dict):
        raise ValueError("passed receipt does not prove adapter tensor changes")
    changed_count = adapter_update.get("changed_tensor_count")
    changed_names = adapter_update.get("changed_tensor_names")
    if (
        type(changed_count) is not int
        or changed_count <= 0
        or not isinstance(changed_names, list)
        or len(changed_names) != changed_count
        or any(not isinstance(name, str) or not name for name in changed_names)
        or len(set(changed_names)) != changed_count
    ):
        raise ValueError("passed receipt does not prove adapter tensor changes")

    forward_count = evidence.get("forward_count")
    train_forward_count = evidence.get("train_forward_count")
    expected_train_forwards = updates * core.MICROBATCHES_PER_UPDATE
    expected_total_forwards = expected_train_forwards + core.DIAGNOSTIC_PRESENTATIONS * (
        2 + len(diagnostics)
    )
    if (
        type(forward_count) is not int
        or forward_count != expected_total_forwards
        or forward_count > core.MAX_FORWARD_COUNT
        or train_forward_count != expected_train_forwards
        or evidence.get("max_forward_count") != core.MAX_FORWARD_COUNT
    ):
        raise ValueError("passed receipt violates the forward budget")
    reload_parity = evidence.get("reload_parity")
    if not isinstance(reload_parity, dict):
        raise ValueError("passed receipt reload parity evidence is missing")
    difference = reload_parity.get("max_candidate_logit_difference")
    if (
        reload_parity.get("presentation_count") != core.DIAGNOSTIC_PRESENTATIONS
        or not isinstance(difference, (int, float))
        or isinstance(difference, bool)
        or not math.isfinite(float(difference))
        or float(difference) < 0.0
        or float(difference) > 1e-3
        or reload_parity.get("winner_mismatches") != 0
        or reload_parity.get("adapter_tensor_keys_shapes_values_match") is not True
        or reload_parity.get("adapter_tensors_unmerged_fp32") is not True
        or reload_parity.get("model_eval_mode") is not True
        or reload_parity.get("use_cache") is not False
    ):
        raise ValueError("passed receipt does not prove fresh reload parity")

    adapter_paths = evidence.get("adapter_paths")
    if not isinstance(adapter_paths, list) or not adapter_paths:
        raise ValueError("passed receipt has no adapter snapshots")
    if [snapshot.get("update") for snapshot in adapter_paths if isinstance(snapshot, dict)] != (
        diagnostic_updates
    ):
        raise ValueError("passed receipt adapter snapshots do not match its diagnostics")
    run_root = PurePosixPath("/artifacts") / "runs" / run_id
    for snapshot in adapter_paths:
        if not isinstance(snapshot, dict):
            raise ValueError("passed receipt adapter snapshot is malformed")
        update = snapshot.get("update")
        path = snapshot.get("path")
        hashes = snapshot.get("files_sha256")
        expected_path = run_root / f"adapter-update-{update:03d}" if type(update) is int else None
        if (
            update not in core.DIAGNOSTIC_UPDATES
            or path != str(expected_path)
            or not isinstance(hashes, dict)
            or not {"adapter_model.safetensors", "adapter_config.json"}.issubset(hashes)
            or any(
                not isinstance(name, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                for name, digest in hashes.items()
            )
        ):
            raise ValueError("passed receipt adapter path or hashes are invalid")
    if adapter_paths[-1].get("update") != updates:
        raise ValueError("passed receipt final adapter snapshot does not match its update count")


def _strict_json_object(raw: bytes) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("progress receipt contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"progress receipt contains invalid JSON constant: {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("progress receipt is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("progress receipt is not a JSON object")
    return parsed


def _read_volume_progress(volume: Any, run_id: str) -> dict[str, object]:
    """Read only this attempt's JSON progress receipt from the Modal Volume."""

    path = f"runs/{core.validate_run_id(run_id)}/progress.json"
    chunks = volume.read_file(path)
    if isinstance(chunks, bytes):
        raise ValueError("Modal Volume progress read did not return a byte stream")
    try:
        pieces = list(chunks)
    except TypeError as exc:
        raise ValueError("Modal Volume progress read did not return an iterable stream") from exc
    if any(not isinstance(chunk, bytes) for chunk in pieces):
        raise ValueError("Modal Volume progress stream contained a non-byte chunk")
    raw = b"".join(pieces)
    return _strict_json_object(raw)


def _write_failure(
    reservation: smoke.ArtifactReservation,
    result: dict[str, object],
    stage: str,
    error: BaseException,
) -> int:
    result["failure"] = _failure(stage, error)
    smoke.write_json_artifact(reservation, _normalize_receipt(result))
    print(json.dumps({"status": "failed", "artifact": str(reservation.destination)}))
    return 1


def _write_json(path: Path, result: dict[str, object], *, replace: bool) -> str:
    """Write strict JSON atomically and return the stored SHA-256."""

    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        return _sha256_file(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_progress(run_dir: Path, receipt: dict[str, object]) -> None:
    path = run_dir / "progress.json"
    _write_json(path, _normalize_receipt(receipt), replace=True)


def _commit_volume() -> None:
    import modal

    modal.Volume.from_name(VOLUME_NAME).commit()


def _compiled_candidate_logits(
    model: Any,
    request: DecisionRequest,
    tokenizer: Any,
    torch: Any,
    forward_counter: dict[str, int],
) -> tuple[Any, CompiledRequest]:
    compiled = compile_request(request, tokenizer, max_tokens=smoke.MAX_INPUT_TOKENS)
    input_ids = torch.tensor([compiled.input_ids], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
    )
    forward_counter["count"] += 1
    candidate_ids = torch.tensor(compiled.candidate_token_ids, dtype=torch.long, device="cuda")
    candidate_logits = output.logits[0, -1].index_select(0, candidate_ids).float()
    if candidate_logits.numel() != len(request.options):
        raise RuntimeError("candidate-only output count did not match the request")
    if not bool(torch.isfinite(candidate_logits).all().item()):
        raise RuntimeError("candidate logits contained a non-finite value")
    return candidate_logits, compiled


def _diagnostic_presentations(
    records: tuple[DecisionRecord, ...],
) -> tuple[tuple[DecisionRecord, str, DecisionRequest], ...]:
    presentations: list[tuple[DecisionRecord, str, DecisionRequest]] = []
    for record in records:
        presentations.append((record, "original", record.request))
        reversed_request = record.request.model_copy(
            update={"options": tuple(reversed(record.request.options))}
        )
        presentations.append((record, "reversed", reversed_request))
    return tuple(presentations)


def _evaluate(
    model: Any,
    tokenizer: Any,
    records: tuple[DecisionRecord, ...],
    torch: Any,
    forward_counter: dict[str, int],
) -> dict[str, object]:
    import torch.nn.functional as functional

    results: list[dict[str, object]] = []
    total_loss = 0.0
    for record, order, request in _diagnostic_presentations(records):
        option_ids = [option.id for option in request.options]
        gold_index = option_ids.index(record.answer_id)
        with torch.inference_mode():
            logits, compiled = _compiled_candidate_logits(
                model, request, tokenizer, torch, forward_counter
            )
            loss = functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([gold_index], dtype=torch.long, device="cuda")
            )
        candidate_logits = [float(value) for value in logits.detach().cpu().tolist()]
        predicted_id = option_ids[int(torch.argmax(logits).item())]
        total_loss += float(loss.item())
        results.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "order": order,
                "gold_option_id": record.answer_id,
                "predicted_option_id": predicted_id,
                "correct": predicted_id == record.answer_id,
                "candidate_logits": candidate_logits,
                "prompt_sha256": compiled.prompt_hash,
            }
        )

    correct_count = sum(row["correct"] is True for row in results)
    by_task: dict[str, dict[str, int | float]] = {}
    by_order: dict[str, dict[str, int | float]] = {}
    for row in results:
        task = str(row["dataset_id"])
        order = str(row["order"])
        bucket = by_task.setdefault(task, {"correct": 0, "count": 0})
        bucket["correct"] = int(bucket["correct"]) + int(row["correct"] is True)
        bucket["count"] = int(bucket["count"]) + 1
        bucket = by_order.setdefault(order, {"correct": 0, "count": 0})
        bucket["correct"] = int(bucket["correct"]) + int(row["correct"] is True)
        bucket["count"] = int(bucket["count"]) + 1
    for buckets in (by_task, by_order):
        for bucket in buckets.values():
            count = int(bucket["count"])
            bucket["accuracy"] = int(bucket["correct"]) / count if count else 0.0
    return {
        "presentation_count": len(results),
        "loss": total_loss / len(results),
        "accuracy": correct_count / len(results),
        "correct": correct_count,
        "by_task": by_task,
        "by_order": by_order,
        "presentations": results,
    }


def _compare_to_baseline(
    baseline: dict[str, object], current: dict[str, object]
) -> dict[str, int | float]:
    baseline_rows = baseline["presentations"]
    current_rows = current["presentations"]
    assert isinstance(baseline_rows, list) and isinstance(current_rows, list)
    if len(baseline_rows) != len(current_rows):
        raise RuntimeError("diagnostic presentation count changed during training")
    changed_predictions = 0
    max_logit_change = 0.0
    for before, after in zip(baseline_rows, current_rows, strict=True):
        if (before["record_id"], before["order"]) != (after["record_id"], after["order"]):
            raise RuntimeError("diagnostic presentation order changed during training")
        changed_predictions += before["predicted_option_id"] != after["predicted_option_id"]
        before_logits = before["candidate_logits"]
        after_logits = after["candidate_logits"]
        if len(before_logits) != len(after_logits):
            raise RuntimeError("candidate count changed during training")
        max_logit_change = max(
            max_logit_change,
            *(abs(float(a) - float(b)) for a, b in zip(before_logits, after_logits, strict=True)),
        )
    return {
        "changed_predictions": int(changed_predictions),
        "max_candidate_logit_change": max_logit_change,
    }


def _adapter_modules_and_parameters(
    model: Any,
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    modules: list[tuple[str, Any]] = []
    parameters: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is not None or lora_b is not None:
            if (
                lora_a is None
                or lora_b is None
                or "default" not in lora_a
                or "default" not in lora_b
            ):
                raise RuntimeError(f"incomplete LoRA module inventory at {name}")
            modules.append((name, module))
    for name, parameter in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            parameters.append((name, parameter))
    return modules, parameters


def _validate_adapter_inventory(
    model: Any,
    torch: Any,
    *,
    require_trainable: bool,
) -> dict[str, object]:
    modules, adapter_parameters = _adapter_modules_and_parameters(model)
    if len(modules) != core.EXPECTED_LORA_MODULES:
        raise RuntimeError(
            f"LoRA target module count was {len(modules)}, expected {core.EXPECTED_LORA_MODULES}"
        )
    if len(adapter_parameters) != core.EXPECTED_ADAPTER_TENSORS:
        raise RuntimeError(
            "LoRA adapter tensor count did not match the pinned 120-tensor inventory"
        )
    total_parameters = sum(parameter.numel() for _name, parameter in adapter_parameters)
    if total_parameters != core.EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            "LoRA parameter count did not match the pinned 2,015,232-parameter inventory"
        )
    if any(parameter.dtype != torch.float32 for _name, parameter in adapter_parameters):
        raise RuntimeError("LoRA adapter tensors must remain FP32")
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            if parameter.dtype != torch.bfloat16:
                raise RuntimeError("frozen Qwen backbone tensors must remain BF16")
            if parameter.requires_grad:
                raise RuntimeError("Qwen backbone parameters must remain frozen")
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if require_trainable:
        if len(trainable) != len(adapter_parameters) or any(
            ".lora_A." not in name and ".lora_B." not in name for name, _parameter in trainable
        ):
            raise RuntimeError("only the exact pinned LoRA tensors may be trainable")
    elif trainable:
        raise RuntimeError("freshly reloaded adapters must be frozen for parity diagnostics")
    return {
        "module_count": len(modules),
        "module_names": [name for name, _module in modules],
        "adapter_tensor_count": len(adapter_parameters),
        "adapter_parameter_count": total_parameters,
        "adapter_tensor_names": [name for name, _parameter in adapter_parameters],
        "adapter_tensor_shapes": {
            name: list(parameter.shape) for name, parameter in adapter_parameters
        },
        "adapter_tensor_dtypes": {
            name: str(parameter.dtype) for name, parameter in adapter_parameters
        },
        "trainable_parameter_count": sum(parameter.numel() for _name, parameter in trainable),
        "base_parameters_frozen_bf16": True,
    }


def _adapter_state(model: Any, peft: Any) -> dict[str, Any]:
    state = peft.get_peft_model_state_dict(model)
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _adapter_state_changes(
    before: dict[str, Any], after: dict[str, Any], torch: Any
) -> dict[str, object]:
    if set(before) != set(after):
        raise RuntimeError("training changed the adapter tensor key inventory")
    changed = []
    for name in sorted(before):
        if before[name].shape != after[name].shape:
            raise RuntimeError("training changed an adapter tensor shape")
        if not torch.equal(before[name], after[name]):
            changed.append(name)
    return {"changed_tensor_count": len(changed), "changed_tensor_names": changed}


def _file_hashes(directory: Path) -> dict[str, str]:
    return {path.name: _sha256_file(path) for path in sorted(directory.iterdir()) if path.is_file()}


def _save_adapter_snapshot(model: Any, run_dir: Path, updates: int) -> dict[str, object]:
    target = run_dir / f"adapter-update-{updates:03d}"
    staging = run_dir / f".adapter-update-{updates:03d}-staging"
    if target.exists() or staging.exists():
        raise FileExistsError("adapter snapshot path already exists")
    staging.mkdir()
    try:
        model.save_pretrained(staging, safe_serialization=True)
        required = ("adapter_model.safetensors", "adapter_config.json")
        if any(not (staging / name).is_file() for name in required):
            raise RuntimeError("PEFT did not write adapter weights and config")
        config_path = staging / "adapter_config.json"
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(adapter_config, dict):
            raise RuntimeError("PEFT adapter config was not a JSON object")
        adapter_config["base_model_name_or_path"] = core.MODEL_ID
        adapter_config["revision"] = core.MODEL_REVISION
        _write_json(config_path, adapter_config, replace=True)
        os.rename(staging, target)
    except Exception:
        for path in staging.glob("*") if staging.exists() else ():
            path.unlink(missing_ok=True)
        if staging.exists():
            staging.rmdir()
        raise
    return {
        "update": updates,
        "path": str(target),
        "files_sha256": _file_hashes(target),
    }


def _load_base(
    tokenizer_cls: Any, model_cls: Any, torch: Any
) -> tuple[Any, Any, dict[str, object]]:
    from huggingface_hub import snapshot_download
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from experiments.baseline_qwen import validate_loading_info

    checkpoint = _download_base_snapshot(snapshot_download)
    tokenizer = tokenizer_cls.from_pretrained(
        checkpoint,
        revision=core.MODEL_REVISION,
        trust_remote_code=False,
        token=False,
    )
    model, load_info = model_cls.from_pretrained(
        checkpoint,
        revision=core.MODEL_REVISION,
        dtype=torch.bfloat16,
        trust_remote_code=False,
        output_loading_info=True,
        attn_implementation="eager",
        use_kernels=False,
        use_safetensors=True,
        token=False,
    )
    diagnostics = validate_loading_info(load_info)
    tied = bool(
        model.config.tie_word_embeddings
        and model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    )
    no_meta = all(parameter.device.type != "meta" for parameter in model.parameters())
    if not (
        isinstance(model, Qwen3_5ForCausalLM)
        and isinstance(model.config, Qwen3_5TextConfig)
        and tied
        and no_meta
        and len(model.model.layers) == 24
    ):
        raise RuntimeError("Qwen checkpoint shape or tied-embedding checks failed")
    tokenizer_hashes = {
        name: _sha256_file(Path(checkpoint) / name)
        for name in TOKENIZER_FILE_NAMES
        if (Path(checkpoint) / name).is_file()
    }
    if not tokenizer_hashes:
        raise RuntimeError("pinned tokenizer files were not present in the checkpoint snapshot")
    model = model.to("cuda").eval()
    provenance = {
        "model_id": core.MODEL_ID,
        "model_revision": core.MODEL_REVISION,
        "model_class": type(model).__name__,
        "config_class": type(model.config).__name__,
        "layer_count": len(model.model.layers),
        "tied_embeddings": tied,
        "no_meta_parameters": no_meta,
        "load_diagnostics": diagnostics,
        "tokenizer_file_sha256": tokenizer_hashes,
        "effective_dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "attention_implementation": "eager",
        "use_kernels": False,
        "use_hub_kernels": os.environ.get("USE_HUB_KERNELS"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "transformers",
                "peft",
                "huggingface-hub",
                "tokenizers",
                "safetensors",
                "pydantic",
            )
        },
    }
    return model, tokenizer, provenance


def _download_base_snapshot(snapshot_download: Any) -> str:
    """Download only the pinned public checkpoint without sending Hub credentials."""

    return snapshot_download(
        repo_id=core.MODEL_ID,
        revision=core.MODEL_REVISION,
        token=False,
    )


def _assert_runtime(torch: Any) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is unavailable")
    if not torch.cuda.get_device_name(0).strip():
        raise RuntimeError("CUDA device name is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)


def _runtime_package_versions() -> dict[str, str]:
    actual = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    expected = {
        "torch": f"{EXPECTED_PACKAGES['torch']}+cu130",
        "torchvision": f"{EXPECTED_PACKAGES['torchvision']}+cu130",
        "transformers": EXPECTED_PACKAGES["transformers"],
        "peft": EXPECTED_PACKAGES["peft"],
    }
    if actual != expected:
        raise RuntimeError("remote training package versions did not match the pinned image")
    return actual


def _remote_train(payload: dict[str, object]) -> dict[str, object]:
    """Run one bounded training rehearsal inside the authorized Modal container."""

    raw_attempt_id = payload.get("attempt_id")
    receipt = _initial_receipt(
        str(payload.get("run_id", "unknown")),
        attempt_id=raw_attempt_id if isinstance(raw_attempt_id, str) else "unknown",
    )
    receipt["phase"] = "payload_validation"
    run_dir: Path | None = None
    owns_run_dir = False
    stage = "payload_validation"
    torch: Any | None = None
    try:
        if set(payload) != {
            "run_id",
            "attempt_id",
            "records_jsonl",
            "manifest_json",
            "protocol_sha256",
            "source_file_sha256",
        }:
            raise ValueError("remote payload contained unexpected fields")
        run_id = core.validate_run_id(payload["run_id"])
        attempt_id = payload["attempt_id"]
        if not isinstance(attempt_id, str) or str(uuid.UUID(attempt_id)) != attempt_id:
            raise ValueError("remote attempt ID must be a canonical UUID")
        records_jsonl = payload["records_jsonl"]
        manifest_json = payload["manifest_json"]
        protocol_sha256 = payload["protocol_sha256"]
        source_file_sha256 = payload["source_file_sha256"]
        if not isinstance(records_jsonl, str) or not isinstance(manifest_json, str):
            raise ValueError("remote data payloads must be UTF-8 strings")
        if protocol_sha256 != core.EXPECTED_PROTOCOL_SHA256:
            raise ValueError("remote protocol hash did not match the pinned recipe")
        manifest, records = core.verify_prepared_bytes(
            records_jsonl.encode("utf-8"), manifest_json.encode("utf-8")
        )
        core.audit_splits(manifest, records)
        measured_source_hashes = _verify_remote_source_fingerprints(source_file_sha256)
        receipt["run_id"] = run_id
        receipt["attempt_id"] = attempt_id
        receipt["provenance"].update(
            {
                "records_sha256": core.EXPECTED_RECORDS_SHA256,
                "manifest_sha256": core.EXPECTED_MANIFEST_SHA256,
                "protocol_sha256": protocol_sha256,
                "source_file_sha256": source_file_sha256,
                "measured_source_file_sha256": measured_source_hashes,
                "seed": core.SEED,
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": 5e-4,
                    "weight_decay": 0.0,
                    "max_gradient_norm": 1.0,
                },
            }
        )

        root = VOLUME_ROOT / "runs"
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / run_id
        run_dir.mkdir()
        owns_run_dir = True
        receipt["phase"] = "run_reserved"
        receipt["evidence"]["run_directory"] = str(run_dir)
        _write_progress(run_dir, receipt)
        _commit_volume()

        stage = "runtime_check"
        os.environ["USE_HUB_KERNELS"] = "NO"
        optional_packages = {
            name: name in sys.modules or importlib.util.find_spec(name) is not None
            for name in ("kernels", "fla", "causal_conv1d")
        }
        if any(optional_packages.values()):
            raise RuntimeError("optional model-kernel packages must not be installed")
        import torch as torch_module
        import torch.nn.functional as functional
        import transformers
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_module
        receipt["provenance"]["training_package_versions"] = _runtime_package_versions()
        _assert_runtime(torch)
        receipt["provenance"]["optional_kernel_packages_present"] = optional_packages
        receipt["provenance"]["cuda_device"] = torch.cuda.get_device_name(0)
        receipt["provenance"]["device_capability"] = list(torch.cuda.get_device_capability(0))
        receipt["provenance"]["torch_cuda_runtime"] = torch.version.cuda
        receipt["provenance"]["transformers_version"] = str(transformers.__version__)
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            lines = driver.stdout.strip().splitlines()
            receipt["provenance"]["driver_version"] = lines[0] if lines else None
        except (OSError, subprocess.SubprocessError):
            receipt["provenance"]["driver_version"] = None

        random.seed(core.SEED)
        torch.manual_seed(core.SEED)
        torch.cuda.manual_seed_all(core.SEED)
        stage = "base_model_load"
        base_model, tokenizer, model_provenance = _load_base(
            AutoTokenizer, AutoModelForCausalLM, torch
        )
        receipt["provenance"].update(model_provenance)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has no usable pad or EOS token")
        forward_counter = {"count": 0}

        stage = "baseline_diagnostics"
        baseline = _evaluate(base_model, tokenizer, records, torch, forward_counter)
        receipt["evidence"]["forward_count"] = forward_counter["count"]
        receipt["evidence"]["baseline"] = {
            key: value for key, value in baseline.items() if key != "presentations"
        }
        receipt["evidence"]["baseline"]["saturated"] = (
            baseline["accuracy"] >= core.EARLY_STOP_ACCURACY
        )
        _write_progress(run_dir, receipt)
        _commit_volume()

        stage = "adapter_setup"
        base_model.requires_grad_(False)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(core.LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        inventory = _validate_adapter_inventory(model, torch, require_trainable=True)
        receipt["evidence"]["adapter_inventory"] = inventory
        initial_adapter_state = _adapter_state(model, importlib.import_module("peft"))
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=5e-4,
            weight_decay=0.0,
        )

        schedule: list[core.TrainingExample] = []
        updates_per_epoch = len(records) // core.MICROBATCHES_PER_UPDATE
        if len(records) % core.MICROBATCHES_PER_UPDATE:
            raise RuntimeError("training record count must divide into complete update epochs")
        if updates_per_epoch == 0:
            raise RuntimeError("training fixture must hold at least one complete update epoch")
        required_examples = core.MAX_UPDATES * core.MICROBATCHES_PER_UPDATE
        for epoch in range(math.ceil(required_examples / len(records))):
            schedule.extend(core.epoch_examples(records, seed=core.SEED, epoch=epoch))
        if len(schedule) < required_examples:
            raise RuntimeError("training schedule did not cover the update limit")
        receipt["evidence"]["training_step_losses"] = []
        receipt["evidence"]["diagnostics"] = []
        receipt["evidence"]["adapter_paths"] = []
        receipt["phase"] = "training"

        stage = "training"
        model.train()
        b_gradient_l1_total = 0.0
        final_diagnostic: dict[str, object] | None = None
        for update in range(1, core.MAX_UPDATES + 1):
            stage = "training"
            optimizer.zero_grad(set_to_none=True)
            microbatch_losses: list[float] = []
            start = (update - 1) * core.MICROBATCHES_PER_UPDATE
            update_examples = schedule[start : start + core.MICROBATCHES_PER_UPDATE]
            if len(update_examples) != core.MICROBATCHES_PER_UPDATE:
                raise RuntimeError("optimizer update did not receive exactly four examples")
            for example in update_examples:
                logits, _compiled = _compiled_candidate_logits(
                    model, example.request, tokenizer, torch, forward_counter
                )
                loss = functional.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([example.gold_index], dtype=torch.long, device="cuda"),
                )
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("training loss was non-finite")
                microbatch_losses.append(float(loss.detach().item()))
                (loss / core.MICROBATCHES_PER_UPDATE).backward()

            b_gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if ".lora_B." in name and parameter.grad is not None
            ]
            gradients = [
                parameter.grad
                for _name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
            if not b_gradients or any(
                not bool(torch.isfinite(gradient).all().item()) for gradient in gradients
            ):
                raise RuntimeError("LoRA-B gradients were missing or non-finite")
            b_gradient_l1 = sum(
                float(gradient.detach().abs().sum().item()) for gradient in b_gradients
            )
            b_gradient_l1_total += b_gradient_l1
            if any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if ".lora_A." not in name and ".lora_B." not in name
            ):
                raise RuntimeError("a frozen backbone parameter received a gradient")
            norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            if not bool(torch.isfinite(norm).item()):
                raise RuntimeError("LoRA gradient norm was non-finite")
            optimizer.step()
            receipt["evidence"]["optimizer_updates_completed"] = update
            receipt["evidence"]["forward_count"] = forward_counter["count"]
            receipt["evidence"]["training_step_losses"].append(
                {
                    "update": update,
                    "mean_loss": sum(microbatch_losses) / len(microbatch_losses),
                    "lo_ra_b_gradient_l1": b_gradient_l1,
                }
            )

            if update not in core.DIAGNOSTIC_UPDATES:
                continue
            stage = "checkpoint_save"
            model.eval()
            snapshot = _save_adapter_snapshot(model, run_dir, update)
            receipt["evidence"]["adapter_paths"].append(snapshot)
            receipt["evidence"]["forward_count"] = forward_counter["count"]
            receipt["phase"] = f"checkpoint_update_{update}"
            _write_progress(run_dir, receipt)
            _commit_volume()

            stage = "training_diagnostics"
            current = _evaluate(model, tokenizer, records, torch, forward_counter)
            comparison = _compare_to_baseline(baseline, current)
            diagnostic = {
                "update": update,
                "loss": current["loss"],
                "accuracy": current["accuracy"],
                "correct": current["correct"],
                "by_task": current["by_task"],
                "by_order": current["by_order"],
                "changed_predictions_from_base": comparison["changed_predictions"],
                "max_candidate_logit_change_from_base": comparison["max_candidate_logit_change"],
            }
            receipt["evidence"]["diagnostics"].append(diagnostic)
            receipt["evidence"]["forward_count"] = forward_counter["count"]
            receipt["phase"] = f"diagnostics_update_{update}"
            _write_progress(run_dir, receipt)
            _commit_volume()
            final_diagnostic = current
            if (
                update >= core.DIAGNOSTIC_UPDATES[0]
                and current["accuracy"] >= core.EARLY_STOP_ACCURACY
            ):
                receipt["evidence"]["early_stop_reason"] = "training_fixture_accuracy_threshold"
                break
            model.train()

        if final_diagnostic is None:
            raise RuntimeError("training completed without the required 16-update diagnostic")
        stage = "training_evidence"
        final_adapter_state = _adapter_state(model, importlib.import_module("peft"))
        update_evidence = _adapter_state_changes(initial_adapter_state, final_adapter_state, torch)
        if update_evidence["changed_tensor_count"] == 0:
            raise RuntimeError("training did not change any LoRA adapter tensor")
        if b_gradient_l1_total <= 0.0:
            raise RuntimeError("training did not produce a nonzero aggregate LoRA-B gradient")
        receipt["evidence"]["adapter_update"] = update_evidence
        receipt["evidence"]["lo_ra_b_gradient_l1_total"] = b_gradient_l1_total
        receipt["evidence"]["forward_count"] = forward_counter["count"]
        if forward_counter["count"] > core.MAX_FORWARD_COUNT:
            raise RuntimeError("training exceeded the pinned forward-count budget")

        stage = "release_training_model"
        del optimizer
        del model
        del base_model
        del trainable_parameters
        del initial_adapter_state
        gc.collect()
        torch.cuda.empty_cache()

        stage = "fresh_base_reload"
        fresh_base, fresh_tokenizer, fresh_provenance = _load_base(
            AutoTokenizer, AutoModelForCausalLM, torch
        )
        if (
            fresh_provenance["tokenizer_file_sha256"]
            != receipt["provenance"]["tokenizer_file_sha256"]
        ):
            raise RuntimeError("fresh reload did not use identical pinned tokenizer files")
        final_snapshot = Path(receipt["evidence"]["adapter_paths"][-1]["path"])
        reloaded_model = PeftModel.from_pretrained(
            fresh_base,
            str(final_snapshot),
            is_trainable=False,
            autocast_adapter_dtype=True,
        ).eval()
        reloaded_inventory = _validate_adapter_inventory(
            reloaded_model, torch, require_trainable=False
        )
        reloaded_adapter_state = _adapter_state(reloaded_model, importlib.import_module("peft"))
        if set(reloaded_adapter_state) != set(final_adapter_state):
            raise RuntimeError("reloaded adapter tensor keys differ from trained adapter")
        for name, trained_tensor in final_adapter_state.items():
            if trained_tensor.shape != reloaded_adapter_state[name].shape or not torch.equal(
                trained_tensor, reloaded_adapter_state[name]
            ):
                raise RuntimeError("reloaded adapter tensor values differ from trained adapter")
        saved_adapter_state = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
            str(final_snapshot / "adapter_model.safetensors"), device="cpu"
        )
        serialized_adapter_state = importlib.import_module("peft").get_peft_model_state_dict(
            reloaded_model
        )
        normalized_serialized_state = {
            name: tensor.detach().cpu() for name, tensor in serialized_adapter_state.items()
        }
        if set(saved_adapter_state) != set(normalized_serialized_state):
            raise RuntimeError("reloaded adapter tensor keys differ from saved safetensors")
        for name, saved_tensor in saved_adapter_state.items():
            if saved_tensor.shape != normalized_serialized_state[name].shape or not torch.equal(
                saved_tensor, normalized_serialized_state[name]
            ):
                raise RuntimeError("reloaded adapter tensor values differ from saved safetensors")
        for _name, module in _adapter_modules_and_parameters(reloaded_model)[0]:
            merged_adapters = getattr(module, "merged_adapters", ())
            if merged_adapters or bool(getattr(module, "merged", False)):
                raise RuntimeError("reloaded LoRA adapter was merged into the base weights")

        stage = "reload_parity"
        reload_counter = {"count": 0}
        reload_result = _evaluate(reloaded_model, fresh_tokenizer, records, torch, reload_counter)
        if final_diagnostic is None:
            raise RuntimeError("final in-memory diagnostic was not retained")
        parity_comparison = _compare_to_baseline(final_diagnostic, reload_result)
        max_reload_difference = float(parity_comparison["max_candidate_logit_change"])
        final_winner_mismatches = int(parity_comparison["changed_predictions"])
        if max_reload_difference > 1e-3 or final_winner_mismatches != 0:
            raise RuntimeError(
                "fresh adapter reload did not match saved candidate logits and winners"
            )
        if forward_counter["count"] + reload_counter["count"] > core.MAX_FORWARD_COUNT:
            raise RuntimeError("total rehearsal forwards exceeded the pinned budget")

        stage = "final_receipt"
        receipt["phase"] = "completed"
        receipt["evidence"]["forward_count"] = forward_counter["count"] + reload_counter["count"]
        receipt["evidence"]["reload_parity"] = {
            "presentation_count": reload_result["presentation_count"],
            "max_candidate_logit_difference": max_reload_difference,
            "winner_mismatches": final_winner_mismatches,
            "adapter_tensor_keys_shapes_values_match": True,
            "adapter_tensors_unmerged_fp32": True,
            "model_eval_mode": not reloaded_model.training,
            "use_cache": False,
            "inventory": reloaded_inventory,
        }
        receipt["evidence"]["max_forward_count"] = core.MAX_FORWARD_COUNT
        receipt["evidence"]["max_gpu_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(0)
        )
        receipt["evidence"]["train_forward_count"] = forward_counter["count"] - (
            128 * (1 + len(receipt["evidence"]["diagnostics"]))
        )
        latest_accuracy = float(receipt["evidence"]["diagnostics"][-1]["accuracy"])
        threshold_met = latest_accuracy >= core.EARLY_STOP_ACCURACY
        evidence_ok = (
            int(update_evidence["changed_tensor_count"]) > 0
            and b_gradient_l1_total > 0.0
            and final_winner_mismatches == 0
            and max_reload_difference <= 1e-3
            and receipt["evidence"]["forward_count"] <= core.MAX_FORWARD_COUNT
        )
        receipt["evidence"]["memorization_threshold_met"] = threshold_met
        receipt["status"] = "passed" if threshold_met and evidence_ok else "failed"
        if not threshold_met:
            receipt["failure"] = {
                "stage": "memorization_threshold",
                "type": "AccuracyThresholdNotReached",
                "message": "the training fixture did not reach 95% accuracy within 128 updates",
            }
        elif not evidence_ok:
            receipt["failure"] = {
                "stage": "training_evidence",
                "type": "TrainingEvidenceIncomplete",
                "message": "one or more adapter update or reload parity checks failed",
            }
        else:
            receipt.pop("failure", None)
        _write_progress(run_dir, receipt)
        _commit_volume()
    except Exception as exc:
        receipt["failure"] = _failure(stage, exc)
        receipt["status"] = "failed"
        receipt["phase"] = "failed"
        if owns_run_dir and run_dir is not None and run_dir.is_dir():
            try:
                if torch is not None and torch.cuda.is_available():
                    receipt["evidence"]["max_gpu_memory_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated(0)
                    )
                _write_progress(run_dir, receipt)
                _commit_volume()
            except Exception:
                pass
    return _normalize_receipt(receipt)


def _launch(
    *,
    profile: str,
    workspace: str,
    run_id: str,
    records_path: str,
    manifest_path: str,
    reservation: smoke.ArtifactReservation,
) -> int:
    attempt_id = str(uuid.uuid4())
    result = _initial_receipt(run_id, attempt_id=attempt_id, profile=profile, workspace=workspace)
    volume: Any | None = None
    remote_call_started = False
    remote_receipt_received = False
    expected_remote_identity: dict[str, object] = {}
    stage = "preflight"
    try:
        core.validate_run_id(run_id)
        manifest, records = core.verify_prepared_data(records_path, manifest_path)
        protocol_sha256 = core.verify_protocol()
        if protocol_sha256 != core.EXPECTED_PROTOCOL_SHA256:
            raise ValueError("training rehearsal protocol SHA-256 mismatch")
        source_hashes = _source_fingerprints()
        expected_remote_identity = {
            "run_id": core.validate_run_id(run_id),
            "attempt_id": attempt_id,
            "source_file_sha256": source_hashes,
        }
        result["provenance"].update(
            {
                "records_sha256": _sha256_file(records_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "protocol_sha256": protocol_sha256,
                "source_file_sha256": source_hashes,
                "record_count": len(records),
                "dataset_count": len(manifest.datasets),
            }
        )
        stage = "environment"
        if any(os.environ.get(name, "") for name in modal_smoke.CREDENTIAL_OVERRIDES):
            raise RuntimeError("Modal credential environment overrides are not accepted")
        stage = "auth"
        result["phase"] = "auth"
        if not modal_smoke._verify_profile(profile, workspace):
            raise RuntimeError("Modal token info did not match the expected workspace")
        stage = "modal_setup"
        result["phase"] = stage
        modal_version = importlib.metadata.version("modal")
        if modal_version != "1.6.1":
            raise RuntimeError("Modal 1.6.1 is required")

        # Volume construction is intentionally after data, protocol, credential, and profile checks.
        modal = importlib.import_module("modal")
        volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        project_root = Path(__file__).resolve().parents[1]
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .env({"USE_HUB_KERNELS": "NO"})
            .pip_install(
                "torch==2.14.1+cu130",
                "torchvision==0.29.1+cu130",
                "transformers==5.18.0",
                "pydantic==2.13.5",
                "peft==0.21.0",
                "pillow==12.0.0",
                extra_index_url="https://download.pytorch.org/whl/cu130",
            )
            .add_local_python_source("reflex_decisions", copy=True)
            .add_local_file(
                str(project_root / "experiments" / "training_rehearsal_core.py"),
                remote_path="/root/experiments/training_rehearsal_core.py",
            )
            .add_local_file(
                str(project_root / "experiments" / "baseline_qwen.py"),
                remote_path="/root/experiments/baseline_qwen.py",
            )
            .add_local_file(
                str(project_root / "experiments" / "modal_smoke.py"),
                remote_path="/root/experiments/modal_smoke.py",
            )
            .add_local_file(
                str(project_root / "experiments" / "modal_train_rehearsal.py"),
                remote_path="/root/experiments/modal_train_rehearsal.py",
            )
        )
        app = modal.App("reflex-qwen-lora-training-rehearsal", image=image)
        run_training = app.function(
            gpu="A10",
            cpu=(2.0, 2.0),
            memory=(16384, 16384),
            max_containers=1,
            min_containers=0,
            buffer_containers=0,
            scaledown_window=2,
            retries=0,
            single_use_containers=True,
            serialized=True,
            include_source=False,
            startup_timeout=300,
            timeout=1800,
            volumes={"/artifacts": volume},
        )(_remote_train)
        payload = {
            "run_id": core.validate_run_id(run_id),
            "attempt_id": attempt_id,
            "records_jsonl": Path(records_path).read_text(encoding="utf-8"),
            "manifest_json": Path(manifest_path).read_text(encoding="utf-8"),
            "protocol_sha256": protocol_sha256,
            "source_file_sha256": source_hashes,
        }
        stage = "remote_training"
        result["phase"] = "remote_training"
        with modal.enable_output(), app.run():
            remote_call_started = True
            remote_result = run_training.remote(payload)
            if not isinstance(remote_result, dict):
                raise RuntimeError("Modal returned an unexpected training receipt")
            remote_receipt_received = True
            result = _validate_remote_receipt(remote_result, expected_remote_identity)
            result["provenance"].update(
                {
                    "profile": profile,
                    "workspace": workspace,
                    "modal_sdk_version": modal_version,
                    "local_records_sha256": _sha256_file(records_path),
                    "local_manifest_sha256": _sha256_file(manifest_path),
                    "local_protocol_sha256": protocol_sha256,
                    "local_source_file_sha256": source_hashes,
                }
            )
    except Exception as exc:
        failure = _failure(stage, exc)
        result["failure"] = failure
        result["status"] = "failed"
        result["phase"] = "failed"
        if remote_call_started and not remote_receipt_received and volume is not None:
            try:
                progress = _read_volume_progress(volume, run_id)
                progress = _validate_remote_receipt(
                    progress,
                    expected_remote_identity,
                    require_full_provenance=True,
                )
                progress_provenance = progress["provenance"]
                progress_provenance.update(result["provenance"])
                progress["status"] = "failed"
                progress["phase"] = "failed"
                progress["failure"] = failure
                result = progress
            except Exception:
                pass

    result = _normalize_receipt(result)
    smoke.write_json_artifact(reservation, result)
    print(
        json.dumps(
            {"status": result.get("status", "failed"), "artifact": str(reservation.destination)}
        )
    )
    return 0 if result.get("status") == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch", action="store_true", help="explicitly launch paid remote training"
    )
    parser.add_argument("--profile", default=PROFILE, help="named personal Modal profile")
    parser.add_argument("--workspace", default=WORKSPACE, help="expected Modal Workspace")
    parser.add_argument("--run-id", help="unique safe run identifier in the artifact volume")
    parser.add_argument("--records", default=core.DEFAULT_RECORDS, help="pinned training JSONL")
    parser.add_argument(
        "--manifest", default=core.DEFAULT_MANIFEST, help="pinned training manifest"
    )
    parser.add_argument("--output", help="new local JSON receipt path")
    args = parser.parse_args(argv)
    if not args.launch:
        print(json.dumps(core.plan(), sort_keys=True, indent=2))
        return 0
    if not args.output or not args.run_id:
        parser.error("--launch requires --run-id and --output")
    try:
        core.validate_run_id(args.run_id)
        with smoke.reserve_output(args.output) as reservation:
            return _launch(
                profile=args.profile.strip(),
                workspace=args.workspace.strip(),
                run_id=args.run_id,
                records_path=args.records,
                manifest_path=args.manifest,
                reservation=reservation,
            )
    except (OSError, ValueError) as exc:
        print(f"output or recipe preflight failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
