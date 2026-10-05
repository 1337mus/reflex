"""Plan or explicitly launch the bounded, request-label-separated real-data pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from experiments import modal_smoke, modal_train_rehearsal
from experiments import real_pilot_core as core
from reflex_decisions.data import DecisionRecord
from reflex_decisions.rendering import CompiledRequest, compile_request
from reflex_decisions.schema import DecisionRequest

PROFILE = modal_train_rehearsal.PROFILE
WORKSPACE = modal_train_rehearsal.WORKSPACE
VOLUME_NAME = modal_train_rehearsal.VOLUME_NAME
VOLUME_ROOT = Path("/artifacts")


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    return {
        "stage": stage,
        "type": type(error).__name__,
        "message": modal_train_rehearsal.smoke.sanitize_exception_message(error),
    }


def _canonical_module_key(path: str | Path) -> str:
    resolved = Path(path).resolve()
    for remote_root, canonical_root in (
        (Path("/root/experiments"), "experiments"),
        (Path("/root/reflex_decisions"), "src/reflex_decisions"),
    ):
        try:
            relative = resolved.relative_to(remote_root)
        except ValueError:
            continue
        if relative.suffix == ".py":
            return f"{canonical_root}/{relative.as_posix()}"
    raise ValueError("remote source module resolved outside approved package roots")


def _measure_remote_source_fingerprints() -> dict[str, str]:
    measured: dict[str, str] = {}
    for module_name, expected_path in core.SOURCE_FINGERPRINT_MODULES:
        module = importlib.import_module(module_name)
        module_path = getattr(module, "__file__", None)
        if not isinstance(module_path, str):
            raise ValueError(f"remote source module has no file: {module_name}")
        canonical = _canonical_module_key(module_path)
        if canonical != expected_path or canonical in measured:
            raise ValueError(f"remote source path did not match its canonical key: {module_name}")
        measured[canonical] = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    protocol_path = Path("/root") / core.PROTOCOL_PATH
    if not protocol_path.is_file():
        raise ValueError("remote protocol file is missing")
    measured[core.PROTOCOL_PATH] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    return measured


def _initial_receipt(run_id: str, nonce: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "failed",
        "run_id": run_id,
        "nonce": nonce,
        "pins": {},
        "phase": "preflight",
        "failure": {
            "stage": "not_started",
            "type": "NotStarted",
            "message": "the pilot did not return a complete receipt",
        },
        "provenance": {
            "model_id": core.MODEL_ID,
            "model_revision": core.MODEL_REVISION,
        },
        "evidence": {},
        "limits": core.MODAL_LIMITS,
    }


def _candidate_tensor_logits(
    model: Any,
    request: DecisionRequest,
    compiled: CompiledRequest,
    torch: Any,
    counter: dict[str, int],
) -> Any:
    input_ids = torch.tensor([compiled.input_ids], dtype=torch.long, device="cuda")
    attention_mask = torch.ones_like(input_ids)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=1,
    )
    counter["count"] += 1
    candidate_ids = torch.tensor(compiled.candidate_token_ids, dtype=torch.long, device="cuda")
    logits = output.logits[0, -1].index_select(0, candidate_ids).float()
    if logits.numel() != len(request.options) or not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("candidate logits were missing, non-finite, or misaligned")
    return logits


def _semantic_winner(option_ids: list[str], logits: list[float]) -> str:
    maximum = max(logits)
    return min(
        option_id for option_id, value in zip(option_ids, logits, strict=True) if value == maximum
    )


def _score_presentations(
    model: Any,
    compiled_rows: list[tuple[dict[str, object], DecisionRequest, CompiledRequest]],
    torch: Any,
    counter: dict[str, int],
) -> list[dict[str, object]]:
    model.eval()
    results: list[dict[str, object]] = []
    with torch.inference_mode():
        for row, request, compiled in compiled_rows:
            logits_tensor = _candidate_tensor_logits(model, request, compiled, torch, counter)
            logits = [float(value) for value in logits_tensor.detach().cpu().tolist()]
            order_ids = [option.id for option in request.options]
            results.append(
                {
                    "presentation_id": row["presentation_id"],
                    "record_id": row["record_id"],
                    "dataset_id": row["dataset_id"],
                    "request_hash": row["request_hash"],
                    "order_ids": order_ids,
                    "candidate_logits": logits,
                    "winner_option_id": _semantic_winner(order_ids, logits),
                    "input_tokens": len(compiled.input_ids),
                    "prompt_sha256": compiled.prompt_hash,
                }
            )
    return results


def _train_records(rows: list[dict[str, object]]) -> tuple[DecisionRecord, ...]:
    return tuple(
        DecisionRecord(
            record_id=str(row["record_id"]),
            dataset_id=str(row["dataset_id"]),
            source_group_id=f"approved-train:{row['record_id']}",
            request=DecisionRequest.model_validate(row["request"]),
            answer_id=str(row["answer_id"]),
        )
        for row in rows
    )


def _write_outputs(run_dir: Path, name: str, rows: list[dict[str, object]]) -> dict[str, str]:
    path = run_dir / f"{name}.json"
    encoded = (
        json.dumps(
            {"presentations": rows},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    with path.open("xb") as stream:
        if stream.write(encoded) != len(encoded):
            raise OSError("evaluation output write was incomplete")
        stream.flush()
        os.fsync(stream.fileno())
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("saved evaluation output hash changed after writing")
    return {"path": str(path), "sha256": digest}


def _persist_progress(run_dir: Path, receipt: dict[str, object]) -> None:
    """Persist compact progress; full output arrays live in their dedicated JSON files."""

    progress = dict(receipt)
    evidence = dict(receipt.get("evidence", {}))
    evidence.pop("base_outputs", None)
    evidence.pop("final_outputs", None)
    progress["evidence"] = evidence
    modal_train_rehearsal._write_progress(run_dir, progress)


def _remote_train(payload: dict[str, object]) -> dict[str, object]:
    """Run one fixed schedule on the ephemeral worker, returning strict JSON only."""

    run_value = payload.get("run_id")
    nonce_value = payload.get("nonce")
    run_id = run_value if isinstance(run_value, str) else "unknown"
    nonce = nonce_value if isinstance(nonce_value, str) else "unknown"
    receipt = _initial_receipt(run_id, nonce)
    stage = "payload_validation"
    run_dir: Path | None = None
    owns_run_dir = False
    forward_counts = {
        "base_evaluation": 0,
        "training": 0,
        "final_evaluation": 0,
        "reload_parity": 0,
        "total": 0,
    }
    torch: Any | None = None
    try:
        remote_payload = core.validate_remote_payload(payload)
        run_id = core.validate_run_id(remote_payload["run_id"])
        nonce = core.validate_nonce(remote_payload["nonce"])
        pins = remote_payload["pins"]
        if not isinstance(pins, dict):
            raise ValueError("remote payload pins are malformed")
        measured_hashes = _measure_remote_source_fingerprints()
        if measured_hashes != pins["source_file_sha256"]:
            raise ValueError("uploaded module hashes differ from the reviewed source map")
        if measured_hashes.get(core.PROTOCOL_PATH) != pins["protocol_sha256"]:
            raise ValueError("uploaded protocol hash differs from the frozen protocol pin")
        receipt.update(
            {
                "run_id": run_id,
                "nonce": nonce,
                "pins": pins,
                "phase": "source_verified",
                "provenance": {
                    "model_id": core.MODEL_ID,
                    "model_revision": core.MODEL_REVISION,
                    "seed": core.SEED,
                    "optimizer": {
                        "name": "AdamW",
                        "learning_rate": 5e-4,
                        "weight_decay": 0.0,
                        "max_gradient_norm": 1.0,
                    },
                    "source_file_sha256": pins["source_file_sha256"],
                    "measured_source_file_sha256": measured_hashes,
                },
                "evidence": {
                    "optimizer_updates_completed": 0,
                    "forward_counts": forward_counts,
                    "training_step_losses": [],
                    "adapter_paths": [],
                    "base_gradients_none": True,
                },
            }
        )

        run_root = VOLUME_ROOT / "runs"
        run_root.mkdir(parents=True, exist_ok=True)
        run_dir = run_root / run_id
        run_dir.mkdir(exist_ok=False)
        owns_run_dir = True
        _persist_progress(run_dir, receipt)
        modal_train_rehearsal._commit_volume()

        stage = "runtime_check"
        os.environ["USE_HUB_KERNELS"] = "NO"
        import torch as torch_module
        import torch.nn.functional as functional
        import transformers
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = torch_module
        modal_train_rehearsal._assert_runtime(torch)
        receipt["provenance"].update(
            {
                "training_package_versions": modal_train_rehearsal._runtime_package_versions(),
                "cuda_device": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "torch_cuda_runtime": torch.version.cuda,
                "transformers_version": str(transformers.__version__),
            }
        )
        import random

        random.seed(core.SEED)
        torch.manual_seed(core.SEED)
        torch.cuda.manual_seed_all(core.SEED)
        stage = "base_model_load"
        base_model, tokenizer, model_provenance = modal_train_rehearsal._load_base(
            AutoTokenizer, AutoModelForCausalLM, torch
        )
        receipt["provenance"].update(model_provenance)

        train_rows = remote_payload["train_records"]
        evaluation_rows = remote_payload["evaluation_presentations"]
        if not isinstance(train_rows, list) or not isinstance(evaluation_rows, list):
            raise ValueError("remote payload rows are malformed")
        train_records = _train_records(train_rows)
        schedule = core.build_training_schedule(train_records)
        compiled_training = [
            (example, compile_request(example.request, tokenizer, max_tokens=core.MAX_INPUT_TOKENS))
            for example in schedule
        ]
        compiled_evaluation = []
        for row in evaluation_rows:
            if not isinstance(row, dict):
                raise ValueError("evaluation request row is malformed")
            request = DecisionRequest.model_validate(row["request"])
            compiled = compile_request(request, tokenizer, max_tokens=core.MAX_INPUT_TOKENS)
            compiled_evaluation.append((row, request, compiled))

        stage = "base_evaluation"
        base_counter = {"count": 0}
        try:
            base_outputs = _score_presentations(
                base_model, compiled_evaluation, torch, base_counter
            )
        finally:
            forward_counts["base_evaluation"] = base_counter["count"]
            forward_counts["total"] = sum(
                forward_counts[name]
                for name in (
                    "base_evaluation",
                    "training",
                    "final_evaluation",
                    "reload_parity",
                )
            )
        forward_counts["base_evaluation"] = base_counter["count"]
        if forward_counts["base_evaluation"] != core.BASE_EVALUATION_COUNT:
            raise RuntimeError("base scoring did not use all 2,572 presentations")
        receipt["evidence"].update(
            {
                "base_outputs": base_outputs,
                "forward_counts": dict(forward_counts),
            }
        )
        stage = "base_outputs_save"
        output_artifacts = {"base_outputs": _write_outputs(run_dir, "base_outputs", base_outputs)}
        receipt["evidence"]["output_artifacts"] = dict(output_artifacts)
        receipt["phase"] = "base_outputs_saved"
        _persist_progress(run_dir, receipt)
        modal_train_rehearsal._commit_volume()

        stage = "adapter_setup"
        base_model.requires_grad_(False)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(modal_train_rehearsal.core.LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        inventory = modal_train_rehearsal._validate_adapter_inventory(
            model, torch, require_trainable=True
        )
        receipt["evidence"]["adapter_inventory"] = inventory
        peft_module = importlib.import_module("peft")
        initial_adapter_state = modal_train_rehearsal._adapter_state(model, peft_module)
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(trainable_parameters, lr=5e-4, weight_decay=0.0)
        training_losses: list[dict[str, object]] = []
        adapter_paths: list[dict[str, object]] = []
        training_counter = {"count": 0}
        forward_counts["training"] = training_counter["count"]
        b_gradient_l1_total = 0.0
        model.train()

        stage = "training"
        for update in range(1, core.MAX_UPDATES + 1):
            optimizer.zero_grad(set_to_none=True)
            microbatch_losses: list[float] = []
            start = (update - 1) * core.MICROBATCHES_PER_UPDATE
            minibatch = compiled_training[start : start + core.MICROBATCHES_PER_UPDATE]
            if len(minibatch) != core.MICROBATCHES_PER_UPDATE:
                raise RuntimeError("optimizer update did not receive exactly four examples")
            for example, compiled in minibatch:
                logits = _candidate_tensor_logits(
                    model, example.request, compiled, torch, training_counter
                )
                forward_counts["training"] = training_counter["count"]
                forward_counts["total"] = sum(
                    forward_counts[name]
                    for name in (
                        "base_evaluation",
                        "training",
                        "final_evaluation",
                        "reload_parity",
                    )
                )
                receipt["evidence"]["forward_counts"] = dict(forward_counts)
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
                raise RuntimeError("LoRA gradients were missing or non-finite")
            if any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if ".lora_A." not in name and ".lora_B." not in name
            ):
                raise RuntimeError("a frozen backbone parameter received a gradient")
            gradient_l1 = sum(
                float(gradient.detach().abs().sum().item()) for gradient in b_gradients
            )
            b_gradient_l1_total += gradient_l1
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise RuntimeError("LoRA gradient norm was non-finite")
            optimizer.step()
            forward_counts["training"] = training_counter["count"]
            forward_counts["total"] = sum(
                forward_counts[name]
                for name in ("base_evaluation", "training", "final_evaluation", "reload_parity")
            )
            training_losses.append(
                {
                    "update": update,
                    "mean_loss": sum(microbatch_losses) / len(microbatch_losses),
                    "lo_ra_b_gradient_l1": gradient_l1,
                }
            )
            receipt["evidence"].update(
                {
                    "optimizer_updates_completed": update,
                    "training_step_losses": list(training_losses),
                    "forward_counts": dict(forward_counts),
                }
            )
            if update % 16 == 0:
                receipt["phase"] = f"training_update_{update}"
                _persist_progress(run_dir, receipt)
                modal_train_rehearsal._commit_volume()
            if update in {126, 252}:
                model.eval()
                snapshot = modal_train_rehearsal._save_adapter_snapshot(model, run_dir, update)
                if (
                    modal_train_rehearsal._file_hashes(Path(snapshot["path"]))
                    != snapshot["files_sha256"]
                ):
                    raise RuntimeError("adapter snapshot files did not match their saved hashes")
                adapter_paths.append(snapshot)
                receipt["evidence"]["adapter_paths"] = list(adapter_paths)
                receipt["phase"] = f"checkpoint_update_{update}"
                _persist_progress(run_dir, receipt)
                modal_train_rehearsal._commit_volume()
                if update != core.MAX_UPDATES:
                    model.train()

        if training_counter["count"] != core.TRAIN_FORWARD_COUNT:
            raise RuntimeError("training did not use exactly 1,008 forwards")
        if b_gradient_l1_total <= 0:
            raise RuntimeError("training did not produce finite nonzero adapter gradients")
        final_adapter_state = modal_train_rehearsal._adapter_state(model, peft_module)
        adapter_update = modal_train_rehearsal._adapter_state_changes(
            initial_adapter_state, final_adapter_state, torch
        )
        if adapter_update["changed_tensor_count"] == 0:
            raise RuntimeError("training did not change any adapter tensor")
        receipt["evidence"].update(
            {
                "adapter_update": adapter_update,
                "base_gradients_none": True,
            }
        )

        stage = "final_evaluation"
        final_counter = {"count": 0}
        try:
            final_outputs = _score_presentations(model, compiled_evaluation, torch, final_counter)
        finally:
            forward_counts["final_evaluation"] = final_counter["count"]
            forward_counts["total"] = sum(
                forward_counts[name]
                for name in (
                    "base_evaluation",
                    "training",
                    "final_evaluation",
                    "reload_parity",
                )
            )
        forward_counts["final_evaluation"] = final_counter["count"]
        if forward_counts["final_evaluation"] != core.FINAL_EVALUATION_COUNT:
            raise RuntimeError("final scoring did not use all 2,572 presentations")
        receipt["evidence"].update(
            {
                "final_outputs": final_outputs,
                "forward_counts": dict(forward_counts),
            }
        )
        stage = "final_outputs_save"
        output_artifacts["final_outputs"] = _write_outputs(run_dir, "final_outputs", final_outputs)
        receipt["evidence"]["output_artifacts"] = dict(output_artifacts)
        receipt["phase"] = "final_outputs_saved"
        _persist_progress(run_dir, receipt)
        modal_train_rehearsal._commit_volume()

        stage = "release_training_model"
        del optimizer, model, base_model, trainable_parameters, initial_adapter_state
        gc.collect()
        torch.cuda.empty_cache()

        stage = "fresh_base_reload"
        fresh_base, fresh_tokenizer, fresh_provenance = modal_train_rehearsal._load_base(
            AutoTokenizer, AutoModelForCausalLM, torch
        )
        if (
            fresh_provenance["tokenizer_file_sha256"]
            != receipt["provenance"]["tokenizer_file_sha256"]
        ):
            raise RuntimeError("fresh reload tokenizer files differed from the pinned base")
        snapshot_path = Path(adapter_paths[-1]["path"])
        reloaded_model = PeftModel.from_pretrained(
            fresh_base,
            str(snapshot_path),
            is_trainable=False,
            autocast_adapter_dtype=True,
        ).eval()
        modal_train_rehearsal._validate_adapter_inventory(
            reloaded_model, torch, require_trainable=False
        )
        reloaded_state = modal_train_rehearsal._adapter_state(reloaded_model, peft_module)
        if set(reloaded_state) != set(final_adapter_state):
            raise RuntimeError("fresh reload adapter tensor keys changed")
        for name, tensor in final_adapter_state.items():
            if tensor.shape != reloaded_state[name].shape or not torch.equal(
                tensor, reloaded_state[name]
            ):
                raise RuntimeError("fresh reload adapter tensor values changed")
        saved_state = importlib.import_module("safetensors.torch").load_file(
            str(snapshot_path / "adapter_model.safetensors"), device="cpu"
        )
        if set(saved_state) != set(reloaded_state) or any(
            saved_state[name].shape != reloaded_state[name].shape
            or not torch.equal(saved_state[name], reloaded_state[name])
            for name in saved_state
        ):
            raise RuntimeError("saved safetensors did not match the reloaded adapter tensors")
        if modal_train_rehearsal._file_hashes(snapshot_path) != adapter_paths[-1]["files_sha256"]:
            raise RuntimeError("final adapter files changed after their snapshot hash")

        stage = "reload_parity"
        sorted_eval = sorted(compiled_evaluation, key=lambda item: str(item[0]["presentation_id"]))
        parity_rows = sorted_eval[: core.RELOAD_PARITY_COUNT]
        fresh_compiled = [
            (
                row,
                request,
                compile_request(request, fresh_tokenizer, max_tokens=core.MAX_INPUT_TOKENS),
            )
            for row, request, _compiled in parity_rows
        ]
        parity_counter = {"count": 0}
        try:
            reloaded_outputs = _score_presentations(
                reloaded_model, fresh_compiled, torch, parity_counter
            )
        finally:
            forward_counts["reload_parity"] = parity_counter["count"]
            forward_counts["total"] = sum(
                forward_counts[name]
                for name in (
                    "base_evaluation",
                    "training",
                    "final_evaluation",
                    "reload_parity",
                )
            )
        forward_counts["reload_parity"] = parity_counter["count"]
        final_by_id = {row["presentation_id"]: row for row in final_outputs}
        parity_ids = [str(row[0]["presentation_id"]) for row in parity_rows]
        reloaded_by_id = {row["presentation_id"]: row for row in reloaded_outputs}
        differences = []
        winner_match = True
        reloaded_logits = []
        for presentation_id in parity_ids:
            expected = final_by_id[presentation_id]
            actual = reloaded_by_id[presentation_id]
            expected_logits = expected["candidate_logits"]
            actual_logits = actual["candidate_logits"]
            if expected["winner_option_id"] != actual["winner_option_id"]:
                winner_match = False
            differences.extend(
                abs(float(left) - float(right))
                for left, right in zip(expected_logits, actual_logits, strict=True)
            )
            reloaded_logits.append(actual_logits)
        maximum_difference = max(differences, default=math.inf)
        if not winner_match or maximum_difference > 1e-3:
            raise RuntimeError("fresh adapter reload parity did not meet the frozen tolerance")
        forward_counts["total"] = sum(
            forward_counts[name]
            for name in ("base_evaluation", "training", "final_evaluation", "reload_parity")
        )
        if forward_counts != {
            "base_evaluation": 2572,
            "training": 1008,
            "final_evaluation": 2572,
            "reload_parity": 32,
            "total": core.MAX_FORWARD_COUNT,
        }:
            raise RuntimeError("real pilot forward count did not match the fixed budget")
        receipt["evidence"].update(
            {
                "forward_counts": dict(forward_counts),
                "reload_parity": {
                    "presentation_ids": parity_ids,
                    "reloaded_logits": reloaded_logits,
                    "adapter_tensor_keys_shapes_values_match": True,
                    "winner_match": winner_match,
                    "max_abs_logit_diff": maximum_difference,
                },
            }
        )
        receipt["status"] = "passed"
        receipt["phase"] = "completed"
        receipt.pop("failure", None)
        core.validate_passed_receipt(receipt, expected_payload=remote_payload)
        _persist_progress(run_dir, receipt)
        modal_train_rehearsal._commit_volume()
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["phase"] = "failed"
        receipt["failure"] = _failure(stage, exc)
        if owns_run_dir and run_dir is not None:
            try:
                _persist_progress(run_dir, receipt)
                modal_train_rehearsal._commit_volume()
            except Exception:
                pass
    finally:
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return modal_train_rehearsal._normalize_receipt(receipt)


def _load_local_data(records_path: str, manifest_path: str, recipe_path: str):
    from reflex_decisions import pilot_data

    manifest, records, recipe = pilot_data.verify_prepared_data(
        records_path, manifest_path, recipe_path
    )
    core.validate_pilot_data(manifest, records)
    return manifest, records, recipe


def _launch(
    *,
    profile: str,
    workspace: str,
    run_id: str,
    records_path: str,
    manifest_path: str,
    recipe_path: str,
    reservation: modal_smoke.smoke.ArtifactReservation,
) -> int:
    nonce = str(uuid.uuid4())
    result = _initial_receipt(run_id, nonce)
    stage = "preflight"
    volume: Any | None = None
    remote_started = False
    remote_returned = False
    expected_payload: dict[str, object] | None = None
    try:
        run_id = core.validate_run_id(run_id)
        if any(os.environ.get(name, "") for name in modal_smoke.CREDENTIAL_OVERRIDES):
            raise RuntimeError("Modal credential environment overrides are not accepted")
        manifest, records, _recipe = _load_local_data(records_path, manifest_path, recipe_path)
        protocol_sha256 = core.verify_protocol()
        source_hashes = core.source_fingerprints()
        expected_payload = core.build_remote_payload(
            manifest,
            records,
            run_id=run_id,
            nonce=nonce,
            records_sha256=hashlib.sha256(Path(records_path).read_bytes()).hexdigest(),
            manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            recipe_sha256=hashlib.sha256(Path(recipe_path).read_bytes()).hexdigest(),
            protocol_sha256=protocol_sha256,
            source_file_sha256=source_hashes,
        )
        result.update(
            {
                "run_id": run_id,
                "nonce": nonce,
                "pins": expected_payload["pins"],
            }
        )
        stage = "profile_verification"
        if not modal_smoke._verify_profile(profile, workspace):
            raise RuntimeError("Modal token info did not match the expected personal workspace")
        if importlib.metadata.version("modal") != "1.6.1":
            raise RuntimeError("Modal SDK 1.6.1 is required")

        stage = "modal_setup"
        import modal

        volume = modal.Volume.from_name(VOLUME_NAME)
        root = Path(__file__).resolve().parents[1]
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
        )
        for module_file in (
            "__init__.py",
            "modal_real_pilot.py",
            "real_pilot_core.py",
            "real_pilot_contracts.py",
            "modal_train_rehearsal.py",
            "training_rehearsal_core.py",
            "baseline_qwen.py",
            "modal_smoke.py",
        ):
            image = image.add_local_file(
                str(root / "experiments" / module_file),
                remote_path=f"/root/experiments/{module_file}",
            )
        image = image.add_local_file(
            str(root / core.PROTOCOL_PATH), remote_path=f"/root/{core.PROTOCOL_PATH}"
        )
        app = modal.App("reflex-qwen-real-data-pilot", image=image)
        run_pilot = app.function(
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
            timeout=3600,
            volumes={"/artifacts": volume},
        )(_remote_train)
        stage = "remote_training"
        with modal.enable_output(), app.run():
            remote_started = True
            remote_value = run_pilot.remote(expected_payload)
            remote_returned = True
        try:
            if isinstance(remote_value, dict) and remote_value.get("status") == "failed":
                result = core.validate_failed_receipt(
                    remote_value, expected_payload=expected_payload
                )
            else:
                result = core.validate_passed_receipt(
                    remote_value, expected_payload=expected_payload
                )
            result["provenance"].update(
                {
                    "profile": profile,
                    "workspace": workspace,
                    "modal_sdk_version": "1.6.1",
                    "local_records_sha256": expected_payload["pins"]["records_sha256"],
                    "local_manifest_sha256": expected_payload["pins"]["manifest_sha256"],
                    "local_recipe_sha256": expected_payload["pins"]["recipe_sha256"],
                }
            )
        except Exception as exc:
            result = _initial_receipt(run_id, nonce)
            result["pins"] = expected_payload["pins"]
            result["failure"] = _failure("receipt_validation", exc)
    except Exception as exc:
        result["status"] = "failed"
        result["phase"] = "failed"
        result["failure"] = _failure(stage, exc)
        if (
            remote_started
            and not remote_returned
            and volume is not None
            and expected_payload is not None
        ):
            try:
                progress = modal_train_rehearsal._read_volume_progress(volume, run_id)
                result = core.validate_recovery_receipt(
                    progress,
                    expected_identity={
                        "run_id": expected_payload["run_id"],
                        "nonce": expected_payload["nonce"],
                        "pins": expected_payload["pins"],
                    },
                )
            except Exception:
                pass
    normalized = modal_train_rehearsal._normalize_receipt(result)
    modal_smoke.smoke.write_json_artifact(reservation, normalized)
    print(
        json.dumps(
            {"status": normalized.get("status", "failed"), "artifact": str(reservation.destination)}
        )
    )
    return 0 if normalized.get("status") == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch", action="store_true", help="explicitly launch paid remote training"
    )
    parser.add_argument("--profile", default=PROFILE, help="named personal Modal profile")
    parser.add_argument("--workspace", default=WORKSPACE, help="expected Modal workspace")
    parser.add_argument("--run-id", help="unique safe run ID in the artifact volume")
    parser.add_argument("--records", default="data/processed/real-pilot-v1.jsonl")
    parser.add_argument("--manifest", default="data/pilots/real-pilot-v1-manifest.json")
    parser.add_argument("--recipe", default="data/pilots/real-pilot-v1-recipe.json")
    parser.add_argument("--output", help="new local strict-JSON receipt path")
    args = parser.parse_args(argv)
    if not args.launch:
        print(json.dumps(core.plan(), sort_keys=True, indent=2))
        return 0
    if not args.output or not args.run_id:
        parser.error("--launch requires --run-id and --output")
    try:
        core.validate_run_id(args.run_id)
        with modal_smoke.smoke.reserve_output(args.output) as reservation:
            return _launch(
                profile=args.profile.strip(),
                workspace=args.workspace.strip(),
                run_id=args.run_id,
                records_path=args.records,
                manifest_path=args.manifest,
                recipe_path=args.recipe,
                reservation=reservation,
            )
    except (OSError, ValueError) as exc:
        print(f"output or recipe preflight failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
