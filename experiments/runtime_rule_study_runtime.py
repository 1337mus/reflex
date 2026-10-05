"""Authenticated remote execution for the runtime-rule study."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import importlib.util
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from experiments import mixture_training_runtime_helpers as runtime_helpers
from experiments import runtime_rule_study_contracts as contracts
from experiments import runtime_rule_study_output_evidence as output_api
from experiments import runtime_rule_study_payloads as payload_api
from experiments import runtime_rule_study_progress as progress_api
from experiments import runtime_rule_study_provenance as provenance_api
from experiments import runtime_rule_study_results as results_api
from experiments import runtime_rule_study_runtime_scoring as scoring_api
from experiments import runtime_rule_study_sources as sources_api
from experiments import runtime_rule_study_training_evidence as training_evidence_api
from reflex_decisions.smoke import sanitize_exception_message

ARTIFACT_RUNS_ROOT = Path("/artifacts/runs")
_FORBIDDEN_KERNEL_PACKAGES = ("kernels", "fla", "causal_conv1d")
_VERSION_UNAVAILABLE = re.compile(r"[^A-Za-z0-9_.:+-]")


@dataclass(frozen=True, slots=True)
class _RuntimeModules:
    torch: Any
    peft: Any
    model_class: Any
    tokenizer_class: Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_runtime_modules() -> _RuntimeModules:
    """Import GPU libraries only after payload, source, and package preflight."""

    torch = importlib.import_module("torch")
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")
    return _RuntimeModules(
        torch=torch,
        peft=peft,
        model_class=transformers.AutoModelForCausalLM,
        tokenizer_class=transformers.AutoTokenizer,
    )


def _load_base(
    model_class: Any, tokenizer_class: Any, torch: Any
) -> tuple[Any, Any, dict[str, object]]:
    from experiments.mixture_training_runtime_execution import _base_model

    return _base_model(model_class, tokenizer_class, torch)


def _measure_runtime_versions(result: dict[str, object]) -> dict[str, str]:
    measured: dict[str, str] = {}
    for name in contracts.RUNTIME_VERSION_PINS:
        try:
            measured[name] = importlib.metadata.version(name)
        except Exception as exc:
            marker = f"unavailable:{type(exc).__name__}"
            measured[name] = _VERSION_UNAVAILABLE.sub("_", marker)[:128]
    provenance = cast(dict[str, object], result["provenance"])
    provenance["runtime_versions"] = dict(measured)
    return measured


def _forbidden_kernel_packages() -> dict[str, bool]:
    return {
        name: name in sys.modules or importlib.util.find_spec(name) is not None
        for name in _FORBIDDEN_KERNEL_PACKAGES
    }


def _check_remote_runtime(result: dict[str, object]) -> None:
    os.environ["USE_HUB_KERNELS"] = "NO"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    measured = _measure_runtime_versions(result)
    optional = _forbidden_kernel_packages()
    if any(optional.values()):
        raise RuntimeError("forbidden optional model-kernel packages are installed")
    if measured != contracts.RUNTIME_VERSION_PINS:
        raise RuntimeError("installed runtime package versions differ from the fixed pins")


def _specs_by_category(payload: Mapping[str, object]) -> dict[str, list[dict[str, object]]]:
    return {
        "training": cast(list[dict[str, object]], payload["training_specs"]),
        "final_evaluation": cast(list[dict[str, object]], payload["evaluation_specs"]),
        "reload_parity": cast(list[dict[str, object]], payload["reload_specs"]),
    }


def _set_progress(result: dict[str, object], ledger: progress_api.ForwardLedger) -> None:
    cast(dict[str, object], result["evidence"]).update(ledger.snapshot())


def _failure(stage: str, error: BaseException) -> dict[str, str]:
    safe_stage = stage[:128] or "worker"
    error_type = _VERSION_UNAVAILABLE.sub("_", type(error).__name__)[:128] or "Exception"
    fallback = f"Worker failed during {safe_stage} ({error_type})."
    try:
        sanitized = sanitize_exception_message(error)
        message = sanitized[:2_000] if isinstance(sanitized, str) else fallback
        if not message.strip():
            message = fallback
    except Exception:
        message = fallback
    return {
        "stage": safe_stage,
        "type": error_type,
        "message": message,
    }


def _failed_envelope(stage: str, error: BaseException) -> dict[str, object]:
    return {"status": "failed", "phase": "preflight", "failure": _failure(stage, error)}


def _commit_artifact_volume() -> None:
    modal = importlib.import_module("modal")
    modal.Volume.from_name(contracts.MODAL_VOLUME).commit()


def _write_progress(run_dir: Path, result: dict[str, object]) -> None:
    if run_dir.parent != ARTIFACT_RUNS_ROOT or not run_dir.is_dir():
        raise ValueError("progress path is outside this worker's run directory")
    runtime_helpers.write_json_atomic(run_dir / "progress.json", result, replace=True)
    _commit_artifact_volume()


def _create_run_directory(run_id: object) -> Path:
    safe_run_id = contracts.validate_safe_run_id(run_id)
    ARTIFACT_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = ARTIFACT_RUNS_ROOT / safe_run_id
    run_dir.mkdir(exist_ok=False)
    return run_dir


def _snapshot_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    study = cast(Mapping[str, object], payload["study"])
    selection = cast(Mapping[str, object], study["selection"])
    return cast(dict[str, object], selection["snapshot"])


def _states_equal(left: Mapping[str, Any], right: Mapping[str, Any], torch: Any) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        first, second = left[name], right[name]
        if (
            getattr(first, "dtype", None) != torch.float32
            or getattr(second, "dtype", None) != torch.float32
            or getattr(first, "shape", None) != getattr(second, "shape", None)
            or not bool(torch.equal(first, second))
        ):
            return False
    return True


def _training_forward_count(ledger: progress_api.ForwardLedger) -> int:
    counts = cast(Mapping[str, int], ledger.snapshot()["completed_forward_counts"])
    return counts["training"]


def _finish_training_evidence(
    result: dict[str, object], payload: Mapping[str, object], ledger: progress_api.ForwardLedger
) -> dict[str, object]:
    evidence = cast(dict[str, object], result["evidence"])
    checked = training_evidence_api.validate_training_evidence(
        evidence["training"],
        run_id=payload["run_id"],
        completed_training_forwards=_training_forward_count(ledger),
        passed=True,
    )
    evidence["training"] = checked
    return checked


def _complete_outputs(
    result: dict[str, object],
    payload: Mapping[str, object],
    ledger: progress_api.ForwardLedger,
) -> None:
    evidence = cast(dict[str, object], result["evidence"])
    counts = cast(Mapping[str, int], ledger.snapshot()["completed_forward_counts"])
    output_api.validate_scored_outputs(
        evidence["outputs"],
        presentations=payload["evaluation"],
        specs=payload["evaluation_specs"],
        completed_forwards=counts["final_evaluation"],
        passed=True,
    )


def _complete_reload(
    result: dict[str, object], payload: Mapping[str, object], ledger: progress_api.ForwardLedger
) -> None:
    evidence = cast(dict[str, object], result["evidence"])
    training = cast(Mapping[str, object], evidence["training"])
    counts = cast(Mapping[str, int], ledger.snapshot()["completed_forward_counts"])
    checked = output_api.validate_reload_evidence(
        evidence["reload"],
        presentations=payload["reload"],
        specs=payload["reload_specs"],
        final_outputs=evidence["outputs"],
        final_tensor_sha256=training["final_tensor_sha256"],
        completed_forwards=counts["reload_parity"],
        passed=True,
    )
    evidence["reload"] = checked


def remote_worker(payload: object, expected_payload_sha256: object) -> dict[str, object]:
    """Run one authenticated study role and preserve every observation on failure."""

    stage = "payload_authentication"
    result: dict[str, object] | None = None
    run_dir: Path | None = None
    try:
        trusted_digest = contracts.validate_sha256(
            expected_payload_sha256, "trusted expected payload SHA-256"
        )
        authenticated = payload_api.validate_payload(
            payload, expected_payload_sha256=trusted_digest
        )
        result = results_api.initial_result(authenticated)
        ledger = progress_api.ForwardLedger(
            authenticated["role"], _specs_by_category(authenticated)
        )
        _set_progress(result, ledger)

        stage = "source_preflight"
        expected_sources = sources_api.validate_source_map(authenticated["source_file_sha256"])
        measured_sources = sources_api.measure_sources(_project_root())
        cast(dict[str, object], result["provenance"])["measured_source_file_sha256"] = dict(
            measured_sources
        )
        if measured_sources != expected_sources:
            raise RuntimeError("packaged source files differ from the exact payload pins")

        stage = "runtime_preflight"
        _check_remote_runtime(result)

        stage = "runtime_imports"
        modules = _load_runtime_modules()
        helper_module = importlib.import_module("experiments.modal_train_rehearsal")
        adapter_scoring = importlib.import_module("experiments.mixture_training_runtime_scoring")
        run_training = importlib.import_module("experiments.runtime_rule_study_runtime_training")

        stage = "base_model_load"
        base_model, tokenizer, base_provenance = _load_base(
            modules.model_class, modules.tokenizer_class, modules.torch
        )
        provenance = cast(dict[str, object], result["provenance"])
        provenance["base_model"] = base_provenance
        provenance["cuda_device"] = modules.torch.cuda.get_device_name(0)

        stage = "loaded_provenance_validation"
        provenance_api.validate_provenance(
            provenance, payload=authenticated, passed=False, require_loaded=True
        )

        role = cast(str, authenticated["role"])
        snapshot = _snapshot_from_payload(authenticated)
        snapshot_hashes = adapter_scoring._snapshot_hashes(snapshot)
        if snapshot_hashes != snapshot["files_sha256"]:
            raise ValueError("selected adapter snapshot hashes differ from the payload pins")

        stage = "selected_adapter_load"
        adapted = modules.peft.PeftModel.from_pretrained(
            base_model,
            cast(str, snapshot["path"]),
            is_trainable=role != contracts.ROLE_UNCHANGED,
            autocast_adapter_dtype=True,
        )
        if role == contracts.ROLE_UNCHANGED:
            adapted.eval()
        helper_module._validate_adapter_inventory(
            adapted, modules.torch, require_trainable=role != contracts.ROLE_UNCHANGED
        )
        selected_state = adapter_scoring._verify_saved_adapter(
            adapted,
            snapshot,
            cast(str, cast(Mapping[str, object], authenticated["study"])["selected_tensor_sha256"]),
            modules.peft,
            modules.torch,
        )
        selected_digest = runtime_helpers.tensor_state_sha256(
            selected_state, torch_module=modules.torch
        )
        identity = {
            "snapshot": snapshot,
            "tensor_sha256": selected_digest,
            "adapter_dtype": "torch.float32",
        }
        provenance_api.validate_selected_adapter_identity(
            identity, payload=authenticated, required=True
        )
        evidence = cast(dict[str, object], result["evidence"])
        evidence["selected_adapter_identity"] = identity

        stage = "request_compilation"
        prepared = scoring_api.prepare_requests(authenticated, tokenizer)

        if role == contracts.ROLE_UNCHANGED:
            stage = "final_evaluation"
            result["phase"] = "final_evaluation"
            adapted.eval()
            scoring_api.score_presentations(
                adapted,
                prepared.evaluation_items,
                authenticated["evaluation"],
                modules.torch,
                ledger,
                evidence,
                "final_evaluation",
            )
        else:
            stage = "training_directory"
            run_dir = _create_run_directory(authenticated["run_id"])
            _write_progress(run_dir, result)

            def persist_training_progress(
                progress_result: dict[str, object], callback_run_dir: Path
            ) -> None:
                if callback_run_dir != run_dir:
                    raise ValueError("training progress callback changed the run directory")
                _write_progress(run_dir, progress_result)

            stage = "training"
            final_state = run_training.train_adapter(
                authenticated,
                result,
                adapted,
                prepared,
                selected_state,
                run_dir,
                modules.torch,
                modules.peft,
                ledger,
                on_progress=persist_training_progress,
            )

            stage = "training_evidence_validation"
            training = _finish_training_evidence(result, authenticated, ledger)
            if not isinstance(final_state, Mapping):
                raise TypeError("training helper did not return the final adapter CPU state")
            final_state = cast(Mapping[str, Any], final_state)
            final_digest = runtime_helpers.tensor_state_sha256(
                final_state, torch_module=modules.torch
            )
            if final_digest != training["final_tensor_sha256"]:
                raise ValueError("training helper final state differs from its tensor digest")

            stage = "final_evaluation"
            result["phase"] = "final_evaluation"
            adapted.eval()

            def evaluation_progress(completed: int, _total: int) -> None:
                if completed % 128 == 0:
                    _write_progress(run_dir, result)

            scoring_api.score_presentations(
                adapted,
                prepared.evaluation_items,
                authenticated["evaluation"],
                modules.torch,
                ledger,
                evidence,
                "final_evaluation",
                on_progress=evaluation_progress,
            )
            _complete_outputs(result, authenticated, ledger)

            final_snapshot = cast(
                dict[str, object], cast(Mapping[str, object], training["adapter_paths"])["336"]
            )
            stage = "release_training_runtime"
            del selected_state, adapted, base_model, tokenizer
            gc.collect()
            modules.torch.cuda.empty_cache()

            stage = "reload_base_load"
            fresh_base, fresh_tokenizer, fresh_base_provenance = _load_base(
                modules.model_class, modules.tokenizer_class, modules.torch
            )
            reload_hashes = fresh_base_provenance.get("tokenizer_file_sha256")
            provenance["reload_tokenizer_file_sha256"] = reload_hashes
            provenance_api.validate_provenance(
                provenance,
                payload=authenticated,
                passed=False,
                require_loaded=True,
                require_reload=True,
            )

            stage = "reload_request_compilation"
            reload_prepared = scoring_api.prepare_requests(authenticated, fresh_tokenizer)

            stage = "reload_snapshot_validation"
            adapter_scoring._snapshot_hashes(final_snapshot)
            training_evidence = cast(dict[str, object], evidence["training"])
            final_state_for_compare = final_state

            stage = "reload_adapter_load"
            reloaded = modules.peft.PeftModel.from_pretrained(
                fresh_base,
                cast(str, final_snapshot["path"]),
                is_trainable=False,
                autocast_adapter_dtype=True,
            )
            reloaded.eval()
            helper_module._validate_adapter_inventory(
                reloaded, modules.torch, require_trainable=False
            )
            reload_state = adapter_scoring._state_dict(reloaded, modules.peft)
            reload_digest = runtime_helpers.tensor_state_sha256(
                reload_state, torch_module=modules.torch
            )
            tensor_equal = _states_equal(final_state_for_compare, reload_state, modules.torch)
            reload_evidence = cast(dict[str, object], evidence["reload"])
            reload_evidence["tensor_sha256"] = reload_digest
            reload_evidence["tensor_equal"] = tensor_equal
            if reload_digest != training_evidence["final_tensor_sha256"] or not tensor_equal:
                raise ValueError("reloaded adapter tensors differ from the final trained state")

            adapter_scoring._verify_saved_adapter(
                reloaded,
                final_snapshot,
                training_evidence["final_tensor_sha256"],
                modules.peft,
                modules.torch,
            )
            training_evidence_api.validate_training_evidence(
                training_evidence,
                run_id=authenticated["run_id"],
                completed_training_forwards=_training_forward_count(ledger),
                passed=True,
            )
            _complete_outputs(result, authenticated, ledger)

            stage = "reload"
            result["phase"] = "reload"

            def reload_progress(completed: int, _total: int) -> None:
                if completed % 128 == 0:
                    _write_progress(run_dir, result)

            scoring_api.score_presentations(
                reloaded,
                reload_prepared.reload_items,
                authenticated["reload"],
                modules.torch,
                ledger,
                evidence,
                "reload_parity",
                on_progress=reload_progress,
            )
            _complete_reload(result, authenticated, ledger)

        stage = "final_validation"
        result["phase"] = "finalize"
        ledger.require_complete()
        progress = progress_api.validate_progress(
            ledger.snapshot(),
            role=authenticated["role"],
            specs_by_category=_specs_by_category(authenticated),
            require_complete=True,
        )
        evidence.update(progress)
        _complete_outputs(result, authenticated, ledger)
        if role != contracts.ROLE_UNCHANGED:
            _finish_training_evidence(result, authenticated, ledger)
            _complete_reload(result, authenticated, ledger)
        if run_dir is not None:
            stage = "final_progress_persistence"
            _write_progress(run_dir, result)

        result["failure"] = None
        result["status"] = "passed"
        result["phase"] = "completed"
        stage = "returned_result_validation"
        validated_result = results_api.validate_result(
            result, payload=authenticated, expected_payload_sha256=trusted_digest
        )
        if run_dir is not None:
            stage = "terminal_progress_persistence"
            _write_progress(run_dir, validated_result)
        return validated_result
    except Exception as exc:
        if result is None:
            return _failed_envelope(stage, exc)
        if result.get("phase") == "completed" or stage in {
            "final_validation",
            "final_progress_persistence",
            "returned_result_validation",
            "terminal_progress_persistence",
        }:
            result["phase"] = "finalize"
        result["status"] = "failed"
        result["failure"] = _failure(stage, exc)
        if run_dir is not None:
            try:
                _write_progress(run_dir, result)
            except Exception:
                pass
        return result
