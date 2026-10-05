"""Plan or explicitly launch the bounded COPA development baseline comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

from experiments import modal_smoke
from experiments.baseline_runner_core import (
    EXPECTED_AUXILIARY_FORWARD_COUNTS,
    MODELS,
    PRESENTATIONS_PER_MODEL,
    RECORD_COUNT,
    validate_adapter_provenance,
)
from experiments.baseline_runner_core import (
    comparison_passed as _comparison_passed,
)
from experiments.baseline_runner_core import (
    plan as _plan,
)
from experiments.baseline_runner_core import (
    presentations as _presentations,
)
from experiments.baseline_runner_core import (
    serialize_remote_records as _serialize_remote_records,
)
from experiments.baseline_runner_core import (
    validate_presentation_result as _validated_presentation_result,
)
from reflex_decisions import baseline_data, smoke
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest

DEFAULT_RECORDS = "data/processed/copa-dev-pilot-v1.jsonl"
DEFAULT_MANIFEST = "data/baselines/copa-dev-manifest.json"
EXPECTED_PROTOCOL = "docs/baseline-protocol.md"


def _remote_model_run(payload: dict[str, object]) -> dict[str, object]:
    model_name = payload.get("model_name")
    receipt: dict[str, object] = {
        "model_name": model_name,
        "status": "failed",
        "provenance": {},
        "presentations": [],
    }
    if model_name not in MODELS:
        receipt["failure"] = {
            "stage": "model_selection",
            "type": "UnknownModel",
            "message": "only the three pinned model names are accepted",
        }
        return receipt

    stage = "payload_validation"
    provenance: dict[str, object] | None = None
    try:
        if set(payload) != {"model_name", "records"}:
            raise ValueError("remote payload must contain only model_name and records")
        rows = payload["records"]
        if not isinstance(rows, list) or len(rows) != RECORD_COUNT:
            raise ValueError("remote payload must contain exactly 32 records")
        if any(set(row) != {"record_id", "request"} for row in rows):
            raise ValueError("remote records may contain only record_id and request")
        presentations = _presentations(rows)
        record_ids = [str(row["record_id"]) for row in rows]
        if len(set(record_ids)) != RECORD_COUNT:
            raise ValueError("remote record IDs must be unique")
        for row in rows:
            request = DecisionRequest.model_validate(row["request"])
            if len(request.options) != 2 or [option.id for option in request.options] != [
                "choice1",
                "choice2",
            ]:
                raise ValueError("remote COPA requests must preserve two original option IDs")

        stage = "runtime_check"
        os.environ["USE_HUB_KERNELS"] = "NO"
        optional = {
            name: name in sys.modules or importlib.util.find_spec(name) is not None
            for name in ("kernels", "fla", "causal_conv1d")
        }
        if any(optional.values()):
            raise RuntimeError("optional model-kernel packages must not be installed")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the Modal worker")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(0)

        stage = "model_load"
        module_path = {
            "qwen": "experiments.baseline_qwen",
            "intern": "experiments.baseline_intern",
            "kev": "experiments.baseline_kev",
        }[model_name]
        adapter = importlib.import_module(module_path)
        scorer, raw_provenance = adapter.load_scorer()
        provenance = validate_adapter_provenance(model_name, adapter, raw_provenance)
        provenance.update(
            {
                "runtime_versions": {
                    package: importlib.metadata.version(package)
                    for package in (
                        "torch",
                        "torchvision",
                        "transformers",
                        "peft",
                        "Pillow",
                        "pydantic",
                        "huggingface-hub",
                        "tokenizers",
                        "safetensors",
                    )
                },
                "use_hub_kernels": os.environ["USE_HUB_KERNELS"],
                "optional_kernel_packages_present": optional,
            }
        )

        stage = "inference"
        for presentation in presentations:
            request = DecisionRequest.model_validate(presentation["request"])
            with torch.inference_mode():
                scored = scorer(request)
            receipt["presentations"].append(_validated_presentation_result(presentation, scored))
        auxiliary_forward_count = provenance.get("auxiliary_forward_count")
        if (
            type(auxiliary_forward_count) is not int
            or auxiliary_forward_count != EXPECTED_AUXILIARY_FORWARD_COUNTS[model_name]
        ):
            raise RuntimeError("adapter auxiliary forward count does not match its pinned path")
        if model_name == "intern" and provenance.get("calibration_parity_passed") is not True:
            raise RuntimeError("Intern calibration parity did not complete successfully")
        scored_presentation_count = len(receipt["presentations"])
        if scored_presentation_count != PRESENTATIONS_PER_MODEL:
            raise RuntimeError("model scorer did not complete all 64 presentations")
        receipt["scored_presentation_count"] = scored_presentation_count
        receipt["auxiliary_forward_count"] = auxiliary_forward_count
        receipt["total_forward_count"] = scored_presentation_count + auxiliary_forward_count
        receipt["max_gpu_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
        receipt["provenance"] = provenance
        receipt["status"] = "passed"
    except Exception as exc:
        if provenance is not None:
            receipt["provenance"] = provenance
        receipt["failure"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": smoke.sanitize_exception_message(exc),
        }
        if "torch" in locals():
            try:
                receipt["max_gpu_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
            except Exception:
                pass
    return receipt


def _sha256_file(path: str | Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    package_roots = (root / "experiments", root / "src" / "reflex_decisions")
    sources = sorted(path for package in package_roots for path in package.rglob("*.py"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def _initial_result(
    records_sha256: str | None,
    manifest_sha256: str | None,
    *,
    profile: str | None = None,
    workspace: str | None = None,
) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    protocol_sha256 = _sha256_file(root / EXPECTED_PROTOCOL)
    return {
        "schema_version": 1,
        "status": "failed",
        "records_sha256": records_sha256,
        "manifest_sha256": manifest_sha256,
        "models": {
            name: {"status": "not_started", "provenance": {}, "presentations": []}
            for name in MODELS
        },
        "provenance": {
            "baseline_protocol": EXPECTED_PROTOCOL,
            "baseline_protocol_sha256": protocol_sha256,
            "source_file_sha256": _source_fingerprints(),
            "profile": profile,
            "workspace": workspace,
        },
        "limits": _plan()["modal"],
    }


def _fail_result(result: dict[str, object], stage: str, exc: BaseException) -> dict[str, object]:
    result["failure"] = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": smoke.sanitize_exception_message(exc),
    }
    return result


def _failed_model(name: str, error: BaseException, *, stage: str) -> dict[str, object]:
    return {
        "model_name": name,
        "status": "failed",
        "provenance": {},
        "presentations": [],
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": smoke.sanitize_exception_message(error),
        },
    }


def _launch(
    profile: str,
    workspace: str,
    records_path: str,
    manifest_path: str,
    records: tuple[DecisionRecord, ...],
    reservation: smoke.ArtifactReservation,
) -> int:
    records_sha256 = _sha256_file(records_path)
    manifest_sha256 = _sha256_file(manifest_path)
    result = _initial_result(records_sha256, manifest_sha256, profile=profile, workspace=workspace)
    if any(os.environ.get(name, "") for name in modal_smoke.CREDENTIAL_OVERRIDES):
        _fail_result(
            result,
            "environment",
            RuntimeError("Modal credential environment overrides are not accepted"),
        )
        smoke.write_json_artifact(reservation, result)
        return 1
    if not modal_smoke._verify_profile(profile, workspace):
        _fail_result(
            result,
            "auth",
            RuntimeError("Modal token info did not match the expected workspace"),
        )
        smoke.write_json_artifact(reservation, result)
        return 1

    try:
        modal_version = importlib.metadata.version("modal")
    except importlib.metadata.PackageNotFoundError as exc:
        _fail_result(result, "modal_sdk", exc)
        smoke.write_json_artifact(reservation, result)
        return 1
    if modal_version != "1.6.1":
        _fail_result(
            result,
            "modal_sdk",
            RuntimeError("Modal 1.6.1 is required"),
        )
        smoke.write_json_artifact(reservation, result)
        return 1

    result["provenance"]["modal_sdk_version"] = modal_version
    try:
        modal = importlib.import_module("modal")
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
            .add_local_python_source("experiments", copy=True)
        )
        app = modal.App("reflex-copa-development-baselines", image=image)
        run_model = app.function(
            gpu="A10",
            cpu=(2.0, 2.0),
            memory=(16384, 16384),
            max_containers=3,
            min_containers=0,
            buffer_containers=0,
            scaledown_window=2,
            retries=0,
            single_use_containers=True,
            serialized=True,
            include_source=False,
            startup_timeout=300,
            timeout=900,
        )(_remote_model_run)
        remote_records = _serialize_remote_records(records)
        payloads = [{"model_name": name, "records": remote_records} for name in MODELS]
        receipts: dict[str, object] = {}
        lifecycle_error: Exception | None = None
        try:
            with modal.enable_output(), app.run():
                mapped = run_model.map(payloads, return_exceptions=True)
                for index, mapped_result in enumerate(mapped):
                    if index >= len(MODELS):
                        break
                    name = MODELS[index]
                    if isinstance(mapped_result, Exception):
                        receipts[name] = _failed_model(name, mapped_result, stage="modal_map")
                    elif (
                        isinstance(mapped_result, dict) and mapped_result.get("model_name") == name
                    ):
                        receipts[name] = mapped_result
                    else:
                        receipts[name] = _failed_model(
                            name,
                            RuntimeError("Modal returned an unexpected model receipt"),
                            stage="result_transfer",
                        )
        except Exception as exc:
            lifecycle_error = exc
            for name in MODELS:
                if name not in receipts:
                    receipts[name] = _failed_model(name, exc, stage="modal_map")

        for name in MODELS:
            receipts.setdefault(
                name,
                _failed_model(
                    name,
                    RuntimeError("Modal did not return a model receipt"),
                    stage="result_transfer",
                ),
            )
        result["models"] = receipts
        record_ids = [record.record_id for record in records]
        passed = _comparison_passed(
            receipts,
            record_ids,
            lifecycle_ok=lifecycle_error is None,
        )
        result["status"] = "passed" if passed else "failed"
        if lifecycle_error is not None:
            _fail_result(result, "modal_lifecycle", lifecycle_error)
        elif not passed:
            result["failure"] = {
                "stage": "comparison",
                "type": "IncompleteModelReceipts",
                "message": "all three models must return all 64 valid presentations",
            }
    except Exception as exc:
        _fail_result(result, "modal_setup", exc)

    smoke.write_json_artifact(reservation, result)
    print(json.dumps({"status": result["status"], "artifact": str(reservation.destination)}))
    return 0 if result["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="explicitly launch paid inference")
    parser.add_argument("--profile", default="reflex-personal", help="named Modal profile")
    parser.add_argument("--workspace", default="rajath-61258", help="expected Modal Workspace")
    parser.add_argument("--records", default=DEFAULT_RECORDS, help="prepared COPA record JSONL")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="prepared COPA manifest")
    parser.add_argument("--output", help="new local JSON artifact path")
    args = parser.parse_args(argv)
    if not args.launch:
        print(json.dumps(_plan(), sort_keys=True, indent=2))
        return 0
    if not args.output:
        parser.error("--launch requires --output")
    try:
        with smoke.reserve_output(args.output) as reservation:
            records_sha256 = _sha256_file(args.records)
            manifest_sha256 = _sha256_file(args.manifest)
            result = _initial_result(
                records_sha256,
                manifest_sha256,
                profile=args.profile.strip(),
                workspace=args.workspace.strip(),
            )
            if result["provenance"]["baseline_protocol_sha256"] is None:
                _fail_result(
                    result,
                    "protocol_preflight",
                    RuntimeError("required baseline protocol document is missing or unreadable"),
                )
                smoke.write_json_artifact(reservation, result)
                print(json.dumps({"status": "failed", "artifact": str(reservation.destination)}))
                return 1
            try:
                _, records = baseline_data.verify_prepared_data(args.records, args.manifest)
            except Exception as exc:
                _fail_result(result, "input_preflight", exc)
                smoke.write_json_artifact(reservation, result)
                print(json.dumps({"status": "failed", "artifact": str(reservation.destination)}))
                return 1
            return _launch(
                args.profile.strip(),
                args.workspace.strip(),
                args.records,
                args.manifest,
                records,
                reservation,
            )
    except (OSError, ValueError) as exc:
        print(f"output preflight failed: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
