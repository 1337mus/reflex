"""CPU-safe helpers shared by the mixture training worker and Modal host."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def tensor_state_sha256(state: Mapping[str, Any], *, torch_module: Any) -> str:
    """Bind sorted adapter names, shapes, FP32 dtype, and contiguous CPU bytes."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("adapter tensor state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise ValueError("adapter tensor names must be non-empty strings")
        tensor = state[name]
        if getattr(tensor, "dtype", None) != torch_module.float32:
            raise ValueError("adapter tensor state must already be FP32")
        normalized = tensor.detach().to(device="cpu").contiguous()
        if normalized.dtype != torch_module.float32:
            raise ValueError("adapter tensor state must remain FP32 on CPU")
        if not bool(torch_module.isfinite(normalized).all().item()):
            raise ValueError("adapter tensor state contains a non-finite value")
        shape = list(normalized.shape)
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("adapter tensor shape is malformed")
        metadata = json.dumps(
            {"dtype": "float32", "name": name, "shape": shape},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        raw = normalized.numpy().tobytes(order="C")
        for chunk in (metadata, raw):
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: object, *, replace: bool) -> str:
    """Write canonical JSON with one newline and return its file hash."""
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError("JSON artifact parent directory does not exist")
    encoded = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256(encoded).hexdigest()
    if not replace:
        with destination.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return digest

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_exclusive(path: str | Path, value: object) -> str:
    """Create a canonical JSON artifact without replacing an existing path."""

    return write_json_atomic(path, value, replace=False)


def strict_json_object(raw: bytes) -> dict[str, object]:
    """Parse UTF-8 JSON while rejecting duplicate keys and non-finite constants."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("progress JSON contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"progress JSON contains non-finite value: {value}")

    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("progress artifact is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("progress artifact must be a JSON object")
    return parsed


def failed_worker_result(
    payload: Mapping[str, object],
    *,
    schema_version: int,
    model_id: str,
    model_revision: str,
    stage: str,
    error: BaseException,
    sanitize: Any,
) -> dict[str, object]:
    """Create a trusted-identity failure result with unknown worker counts."""

    pins = payload.get("pins")
    source_hashes = pins.get("source_file_sha256", {}) if isinstance(pins, Mapping) else {}
    provenance: dict[str, object] = {
        "model_id": model_id,
        "model_revision": model_revision,
        "versions": {},
        "source_file_sha256": source_hashes,
        "measured_source_file_sha256": {},
        "base_model": None,
        "cuda_device": None,
    }
    identity = {
        key: payload[key]
        for key in (
            "experiment_id",
            "run_id",
            "nonce",
            "arm",
            "payload_sha256",
        )
    }
    phase = payload.get("phase")
    return {
        "schema_version": schema_version,
        **identity,
        "phase": f"{phase}_failed"[:64],
        "status": "failed",
        "provenance": provenance,
        "evidence": {
            "forward_counts": {
                "base_evaluation": None,
                "training": None,
                "final_evaluation": None,
                "reload_parity": None,
                "total": None,
            },
            "input_token_counts": {
                "base_evaluation": None,
                "training": None,
                "final_evaluation": None,
                "reload_parity": None,
                "total": None,
            },
            "outputs": [],
            "output_artifact": None,
            "initialization": payload.get("initialization"),
        },
        "failure": {
            "stage": stage,
            "type": type(error).__name__,
            "message": sanitize(error),
        },
    }


def increment_completed_forward(
    evidence: dict[str, object], category: str, input_tokens: int
) -> None:
    """Record one returned model forward and its prompt-token exposure."""

    allowed = {"base_evaluation", "training", "final_evaluation", "reload_parity"}
    if category not in allowed:
        raise ValueError("forward category is invalid")
    if type(input_tokens) is not int or not 1 <= input_tokens <= 2048:
        raise ValueError("completed forward token count is outside the pinned limit")
    for field in ("forward_counts", "input_token_counts"):
        values = evidence.get(field)
        if not isinstance(values, dict):
            raise ValueError(f"worker {field} evidence is malformed")
        count = values.get(category)
        total = values.get("total")
        if type(count) is not int or type(total) is not int:
            raise ValueError(f"worker {field} evidence is unknown during execution")
        values[category] = count + 1 if field == "forward_counts" else count + input_tokens
        values["total"] = total + 1 if field == "forward_counts" else total + input_tokens


def clean_partial_update(cleanup: Any, *, original_error: BaseException | None) -> None:
    """Run best-effort cleanup without replacing the original lifecycle failure."""

    try:
        cleanup()
    except Exception:
        if original_error is None:
            raise


def build_worker_result(
    payload: Mapping[str, object],
    *,
    schema_version: int,
    phase: str,
    status: str,
    provenance: Mapping[str, object],
    evidence: Mapping[str, object],
    failure: Mapping[str, object] | None,
) -> dict[str, object]:
    """Assemble the exact JSON result envelope consumed by the CPU validator."""

    if status not in {"passed", "failed"}:
        raise ValueError("worker result status must be passed or failed")
    if (status == "passed") != (failure is None):
        raise ValueError("worker result failure must agree with its status")
    result = {
        "schema_version": schema_version,
        "experiment_id": payload["experiment_id"],
        "run_id": payload["run_id"],
        "nonce": payload["nonce"],
        "phase": phase,
        "arm": payload["arm"],
        "payload_sha256": payload["payload_sha256"],
        "status": status,
        "provenance": dict(provenance),
        "evidence": dict(evidence),
        "failure": dict(failure) if failure is not None else None,
    }
    return result
