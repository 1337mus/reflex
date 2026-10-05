"""Immutable, label-free payload contracts for the public fresh evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from experiments.adapter_transfer_contracts import (
    SOURCE_FINGERPRINT_PATHS as TRANSFER_SOURCE_FINGERPRINT_PATHS,
)
from experiments.mixture_training_contracts import (
    json_sha256,
    normalize_json_object,
    validate_safe_run_id,
    validate_sha256,
    validate_uuid,
)
from reflex_decisions.schema import DecisionRequest

EXPERIMENT_ID = "fresh-eval-v1"
SCHEMA_VERSION = 1
DATASET_COUNTS = {
    "hans-eval-v1": 300,
    "winogrande-dev-v1": 200,
    "arc-challenge-dev-v1": 200,
}
BASE_PRESENTATION_COUNT = 1400
ADAPTER_PRESENTATION_COUNT = 1400
PARITY_PRESENTATION_COUNT = 24
MAX_TOTAL_FORWARDS = 2824
MAX_INPUT_TOKENS = 2048
MAX_TOTAL_INPUT_TOKENS = 5_783_552
_PRESENTATION_FIELDS = {
    "presentation_id",
    "record_id",
    "dataset_id",
    "source_group_id",
    "request_hash",
    "order_index",
    "order_ids",
    "request",
}
_COMPILED_FIELDS = {
    "presentation_id",
    "request_hash",
    "order_ids",
    "prompt_sha256",
    "input_tokens",
    "candidate_token_ids",
    "input_ids_sha256",
}
_GPU_PIN_FIELDS = {
    "data_file_sha256",
    "dataset_counts",
    "panel_sha256",
    "source_manifest_sha256",
}
_PAYLOAD_FIELDS = {
    "schema_version",
    "experiment_id",
    "run_id",
    "nonce",
    "selection_sha256",
    "selection",
    "pins",
    "presentations",
    "parity_presentations",
    "compiled_requests",
    "payload_sha256",
}
_PIN_FIELDS = {
    "gpu_pins",
    "selection_file_sha256",
    "adapter_tensor_sha256",
    "model_id",
    "model_revision",
    "protocol_sha256",
    "source_file_sha256",
    "tokenizer_file_sha256",
    "tokenizers_version",
}
PROTOCOL_PATH = "docs/fresh-eval-protocol.md"
ADAPTER_TENSOR_SHA256 = "b9ade96b9f6077934985a4b261a6f7400a1e210003b02094b492f36c8a844324"
TOKENIZER_FILE_SHA256 = "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"
TOKENIZERS_VERSION = "0.23.2"
_FRESH_SOURCE_FINGERPRINT_PATHS = (
    "experiments/fresh_eval_core.py",
    "experiments/fresh_eval_runtime.py",
    "experiments/fresh_eval_results.py",
    "experiments/modal_fresh_eval.py",
    "experiments/analyze_fresh_eval.py",
    "experiments/fresh_eval_analysis_validation.py",
    "experiments/prepare_fresh_eval.py",
    "src/reflex_decisions/fresh_eval_data.py",
    "src/reflex_decisions/fresh_eval_build.py",
    "src/reflex_decisions/fresh_eval_bundle.py",
    "experiments/adapter_transfer_contracts.py",
    "experiments/mixture_training_contracts.py",
    "experiments/mixture_training_runtime_execution.py",
    "experiments/mixture_training_runtime_scoring.py",
    "experiments/mixture_training_runtime_helpers.py",
    "experiments/modal_train_rehearsal.py",
    "experiments/modal_smoke.py",
    "src/reflex_decisions/rendering.py",
    "src/reflex_decisions/schema.py",
    "src/reflex_decisions/smoke.py",
    PROTOCOL_PATH,
)
SOURCE_FINGERPRINT_PATHS = tuple(
    dict.fromkeys((*TRANSFER_SOURCE_FINGERPRINT_PATHS, *_FRESH_SOURCE_FINGERPRINT_PATHS))
)


def build_presentations(records: object) -> list[dict[str, object]]:
    """Use the public-data builder so evaluation code never reinterprets records."""

    from reflex_decisions.fresh_eval_data import build_presentations as build

    return build(records)


def validate_presentations(
    value: object, *, expected_counts: Mapping[str, int] | None = None
) -> list[dict[str, object]]:
    """Require the fixed, label-free original/left-rotation presentation panel."""

    counts = dict(DATASET_COUNTS if expected_counts is None else expected_counts)
    if not isinstance(value, list) or len(value) != 2 * sum(counts.values()):
        raise ValueError("fresh evaluation presentations have an unexpected count")
    rows: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != _PRESENTATION_FIELDS:
            raise ValueError("fresh evaluation presentation has an unexpected schema")
        if (
            not isinstance(row["record_id"], str)
            or not isinstance(row["dataset_id"], str)
            or row["dataset_id"] not in counts
            or not isinstance(row["source_group_id"], str)
            or type(row["order_index"]) is not int
            or row["order_index"] not in (0, 1)
            or not isinstance(row["order_ids"], list)
        ):
            raise ValueError("fresh evaluation presentation identity is malformed")
        request = DecisionRequest.model_validate(row["request"])
        if (
            row["request_hash"] != request.request_hash
            or row["order_ids"] != [option.id for option in request.options]
            or row["presentation_id"]
            != f"{EXPERIMENT_ID}:{row['record_id']}:order-{row['order_index']}"
        ):
            raise ValueError("fresh evaluation presentation request identity is inconsistent")
        rows.append(row)
    pairs = list(zip(rows[::2], rows[1::2], strict=True))
    seen: Counter[str] = Counter()
    previous: tuple[str, str] | None = None
    for original, rotated in pairs:
        if (
            original["order_index"] != 0
            or rotated["order_index"] != 1
            or any(
                original[key] != rotated[key]
                for key in ("record_id", "dataset_id", "source_group_id")
            )
        ):
            raise ValueError("fresh evaluation presentations are not paired identity orders")
        first = DecisionRequest.model_validate(original["request"])
        second = DecisionRequest.model_validate(rotated["request"])
        if (
            len(first.options) < 2
            or second.options != first.options[1:] + first.options[:1]
            or original["request_hash"] != rotated["request_hash"]
        ):
            raise ValueError("fresh evaluation order one is not the cyclic rotation")
        order = (str(original["dataset_id"]), str(original["record_id"]))
        if previous is not None and order <= previous:
            raise ValueError("fresh evaluation presentations are not sorted")
        previous = order
        seen[str(original["dataset_id"])] += 1
    if seen != Counter(counts):
        raise ValueError("fresh evaluation presentation datasets differ from the frozen panel")
    return rows


def compile_requests(
    presentations: object, tokenizer: object, *, expected_counts: Mapping[str, int] | None = None
) -> list[dict[str, object]]:
    """Compile every fixed prompt on the host before any GPU worker is eligible."""

    from reflex_decisions.rendering import compile_request

    rows = validate_presentations(presentations, expected_counts=expected_counts)
    compiled_rows: list[dict[str, object]] = []
    for presentation in rows:
        request = DecisionRequest.model_validate(presentation["request"])
        compiled = compile_request(request, tokenizer, max_tokens=2048)
        input_ids = list(compiled.input_ids)
        compiled_rows.append(
            {
                "presentation_id": presentation["presentation_id"],
                "request_hash": compiled.request_hash,
                "order_ids": list(presentation["order_ids"]),
                "prompt_sha256": compiled.prompt_hash,
                "input_tokens": len(input_ids),
                "candidate_token_ids": list(compiled.candidate_token_ids),
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        )
    return compiled_rows


def validate_compiled_requests(
    value: object,
    presentations: object,
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> list[dict[str, object]]:
    """Bind CPU-compiled identities to each immutable GPU request."""

    rows = validate_presentations(presentations, expected_counts=expected_counts)
    if not isinstance(value, list) or len(value) != len(rows):
        raise ValueError("fresh evaluation compiled requests have an unexpected count")
    compiled: list[dict[str, object]] = []
    for item, presentation in zip(value, rows, strict=True):
        if not isinstance(item, dict) or set(item) != _COMPILED_FIELDS:
            raise ValueError("fresh evaluation compiled request has an unexpected schema")
        if any(
            item[key] != presentation[key]
            for key in ("presentation_id", "request_hash", "order_ids")
        ):
            raise ValueError("fresh evaluation compiled request identity is inconsistent")
        if type(item["input_tokens"]) is not int or not 0 < item["input_tokens"] <= 2048:
            raise ValueError("fresh evaluation compiled request token count is invalid")
        if (
            not isinstance(item["candidate_token_ids"], list)
            or len(item["candidate_token_ids"]) != len(presentation["order_ids"])
            or len(set(item["candidate_token_ids"])) != len(item["candidate_token_ids"])
            or any(type(token) is not int or token < 0 for token in item["candidate_token_ids"])
        ):
            raise ValueError("fresh evaluation compiled candidate IDs are invalid")
        for field in ("prompt_sha256", "input_ids_sha256"):
            validate_sha256(item[field], f"compiled request {field}")
        compiled.append(item)
    return compiled


def validate_gpu_pins(value: object) -> dict[str, object]:
    """Allow only aggregate, label-free public-data pins into the GPU payload."""

    if not isinstance(value, dict) or set(value) != _GPU_PIN_FIELDS:
        raise ValueError("fresh evaluation GPU data pins have an unexpected schema")
    hashes = value["data_file_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("fresh evaluation data-file pins are malformed")
    normalized_hashes: dict[str, str] = {}
    for path, digest in hashes.items():
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("fresh evaluation data-file pin path is unsafe")
        normalized_hashes[path] = validate_sha256(digest, f"data-file pin {path}")
    if value["dataset_counts"] != DATASET_COUNTS:
        raise ValueError("fresh evaluation dataset counts differ from the frozen panel")
    return {
        "data_file_sha256": normalized_hashes,
        "dataset_counts": dict(DATASET_COUNTS),
        "panel_sha256": validate_sha256(value["panel_sha256"], "panel SHA-256"),
        "source_manifest_sha256": validate_sha256(
            value["source_manifest_sha256"], "source manifest SHA-256"
        ),
    }


def source_fingerprints(root: str | Path | None = None) -> dict[str, str]:
    """Hash only the transitive, explicit fresh-evaluation source allowlist."""

    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        canonical_root = project_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("fresh evaluation project root is unavailable") from exc
    hashes: dict[str, str] = {}
    for relative in SOURCE_FINGERPRINT_PATHS:
        try:
            path = (project_root / relative).resolve(strict=True)
            path.relative_to(canonical_root)
            if not path.is_file():
                raise ValueError("source path is not a file")
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise ValueError(f"fresh evaluation source is unavailable: {relative}") from exc
    return hashes


def _parity_presentations(presentations: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: set[tuple[str, str]] = set()
    for dataset_id in sorted(DATASET_COUNTS):
        record_ids = [
            str(row["record_id"])
            for row in presentations
            if row["dataset_id"] == dataset_id and row["order_index"] == 0
        ]
        selected.update((dataset_id, record_id) for record_id in record_ids[:4])
    parity = [
        row for row in presentations if (str(row["dataset_id"]), str(row["record_id"])) in selected
    ]
    if len(parity) != 24:
        raise ValueError("fresh evaluation parity presentations are malformed")
    return parity


def _validate_pins(value: object, *, root: str | Path | None) -> dict[str, object]:
    from experiments import adapter_transfer_contracts as contracts

    pins = normalize_json_object(value, "fresh evaluation pins")
    if set(pins) != _PIN_FIELDS:
        raise ValueError("fresh evaluation pins have an unexpected schema")
    sources = pins["source_file_sha256"]
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_FINGERPRINT_PATHS):
        raise ValueError("fresh evaluation source pins have an unexpected schema")
    actual_sources = source_fingerprints(root)
    if dict(sources) != actual_sources:
        raise ValueError("fresh evaluation source pins differ from local source files")
    protocol_sha = validate_sha256(pins["protocol_sha256"], "fresh evaluation protocol SHA-256")
    if protocol_sha != actual_sources[PROTOCOL_PATH]:
        raise ValueError("fresh evaluation protocol pin differs from its source hash")
    if pins["model_id"] != contracts.MODEL_ID or pins["model_revision"] != contracts.MODEL_REVISION:
        raise ValueError("fresh evaluation model pin differs from the selected base model")
    if pins["adapter_tensor_sha256"] != ADAPTER_TENSOR_SHA256:
        raise ValueError("fresh evaluation adapter tensor pin differs from the selected adapter")
    if pins["tokenizer_file_sha256"] != TOKENIZER_FILE_SHA256:
        raise ValueError("fresh evaluation tokenizer file differs from the reviewed cache")
    if pins["tokenizers_version"] != TOKENIZERS_VERSION:
        raise ValueError("fresh evaluation tokenizers version differs from the reviewed cache")
    return {
        "gpu_pins": validate_gpu_pins(pins["gpu_pins"]),
        "selection_file_sha256": validate_sha256(
            pins["selection_file_sha256"], "selection file SHA-256"
        ),
        "adapter_tensor_sha256": ADAPTER_TENSOR_SHA256,
        "model_id": contracts.MODEL_ID,
        "model_revision": contracts.MODEL_REVISION,
        "protocol_sha256": protocol_sha,
        "source_file_sha256": actual_sources,
        "tokenizer_file_sha256": TOKENIZER_FILE_SHA256,
        "tokenizers_version": TOKENIZERS_VERSION,
    }


def build_payload(
    run_id: str,
    nonce: str,
    presentations: object,
    compiled_requests: object,
    selection: object,
    selection_file_sha256: str,
    gpu_pins: object,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Freeze the only label-free input accepted by the remote worker."""

    from experiments import adapter_transfer_contracts as contracts

    validate_safe_run_id(run_id)
    validate_uuid(nonce, "fresh evaluation nonce")
    rows = validate_presentations(presentations)
    compiled = validate_compiled_requests(compiled_requests, rows)
    normalized_selection = contracts.validate_selection(selection)
    sources = source_fingerprints(root)
    pins = {
        "gpu_pins": validate_gpu_pins(gpu_pins),
        "selection_file_sha256": validate_sha256(selection_file_sha256, "selection file SHA-256"),
        "adapter_tensor_sha256": ADAPTER_TENSOR_SHA256,
        "model_id": contracts.MODEL_ID,
        "model_revision": contracts.MODEL_REVISION,
        "protocol_sha256": sources[PROTOCOL_PATH],
        "source_file_sha256": sources,
        "tokenizer_file_sha256": TOKENIZER_FILE_SHA256,
        "tokenizers_version": TOKENIZERS_VERSION,
    }
    if pins["selection_file_sha256"] != contracts.require_selection_pin():
        raise ValueError("fresh evaluation selection file differs from the reviewed pin")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "nonce": nonce,
        "selection_sha256": json_sha256(normalized_selection),
        "selection": normalized_selection,
        "pins": pins,
        "presentations": rows,
        "parity_presentations": _parity_presentations(rows),
        "compiled_requests": compiled,
    }
    return validate_payload({**unsigned, "payload_sha256": json_sha256(unsigned)}, root=root)


def prepare_payload(
    root: str | Path, run_id: str, nonce: str, tokenizer: object
) -> tuple[dict[str, object], dict[str, object]]:
    """Keep complete metadata on the host while freezing only its safe aggregate pins."""

    from experiments import adapter_transfer_contracts as contracts
    from reflex_decisions.fresh_eval_data import load_panel

    records, metadata = load_panel(root)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("gpu_pins"), dict):
        raise ValueError("fresh evaluation data loader did not provide GPU-safe aggregate pins")
    presentations = build_presentations(records)
    compiled = compile_requests(presentations, tokenizer)
    selection, selection_file_sha256 = contracts.load_selection(root)
    return (
        build_payload(
            run_id,
            nonce,
            presentations,
            compiled,
            selection,
            selection_file_sha256,
            metadata["gpu_pins"],
            root=root,
        ),
        metadata,
    )


def validate_payload(value: object, *, root: str | Path | None = None) -> dict[str, object]:
    """Recheck all frozen request identities and host-visible local source pins."""

    from experiments import adapter_transfer_contracts as contracts

    payload = normalize_json_object(value, "fresh evaluation payload")
    if set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("fresh evaluation payload has an unexpected schema")
    if payload["schema_version"] != SCHEMA_VERSION or type(payload["schema_version"]) is not int:
        raise ValueError("fresh evaluation payload schema version is unsupported")
    if payload["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("fresh evaluation payload experiment ID is unsupported")
    validate_safe_run_id(payload["run_id"])
    validate_uuid(payload["nonce"], "fresh evaluation payload nonce")
    selection = contracts.validate_selection(payload["selection"])
    if payload["selection_sha256"] != json_sha256(selection):
        raise ValueError("fresh evaluation selection hash differs from the selection")
    pins = _validate_pins(payload["pins"], root=root)
    if pins["selection_file_sha256"] != contracts.require_selection_pin():
        raise ValueError("fresh evaluation selection file differs from the reviewed pin")
    rows = validate_presentations(payload["presentations"])
    if payload["parity_presentations"] != _parity_presentations(rows):
        raise ValueError("fresh evaluation parity panel differs from the frozen first-four panel")
    compiled = validate_compiled_requests(payload["compiled_requests"], rows)
    digest = validate_sha256(payload["payload_sha256"], "fresh evaluation payload SHA-256")
    unsigned = {key: item for key, item in payload.items() if key != "payload_sha256"}
    if digest != json_sha256(unsigned):
        raise ValueError("fresh evaluation payload SHA-256 differs from canonical fields")
    return {
        **payload,
        "selection": selection,
        "pins": pins,
        "presentations": rows,
        "compiled_requests": compiled,
    }


def validate_result(
    value: object, payload: object, *, root: str | Path | None = None
) -> dict[str, object]:
    """Expose the receipt validator beside the frozen payload contract."""

    from experiments.fresh_eval_results import validate_result as validate

    return validate(value, payload, root=root)
