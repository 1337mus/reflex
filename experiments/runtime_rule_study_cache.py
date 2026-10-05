"""Verified reuse of historical outputs for the runtime-rule retention panel."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from experiments import (
    adapter_transfer_contracts,
    adapter_transfer_results,
    mixture_training_contracts,
    mixture_training_data,
    mixture_training_results,
    runtime_rule_study_contracts,
    runtime_rule_study_inputs,
    runtime_rule_study_tokens,
)
from experiments import runtime_rule_study_data as study
from experiments.mixture_training_contracts import MAX_INPUT_TOKENS, strict_json_loads
from experiments.mixture_training_outputs import _validate_output_rows
from reflex_decisions.rendering import render_prompt
from reflex_decisions.schema import DecisionRequest

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NATURAL_RECEIPT_PATH = "artifacts/natural-reasoning-2026-10-04-r1-receipt.json"
NATURAL_RECEIPT_SHA256 = "d71ed1d77b46d19022850736c88d9ec3528d4dbfb5d9131a8a01b900454aaf67"
TRANSFER_RECEIPT_PATH = "artifacts/adapter-transfer-2026-10-05-r1-receipt.json"
TRANSFER_RECEIPT_SHA256 = "8797c697afe40104a45dfc73a449f2c79b2118899febf6809ca951669a27515d"
SELECTED_TENSOR_SHA256 = "b9ade96b9f6077934985a4b261a6f7400a1e210003b02094b492f36c8a844324"
_TRANSFER_RUN_ID = "adapter-transfer-2026-10-05-r1"
_OLD_RETENTION_DATASETS = frozenset(tuple(runtime_rule_study_tokens.RETENTION_COUNTS)[:5])
_OLD_RETENTION_COUNT = 3_654
_RETENTION_COUNT = 3_782
_NEW_DEVELOPMENT_COUNT = 164
_COMPILATION_FIELDS = {
    "training",
    "evaluation",
    "unchanged",
    "reload",
    "schedule_sha256",
    "counts",
    "input_token_counts",
    "max_input_tokens",
    "compilation_sha256",
}
_BASE_MODEL_FIELDS = {
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
    "device",
    "attention_implementation",
    "use_kernels",
    "use_hub_kernels",
    "versions",
}


@dataclass(frozen=True, slots=True)
class CachedRetention:
    outputs: tuple[dict[str, object], ...]
    receipt_file_sha256: dict[str, str]
    outputs_sha256: str
    selection: dict[str, object]
    tensor_sha256: str
    tokenizer_file_sha256: dict[str, str]
    runtime_versions: dict[str, str]
    base_model: dict[str, object]
    source_file_sha256: dict[str, str]
    compilation_sha256: str
    cache_sha256: str


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_pinned_json(
    root: Path, relative: str, expected_sha256: str
) -> tuple[Mapping[str, object], str]:
    _sha256(expected_sha256, f"{relative} expected SHA-256")
    try:
        canonical_root = root.resolve(strict=True)
        resolved = (canonical_root / relative).resolve(strict=True)
        resolved.relative_to(canonical_root)
        if not canonical_root.is_dir() or not resolved.is_file():
            raise ValueError("pinned path is not a regular file inside the project root")
        raw = resolved.read_bytes()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"pinned receipt is unavailable inside the project root: {relative}"
        ) from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"pinned receipt SHA-256 mismatch: {relative}")
    parsed = strict_json_loads(raw)
    return _object(parsed, f"{relative} receipt"), digest


def _mapping_field(value: object, field: str, label: str) -> Mapping[str, object]:
    return _object(_object(value, label).get(field), f"{label}.{field}")


def _validate_compilation(
    value: object, inputs: runtime_rule_study_inputs.StudyInputs
) -> tuple[list[Mapping[str, object]], str]:
    compilation = _object(value, "compilation")
    if set(compilation) != _COMPILATION_FIELDS:
        raise ValueError("compilation has an unexpected schema")
    digest = _sha256(compilation.get("compilation_sha256"), "compilation SHA-256")
    unsigned = {key: item for key, item in compilation.items() if key != "compilation_sha256"}
    if study._digest(unsigned) != digest:
        raise ValueError("compilation SHA-256 does not match its JSON-safe fields")
    for role in runtime_rule_study_contracts.ROLES:
        runtime_rule_study_contracts.expected_token_counts(role, compilation)
    expected_presentations = (*inputs.retention_presentations, *inputs.new_presentations)
    if (
        len(inputs.retention_presentations) != _RETENTION_COUNT
        or len(inputs.new_presentations) != _NEW_DEVELOPMENT_COUNT
        or Counter(row.get("dataset_id") for row in inputs.retention_presentations)
        != Counter(runtime_rule_study_tokens.RETENTION_COUNTS)
    ):
        raise ValueError("StudyInputs does not contain the exact fixed evaluation panels")
    evaluation_value = compilation.get("evaluation")
    if not isinstance(evaluation_value, list) or len(evaluation_value) != len(
        expected_presentations
    ):
        raise ValueError("compiled evaluation does not contain the exact frozen panel")
    seen: set[str] = set()
    for index, (expected, compiled) in enumerate(
        zip(expected_presentations, evaluation_value, strict=True)
    ):
        if not isinstance(expected, Mapping) or not isinstance(compiled, Mapping):
            raise ValueError(f"compiled evaluation row {index} must be an object")
        presentation_id = expected.get("presentation_id")
        record_id = expected.get("record_id")
        if (
            not isinstance(presentation_id, str)
            or not presentation_id
            or not isinstance(record_id, str)
            or not record_id
            or presentation_id in seen
        ):
            raise ValueError("StudyInputs evaluation presentation IDs must be unique strings")
        seen.add(presentation_id)
        if type(expected.get("order_index")) is not int:
            raise ValueError("StudyInputs order_index must be a strict integer")
        if (
            compiled.get("presentation_id") != presentation_id
            or compiled.get("record_id") != record_id
        ):
            raise ValueError("compiled evaluation identities differ from StudyInputs order")
        compiled_order = compiled.get("order_index", expected["order_index"])
        if type(compiled_order) is not int or compiled_order != expected["order_index"]:
            raise ValueError("compiled evaluation order_index differs from StudyInputs")
        request = DecisionRequest.model_validate(expected.get("request"))
        if compiled.get("request_hash") != request.request_hash:
            raise ValueError("compiled request hash differs from StudyInputs")
        if _sha256(
            compiled.get("request_json_sha256"), "compiled request SHA-256"
        ) != study._digest(request.model_dump(mode="json")):
            raise ValueError("compiled ordered request differs from StudyInputs")
        _sha256(compiled.get("schema_hash"), "compiled schema SHA-256")
        _sha256(compiled.get("input_ids_sha256"), "compiled token IDs SHA-256")
        prompt_sha256 = _sha256(compiled.get("prompt_sha256"), "compiled prompt SHA-256")
        expected_prompt_sha256 = hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest()
        if prompt_sha256 != expected_prompt_sha256:
            raise ValueError("compiled prompt hash differs from the rendered StudyInputs request")
        token_count = compiled.get("input_tokens")
        if type(token_count) is not int or not 1 <= token_count <= MAX_INPUT_TOKENS:
            raise ValueError("compiled evaluation token count must be a strict positive integer")
    maximum = compilation.get("max_input_tokens")
    if type(maximum) is not int or not 1 <= maximum <= MAX_INPUT_TOKENS:
        raise ValueError("compilation maximum input token count is invalid")
    return evaluation_value, digest


def _validate_source_map(
    value: object,
    paths: Sequence[str],
    verified_files: Mapping[str, str],
    label: str,
) -> dict[str, str]:
    source_map = _object(value, label)
    if set(source_map) != set(paths):
        raise ValueError(f"{label} does not contain its exact historical source allowlist")
    verified: dict[str, str] = {}
    for path, digest_value in source_map.items():
        digest = _sha256(digest_value, f"{label} {path} SHA-256")
        if verified_files.get(path) != digest:
            raise ValueError(f"{label} differs from verified StudyInputs source pins: {path}")
        verified[path] = digest
    return verified


def _validate_receipt_sources(
    result: Mapping[str, object],
    payload: Mapping[str, object],
    paths: Sequence[str],
    verified_files: Mapping[str, str],
    label: str,
) -> dict[str, str]:
    pins = _mapping_field(payload, "pins", f"{label} payload")
    declared_pins = _validate_source_map(
        pins.get("source_file_sha256"), paths, verified_files, f"{label} payload source_file_sha256"
    )
    provenance = _mapping_field(result, "provenance", f"{label} result")
    declared = _validate_source_map(
        provenance.get("source_file_sha256"),
        paths,
        verified_files,
        f"{label} source_file_sha256",
    )
    measured = _validate_source_map(
        provenance.get("measured_source_file_sha256"),
        paths,
        verified_files,
        f"{label} measured_source_file_sha256",
    )
    if declared_pins != declared or declared != measured:
        raise ValueError(f"{label} declared and measured source maps disagree")
    return declared


def _validate_receipt_provenance(
    natural: Mapping[str, object],
    natural_payload: Mapping[str, object],
    transfer: Mapping[str, object],
    transfer_payload: Mapping[str, object],
    inputs: runtime_rule_study_inputs.StudyInputs,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, object]]:
    verified_files = _object(inputs.file_sha256, "StudyInputs file_sha256")
    verified_source_hashes = {
        path: _sha256(digest, f"StudyInputs source {path} SHA-256")
        for path, digest in verified_files.items()
    }
    natural_sources = _validate_receipt_sources(
        natural,
        natural_payload,
        mixture_training_contracts.SOURCE_FINGERPRINT_PATHS,
        verified_source_hashes,
        "natural-reasoning",
    )
    transfer_sources = _validate_receipt_sources(
        transfer,
        transfer_payload,
        adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS,
        verified_source_hashes,
        "adapter-transfer",
    )
    if not set(natural_sources).issubset(transfer_sources):
        raise ValueError("natural-reasoning source pins are not a subset of adapter-transfer pins")
    if any(transfer_sources[path] != digest for path, digest in natural_sources.items()):
        raise ValueError("natural-reasoning and adapter-transfer historical source hashes differ")

    natural_provenance = _mapping_field(natural, "provenance", "natural result")
    transfer_provenance = _mapping_field(transfer, "provenance", "transfer result")
    for provenance in (natural_provenance, transfer_provenance):
        if (
            provenance.get("model_id") != mixture_training_contracts.MODEL_ID
            or provenance.get("model_revision") != mixture_training_contracts.MODEL_REVISION
        ):
            raise ValueError("historical receipts do not match the pinned model revision")
    natural_versions = mixture_training_contracts.validate_runtime_versions(
        natural_provenance.get("versions")
    )
    transfer_versions = mixture_training_contracts.validate_runtime_versions(
        transfer_provenance.get("versions")
    )
    if natural_versions != transfer_versions:
        raise ValueError("historical receipts use different nine-package runtimes")

    natural_base = _mapping_field(natural_provenance, "base_model", "natural provenance")
    transfer_base = _mapping_field(transfer_provenance, "base_model", "transfer provenance")
    if set(natural_base) != _BASE_MODEL_FIELDS or dict(natural_base) != dict(transfer_base):
        raise ValueError("historical receipts do not share complete base-model provenance")
    if (
        natural_base.get("model_id") != mixture_training_contracts.MODEL_ID
        or natural_base.get("model_revision") != mixture_training_contracts.MODEL_REVISION
        or natural_base.get("effective_dtype") != "torch.bfloat16"
        or natural_base.get("attention_implementation") != "eager"
        or natural_base.get("use_kernels") is not False
        or natural_base.get("use_hub_kernels") != "NO"
        or natural_base.get("layer_count") != 24
        or natural_base.get("tied_embeddings") is not True
        or natural_base.get("no_meta_parameters") is not True
    ):
        raise ValueError("historical base provenance differs from the frozen BF16/eager Qwen model")
    base_versions = _object(natural_base.get("versions"), "base model versions")
    expected_base_versions = {
        key: version
        for key, version in mixture_training_contracts.RUNTIME_VERSION_PINS.items()
        if key != "Pillow"
    }
    if dict(base_versions) != expected_base_versions:
        raise ValueError("historical base package versions differ from the frozen pins")
    tokenizer_files = _object(
        natural_base.get("tokenizer_file_sha256"), "base tokenizer_file_sha256"
    )
    if not tokenizer_files or any(
        not isinstance(name, str) or not name or _SHA256.fullmatch(str(digest)) is None
        for name, digest in tokenizer_files.items()
    ):
        raise ValueError("historical tokenizer file hashes are malformed")
    normalized_tokenizers = {name: str(digest) for name, digest in tokenizer_files.items()}
    if normalized_tokenizers.get("tokenizer.json") != runtime_rule_study_tokens.TOKENIZER_SHA256:
        raise ValueError("historical tokenizer.json differs from the CPU compiler pin")
    reload_tokenizers = _object(
        transfer_provenance.get("reload_tokenizer_file_sha256"),
        "transfer reload tokenizer_file_sha256",
    )
    if dict(reload_tokenizers) != normalized_tokenizers:
        raise ValueError(
            "transfer reload tokenizer files differ from the pinned full tokenizer map"
        )
    return natural_sources, transfer_sources, normalized_tokenizers, dict(natural_base)


def _validate_receipt_identities(
    natural: Mapping[str, object],
    transfer: Mapping[str, object],
    transfer_payload: Mapping[str, object],
    selection: Mapping[str, object],
    natural_receipt_sha256: str,
) -> str:
    selected = adapter_transfer_contracts.validate_selection(selection)
    if selected.get("training_receipt_sha256") != natural_receipt_sha256:
        raise ValueError("selected adapter does not bind the pinned natural-reasoning receipt")
    natural_evidence = _mapping_field(natural, "evidence", "natural result")
    training_run_id = selected.get("training_run_id")
    arm = selected.get("arm")
    update = selected.get("update")
    if not isinstance(training_run_id, str) or not isinstance(arm, str):
        raise ValueError("selected adapter training identity is malformed")
    run_suffix = mixture_training_data.RUN_SUFFIXES.get(arm)
    snapshot = _mapping_field(selected, "snapshot", "selection")
    snapshot_path = snapshot.get("path")
    if run_suffix is None or not isinstance(snapshot_path, str):
        raise ValueError("selected adapter training identity is unsupported")
    expected_run_id = f"{training_run_id}{run_suffix}"
    if Path(snapshot_path).parent.name != expected_run_id:
        raise ValueError("selected adapter snapshot run ID differs from its training identity")
    if (
        natural.get("status") != "passed"
        or natural.get("phase") != "train"
        or natural.get("run_id") != expected_run_id
        or natural.get("arm") != arm
        or type(natural_evidence.get("optimizer_updates_completed")) is not int
        or type(update) is not int
        or natural_evidence["optimizer_updates_completed"] != update
    ):
        raise ValueError("natural-reasoning receipt is not the passed SNLI update-378 execution")
    paths = _mapping_field(natural_evidence, "adapter_paths", "natural evidence")
    if paths.get(str(update)) != snapshot:
        raise ValueError("natural final adapter snapshot differs from the selected descriptor")

    validated_payload_selection = adapter_transfer_contracts.validate_selection(
        transfer_payload.get("selection")
    )
    if validated_payload_selection != dict(selection):
        raise ValueError("transfer payload selection differs from the selected descriptor")
    transfer_evidence = _mapping_field(transfer, "evidence", "transfer result")
    if (
        transfer.get("status") != "passed"
        or transfer.get("phase") != "completed"
        or transfer.get("experiment_id") != adapter_transfer_contracts.EXPERIMENT_ID
        or transfer.get("run_id") != _TRANSFER_RUN_ID
    ):
        raise ValueError("adapter-transfer receipt is not the passed pinned execution")
    identity = _mapping_field(transfer_evidence, "adapter_identity", "transfer evidence")
    snapshot = _mapping_field(selection, "snapshot", "selection")
    if (
        identity.get("snapshot_path") != snapshot.get("path")
        or identity.get("files_sha256") != snapshot.get("files_sha256")
        or identity.get("dtype") != "torch.float32"
        or identity.get("tensor_sha256") != SELECTED_TENSOR_SHA256
        or identity.get("reloaded_tensor_sha256") != SELECTED_TENSOR_SHA256
    ):
        raise ValueError("transfer adapter identity differs from the selected snapshot and tensor")
    return SELECTED_TENSOR_SHA256


def _retained_outputs(
    natural: Mapping[str, object], transfer: Mapping[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    natural_evidence = _mapping_field(natural, "evidence", "natural result")
    natural_outputs = natural_evidence.get("outputs")
    if not isinstance(natural_outputs, list):
        raise ValueError("natural-reasoning outputs must be a JSON array")
    old = [
        row
        for row in natural_outputs
        if isinstance(row, Mapping) and row.get("dataset_id") in _OLD_RETENTION_DATASETS
    ]
    if len(old) != _OLD_RETENTION_COUNT:
        raise ValueError("natural receipt does not contain all 3,654 retained development outputs")
    transfer_evidence = _mapping_field(transfer, "evidence", "transfer result")
    outputs = _mapping_field(transfer_evidence, "outputs", "transfer evidence")
    adapter = outputs.get("adapter")
    if not isinstance(adapter, list) or len(adapter) != 128:
        raise ValueError("transfer receipt must contain all 128 selected-adapter outputs")
    return [dict(row) for row in old], [dict(row) for row in adapter]


def _cache_payload(cache: CachedRetention) -> dict[str, object]:
    return {
        "outputs": [dict(row) for row in cache.outputs],
        "receipt_file_sha256": dict(cache.receipt_file_sha256),
        "outputs_sha256": cache.outputs_sha256,
        "selection": dict(cache.selection),
        "tensor_sha256": cache.tensor_sha256,
        "tokenizer_file_sha256": dict(cache.tokenizer_file_sha256),
        "runtime_versions": dict(cache.runtime_versions),
        "base_model": dict(cache.base_model),
        "source_file_sha256": dict(cache.source_file_sha256),
        "compilation_sha256": cache.compilation_sha256,
    }


def _join_retention_rows(
    presentations: Sequence[Mapping[str, object]],
    cached_rows: object,
    compiled_rows: object,
) -> tuple[dict[str, object], ...]:
    """Join cached scores to the exact ordered prompt compilation."""

    if not isinstance(presentations, Sequence) or isinstance(presentations, (str, bytes)):
        raise ValueError("retention presentations must be a sequence")
    if not isinstance(cached_rows, list) or len(cached_rows) != len(presentations):
        raise ValueError("cached outputs must contain every retention presentation")
    if not isinstance(compiled_rows, list) or len(compiled_rows) != len(presentations):
        raise ValueError("compiled evaluation must contain every retention presentation")

    presentation_ids: set[str] = set()
    for presentation in presentations:
        if not isinstance(presentation, Mapping):
            raise ValueError("retention presentation must be an object")
        presentation_id = presentation.get("presentation_id")
        record_id = presentation.get("record_id")
        if (
            not isinstance(presentation_id, str)
            or not presentation_id
            or not isinstance(record_id, str)
            or not record_id
        ):
            raise ValueError("retention presentation identity must use nonblank strings")
        if presentation_id in presentation_ids:
            raise ValueError("retention presentation IDs must be unique")
        presentation_ids.add(presentation_id)
        if type(presentation.get("order_index")) is not int:
            raise ValueError("retention order_index must be an integer")

    if any(
        not isinstance(row, Mapping) or type(row.get("order_index")) is not int
        for row in cached_rows
    ):
        raise ValueError("output order_index must be an integer")

    normalized = _validate_output_rows(
        cached_rows, presentations, label="cached retention outputs", require_complete=True
    )
    for index, (presentation, output, compiled) in enumerate(
        zip(presentations, normalized, compiled_rows, strict=True)
    ):
        if not isinstance(compiled, Mapping):
            raise ValueError(f"compiled retention row {index} must be an object")
        if (
            compiled.get("presentation_id") != presentation["presentation_id"]
            or compiled.get("record_id") != presentation["record_id"]
        ):
            raise ValueError("compiled retention identities do not match panel order")
        compiled_order = compiled.get("order_index", presentation["order_index"])
        if type(compiled_order) is not int or compiled_order != presentation["order_index"]:
            raise ValueError("compiled retention order_index differs from panel order")
        request = DecisionRequest.model_validate(presentation["request"])
        if compiled.get("request_hash") != presentation["request_hash"]:
            raise ValueError("compiled request identity differs from the presentation")
        request_json_sha256 = _sha256(
            compiled.get("request_json_sha256"), "compiled ordered request SHA-256"
        )
        if request_json_sha256 != study._digest(request.model_dump(mode="json")):
            raise ValueError("compiled ordered request differs from the presentation options")
        compiled_prompt = _sha256(compiled.get("prompt_sha256"), "compiled prompt SHA-256")
        expected_prompt = hashlib.sha256(render_prompt(request).encode("utf-8")).hexdigest()
        if compiled_prompt != expected_prompt:
            raise ValueError("compiled prompt hash differs from the rendered presentation request")
        if compiled_prompt != output["prompt_sha256"]:
            raise ValueError("compiled prompt hash differs from the cached output")
        token_count = compiled.get("input_tokens")
        if type(token_count) is not int or not 1 <= token_count <= MAX_INPUT_TOKENS:
            raise ValueError("compiled input token count must be a strict positive integer")
        if token_count != output["input_tokens"]:
            raise ValueError("compiled input token count differs from the cached output")
    return tuple(dict(row) for row in normalized)


def load_cached_retention(
    root: str | Path,
    inputs: object,
    compilation: object,
) -> CachedRetention:
    """Load the exact selected-adapter retention outputs from pinned receipts."""

    if not isinstance(inputs, runtime_rule_study_inputs.StudyInputs):
        raise TypeError("inputs must be a validated StudyInputs value")
    evaluation, compilation_sha256 = _validate_compilation(compilation, inputs)
    try:
        project_root = Path(root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("cache project root is unavailable") from exc
    if not project_root.is_dir():
        raise ValueError("cache project root must be a directory")

    natural_envelope, natural_receipt_sha256 = _read_pinned_json(
        project_root, NATURAL_RECEIPT_PATH, NATURAL_RECEIPT_SHA256
    )
    transfer_envelope, transfer_receipt_sha256 = _read_pinned_json(
        project_root, TRANSFER_RECEIPT_PATH, TRANSFER_RECEIPT_SHA256
    )
    try:
        selection, selection_file_sha256 = adapter_transfer_contracts.load_selection(project_root)
    except (OSError, ValueError) as exc:
        raise ValueError("pinned adapter selection is unavailable or invalid") from exc
    if dict(inputs.selection) != selection:
        raise ValueError("StudyInputs selection differs from the pinned selection descriptor")
    if inputs.file_sha256.get(adapter_transfer_contracts.SELECTION_PATH) != selection_file_sha256:
        raise ValueError("StudyInputs selection file hash differs from its pinned bytes")

    natural_payloads = _mapping_field(natural_envelope, "payloads", "natural receipt")
    natural_results = _mapping_field(natural_envelope, "results", "natural receipt")
    natural_payload = _object(natural_payloads.get("snli_mix"), "natural snli_mix payload")
    natural_value = _object(natural_results.get("snli_mix"), "natural snli_mix result")
    transfer_payload = _object(transfer_envelope.get("payload"), "transfer payload")
    transfer_value = _object(transfer_envelope.get("result"), "transfer result")
    try:
        natural = _object(
            mixture_training_results.validate_result(natural_value, natural_payload),
            "validated natural result",
        )
        transfer = _object(
            adapter_transfer_results.validate_result(
                transfer_value, transfer_payload, root=project_root
            ),
            "validated transfer result",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("historical receipt validation failed") from exc

    tensor_sha256 = _validate_receipt_identities(
        natural, transfer, transfer_payload, selection, natural_receipt_sha256
    )
    _natural_sources, source_files, tokenizer_files, base_model = _validate_receipt_provenance(
        natural, natural_payload, transfer, transfer_payload, inputs
    )
    old_outputs, transfer_outputs = _retained_outputs(natural, transfer)
    if len(inputs.retention_presentations[:_OLD_RETENTION_COUNT]) != _OLD_RETENTION_COUNT:
        raise ValueError("StudyInputs old retention panel must contain 3,654 rows")
    joined_outputs = _join_retention_rows(
        inputs.retention_presentations,
        [*old_outputs, *transfer_outputs],
        evaluation[:_RETENTION_COUNT],
    )
    if len(joined_outputs) != _RETENTION_COUNT:
        raise ValueError("cached retention join must contain exactly 3,782 rows")

    outputs_sha256 = study._digest(list(joined_outputs))
    runtime_versions = mixture_training_contracts.validate_runtime_versions(
        _mapping_field(natural, "provenance", "natural result").get("versions")
    )
    cache_without_digest = CachedRetention(
        outputs=joined_outputs,
        receipt_file_sha256={
            NATURAL_RECEIPT_PATH: natural_receipt_sha256,
            TRANSFER_RECEIPT_PATH: transfer_receipt_sha256,
        },
        outputs_sha256=outputs_sha256,
        selection=dict(selection),
        tensor_sha256=tensor_sha256,
        tokenizer_file_sha256=tokenizer_files,
        runtime_versions=runtime_versions,
        base_model=base_model,
        source_file_sha256=dict(sorted(source_files.items())),
        compilation_sha256=compilation_sha256,
        cache_sha256="",
    )
    return replace(
        cache_without_digest,
        cache_sha256=study._digest(_cache_payload(cache_without_digest)),
    )


def serialize_cached_retention(cache: CachedRetention) -> dict[str, object]:
    """Return the shared JSON-safe cache representation after digest checks."""

    if not isinstance(cache, CachedRetention):
        raise TypeError("cache must be a CachedRetention value")
    payload = _cache_payload(cache)
    if study._digest(payload["outputs"]) != cache.outputs_sha256:
        raise ValueError("cached retention output digest no longer matches its rows")
    if study._digest(payload) != cache.cache_sha256:
        raise ValueError("cached retention cache SHA-256 no longer matches its fields")
    return {**payload, "cache_sha256": cache.cache_sha256}
