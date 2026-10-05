"""CPU validation of SNLI model states, provenance, and run envelopes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from math import prod
from numbers import Real

from experiments import real_pilot_baseline_core
from experiments import snli_diagnostic_core as core

_QWEN_BASE_VERSION_PINS = {
    name: version for name, version in core.RUNTIME_VERSION_PINS.items() if name != "Pillow"
}
_RESULT_FIELDS = {
    "model_name",
    "status",
    "run_id",
    "nonce",
    "pins",
    "model_id",
    "model_revision",
    "forward_counts",
    "provenance",
    "presentations",
}
_OUTPUT_FIELDS = {
    "presentation_id",
    "record_id",
    "source_group_id",
    "request_hash",
    "order_index",
    "order_ids",
    "candidate_logits",
    "winner_option_id",
    "input_tokens",
    "prompt_sha256",
}
_PROVENANCE_FIELDS = {
    "measured_source_file_sha256",
    "runtime_versions",
    "effective_dtype",
    "scorer",
    "base",
    "adapter",
}


def _validate_qwen_base(value: object) -> dict[str, object]:
    base = core._normalize_object(value, "Qwen base evidence")
    fields = {
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
    pin = core.MODEL_PINS["qwen_base"]
    if set(base) != fields or (base["model_id"], base["model_revision"]) != (
        pin["model_id"],
        pin["model_revision"],
    ):
        raise ValueError("Qwen base evidence differs from its frozen checkpoint")
    diagnostics = base["load_diagnostics"]
    if (
        base["model_class"] != "Qwen3_5ForCausalLM"
        or base["config_class"] != "Qwen3_5TextConfig"
        or base["layer_count"] != 24
        or base["tied_embeddings"] is not True
        or base["no_meta_parameters"] is not True
        or not isinstance(diagnostics, dict)
        or set(diagnostics) != {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
        or any(diagnostics[name] != [] for name in diagnostics)
        or base["effective_dtype"] != "torch.bfloat16"
        or not isinstance(base["device"], str)
        or not base["device"].startswith("cuda")
        or base["attention_implementation"] != "eager"
        or base["use_kernels"] is not False
        or base["use_hub_kernels"] != "NO"
        or base["versions"] != _QWEN_BASE_VERSION_PINS
    ):
        raise ValueError("Qwen base loading evidence failed its runtime checks")
    tokenizer_hashes = base["tokenizer_file_sha256"]
    if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
        raise ValueError("Qwen tokenizer snapshot hashes are missing")
    for name, digest in tokenizer_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Qwen tokenizer snapshot hash is malformed")
        core._digest(digest, f"tokenizer snapshot {name}")
    return base


def _validate_qwen_adapter(value: object) -> dict[str, object]:
    adapter = core._normalize_object(value, "Qwen adapter evidence")
    fields = {
        "expected_file_sha256",
        "actual_file_sha256",
        "tensor_keys_shapes_values_match",
        "default_adapter_active",
        "adapter_unmerged",
        "inventory",
        "all_adapter_parameters_frozen",
    }
    if (
        set(adapter) != fields
        or any(
            adapter[name] != core.QWEN_ADAPTER_FILE_SHA256
            for name in ("expected_file_sha256", "actual_file_sha256")
        )
        or any(
            adapter[name] is not True
            for name in (
                "tensor_keys_shapes_values_match",
                "default_adapter_active",
                "adapter_unmerged",
                "all_adapter_parameters_frozen",
            )
        )
    ):
        raise ValueError("Qwen adapter evidence differs from the frozen snapshot")
    inventory = adapter["inventory"]
    fields = {
        "module_count",
        "module_names",
        "adapter_tensor_count",
        "adapter_parameter_count",
        "adapter_tensor_names",
        "adapter_tensor_shapes",
        "adapter_tensor_dtypes",
        "trainable_parameter_count",
        "base_parameters_frozen_bf16",
    }
    if not isinstance(inventory, dict) or set(inventory) != fields:
        raise ValueError("Qwen adapter inventory has an unexpected schema")
    names, tensors = inventory["module_names"], inventory["adapter_tensor_names"]
    shapes, dtypes = inventory["adapter_tensor_shapes"], inventory["adapter_tensor_dtypes"]
    if (
        inventory["module_count"] != 60
        or not isinstance(names, list)
        or len(names) != 60
        or len(set(names)) != 60
        or inventory["adapter_tensor_count"] != 120
        or not isinstance(tensors, list)
        or len(tensors) != 120
        or len(set(tensors)) != 120
        or inventory["adapter_parameter_count"] != 2_015_232
        or inventory["trainable_parameter_count"] != 0
        or inventory["base_parameters_frozen_bf16"] is not True
        or not isinstance(shapes, dict)
        or set(shapes) != set(tensors)
        or not isinstance(dtypes, dict)
        or set(dtypes) != set(tensors)
        or any(
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(dimension) is not int or dimension < 1 for dimension in shape)
            for shape in shapes.values()
        )
        or sum(prod(shape) for shape in shapes.values()) != 2_015_232
        or any(dtype != "torch.float32" for dtype in dtypes.values())
    ):
        raise ValueError("Qwen adapter inventory does not match the frozen LoRA state")
    return adapter


def _validate_versions(value: object, *, complete: bool) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("runtime versions must be an object")
    versions = dict(value)
    if complete and versions != core.RUNTIME_VERSION_PINS:
        raise ValueError("runtime versions differ from the frozen Modal image")
    if any(
        name not in core.RUNTIME_VERSION_PINS or not isinstance(version, str) or not version.strip()
        for name, version in versions.items()
    ):
        raise ValueError("partial runtime versions are malformed")
    return versions


def _validate_source_measurements(
    value: object, pins: Mapping[str, object], *, complete: bool
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("measured source fingerprints must be an object")
    measured = dict(value)
    if complete and measured != pins["source_file_sha256"]:
        raise ValueError("measured source files differ from the payload pins")
    for path, digest in measured.items():
        if path not in core.SOURCE_FINGERPRINT_PATHS:
            raise ValueError("measured source fingerprints are malformed")
        core._digest(digest, f"measured source {path}")
    return measured


def _validate_provenance(
    value: object, model_name: str, pins: Mapping[str, object], *, complete: bool
) -> dict[str, object]:
    provenance = core._normalize_object(value, "model provenance")
    if set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("model provenance has an unexpected schema")
    measured = _validate_source_measurements(
        provenance["measured_source_file_sha256"], pins, complete=complete
    )
    versions = _validate_versions(provenance["runtime_versions"], complete=complete)
    dtype, scorer = provenance["effective_dtype"], provenance["scorer"]
    if dtype is not None and (not isinstance(dtype, str) or not dtype.strip()):
        raise ValueError("effective dtype must be nonblank when present")
    if not isinstance(scorer, dict):
        raise ValueError("scorer provenance must be an object")
    base, adapter = provenance["base"], provenance["adapter"]
    if model_name.startswith("qwen_"):
        if base is not None:
            base = _validate_qwen_base(base)
        if adapter is not None:
            adapter = _validate_qwen_adapter(adapter)
        if scorer:
            raise ValueError("Qwen provenance may not contain reference scorer fields")
        if complete and (base is None or dtype != "torch.bfloat16"):
            raise ValueError("passed Qwen result requires pinned BF16 base evidence")
        if complete and ((model_name == "qwen_base") != (adapter is None)):
            raise ValueError("Qwen base/final adapter evidence is inconsistent")
        if base is None and adapter is not None:
            raise ValueError("Qwen adapter evidence requires base checkpoint evidence")
    else:
        if base is not None or adapter is not None:
            raise ValueError("reference model provenance may not contain Qwen evidence")
        if scorer:
            validator = (
                real_pilot_baseline_core._validate_scorer_provenance
                if complete
                else real_pilot_baseline_core._validate_partial_scorer_provenance
            )
            scorer = validator(model_name, scorer)
        elif complete:
            raise ValueError("passed reference result requires official scorer provenance")
        if complete:
            runtime = scorer.get("runtime")
            expected_dtype = (
                scorer.get("effective_dtype")
                if model_name == "intern"
                else runtime.get("effective_dtype")
                if isinstance(runtime, dict)
                else None
            )
            if dtype != expected_dtype:
                raise ValueError("effective dtype differs from official scorer evidence")
    return {
        "measured_source_file_sha256": measured,
        "runtime_versions": versions,
        "effective_dtype": dtype,
        "scorer": scorer,
        "base": base,
        "adapter": adapter,
    }


def _validate_output_row(value: object, expected: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _OUTPUT_FIELDS:
        raise ValueError("model presentation result has an unexpected schema")
    identity_fields = _OUTPUT_FIELDS - {
        "candidate_logits",
        "winner_option_id",
        "input_tokens",
        "prompt_sha256",
    }
    if type(value["order_index"]) is not int or any(
        value[key] != expected[key] for key in identity_fields
    ):
        raise ValueError("model presentation metadata differs from the request-only payload")
    option_ids, logits = expected["order_ids"], value["candidate_logits"]
    if (
        not isinstance(logits, list)
        or len(logits) != len(option_ids)
        or any(
            isinstance(number, bool)
            or not isinstance(number, Real)
            or not math.isfinite(float(number))
            for number in logits
        )
    ):
        raise ValueError("candidate logits must be finite and match the option count")
    normalized = [float(number) for number in logits]
    winner = min(
        option_id
        for option_id, logit in zip(option_ids, normalized, strict=True)
        if logit == max(normalized)
    )
    if value["winner_option_id"] != winner:
        raise ValueError("winner does not match semantic lexical tie-breaking")
    if (
        type(value["input_tokens"]) is not int
        or not 1 <= value["input_tokens"] <= core.MAX_INPUT_TOKENS
    ):
        raise ValueError("input token count is outside the 1–2,048 limit")
    core._digest(value["prompt_sha256"], "prompt_sha256")
    output = dict(value)
    output["candidate_logits"] = normalized
    return output


def validate_model_result(result: object, payload: object, model_name: str) -> dict[str, object]:
    """Validate complete results or a failed worker's exact ordered prefix."""

    if model_name not in core.MODEL_NAMES:
        raise ValueError("model name is outside the frozen allowlist")
    expected_payload = core.validate_payload(payload)
    state = core._normalize_object(result, "model result")
    status = state.get("status")
    fields = _RESULT_FIELDS | ({"failure"} if status == "failed" else set())
    if set(state) != fields or state.get("model_name") != model_name:
        raise ValueError("model result has an unexpected schema or name")
    if status not in {"passed", "failed"}:
        raise ValueError("model result status must be passed or failed")
    model_pin = core.MODEL_PINS[model_name]
    if (
        state["run_id"] != expected_payload["run_id"]
        or state["nonce"] != expected_payload["nonce"]
        or core._validate_pins(state["pins"]) != expected_payload["pins"]
        or state.get("model_id") != model_pin["model_id"]
        or state.get("model_revision") != model_pin["model_revision"]
    ):
        raise ValueError("model result identity differs from the frozen payload or checkpoint")
    outputs, expected_rows = state["presentations"], expected_payload["presentations"]
    if not isinstance(outputs, list) or len(outputs) > core.PRESENTATION_COUNT:
        raise ValueError("model presentations must be an ordered panel prefix")
    if status == "passed" and len(outputs) != core.PRESENTATION_COUNT:
        raise ValueError("passed model result must contain the complete panel")
    normalized_outputs = []
    for index, output in enumerate(outputs):
        if (
            not isinstance(output, dict)
            or output.get("presentation_id") != expected_rows[index]["presentation_id"]
        ):
            raise ValueError("failed model presentations must be a validated prefix")
        normalized_outputs.append(_validate_output_row(output, expected_rows[index]))
    counts = state["forward_counts"]
    if not isinstance(counts, dict) or set(counts) != {"scored", "auxiliary", "total"}:
        raise ValueError("forward counts have an unexpected schema")
    scored, auxiliary, total = counts["scored"], counts["auxiliary"], counts["total"]
    if type(scored) is not int or scored != len(normalized_outputs):
        raise ValueError("scored forward count must match validated presentations")
    if status == "passed":
        auxiliary_expected = core.AUXILIARY_FORWARD_COUNTS[model_name]
        expected_counts = {
            "scored": core.PRESENTATION_COUNT,
            "auxiliary": auxiliary_expected,
            "total": core.PRESENTATION_COUNT + auxiliary_expected,
        }
        if counts != expected_counts or any(
            type(counts[key]) is not int for key in expected_counts
        ):
            raise ValueError("passed forward counts differ from the frozen limit")
    else:
        allowed_aux = {None, core.AUXILIARY_FORWARD_COUNTS[model_name]}
        if model_name == "intern":
            allowed_aux.add(0)
        if (
            (auxiliary is not None and type(auxiliary) is not int)
            or auxiliary not in allowed_aux
            or (auxiliary is None and total is not None)
            or (total is not None and (type(total) is not int or total != scored + auxiliary))
        ):
            raise ValueError("failed forward counts do not match observed work")
        failure = state["failure"]
        if (
            not isinstance(failure, dict)
            or set(failure) != {"stage", "type", "message"}
            or any(
                not isinstance(failure[key], str) or not failure[key].strip()
                for key in ("stage", "type", "message")
            )
        ):
            raise ValueError("failed model details are malformed")
        state["failure"] = dict(failure)
    state["forward_counts"] = {"scored": scored, "auxiliary": auxiliary, "total": total}
    state["provenance"] = _validate_provenance(
        state["provenance"], model_name, expected_payload["pins"], complete=status == "passed"
    )
    state["presentations"] = normalized_outputs
    return state


def validate_receipt(receipt: object, payload: object) -> dict[str, object]:
    """Validate the envelope and retain successful models across lifecycle failures."""

    expected_payload = core.validate_payload(payload)
    envelope = core._normalize_object(receipt, "receipt")
    fields = {"schema_version", "status", "payload", "models", "provenance"}
    if set(envelope) != fields and set(envelope) != fields | {"failure"}:
        raise ValueError("receipt has an unexpected schema")
    if (
        type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != core.SCHEMA_VERSION
    ):
        raise ValueError("receipt schema version is unsupported")
    receipt_payload = core.validate_payload(envelope["payload"])
    if receipt_payload != expected_payload:
        raise ValueError("receipt payload differs from the locally reserved identity")
    models = envelope["models"]
    if not isinstance(models, dict) or set(models) != set(core.MODEL_NAMES):
        raise ValueError("receipt must contain all four model states")
    normalized_models = {
        name: validate_model_result(models[name], expected_payload, name)
        for name in core.MODEL_NAMES
    }
    passed = sum(state["status"] == "passed" for state in normalized_models.values())
    status = "passed" if passed == len(core.MODEL_NAMES) else "partial" if passed else "failed"
    failure = envelope.get("failure")
    if failure is not None:
        if (
            not isinstance(failure, dict)
            or set(failure) != {"stage", "type", "message"}
            or any(
                not isinstance(failure[key], str) or not failure[key].strip()
                for key in ("stage", "type", "message")
            )
        ):
            raise ValueError("receipt lifecycle failure is malformed")
        if failure["stage"] != "modal_lifecycle":
            raise ValueError("receipt-level failure must identify a Modal lifecycle failure")
        status = "failed"
    if envelope["status"] != status:
        raise ValueError("receipt status does not match model and lifecycle states")
    provenance = core._normalize_object(envelope["provenance"], "receipt provenance")
    fields = {"modal_profile", "workspace", "modal_sdk_version", "limits"}
    modal_sdk_version = provenance.get("modal_sdk_version")
    if set(provenance) != fields or (
        provenance["modal_profile"] != "reflex-personal"
        or provenance["workspace"] != "rajath-61258"
        or (
            modal_sdk_version is not None
            and (not isinstance(modal_sdk_version, str) or not modal_sdk_version.strip())
        )
        or provenance["limits"] != core.MODAL_LIMITS
        or (passed > 0 and modal_sdk_version != "1.6.1")
    ):
        raise ValueError("receipt Modal provenance differs from the frozen run limits")
    normalized: dict[str, object] = {
        "schema_version": core.SCHEMA_VERSION,
        "status": status,
        "payload": receipt_payload,
        "models": normalized_models,
        "provenance": provenance,
    }
    if failure is not None:
        normalized["failure"] = dict(failure)
    return normalized
