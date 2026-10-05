"""Contracts and CPU-only accounting for targeted matched training."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path

EXPERIMENT_ID = "targeted-reasoning-v1"
ROLE_UNCHANGED = "unchanged"
ROLE_CONTROL = "control"
ROLE_TREATMENT = "treatment"
ROLES = (ROLE_UNCHANGED, ROLE_CONTROL, ROLE_TREATMENT)

TRAINING_FORWARDS = 4_800
FINAL_EVALUATION_FORWARDS = 6_182
RELOAD_PARITY_FORWARDS = 32
MAX_TOTAL_FORWARDS = 28_210
MAX_TOTAL_INPUT_TOKENS = 57_774_080
MAX_INPUT_TOKENS = 2_048
FRESH_SUMMARY_SHA256 = "00b6331faf6392dbd6f8ce84e7eac98053ae54b8a7d90017df003c1eb237d288"
TRANSFER_SUMMARY_SHA256 = "9594d10665e18c725f2312f636198f69277b7ac158bbbe47c1d6ca5be27e9441"
TARGETED_SOURCE_PATHS = (
    "experiments/targeted_training_contracts.py",
    "experiments/targeted_training_core.py",
    "experiments/targeted_training_data.py",
    "experiments/targeted_training_runtime.py",
    "experiments/modal_targeted_training.py",
    "experiments/analyze_targeted_training.py",
    "docs/targeted-training-protocol.md",
    "pyproject.toml",
    "uv.lock",
    "docs/verification/fresh-eval-summary.json",
    "docs/verification/adapter-transfer-summary.json",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")


def canonical_sha256(value: object) -> str:
    """Hash strictly JSON-safe immutable launch and role documents."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("immutable payload is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def compile_presentations(rows: object, tokenizer: object) -> tuple[dict[str, object], ...]:
    """Compile every request locally and retain only immutable accounting evidence.

    This deliberately has no truncation path.  The remote worker recompiles the
    same requests and compares these identities before a model forward.
    """

    from reflex_decisions.rendering import compile_request
    from reflex_decisions.schema import DecisionRequest

    if not isinstance(rows, (tuple, list)):
        raise ValueError("presentations must be an ordered sequence")
    compiled_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in rows:
        if not isinstance(value, dict):
            raise ValueError("presentation is malformed")
        presentation_id = value.get("presentation_id")
        request_payload = value.get("request")
        order_ids = value.get("order_ids")
        expected_request_hash = value.get("request_hash")
        if (
            not isinstance(presentation_id, str)
            or not presentation_id
            or presentation_id in seen
            or not isinstance(request_payload, dict)
            or not isinstance(order_ids, list)
            or not isinstance(expected_request_hash, str)
        ):
            raise ValueError("presentation identity is malformed")
        request = DecisionRequest.model_validate(request_payload)
        compiled = compile_request(request, tokenizer, max_tokens=MAX_INPUT_TOKENS)
        if (
            compiled.request_hash != expected_request_hash
            or tuple(order_ids) != tuple(option.id for option in request.options)
            or len(compiled.input_ids) > MAX_INPUT_TOKENS
        ):
            raise ValueError("compiled presentation identity differs from its frozen request")
        seen.add(presentation_id)
        compiled_rows.append(
            {
                "presentation_id": presentation_id,
                "request_hash": compiled.request_hash,
                "prompt_sha256": compiled.prompt_hash,
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(compiled.input_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "input_tokens": len(compiled.input_ids),
                "candidate_token_ids": list(compiled.candidate_token_ids),
            }
        )
    return tuple(compiled_rows)


def expected_forward_counts(role: object) -> dict[str, int]:
    """Return the exact forward caps for one immutable worker role."""

    if role == ROLE_UNCHANGED:
        return {
            "training": 0,
            "final_evaluation": FINAL_EVALUATION_FORWARDS,
            "reload_parity": 0,
            "total": FINAL_EVALUATION_FORWARDS,
        }
    if role in {ROLE_CONTROL, ROLE_TREATMENT}:
        total = TRAINING_FORWARDS + FINAL_EVALUATION_FORWARDS + RELOAD_PARITY_FORWARDS
        return {
            "training": TRAINING_FORWARDS,
            "final_evaluation": FINAL_EVALUATION_FORWARDS,
            "reload_parity": RELOAD_PARITY_FORWARDS,
            "total": total,
        }
    raise ValueError("role must be unchanged, control, or treatment")


def validate_completed_counts(role: object, value: object) -> dict[str, int]:
    """Require completed forward counts to match one finished role exactly."""

    expected = expected_forward_counts(role)
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("forward counts must contain the exact role categories")
    counts: dict[str, int] = {}
    for category, cap in expected.items():
        count = value[category]
        if type(count) is not int or count < 0:
            raise ValueError(f"{category} count must be a nonnegative integer")
        if count != cap:
            raise ValueError(f"{category} count differs from its role cap")
        counts[category] = count
    return counts


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"pinned summary is unavailable or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"pinned summary must be a JSON object: {path}")
    return value


def _validate_source_map(
    root: Path, value: object, expected_paths: tuple[str, ...], label: str
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(expected_paths):
        raise ValueError(f"{label} source paths differ from the exact declared allowlist")
    normalized: dict[str, str] = {}
    for relative in expected_paths:
        digest = value[relative]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{label} source digest is malformed: {relative}")
        try:
            source = (root / relative).resolve(strict=True)
            source.relative_to(root)
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            raise ValueError(f"{label} pinned source is unavailable: {relative}") from exc
        if actual != digest:
            raise ValueError(f"{label} pinned source differs from its summary: {relative}")
        normalized[relative] = digest
    return normalized


def load_source_maps(root: str | Path) -> dict[str, dict[str, str]]:
    """Read and verify independent fresh and historical source proofs."""

    from experiments import adapter_transfer_contracts as transfer
    from experiments import fresh_eval_core as fresh

    project_root = Path(root).resolve(strict=True)
    fresh_path = project_root / "docs/verification/fresh-eval-summary.json"
    transfer_path = project_root / "docs/verification/adapter-transfer-summary.json"
    if hashlib.sha256(fresh_path.read_bytes()).hexdigest() != FRESH_SUMMARY_SHA256:
        raise ValueError("fresh source summary hash differs from its reviewed pin")
    if hashlib.sha256(transfer_path.read_bytes()).hexdigest() != TRANSFER_SUMMARY_SHA256:
        raise ValueError("historical source summary hash differs from its reviewed pin")
    fresh_summary = _read_json_object(fresh_path)
    transfer_summary = _read_json_object(transfer_path)
    fresh_pins = fresh_summary.get("pins")
    transfer_pins = transfer_summary.get("pins")
    if not isinstance(fresh_pins, dict) or not isinstance(transfer_pins, dict):
        raise ValueError("pinned summaries do not contain pin objects")
    fresh_map = _validate_source_map(
        project_root,
        fresh_pins.get("source_file_sha256"),
        fresh.SOURCE_FINGERPRINT_PATHS,
        "fresh",
    )
    historical_map = _validate_source_map(
        project_root,
        transfer_pins.get("candidate_source_file_sha256"),
        transfer.SOURCE_FINGERPRINT_PATHS,
        "historical",
    )
    if len(fresh_map) != 71 or len(historical_map) != 60:
        raise ValueError("source summary cardinalities differ from the reviewed 71/60 proofs")
    return {"fresh": fresh_map, "historical": historical_map}


def targeted_source_manifest(root: str | Path) -> dict[str, str]:
    """Hash the complete bounded source image required by the targeted lane.

    It composes, rather than conflates, the separate 71/60 historical proofs;
    a missing additive source therefore blocks launch instead of being silently
    omitted from a whole-repository mount.
    """

    project_root = Path(root).resolve(strict=True)
    inherited = load_source_maps(project_root)
    manifest = {**inherited["fresh"], **inherited["historical"]}
    for relative in TARGETED_SOURCE_PATHS:
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
            contents = path.read_bytes()
        except (OSError, ValueError) as exc:
            raise ValueError(f"targeted source is unavailable: {relative}") from exc
        manifest[relative] = hashlib.sha256(contents).hexdigest()
    return dict(sorted(manifest.items()))


def build_role_payload(
    inputs: object,
    role: object,
    tokenizer: object,
    *,
    run_id: str,
    source_pins: Mapping[str, str],
) -> dict[str, object]:
    """Construct a label-minimal role payload after CPU prompt compilation.

    ``TargetedInputs`` is intentionally duck-typed here: importing the data
    builder would couple launch planning to its host-only data implementation.
    """

    from experiments import adapter_transfer_contracts
    from experiments.targeted_training_contracts import (
        MODEL_ID,
        MODEL_REVISION,
        TOKENIZER_FILE_SHA256,
    )

    if role not in ROLES:
        raise ValueError("role must be unchanged, control, or treatment")
    if (
        not isinstance(run_id, str)
        or _RUN_ID.fullmatch(run_id) is None
        or not isinstance(source_pins, Mapping)
    ):
        raise ValueError("role run or source binding is missing")
    presentations = getattr(inputs, "presentations", None)
    schedules = getattr(inputs, "schedules", None)
    selection = getattr(inputs, "selection", None)
    strata = getattr(inputs, "strata", None)
    input_pins = getattr(inputs, "file_sha256", None)
    input_audit = getattr(inputs, "audit", None)
    if (
        not isinstance(presentations, (tuple, list))
        or len(presentations) != FINAL_EVALUATION_FORWARDS
        or not isinstance(schedules, dict)
        or not isinstance(selection, dict)
        or not isinstance(strata, dict)
        or not isinstance(input_pins, dict)
        or not isinstance(input_audit, dict)
    ):
        raise ValueError("targeted inputs do not satisfy the frozen launch interface")
    selection = adapter_transfer_contracts.validate_selection(selection)
    evaluations = tuple(dict(row) for row in presentations if isinstance(row, dict))
    if len(evaluations) != len(presentations):
        raise ValueError("evaluation presentations are malformed")
    evaluation_compiled = compile_presentations(evaluations, tokenizer)
    if role == ROLE_UNCHANGED:
        training: tuple[dict[str, object], ...] = ()
        training_compiled: tuple[dict[str, object], ...] = ()
    else:
        training = tuple(dict(row) for row in schedules.get(role, ()) if isinstance(row, dict))
        if len(training) != len(schedules.get(role, ())):
            raise ValueError("training schedule is malformed")
        from experiments.targeted_training_runtime import validate_training_schedule

        validate_training = validate_training_schedule
        training = validate_training(role, training)
        training_compiled = compile_presentations(training, tokenizer)
    reload_ids = [
        row["presentation_id"]
        for row in sorted(
            evaluations,
            key=lambda row: (
                str(row.get("dataset_id")),
                str(row.get("record_id")),
                int(row.get("order_index", -1)),
                str(row.get("presentation_id")),
            ),
        )
        if strata.get(row.get("dataset_id")) == "retention"
    ][:RELOAD_PARITY_FORWARDS]
    if role == ROLE_UNCHANGED:
        reload_ids = []
    elif len(reload_ids) != RELOAD_PARITY_FORWARDS:
        raise ValueError("reload parity selection does not contain 32 retention presentations")
    # Evaluation payloads never carry answer fields; training gets only its own losses.
    public_evaluations = tuple(
        {
            key: row[key]
            for key in (
                "presentation_id",
                "record_id",
                "dataset_id",
                "source_group_id",
                "request_hash",
                "order_index",
                "order_ids",
                "request",
            )
        }
        for row in evaluations
    )
    counts = expected_forward_counts(role)
    compiled_by_id = {row["presentation_id"]: row for row in evaluation_compiled}
    reload_tokens = sum(
        int(compiled_by_id[identifier]["input_tokens"]) for identifier in reload_ids
    )
    input_tokens = sum(
        int(row["input_tokens"]) for row in (*training_compiled, *evaluation_compiled)
    )
    if role != ROLE_UNCHANGED:
        input_tokens += reload_tokens
    if input_tokens > MAX_TOTAL_INPUT_TOKENS:
        raise ValueError("role compilation exceeds the absolute input-token budget")
    token_counts = {
        "training": sum(int(row["input_tokens"]) for row in training_compiled),
        "final_evaluation": sum(int(row["input_tokens"]) for row in evaluation_compiled),
        "reload_parity": reload_tokens if role != ROLE_UNCHANGED else 0,
    }
    token_counts["total"] = sum(token_counts.values())
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": run_id,
        "role": role,
        "source_pins": dict(source_pins),
        "input_file_sha256": dict(input_pins),
        "input_audit": dict(input_audit),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "base_dtype": "bfloat16",
            "adapter_dtype": "float32",
        },
        "tokenizer_file_sha256": dict(TOKENIZER_FILE_SHA256),
        "selection": dict(selection),
        "training": training,
        "evaluation_presentations": public_evaluations,
        "compiled": {"training": training_compiled, "final_evaluation": evaluation_compiled},
        "reload_presentation_ids": tuple(reload_ids),
        "forward_caps": counts,
        "compiled_input_token_counts": token_counts,
        "compiled_input_tokens": input_tokens,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def validate_role_payload(payload: object) -> dict[str, object]:
    """Validate one immutable role payload without model dependencies."""
    from collections import Counter

    from experiments import adapter_transfer_contracts
    from experiments import analyze_targeted_training as analysis
    from experiments.targeted_training_contracts import (
        MODEL_ID,
        MODEL_REVISION,
        TOKENIZER_FILE_SHA256,
    )
    from experiments.targeted_training_runtime import validate_training_schedule
    from reflex_decisions.schema import DecisionRequest

    required = {
        "experiment_id",
        "run_id",
        "role",
        "source_pins",
        "input_file_sha256",
        "input_audit",
        "model",
        "tokenizer_file_sha256",
        "selection",
        "training",
        "evaluation_presentations",
        "compiled",
        "reload_presentation_ids",
        "forward_caps",
        "compiled_input_token_counts",
        "compiled_input_tokens",
        "payload_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("role payload digest or schema is incomplete")
    role = payload["role"]
    if (
        payload["experiment_id"] != EXPERIMENT_ID
        or role not in ROLES
        or not isinstance(payload["run_id"], str)
        or _RUN_ID.fullmatch(payload["run_id"]) is None
    ):
        raise ValueError("role payload run or experiment identity differs")
    digest = payload["payload_sha256"]
    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or digest
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "payload_sha256"}
        )
    ):
        raise ValueError("role payload digest differs from canonical contents")
    source_pins = payload["source_pins"]
    if (
        not isinstance(source_pins, dict)
        or not set(TARGETED_SOURCE_PATHS) <= set(source_pins)
        or any(
            not isinstance(key, str) or not isinstance(value, str) or not _SHA256.fullmatch(value)
            for key, value in source_pins.items()
        )
    ):
        raise ValueError("role source pins are malformed")
    input_pins = payload["input_file_sha256"]
    if (
        not isinstance(input_pins, dict)
        or not input_pins
        or any(
            not isinstance(key, str) or not isinstance(value, str) or not _SHA256.fullmatch(value)
            for key, value in input_pins.items()
        )
        or not isinstance(payload["input_audit"], dict)
        or not payload["input_audit"]
    ):
        raise ValueError("role input pins or audit are malformed")
    if payload["model"] != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "base_dtype": "bfloat16",
        "adapter_dtype": "float32",
    }:
        raise ValueError("role model identity or dtype differs")
    if payload["tokenizer_file_sha256"] != TOKENIZER_FILE_SHA256:
        raise ValueError("role tokenizer files differ from reviewed pin")
    adapter_transfer_contracts.validate_selection(payload["selection"])
    if payload["forward_caps"] != expected_forward_counts(role):
        raise ValueError("role forward caps differ from fixed budget")

    presentations = payload["evaluation_presentations"]
    if (
        not isinstance(presentations, (tuple, list))
        or len(presentations) != FINAL_EVALUATION_FORWARDS
    ):
        raise ValueError("role final panel has the wrong count")
    presentation_ids: set[str] = set()
    counts: Counter[str] = Counter()
    expected_eval_keys = set((*analysis.IDENTITY_FIELDS, "request"))
    for value in presentations:
        if not isinstance(value, dict) or set(value) != expected_eval_keys:
            raise ValueError("role final panel must be exactly label-free")
        identifier = value["presentation_id"]
        if not isinstance(identifier, str) or not identifier or identifier in presentation_ids:
            raise ValueError("role final panel identity is malformed or duplicated")
        presentation_ids.add(identifier)
        request = DecisionRequest.model_validate(value["request"])
        if (
            value["request_hash"] != request.request_hash
            or value["order_ids"] != [option.id for option in request.options]
            or type(value["order_index"]) is not int
            or value["order_index"] < 0
            or any(
                not isinstance(value[key], str) or not value[key]
                for key in ("record_id", "dataset_id", "source_group_id")
            )
        ):
            raise ValueError("role final panel request or order identity differs")
        counts[value["dataset_id"]] += 1
    if counts != analysis.TASK_COUNTS:
        raise ValueError("role final task counts differ from fixed panel")

    training = validate_training_schedule(role, payload["training"])
    compiled = payload["compiled"]
    if not isinstance(compiled, dict) or set(compiled) != {"training", "final_evaluation"}:
        raise ValueError("role compiled phase set is malformed")
    token_counts: dict[str, int] = {}
    for phase, rows in (("training", training), ("final_evaluation", presentations)):
        compiled_rows = compiled[phase]
        if not isinstance(compiled_rows, (tuple, list)) or len(compiled_rows) != len(rows):
            raise ValueError("role compiled phase row count differs")
        tokens = 0
        for row, evidence in zip(rows, compiled_rows, strict=True):
            if not isinstance(evidence, dict) or set(evidence) != {
                "presentation_id",
                "request_hash",
                "prompt_sha256",
                "input_ids_sha256",
                "input_tokens",
                "candidate_token_ids",
            }:
                raise ValueError("role compiled row identity is incomplete")
            if (
                evidence["presentation_id"] != row["presentation_id"]
                or evidence["request_hash"] != row["request_hash"]
            ):
                raise ValueError("role compiled row identity differs from presentation")
            if (
                any(
                    not isinstance(evidence[key], str) or not _SHA256.fullmatch(evidence[key])
                    for key in ("prompt_sha256", "input_ids_sha256")
                )
                or type(evidence["input_tokens"]) is not int
                or not 0 < evidence["input_tokens"] <= MAX_INPUT_TOKENS
                or not isinstance(evidence["candidate_token_ids"], list)
                or len(evidence["candidate_token_ids"]) != len(row["order_ids"])
                or any(
                    type(token) is not int or token < 0 for token in evidence["candidate_token_ids"]
                )
            ):
                raise ValueError("role compiled prompt, tokens, or candidate IDs are malformed")
            tokens += evidence["input_tokens"]
        token_counts[phase] = tokens
    reload_ids = payload["reload_presentation_ids"]
    if not isinstance(reload_ids, (tuple, list)):
        raise ValueError("role reload IDs are malformed")
    retention = [
        row["presentation_id"]
        for row in sorted(
            presentations,
            key=lambda row: (
                row["dataset_id"],
                row["record_id"],
                row["order_index"],
                row["presentation_id"],
            ),
        )
        if row["dataset_id"] in analysis.RETENTION_COUNTS
    ][:RELOAD_PARITY_FORWARDS]
    if list(reload_ids) != ([] if role == ROLE_UNCHANGED else retention):
        raise ValueError("role reload IDs differ from fixed retention selection")
    compiled_by_id = {row["presentation_id"]: row for row in compiled["final_evaluation"]}
    token_counts["reload_parity"] = sum(
        compiled_by_id[identifier]["input_tokens"] for identifier in reload_ids
    )
    token_counts["total"] = sum(token_counts.values())
    if (
        payload["compiled_input_token_counts"] != token_counts
        or payload["compiled_input_tokens"] != token_counts["total"]
    ):
        raise ValueError("role compiled token totals differ from fixed rows")
    if token_counts["total"] > MAX_TOTAL_INPUT_TOKENS:
        raise ValueError("role compiled token total exceeds study ceiling")
    return payload


def validate_role_result(value: object, role: object) -> dict[str, object]:
    """Check fixed role count and successful worker status before deeper adoption."""
    if not isinstance(value, dict) or value.get("role") != role or value.get("status") != "passed":
        raise ValueError("worker result identity or status is malformed")
    execution = value.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "completed",
        "attempted_unknown",
        "not_started",
        "input_token_counts",
    }:
        raise ValueError("worker execution ledger is incomplete")
    validate_completed_counts(role, execution["completed"])
    zeros = {category: 0 for category in ("training", "final_evaluation", "reload_parity")}
    if execution["attempted_unknown"] != zeros or execution["not_started"] != zeros:
        raise ValueError("completed result has unresolved worker work")
    return value


def _require_provenance(payload: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    from experiments.mixture_training_contracts import RUNTIME_VERSION_PINS
    from experiments.targeted_training_contracts import (
        MODEL_ID,
        MODEL_REVISION,
        SELECTED_ADAPTER_TENSOR_SHA256,
        TOKENIZER_FILE_SHA256,
    )

    provenance = result.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("worker provenance is missing")
    if provenance.get("source_pins_sha256") != canonical_sha256(payload["source_pins"]):
        raise ValueError("worker source proof differs from frozen source pins")
    base = provenance.get("base_model")
    if not isinstance(base, dict) or (
        base.get("model_id") != MODEL_ID
        or base.get("model_revision") != MODEL_REVISION
        or base.get("tokenizer_file_sha256") != TOKENIZER_FILE_SHA256
        or base.get("effective_dtype") != "torch.bfloat16"
        or not isinstance(base.get("device"), str)
        or not base["device"].startswith("cuda")
        or base.get("layer_count") != 24
        or base.get("tied_embeddings") is not True
        or base.get("no_meta_parameters") is not True
        or base.get("attention_implementation") != "eager"
        or base.get("use_kernels") is not False
        or base.get("use_hub_kernels") != "NO"
    ):
        raise ValueError("worker model or tokenizer provenance differs from pinned runtime")
    expected_versions = {
        name: version for name, version in RUNTIME_VERSION_PINS.items() if name != "Pillow"
    }
    if base.get("versions") != expected_versions:
        raise ValueError("worker package versions differ from pinned runtime image")
    if provenance.get("adapter_tensor_sha256") != SELECTED_ADAPTER_TENSOR_SHA256:
        raise ValueError("selected adapter tensor digest differs from reviewed pin")
    inventory = provenance.get("adapter_inventory")
    role = payload["role"]
    if not isinstance(inventory, dict) or (
        inventory.get("module_count") != 60
        or inventory.get("adapter_tensor_count") != 120
        or inventory.get("adapter_parameter_count") != 2_015_232
        or inventory.get("trainable_parameter_count")
        != (0 if role == ROLE_UNCHANGED else 2_015_232)
        or inventory.get("base_parameters_frozen_bf16") is not True
    ):
        raise ValueError("worker adapter inventory differs from fixed LoRA")
    names = inventory.get("adapter_tensor_names")
    dtypes = inventory.get("adapter_tensor_dtypes")
    if (
        not isinstance(names, list)
        or len(names) != 120
        or len(set(names)) != 120
        or not isinstance(dtypes, dict)
        or set(dtypes) != set(names)
        or any(dtype != "torch.float32" for dtype in dtypes.values())
    ):
        raise ValueError("worker FP32 LoRA tensor inventory is incomplete")
    return provenance


def _require_snapshot(payload: dict[str, object], snapshot: object) -> dict[str, object]:
    if not isinstance(snapshot, dict):
        raise ValueError("trained result lacks final saved adapter descriptor")
    expected_path = f"/artifacts/runs/{payload['run_id']}-{payload['role']}/adapter-update-1200"
    if (
        type(snapshot.get("update")) is not int
        or snapshot["update"] != 1200
        or snapshot.get("path") != expected_path
    ):
        raise ValueError("trained saved adapter path or update differs")
    hashes = snapshot.get("files_sha256")
    sizes = snapshot.get("files_size_bytes")
    required = {"adapter_model.safetensors", "adapter_config.json"}
    if (
        not isinstance(hashes, dict)
        or not required <= set(hashes)
        or set(hashes) - (required | {"README.md"})
        or not isinstance(sizes, dict)
        or set(sizes) != set(hashes)
        or any(
            not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            for digest in hashes.values()
        )
        or any(type(size) is not int or size <= 0 for size in sizes.values())
    ):
        raise ValueError("trained saved adapter files, sizes, or hashes are incomplete")
    return snapshot


def _require_training_evidence(payload: dict[str, object], result: dict[str, object]) -> None:
    from experiments import analyze_targeted_training as analysis
    from experiments.targeted_training_contracts import (
        BETAS,
        EPSILON,
        LEARNING_RATE,
        MAX_GRADIENT_NORM,
        TRAINING_SEED,
        WEIGHT_DECAY,
    )
    from experiments.targeted_training_runtime import validate_reload_parity

    role = payload["role"]
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("worker evidence is missing")
    if role == ROLE_UNCHANGED:
        if evidence != {
            "optimizer": None,
            "saved_adapter": None,
            "reload_parity": None,
            "adapter_update": None,
            "training_step_losses": [],
        }:
            raise ValueError("unchanged result contains training, save, or reload evidence")
        return
    optimizer = evidence.get("optimizer")
    expected_optimizer = {
        "lr": LEARNING_RATE,
        "betas": list(BETAS),
        "eps": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "max_gradient_norm": MAX_GRADIENT_NORM,
        "training_seed": TRAINING_SEED,
        "initial_state_empty": True,
        "scheduler": None,
        "updates": 1200,
        "microbatches_per_update": 4,
        "base_gradients_absent": True,
    }
    if optimizer != expected_optimizer:
        raise ValueError("trained optimizer/update evidence differs from fixed recipe")
    losses = evidence.get("training_step_losses")
    if not isinstance(losses, list) or len(losses) != 1200:
        raise ValueError("trained result lacks 1200 update-loss rows")
    for update, item in enumerate(losses, start=1):
        if (
            not isinstance(item, dict)
            or set(item) != {"update", "mean_loss"}
            or type(item["update"]) is not int
            or item["update"] != update
            or type(item["mean_loss"]) not in {float, int}
            or not math.isfinite(float(item["mean_loss"]))
            or item["mean_loss"] < 0
        ):
            raise ValueError("trained update-loss evidence is incomplete")
    changes = evidence.get("adapter_update")
    inventory_names = result["provenance"]["adapter_inventory"]["adapter_tensor_names"]
    # PEFT exposes module parameter names with a `.default` adapter component,
    # while get_peft_model_state_dict (used for the before/after proof) strips it.
    names = [
        name.replace(".lora_A.default.weight", ".lora_A.weight").replace(
            ".lora_B.default.weight", ".lora_B.weight"
        )
        for name in inventory_names
    ]
    if len(set(names)) != len(names):
        raise ValueError("PEFT saved-state name mapping collides")
    if (
        not isinstance(changes, dict)
        or type(changes.get("changed_tensor_count")) is not int
        or not 0 < changes["changed_tensor_count"] <= 120
        or not isinstance(changes.get("changed_tensor_names"), list)
        or len(changes["changed_tensor_names"]) != changes["changed_tensor_count"]
        or len(set(changes["changed_tensor_names"])) != changes["changed_tensor_count"]
        or not set(changes["changed_tensor_names"]) <= set(names)
    ):
        raise ValueError("trained adapter change proof is incomplete")
    _require_snapshot(payload, evidence.get("saved_adapter"))
    parity = evidence.get("reload_parity")
    if not isinstance(parity, dict) or set(parity) != {
        "tensor_values_exact",
        "winner_match",
        "max_candidate_logit_delta",
        "presentation_ids",
        "pre_save_tensor_sha256",
        "reloaded_tensor_sha256",
        "reload_outputs",
    }:
        raise ValueError("trained reload proof is incomplete")
    if (
        parity["tensor_values_exact"] is not True
        or parity["winner_match"] is not True
        or parity["presentation_ids"] != list(payload["reload_presentation_ids"])
        or not isinstance(parity["pre_save_tensor_sha256"], str)
        or not _SHA256.fullmatch(parity["pre_save_tensor_sha256"])
        or parity["reloaded_tensor_sha256"] != parity["pre_save_tensor_sha256"]
    ):
        raise ValueError("trained reload tensor or presentation proof differs")
    if type(parity["max_candidate_logit_delta"]) not in {float, int} or not math.isfinite(
        float(parity["max_candidate_logit_delta"])
    ):
        raise ValueError("trained reload score delta is malformed")
    by_id = {row["presentation_id"]: row for row in payload["evaluation_presentations"]}
    compiled_by_id = {
        row["presentation_id"]: row for row in payload["compiled"]["final_evaluation"]
    }
    ids = payload["reload_presentation_ids"]
    reload_rows = parity["reload_outputs"]
    analysis.validate_outputs(
        reload_rows,
        [by_id[identifier] for identifier in ids],
        [compiled_by_id[identifier] for identifier in ids],
    )
    final_by_id = {row["presentation_id"]: row for row in result["outputs"]}
    maximum = validate_reload_parity(
        {identifier: final_by_id[identifier] for identifier in ids},
        {row["presentation_id"]: row for row in reload_rows},
    )
    if parity["max_candidate_logit_delta"] != maximum:
        raise ValueError("trained reload score delta differs from raw parity rows")


def validate_completed_result(payload: object, result: object) -> dict[str, object]:
    """Adopt only a fully bound, independently recountable worker result."""
    from experiments import analyze_targeted_training as analysis

    frozen = validate_role_payload(payload)
    if not isinstance(result, dict):
        raise ValueError("worker result must be a JSON object")
    role = frozen["role"]
    if (
        result.get("experiment_id") != EXPERIMENT_ID
        or result.get("run_id") != frozen["run_id"]
        or result.get("role") != role
        or result.get("payload_sha256") != frozen["payload_sha256"]
    ):
        raise ValueError("worker run, role, or payload identity differs")
    accepted = validate_role_result(result, role)
    if result["execution"]["input_token_counts"] != frozen["compiled_input_token_counts"]:
        raise ValueError("actual input-token ledger differs from frozen compilation")
    if canonical_sha256(result.get("compiled")) != canonical_sha256(frozen["compiled"]):
        raise ValueError("worker compiled rows differ from frozen payload")
    analysis.validate_outputs(
        result.get("outputs"),
        frozen["evaluation_presentations"],
        frozen["compiled"]["final_evaluation"],
    )
    _require_provenance(frozen, result)
    _require_training_evidence(frozen, result)
    return accepted
