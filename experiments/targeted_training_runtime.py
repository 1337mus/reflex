"""Small GPU-worker primitives for targeted matched training.

This module intentionally imports neither Modal nor model libraries at import time.
The Modal entrypoint injects those dependencies only inside an authorized worker.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from experiments.targeted_training_contracts import (
    BETAS,
    EPSILON,
    LEARNING_RATE,
    MAX_GRADIENT_NORM,
    SELECTED_ADAPTER_TENSOR_SHA256,
    WEIGHT_DECAY,
)
from experiments.targeted_training_core import (
    ROLE_UNCHANGED,
    TRAINING_FORWARDS,
    compile_presentations,
    expected_forward_counts,
    targeted_source_manifest,
    validate_completed_counts,
    validate_role_payload,
)


@dataclass
class PendingForwardLedger:
    """Conservative progress ledger: never infer completion after an interruption."""

    role: str
    completed: dict[str, int] = field(init=False)
    input_token_counts: dict[str, int] = field(init=False)
    attempted_unknown: dict[str, int] = field(init=False)
    failure: dict[str, str] | None = field(default=None, init=False)
    pending: dict[str, str] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.completed = {
            category: 0 for category in expected_forward_counts(self.role) if category != "total"
        }
        self.attempted_unknown = {category: 0 for category in self.completed}
        self.input_token_counts = {category: 0 for category in self.completed}

    @property
    def not_started(self) -> dict[str, int]:
        caps = expected_forward_counts(self.role)
        return {
            category: caps[category] - self.completed[category] - self.attempted_unknown[category]
            for category in self.completed
        }

    def complete(self, category: str, count: int = 1, *, input_tokens: int = 0) -> None:
        caps = expected_forward_counts(self.role)
        if self.failure is not None:
            raise ValueError("cannot complete a forward after a recorded failure")
        if (
            category not in self.completed
            or type(count) is not int
            or count <= 0
            or type(input_tokens) is not int
            or not 0 <= input_tokens <= 2048 * count
        ):
            raise ValueError("forward category or count is malformed")
        if self.completed[category] + count > caps[category]:
            raise ValueError("forward completion exceeds its immutable cap")
        self.completed[category] += count
        self.input_token_counts[category] += input_tokens
        self.pending = None

    def begin(self, category: str, presentation_id: str) -> None:
        if category not in self.completed or not presentation_id or self.pending is not None:
            raise ValueError("pending forward identity is malformed")
        if (
            self.completed[category] + self.attempted_unknown[category]
            >= expected_forward_counts(self.role)[category]
        ):
            raise ValueError("pending forward exceeds immutable category cap")
        self.pending = {"category": category, "presentation_id": presentation_id}

    def fail_pending(self, category: str, presentation_id: str) -> None:
        if (
            category not in self.completed
            or not isinstance(presentation_id, str)
            or not presentation_id
        ):
            raise ValueError("failed forward identity is malformed")
        if self.failure is not None:
            raise ValueError("only the first failed pending forward may be recorded")
        self.failure = {"category": category, "presentation_id": presentation_id}
        self.attempted_unknown[category] = 1
        self.pending = None

    def receipt(self) -> dict[str, object]:
        token_counts = dict(self.input_token_counts)
        token_counts["total"] = sum(token_counts.values())
        return {
            "completed": dict(self.completed),
            "attempted_unknown": dict(self.attempted_unknown),
            "not_started": self.not_started,
            "input_token_counts": token_counts,
            "failure": self.failure,
        }

    def require_complete(self) -> dict[str, int]:
        """Accept a ledger only when no forward remains unknown."""

        if self.failure is not None:
            raise ValueError("a worker with a failed pending forward cannot complete")
        counts = dict(self.completed)
        counts["total"] = sum(counts.values())
        return validate_completed_counts(self.role, counts)


def run_scored_forward(
    ledger: PendingForwardLedger,
    category: str,
    presentation: object,
    tokenizer: object,
    forward: object,
) -> object:
    """Compile before the model call and persist conservative failure evidence.

    ``forward`` is injected so this host-safe module remains importable without
    Torch.  A GPU worker supplies a single candidate-only forward callable.
    """

    if not callable(forward) or not isinstance(presentation, Mapping):
        raise ValueError("scored forward dependencies are malformed")
    compiled = compile_presentations((dict(presentation),), tokenizer)
    if len(compiled) != 1:
        raise ValueError("compiled forward accounting is incomplete")
    presentation_id = presentation.get("presentation_id")
    if not isinstance(presentation_id, str) or not presentation_id:
        raise ValueError("scored forward identity is malformed")
    ledger.begin(category, presentation_id)
    try:
        result = forward(presentation, compiled[0])
    except BaseException:
        ledger.fail_pending(category, presentation_id)
        raise
    ledger.complete(category, input_tokens=compiled[0]["input_tokens"])
    return result


def _checked_score(
    scoring: object,
    model: object,
    row: Mapping[str, object],
    expected: object,
    tokenizer: object,
    torch: object,
    progress: dict[str, object],
    ledger: PendingForwardLedger,
    category: str,
) -> tuple[object, object]:
    """Recompile and bind a pending forward before the only model invocation."""
    from reflex_decisions.schema import DecisionRequest

    if not isinstance(expected, Mapping):
        raise ValueError("frozen compiled forward is malformed")
    actual = compile_presentations((dict(row),), tokenizer)[0]
    if actual != dict(expected):
        raise ValueError("remote compilation differs from frozen payload accounting")
    presentation_id = row.get("presentation_id")
    if not isinstance(presentation_id, str):
        raise ValueError("forward presentation ID is malformed")
    request = DecisionRequest.model_validate(row["request"])
    ledger.begin(category, presentation_id)
    try:
        logits, compiled = scoring._score_forward(
            model, request, tokenizer, torch, progress, category
        )
        if (
            compiled.request_hash != expected["request_hash"]
            or compiled.prompt_hash != expected["prompt_sha256"]
            or len(compiled.input_ids) != expected["input_tokens"]
            or list(compiled.candidate_token_ids) != expected["candidate_token_ids"]
        ):
            raise ValueError("scoring returned a different compiled request")
    except BaseException:
        ledger.fail_pending(category, presentation_id)
        raise
    ledger.complete(category, input_tokens=actual["input_tokens"])
    return logits, compiled


def validate_optimizer_configuration(value: object) -> dict[str, object]:
    """Require exactly the fresh, scheduler-free AdamW recipe before training."""

    if not isinstance(value, Mapping):
        raise ValueError("optimizer configuration is malformed")
    expected = {
        "lr": LEARNING_RATE,
        "betas": list(BETAS),
        "eps": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "max_gradient_norm": MAX_GRADIENT_NORM,
        "state": {},
    }
    normalized = dict(value)
    if "max_gradient_norm" not in normalized:
        normalized["max_gradient_norm"] = MAX_GRADIENT_NORM
    if normalized != expected:
        if normalized.get("state") not in ({}, None):
            raise ValueError("optimizer state must be empty before the first update")
        raise ValueError("optimizer configuration differs from the frozen AdamW recipe")
    return expected


def validate_training_schedule(role: object, schedule: object) -> tuple[dict[str, object], ...]:
    """Validate the exact four-microbatch shape without reading labels from another arm."""
    from reflex_decisions.schema import DecisionRequest

    if role == ROLE_UNCHANGED:
        if schedule not in ((), [], None):
            raise ValueError("unchanged reference cannot receive training rows")
        return ()
    if not isinstance(schedule, (tuple, list)) or len(schedule) != TRAINING_FORWARDS:
        raise ValueError("trained arm must receive exactly 4,800 loss rows")
    rows: list[dict[str, object]] = []
    identifiers: set[str] = set()
    required = {
        "presentation_id",
        "record_id",
        "dataset_id",
        "source_group_id",
        "request_hash",
        "order_index",
        "order_ids",
        "request",
        "gold_option_id",
        "gold_index",
        "update",
        "microbatch_index",
    }
    for position, value in enumerate(schedule):
        if not isinstance(value, Mapping):
            raise ValueError("training row is malformed")
        row = dict(value)
        update = row.get("update")
        microbatch = row.get("microbatch_index")
        identifier = row.get("presentation_id")
        if (
            type(update) is not int
            or update != position // 4 + 1
            or type(microbatch) is not int
            or microbatch != position % 4
            or not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or identifier != f"targeted-reasoning-v1:{role}:update-{update}:micro-{microbatch}"
            or set(row) != required
            or row.get("order_index") != 0
            or type(row.get("order_index")) is not int
        ):
            raise ValueError("training schedule ordering or labels are malformed")
        request = DecisionRequest.model_validate(row["request"])
        order_ids = row["order_ids"]
        gold_index = row["gold_index"]
        gold_option_id = row["gold_option_id"]
        if (
            not isinstance(order_ids, list)
            or order_ids != [option.id for option in request.options]
            or row["request_hash"] != request.request_hash
            or type(gold_index) is not int
            or not 0 <= gold_index < len(order_ids)
            or order_ids[gold_index] != gold_option_id
        ):
            raise ValueError("training row gold index, order, or request differs")
        identifiers.add(identifier)
        rows.append(row)
    return tuple(rows)


def validate_reload_parity(reference: object, reloaded: object, tolerance: float = 1e-6) -> float:
    """Require semantic winner equality and bounded candidate-score drift."""

    if type(tolerance) not in {float, int} or tolerance < 0:
        raise ValueError("reload parity tolerance is malformed")
    if (
        not isinstance(reference, Mapping)
        or not isinstance(reloaded, Mapping)
        or set(reference) != set(reloaded)
    ):
        raise ValueError("reload parity outputs have incompatible presentation identities")
    maximum = 0.0
    for identifier in sorted(reference):
        before, after = reference[identifier], reloaded[identifier]
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise ValueError("reload parity rows are malformed")
        if before.get("winner_option_id") != after.get("winner_option_id"):
            raise ValueError("reload parity changed a semantic winner")
        left, right = before.get("candidate_logits"), after.get("candidate_logits")
        if (
            not isinstance(left, (tuple, list))
            or not isinstance(right, (tuple, list))
            or len(left) != len(right)
        ):
            raise ValueError("reload parity candidate scores are malformed")
        for old, new in zip(left, right, strict=True):
            if type(old) not in {int, float} or type(new) not in {int, float}:
                raise ValueError("reload parity candidate score is malformed")
            delta = abs(float(old) - float(new))
            if not math.isfinite(delta):
                raise ValueError("reload parity candidate score is non-finite")
            maximum = max(maximum, delta)
    if maximum > tolerance:
        raise ValueError("reload parity candidate score delta exceeds tolerance")
    return maximum


def _verify_preimport(payload: dict[str, object], source_root: Path) -> None:
    """Recheck source bytes and selected snapshot files before any model import."""
    if targeted_source_manifest(source_root) != payload["source_pins"]:
        raise ValueError("worker source files differ from frozen source pins")
    snapshot = payload["selection"]["snapshot"]
    directory = Path(snapshot["path"])
    files = snapshot["files_sha256"]
    if not directory.is_dir() or not isinstance(files, dict):
        raise ValueError("selected adapter snapshot files are unavailable")
    actual = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }
    if actual != files:
        raise ValueError("selected adapter snapshot file hashes differ from pin")


def _load_dependencies() -> dict[str, object]:
    """Load GPU packages only after immutable CPU preflight succeeds."""
    transformers = importlib.import_module("transformers")
    return {
        "torch": importlib.import_module("torch"),
        "functional": importlib.import_module("torch.nn.functional"),
        "peft": importlib.import_module("peft"),
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoTokenizer": transformers.AutoTokenizer,
        "execution": importlib.import_module("experiments.mixture_training_runtime_execution"),
        "scoring": importlib.import_module("experiments.mixture_training_runtime_scoring"),
        "rehearsal": importlib.import_module("experiments.modal_train_rehearsal"),
        "helpers": importlib.import_module("experiments.mixture_training_runtime_helpers"),
    }


def run_worker(
    payload: object,
    *,
    source_root: Path = Path("/root"),
    artifact_root: Path = Path("/artifacts"),
    preimport_verifier: object = None,
    dependency_loader: object = None,
) -> dict[str, object]:
    """Execute one real, bounded role worker on its injected Modal GPU image.

    Model packages are imported only after the immutable payload has been
    checked.  Any exception returns raw partial evidence with one pending
    attempted forward, allowing the host to retain it without retrying.
    """
    from experiments.targeted_training_core import validate_completed_result

    payload = validate_role_payload(payload)
    verifier = _verify_preimport if preimport_verifier is None else preimport_verifier
    if not callable(verifier):
        raise ValueError("worker preimport verifier is malformed")
    verifier(payload, source_root)
    load = _load_dependencies if dependency_loader is None else dependency_loader
    if not callable(load):
        raise ValueError("worker dependency loader is malformed")
    role = payload["role"]
    schedules = validate_training_schedule(role, payload.get("training"))
    presentations = payload.get("evaluation_presentations")
    compiled = payload.get("compiled")
    if not isinstance(presentations, (tuple, list)) or not isinstance(compiled, Mapping):
        raise ValueError("worker payload presentations or compilation are malformed")
    if len(presentations) != expected_forward_counts(role)["final_evaluation"]:
        raise ValueError("worker payload final panel has the wrong count")
    ledger = PendingForwardLedger(role)

    progress = {
        "evidence": {
            "forward_counts": {
                "training": 0,
                "final_evaluation": 0,
                "reload_parity": 0,
                "base_evaluation": 0,
                "total": 0,
            },
            "input_token_counts": {
                "training": 0,
                "final_evaluation": 0,
                "reload_parity": 0,
                "base_evaluation": 0,
                "total": 0,
            },
        }
    }
    result: dict[str, object] = {
        "status": "failed",
        "experiment_id": payload.get("experiment_id"),
        "run_id": payload.get("run_id"),
        "role": role,
        "payload_sha256": payload.get("payload_sha256"),
        "outputs": [],
        "compiled": compiled,
        "provenance": {},
        "evidence": {
            "optimizer": None,
            "saved_adapter": None,
            "reload_parity": None,
            "adapter_update": None,
            "training_step_losses": [],
        },
    }
    stage = "load_dependencies"
    run_dir: Path | None = None
    helpers: object = None
    try:
        deps = load()
        if not isinstance(deps, Mapping) or set(deps) != {
            "torch",
            "functional",
            "peft",
            "AutoModelForCausalLM",
            "AutoTokenizer",
            "execution",
            "scoring",
            "rehearsal",
            "helpers",
        }:
            raise ValueError("worker dependency bundle is malformed")
        torch = deps["torch"]
        functional = deps["functional"]
        peft = deps["peft"]
        AutoModelForCausalLM = deps["AutoModelForCausalLM"]
        AutoTokenizer = deps["AutoTokenizer"]
        execution = deps["execution"]
        scoring = deps["scoring"]
        rehearsal = deps["rehearsal"]
        helpers = deps["helpers"]
        stage = "load_model"
        model, tokenizer, provenance = execution._base_model(
            AutoModelForCausalLM, AutoTokenizer, torch
        )
        selection = payload.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("worker selection is malformed")
        snapshot = selection.get("snapshot")
        adapter_path = snapshot.get("path") if isinstance(snapshot, Mapping) else None
        expected_digest = SELECTED_ADAPTER_TENSOR_SHA256
        if not isinstance(adapter_path, str):
            raise ValueError("selected adapter descriptor is malformed")
        adapted = peft.PeftModel.from_pretrained(
            model, adapter_path, is_trainable=role != ROLE_UNCHANGED, autocast_adapter_dtype=True
        )
        inventory = rehearsal._validate_adapter_inventory(
            adapted, torch, require_trainable=role != ROLE_UNCHANGED
        )
        initial_state = scoring._verify_saved_adapter(
            adapted, dict(snapshot), expected_digest, peft, torch
        )
        result["provenance"] = {
            "base_model": provenance,
            "source_pins_sha256": hashlib.sha256(
                json.dumps(payload["source_pins"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "adapter_tensor_sha256": expected_digest,
            "adapter_inventory": inventory,
        }
        from experiments.targeted_training_contracts import TRAINING_SEED

        random.seed(TRAINING_SEED)
        torch.manual_seed(TRAINING_SEED)
        torch.cuda.manual_seed_all(TRAINING_SEED)
        if role != ROLE_UNCHANGED:
            stage = "training"
            run_id = payload["run_id"]
            run_dir = artifact_root / "runs" / f"{run_id}-{role}"
            run_dir.mkdir(parents=True, exist_ok=False)
            trainable = [parameter for parameter in adapted.parameters() if parameter.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable, lr=LEARNING_RATE, betas=BETAS, eps=EPSILON, weight_decay=WEIGHT_DECAY
            )
            if optimizer.state:
                raise RuntimeError("fresh optimizer unexpectedly has state")
            adapted.train()
            result["evidence"]["training_step_losses"] = []
            for update in range(1, 1201):
                optimizer.zero_grad(set_to_none=True)
                batch = schedules[(update - 1) * 4 : update * 4]
                if len(batch) != 4:
                    raise RuntimeError("training update has fewer than four microbatches")
                losses: list[float] = []
                for offset, row in enumerate(batch):
                    logits, _ = _checked_score(
                        scoring,
                        adapted,
                        row,
                        compiled["training"][(update - 1) * 4 + offset],
                        tokenizer,
                        torch,
                        progress,
                        ledger,
                        "training",
                    )
                    loss = functional.cross_entropy(
                        logits.unsqueeze(0), torch.tensor([row["gold_index"]], device="cuda")
                    )
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError("non-finite training loss")
                    (loss / 4).backward()
                    losses.append(float(loss.detach().item()))
                    if any(
                        parameter.grad is not None
                        for name, parameter in adapted.named_parameters()
                        if ".lora_A." not in name and ".lora_B." not in name
                    ):
                        raise RuntimeError("a frozen base parameter received a gradient")
                norm = torch.nn.utils.clip_grad_norm_(trainable, MAX_GRADIENT_NORM)
                if not bool(torch.isfinite(norm).item()):
                    raise RuntimeError("non-finite gradient norm")
                optimizer.step()
                result["evidence"]["training_step_losses"].append(
                    {"update": update, "mean_loss": sum(losses) / len(losses)}
                )
                if update % 50 == 0 or update == 1200:
                    helpers.write_json_atomic(
                        run_dir / "progress.json",
                        {
                            "run_id": run_id,
                            "role": role,
                            "phase": "training",
                            "completed_updates": update,
                            "execution": ledger.receipt(),
                            "training_step_losses": result["evidence"]["training_step_losses"],
                        },
                        replace=update != 50,
                    )
            result["evidence"]["optimizer"] = {
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
            final_training_state = scoring._state_dict(adapted, peft)
            changes = rehearsal._adapter_state_changes(initial_state, final_training_state, torch)
            if changes["changed_tensor_count"] == 0:
                raise RuntimeError("training did not change any adapter tensor")
            result["evidence"]["adapter_update"] = changes
        stage = "final_evaluation"
        adapted.eval()
        outputs: list[dict[str, object]] = []
        result["outputs"] = outputs
        for index, row in enumerate(presentations):
            with torch.inference_mode():
                logits, compiled_request = _checked_score(
                    scoring,
                    adapted,
                    row,
                    compiled["final_evaluation"][index],
                    tokenizer,
                    torch,
                    progress,
                    ledger,
                    "final_evaluation",
                )
            outputs.append(scoring._scored_row(row, logits, compiled_request))
            if role != ROLE_UNCHANGED:
                with (run_dir / "final_scores.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(outputs[-1], sort_keys=True, separators=(",", ":")) + "\n"
                    )
        result["outputs"] = outputs
        if role != ROLE_UNCHANGED:
            stage = "save_reload"
            saved = rehearsal._save_adapter_snapshot(adapted, run_dir, 1200)
            saved["files_size_bytes"] = {
                name: (Path(saved["path"]) / name).stat().st_size for name in saved["files_sha256"]
            }
            result["evidence"]["saved_adapter"] = saved
            final_state = scoring._state_dict(adapted, peft)
            final_digest = helpers.tensor_state_sha256(final_state, torch_module=torch)
            fresh_base, fresh_tokenizer, _ = execution._base_model(
                AutoModelForCausalLM, AutoTokenizer, torch
            )
            reloaded = peft.PeftModel.from_pretrained(
                fresh_base, str(saved["path"]), is_trainable=False, autocast_adapter_dtype=True
            ).eval()
            rehearsal._validate_adapter_inventory(reloaded, torch, require_trainable=False)
            reloaded_state = scoring._verify_saved_adapter(
                reloaded, saved, final_digest, peft, torch
            )
            scoring._equal_states(final_state, reloaded_state, torch)
            reloaded_digest = helpers.tensor_state_sha256(reloaded_state, torch_module=torch)
            final_by_id = {row["presentation_id"]: row for row in outputs}
            reload_ids = payload.get("reload_presentation_ids")
            if not isinstance(reload_ids, (tuple, list)) or len(reload_ids) != 32:
                raise RuntimeError("reload parity IDs are malformed")
            presentation_by_id = {row["presentation_id"]: row for row in presentations}
            reload_outputs: dict[str, object] = {}
            expected_final = {
                item["presentation_id"]: item for item in compiled["final_evaluation"]
            }
            for identifier in reload_ids:
                row = presentation_by_id.get(identifier)
                if not isinstance(row, Mapping) or identifier not in final_by_id:
                    raise RuntimeError("reload parity presentation is absent from final scoring")
                with torch.inference_mode():
                    logits, compiled_request = _checked_score(
                        scoring,
                        reloaded,
                        row,
                        expected_final[identifier],
                        fresh_tokenizer,
                        torch,
                        progress,
                        ledger,
                        "reload_parity",
                    )
                reload_outputs[str(identifier)] = scoring._scored_row(row, logits, compiled_request)
            maximum = validate_reload_parity(
                {identifier: final_by_id[identifier] for identifier in reload_ids}, reload_outputs
            )
            result["evidence"]["reload_parity"] = {
                "tensor_values_exact": True,
                "winner_match": True,
                "max_candidate_logit_delta": maximum,
                "presentation_ids": list(reload_ids),
                "pre_save_tensor_sha256": final_digest,
                "reloaded_tensor_sha256": reloaded_digest,
                "reload_outputs": list(reload_outputs.values()),
            }
        completed = ledger.require_complete()
        zero_categories = {category: 0 for category in ledger.completed}
        token_counts = dict(ledger.input_token_counts)
        token_counts["total"] = sum(token_counts.values())
        result["execution"] = {
            "completed": completed,
            "attempted_unknown": zero_categories,
            "not_started": zero_categories,
            "input_token_counts": token_counts,
        }
        result["status"] = "passed"
        return validate_completed_result(payload, result)
    except BaseException as error:
        result["execution"] = ledger.receipt()
        result["failure"] = {"type": type(error).__name__, "stage": stage}
        if run_dir is not None and run_dir.is_dir() and helpers is not None:
            try:
                helpers.write_json_atomic(run_dir / "failure-receipt.json", result, replace=False)
            except BaseException as persistence_error:
                result["failure"]["persistence_failure"] = type(persistence_error).__name__
        return result
