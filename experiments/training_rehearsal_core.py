"""CPU-only data and plan contracts for the synthetic LoRA mechanics rehearsal."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reflex_decisions.data import (
    DatasetSpec,
    DecisionRecord,
    SplitManifest,
    audit_splits,
)
from reflex_decisions.schema import DecisionRequest, Option

SOURCE_ID = "reflex-authored-rehearsal-v1"
DATASET_SUPPORT = "rehearsal-support-v1"
DATASET_INFRA = "rehearsal-infra-v1"
DEFAULT_RECORDS = "data/training/rehearsal-v1.jsonl"
DEFAULT_MANIFEST = "data/training/rehearsal-v1-manifest.json"
EXPECTED_PROTOCOL = "docs/training-rehearsal-protocol.md"
EXPECTED_RECORDS_SHA256 = "39c61a9b0f91442e34c006fa797011db06e4db6290f31b4b198ac3fdf7e72cc0"
EXPECTED_MANIFEST_SHA256 = "6270fe59969123650c014da96c37c9aeb3253b26e0425290ccc206b293d0ee2a"
EXPECTED_PROTOCOL_SHA256 = "15597062fc9f54a6d5d9fa7d94e1823d16bc8ca0aa199130faae2991a0f6c78a"
SOURCE_FINGERPRINT_MODULES = (
    ("experiments.training_rehearsal_core", "experiments/training_rehearsal_core.py"),
    ("experiments.modal_train_rehearsal", "experiments/modal_train_rehearsal.py"),
    ("experiments.baseline_qwen", "experiments/baseline_qwen.py"),
    ("experiments.modal_smoke", "experiments/modal_smoke.py"),
    ("reflex_decisions.data", "src/reflex_decisions/data.py"),
    ("reflex_decisions.rendering", "src/reflex_decisions/rendering.py"),
    ("reflex_decisions.schema", "src/reflex_decisions/schema.py"),
    ("reflex_decisions.scoring", "src/reflex_decisions/scoring.py"),
    ("reflex_decisions.smoke", "src/reflex_decisions/smoke.py"),
)
SOURCE_FINGERPRINT_PATHS = tuple(path for _module, path in SOURCE_FINGERPRINT_MODULES)
MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
SEED = 20261004
MAX_UPDATES = 128
MICROBATCHES_PER_UPDATE = 4
DIAGNOSTIC_UPDATES = (16, 32, 64, 128)
EARLY_STOP_ACCURACY = 0.95
TRAINING_PRESENTATIONS = MAX_UPDATES * MICROBATCHES_PER_UPDATE
DIAGNOSTIC_PRESENTATIONS = 2 * 64
MAX_FORWARD_COUNT = (
    TRAINING_PRESENTATIONS
    + DIAGNOSTIC_PRESENTATIONS * (1 + len(DIAGNOSTIC_UPDATES))
    + DIAGNOSTIC_PRESENTATIONS
)
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "out_proj",
)
EXPECTED_LORA_MODULES = 60
EXPECTED_ADAPTER_TENSORS = 120
EXPECTED_TRAINABLE_PARAMETERS = 2_015_232
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def source_fingerprints() -> dict[str, str]:
    """Fingerprint only the training path and the runtime helpers it uses."""

    root = Path(__file__).resolve().parents[1]
    paths = [root / relative_path for relative_path in SOURCE_FINGERPRINT_PATHS]
    if any(not path.is_file() for path in paths):
        raise ValueError("a required training rehearsal source file is missing")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


_TEMPLATES = (
    "I need help with my {problem}; please direct me to the right support team.",
    "A recent change caused a problem with my {problem}, and I need assistance.",
    "I cannot resolve the {problem} issue through the customer portal.",
    "Please help me fix the {problem} issue before my next scheduled use.",
)
_SUPPORT_TYPES = (
    ("invoice", "billing"),
    ("refund", "billing"),
    ("charge", "billing"),
    ("receipt", "billing"),
    ("password", "account"),
    ("sign-in", "account"),
    ("profile", "account"),
    ("email", "account"),
)
_INFRA_TYPES = (
    ("DNS lookup", "network"),
    ("firewall rule", "network"),
    ("packet delivery", "network"),
    ("gateway route", "network"),
    ("disk volume", "storage"),
    ("snapshot", "storage"),
    ("filesystem", "storage"),
    ("object store", "storage"),
)
_INFRA_TEMPLATES = (
    "The {problem} stopped working after a service deployment.",
    "We see repeated errors involving the {problem} on one host.",
    "A recent configuration change left the {problem} unavailable to the application.",
    "Please route the {problem} incident to the correct operations team.",
)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    record_id: str
    request: DecisionRequest
    gold_option_id: str
    gold_index: int


def build_fixture() -> tuple[SplitManifest, tuple[DecisionRecord, ...]]:
    """Build the fixed 64-example self-authored training-only corpus."""

    records: list[DecisionRecord] = []
    support_options = (
        Option(
            id="billing", label="Billing support", description="Invoices, charges, and refunds."
        ),
        Option(
            id="account",
            label="Account support",
            description="Access, profile, and email settings.",
        ),
    )
    infra_options = (
        Option(
            id="network",
            label="Network operations",
            description="DNS, firewall, packet, and routing.",
        ),
        Option(
            id="storage",
            label="Storage operations",
            description="Disks, snapshots, filesystems, and objects.",
        ),
    )
    for dataset_id, issue_types, options, question, templates in (
        (
            DATASET_SUPPORT,
            _SUPPORT_TYPES,
            support_options,
            "Which customer support team should handle this issue?",
            _TEMPLATES,
        ),
        (
            DATASET_INFRA,
            _INFRA_TYPES,
            infra_options,
            "Which infrastructure team should handle this incident?",
            _INFRA_TEMPLATES,
        ),
    ):
        kind = "support" if dataset_id == DATASET_SUPPORT else "infra"
        for problem, answer_id in issue_types:
            normalized_problem = problem.casefold().replace(" ", "-")
            for pattern_index, template in enumerate(templates, start=1):
                records.append(
                    DecisionRecord(
                        record_id=(f"{SOURCE_ID}-{kind}-{normalized_problem}-{pattern_index}"),
                        dataset_id=dataset_id,
                        source_group_id=f"{kind}-{normalized_problem}",
                        request=DecisionRequest(
                            context=template.format(problem=problem),
                            question=question,
                            options=options,
                        ),
                        answer_id=answer_id,
                    )
                )
    manifest = SplitManifest(
        data_kind="fixture",
        datasets=(
            DatasetSpec(
                dataset_id=DATASET_SUPPORT,
                source_id=SOURCE_ID,
                task_family="customer-support-routing",
                split="train",
                source_uri="local://reflex/training-rehearsal-v1/support",
                source_revision="self-authored-v1",
                license="Personal Reflex-authored synthetic fixture text; no employer data",
            ),
            DatasetSpec(
                dataset_id=DATASET_INFRA,
                source_id=SOURCE_ID,
                task_family="infrastructure-incident-routing",
                split="train",
                source_uri="local://reflex/training-rehearsal-v1/infrastructure",
                source_revision="self-authored-v1",
                license="Personal Reflex-authored synthetic fixture text; no employer data",
            ),
        ),
    )
    return manifest, tuple(records)


def serialize_records(records: Sequence[DecisionRecord]) -> bytes:
    lines = [
        json.dumps(
            record.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for record in records
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def serialize_manifest(manifest: SplitManifest) -> bytes:
    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def verify_prepared_data(
    records_path: str | Path = DEFAULT_RECORDS,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> tuple[SplitManifest, tuple[DecisionRecord, ...]]:
    """Require the exact pinned self-authored records and train-only manifest."""

    records_bytes = Path(records_path).read_bytes()
    manifest_bytes = Path(manifest_path).read_bytes()
    return verify_prepared_bytes(records_bytes, manifest_bytes)


def verify_prepared_bytes(
    records_bytes: bytes, manifest_bytes: bytes
) -> tuple[SplitManifest, tuple[DecisionRecord, ...]]:
    """Validate exact data bytes before the caller sends them to a training worker."""

    if not isinstance(records_bytes, bytes) or not isinstance(manifest_bytes, bytes):
        raise TypeError("training rehearsal payloads must be bytes")
    expected_manifest, expected_records = build_fixture()
    if hashlib.sha256(records_bytes).hexdigest() != EXPECTED_RECORDS_SHA256:
        raise ValueError("training rehearsal records SHA-256 mismatch")
    if hashlib.sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("training rehearsal manifest SHA-256 mismatch")
    if records_bytes != serialize_records(expected_records):
        raise ValueError("training rehearsal records differ from the self-authored recipe")
    if manifest_bytes != serialize_manifest(expected_manifest):
        raise ValueError("training rehearsal manifest differs from the pinned fixture")
    try:
        manifest = SplitManifest.model_validate_json(manifest_bytes)
        records = tuple(
            DecisionRecord.model_validate_json(line)
            for line in records_bytes.splitlines()
            if line.strip()
        )
    except Exception as exc:
        raise ValueError("training rehearsal payload has an invalid schema") from exc
    if manifest != expected_manifest or records != expected_records:
        raise ValueError("prepared training data does not match the approved fixture")
    if len(records) != 64 or any(dataset.split != "train" for dataset in manifest.datasets):
        raise ValueError("training rehearsal must contain 64 training-only records")
    audit_splits(manifest, records)
    return manifest, records


def verify_protocol(path: str | Path = EXPECTED_PROTOCOL) -> str:
    """Require the protocol document to match the checked-in training recipe pin."""

    try:
        protocol_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("training rehearsal protocol is unavailable") from exc
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("training rehearsal protocol SHA-256 mismatch")
    return digest


def validate_run_id(value: str) -> str:
    if not isinstance(value, str) or _SAFE_RUN_ID.fullmatch(value) is None:
        raise ValueError(
            "run ID must be 1–64 safe ASCII letters, digits, dots, underscores, or hyphens"
        )
    return value


def plan() -> dict[str, object]:
    return {
        "mode": "plan-only",
        "purpose": "synthetic training mechanics and fixture memorization only",
        "records": 64,
        "dataset_counts": {DATASET_SUPPORT: 32, DATASET_INFRA: 32},
        "data_sha256": {
            "records": EXPECTED_RECORDS_SHA256,
            "manifest": EXPECTED_MANIFEST_SHA256,
            "protocol": EXPECTED_PROTOCOL_SHA256,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "optimizer_updates_max": MAX_UPDATES,
        "microbatches_per_update": MICROBATCHES_PER_UPDATE,
        "train_forwards_max": TRAINING_PRESENTATIONS,
        "diagnostic_updates": list(DIAGNOSTIC_UPDATES),
        "max_forward_count": MAX_FORWARD_COUNT,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
            "max_gradient_norm": 1.0,
            "seed": SEED,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "bias": "none",
            "target_modules": list(LORA_TARGET_MODULES),
            "expected_modules": EXPECTED_LORA_MODULES,
            "expected_adapter_tensors": EXPECTED_ADAPTER_TENSORS,
            "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        },
        "modal": {
            "gpu": "A10",
            "cpu": [2.0, 2.0],
            "memory_mib": [16384, 16384],
            "max_containers": 1,
            "min_containers": 0,
            "buffer_containers": 0,
            "retries": 0,
            "single_use_containers": True,
            "startup_timeout_seconds": 300,
            "timeout_seconds": 1800,
            "volume": "reflex-rehearsal-artifacts",
        },
    }


def make_example(record: DecisionRecord, option_ids: Sequence[str]) -> TrainingExample:
    """Apply a semantic option order and remap its gold option to the candidate index."""

    original = {option.id: option for option in record.request.options}
    if any(not isinstance(option_id, str) for option_id in option_ids):
        raise ValueError("presented option IDs must be strings")
    if len(option_ids) != len(original) or set(option_ids) != set(original):
        raise ValueError("presented option IDs must be an exact permutation of record options")
    ordered_options = tuple(original[option_id] for option_id in option_ids)
    request = record.request.model_copy(update={"options": ordered_options})
    return TrainingExample(
        record_id=record.record_id,
        request=request,
        gold_option_id=record.answer_id,
        gold_index=list(option_ids).index(record.answer_id),
    )


def epoch_examples(
    records: Sequence[DecisionRecord], *, seed: int, epoch: int
) -> tuple[TrainingExample, ...]:
    """Shuffle records and their candidate order reproducibly for one training epoch."""

    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError("training seed and epoch must be integers, with a nonnegative epoch")
    rng = random.Random(seed + epoch)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)
    examples: list[TrainingExample] = []
    for record in shuffled_records:
        option_ids = [option.id for option in record.request.options]
        rng.shuffle(option_ids)
        examples.append(make_example(record, option_ids))
    return tuple(examples)
