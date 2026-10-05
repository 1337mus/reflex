"""CPU contracts for the bounded real-data LoRA pilot."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
from pathlib import Path

from experiments import training_rehearsal_core
from reflex_decisions.data import DecisionRecord, SplitManifest, audit_splits

SEED = 20261005
TRAIN_DATASET_IDS = ("dbpedia14-pilot-v1-train", "sms-pilot-v1-train")
TRAIN_RECORD_COUNT = 504
MICROBATCHES_PER_UPDATE = 4
MAX_UPDATES = 252
TRAINING_PRESENTATIONS = MICROBATCHES_PER_UPDATE * MAX_UPDATES
DATASET_RECORD_COUNTS = {
    "dbpedia14-pilot-v1-train": 252,
    "dbpedia14-pilot-v1-development": 56,
    "dbpedia14-pilot-v1-calibration": 56,
    "sms-pilot-v1-train": 252,
    "sms-pilot-v1-development": 60,
    "sms-pilot-v1-calibration": 60,
    "snli-pilot-v1-development": 128,
}
EVALUATION_PRESENTATION_COUNT = 2572
SPLIT_RECORD_COUNTS = {"train": 504, "development": 244, "calibration": 116, "test": 0}
MODEL_ID = training_rehearsal_core.MODEL_ID
MODEL_REVISION = training_rehearsal_core.MODEL_REVISION
EXPECTED_PROTOCOL_SHA256: str | None = (
    "88d3bb26a984c95f118dea7cd794d9ee82e38db31e87ecb265a52d49bd37632b"
)
PROTOCOL_PATH = "docs/real-pilot-protocol.md"
SOURCE_FINGERPRINT_MODULES = (
    ("experiments", "experiments/__init__.py"),
    ("experiments.real_pilot_core", "experiments/real_pilot_core.py"),
    ("experiments.real_pilot_contracts", "experiments/real_pilot_contracts.py"),
    ("experiments.modal_real_pilot", "experiments/modal_real_pilot.py"),
    ("experiments.modal_train_rehearsal", "experiments/modal_train_rehearsal.py"),
    ("experiments.training_rehearsal_core", "experiments/training_rehearsal_core.py"),
    ("experiments.baseline_qwen", "experiments/baseline_qwen.py"),
    ("experiments.modal_smoke", "experiments/modal_smoke.py"),
    ("reflex_decisions", "src/reflex_decisions/__init__.py"),
    ("reflex_decisions.data", "src/reflex_decisions/data.py"),
    ("reflex_decisions.broader_data", "src/reflex_decisions/broader_data.py"),
    ("reflex_decisions.pilot_data", "src/reflex_decisions/pilot_data.py"),
    ("reflex_decisions.rendering", "src/reflex_decisions/rendering.py"),
    ("reflex_decisions.schema", "src/reflex_decisions/schema.py"),
    ("reflex_decisions.scoring", "src/reflex_decisions/scoring.py"),
    ("reflex_decisions.smoke", "src/reflex_decisions/smoke.py"),
)
SOURCE_FINGERPRINT_PATHS = tuple(path for _module, path in SOURCE_FINGERPRINT_MODULES) + (
    PROTOCOL_PATH,
)
TRAIN_FORWARD_COUNT = TRAINING_PRESENTATIONS
BASE_EVALUATION_COUNT = EVALUATION_PRESENTATION_COUNT
FINAL_EVALUATION_COUNT = EVALUATION_PRESENTATION_COUNT
RELOAD_PARITY_COUNT = 32
MAX_FORWARD_COUNT = (
    TRAIN_FORWARD_COUNT + BASE_EVALUATION_COUNT + FINAL_EVALUATION_COUNT + RELOAD_PARITY_COUNT
)
MODAL_LIMITS = {
    "gpu": "A10",
    "cpu_physical_cores": [2.0, 2.0],
    "memory_mib": [16384, 16384],
    "max_containers": 1,
    "min_containers": 0,
    "buffer_containers": 0,
    "retries": 0,
    "single_use_containers": True,
    "startup_timeout_seconds": 300,
    "timeout_seconds": 3600,
    "volume": "reflex-rehearsal-artifacts",
}
MAX_INPUT_TOKENS = 2048


@dataclass(frozen=True, slots=True)
class PilotDataAudit:
    record_count: int
    split_counts: tuple[tuple[str, int], ...]


def source_fingerprints(root: str | Path | None = None) -> dict[str, str]:
    """Hash every uploaded Python dependency plus the frozen protocol document."""

    project_root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    paths = [project_root / relative for relative in SOURCE_FINGERPRINT_PATHS]
    if any(not path.is_file() for path in paths):
        raise ValueError("a required real pilot source file or protocol is missing")
    return {
        str(path.relative_to(project_root)): sha256(path.read_bytes()).hexdigest() for path in paths
    }


def verify_protocol(path: str | Path = PROTOCOL_PATH, *, require_frozen_pin: bool = True) -> str:
    try:
        digest = sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError("real pilot protocol is unavailable") from exc
    if require_frozen_pin and EXPECTED_PROTOCOL_SHA256 is None:
        raise ValueError("real pilot protocol hash has not been frozen in the runner")
    if EXPECTED_PROTOCOL_SHA256 is not None and digest != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("real pilot protocol SHA-256 mismatch")
    return digest


def plan() -> dict[str, object]:
    """Return the frozen compute/data contract without importing Modal or loading corpora."""

    return {
        "mode": "plan-only",
        "purpose": "one bounded real-data supervised LoRA feasibility pilot",
        "data": {
            "records": "data/processed/real-pilot-v1.jsonl",
            "manifest": "data/pilots/real-pilot-v1-manifest.json",
            "recipe": "data/pilots/real-pilot-v1-recipe.json",
            "record_count": sum(DATASET_RECORD_COUNTS.values()),
            "dataset_record_counts": DATASET_RECORD_COUNTS,
            "split_counts": SPLIT_RECORD_COUNTS,
            "official_tests_opened": False,
            "snli_role": "development diagnostic; not a sealed test",
            "financial_phrasebank_allowed": False,
        },
        "pins": {
            "protocol_path": PROTOCOL_PATH,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "source_fingerprint_paths": list(SOURCE_FINGERPRINT_PATHS),
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "training": {
            "seed": SEED,
            "optimizer_updates": MAX_UPDATES,
            "epochs": 2,
            "microbatches_per_update": MICROBATCHES_PER_UPDATE,
            "train_records": TRAIN_RECORD_COUNT,
            "train_forwards": TRAIN_FORWARD_COUNT,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": 5e-4,
                "weight_decay": 0.0,
                "max_gradient_norm": 1.0,
            },
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "target_modules": list(training_rehearsal_core.LORA_TARGET_MODULES),
                "expected_modules": training_rehearsal_core.EXPECTED_LORA_MODULES,
                "expected_adapter_tensors": training_rehearsal_core.EXPECTED_ADAPTER_TENSORS,
                "expected_trainable_parameters": (
                    training_rehearsal_core.EXPECTED_TRAINABLE_PARAMETERS
                ),
            },
            "snapshots_at_updates": [126, 252],
        },
        "evaluation": {
            "presentations_per_state": EVALUATION_PRESENTATION_COUNT,
            "reload_parity_presentations": RELOAD_PARITY_COUNT,
            "reload_logit_tolerance": 1e-3,
            "total_forward_limit": MAX_FORWARD_COUNT,
        },
        "modal": MODAL_LIMITS,
        "launch_requires_explicit_flag": True,
    }


def validate_pilot_data(
    manifest: SplitManifest, records: Sequence[DecisionRecord]
) -> PilotDataAudit:
    """Validate the fixed two-task pilot allocation and its split lineage."""

    from reflex_decisions import pilot_data

    if manifest.data_kind != "benchmark":
        raise ValueError("real pilot data must be marked as benchmark data")
    specs = {spec.dataset_id: spec for spec in manifest.datasets}
    if set(specs) != set(DATASET_RECORD_COUNTS):
        raise ValueError("pilot manifest does not match the approved dataset allowlist")
    if len(specs) != len(manifest.datasets):
        raise ValueError("pilot manifest dataset IDs must be unique")
    if set(pilot_data.DATASET_SPECS) != set(DATASET_RECORD_COUNTS):
        raise ValueError("pilot source module dataset allowlist differs from the runner")
    for dataset_id, (split, task_family) in pilot_data.DATASET_SPECS.items():
        spec = specs[dataset_id]
        if (spec.split, spec.task_family) != (split, task_family):
            raise ValueError(
                "pilot dataset split or task family does not match the frozen contract"
            )
    dbpedia_sources = {specs[dataset_id].source_id for dataset_id in pilot_data.DBPEDIA_DATASET_IDS}
    sms_sources = {specs[dataset_id].source_id for dataset_id in pilot_data.SMS_DATASET_IDS}
    snli_source = specs[pilot_data.SNLI_DATASET_ID].source_id
    if len(dbpedia_sources) != 1 or len(sms_sources) != 1:
        raise ValueError("official splits must retain one canonical source ID per corpus")
    if dbpedia_sources & sms_sources or snli_source in dbpedia_sources | sms_sources:
        raise ValueError("approved pilot datasets must come from distinct source identities")
    if (
        next(iter(dbpedia_sources)) != pilot_data.DBPEDIA_SOURCE_ID
        or next(iter(sms_sources)) != pilot_data.SMS_SOURCE_ID
        or snli_source != pilot_data.SNLI_SOURCE_ID
    ):
        raise ValueError("pilot manifest source IDs do not match the canonical source allowlist")
    if set(manifest.group_partitioned_sources) != dbpedia_sources | sms_sources:
        raise ValueError("only DBpedia14 and SMS may use the source-group split policy")

    if len(records) != sum(DATASET_RECORD_COUNTS.values()):
        raise ValueError("pilot data does not contain exactly 864 records")
    counts = Counter(record.dataset_id for record in records)
    if counts != Counter(DATASET_RECORD_COUNTS):
        raise ValueError("pilot records do not match the frozen split dataset counts")
    for record in records:
        expected_options = 14 if record.dataset_id.startswith("dbpedia14-") else 2
        if record.dataset_id == "snli-pilot-v1-development":
            expected_options = 3
        if len(record.request.options) != expected_options:
            raise ValueError("pilot record option count does not match its approved task")
    audit = audit_splits(manifest, list(records))
    split_counts = dict(audit.split_counts)
    if split_counts != SPLIT_RECORD_COUNTS:
        raise ValueError("pilot split totals do not match the frozen allocation")
    if audit.held_out_families:
        raise ValueError("SNLI is a development diagnostic, not a held-out test family")
    return PilotDataAudit(
        record_count=audit.record_count,
        split_counts=audit.split_counts,
    )


def _option_orders(dataset_id: str, record: DecisionRecord) -> tuple[tuple[object, ...], ...]:
    options = record.request.options
    if dataset_id.startswith("dbpedia14-"):
        if len(options) != 14:
            raise ValueError("DBpedia14 records must have exactly 14 options")
        if dataset_id.endswith("-calibration"):
            return (tuple(options),)
        forward = tuple(tuple(options[index:] + options[:index]) for index in range(len(options)))
        reversed_options = tuple(reversed(options))
        reverse = tuple(
            tuple(reversed_options[index:] + reversed_options[:index])
            for index in range(len(reversed_options))
        )
        return tuple(dict.fromkeys((*forward, *reverse)))
    if dataset_id.startswith("sms-"):
        if len(options) != 2:
            raise ValueError("SMS records must have exactly two options")
        if dataset_id.endswith("-calibration"):
            return (tuple(options),)
        return tuple(permutations(options))
    if dataset_id == "snli-pilot-v1-development":
        if len(options) != 3:
            raise ValueError("SNLI records must have exactly three options")
        return tuple(permutations(options))
    raise ValueError(f"unsupported evaluation dataset: {dataset_id}")


def build_evaluation_presentations(
    records: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    """Build the frozen development/calibration/SNLI request-only order panel."""

    counts = Counter(record.dataset_id for record in records)
    if counts != Counter(DATASET_RECORD_COUNTS):
        raise ValueError("pilot records do not match the frozen split dataset counts")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("pilot record IDs must be unique")

    presentations: list[dict[str, object]] = []
    for record in records:
        if record.dataset_id in TRAIN_DATASET_IDS:
            continue
        option_orders = _option_orders(record.dataset_id, record)
        for order_index, options in enumerate(option_orders):
            order_ids = [option.id for option in options]
            request = record.request.model_copy(update={"options": options})
            identity = "\0".join(
                ("reflex-real-pilot-presentation-v1", record.record_id, *order_ids)
            )
            presentations.append(
                {
                    "presentation_id": sha256(identity.encode("utf-8")).hexdigest(),
                    "record_id": record.record_id,
                    "dataset_id": record.dataset_id,
                    "request_hash": request.request_hash,
                    "order_index": order_index,
                    "order_ids": order_ids,
                    "request": request.model_dump(mode="json"),
                }
            )
    if len(presentations) != EVALUATION_PRESENTATION_COUNT:
        raise ValueError("evaluation panel does not match the frozen presentation count")
    if len({row["presentation_id"] for row in presentations}) != len(presentations):
        raise ValueError("evaluation presentation IDs are not unique")
    return tuple(presentations)


def build_training_schedule(
    records: Sequence[DecisionRecord],
) -> tuple[training_rehearsal_core.TrainingExample, ...]:
    """Build exactly two deterministic shuffled epochs of train-only records."""

    if len(records) != TRAIN_RECORD_COUNT:
        raise ValueError(f"training requires exactly {TRAIN_RECORD_COUNT} records")
    counts = Counter(record.dataset_id for record in records)
    if counts != Counter({TRAIN_DATASET_IDS[0]: 252, TRAIN_DATASET_IDS[1]: 252}):
        raise ValueError("training records must contain 252 examples from each approved task")
    if len({record.record_id for record in records}) != TRAIN_RECORD_COUNT:
        raise ValueError("training record IDs must be unique")

    schedule = tuple(
        example
        for epoch in range(2)
        for example in training_rehearsal_core.epoch_examples(records, seed=SEED, epoch=epoch)
    )
    if len(schedule) != TRAINING_PRESENTATIONS:
        raise ValueError("training schedule does not match the fixed update budget")
    return schedule


def build_training_payload(records: Sequence[DecisionRecord]) -> tuple[dict[str, object], ...]:
    from experiments.real_pilot_contracts import build_training_payload as build

    return build(records)


def build_remote_payload(*args: object, **kwargs: object) -> dict[str, object]:
    from experiments.real_pilot_contracts import build_remote_payload as build

    return build(*args, **kwargs)  # type: ignore[arg-type]


def validate_remote_payload(
    payload: object, *, expected_identity: dict[str, object] | None = None
) -> dict[str, object]:
    from experiments.real_pilot_contracts import validate_remote_payload as validate

    return validate(payload, expected_identity=expected_identity)


def validate_passed_receipt(
    value: object, *, expected_payload: dict[str, object]
) -> dict[str, object]:
    from experiments.real_pilot_contracts import validate_passed_receipt as validate

    return validate(value, expected_payload=expected_payload)


def validate_failed_receipt(
    value: object, *, expected_payload: dict[str, object]
) -> dict[str, object]:
    from experiments.real_pilot_contracts import validate_failed_receipt as validate

    return validate(value, expected_payload=expected_payload)


def validate_recovery_receipt(
    value: object, *, expected_identity: dict[str, object]
) -> dict[str, object]:
    from experiments.real_pilot_contracts import validate_recovery_receipt as validate

    return validate(value, expected_identity=expected_identity)


def validate_run_id(value: object) -> str:
    from experiments.real_pilot_contracts import validate_run_id as validate

    return validate(value)


def validate_nonce(value: object) -> str:
    from experiments.real_pilot_contracts import validate_nonce as validate

    return validate(value)
