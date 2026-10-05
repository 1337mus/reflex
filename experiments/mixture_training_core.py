"""CPU payload construction and verification for controlled mixture experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from experiments import mixture_training_data as mixture_data
from experiments import training_rehearsal_core
from experiments.mixture_training_contracts import (
    DATA_FILE_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    MAX_INPUT_TOKENS,
    MAX_TOTAL_FORWARDS,
    MODEL_ID,
    MODEL_REVISION,
    PROFILE,
    PROTOCOL_PATH,
    RELOAD_PARITY_COUNT,
    RUNTIME_VERSION_PINS,
    SCHEMA_VERSION,
    SOURCE_FINGERPRINT_PATHS,
    WORKSPACE,
    json_sha256,
    normalize_json_object,
    source_fingerprints,
    validate_pins,
    validate_safe_run_id,
    validate_sha256,
    validate_uuid,
    verify_protocol,
)
from experiments.mixture_training_evaluations import _validate_evaluation_rows
from experiments.mixture_training_inputs import load_local_data
from reflex_decisions.data import DecisionRecord

_PAYLOAD_FIELDS = {
    "schema_version",
    "experiment_id",
    "run_id",
    "nonce",
    "phase",
    "arm",
    "pins",
    "train_records",
    "evaluation_presentations",
    "schedule_sha256",
    "initialization",
    "payload_sha256",
}
_TRAIN_RECORD_FIELDS = {"record_id", "dataset_id", "source_group_id", "request", "answer_id"}
_INITIALIZATION_FIELDS = {"snapshot", "tensor_sha256"}
_SNAPSHOT_FIELDS = {"update", "path", "files_sha256"}
_SNAPSHOT_FILES = {"adapter_config.json", "adapter_model.safetensors"}


def _train_record_dict(value: object) -> dict[str, object]:
    if isinstance(value, DecisionRecord):
        row = value.model_dump(mode="json")
    else:
        row = normalize_json_object(value, "training record")
    if set(row) != _TRAIN_RECORD_FIELDS:
        raise ValueError("training record has an unexpected schema")
    try:
        record = DecisionRecord.model_validate(row)
    except Exception as exc:
        raise ValueError("training record does not match the decision record schema") from exc
    normalized = record.model_dump(mode="json")
    if normalized != row:
        raise ValueError("training record is not in canonical JSON form")
    if not all(
        isinstance(row[name], str) and row[name]
        for name in ("record_id", "dataset_id", "source_group_id", "answer_id")
    ):
        raise ValueError("training record identifiers must be nonempty strings")
    return normalized


def _record_from_row(row: Mapping[str, object]) -> DecisionRecord:
    return DecisionRecord.model_validate(dict(row))


def _validate_snapshot(value: object, *, experiment_id: str, update: int) -> dict[str, object]:
    snapshot = normalize_json_object(value, "adapter snapshot")
    if set(snapshot) != _SNAPSHOT_FIELDS:
        raise ValueError("adapter snapshot has an unexpected schema")
    if type(snapshot["update"]) is not int or snapshot["update"] != update:
        raise ValueError("adapter snapshot update does not match the required checkpoint")
    expected_path = f"/artifacts/runs/{experiment_id}-init/adapter-update-000"
    if snapshot["path"] != expected_path:
        raise ValueError("initial adapter snapshot path is not canonical")
    files = snapshot["files_sha256"]
    if not isinstance(files, dict) or not _SNAPSHOT_FILES.issubset(files):
        raise ValueError("adapter snapshot is missing required file hashes")
    if set(files) - (_SNAPSHOT_FILES | {"README.md"}):
        raise ValueError("adapter snapshot contains an unexpected file")
    for name, digest in files.items():
        validate_sha256(digest, f"adapter snapshot {name} SHA-256")
    return snapshot


def _validate_initialization(value: object, experiment_id: str) -> dict[str, object]:
    descriptor = normalize_json_object(value, "initialization descriptor")
    if set(descriptor) != _INITIALIZATION_FIELDS:
        raise ValueError("initialization descriptor has an unexpected schema")
    snapshot = _validate_snapshot(descriptor["snapshot"], experiment_id=experiment_id, update=0)
    tensor_sha = validate_sha256(descriptor["tensor_sha256"], "initial tensor SHA-256")
    return {"snapshot": snapshot, "tensor_sha256": tensor_sha}


def _validate_training_rows(
    value: object,
) -> tuple[list[dict[str, object]], tuple[DecisionRecord, ...]]:
    if not isinstance(value, list) or len(value) != 1004:
        raise ValueError("training payload must contain exactly 1,004 approved rows")
    rows = [_train_record_dict(item) for item in value]
    records = tuple(_record_from_row(row) for row in rows)
    real_records = tuple(
        row for row in records if row.dataset_id in mixture_data.REAL_TRAIN_DATASET_COUNTS
    )
    synthetic_records = tuple(
        row for row in records if row.dataset_id in mixture_data.SYNTHETIC_TRAIN_DATASET_COUNTS
    )
    if len(real_records) + len(synthetic_records) != len(records):
        raise ValueError("training records contain a dataset outside the exact allowlist")
    mixture_data.build_training_schedules(real_records, synthetic_records)
    return rows, records


def _validate_group_separation(
    train_records: Sequence[DecisionRecord], presentations: Sequence[Mapping[str, object]]
) -> None:
    train_ids = {record.record_id for record in train_records}
    train_groups = {record.source_group_id for record in train_records}
    train_requests = {record.request.request_hash for record in train_records}
    eval_ids = {str(row["record_id"]) for row in presentations}
    eval_groups = {str(row["source_group_id"]) for row in presentations}
    eval_requests = {str(row["request_hash"]) for row in presentations}
    if train_ids & eval_ids:
        raise ValueError("training and evaluation record IDs must be disjoint")
    if train_groups & eval_groups:
        raise ValueError("training and evaluation source groups must be disjoint")
    if train_requests & eval_requests:
        raise ValueError("training and evaluation semantic requests must be disjoint")


def _normalize_identity(
    *, experiment_id: object, run_id: object, nonce: object, phase: object, arm: object
) -> tuple[str, str, str, str, str | None]:
    experiment = validate_safe_run_id(experiment_id, "experiment_id")
    if len(experiment) > 59:
        raise ValueError("experiment_id is too long for derived run IDs")
    if phase not in {"initialize", "train"}:
        raise ValueError("phase must be initialize or train")
    if phase == "initialize":
        if arm is not None:
            raise ValueError("initialization payload arm must be null")
        suffix = "-init"
        normalized_arm = None
    else:
        if arm not in mixture_data.ARM_NAMES:
            raise ValueError("training arm is outside the exact allowlist")
        normalized_arm = str(arm)
        suffix = "-real" if arm == "real_only" else "-mix"
    expected_run = experiment + suffix
    if validate_safe_run_id(run_id) != expected_run:
        raise ValueError("run_id does not match experiment phase and arm")
    normalized_nonce = validate_uuid(nonce, "nonce")
    if normalized_nonce == expected_run:
        raise ValueError("nonce must be separate from run_id")
    return experiment, expected_run, normalized_nonce, str(phase), normalized_arm


def build_payload(
    *,
    experiment_id: str,
    run_id: str,
    nonce: str,
    phase: str,
    arm: str | None,
    pins: Mapping[str, object],
    train_records: Sequence[DecisionRecord | Mapping[str, object]],
    evaluation_presentations: Sequence[Mapping[str, object]],
    initialization: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build, hash, and fully validate an initialization or training payload."""

    experiment, run, nonce_value, normalized_phase, normalized_arm = _normalize_identity(
        experiment_id=experiment_id, run_id=run_id, nonce=nonce, phase=phase, arm=arm
    )
    normalized_pins = validate_pins(pins)
    if source_fingerprints() != normalized_pins["source_file_sha256"]:
        raise ValueError("payload source fingerprints do not match current source bytes")
    if normalized_phase == "initialize":
        if train_records:
            raise ValueError("initialization payload must not contain training records")
        if initialization is not None:
            raise ValueError("initialization payload must not contain an initialization descriptor")
        normalized_train: list[dict[str, object]] = []
        schedule_sha256 = None
        normalized_initialization = None
    else:
        if initialization is None:
            raise ValueError("training payload requires the successful initialization descriptor")
        normalized_train, parsed_train = _validate_training_rows(
            [_train_record_dict(row) for row in train_records]
        )
        normalized_initialization = _validate_initialization(initialization, experiment)
        real_records = tuple(
            record
            for record in parsed_train
            if record.dataset_id in mixture_data.REAL_TRAIN_DATASET_COUNTS
        )
        synthetic_records = tuple(
            record
            for record in parsed_train
            if record.dataset_id in mixture_data.SYNTHETIC_TRAIN_DATASET_COUNTS
        )
        audit = mixture_data.build_training_schedule_audit(real_records, synthetic_records)
        schedule_sha256 = audit["schedule_sha256"][normalized_arm]
    normalized_eval = [
        normalize_json_object(row, "evaluation presentation") for row in evaluation_presentations
    ]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment,
        "run_id": run,
        "nonce": nonce_value,
        "phase": normalized_phase,
        "arm": normalized_arm,
        "pins": normalized_pins,
        "train_records": normalized_train,
        "evaluation_presentations": normalized_eval,
        "schedule_sha256": schedule_sha256,
        "initialization": normalized_initialization,
    }
    payload["payload_sha256"] = json_sha256(payload)
    return validate_payload(payload)


def validate_payload(value: object) -> dict[str, object]:
    """Validate exact payload fields, semantic panels, hashes, and deterministic schedule."""

    payload = normalize_json_object(value, "payload")
    if set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("payload has an unexpected schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("payload schema version is unsupported")
    experiment, run, nonce, phase, arm = _normalize_identity(
        experiment_id=payload["experiment_id"],
        run_id=payload["run_id"],
        nonce=payload["nonce"],
        phase=payload["phase"],
        arm=payload["arm"],
    )
    pins = validate_pins(payload["pins"])
    payload_digest = validate_sha256(payload["payload_sha256"], "payload_sha256")
    unsigned = {key: item for key, item in payload.items() if key != "payload_sha256"}
    if json_sha256(unsigned) != payload_digest:
        raise ValueError("payload SHA-256 does not match its canonical fields")

    if phase == "initialize":
        if payload["train_records"] != [] or payload["schedule_sha256"] is not None:
            raise ValueError("initialization payload must have no training rows or schedule digest")
        if payload["initialization"] is not None:
            raise ValueError("initialization payload cannot refer to a prior adapter")
        train_records: tuple[DecisionRecord, ...] = ()
    else:
        train_rows, train_records = _validate_training_rows(payload["train_records"])
        if train_rows != payload["train_records"]:
            raise ValueError("training record rows are not normalized JSON")
        initialization = _validate_initialization(payload["initialization"], experiment)
        if initialization != payload["initialization"]:
            raise ValueError("initialization descriptor is not normalized JSON")
        real_records = tuple(
            record
            for record in train_records
            if record.dataset_id in mixture_data.REAL_TRAIN_DATASET_COUNTS
        )
        synthetic_records = tuple(
            record
            for record in train_records
            if record.dataset_id in mixture_data.SYNTHETIC_TRAIN_DATASET_COUNTS
        )
        audit = mixture_data.build_training_schedule_audit(real_records, synthetic_records)
        schedule = audit["schedule_sha256"][arm]
        if validate_sha256(payload["schedule_sha256"], "schedule_sha256") != schedule:
            raise ValueError("training schedule digest differs from regenerated approved schedule")

    presentations = _validate_evaluation_rows(payload["evaluation_presentations"], phase)
    _validate_group_separation(train_records, presentations)
    payload["pins"] = pins
    payload["evaluation_presentations"] = presentations
    return payload


def training_examples(payload: object) -> tuple[training_rehearsal_core.TrainingExample, ...]:
    """Regenerate the deterministic arm schedule after validating the full payload."""

    normalized = validate_payload(payload)
    if normalized["phase"] != "train":
        return ()
    records = tuple(_record_from_row(row) for row in normalized["train_records"])
    real_records = tuple(
        record for record in records if record.dataset_id in mixture_data.REAL_TRAIN_DATASET_COUNTS
    )
    synthetic_records = tuple(
        record
        for record in records
        if record.dataset_id in mixture_data.SYNTHETIC_TRAIN_DATASET_COUNTS
    )
    schedules = mixture_data.build_training_schedules(real_records, synthetic_records)
    return tuple(schedules[str(normalized["arm"])])


from experiments.mixture_training_results import validate_result  # noqa: E402

__all__ = [
    "DATA_FILE_SHA256",
    "EXPECTED_PROTOCOL_SHA256",
    "MAX_INPUT_TOKENS",
    "MAX_TOTAL_FORWARDS",
    "MODEL_ID",
    "MODEL_REVISION",
    "PROFILE",
    "PROTOCOL_PATH",
    "RELOAD_PARITY_COUNT",
    "RUNTIME_VERSION_PINS",
    "SCHEMA_VERSION",
    "SOURCE_FINGERPRINT_PATHS",
    "WORKSPACE",
    "build_payload",
    "load_local_data",
    "source_fingerprints",
    "training_examples",
    "validate_payload",
    "validate_result",
    "verify_protocol",
]
