"""Pinned CPU-only inputs for the targeted matched-training study."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments import adapter_transfer_contracts
from experiments.mixture_training_contracts import strict_json_loads
from reflex_decisions import fresh_eval_data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option

EXPERIMENT_ID = "targeted-reasoning-v1"
SELECTION_SEED = 20261013
HANS_TRAIN_DATASET = "targeted-hans-train-v1"
HANS_RESERVED_DATASET = "targeted-hans-reserved-v1"
WINOGRANDE_TRAIN_DATASET = "targeted-winogrande-train-v1"
WINOGRANDE_RESERVED_DATASET = "targeted-winogrande-reserved-v1"
SNLI_TRAIN_DATASET = "snli-training-v1"
_HANS_QUESTION_PREFIX = "Does the following statement necessarily follow from the context? "
_SNLI_CONTEXT_PREFIX = "Premise: "
_SNLI_CONTEXT_DELIMITER = "\nHypothesis: "
PRESENTATION_FIELDS = frozenset(
    {
        "presentation_id",
        "record_id",
        "dataset_id",
        "source_group_id",
        "request_hash",
        "order_index",
        "order_ids",
        "request",
    }
)
_TRAINING_POOL_NAMES = ("real", "synthetic", "snli", "hans", "winogrande")
_STREAM_SEEDS = {"real": 20261114, "synthetic": 20261214, "snli": 20261314}
_HANS_STREAM_SEED = 20261414
_WINOGRANDE_STREAM_SEED = 20261514
WINOGRANDE_MEMBERS = (
    "winogrande_1.1/train_xl.jsonl",
    "winogrande_1.1/train_xl-labels.lst",
    "winogrande_1.1/dev.jsonl",
    "winogrande_1.1/dev-labels.lst",
)
PINNED_FILE_SHA256 = {
    "data/raw/targeted-reasoning-v1/heuristics_train_set.txt": (
        "49245bd5fdb0b185dcbfbf48f0f16513c62ad5bc9fad0b8800dc48d6818ee5cf"
    ),
    "data/raw/fresh-eval-v1/hans-heuristics_evaluation_set.txt": (
        "c55b62feef9913070e88f38938dc2492018c945ac81f70139346472494124e79"
    ),
    "data/raw/fresh-eval-v1/winogrande_1.1.zip": (
        "3619ab104d8be2977b25c90ff420cb42d491707dcc75362a1e5d22bc082b7318"
    ),
    "data/processed/fresh-eval-v1/records.jsonl": (
        "8f8b05fee9aa25fdbf6d72079ddad64b007443b9fa41b29f624effb5d55e6121"
    ),
    "data/processed/fresh-eval-v1/record-metadata.json": (
        "14c4eb3f7c5b0b7cb76a958a6ac255d7b9a42fa4d480a3fdc6fc71894ea17dfe"
    ),
    "artifacts/natural-reasoning-2026-10-04-r1-receipt.json": (
        "d71ed1d77b46d19022850736c88d9ec3528d4dbfb5d9131a8a01b900454aaf67"
    ),
    "artifacts/adapter-transfer-2026-10-05-r1-receipt.json": (
        "8797c697afe40104a45dfc73a449f2c79b2118899febf6809ca951669a27515d"
    ),
    "artifacts/fresh-eval-2026-10-05-r1-receipt.json": (
        "7bed5a26faaed9b5abe84609f6b5a21ebe3313d6527e3f4c45f3708305b56644"
    ),
    "data/evaluations/adapter-transfer-v1-selection.json": (
        "5b274b9cf947b497dee5f2c05547d7fe7d4cae2dcd546a8245180406885569e5"
    ),
    "data/processed/real-pilot-v1.jsonl": (
        "8d803048df38b1120d2a1e91f404a480fba6097f6f300c0df9b1ec2512cff7ba"
    ),
    "data/processed/synthetic-seed-v1-r2/records.jsonl": (
        "30b0e07b89977b1345d403936ab666b06f822e13053ffe71cf11fb0243321e5c"
    ),
    "data/processed/snli-balanced-v1/records.jsonl": (
        "a1fa41d19b381e227ebca258b1561e7e39184f6f58a325aaa5a2b1ff61ddd98c"
    ),
    "data/processed/copa-dev-pilot-v1.jsonl": (
        "a8daefa5a7300cc6243f3af1208bbd2887d41c306416bb37bf95263748657c5b"
    ),
    "data/processed/broader-dev-pilot-v1.jsonl": (
        "5d09ac1c1df2b7c1849e675179cb57306040f473c0ebad60527f54d186ac6845"
    ),
}
RETENTION_PRESENTATION_COUNTS = {
    "dbpedia14-pilot-v1-development": 1568,
    "sms-pilot-v1-development": 120,
    "snli-balanced-v1-development": 1152,
    "synthetic-atomic-fact-inference-v1-development": 450,
    "synthetic-numeric-selection-v1-development": 364,
    "boolq-dev-pilot-v1": 64,
    "copa-dev-pilot-v1": 64,
}
MONITORING_PRESENTATION_COUNTS = {
    "hans-eval-v1": 600,
    "winogrande-dev-v1": 400,
    "arc-challenge-dev-v1": 400,
}
RETENTION_DATASETS = frozenset(RETENTION_PRESENTATION_COUNTS)
MONITORING_DATASETS = frozenset(MONITORING_PRESENTATION_COUNTS)
RESERVED_DATASETS = frozenset({HANS_RESERVED_DATASET, WINOGRANDE_RESERVED_DATASET})
_RETENTION_RECORD_FILES = {
    "data/processed/real-pilot-v1.jsonl": frozenset(
        {"dbpedia14-pilot-v1-development", "sms-pilot-v1-development"}
    ),
    "data/processed/synthetic-seed-v1-r2/records.jsonl": frozenset(
        {
            "synthetic-atomic-fact-inference-v1-development",
            "synthetic-numeric-selection-v1-development",
        }
    ),
    "data/processed/snli-balanced-v1/records.jsonl": frozenset({"snli-balanced-v1-development"}),
    "data/processed/copa-dev-pilot-v1.jsonl": frozenset({"copa-dev-pilot-v1"}),
    "data/processed/broader-dev-pilot-v1.jsonl": frozenset({"boolq-dev-pilot-v1"}),
}
_OLD_POOL_DATASETS = {
    "real": frozenset({"dbpedia14-pilot-v1-train", "sms-pilot-v1-train"}),
    "synthetic": frozenset(
        {
            "synthetic-atomic-fact-inference-v1-train",
            "synthetic-numeric-selection-v1-train",
        }
    ),
    "snli": frozenset({SNLI_TRAIN_DATASET}),
}
_OLD_POOL_COUNTS = {"real": 504, "synthetic": 500, "snli": 500}
_OLD_POOL_DATASET_COUNTS = {
    "dbpedia14-pilot-v1-train": 252,
    "sms-pilot-v1-train": 252,
    "synthetic-atomic-fact-inference-v1-train": 300,
    "synthetic-numeric-selection-v1-train": 200,
    SNLI_TRAIN_DATASET: 500,
}
_TRAINING_POOL_COUNTS = {**_OLD_POOL_COUNTS, "hans": 600, "winogrande": 600}
_RETENTION_RECORD_COUNT = 497
_MONITORING_RECORD_COUNT = 700
_RESERVED_RECORD_COUNT = 500
_EVALUATION_RECORD_COUNT = 1697
_INPUT_FILE_COUNT = len(PINNED_FILE_SHA256)
_WINOGRANDE_MAX_APPROVED_UNCOMPRESSED_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TargetedInputs:
    """Immutable container for pinned CPU inputs and label-free presentations."""

    training_pools: dict[str, tuple[DecisionRecord, ...]]
    schedules: dict[str, tuple[dict[str, object], ...]]
    evaluation_records: tuple[DecisionRecord, ...]
    presentations: tuple[dict[str, object], ...]
    strata: dict[str, str]
    selection: dict[str, object]
    file_sha256: dict[str, str]
    audit: dict[str, object]


def _read_pinned_file(root: str | Path, relative: str, expected_sha256: str) -> bytes:
    """Read one regular pinned file only after proving it resolves inside root."""

    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("pinned input path or SHA-256 is malformed")
    try:
        canonical_root = Path(root).resolve(strict=True)
        resolved = (canonical_root / relative).resolve(strict=True)
        resolved.relative_to(canonical_root)
        if not canonical_root.is_dir() or not resolved.is_file():
            raise ValueError("pinned path is not a regular file inside the project root")
        raw = resolved.read_bytes()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"pinned input is unavailable inside the project root: {relative}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"pinned input SHA-256 mismatch: {relative}")
    return raw


def _read_wino_members(archive_bytes: bytes) -> dict[str, bytes]:
    """Read exactly the four approved train/dev members and no official test member."""

    if not isinstance(archive_bytes, bytes):
        raise TypeError("WinoGrande archive must be provided as bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            infos = archive.infolist()
            by_name: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                if info.filename in by_name:
                    raise ValueError("WinoGrande archive contains duplicate member names")
                by_name[info.filename] = info
            if any(name not in by_name for name in WINOGRANDE_MEMBERS):
                raise ValueError("WinoGrande archive is missing an approved member")
            if sum(by_name[name].file_size for name in WINOGRANDE_MEMBERS) > (
                _WINOGRANDE_MAX_APPROVED_UNCOMPRESSED_BYTES
            ):
                raise ValueError("WinoGrande approved members exceed the size limit")
            members: dict[str, bytes] = {}
            for name in WINOGRANDE_MEMBERS:
                info = by_name[name]
                if info.is_dir():
                    raise ValueError("WinoGrande approved member has an invalid size")
                with archive.open(info) as member:
                    members[name] = member.read()
    except (OSError, zipfile.BadZipFile, RuntimeError, KeyError) as exc:
        raise ValueError("could not read approved WinoGrande archive members") from exc
    return members


def _validate_presentation_row(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != PRESENTATION_FIELDS:
        raise ValueError("presentation row has an unexpected or label-bearing schema")
    row = dict(value)
    if any(
        not isinstance(row[name], str) or not row[name].strip()
        for name in (
            "presentation_id",
            "record_id",
            "dataset_id",
            "source_group_id",
            "request_hash",
        )
    ):
        raise ValueError("presentation identity fields must be nonblank strings")
    if type(row["order_index"]) is not int or row["order_index"] < 0:
        raise ValueError("presentation order index is malformed")
    if not isinstance(row["order_ids"], list) or any(
        not isinstance(option_id, str) or not option_id for option_id in row["order_ids"]
    ):
        raise ValueError("presentation option order is malformed")
    try:
        request = DecisionRequest.model_validate(row["request"])
    except Exception as exc:
        raise ValueError("presentation request is malformed") from exc
    if row["request_hash"] != request.request_hash or row["order_ids"] != [
        option.id for option in request.options
    ]:
        raise ValueError("presentation request identity is inconsistent")
    return row


def _filter_presentation_rows(
    value: object, *, allowed_dataset_ids: set[str] | frozenset[str]
) -> tuple[dict[str, object], ...]:
    """Filter by dataset before validating rows, preserving approved source order."""

    if (
        not isinstance(value, (list, tuple))
        or not allowed_dataset_ids
        or any(
            not isinstance(dataset_id, str) or not dataset_id for dataset_id in allowed_dataset_ids
        )
    ):
        raise ValueError("presentation source or dataset allowlist is malformed")
    selected = tuple(
        _validate_presentation_row(row)
        for row in value
        if isinstance(row, Mapping)
        and isinstance(row.get("dataset_id"), str)
        and row.get("dataset_id") in allowed_dataset_ids
    )
    if len({row["presentation_id"] for row in selected}) != len(selected):
        raise ValueError("approved presentations contain duplicate IDs")
    return selected


@dataclass(frozen=True)
class _Candidate:
    """A parsed public-source candidate before exact-overlap selection."""

    split: str
    raw_source_id: str
    source_group_id: str
    request: DecisionRequest
    answer_id: str
    source_metadata: dict[str, object]

    @property
    def qualified_source_id(self) -> str:
        return f"{self.split}/{self.raw_source_id}"


def normalize_text(value: str) -> str:
    """Normalize public-source text for conservative exact-overlap checks."""

    if not isinstance(value, str):
        raise TypeError("text to normalize must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _normalized_request_key(request: DecisionRequest) -> tuple[str, str]:
    return normalize_text(request.context), normalize_text(request.question)


def _hans_pair_key(request: DecisionRequest) -> tuple[str, str]:
    question = request.question
    if question.startswith(_HANS_QUESTION_PREFIX):
        hypothesis = question[len(_HANS_QUESTION_PREFIX) :]
        if not hypothesis.strip():
            raise ValueError("HANS request has an empty statement")
        return normalize_text(request.context), normalize_text(hypothesis)
    return _normalized_request_key(request)


def _snli_premise_hypothesis(request: DecisionRequest) -> tuple[str, str]:
    context = request.context
    if not context.startswith(_SNLI_CONTEXT_PREFIX):
        raise ValueError("pinned SNLI training context has an unexpected prefix")
    premise_and_hypothesis = context[len(_SNLI_CONTEXT_PREFIX) :]
    if premise_and_hypothesis.count(_SNLI_CONTEXT_DELIMITER) != 1:
        raise ValueError("pinned SNLI training context has an unexpected delimiter")
    premise, hypothesis = premise_and_hypothesis.split(_SNLI_CONTEXT_DELIMITER, 1)
    if not premise.strip() or not hypothesis.strip():
        raise ValueError("pinned SNLI training pair is incomplete")
    return normalize_text(premise), normalize_text(hypothesis)


def _old_hans_request_key(record: DecisionRecord) -> tuple[str, str]:
    if record.dataset_id == SNLI_TRAIN_DATASET or record.dataset_id.startswith("snli-balanced-v1-"):
        return _snli_premise_hypothesis(record.request)
    return _hans_pair_key(record.request)


def _old_hans_premise(record: DecisionRecord) -> str:
    if record.dataset_id == SNLI_TRAIN_DATASET or record.dataset_id.startswith("snli-balanced-v1-"):
        premise, _hypothesis = _snli_premise_hypothesis(record.request)
        return premise
    return normalize_text(record.request.context)


def _selection_rank(candidate: _Candidate) -> tuple[str, str]:
    qualified_id = candidate.qualified_source_id
    rank = hashlib.sha256(f"{EXPERIMENT_ID}:{SELECTION_SEED}:{qualified_id}".encode()).hexdigest()
    return rank, qualified_id


def _select_hans_source(
    candidates: Sequence[_Candidate],
    *,
    quota: int,
    excluded_premises: set[str],
    excluded_requests: set[tuple[str, str]],
    excluded_request_hashes: set[str],
    conflict_premises: set[str],
    already_selected: set[str],
) -> tuple[_Candidate, ...]:
    if type(quota) is not int or quota < 0:
        raise ValueError("HANS quota must be a nonnegative integer")
    if quota == 0:
        return ()
    by_subcase: dict[str, list[_Candidate]] = defaultdict(list)
    group_by_name: dict[str, str] = {}
    name_by_group: dict[str, str] = {}
    for candidate in candidates:
        raw_subcase = candidate.source_metadata.get("subcase")
        if not isinstance(raw_subcase, str) or not raw_subcase.strip():
            raise ValueError("HANS candidate is missing its raw subcase name")
        previous_group = group_by_name.setdefault(raw_subcase, candidate.source_group_id)
        previous_name = name_by_group.setdefault(candidate.source_group_id, raw_subcase)
        if previous_group != candidate.source_group_id or previous_name != raw_subcase:
            raise ValueError("HANS raw subcase names and group IDs are not one-to-one")
        by_subcase[raw_subcase].append(candidate)
    if not by_subcase:
        raise ValueError("HANS quota cannot be filled from an empty source")

    selected: list[_Candidate] = []
    used_premises = set(already_selected)
    for raw_subcase in sorted(by_subcase):
        eligible = sorted(by_subcase[raw_subcase], key=_selection_rank)
        picked = 0
        for candidate in eligible:
            premise = normalize_text(candidate.request.context)
            request_key = _hans_pair_key(candidate.request)
            if (
                premise in excluded_premises
                or premise in conflict_premises
                or premise in used_premises
                or request_key in excluded_requests
                or candidate.request.request_hash in excluded_request_hashes
            ):
                continue
            selected.append(candidate)
            used_premises.add(premise)
            picked += 1
            if picked == quota:
                break
        if picked != quota:
            raise ValueError("HANS per-subcase quota cannot be filled")
    return tuple(selected)


def _hans_records(candidates: Sequence[_Candidate]) -> tuple[DecisionRecord, ...]:
    records: list[DecisionRecord] = []
    for candidate in candidates:
        if candidate.split not in {"hans/train", "hans/evaluation"}:
            raise ValueError("HANS candidate has an unsupported split")
        suffix = "train" if candidate.split == "hans/train" else "evaluation"
        dataset_id = HANS_TRAIN_DATASET if suffix == "train" else HANS_RESERVED_DATASET
        records.append(
            DecisionRecord(
                record_id=f"{EXPERIMENT_ID}:hans/{suffix}/{candidate.raw_source_id}",
                dataset_id=dataset_id,
                source_group_id=candidate.source_group_id,
                request=candidate.request,
                answer_id=candidate.answer_id,
            )
        )
    return tuple(records)


def select_hans_candidates(
    train_candidates: Sequence[_Candidate],
    evaluation_candidates: Sequence[_Candidate],
    *,
    old_training: Sequence[DecisionRecord],
    old_monitoring: Sequence[DecisionRecord],
    train_quota: int,
    reserve_quota: int,
) -> tuple[tuple[DecisionRecord, ...], tuple[DecisionRecord, ...], dict[str, object]]:
    """Select HANS rows using premise-wide and exact-request exclusions."""

    train = tuple(train_candidates)
    evaluation = tuple(evaluation_candidates)
    if any(row.split != "hans/train" for row in train) or any(
        row.split != "hans/evaluation" for row in evaluation
    ):
        raise ValueError("HANS candidates are assigned to an unexpected split")
    for rows in (train, evaluation):
        identities = [row.raw_source_id for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError("HANS source IDs must be unique within each split")
    combined = (*train, *evaluation)
    answers_by_request: dict[tuple[str, str], set[str]] = defaultdict(set)
    premises_by_request: dict[tuple[str, str], set[str]] = defaultdict(set)
    for candidate in combined:
        key = _hans_pair_key(candidate.request)
        answers_by_request[key].add(candidate.answer_id)
        premises_by_request[key].add(normalize_text(candidate.request.context))
    conflicting_keys = {key for key, answers in answers_by_request.items() if len(answers) > 1}
    conflict_premises = {
        premise for key in conflicting_keys for premise in premises_by_request[key]
    }

    old_train_premises = {_old_hans_premise(row) for row in old_training}
    old_train_requests = {_old_hans_request_key(row) for row in old_training}
    old_train_request_hashes = {row.request.request_hash for row in old_training}
    old_monitor_premises = {
        normalize_text(row.request.context)
        for row in old_monitoring
        if row.dataset_id == "hans-eval-v1"
    }
    eval_premises = {normalize_text(row.request.context) for row in evaluation}
    selected_train = _select_hans_source(
        train,
        quota=train_quota,
        excluded_premises=eval_premises | old_train_premises,
        excluded_requests=old_train_requests,
        excluded_request_hashes=old_train_request_hashes,
        conflict_premises=conflict_premises,
        already_selected=set(),
    )
    selected_train_premises = {normalize_text(row.request.context) for row in selected_train}
    selected_reserved = _select_hans_source(
        evaluation,
        quota=reserve_quota,
        excluded_premises=old_train_premises | old_monitor_premises,
        excluded_requests=old_train_requests,
        excluded_request_hashes=old_train_request_hashes,
        conflict_premises=conflict_premises,
        already_selected=selected_train_premises,
    )
    raw_id_collisions = {row.raw_source_id for row in train} & {
        row.raw_source_id for row in evaluation
    }
    audit: dict[str, object] = {
        "train_source_rows": len(train),
        "evaluation_source_rows": len(evaluation),
        "cross_split_raw_id_collisions": len(raw_id_collisions),
        "conflicting_request_groups": len(conflicting_keys),
        "conflicting_premises": len(conflict_premises),
        "excluded_training_evaluation_premises": len(
            {normalize_text(row.request.context) for row in train} & eval_premises
        ),
        "selected_training_records": len(selected_train),
        "selected_reserved_records": len(selected_reserved),
    }
    return _hans_records(selected_train), _hans_records(selected_reserved), audit


def _wino_option_pair(request: DecisionRequest) -> tuple[str, str]:
    labels = tuple(sorted(normalize_text(option.label) for option in request.options))
    if len(labels) != 2:
        raise ValueError("WinoGrande candidate must have exactly two options")
    return labels[0], labels[1]


def _semantic_request_key(request: DecisionRequest) -> tuple[str, str, tuple[str, ...]]:
    return (
        normalize_text(request.context),
        normalize_text(request.question),
        tuple(sorted(normalize_text(option.label) for option in request.options)),
    )


def _wino_request_key(request: DecisionRequest) -> tuple[str, str, tuple[str, ...]]:
    _wino_option_pair(request)
    return _semantic_request_key(request)


def _wino_answer_text(request: DecisionRequest, answer_id: str) -> str:
    for option in request.options:
        if option.id == answer_id:
            return normalize_text(option.label)
    raise ValueError("WinoGrande answer does not identify a source option")


def _wino_group_id(pair: tuple[str, str]) -> str:
    payload = json.dumps(pair, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"winogrande-option-pair:{hashlib.sha256(payload).hexdigest()}"


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("source JSON contains duplicate object keys")
        output[key] = value
    return output


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"source JSON contains unsupported numeric constant: {value}")


def parse_winogrande_rows(data: bytes, labels: bytes, *, split: str) -> tuple[_Candidate, ...]:
    """Parse one approved WinoGrande split with labels kept outside requests."""

    if split not in {"winogrande/train_xl", "winogrande/dev"}:
        raise ValueError("WinoGrande split is outside the exact source allowlist")
    try:
        data_text = data.decode("utf-8-sig", errors="strict")
        labels_text = labels.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("WinoGrande source is not UTF-8") from exc
    data_lines = data_text.splitlines()
    label_lines = labels_text.splitlines()
    if not data_lines or len(data_lines) != len(label_lines):
        raise ValueError("WinoGrande labels are not fully aligned")
    output: list[_Candidate] = []
    seen_ids: set[str] = set()
    for data_line, label_line in zip(data_lines, label_lines, strict=True):
        if not data_line.strip():
            raise ValueError("WinoGrande source contains a blank row")
        try:
            row = json.loads(
                data_line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("WinoGrande source contains malformed JSON") from exc
        required_fields = {"qID", "sentence", "option1", "option2"}
        if (
            not isinstance(row, dict)
            or not required_fields <= set(row)
            or set(row) - (required_fields | {"answer"})
        ):
            raise ValueError("WinoGrande source row has an unexpected schema")
        source_id = row["qID"]
        sentence = row["sentence"]
        first, second = row["option1"], row["option2"]
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (source_id, sentence, first, second)
        ):
            raise ValueError("WinoGrande source row has blank or malformed fields")
        source_id, sentence, first, second = (
            source_id.strip(),
            sentence.strip(),
            first.strip(),
            second.strip(),
        )
        answer = label_line.strip()
        if answer not in {"1", "2"}:
            raise ValueError("WinoGrande source contains an unsupported answer label")
        if "answer" in row:
            embedded_answer = row["answer"]
            if not isinstance(embedded_answer, str) or embedded_answer not in {"1", "2"}:
                raise ValueError("WinoGrande embedded answer is malformed")
            if embedded_answer != answer:
                raise ValueError("WinoGrande embedded answer conflicts with aligned labels")
        if sentence.count("_") != 1 or normalize_text(first) == normalize_text(second):
            raise ValueError("WinoGrande sentence or option pair is malformed")
        if source_id in seen_ids:
            raise ValueError("WinoGrande source contains a duplicate question ID")
        seen_ids.add(source_id)
        output.append(
            _Candidate(
                split=split,
                raw_source_id=source_id,
                source_group_id=source_id,
                request=DecisionRequest(
                    context=sentence,
                    question="Which option fills the blank?",
                    options=(Option(id="option1", label=first), Option(id="option2", label=second)),
                ),
                answer_id=f"option{answer}",
                source_metadata={"split": split.rsplit("/", 1)[-1]},
            )
        )
    return tuple(output)


def _winogrande_records(candidates: Sequence[_Candidate]) -> tuple[DecisionRecord, ...]:
    records: list[DecisionRecord] = []
    for candidate in candidates:
        if candidate.split not in {"winogrande/train_xl", "winogrande/dev"}:
            raise ValueError("WinoGrande candidate has an unsupported split")
        train_split = candidate.split == "winogrande/train_xl"
        dataset_id = WINOGRANDE_TRAIN_DATASET if train_split else WINOGRANDE_RESERVED_DATASET
        records.append(
            DecisionRecord(
                record_id=f"{EXPERIMENT_ID}:{candidate.qualified_source_id}",
                dataset_id=dataset_id,
                source_group_id=_wino_group_id(_wino_option_pair(candidate.request)),
                request=candidate.request,
                answer_id=candidate.answer_id,
            )
        )
    return tuple(records)


def _epoch_stream(
    records: Sequence[DecisionRecord], *, seed: int, count: int
) -> tuple[object, ...]:
    from experiments.training_rehearsal_core import epoch_examples

    if not records or type(count) is not int or count <= 0:
        raise ValueError("training stream requires records and a positive count")
    ordered = tuple(sorted(records, key=lambda row: row.record_id))
    examples: list[object] = []
    epoch = 0
    while len(examples) < count:
        examples.extend(epoch_examples(ordered, seed=seed, epoch=epoch))
        epoch += 1
    return tuple(examples[:count])


def _training_row(
    example: object,
    record_by_id: Mapping[str, DecisionRecord],
    *,
    role: str,
    update: int,
    microbatch_index: int,
) -> dict[str, object]:
    record_id = getattr(example, "record_id", None)
    request = getattr(example, "request", None)
    gold_option_id = getattr(example, "gold_option_id", None)
    gold_index = getattr(example, "gold_index", None)
    record = record_by_id.get(record_id) if isinstance(record_id, str) else None
    if (
        record is None
        or not isinstance(request, DecisionRequest)
        or gold_option_id != record.answer_id
        or type(gold_index) is not int
        or not 0 <= gold_index < len(request.options)
        or request.options[gold_index].id != gold_option_id
    ):
        raise ValueError("training schedule example has a malformed gold mapping")
    return {
        "presentation_id": (f"{EXPERIMENT_ID}:{role}:update-{update}:micro-{microbatch_index}"),
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "source_group_id": record.source_group_id,
        "request_hash": request.request_hash,
        "order_index": 0,
        "order_ids": [option.id for option in request.options],
        "request": request.model_dump(mode="json"),
        "gold_option_id": gold_option_id,
        "gold_index": gold_index,
        "update": update,
        "microbatch_index": microbatch_index,
    }


def build_schedules(
    training_pools: Mapping[str, Sequence[DecisionRecord]], *, rows_per_stream: int = 1200
) -> dict[str, tuple[dict[str, object], ...]]:
    """Build the paired 4-microbatch schedules from five frozen record pools."""

    if not isinstance(training_pools, Mapping) or set(training_pools) != set(_TRAINING_POOL_NAMES):
        raise ValueError("training pools do not match the exact five-source allowlist")
    if type(rows_per_stream) is not int or rows_per_stream <= 0 or rows_per_stream % 2:
        raise ValueError("schedule stream length must be a positive even integer")
    pools: dict[str, tuple[DecisionRecord, ...]] = {}
    for name in _TRAINING_POOL_NAMES:
        records = tuple(training_pools[name])
        if not records or any(not isinstance(row, DecisionRecord) for row in records):
            raise ValueError("training pool is empty or contains a malformed record")
        if len({row.record_id for row in records}) != len(records):
            raise ValueError("training pool contains duplicate record IDs")
        pools[name] = tuple(sorted(records, key=lambda row: row.record_id))
    all_records = tuple(row for name in _TRAINING_POOL_NAMES for row in pools[name])
    record_by_id = {row.record_id: row for row in all_records}
    if len(record_by_id) != len(all_records):
        raise ValueError("training record IDs collide across source pools")

    shared = {
        name: _epoch_stream(pools[name], seed=_STREAM_SEEDS[name], count=rows_per_stream)
        for name in _STREAM_SEEDS
    }
    control_extra = tuple(
        shared[name][index]
        for index in range(rows_per_stream)
        for name in ("real", "synthetic", "snli")
    )[:rows_per_stream]
    half = rows_per_stream // 2
    treatment_hans = _epoch_stream(pools["hans"], seed=_HANS_STREAM_SEED, count=half)
    treatment_wino = _epoch_stream(pools["winogrande"], seed=_WINOGRANDE_STREAM_SEED, count=half)
    treatment_extra = tuple(
        example
        for hans_example, wino_example in zip(treatment_hans, treatment_wino, strict=True)
        for example in (hans_example, wino_example)
    )
    if len(control_extra) != rows_per_stream or len(treatment_extra) != rows_per_stream:
        raise ValueError("arm-specific extra stream has an unexpected length")

    schedules: dict[str, tuple[dict[str, object], ...]] = {}
    for role, extra in (("control", control_extra), ("treatment", treatment_extra)):
        rows = tuple(
            _training_row(
                example,
                record_by_id,
                role=role,
                update=index + 1,
                microbatch_index=microbatch,
            )
            for index in range(rows_per_stream)
            for microbatch, example in enumerate(
                (
                    shared["real"][index],
                    shared["synthetic"][index],
                    shared["snli"][index],
                    extra[index],
                )
            )
        )
        schedules[role] = rows
    return schedules


def _presentation_row(
    record: DecisionRecord, options: Sequence[object], order_index: int, *, identifier: str
) -> dict[str, object]:
    ordered_options = tuple(options)
    request = record.request.model_copy(update={"options": ordered_options})
    row = {
        "presentation_id": identifier,
        "record_id": record.record_id,
        "dataset_id": record.dataset_id,
        "source_group_id": record.source_group_id,
        "request_hash": request.request_hash,
        "order_index": order_index,
        "order_ids": [option.id for option in request.options],
        "request": request.model_dump(mode="json"),
    }
    if set(row) != PRESENTATION_FIELDS:
        raise ValueError("presentation row does not match its label-free schema")
    return row


def build_reserved_presentations(
    records: Sequence[DecisionRecord],
) -> tuple[dict[str, object], ...]:
    """Build the two fixed orders for the new host-only reserved records."""

    typed = tuple(records)
    if any(
        not isinstance(row, DecisionRecord)
        or row.dataset_id not in {HANS_RESERVED_DATASET, WINOGRANDE_RESERVED_DATASET}
        or len(row.request.options) != 2
        for row in typed
    ):
        raise ValueError("reserved records contain an unsupported dataset or option count")
    if len({row.record_id for row in typed}) != len(typed):
        raise ValueError("reserved records contain duplicate identities")
    output: list[dict[str, object]] = []
    for record in sorted(typed, key=lambda row: (row.dataset_id, row.record_id)):
        for order_index, options in enumerate(
            (record.request.options, record.request.options[1:] + record.request.options[:1])
        ):
            output.append(
                _presentation_row(
                    record,
                    options,
                    order_index,
                    identifier=f"{EXPERIMENT_ID}:{record.record_id}:order-{order_index}",
                )
            )
    return tuple(output)


def select_winogrande_candidates(
    train_candidates: Sequence[_Candidate],
    development_candidates: Sequence[_Candidate],
    *,
    old_training: Sequence[DecisionRecord],
    old_monitoring: Sequence[DecisionRecord],
    old_monitor_source_ids: set[str] | None = None,
    train_quota: int,
    reserve_quota: int,
) -> tuple[tuple[DecisionRecord, ...], tuple[DecisionRecord, ...], dict[str, object]]:
    """Select WinoGrande rows by conservative option-pair and exact-request rules."""

    train = tuple(train_candidates)
    development = tuple(development_candidates)
    if any(row.split != "winogrande/train_xl" for row in train) or any(
        row.split != "winogrande/dev" for row in development
    ):
        raise ValueError("WinoGrande candidates are assigned to an unexpected split")
    for rows in (train, development):
        identifiers = [row.raw_source_id for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("WinoGrande source IDs must be unique within each split")
    all_candidates = (*train, *development)
    answers_by_request: dict[tuple[str, str, tuple[str, ...]], set[str]] = defaultdict(set)
    blocks_by_request: dict[tuple[str, str, tuple[str, ...]], set[tuple[str, str]]] = defaultdict(
        set
    )
    for candidate in all_candidates:
        key = _wino_request_key(candidate.request)
        answers_by_request[key].add(_wino_answer_text(candidate.request, candidate.answer_id))
        blocks_by_request[key].add(_wino_option_pair(candidate.request))
    conflicting_requests = {key for key, answers in answers_by_request.items() if len(answers) > 1}
    conflicting_blocks = {block for key in conflicting_requests for block in blocks_by_request[key]}

    development_blocks = {_wino_option_pair(row.request) for row in development}
    development_ids = {row.raw_source_id for row in development}
    training_ids = {row.raw_source_id for row in train}
    development_sentences = {normalize_text(row.request.context) for row in development}
    old_train_requests = {_semantic_request_key(row.request) for row in old_training}
    monitor_records = tuple(row for row in old_monitoring if row.dataset_id == "winogrande-dev-v1")
    old_monitor_blocks = {_wino_option_pair(row.request) for row in monitor_records}
    old_monitor_sentences = {normalize_text(row.request.context) for row in monitor_records}
    old_monitor_ids = old_monitor_source_ids or set()

    def choose(
        candidates: Sequence[_Candidate],
        *,
        quota: int,
        forbidden_blocks: set[tuple[str, str]],
        forbidden_ids: set[str],
        forbidden_sentences: set[str],
        exclude_development_sentence_and_id: bool,
    ) -> tuple[_Candidate, ...]:
        if type(quota) is not int or quota < 0:
            raise ValueError("WinoGrande quota must be a nonnegative integer")
        if quota == 0:
            return ()
        selected: list[_Candidate] = []
        used_blocks: set[tuple[str, str]] = set()
        for candidate in sorted(candidates, key=_selection_rank):
            block = _wino_option_pair(candidate.request)
            sentence = normalize_text(candidate.request.context)
            request_key = _semantic_request_key(candidate.request)
            if (
                block in forbidden_blocks
                or block in conflicting_blocks
                or block in used_blocks
                or candidate.raw_source_id in forbidden_ids
                or sentence in forbidden_sentences
                or request_key in old_train_requests
            ):
                continue
            if exclude_development_sentence_and_id and (
                candidate.raw_source_id in development_ids or sentence in development_sentences
            ):
                continue
            selected.append(candidate)
            used_blocks.add(block)
            if len(selected) == quota:
                return tuple(selected)
        raise ValueError("WinoGrande quota cannot be filled")

    selected_train = choose(
        train,
        quota=train_quota,
        forbidden_blocks=development_blocks,
        forbidden_ids=set(),
        forbidden_sentences=set(),
        exclude_development_sentence_and_id=True,
    )
    selected_reserved = choose(
        development,
        quota=reserve_quota,
        forbidden_blocks=old_monitor_blocks,
        forbidden_ids=old_monitor_ids | training_ids,
        forbidden_sentences=old_monitor_sentences,
        exclude_development_sentence_and_id=False,
    )
    train_blocks = {_wino_option_pair(row.request) for row in selected_train}
    reserve_blocks = {_wino_option_pair(row.request) for row in selected_reserved}
    if train_blocks & reserve_blocks:
        raise ValueError("WinoGrande training and reserved blocks overlap")
    audit: dict[str, object] = {
        "train_source_rows": len(train),
        "development_source_rows": len(development),
        "train_development_raw_id_collisions": len(
            {row.raw_source_id for row in train} & {row.raw_source_id for row in development}
        ),
        "conflicting_request_groups": len(conflicting_requests),
        "conflicting_option_blocks": len(conflicting_blocks),
        "excluded_train_dev_blocks": len(
            {_wino_option_pair(row.request) for row in train} & development_blocks
        ),
        "excluded_train_dev_sentences": len(
            {normalize_text(row.request.context) for row in train} & development_sentences
        ),
        "selected_training_records": len(selected_train),
        "selected_reserved_records": len(selected_reserved),
    }
    return _winogrande_records(selected_train), _winogrande_records(selected_reserved), audit


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = strict_json_loads(raw)
    except Exception as exc:
        raise ValueError(f"pinned {label} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"pinned {label} must be a JSON object")
    return value


def _receipt_payload(raw: bytes, *, kind: str) -> dict[str, object]:
    receipt = _json_object(raw, f"{kind} receipt")
    if kind == "natural-reasoning":
        expected_fields = {
            "schema_version",
            "experiment_id",
            "status",
            "payloads",
            "results",
            "failure",
            "modal",
        }
        if (
            set(receipt) != expected_fields
            or type(receipt["schema_version"]) is not int
            or receipt["schema_version"] != 2
            or not isinstance(receipt["experiment_id"], str)
            or receipt["status"] != "passed"
            or receipt["failure"] is not None
        ):
            raise ValueError("pinned natural-reasoning receipt is incomplete")
        payloads = receipt["payloads"]
        if not isinstance(payloads, dict) or set(payloads) != {
            "initialize",
            "synthetic_repeat",
            "snli_mix",
        }:
            raise ValueError("pinned natural-reasoning receipt has an unexpected payload map")
        payload = payloads["snli_mix"]
        if (
            not isinstance(payload, dict)
            or payload.get("phase") != "train"
            or payload.get("arm") != "snli_mix"
        ):
            raise ValueError("pinned natural-reasoning SNLI-mix payload is malformed")
        return payload

    if kind == "adapter-transfer":
        expected_fields = {
            "schema_version",
            "run_id",
            "status",
            "phase",
            "payload",
            "result",
            "failure",
            "modal",
        }
    elif kind == "fresh-eval":
        expected_fields = {
            "schema_version",
            "experiment_id",
            "run_id",
            "status",
            "phase",
            "payload",
            "result",
            "raw_worker_result",
            "failure",
            "execution",
            "modal",
        }
    else:
        raise ValueError("receipt kind is outside the exact allowlist")
    if (
        set(receipt) != expected_fields
        or type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or not isinstance(receipt.get("run_id"), str)
        or not receipt["run_id"].strip()
        or receipt["status"] != "passed"
        or receipt["phase"] not in {"complete", "completed"}
        or receipt["failure"] is not None
        or not isinstance(receipt["payload"], dict)
    ):
        raise ValueError(f"pinned {kind} receipt is incomplete")
    payload = receipt["payload"]
    if kind == "adapter-transfer" and payload.get("experiment_id") != "adapter-transfer-v1":
        raise ValueError("pinned adapter-transfer payload has an unexpected experiment ID")
    if kind == "fresh-eval" and (
        receipt.get("experiment_id") != "fresh-eval-v1"
        or payload.get("experiment_id") != "fresh-eval-v1"
    ):
        raise ValueError("pinned fresh-eval payload has an unexpected experiment ID")
    return payload


def _record_matches_canonical_shape(record: DecisionRecord, row: Mapping[str, object]) -> bool:
    """Allow omitted option descriptions while rejecting other representation changes."""

    canonical = record.model_dump(mode="json")
    raw_request = row.get("request")
    raw_options = raw_request.get("options") if isinstance(raw_request, Mapping) else None
    canonical_options = canonical["request"]["options"]
    if not isinstance(raw_options, (list, tuple)) or len(raw_options) != len(canonical_options):
        return False
    for raw_option, canonical_option in zip(raw_options, canonical_options, strict=True):
        if not isinstance(raw_option, Mapping):
            return False
        if "description" not in raw_option and canonical_option["description"] is None:
            del canonical_option["description"]
    return canonical == row


def _records_from_rows(value: object, *, label: str) -> tuple[DecisionRecord, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a record list")
    records: list[DecisionRecord] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a malformed record")
        try:
            record = DecisionRecord.model_validate(row)
        except Exception as exc:
            raise ValueError(f"{label} contains a malformed record") from exc
        if not _record_matches_canonical_shape(record, row):
            raise ValueError(f"{label} contains a noncanonical record")
        records.append(record)
    if len({record.record_id for record in records}) != len(records):
        raise ValueError(f"{label} contains duplicate record IDs")
    return tuple(records)


def _parse_jsonl_records(
    raw: bytes,
    relative: str,
    *,
    allowed_dataset_ids: set[str] | frozenset[str],
    membership: Mapping[str, str],
) -> tuple[DecisionRecord, ...]:
    """Parse only approved dataset rows whose IDs occur in the frozen panel."""

    if not isinstance(allowed_dataset_ids, (set, frozenset)) or not membership:
        raise ValueError("approved JSONL dataset or record allowlist is malformed")
    expected_ids = set(membership)
    selected: dict[str, DecisionRecord] = {}
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"pinned record file contains a blank line: {relative}:{line_number}")
        try:
            row = strict_json_loads(line)
        except Exception as exc:
            raise ValueError(
                f"pinned record file has malformed JSON: {relative}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"pinned record file row is not an object: {relative}:{line_number}")
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str) or dataset_id not in allowed_dataset_ids:
            continue
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or record_id not in expected_ids:
            continue
        if membership[record_id] != dataset_id:
            raise ValueError(f"approved record is in the wrong source dataset: {relative}")
        try:
            record = DecisionRecord.model_validate(row)
        except Exception as exc:
            raise ValueError(f"approved record is malformed: {relative}") from exc
        if not _record_matches_canonical_shape(record, row) or record_id in selected:
            raise ValueError(f"approved record is noncanonical or duplicated: {relative}")
        selected[record_id] = record
    if set(selected) != expected_ids:
        raise ValueError(f"approved record membership differs from the panel: {relative}")
    return tuple(selected[record_id] for record_id in sorted(selected))


def _require_panel_counts(
    rows: Sequence[Mapping[str, object]], expected: Mapping[str, int], label: str
) -> None:
    counts = Counter(row.get("dataset_id") for row in rows)
    if counts != Counter(expected):
        raise ValueError(f"{label} dataset IDs or presentation counts differ from the allowlist")


def _hans_candidates(raw: bytes, *, split: str) -> tuple[_Candidate, ...]:
    if split not in {"hans/train", "hans/evaluation"}:
        raise ValueError("HANS split is outside the exact allowlist")
    parsed = fresh_eval_data.parse_hans_tsv(raw)
    return tuple(
        _Candidate(
            split=split,
            raw_source_id=row.raw_source_id,
            source_group_id=row.source_group_id,
            request=row.request,
            answer_id=row.answer_id,
            source_metadata=row.source_metadata,
        )
        for row in parsed
    )


def _request_semantic_key(
    request: DecisionRequest,
) -> tuple[str, str, tuple[tuple[str, str, str], ...]]:
    return (
        normalize_text(request.context),
        normalize_text(request.question),
        tuple(
            sorted(
                (
                    normalize_text(option.id),
                    normalize_text(option.label),
                    normalize_text(option.description or ""),
                )
                for option in request.options
            )
        ),
    )


def _join_evaluation_records(
    presentations: Sequence[Mapping[str, object]],
    source_records: Sequence[DecisionRecord],
) -> tuple[DecisionRecord, ...]:
    by_id = {record.record_id: record for record in source_records}
    if len(by_id) != len(source_records):
        raise ValueError("evaluation sources contain duplicate record IDs")
    used: set[str] = set()
    groups_by_id: dict[str, str] = {}
    result: list[DecisionRecord] = []
    for raw in presentations:
        row = _validate_presentation_row(raw)
        record_id = str(row["record_id"])
        record = by_id.get(record_id)
        if record is None or row["dataset_id"] != record.dataset_id:
            raise ValueError("presentation references a record outside the approved gold inputs")
        group_id = str(row["source_group_id"])
        if record.dataset_id in {"boolq-dev-pilot-v1", "copa-dev-pilot-v1"}:
            if not group_id.startswith(f"adapter-transfer-v1:{record.dataset_id}:"):
                raise ValueError("transfer presentation is missing its analysis group alias")
        elif group_id != record.source_group_id:
            raise ValueError("presentation source group differs from its host record")
        previous_group = groups_by_id.setdefault(record_id, group_id)
        if previous_group != group_id:
            raise ValueError("one record has inconsistent presentation group IDs")
        try:
            presentation_request = DecisionRequest.model_validate(row["request"])
        except Exception as exc:
            raise ValueError("presentation request is malformed") from exc
        if _request_semantic_key(presentation_request) != _request_semantic_key(record.request):
            raise ValueError("presentation options or request differ from the host record")
        if record.answer_id not in {option.id for option in record.request.options}:
            raise ValueError("host evaluation record has an invalid gold option")
        if record_id not in used:
            used.add(record_id)
            result.append(
                record.model_copy(update={"source_group_id": group_id})
                if group_id != record.source_group_id
                else record
            )
    if used != set(by_id):
        raise ValueError("evaluation gold membership differs from the final panel")
    return tuple(result)


def _validate_new_training_separation(
    hans_training: Sequence[DecisionRecord],
    winogrande_training: Sequence[DecisionRecord],
    evaluation_records: Sequence[DecisionRecord],
) -> None:
    evaluation_ids = {record.record_id for record in evaluation_records}
    if {record.record_id for record in (*hans_training, *winogrande_training)} & evaluation_ids:
        raise ValueError("new training record IDs overlap the evaluation panel")
    eval_hans_keys = {_old_hans_request_key(record) for record in evaluation_records}
    eval_wino_keys = {_semantic_request_key(record.request) for record in evaluation_records}
    if any(_hans_pair_key(record.request) in eval_hans_keys for record in hans_training):
        raise ValueError("new HANS training request overlaps retention or evaluation")
    if any(
        _semantic_request_key(record.request) in eval_wino_keys for record in winogrande_training
    ):
        raise ValueError("new WinoGrande training request overlaps retention or evaluation")


def _lineage_rows(
    candidates: Sequence[_Candidate], records: Sequence[DecisionRecord]
) -> list[dict[str, object]]:
    by_id = {f"{EXPERIMENT_ID}:{row.qualified_source_id}": row for row in candidates}
    output: list[dict[str, object]] = []
    for record in sorted(records, key=lambda row: row.record_id):
        candidate = by_id.get(record.record_id)
        if candidate is None:
            raise ValueError("selected record is missing source lineage")
        output.append(
            {
                "record_id": record.record_id,
                "dataset_id": record.dataset_id,
                "source_group_id": record.source_group_id,
                "request_hash": record.request.request_hash,
                "source_split": candidate.split,
                "raw_source_id": candidate.raw_source_id,
                "selection_rank_sha256": _selection_rank(candidate)[0],
                "subcase": candidate.source_metadata.get("subcase"),
            }
        )
    return output


def load_inputs(root: str | Path) -> TargetedInputs:
    """Build the frozen study data from the exact pinned, approved CPU inputs."""

    project_root = Path(root).resolve(strict=True)
    if not project_root.is_dir():
        raise ValueError("targeted-training project root must be a directory")
    input_bytes = {
        relative: _read_pinned_file(project_root, relative, digest)
        for relative, digest in PINNED_FILE_SHA256.items()
    }
    file_sha256 = dict(PINNED_FILE_SHA256)
    selection_path = adapter_transfer_contracts.SELECTION_PATH
    if adapter_transfer_contracts.require_selection_pin() != PINNED_FILE_SHA256[selection_path]:
        raise ValueError("adapter selection pin differs from the targeted input allowlist")
    selection = adapter_transfer_contracts.validate_selection(
        strict_json_loads(input_bytes[selection_path])
    )

    natural_payload = _receipt_payload(
        input_bytes["artifacts/natural-reasoning-2026-10-04-r1-receipt.json"],
        kind="natural-reasoning",
    )
    transfer_payload = _receipt_payload(
        input_bytes["artifacts/adapter-transfer-2026-10-05-r1-receipt.json"],
        kind="adapter-transfer",
    )
    fresh_payload = _receipt_payload(
        input_bytes["artifacts/fresh-eval-2026-10-05-r1-receipt.json"], kind="fresh-eval"
    )
    if (
        transfer_payload.get("selection") != selection
        or fresh_payload.get("selection") != selection
    ):
        raise ValueError("pinned receipts do not name the selected adapter descriptor")
    old_training = _records_from_rows(
        natural_payload.get("train_records"), label="pinned SNLI-mix training records"
    )
    if len(old_training) != 1504 or Counter(
        record.dataset_id for record in old_training
    ) != Counter(_OLD_POOL_DATASET_COUNTS):
        raise ValueError("pinned SNLI-mix training records differ from the exact old pools")
    training_pools: dict[str, tuple[DecisionRecord, ...]] = {
        name: tuple(
            sorted(
                (record for record in old_training if record.dataset_id in dataset_ids),
                key=lambda record: record.record_id,
            )
        )
        for name, dataset_ids in _OLD_POOL_DATASETS.items()
    }
    if any(len(training_pools[name]) != _OLD_POOL_COUNTS[name] for name in _OLD_POOL_COUNTS):
        raise ValueError("pinned SNLI-mix training pools have unexpected counts")

    retention_presentations = (
        *_filter_presentation_rows(
            natural_payload.get("evaluation_presentations"),
            allowed_dataset_ids=RETENTION_DATASETS,
        ),
        *_filter_presentation_rows(
            transfer_payload.get("presentations"),
            allowed_dataset_ids={"boolq-dev-pilot-v1", "copa-dev-pilot-v1"},
        ),
    )
    _require_panel_counts(retention_presentations, RETENTION_PRESENTATION_COUNTS, "retention panel")
    monitoring_presentations = _filter_presentation_rows(
        fresh_payload.get("presentations"), allowed_dataset_ids=MONITORING_DATASETS
    )
    _require_panel_counts(
        monitoring_presentations, MONITORING_PRESENTATION_COUNTS, "monitoring panel"
    )
    if len(retention_presentations) != 3782 or len(monitoring_presentations) != 1400:
        raise ValueError("historical presentation counts differ from the frozen panel")

    hans_training_candidates = _hans_candidates(
        input_bytes["data/raw/targeted-reasoning-v1/heuristics_train_set.txt"],
        split="hans/train",
    )
    hans_evaluation_candidates = _hans_candidates(
        input_bytes["data/raw/fresh-eval-v1/hans-heuristics_evaluation_set.txt"],
        split="hans/evaluation",
    )
    wino_members = _read_wino_members(input_bytes["data/raw/fresh-eval-v1/winogrande_1.1.zip"])
    wino_training_candidates = parse_winogrande_rows(
        wino_members["winogrande_1.1/train_xl.jsonl"],
        wino_members["winogrande_1.1/train_xl-labels.lst"],
        split="winogrande/train_xl",
    )
    wino_development_candidates = parse_winogrande_rows(
        wino_members["winogrande_1.1/dev.jsonl"],
        wino_members["winogrande_1.1/dev-labels.lst"],
        split="winogrande/dev",
    )

    fresh_record_ids = {
        str(row["record_id"]): str(row["dataset_id"]) for row in monitoring_presentations
    }
    fresh_records = _parse_jsonl_records(
        input_bytes["data/processed/fresh-eval-v1/records.jsonl"],
        "data/processed/fresh-eval-v1/records.jsonl",
        allowed_dataset_ids=MONITORING_DATASETS,
        membership=fresh_record_ids,
    )
    metadata = _json_object(
        input_bytes["data/processed/fresh-eval-v1/record-metadata.json"],
        "fresh evaluation record metadata",
    )
    if set(metadata) != set(fresh_record_ids):
        raise ValueError("fresh evaluation metadata membership differs from monitoring panel")
    metadata_by_id: dict[str, Mapping[str, object]] = {}
    for record in fresh_records:
        value = metadata.get(record.record_id)
        if not isinstance(value, Mapping):
            raise ValueError("fresh evaluation metadata row is malformed")
        if (
            value.get("dataset_id") != record.dataset_id
            or value.get("source_group_id") != record.source_group_id
            or value.get("semantic_request_sha256") != record.request.request_hash
            or value.get("record_sha256")
            != hashlib.sha256(_canonical_json(record.model_dump(mode="json"))).hexdigest()
            or not isinstance(value.get("raw_source_id"), str)
            or not value["raw_source_id"]
        ):
            raise ValueError("fresh evaluation metadata differs from its host record")
        metadata_by_id[record.record_id] = value
    old_wino_source_ids = {
        str(metadata_by_id[record.record_id]["raw_source_id"])
        for record in fresh_records
        if record.dataset_id == "winogrande-dev-v1"
    }

    selected_hans_training, selected_hans_reserved, hans_audit = select_hans_candidates(
        hans_training_candidates,
        hans_evaluation_candidates,
        old_training=old_training,
        old_monitoring=fresh_records,
        train_quota=20,
        reserve_quota=10,
    )
    selected_wino_training, selected_wino_reserved, wino_audit = select_winogrande_candidates(
        wino_training_candidates,
        wino_development_candidates,
        old_training=old_training,
        old_monitoring=fresh_records,
        old_monitor_source_ids=old_wino_source_ids,
        train_quota=600,
        reserve_quota=200,
    )
    training_pools["hans"] = selected_hans_training
    training_pools["winogrande"] = selected_wino_training
    if {name: len(records) for name, records in training_pools.items()} != _TRAINING_POOL_COUNTS:
        raise ValueError("selected training pools differ from the fixed recipe")
    schedules = build_schedules(training_pools)
    if any(len(rows) != 4800 for rows in schedules.values()):
        raise ValueError("paired schedules must each contain exactly 4,800 rows")

    reserved_records = (*selected_hans_reserved, *selected_wino_reserved)
    reserved_presentations = build_reserved_presentations(reserved_records)
    if len(reserved_presentations) != 1000:
        raise ValueError("reserved presentation count differs from the fixed panel")
    presentations = (
        *retention_presentations,
        *monitoring_presentations,
        *reserved_presentations,
    )
    if len(presentations) != 6182 or len({row["presentation_id"] for row in presentations}) != len(
        presentations
    ):
        raise ValueError("final panel has an unexpected count or duplicate presentation ID")

    retention_membership = {
        str(row["record_id"]): str(row["dataset_id"]) for row in retention_presentations
    }
    if len(retention_membership) != _RETENTION_RECORD_COUNT:
        raise ValueError("retention gold record membership differs from the exact allowlist")
    retention_sources: list[DecisionRecord] = []
    for relative, allowed in _RETENTION_RECORD_FILES.items():
        expected_membership = {
            record_id: dataset_id
            for record_id, dataset_id in retention_membership.items()
            if dataset_id in allowed
        }
        if not expected_membership:
            raise ValueError("retention record source has no panel membership")
        retention_sources.extend(
            _parse_jsonl_records(
                input_bytes[relative],
                relative,
                allowed_dataset_ids=allowed,
                membership=expected_membership,
            )
        )
    fresh_membership = {
        str(row["record_id"]): str(row["dataset_id"]) for row in monitoring_presentations
    }
    reserved_membership = {record.record_id: record.dataset_id for record in reserved_records}
    evaluation_source_records = (
        *retention_sources,
        *fresh_records,
        *reserved_records,
    )
    evaluation_records = _join_evaluation_records(presentations, evaluation_source_records)
    if len(evaluation_records) != _EVALUATION_RECORD_COUNT:
        raise ValueError("evaluation gold record count differs from the fixed panel")
    if (
        set(
            record.record_id
            for record in evaluation_records
            if record.dataset_id in RETENTION_DATASETS
        )
        != set(retention_membership)
        or set(
            record.record_id
            for record in evaluation_records
            if record.dataset_id in MONITORING_DATASETS
        )
        != set(fresh_membership)
        or set(
            record.record_id
            for record in evaluation_records
            if record.dataset_id in RESERVED_DATASETS
        )
        != set(reserved_membership)
    ):
        raise ValueError(
            "evaluation gold membership differs from retention, monitoring, or reserve"
        )
    _validate_new_training_separation(
        selected_hans_training,
        selected_wino_training,
        evaluation_records,
    )

    strata = {
        **{dataset_id: "retention" for dataset_id in RETENTION_DATASETS},
        **{dataset_id: "monitoring" for dataset_id in MONITORING_DATASETS},
        **{dataset_id: "reserved" for dataset_id in RESERVED_DATASETS},
    }
    selection_rows = [
        {
            "record_id": record.record_id,
            "dataset_id": record.dataset_id,
            "source_group_id": record.source_group_id,
            "request_hash": record.request.request_hash,
        }
        for record in sorted(
            (
                *selected_hans_training,
                *selected_wino_training,
                *selected_hans_reserved,
                *selected_wino_reserved,
            ),
            key=lambda row: row.record_id,
        )
    ]
    schedule_sha256 = {role: _digest(list(rows)) for role, rows in sorted(schedules.items())}
    panel_sha256 = _digest(list(presentations))
    audit: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "selection_seed": SELECTION_SEED,
        "input_file_count": _INPUT_FILE_COUNT,
        "input_file_sha256": dict(sorted(file_sha256.items())),
        "source_candidate_rows": {
            "hans_train": len(hans_training_candidates),
            "hans_evaluation": len(hans_evaluation_candidates),
            "winogrande_train_xl": len(wino_training_candidates),
            "winogrande_dev": len(wino_development_candidates),
        },
        "selection": {
            "hans": hans_audit,
            "winogrande": wino_audit,
            "selected_records_sha256": _digest(selection_rows),
            "selected_lineage": _lineage_rows(
                (
                    *hans_training_candidates,
                    *hans_evaluation_candidates,
                    *wino_training_candidates,
                    *wino_development_candidates,
                ),
                (
                    *selected_hans_training,
                    *selected_hans_reserved,
                    *selected_wino_training,
                    *selected_wino_reserved,
                ),
            ),
        },
        "training_pools": {name: len(records) for name, records in sorted(training_pools.items())},
        "schedules": {
            "rows_per_arm": 4800,
            "sha256": schedule_sha256,
        },
        "panel": {
            "presentation_count": len(presentations),
            "presentations_sha256": panel_sha256,
            "evaluation_record_count": len(evaluation_records),
            "evaluation_records_sha256": _digest(
                [record.model_dump(mode="json") for record in evaluation_records]
            ),
            "retention_presentation_count": len(retention_presentations),
            "monitoring_presentation_count": len(monitoring_presentations),
            "reserved_presentation_count": len(reserved_presentations),
        },
    }
    return TargetedInputs(
        training_pools=training_pools,
        schedules=schedules,
        evaluation_records=evaluation_records,
        presentations=tuple(dict(row) for row in presentations),
        strata=strata,
        selection=selection,
        file_sha256=file_sha256,
        audit=audit,
    )


def _record_jsonl(records: Sequence[DecisionRecord]) -> bytes:
    return b"".join(_canonical_json(record.model_dump(mode="json")) + b"\n" for record in records)


def _json_file_bytes(value: object) -> bytes:
    return _canonical_json(value) + b"\n"


def write_bundle(inputs: TargetedInputs, output_dir: str | Path) -> dict[str, object]:
    """Write one exclusive, label-separated bundle with its manifest published last."""

    if not isinstance(inputs, TargetedInputs):
        raise TypeError("bundle input must be TargetedInputs")
    if (
        set(inputs.training_pools) != set(_TRAINING_POOL_NAMES)
        or set(inputs.schedules) != {"control", "treatment"}
        or any(
            not isinstance(record, DecisionRecord)
            for pool in inputs.training_pools.values()
            for record in pool
        )
        or any(not isinstance(record, DecisionRecord) for record in inputs.evaluation_records)
        or not isinstance(inputs.selection, Mapping)
        or not isinstance(inputs.audit, Mapping)
        or not isinstance(inputs.strata, Mapping)
        or not isinstance(inputs.file_sha256, Mapping)
    ):
        raise ValueError("bundle inputs do not satisfy the targeted-data schema")
    presentations = tuple(_validate_presentation_row(row) for row in inputs.presentations)
    schedule_rows = {
        role: [dict(row) for row in inputs.schedules[role]] for role in ("control", "treatment")
    }
    training_records = tuple(
        record
        for pool_name in _TRAINING_POOL_NAMES
        for record in sorted(inputs.training_pools[pool_name], key=lambda row: row.record_id)
    )
    file_hashes = {relative: digest for relative, digest in sorted(inputs.file_sha256.items())}
    if any(
        not isinstance(relative, str)
        or not relative
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for relative, digest in file_hashes.items()
    ):
        raise ValueError("bundle input fingerprints are malformed")
    data_files = {
        "training-records.jsonl": _record_jsonl(training_records),
        "evaluation-records.jsonl": _record_jsonl(inputs.evaluation_records),
        "presentations.json": _json_file_bytes(list(presentations)),
        "schedules.json": _json_file_bytes(schedule_rows),
        "audit.json": _json_file_bytes(dict(inputs.audit)),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "input_file_sha256": file_hashes,
        "bundle_file_sha256": {
            name: hashlib.sha256(contents).hexdigest()
            for name, contents in sorted(data_files.items())
        },
    }
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(mode=0o700, exist_ok=False)
    try:
        for filename, content in (
            *data_files.items(),
            ("manifest.json", _json_file_bytes(manifest)),
        ):
            temporary = output_path / f".{filename}.tmp"
            destination = output_path / filename
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, destination)
            temporary.unlink()
        directory_fd = os.open(output_path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        shutil.rmtree(output_path, ignore_errors=True)
        raise
    return manifest
