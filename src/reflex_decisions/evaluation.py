"""Offline prediction joins and decision metrics."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .data import DataKind, DecisionRecord, SplitManifest, SplitName, audit_splits
from .scoring import DecisionResult, score_candidates


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logits: dict[str, float]
    model_revision: str

    @field_validator("record_id", "model_revision")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("logits", mode="before")
    @classmethod
    def validate_logits(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise ValueError("logits must be a nonempty mapping")
        result: dict[str, float] = {}
        for option_id, raw_logit in value.items():
            if not isinstance(option_id, str) or not option_id.strip():
                raise ValueError("logit option IDs must be nonblank strings")
            if isinstance(raw_logit, bool) or not isinstance(raw_logit, Real):
                raise ValueError("logits must be finite real numbers")
            try:
                logit = float(raw_logit)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError("logits must be finite real numbers") from exc
            if not math.isfinite(logit):
                raise ValueError("logits must be finite real numbers")
            result[option_id] = logit
        return result


class RiskCoveragePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    risk: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)


class DatasetMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    task_family: str
    split: SplitName
    record_count: int
    accuracy: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    macro_f1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    nll: float = Field(ge=0.0, allow_inf_nan=False)
    brier: float = Field(ge=0.0, allow_inf_nan=False)
    ece: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    accepted_risk: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    risk_coverage_curve: tuple[RiskCoveragePoint, ...]
    aurc: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class OverallMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_count: int
    micro_accuracy: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    macro_dataset_accuracy: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    macro_dataset_f1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    macro_dataset_nll: float = Field(ge=0.0, allow_inf_nan=False)
    macro_dataset_brier: float = Field(ge=0.0, allow_inf_nan=False)
    macro_dataset_ece: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    pooled_nll: float = Field(ge=0.0, allow_inf_nan=False)
    pooled_brier: float = Field(ge=0.0, allow_inf_nan=False)
    pooled_ece: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    accepted_risk: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    risk_coverage_curve: tuple[RiskCoveragePoint, ...]
    aurc: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_kind: DataKind
    evaluation_split: SplitName
    model_revision: str
    temperature: float = Field(gt=0.0, allow_inf_nan=False)
    calibration_id: str | None
    calibration_status: str
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    records_sha256: str
    predictions_sha256: str
    manifest_sha256: str
    python_version: str
    package_version: str
    per_dataset: tuple[DatasetMetrics, ...]
    overall: OverallMetrics


@dataclass(frozen=True, slots=True)
class _ScoredRow:
    record: DecisionRecord
    result: DecisionResult
    correct: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class _ComputedMetrics:
    accuracy: float
    macro_f1: float
    nll: float
    brier: float
    ece: float
    coverage: float
    accepted_risk: float | None
    risk_coverage_curve: tuple[RiskCoveragePoint, ...]
    aurc: float


class _DuplicateKeyError(ValueError):
    pass


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError("duplicate object key")
        result[key] = value
    return result


def load_predictions(path: str | Path) -> tuple[PredictionRecord, ...]:
    """Load strict JSON/JSONL prediction records without echoing input data."""

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{source}: could not read predictions") from exc
    entries: list[tuple[int, Any]] = []
    if source.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(text.split("\n"), start=1):
            if line.endswith("\r"):
                line = line[:-1]
            if not line.strip():
                continue
            entries.append((line_number, _decode_json(line, source, line_number)))
    elif source.suffix.lower() == ".json":
        if not text.strip():
            raise ValueError(f"{source}:1: empty predictions file")
        raw = _decode_json(text, source, 1)
        if not isinstance(raw, list):
            raise ValueError(f"{source}:1: JSON predictions must be an array")
        entries = [(index + 1, item) for index, item in enumerate(raw)]
    else:
        raise ValueError(f"{source}: unsupported predictions extension; use .json or .jsonl")
    if not entries:
        raise ValueError(f"{source}:1: predictions must not be empty")

    records: list[PredictionRecord] = []
    seen_ids: set[str] = set()
    for line_number, item in entries:
        try:
            record = PredictionRecord.model_validate(item)
        except Exception as exc:
            errors = (
                exc.errors(include_input=False, include_context=False)
                if hasattr(exc, "errors")
                else []
            )
            fields = ", ".join(
                ".".join(str(part) for part in error.get("loc", ())) or "document"
                for error in errors
            )
            raise ValueError(
                f"{source}:{line_number}: invalid prediction fields: {fields or 'document'}"
            ) from exc
        if record.record_id in seen_ids:
            raise ValueError(f"{source}:{line_number}: duplicate prediction record_id")
        seen_ids.add(record.record_id)
        records.append(record)
    return tuple(records)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_json(text: str, path: Path, line_number: int) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except _DuplicateKeyError as exc:
        raise ValueError(f"{path}:{line_number}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: malformed JSON") from exc


def _ece(rows: Sequence[_ScoredRow]) -> float:
    if not rows:
        raise ValueError("metrics require at least one row")
    total = len(rows)
    value = 0.0
    for bin_index in range(10):
        lower = bin_index / 10
        upper = (bin_index + 1) / 10
        members = [
            row
            for row in rows
            if lower <= row.confidence
            and (row.confidence < upper or (bin_index == 9 and row.confidence <= 1.0))
        ]
        if members:
            mean_confidence = math.fsum(row.confidence for row in members) / len(members)
            mean_accuracy = sum(row.correct for row in members) / len(members)
            value += (len(members) / total) * abs(mean_confidence - mean_accuracy)
    return value


def _macro_f1(rows: Sequence[_ScoredRow]) -> float:
    gold = {row.record.answer_id for row in rows}
    predicted = {row.result.top_option_id for row in rows}
    labels = gold | predicted
    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            row.record.answer_id == label and row.result.top_option_id == label for row in rows
        )
        false_positive = sum(
            row.record.answer_id != label and row.result.top_option_id == label for row in rows
        )
        false_negative = sum(
            row.record.answer_id == label and row.result.top_option_id != label for row in rows
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return math.fsum(scores) / len(scores)


def _risk_coverage(rows: Sequence[_ScoredRow]) -> tuple[tuple[RiskCoveragePoint, ...], float]:
    ordered = sorted(rows, key=lambda row: row.confidence, reverse=True)
    points: list[RiskCoveragePoint] = [RiskCoveragePoint(coverage=0.0, risk=None)]
    consumed = 0
    errors = 0
    previous_coverage = 0.0
    aurc = 0.0
    index = 0
    while index < len(ordered):
        confidence = ordered[index].confidence
        end = index
        while end < len(ordered) and ordered[end].confidence == confidence:
            end += 1
        group = ordered[index:end]
        consumed += len(group)
        errors += sum(not row.correct for row in group)
        coverage = consumed / len(ordered)
        risk = errors / consumed
        points.append(RiskCoveragePoint(coverage=coverage, risk=risk))
        aurc += (coverage - previous_coverage) * risk
        previous_coverage = coverage
        index = end
    return tuple(points), aurc


def _metrics(rows: Sequence[_ScoredRow]) -> _ComputedMetrics:
    if not rows:
        raise ValueError("metrics require at least one row")
    correct_count = sum(row.correct for row in rows)
    accepted = [row for row in rows if not row.result.abstained]
    accepted_errors = sum(not row.correct for row in accepted)
    curve, aurc = _risk_coverage(rows)
    return _ComputedMetrics(
        accuracy=correct_count / len(rows),
        macro_f1=_macro_f1(rows),
        nll=math.fsum(
            -next(
                score.log_probability
                for score in row.result.scores
                if score.option_id == row.record.answer_id
            )
            for row in rows
        )
        / len(rows),
        brier=math.fsum(
            math.fsum(
                (score.probability - (1.0 if score.option_id == row.record.answer_id else 0.0)) ** 2
                for score in row.result.scores
            )
            for row in rows
        )
        / len(rows),
        ece=_ece(rows),
        coverage=len(accepted) / len(rows),
        accepted_risk=(accepted_errors / len(accepted)) if accepted else None,
        risk_coverage_curve=curve,
        aurc=aurc,
    )


def _metric_fields(metrics: _ComputedMetrics) -> dict[str, Any]:
    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "nll": metrics.nll,
        "brier": metrics.brier,
        "ece": metrics.ece,
        "coverage": metrics.coverage,
        "accepted_risk": metrics.accepted_risk,
        "risk_coverage_curve": metrics.risk_coverage_curve,
        "aurc": metrics.aurc,
    }


def _sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def evaluate(
    records: Sequence[DecisionRecord],
    predictions: Sequence[PredictionRecord],
    manifest: SplitManifest,
    *,
    records_sha256: str,
    predictions_sha256: str,
    manifest_sha256: str,
    temperature: float | int = 1.0,
    calibration_id: str | None = None,
    min_confidence: float | int | None = None,
) -> EvaluationReport:
    """Join saved logits by record ID and produce frozen-policy offline metrics."""

    if not records:
        raise ValueError("evaluation records must not be empty")
    audit_splits(manifest, tuple(records))
    dataset_specs = {dataset.dataset_id: dataset for dataset in manifest.datasets}
    evaluation_splits = {dataset_specs[record.dataset_id].split for record in records}
    if len(evaluation_splits) != 1:
        raise ValueError("one evaluation report must contain records from one split")
    evaluation_split = next(iter(evaluation_splits))
    record_by_id: dict[str, DecisionRecord] = {}
    for record in records:
        if record.record_id in record_by_id:
            raise ValueError("duplicate evaluation record_id")
        record_by_id[record.record_id] = record
    prediction_by_id: dict[str, PredictionRecord] = {}
    for prediction in predictions:
        if prediction.record_id in prediction_by_id:
            raise ValueError("duplicate prediction record_id")
        prediction_by_id[prediction.record_id] = prediction
    missing = set(record_by_id) - set(prediction_by_id)
    extra = set(prediction_by_id) - set(record_by_id)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing predictions for {len(missing)} record(s)")
        if extra:
            details.append(f"unexpected predictions for {len(extra)} record(s)")
        raise ValueError("; ".join(details))
    revisions = {prediction.model_revision for prediction in prediction_by_id.values()}
    if len(revisions) != 1:
        raise ValueError("all predictions must use one model_revision")
    model_revision = next(iter(revisions))

    scored: list[_ScoredRow] = []
    datasets = dataset_specs
    for record in records:
        prediction = prediction_by_id[record.record_id]
        if prediction.request_hash != record.request.request_hash:
            raise ValueError(
                f"prediction for record {record.record_id!r} has a mismatched request_hash"
            )
        expected_ids = {option.id for option in record.request.options}
        if set(prediction.logits) != expected_ids:
            raise ValueError(f"prediction for record {record.record_id!r} has the wrong option IDs")
        logits = [prediction.logits[option.id] for option in record.request.options]
        result = score_candidates(
            record.request,
            logits,
            model_revision=prediction.model_revision,
            temperature=temperature,
            calibration_id=calibration_id,
            min_confidence=min_confidence,
        )
        scores = {score.option_id: score.probability for score in result.scores}
        scored.append(
            _ScoredRow(
                record=record,
                result=result,
                correct=result.top_option_id == record.answer_id,
                confidence=max(scores.values()),
            )
        )

    rows_by_dataset: dict[str, list[_ScoredRow]] = defaultdict(list)
    for row in scored:
        rows_by_dataset[row.record.dataset_id].append(row)
    per_dataset: list[DatasetMetrics] = []
    dataset_metrics: list[_ComputedMetrics] = []
    for dataset_id in sorted(rows_by_dataset):
        dataset = datasets[dataset_id]
        rows = rows_by_dataset[dataset_id]
        metrics = _metrics(rows)
        dataset_metrics.append(metrics)
        per_dataset.append(
            DatasetMetrics(
                dataset_id=dataset_id,
                task_family=dataset.task_family,
                split=dataset.split,
                record_count=len(rows),
                **_metric_fields(metrics),
            )
        )
    pooled = _metrics(scored)
    overall = OverallMetrics(
        record_count=len(scored),
        micro_accuracy=pooled.accuracy,
        macro_dataset_accuracy=math.fsum(metric.accuracy for metric in dataset_metrics)
        / len(dataset_metrics),
        macro_dataset_f1=math.fsum(metric.macro_f1 for metric in dataset_metrics)
        / len(dataset_metrics),
        macro_dataset_nll=math.fsum(metric.nll for metric in dataset_metrics)
        / len(dataset_metrics),
        macro_dataset_brier=math.fsum(metric.brier for metric in dataset_metrics)
        / len(dataset_metrics),
        macro_dataset_ece=math.fsum(metric.ece for metric in dataset_metrics)
        / len(dataset_metrics),
        pooled_nll=pooled.nll,
        pooled_brier=pooled.brier,
        pooled_ece=pooled.ece,
        coverage=pooled.coverage,
        accepted_risk=pooled.accepted_risk,
        risk_coverage_curve=pooled.risk_coverage_curve,
        aurc=pooled.aurc,
    )
    return EvaluationReport(
        data_kind=manifest.data_kind,
        evaluation_split=evaluation_split,
        model_revision=model_revision,
        temperature=float(temperature),
        calibration_id=calibration_id,
        calibration_status="uncalibrated" if calibration_id is None else "profile_id_supplied",
        min_confidence=float(min_confidence) if min_confidence is not None else None,
        records_sha256=_sha256(records_sha256, "records_sha256"),
        predictions_sha256=_sha256(predictions_sha256, "predictions_sha256"),
        manifest_sha256=_sha256(manifest_sha256, "manifest_sha256"),
        python_version=platform.python_version(),
        package_version=__version__,
        per_dataset=tuple(per_dataset),
        overall=overall,
    )
