"""Data contract for the pinned fresh-eval-v1 evaluation panel."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from .data import DecisionRecord
from .schema import DecisionRequest, Option

SELECTION_SEED = 20261011
HANS_DATASET_ID = "hans-eval-v1"
WINOGRANDE_DATASET_ID = "winogrande-dev-v1"
ARC_DATASET_ID = "arc-challenge-dev-v1"
DATASET_COUNTS = {
    HANS_DATASET_ID: 300,
    WINOGRANDE_DATASET_ID: 200,
    ARC_DATASET_ID: 200,
}
HANS_SUBCASE_COUNT = 30
HANS_PER_SUBCASE = 10


@dataclass(frozen=True)
class _Candidate:
    dataset_id: str
    raw_source_id: str
    source_group_id: str
    request: DecisionRequest
    answer_id: str
    source_metadata: dict[str, object]


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record_sha256(record: DecisionRecord) -> str:
    """Hash canonical record JSON, including the host-only gold label."""

    return _sha256(_canonical_json(record.model_dump(mode="json")))


def _selection_rank_sha256(dataset_id: str, raw_source_id: str) -> str:
    payload = f"fresh-eval-v1:{SELECTION_SEED}:{dataset_id}:{raw_source_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_hans_tsv(payload: bytes) -> list[_Candidate]:
    """Parse HANS evaluation rows into gold-separated decision candidates."""

    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("HANS source is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    required = {
        "gold_label",
        "sentence1",
        "sentence2",
        "pairID",
        "heuristic",
        "subcase",
        "template",
    }
    if reader.fieldnames is None or not required <= set(reader.fieldnames):
        raise ValueError("HANS source has an unexpected header")
    output: list[_Candidate] = []
    seen_ids: dict[str, _Candidate] = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("HANS source contains a malformed row")
        values = {key: row[key].strip() for key in required}
        if any(not value for value in values.values()):
            raise ValueError("HANS source contains a blank required field")
        if values["gold_label"] not in {"entailment", "non-entailment"}:
            raise ValueError("HANS source contains an unsupported gold label")
        subcase_digest = hashlib.sha256(values["subcase"].encode("utf-8")).hexdigest()
        candidate = _Candidate(
            dataset_id=HANS_DATASET_ID,
            raw_source_id=values["pairID"],
            source_group_id=f"hans-subcase:{subcase_digest}",
            request=DecisionRequest(
                context=values["sentence1"],
                question=(
                    "Does the following statement necessarily follow from the context? "
                    f"{values['sentence2']}"
                ),
                options=(
                    Option(id="entailment", label="Follows from the sentence"),
                    Option(
                        id="non-entailment",
                        label="Does not necessarily follow",
                    ),
                ),
            ),
            answer_id=values["gold_label"],
            source_metadata={key: values[key] for key in ("heuristic", "subcase", "template")},
        )
        previous = seen_ids.get(candidate.raw_source_id)
        if previous is not None:
            if previous == candidate:
                continue
            if (
                previous.request.request_hash == candidate.request.request_hash
                and previous.answer_id != candidate.answer_id
            ):
                raise ValueError("HANS duplicate source ID has conflicting answer labels")
            raise ValueError("duplicate HANS source ID is ambiguous")
        seen_ids[candidate.raw_source_id] = candidate
        output.append(candidate)
    return output


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("source JSON contains duplicate object keys")
        result[key] = value
    return result


def parse_winogrande_dev(data: bytes, labels: bytes) -> list[_Candidate]:
    """Parse the v1.1 WinoGrande dev rows with their separately aligned labels."""

    try:
        data_text = data.decode("utf-8-sig", errors="strict")
        labels_text = labels.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("WinoGrande dev source is not UTF-8") from exc
    data_lines = data_text.splitlines()
    label_lines = labels_text.splitlines()
    if not data_lines or len(data_lines) != len(label_lines):
        raise ValueError("WinoGrande dev labels are not fully aligned")
    output: list[_Candidate] = []
    seen_ids: dict[str, _Candidate] = {}
    for data_line, label_line in zip(data_lines, label_lines, strict=True):
        if not data_line.strip():
            raise ValueError("WinoGrande dev source contains a blank row")
        try:
            row = json.loads(data_line, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("WinoGrande dev source contains malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError("WinoGrande dev source row is not an object")
        raw_id = row.get("qID")
        sentence = row.get("sentence")
        first = row.get("option1")
        second = row.get("option2")
        if (
            not isinstance(raw_id, str)
            or not raw_id.strip()
            or not isinstance(sentence, str)
            or not sentence.strip()
            or not isinstance(first, str)
            or not first.strip()
            or not isinstance(second, str)
            or not second.strip()
        ):
            raise ValueError("WinoGrande dev source row has blank or malformed fields")
        raw_id, sentence, first, second = (
            raw_id.strip(),
            sentence.strip(),
            first.strip(),
            second.strip(),
        )
        answer = label_line.strip()
        if answer not in {"1", "2"}:
            raise ValueError("WinoGrande dev source contains an unsupported answer label")
        if sentence.count("_") != 1:
            raise ValueError("WinoGrande sentence must contain exactly one blank")
        candidate = _Candidate(
            dataset_id=WINOGRANDE_DATASET_ID,
            raw_source_id=raw_id,
            source_group_id=raw_id,
            request=DecisionRequest(
                context=sentence,
                question="Which option fills the blank?",
                options=(
                    Option(id="option1", label=first),
                    Option(id="option2", label=second),
                ),
            ),
            answer_id=f"option{answer}",
            source_metadata={"split": "dev"},
        )
        previous = seen_ids.get(raw_id)
        if previous is not None:
            if previous == candidate:
                continue
            if (
                previous.request.request_hash == candidate.request.request_hash
                and previous.answer_id != candidate.answer_id
            ):
                raise ValueError("WinoGrande duplicate source ID has conflicting answer labels")
            raise ValueError("duplicate WinoGrande source ID is ambiguous")
        seen_ids[raw_id] = candidate
        output.append(candidate)
    return output


def parse_arc_dev_jsonl(payload: bytes) -> list[_Candidate]:
    """Parse ARC-Challenge-Dev rows while keeping the answer key outside requests."""

    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("ARC dev source is not UTF-8") from exc
    output: list[_Candidate] = []
    seen_ids: dict[str, _Candidate] = {}
    for line in text.splitlines():
        if not line.strip():
            raise ValueError("ARC dev source contains a blank row")
        try:
            row = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("ARC dev source contains malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError("ARC dev source row is not an object")
        raw_id = row.get("id")
        question = row.get("question")
        answer_key = row.get("answerKey")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("ARC dev source row has a blank question ID")
        if not isinstance(question, dict):
            raise ValueError("ARC dev question has an unexpected schema")
        stem = question.get("stem")
        choices = question.get("choices")
        if not isinstance(stem, str) or not stem.strip():
            raise ValueError("ARC dev question has a blank stem")
        if not isinstance(choices, list) or not 2 <= len(choices) <= 16:
            raise ValueError("ARC dev question must have between 2 and 16 choices")
        options: list[Option] = []
        labels: set[str] = set()
        for choice in choices:
            if not isinstance(choice, dict):
                raise ValueError("ARC dev choice has an unexpected schema")
            label = choice.get("label")
            choice_text = choice.get("text")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("ARC dev choice has a blank label")
            if label in labels:
                raise ValueError("ARC dev question contains duplicate choice IDs")
            if not isinstance(choice_text, str) or not choice_text.strip():
                raise ValueError("ARC dev choice has blank text")
            labels.add(label)
            options.append(Option(id=label, label=choice_text))
        if not isinstance(answer_key, str) or answer_key not in labels:
            raise ValueError("ARC dev answerKey does not match a choice label")
        candidate = _Candidate(
            dataset_id=ARC_DATASET_ID,
            raw_source_id=raw_id,
            source_group_id=raw_id,
            request=DecisionRequest(
                context=stem,
                question="Choose the correct answer.",
                options=tuple(options),
            ),
            answer_id=answer_key,
            source_metadata={"split": "ARC-Challenge-Dev", "choice_count": len(options)},
        )
        previous = seen_ids.get(raw_id)
        if previous is not None:
            if previous == candidate:
                continue
            if (
                previous.request.request_hash == candidate.request.request_hash
                and previous.answer_id != candidate.answer_id
            ):
                raise ValueError("ARC duplicate source ID has conflicting answer labels")
            raise ValueError("duplicate ARC source ID is ambiguous")
        seen_ids[raw_id] = candidate
        output.append(candidate)
    if not output:
        raise ValueError("ARC dev source contains no records")
    return output


def _dedupe_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """Collapse exact duplicate IDs/requests and reject conflicting gold mappings."""

    by_source_id: dict[tuple[str, str], _Candidate] = {}
    for candidate in candidates:
        source_key = (candidate.dataset_id, candidate.raw_source_id)
        previous_id = by_source_id.get(source_key)
        if previous_id is None:
            by_source_id[source_key] = candidate
            continue
        if previous_id == candidate:
            continue
        if (
            previous_id.request.request_hash == candidate.request.request_hash
            and previous_id.answer_id != candidate.answer_id
        ):
            raise ValueError("semantic request has conflicting answer labels")
        raise ValueError("duplicate source ID is ambiguous")

    ranked = sorted(
        by_source_id.values(),
        key=lambda candidate: (
            _selection_rank_sha256(candidate.dataset_id, candidate.raw_source_id),
            candidate.dataset_id,
            candidate.raw_source_id,
        ),
    )
    by_request: dict[str, _Candidate] = {}
    output: list[_Candidate] = []
    for candidate in ranked:
        fingerprint = candidate.request.request_hash
        previous_request = by_request.get(fingerprint)
        if previous_request is not None:
            if previous_request.answer_id != candidate.answer_id:
                raise ValueError("semantic request has conflicting answer labels")
            continue
        by_request[fingerprint] = candidate
        output.append(candidate)
    return output


# Keep the stable public data API here; the larger preparation implementation is
# separated so parsers and bundle verification remain independently reviewable.
def _select_panel(
    candidates: list[_Candidate],
    *,
    preview_questions: set[str],
    preview_question_sha256s: set[str] | None = None,
) -> tuple[list[DecisionRecord], dict[str, object]]:
    from .fresh_eval_build import _select_panel as select

    return select(
        candidates,
        preview_questions=preview_questions,
        preview_question_sha256s=preview_question_sha256s,
    )


def build_presentations(records: object) -> list[dict[str, object]]:
    from .fresh_eval_build import build_presentations as build

    return build(records)


def prepare_panel(root: str | Path) -> tuple[list[DecisionRecord], dict[str, object]]:
    from .fresh_eval_bundle import prepare_panel as prepare

    return prepare(root)


def load_panel(root: str | Path) -> tuple[list[DecisionRecord], dict[str, object]]:
    from .fresh_eval_bundle import load_panel as load

    return load(root)
