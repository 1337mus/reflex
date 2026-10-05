"""Deterministic selection and label-free presentations for fresh evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from .data import DecisionRecord
from .fresh_eval_data import (
    ARC_DATASET_ID,
    DATASET_COUNTS,
    HANS_DATASET_ID,
    HANS_PER_SUBCASE,
    HANS_SUBCASE_COUNT,
    SELECTION_SEED,
    WINOGRANDE_DATASET_ID,
    _Candidate,
    _canonical_json,
    _dedupe_candidates,
    _record_sha256,
    _selection_rank_sha256,
    _sha256,
)


def _opaque_record_id(candidate: _Candidate) -> str:
    rank = _selection_rank_sha256(candidate.dataset_id, candidate.raw_source_id)
    return f"fresh-eval-v1:{candidate.dataset_id}:{rank}"


def _opaque_group_id(candidate: _Candidate) -> str:
    digest = _sha256(f"fresh-eval-v1:{candidate.dataset_id}:{candidate.source_group_id}".encode())
    return f"fresh-eval-v1:{candidate.dataset_id}:group:{digest}"


def _select_panel(
    candidates: list[_Candidate],
    *,
    preview_questions: set[str],
    preview_question_sha256s: set[str] | None = None,
) -> tuple[list[DecisionRecord], dict[str, object]]:
    """Select the label-independent frozen panel and retain source details on CPU."""

    if preview_question_sha256s is None:
        preview_question_sha256s = set()
    deduped = _dedupe_candidates(candidates)
    candidates_by_dataset: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in deduped:
        if candidate.dataset_id not in DATASET_COUNTS:
            raise ValueError("fresh evaluation source contains an unexpected dataset")
        candidates_by_dataset[candidate.dataset_id].append(candidate)

    hans_groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates_by_dataset[HANS_DATASET_ID]:
        hans_groups[candidate.source_group_id].append(candidate)
    if len(hans_groups) != HANS_SUBCASE_COUNT:
        raise ValueError("HANS evaluation source must contain exactly 30 subcases")
    selected_candidates: list[_Candidate] = []
    for group_id in sorted(hans_groups):
        ranked = sorted(
            hans_groups[group_id],
            key=lambda candidate: (
                _selection_rank_sha256(candidate.dataset_id, candidate.raw_source_id),
                candidate.raw_source_id,
            ),
        )
        if len(ranked) < HANS_PER_SUBCASE:
            raise ValueError("HANS evaluation subcase has fewer than ten unique rows")
        selected_candidates.extend(ranked[:HANS_PER_SUBCASE])

    for dataset_id in (WINOGRANDE_DATASET_ID, ARC_DATASET_ID):
        eligible = candidates_by_dataset[dataset_id]
        preview_excluded = 0
        if dataset_id == ARC_DATASET_ID:
            retained: list[_Candidate] = []
            for candidate in eligible:
                stem = candidate.request.context.strip()
                stem_digest = _sha256(stem.encode("utf-8"))
                if stem in preview_questions or stem_digest in preview_question_sha256s:
                    preview_excluded += 1
                else:
                    retained.append(candidate)
            eligible = retained
        ranked = sorted(
            eligible,
            key=lambda candidate: (
                _selection_rank_sha256(candidate.dataset_id, candidate.raw_source_id),
                candidate.raw_source_id,
            ),
        )
        if len(ranked) < DATASET_COUNTS[dataset_id]:
            raise ValueError(f"{dataset_id} source has too few unique eligible rows")
        selected_candidates.extend(ranked[: DATASET_COUNTS[dataset_id]])
        if dataset_id == ARC_DATASET_ID:
            arc_preview_excluded = preview_excluded

    records = [
        DecisionRecord(
            record_id=_opaque_record_id(candidate),
            dataset_id=candidate.dataset_id,
            source_group_id=_opaque_group_id(candidate),
            request=candidate.request,
            answer_id=candidate.answer_id,
        )
        for candidate in selected_candidates
    ]
    records.sort(key=lambda record: (record.dataset_id, record.record_id))
    if len({record.record_id for record in records}) != sum(DATASET_COUNTS.values()):
        raise ValueError("fresh evaluation selected duplicate opaque record IDs")
    counts = Counter(record.dataset_id for record in records)
    if dict(counts) != DATASET_COUNTS:
        raise ValueError("fresh evaluation selected task counts differ from the frozen panel")

    source_by_identity = {
        (candidate.dataset_id, candidate.raw_source_id): candidate
        for candidate in selected_candidates
    }
    metadata_by_id: dict[str, dict[str, object]] = {}
    selection_source_ids: dict[str, list[str]] = {dataset_id: [] for dataset_id in DATASET_COUNTS}
    for record in records:
        candidates_match = [
            candidate
            for (dataset_id, _raw_id), candidate in source_by_identity.items()
            if dataset_id == record.dataset_id and _opaque_record_id(candidate) == record.record_id
        ]
        if len(candidates_match) != 1:
            raise ValueError("fresh evaluation record metadata cannot be matched uniquely")
        candidate = candidates_match[0]
        rank_digest = _selection_rank_sha256(candidate.dataset_id, candidate.raw_source_id)
        row: dict[str, object] = {
            "dataset_id": record.dataset_id,
            "source_group_id": record.source_group_id,
            "raw_source_id": candidate.raw_source_id,
            "raw_source_group_id": candidate.source_group_id,
            "selection_rank_sha256": rank_digest,
            "semantic_request_sha256": record.request.request_hash,
            "record_sha256": _record_sha256(record),
        }
        row.update(candidate.source_metadata)
        metadata_by_id[record.record_id] = row
        selection_source_ids[record.dataset_id].append(candidate.raw_source_id)

    hans_sizes = Counter(
        record.source_group_id for record in records if record.dataset_id == HANS_DATASET_ID
    )
    if len(hans_sizes) != HANS_SUBCASE_COUNT or set(hans_sizes.values()) != {HANS_PER_SUBCASE}:
        raise ValueError("fresh evaluation HANS panel is not ten rows from each subcase")
    selected_identity_hashes = {
        dataset_id: sorted(_selection_rank_sha256(dataset_id, raw_id) for raw_id in raw_ids)
        for dataset_id, raw_ids in selection_source_ids.items()
    }
    selection_digest = _sha256(
        _canonical_json(
            {
                "seed": SELECTION_SEED,
                "selected_rank_sha256_by_dataset": selected_identity_hashes,
                "known_arc_preview_excluded": arc_preview_excluded,
                "known_arc_preview_stem_sha256": sorted(preview_question_sha256s),
            }
        )
    )
    panel_metadata = {
        "selection": {
            "seed": SELECTION_SEED,
            "counts_by_dataset": dict(counts),
            "hans_subcase_count": HANS_SUBCASE_COUNT,
            "hans_per_subcase": HANS_PER_SUBCASE,
            "hans_per_label": {
                label: sum(
                    record.answer_id == label
                    for record in records
                    if record.dataset_id == HANS_DATASET_ID
                )
                for label in ("entailment", "non-entailment")
            },
            "known_arc_preview_excluded": arc_preview_excluded,
            "known_arc_preview_stem_sha256": sorted(preview_question_sha256s),
            "selection_digest_sha256": selection_digest,
        },
        "record_metadata_by_id": metadata_by_id,
    }
    return records, panel_metadata


def build_presentations(records: object) -> list[dict[str, object]]:
    """Build the fixed label-free panel with original and left-rotated choices."""

    if not isinstance(records, Sequence) or len(records) != sum(DATASET_COUNTS.values()):
        raise ValueError("fresh evaluation panel must contain exactly 700 records")
    typed_records = tuple(records)
    if any(not isinstance(record, DecisionRecord) for record in typed_records):
        raise ValueError("fresh evaluation panel contains a malformed record")
    counts = Counter(record.dataset_id for record in typed_records)
    if dict(counts) != DATASET_COUNTS or len({row.record_id for row in typed_records}) != len(
        typed_records
    ):
        raise ValueError("fresh evaluation panel has unexpected dataset counts or duplicate IDs")
    if tuple(typed_records) != tuple(
        sorted(typed_records, key=lambda row: (row.dataset_id, row.record_id))
    ):
        raise ValueError("fresh evaluation records are not sorted by dataset and record ID")

    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in typed_records:
        group_counts[record.dataset_id][record.source_group_id] += 1
    if len(group_counts[HANS_DATASET_ID]) != HANS_SUBCASE_COUNT or set(
        group_counts[HANS_DATASET_ID].values()
    ) != {HANS_PER_SUBCASE}:
        raise ValueError("fresh evaluation HANS groups differ from the frozen subcase panel")
    for dataset_id in (WINOGRANDE_DATASET_ID, ARC_DATASET_ID):
        if len(group_counts[dataset_id]) != DATASET_COUNTS[dataset_id] or set(
            group_counts[dataset_id].values()
        ) != {1}:
            raise ValueError(f"{dataset_id} must use each question as a source group")

    output: list[dict[str, object]] = []
    for record in typed_records:
        for order_index, options in enumerate(
            (record.request.options, record.request.options[1:] + record.request.options[:1])
        ):
            request = record.request.model_copy(update={"options": options})
            output.append(
                {
                    "presentation_id": f"fresh-eval-v1:{record.record_id}:order-{order_index}",
                    "record_id": record.record_id,
                    "dataset_id": record.dataset_id,
                    "source_group_id": record.source_group_id,
                    "request_hash": request.request_hash,
                    "order_index": order_index,
                    "order_ids": [option.id for option in request.options],
                    "request": request.model_dump(mode="json"),
                }
            )
    return output
