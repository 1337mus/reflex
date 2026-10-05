"""Conservative streaming parser for the private SNLI training candidate."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from . import broader_data
from .broader_data import SNLI_LABELS, SnliItem


@dataclass(frozen=True)
class SnliParseStats:
    """Aggregate source counts retained without storing raw source lines."""

    raw_row_count: int
    label_row_counts: tuple[tuple[str, int], ...]
    unlabeled_row_count: int
    normalized_pair_count: int
    conflicting_pair_count: int
    conflicting_labeled_row_count: int
    consistent_labeled_pair_count: int
    unlabeled_only_pair_count: int
    source_group_count: int


@dataclass(frozen=True)
class ParsedSnliSource:
    """Deduplicated items plus all source links needed for conservative exclusions."""

    items: tuple[SnliItem, ...]
    pair_group_ids: dict[tuple[str, str], str]
    pair_anchors: dict[tuple[str, str], frozenset[tuple[str, str]]]
    group_anchors: dict[str, frozenset[tuple[str, str]]]
    pair_keys: frozenset[tuple[str, str]]
    stats: SnliParseStats


def _parse_snli_jsonl(
    lines: Iterable[str],
    *,
    include_unlabeled_anchors: bool = False,
    conflict_policy: Literal["raise", "exclude"] = "raise",
    source_name: str = "SNLI source",
) -> ParsedSnliSource:
    """Parse, deduplicate, and group SNLI rows before selecting labeled items."""

    parsed: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    labels_by_pair: dict[tuple[str, str], set[str]] = {}
    labeled_rows_by_pair: dict[tuple[str, str], int] = {}
    row_anchors: dict[tuple[str, str], set[tuple[str, str]]] = {}
    label_row_counts = {label: 0 for label in SNLI_LABELS}
    raw_row_count = 0
    unlabeled_row_count = 0
    allowed_labels = set(SNLI_LABELS)

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        raw_row_count += 1
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source_name} line {line_number} is malformed JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{source_name} line {line_number} must be an object")
        label = row.get("gold_label")
        is_unlabeled = label == "-"
        if is_unlabeled and not include_unlabeled_anchors:
            continue
        if include_unlabeled_anchors and not isinstance(label, str):
            raise ValueError(f"{source_name} line {line_number} has an unknown gold label")
        if not is_unlabeled and label not in allowed_labels:
            raise ValueError(f"{source_name} line {line_number} has an unknown gold label")
        pair_id = row.get("pairID")
        premise, hypothesis = row.get("sentence1"), row.get("sentence2")
        if include_unlabeled_anchors and "captionID" not in row:
            raise ValueError(f"{source_name} line {line_number} captionID is missing")
        caption_id = row.get("captionID", "")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"{source_name} line {line_number} has a blank pairID")
        if not isinstance(premise, str) or not broader_data._normalize_text(premise):
            raise ValueError(f"{source_name} line {line_number} has a blank premise")
        if not isinstance(hypothesis, str) or not broader_data._normalize_text(hypothesis):
            raise ValueError(f"{source_name} line {line_number} has a blank hypothesis")
        if not isinstance(caption_id, str):
            raise ValueError(f"{source_name} line {line_number} captionID must be text")

        premise_norm = broader_data._normalize_text(premise)
        hypothesis_norm = broader_data._normalize_text(hypothesis)
        semantic_key = (premise_norm, hypothesis_norm)
        caption_norm = broader_data._normalize_text(caption_id)
        anchors = row_anchors.setdefault(semantic_key, set())
        anchors.add(("premise", premise_norm))
        if caption_norm:
            anchors.add(("caption", caption_norm))
        labels = labels_by_pair.setdefault(semantic_key, set())
        if is_unlabeled:
            unlabeled_row_count += 1
            continue
        if not isinstance(label, str):
            raise ValueError(f"{source_name} line {line_number} has an unknown gold label")

        label_row_counts[label] += 1
        labeled_rows_by_pair[semantic_key] = labeled_rows_by_pair.get(semantic_key, 0) + 1
        if label not in labels and labels and conflict_policy == "raise":
            raise ValueError("normalized duplicate SNLI premise/hypothesis has conflicting labels")
        labels.add(label)
        candidate = (
            pair_id.strip(),
            premise.strip(),
            hypothesis.strip(),
            caption_id.strip(),
            label,
        )
        previous = parsed.get(semantic_key)
        if previous is None or candidate[:4] < previous[:4]:
            parsed[semantic_key] = candidate

    keys = sorted(row_anchors)
    parents = list(range(len(keys)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[max(root_left, root_right)] = min(root_left, root_right)

    by_source: dict[tuple[str, str], int] = {}
    for index, key in enumerate(keys):
        for anchor in row_anchors[key]:
            previous_index = by_source.setdefault(anchor, index)
            union(index, previous_index)

    component_anchors: dict[int, set[tuple[str, str]]] = {}
    for index, key in enumerate(keys):
        component_anchors.setdefault(find(index), set()).update(row_anchors[key])
    group_ids = {
        root: broader_data._digest(
            json.dumps(sorted(anchors), ensure_ascii=False, separators=(",", ":"))
        )
        for root, anchors in component_anchors.items()
    }
    pair_group_ids = {key: group_ids[find(index)] for index, key in enumerate(keys)}
    group_anchors = {
        group_ids[root]: frozenset(anchors) for root, anchors in component_anchors.items()
    }
    items = tuple(
        SnliItem(
            source_item_id=parsed[key][0],
            source_group_id=pair_group_ids[key],
            premise=parsed[key][1],
            hypothesis=parsed[key][2],
            label=parsed[key][4],
        )
        for key in keys
        if len(labels_by_pair[key]) == 1 and key in parsed
    )
    conflicting_keys = {key for key, labels in labels_by_pair.items() if len(labels) > 1}
    consistent_keys = {key for key, labels in labels_by_pair.items() if len(labels) == 1}
    stats = SnliParseStats(
        raw_row_count=raw_row_count,
        label_row_counts=tuple((label, label_row_counts[label]) for label in SNLI_LABELS),
        unlabeled_row_count=unlabeled_row_count,
        normalized_pair_count=len(row_anchors),
        conflicting_pair_count=len(conflicting_keys),
        conflicting_labeled_row_count=sum(labeled_rows_by_pair[key] for key in conflicting_keys),
        consistent_labeled_pair_count=len(consistent_keys),
        unlabeled_only_pair_count=sum(not labels for labels in labels_by_pair.values()),
        source_group_count=len(component_anchors),
    )
    return ParsedSnliSource(
        items=items,
        pair_group_ids=pair_group_ids,
        pair_anchors={key: frozenset(value) for key, value in row_anchors.items()},
        group_anchors=group_anchors,
        pair_keys=frozenset(row_anchors),
        stats=stats,
    )
