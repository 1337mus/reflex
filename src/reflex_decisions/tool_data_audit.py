"""Validation and aggregate audits for tool-choice corpus candidates."""

from __future__ import annotations

from dataclasses import replace

from .data import DatasetSpec, SplitManifest, SplitName, audit_splits
from .rendering import render_prompt
from .tool_data_identity import (
    catalog_hash,
    scenario_hash,
    split_for_structure,
    structural_catalog_hash,
)
from .tool_data_spec import (
    CASE_KINDS,
    FAMILY,
    SOURCE_ID,
    SPLITS,
    TEMPLATE_IDS,
    VOCABULARIES,
    ToolDatum,
)
from .tool_rules import (
    INSUFFICIENT_INFORMATION,
    NO_ELIGIBLE_TOOL,
    ToolScenario,
    solve,
)
from .tool_text_parser import parse_tool_prompt, verify_tool_request


def classify_scenario(scenario: ToolScenario) -> str:
    if not isinstance(scenario, ToolScenario):
        raise TypeError("scenario must be a ToolScenario")
    missing = tuple(
        index for index, value in enumerate(scenario.permission_values) if value is None
    )
    if len(missing) >= 2:
        solve(scenario)
        return "multiple_missing"
    if not missing:
        answer = solve(scenario)
        return "complete_no_eligible" if answer == NO_ELIGIBLE_TOOL else "complete_tool"

    index = missing[0]
    outcomes = []
    for value in (False, True):
        permission_values = list(scenario.permission_values)
        permission_values[index] = value
        outcomes.append(solve(replace(scenario, permission_values=tuple(permission_values))))
    if outcomes[0] == outcomes[1]:
        return "missing_none_agree" if outcomes[0] == NO_ELIGIBLE_TOOL else "missing_tool_agree"
    if NO_ELIGIBLE_TOOL in outcomes:
        return "missing_tool_vs_none"
    if INSUFFICIENT_INFORMATION in outcomes:
        raise ValueError("a complete permission assignment cannot be insufficient")
    return "missing_tool_conflict"


def audit_tool_data(rows: tuple[ToolDatum, ...], manifest: SplitManifest) -> dict[str, object]:
    if not rows:
        raise ValueError("tool data rows must not be empty")

    vocabulary_terms = [
        set(vocabulary.capabilities)
        | set(vocabulary.input_types)
        | set(vocabulary.permissions)
        | set(vocabulary.tool_ids)
        for vocabulary in (VOCABULARIES[split] for split in SPLITS)
    ]
    if any(
        left & right
        for index, left in enumerate(vocabulary_terms)
        for right in vocabulary_terms[index + 1 :]
    ):
        raise ValueError("protected vocabularies must be disjoint across splits")

    if len(manifest.datasets) != len(SPLITS):
        raise ValueError("manifest must declare exactly one dataset per split")
    datasets: dict[SplitName, DatasetSpec] = {}
    for dataset_spec in manifest.datasets:
        if dataset_spec.source_id != SOURCE_ID or dataset_spec.task_family != FAMILY:
            raise ValueError("manifest has an unapproved source or task family")
        if dataset_spec.split in datasets:
            raise ValueError("manifest must declare exactly one dataset per split")
        datasets[dataset_spec.split] = dataset_spec
    if set(datasets) != set(SPLITS):
        raise ValueError("manifest must declare every tool-data split")
    if SOURCE_ID not in manifest.group_partitioned_sources:
        raise ValueError("manifest must group-partition the tool-data source")

    records = tuple(row.record for row in rows)
    audit_splits(manifest, records)
    datasets_by_id = {dataset.dataset_id: dataset for dataset in manifest.datasets}
    source_groups: set[str] = set()
    seen_scenarios: set[str] = set()
    seen_catalogs: set[str] = set()
    seen_structures: set[str] = set()
    split_counts: dict[SplitName, int] = {split: 0 for split in SPLITS}
    case_counts: dict[SplitName, dict[str, int]] = {
        split: {kind: 0 for kind in CASE_KINDS} for split in SPLITS
    }
    menu_counts: dict[SplitName, dict[int, int]] = {split: {} for split in SPLITS}
    answer_counts: dict[SplitName, dict[int, dict[str, int]]] = {split: {} for split in SPLITS}
    multiple_counts: dict[SplitName, dict[int, dict[str, int]]] = {split: {} for split in SPLITS}
    classes = ("tool", "no_eligible_tool", "insufficient_information")

    for row in rows:
        if row.split not in SPLITS:
            raise ValueError("row has an unapproved split")
        row_dataset = datasets_by_id.get(row.record.dataset_id)
        if row_dataset is None or row_dataset.split != row.split:
            raise ValueError("row split does not match its manifest dataset")
        if row.template_id != TEMPLATE_IDS[row.split]:
            raise ValueError("template ID does not match the declared split")

        scenario = row.scenario
        vocabulary = VOCABULARIES[row.split]
        if (
            set(scenario.capabilities) != set(vocabulary.capabilities)
            or set(scenario.input_types) != set(vocabulary.input_types)
            or set(scenario.permissions) != set(vocabulary.permissions)
        ):
            raise ValueError("scenario domains do not match the protected split vocabulary")
        if len(scenario.tools) not in (2, 4, 6) or not {tool.id for tool in scenario.tools} <= set(
            vocabulary.tool_ids
        ):
            raise ValueError("tool menu does not match the protected split vocabulary")

        _verify_visible_prompt(row)
        answer = solve(scenario)
        if row.record.answer_id != answer:
            raise ValueError("record answer does not match the tool solver")
        expected_case = classify_scenario(scenario)
        if row.case_kind != expected_case:
            raise ValueError("case kind does not match the scenario outcomes")

        scenario_digest = scenario_hash(scenario)
        if scenario_digest in seen_scenarios:
            raise ValueError("duplicate tool scenario hash")
        seen_scenarios.add(scenario_digest)
        catalog_digest = catalog_hash(scenario)
        if catalog_digest in seen_catalogs:
            raise ValueError("duplicate named tool catalog")
        seen_catalogs.add(catalog_digest)
        structure_digest = structural_catalog_hash(scenario)
        if structure_digest in seen_structures:
            raise ValueError("duplicate structural tool catalog")
        seen_structures.add(structure_digest)
        expected_group = "tool-group-" + structure_digest
        if row.record.source_group_id != expected_group:
            raise ValueError("source group ID does not match the structural catalog hash")
        if split_for_structure(structure_digest) != row.split:
            raise ValueError("structural catalog is assigned to a different split")

        source_groups.add(row.record.source_group_id)
        split_counts[row.split] += 1
        case_counts[row.split][row.case_kind] += 1
        menu_size = len(row.record.request.options)
        menu_counts[row.split][menu_size] = menu_counts[row.split].get(menu_size, 0) + 1
        answer_class = _answer_class(answer)
        _increment(answer_counts[row.split], menu_size, answer_class, classes)
        if row.case_kind == "multiple_missing":
            _increment(multiple_counts[row.split], menu_size, answer_class, classes)

    return {
        "rows": len(rows),
        "groups": len(source_groups),
        "split_counts": split_counts,
        "case_kind_counts": case_counts,
        "menu_size_counts": menu_counts,
        "answer_class_by_menu_size": answer_counts,
        "multiple_missing_answer_class_by_menu_size": multiple_counts,
    }


def _verify_visible_prompt(row: ToolDatum) -> None:
    request = row.record.request
    verify_tool_request(request, row.scenario)
    parsed = parse_tool_prompt(render_prompt(request))
    if parsed.template_id != row.template_id:
        raise ValueError("rendered prompt template does not match the row template ID")


def _answer_class(answer: str) -> str:
    if answer == NO_ELIGIBLE_TOOL:
        return "no_eligible_tool"
    if answer == INSUFFICIENT_INFORMATION:
        return "insufficient_information"
    return "tool"


def _increment(
    counts: dict[int, dict[str, int]], menu_size: int, answer_class: str, classes: tuple[str, ...]
) -> None:
    by_class = counts.setdefault(menu_size, {name: 0 for name in classes})
    by_class[answer_class] += 1
