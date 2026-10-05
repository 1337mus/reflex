"""End-to-end integrity checks for tool-choice rows and manifests."""

from dataclasses import replace

import pytest

from reflex_decisions.data import DatasetSpec, DecisionRecord, SplitManifest
from reflex_decisions.tool_data_audit import audit_tool_data, classify_scenario
from reflex_decisions.tool_data_identity import (
    catalog_hash,
    split_for_structure,
    structural_catalog_hash,
)
from reflex_decisions.tool_data_spec import (
    CASE_KINDS,
    FAMILY,
    SOURCE_ID,
    SPLITS,
    TEMPLATE_IDS,
    VERSION,
    VOCABULARIES,
    ToolDatum,
)
from reflex_decisions.tool_rules import (
    ToolScenario,
    ToolSpec,
    answer_ids,
    solve,
)
from reflex_decisions.tool_text import render_tool_request


def _scenario(
    tools: tuple[ToolSpec, ...],
    *,
    permissions: tuple[str, ...] = ("p1", "p2"),
    permission_values: tuple[bool | None, ...] = (False, False),
    required_capabilities: tuple[str, ...] = ("read",),
    input_type: str = "text",
) -> ToolScenario:
    return ToolScenario(
        ("read", "write"),
        ("text", "image"),
        permissions,
        tools,
        required_capabilities,
        input_type,
        permission_values,
    )


def test_classify_scenario_distinguishes_all_seven_outcome_shapes() -> None:
    stable = ToolSpec("stable", ("read",), ("text",), (), 4)
    private = ToolSpec("private", ("read",), ("text",), ("p1",), 1)
    p2_a = ToolSpec("p2-a", ("read",), ("text",), ("p2",), 1)
    p2_b = ToolSpec("p2-b", ("read",), ("text",), ("p2",), 2)
    wrong = ToolSpec("wrong", ("write",), ("text",), (), 0)
    image = ToolSpec("image", ("read",), ("image",), (), 0)
    scenarios = (
        (_scenario((private, stable), permission_values=(True, False)), "complete_tool"),
        (_scenario((wrong, image)), "complete_no_eligible"),
        (
            _scenario(
                (replace(stable, cost=1), replace(private, cost=5)),
                permission_values=(None, False),
            ),
            "missing_tool_agree",
        ),
        (_scenario((p2_a, p2_b), permission_values=(None, False)), "missing_none_agree"),
        (
            _scenario((stable, private), permission_values=(None, False)),
            "missing_tool_conflict",
        ),
        (
            _scenario((private, wrong), permission_values=(None, False)),
            "missing_tool_vs_none",
        ),
        (
            _scenario((replace(stable, cost=0), private), permission_values=(None, None)),
            "multiple_missing",
        ),
    )

    assert tuple(classify_scenario(scenario) for scenario, _ in scenarios) == tuple(
        kind for _, kind in scenarios
    )


def _scenario_for_split(target_split: str, *, vocabulary_split: str | None = None) -> ToolScenario:
    vocabulary = VOCABULARIES[vocabulary_split or target_split]  # type: ignore[index]
    capability_count = len(vocabulary.capabilities)
    input_count = len(vocabulary.input_types)
    for attempt in range(8192):
        tool_count = (2, 4, 6)[attempt % 3]
        tools = tuple(
            ToolSpec(
                id=vocabulary.tool_ids[index],
                capabilities=tuple(
                    value
                    for bit, value in enumerate(vocabulary.capabilities)
                    if (1 + (attempt * (index + 1) + index * 3) % ((1 << capability_count) - 1))
                    & (1 << bit)
                ),
                input_types=tuple(
                    value
                    for bit, value in enumerate(vocabulary.input_types)
                    if (1 + (attempt * (index + 2) + index * 5) % ((1 << input_count) - 1))
                    & (1 << bit)
                ),
                required_permissions=tuple(
                    value
                    for bit, value in enumerate(vocabulary.permissions)
                    if (attempt + index * 3) % 4 & (1 << bit)
                ),
                cost=index * 7 + attempt % 7,
            )
            for index in range(tool_count)
        )
        scenario = ToolScenario(
            vocabulary.capabilities,
            vocabulary.input_types,
            vocabulary.permissions,
            tools,
            (vocabulary.capabilities[attempt % capability_count],),
            vocabulary.input_types[attempt % input_count],
            (False, True),
        )
        if split_for_structure(structural_catalog_hash(scenario)) == target_split:
            return scenario
    raise AssertionError(f"test fixture search did not find a {target_split} catalog")


def _manifest() -> SplitManifest:
    return SplitManifest(
        data_kind="fixture",
        datasets=tuple(
            DatasetSpec(
                dataset_id=f"{VERSION}-{split}",
                source_id=SOURCE_ID,
                task_family=FAMILY,
                split=split,
                source_uri="synthetic://reflex/tool-choice/v1",
                source_revision=VERSION,
                license="self-authored synthetic",
            )
            for split in SPLITS
        ),
        group_partitioned_sources=(SOURCE_ID,),
    )


def _row_for_scenario(
    scenario: ToolScenario, split: str, *, record_id: str = "toy-record"
) -> ToolDatum:
    structure = structural_catalog_hash(scenario)
    request = render_tool_request(
        scenario,
        option_order=tuple(reversed(answer_ids(scenario))),
        template_id=TEMPLATE_IDS[split],  # type: ignore[index]
    )
    record = DecisionRecord(
        record_id=record_id,
        dataset_id=f"{VERSION}-{split}",
        source_group_id="tool-group-" + structure,
        request=request,
        answer_id=solve(scenario),
    )
    return ToolDatum(
        split=split,  # type: ignore[arg-type]
        scenario=scenario,
        record=record,
        template_id=TEMPLATE_IDS[split],  # type: ignore[index]
        case_kind=classify_scenario(scenario),
    )


def _row(split: str = "train", *, record_id: str = "toy-record") -> ToolDatum:
    return _row_for_scenario(_scenario_for_split(split), split, record_id=record_id)


def test_audit_accepts_a_valid_toy_row_and_returns_aggregate_only() -> None:
    row = _row()

    summary = audit_tool_data((row,), _manifest())

    assert summary["rows"] == 1
    assert summary["groups"] == 1
    assert summary["split_counts"] == {split: int(split == row.split) for split in SPLITS}
    assert sum(sum(counts.values()) for counts in summary["case_kind_counts"].values()) == 1
    assert summary["menu_size_counts"]["train"][len(row.scenario.tools) + 2] == 1


def test_audit_counts_answer_classes_and_multiple_missing_by_menu_size() -> None:
    scenario = replace(_scenario_for_split("train"), permission_values=(None, None))
    row = _row_for_scenario(scenario, "train", record_id="multiple-missing")

    summary = audit_tool_data((row,), _manifest())
    menu_size = len(row.scenario.tools) + 2

    assert sum(summary["answer_class_by_menu_size"]["train"][menu_size].values()) == 1
    assert (
        sum(summary["multiple_missing_answer_class_by_menu_size"]["train"][menu_size].values()) == 1
    )


def test_audit_rejects_unapproved_or_incomplete_manifest_metadata() -> None:
    row = _row()
    manifest = _manifest()
    datasets = list(manifest.datasets)
    datasets[0] = datasets[0].model_copy(update={"task_family": "other-family"})
    wrong_family = manifest.model_copy(update={"datasets": tuple(datasets)})

    with pytest.raises(ValueError, match="unapproved source or task family"):
        audit_tool_data((row,), wrong_family)
    with pytest.raises(ValueError, match="exactly one dataset per split"):
        audit_tool_data((row,), manifest.model_copy(update={"datasets": manifest.datasets[:-1]}))


def test_audit_accepts_reordered_domains_with_permission_pairing_preserved() -> None:
    row = _row()
    scenario = row.scenario
    reordered = replace(
        scenario,
        capabilities=tuple(reversed(scenario.capabilities)),
        input_types=tuple(reversed(scenario.input_types)),
        permissions=tuple(reversed(scenario.permissions)),
        permission_values=tuple(reversed(scenario.permission_values)),
    )

    summary = audit_tool_data((_row_for_scenario(reordered, "train"),), _manifest())

    assert summary["rows"] == 1


def test_audit_rejects_wrong_group_case_label_template_and_split() -> None:
    row = _row()
    mutations = (
        (
            replace(row, record=row.record.model_copy(update={"source_group_id": "wrong"})),
            "source group ID",
        ),
        (
            replace(row, case_kind=next(kind for kind in CASE_KINDS if kind != row.case_kind)),
            "case kind",
        ),
        (replace(row, template_id="tool-development-v1"), "template ID"),
        (replace(row, split="development"), "row split"),  # type: ignore[arg-type]
    )
    for changed, message in mutations:
        with pytest.raises(ValueError, match=message):
            audit_tool_data((changed,), _manifest())


def test_audit_rejects_an_altered_answer_id() -> None:
    row = _row()
    wrong_answer = next(
        answer for answer in answer_ids(row.scenario) if answer != row.record.answer_id
    )
    changed = replace(
        row,
        record=row.record.model_copy(update={"answer_id": wrong_answer}),
    )

    with pytest.raises(ValueError, match="record answer does not match the tool solver"):
        audit_tool_data((changed,), _manifest())


def test_audit_rejects_changed_prompt_facts_and_option_labels() -> None:
    row = _row()
    changed_scenario = replace(
        row.scenario,
        required_capabilities=(
            next(
                cap
                for cap in row.scenario.capabilities
                if cap != row.scenario.required_capabilities[0]
            ),
        ),
    )
    changed_prompt = render_tool_request(changed_scenario, template_id=row.template_id)
    changed_facts = row.record.request.model_copy(update={"context": changed_prompt.context})
    changed_label_options = list(row.record.request.options)
    changed_label_options[0] = changed_label_options[0].model_copy(update={"label": "Use another"})
    changed_label = row.record.request.model_copy(update={"options": tuple(changed_label_options)})
    for request, message in (
        (changed_facts, "rendered tool facts do not match"),
        (changed_label, "unknown or duplicate choice"),
    ):
        changed = replace(row, record=row.record.model_copy(update={"request": request}))
        with pytest.raises(ValueError, match=message):
            audit_tool_data((changed,), _manifest())


def test_audit_rejects_foreign_vocabulary_and_wrong_structural_split() -> None:
    row = _row()
    foreign = _row("development")
    wrong_vocabulary = replace(row, scenario=foreign.scenario)
    with pytest.raises(ValueError, match="protected split vocabulary"):
        audit_tool_data((wrong_vocabulary,), _manifest())

    train_structure_with_development_names = _scenario_for_split(
        "train", vocabulary_split="development"
    )
    wrong_split = _row_for_scenario(
        train_structure_with_development_names, "development", record_id="wrong-split"
    )
    with pytest.raises(ValueError, match="assigned to a different split"):
        audit_tool_data((wrong_split,), _manifest())


def test_audit_rejects_duplicate_named_and_structural_catalogs() -> None:
    row = _row()
    changed_query = replace(
        row.scenario,
        required_capabilities=(
            next(
                cap
                for cap in row.scenario.capabilities
                if cap != row.scenario.required_capabilities[0]
            ),
        ),
    )
    sibling = _row_for_scenario(changed_query, "train", record_id="sibling")
    with pytest.raises(ValueError, match="duplicate named tool catalog"):
        audit_tool_data((row, sibling), _manifest())

    rotated_ids = tuple(
        replace(tool, id=VOCABULARIES["train"].tool_ids[(index + 1) % 6])
        for index, tool in enumerate(row.scenario.tools)
    )
    renamed = _row_for_scenario(
        replace(row.scenario, tools=rotated_ids), "train", record_id="renamed"
    )
    assert catalog_hash(row.scenario) != catalog_hash(renamed.scenario)
    assert structural_catalog_hash(row.scenario) == structural_catalog_hash(renamed.scenario)
    with pytest.raises(ValueError, match="duplicate structural tool catalog"):
        audit_tool_data((row, renamed), _manifest())
