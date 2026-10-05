"""Tests for exact-rule synthetic data generation."""

from __future__ import annotations

import importlib
import json
import re
from collections import Counter, defaultdict

import pytest

from reflex_decisions.data import audit_splits


def test_atomic_fact_solver_marks_matching_exact_fact_entailed() -> None:
    try:
        solve_atomic_fact = importlib.import_module(
            "reflex_decisions.synthetic_data"
        ).solve_atomic_fact
    except (ModuleNotFoundError, AttributeError):
        pytest.fail("the atomic fact solver is not implemented")

    assert solve_atomic_fact({"elm": "blue"}, "elm", "blue") == "entailed"


def test_numeric_solver_selects_the_smallest_value_including_negative_and_zero() -> None:
    try:
        solve_numeric_selection = importlib.import_module(
            "reflex_decisions.synthetic_data"
        ).solve_numeric_selection
    except (ModuleNotFoundError, AttributeError):
        pytest.fail("the numeric selection solver is not implemented")

    assert solve_numeric_selection({"item-a": 0, "item-b": -2, "item-c": 4}, "smallest") == "item-b"


def test_default_candidate_is_reproducible_and_keeps_scenario_groups_in_one_split() -> None:
    try:
        module = importlib.import_module("reflex_decisions.synthetic_data")
        build_candidate = module.build_candidate
    except (ModuleNotFoundError, AttributeError):
        pytest.fail("the synthetic candidate generator is not implemented")

    first = build_candidate()
    second = build_candidate()

    assert first.manifest.data_kind == "fixture"
    assert first.manifest.held_out_families == ()
    assert len(first.records) == 750
    assert first.records == second.records
    audited = audit_splits(first.manifest, first.records)
    assert dict(audited.split_counts) == {
        "train": 500,
        "development": 125,
        "calibration": 125,
        "test": 0,
    }
    group_sizes = Counter(record.source_group_id for record in first.records)
    assert Counter(group_sizes.values()) == {2: 150, 3: 150}
    numeric_values = first.audit["numeric_candidate_values"]
    assert all(numeric_values[f"{sign}_count"] > 0 for sign in ("negative", "zero", "positive"))
    records_by_id = {record.record_id: record for record in first.records}
    fact_traces = [trace for trace in first.provenance if trace["family"] == module.FACT_FAMILY]
    assert sum(len(trace["derived_examples"]) for trace in fact_traces) == 450
    for trace in fact_traces:
        for example in trace["derived_examples"]:
            record = records_by_id[example["record_id"]]
            claim = re.search(r"“(\S+) has color (\S+).”$", record.request.question)
            assert claim is not None
            assert (example["subject"], example["color"]) == claim.groups()
            assert {"subject": example["subject"], "color": example["color"]} in trace[
                "structured_problem"
            ]["claims"]


def test_count_expansion_preserves_existing_scenarios_and_numeric_option_counts() -> None:
    module = importlib.import_module("reflex_decisions.synthetic_data")
    config = module.GenerationConfig(fact_scenarios=(2, 2, 2), numeric_scenarios=(2, 2, 2))
    expanded_config = module.GenerationConfig(fact_scenarios=(4, 2, 2), numeric_scenarios=(4, 2, 2))
    original, expanded = module.build_candidate(config), module.build_candidate(expanded_config)
    expanded_traces = {trace["scenario_id"]: trace for trace in expanded.provenance}
    expanded_records = {record.record_id: record for record in expanded.records}

    for trace in original.provenance:
        assert expanded_traces[trace["scenario_id"]] == trace
    for record in original.records:
        assert expanded_records[record.record_id] == record


def test_fact_requests_define_unlisted_object_colors_as_unknown() -> None:
    module = importlib.import_module("reflex_decisions.synthetic_data")
    candidate = module.build_candidate(
        module.GenerationConfig(fact_scenarios=(1, 1, 1), numeric_scenarios=(1, 1, 1))
    )
    fact_records = [record for record in candidate.records if len(record.request.options) == 3]

    assert len(fact_records) == 9
    assert all(
        "unlisted object's color is unknown" in record.request.context for record in fact_records
    )


def test_synthetic_cli_is_plan_only_by_default(tmp_path, capsys) -> None:
    try:
        main = importlib.import_module("experiments.prepare_synthetic_data").main
    except (ModuleNotFoundError, AttributeError):
        pytest.fail("the synthetic preparation CLI is not implemented")

    output_dir = tmp_path / "candidate"
    assert main(["--output-dir", str(output_dir)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["mode"] == "plan-only"
    assert output["audit"]["record_counts_by_split"] == {
        "train": 500,
        "development": 125,
        "calibration": 125,
    }
    assert not output_dir.exists()


def test_rendered_templates_independently_reproduce_solver_answers_and_balance_options() -> None:
    module = importlib.import_module("reflex_decisions.synthetic_data")
    candidate = module.build_candidate(
        module.GenerationConfig(fact_scenarios=(8, 8, 8), numeric_scenarios=(8, 8, 8))
    )
    records_by_id = {record.record_id: record for record in candidate.records}
    fact_relations: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_directions: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_counts_by_template: Counter[str] = Counter()
    contradicted_colors: set[str] = set()
    option_counts_by_split: dict[str, set[int]] = defaultdict(set)
    option_counts_by_template: dict[str, set[int]] = defaultdict(set)
    template_signatures: dict[str, set[tuple[str, str]]] = defaultdict(set)
    group_order: dict[str, tuple[str, ...]] = {}
    group_split: dict[str, str] = {}

    for trace in candidate.provenance:
        family = trace["family"]
        split = trace["split"]
        template_id = trace["template_id"]
        bank = (
            module.FACT_TEMPLATE_BANKS[split]
            if family == module.FACT_FAMILY
            else module.NUMERIC_TEMPLATE_BANKS[split]
        )
        assert template_id in bank
        scenario_counts_by_template[template_id] += 1
        for example in trace["derived_examples"]:
            record = records_by_id[example["record_id"]]
            prompt = record.request.question
            if family == module.FACT_FAMILY:
                question_template = prompt.split(" “", maxsplit=1)[0]
            else:
                question_template = re.sub(r"\b(smallest|largest)\b", "{direction}", prompt)
            template_signatures[template_id].add(
                (record.request.context.splitlines()[0], question_template)
            )
            previous_split = group_split.setdefault(record.source_group_id, split)
            assert previous_split == split
            order = tuple(option.id for option in record.request.options)
            previous_order = group_order.setdefault(record.source_group_id, order)
            assert previous_order == order

            if family == module.FACT_FAMILY:
                facts = dict(
                    re.findall(r"^- (\S+) has exactly color (\S+)\.$", record.request.context, re.M)
                )
                claim = re.search(r"“(\S+) has color (\S+)\.”$", record.request.question)
                assert len(facts) == 3
                assert claim is not None
                subject, color = claim.groups()
                fact_color = facts.get(subject)
                expected = (
                    "unknown"
                    if fact_color is None
                    else ("entailed" if fact_color == color else "contradicted")
                )
                answer = next(
                    option for option in record.request.options if option.id == record.answer_id
                )
                assert answer.label == expected
                if expected == "contradicted":
                    contradicted_colors.add(color)
                fact_relations[template_id][expected] += 1
            else:
                measurements = {
                    item: int(value)
                    for item, value in re.findall(
                        r"^- (\S+) has value (-?\d+)\.$", record.request.context, re.M
                    )
                }
                direction_match = re.search(r"\b(smallest|largest)\b", record.request.question)
                assert len(measurements) == len(record.request.options)
                assert len(set(measurements.values())) == len(measurements)
                option_counts_by_template[template_id].add(len(measurements))
                assert direction_match is not None
                direction = direction_match.group(1)
                expected = (
                    min(measurements, key=measurements.__getitem__)
                    if direction == "smallest"
                    else max(measurements, key=measurements.__getitem__)
                )
                answer = next(
                    option for option in record.request.options if option.id == record.answer_id
                )
                assert answer.label == expected
                numeric_directions[template_id][direction] += 1
                option_counts_by_split[split].add(len(measurements))

    for template_id, counts in fact_relations.items():
        assert counts == Counter(
            {
                relation: scenario_counts_by_template[template_id]
                for relation in module.FACT_RELATIONS
            }
        )
    for template_id, counts in numeric_directions.items():
        assert counts == Counter(
            {
                direction: scenario_counts_by_template[template_id]
                for direction in ("smallest", "largest")
            }
        )
    assert all(counts == {2, 4, 8, 16} for counts in option_counts_by_split.values())
    assert all(counts == {2, 4, 8, 16} for counts in option_counts_by_template.values())
    assert set(group_split.values()) == set(module.SPLITS)
    assert len(contradicted_colors) > 1
    expected_templates = {
        template
        for split in module.SPLITS
        for template in (*module.FACT_TEMPLATE_BANKS[split], *module.NUMERIC_TEMPLATE_BANKS[split])
    }
    assert set(template_signatures) == expected_templates
    assert all(len(signatures) == 1 for signatures in template_signatures.values())
    rendered = {
        template: next(iter(signatures)) for template, signatures in template_signatures.items()
    }
    assert len({context for context, _ in rendered.values()}) == len(expected_templates)
    assert len({question for _, question in rendered.values()}) == len(expected_templates)
    assert len({next(iter(values)) for values in template_signatures.values()}) == len(
        template_signatures
    )


def test_canonical_scenario_hashes_reject_reuse_across_splits() -> None:
    module = importlib.import_module("reflex_decisions.synthetic_data")
    repeated_problem = "same-canonical-facts"

    with pytest.raises(ValueError, match="appears across data splits"):
        module.validate_scenario_hash_partition(
            (
                {"canonical_problem_sha256": repeated_problem, "split": "train"},
                {"canonical_problem_sha256": repeated_problem, "split": "development"},
            )
        )


def test_candidate_writer_hashes_outputs_and_refuses_existing_directory(tmp_path) -> None:
    module = importlib.import_module("reflex_decisions.synthetic_data")
    candidate = module.build_candidate(
        module.GenerationConfig(fact_scenarios=(1, 1, 1), numeric_scenarios=(1, 1, 1))
    )
    output_dir = tmp_path / "candidate"

    report = module.write_candidate(candidate, output_dir)

    assert set(report["file_sha256"]) == {
        "records.jsonl",
        "manifest.json",
        "recipe.json",
        "provenance.jsonl",
        "audit.json",
    }
    for name, digest in report["file_sha256"].items():
        assert module.hashlib.sha256((output_dir / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="choose a fresh directory"):
        module.write_candidate(candidate, output_dir)
