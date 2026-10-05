from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pytest

from experiments import prepare_snli_diagnostic
from reflex_decisions import snli_diagnostic
from reflex_decisions.broader_data import (
    SNLI_LABELS,
    SnliItem,
    record_from_snli_item,
)
from reflex_decisions.data import audit_splits


def _item(pair_id: str, group_id: str, label: str) -> SnliItem:
    return SnliItem(
        source_item_id=pair_id,
        source_group_id=group_id,
        premise=f"premise {group_id}",
        hypothesis=f"hypothesis {pair_id}",
        label=label,
    )


def _ranked_fixture() -> tuple[SnliItem, ...]:
    items: list[SnliItem] = []
    for short_group in ("g-a", "g-b", "g-c", "g-d"):
        group_id = hashlib.sha256(short_group.encode()).hexdigest()
        suffix = short_group[-1]
        for label in SNLI_LABELS:
            for prefix in ("aaa", "zzz"):
                items.append(_item(f"{prefix}-{label}-{suffix}", group_id, label))
    return tuple(items)


def test_selection_uses_hash_ranks_and_chooses_a_complete_deterministic_panel() -> None:
    select = getattr(snli_diagnostic, "select_balanced_items", None)
    assert callable(select), "balanced group and pair selection behavior is not implemented"

    items = _ranked_fixture()
    selected = select(items, excluded_group_ids=frozenset(), group_count=2)
    reversed_selected = select(
        tuple(reversed(items)), excluded_group_ids=frozenset(), group_count=2
    )

    groups = {
        item.source_group_id: short
        for short in ("g-a", "g-b", "g-c", "g-d")
        for item in items
        if item.source_group_id == hashlib.sha256(short.encode()).hexdigest()
    }
    actual = [(groups[item.source_group_id], item.label, item.source_item_id) for item in selected]
    assert actual == [
        ("g-d", "entailment", "aaa-entailment-d"),
        ("g-d", "neutral", "zzz-neutral-d"),
        ("g-d", "contradiction", "zzz-contradiction-d"),
        ("g-b", "entailment", "zzz-entailment-b"),
        ("g-b", "neutral", "aaa-neutral-b"),
        ("g-b", "contradiction", "aaa-contradiction-b"),
    ]
    assert reversed_selected == selected
    assert Counter(item.label for item in selected) == Counter({label: 2 for label in SNLI_LABELS})


def test_exclusions_and_incomplete_groups_are_removed_before_group_ranking() -> None:
    select = getattr(snli_diagnostic, "select_balanced_items", None)
    assert callable(select), "balanced group and pair selection behavior is not implemented"

    items = list(_ranked_fixture())
    incomplete_group = hashlib.sha256(b"incomplete").hexdigest()
    items.append(_item("incomplete-only", incomplete_group, "entailment"))
    excluded_group = hashlib.sha256(b"g-d").hexdigest()

    selected = select(items, excluded_group_ids=frozenset({excluded_group}), group_count=2)

    selected_groups = tuple(dict.fromkeys(item.source_group_id for item in selected))
    assert selected_groups == (
        hashlib.sha256(b"g-b").hexdigest(),
        hashlib.sha256(b"g-a").hexdigest(),
    )
    assert all(
        Counter(item.label for item in selected if item.source_group_id == group_id)
        == Counter({label: 1 for label in SNLI_LABELS})
        for group_id in selected_groups
    )


def test_selection_rejects_a_panel_larger_than_eligible_groups() -> None:
    select = getattr(snli_diagnostic, "select_balanced_items", None)
    assert callable(select), "balanced group and pair selection behavior is not implemented"

    with pytest.raises(ValueError, match="eligible SNLI groups"):
        select(_ranked_fixture(), excluded_group_ids=frozenset(), group_count=5)


def test_prior_recipe_bytes_must_match_the_pinned_hashes(tmp_path) -> None:
    load_exclusions = getattr(snli_diagnostic, "load_prior_exclusions", None)
    assert callable(load_exclusions), "pinned prior-panel exclusion validation is not implemented"

    recipe_path = tmp_path / "data/baselines/broader-dev-recipe.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="recipe SHA-256 mismatch"):
        load_exclusions(tmp_path)


@pytest.mark.parametrize(
    ("variant", "message"),
    (
        ("wrong_count", "group count mismatch"),
        ("invalid_hash", "invalid source group hash"),
        ("duplicate", "duplicate source groups"),
        ("overlap", "not source-group disjoint"),
    ),
)
def test_prior_recipe_selection_must_have_valid_disjoint_group_hashes(
    tmp_path, monkeypatch, variant: str, message: str
) -> None:
    load_exclusions = getattr(snli_diagnostic, "load_prior_exclusions", None)
    assert callable(load_exclusions), "pinned prior-panel exclusion validation is not implemented"

    group_ids = [hashlib.sha256(f"group-{index}".encode()).hexdigest() for index in range(160)]
    first = group_ids[:32]
    second = group_ids[32:]
    if variant == "wrong_count":
        first.pop()
    elif variant == "invalid_hash":
        first[-1] = "invalid"
    elif variant == "duplicate":
        first[-1] = first[0]
    elif variant == "overlap":
        second[-1] = first[0]

    specs = [
        snli_diagnostic.PriorRecipeSpec("first.json", "", "snli", "source_group_ids", 32),
        snli_diagnostic.PriorRecipeSpec("second.json", "", "pilot", "group_ids", 128),
    ]
    documents = (
        {"selected": {"snli": {"source_group_ids": first}}},
        {"selected": {"pilot": {"group_ids": second}}},
    )
    for index, document in enumerate(documents):
        path = tmp_path / specs[index].path
        path.write_text(json.dumps(document), encoding="utf-8")
        specs[index] = snli_diagnostic.PriorRecipeSpec(
            specs[index].path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            specs[index].selection_key,
            specs[index].group_key,
            specs[index].group_count,
        )
    monkeypatch.setattr(snli_diagnostic, "PRIOR_RECIPE_SPECS", tuple(specs))

    with pytest.raises(ValueError, match=message):
        load_exclusions(tmp_path)


def test_records_preserve_snli_request_semantics_and_use_balanced_dataset_identity() -> None:
    build = getattr(snli_diagnostic, "build_balanced_dataset", None)
    assert callable(build), "balanced SNLI record and manifest construction is not implemented"

    bundle = build(_ranked_fixture(), excluded_group_ids=frozenset(), group_count=2)

    assert len(bundle.records) == 6
    assert {record.dataset_id for record in bundle.records} == {"snli-balanced-v1-development"}
    assert Counter(record.answer_id for record in bundle.records) == Counter(
        {label: 2 for label in SNLI_LABELS}
    )
    assert {record.source_group_id for record in bundle.records} == {
        item.source_group_id for item in bundle.items
    }
    for item, record in zip(bundle.items, bundle.records, strict=True):
        original = record_from_snli_item(item)
        assert record.request == original.request
        assert record.answer_id == item.label
        assert record.record_id.startswith("snli-balanced-v1-development-")
        assert record.record_id != original.record_id
    assert bundle.manifest.data_kind == "benchmark"
    assert len(bundle.manifest.datasets) == 1
    assert bundle.manifest.datasets[0].split == "development"
    assert bundle.manifest.held_out_families == ()
    assert audit_splits(bundle.manifest, bundle.records).record_count == 6


def test_conflicting_normalized_source_labels_still_raise_through_the_pipeline() -> None:
    parse_and_build = getattr(snli_diagnostic, "parse_and_build_balanced_dataset", None)
    assert callable(parse_and_build), "the balanced pipeline must reuse the pinned SNLI parser"

    rows = (
        {
            "pairID": "pair-one",
            "sentence1": "  A premise",
            "sentence2": "A hypothesis",
            "captionID": "caption-one",
            "gold_label": "entailment",
        },
        {
            "pairID": "pair-two",
            "sentence1": "a premise ",
            "sentence2": "a hypothesis",
            "captionID": "caption-one",
            "gold_label": "neutral",
        },
    )
    payload = "\n".join(json.dumps(row) for row in rows).encode()

    with pytest.raises(ValueError, match="conflicting labels"):
        parse_and_build(payload, excluded_group_ids=frozenset(), group_count=1)


def test_candidate_pipeline_builds_a_source_only_recipe_from_validated_inputs(monkeypatch) -> None:
    build_candidate = getattr(snli_diagnostic, "build_candidate", None)
    assert callable(build_candidate), (
        "pinned balanced SNLI candidate preparation is not implemented"
    )

    items = _ranked_fixture()
    excluded = frozenset({hashlib.sha256(b"g-d").hexdigest()})
    pair_rows = {item.source_item_id: row for row, item in enumerate(items)}
    receipts = (snli_diagnostic.PriorRecipeReceipt("pinned-prior.json", "a" * 64, tuple(excluded)),)
    monkeypatch.setattr(
        snli_diagnostic,
        "read_verified_snli_development",
        lambda _root: (b"dev-only", "b" * 64, "c" * 64),
    )
    monkeypatch.setattr(
        snli_diagnostic,
        "parse_snli_items",
        lambda payload: items if payload == b"dev-only" else pytest.fail("unexpected source bytes"),
    )
    monkeypatch.setattr(
        snli_diagnostic,
        "load_prior_exclusions",
        lambda _root: (receipts, excluded),
    )
    monkeypatch.setattr(
        snli_diagnostic,
        "parse_snli_native_row_ids",
        lambda _payload, expected_rows: (pair_rows, 0, expected_rows),
    )
    monkeypatch.setattr(snli_diagnostic, "EXPECTED_SOURCE_GROUPS", 4)
    monkeypatch.setattr(snli_diagnostic, "EXPECTED_ELIGIBLE_GROUPS", 3)
    monkeypatch.setattr(snli_diagnostic, "SELECTED_GROUP_COUNT", 2)

    candidate = build_candidate("unused-fixture-root")
    recipe = json.loads(candidate.recipe_bytes)
    records = candidate.bundle.records

    assert len(records) == 6
    assert Counter(record.answer_id for record in records) == Counter(
        {label: 2 for label in SNLI_LABELS}
    )
    assert len({record.source_group_id for record in records}) == 2
    assert recipe["selection"]["source_group_count"] == 4
    assert recipe["selection"]["excluded_group_count"] == 1
    assert recipe["selection"]["eligible_group_count"] == 3
    assert recipe["selection"]["selected_group_count"] == 2
    selected_groups = {record.source_group_id for record in records}
    excluded_groups = set(recipe["selection"]["excluded_group_ids"])
    assert selected_groups.isdisjoint(excluded_groups)
    assert not any(term in candidate.recipe_bytes.lower() for term in (b"premise", b"hypothesis"))
    assert all(
        member["source_row_id"] >= 0
        for group in recipe["selection"]["selected_groups"]
        for member in group["members"]
    )


def test_source_reader_verifies_the_archive_but_reads_only_the_dev_member(
    tmp_path, monkeypatch
) -> None:
    read_development = getattr(snli_diagnostic, "read_verified_snli_development", None)
    assert callable(read_development), (
        "pinned SNLI development source verification is not implemented"
    )

    dev_bytes = b'{"gold_label":"entailment"}\n'
    archive_buffer = io.BytesIO()
    with ZipFile(archive_buffer, "w") as archive:
        archive.writestr("snli_1.0/snli_1.0_dev.jsonl", dev_bytes)
        archive.writestr("snli_1.0/snli_1.0_train.jsonl", b"not JSON and must not be read")
        archive.writestr("snli_1.0/snli_1.0_test.jsonl", b"not JSON and must not be read")
    archive_bytes = archive_buffer.getvalue()
    root = tmp_path / "data/raw"
    root.mkdir(parents=True)
    (root / "snli_1.0.zip").write_bytes(archive_bytes)
    (root / "snli_1.0_dev.jsonl").write_bytes(dev_bytes)
    monkeypatch.setattr(
        snli_diagnostic,
        "SNLI_SOURCE_ARCHIVE_SHA256",
        hashlib.sha256(archive_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        snli_diagnostic,
        "SNLI_DEV_SHA256",
        hashlib.sha256(dev_bytes).hexdigest(),
    )
    archive_reads: list[str] = []
    original_read = ZipFile.read

    def track_archive_read(self, name, *args, **kwargs):
        archive_reads.append(name)
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "read", track_archive_read)

    actual = read_development(tmp_path)

    assert actual == (
        dev_bytes,
        hashlib.sha256(archive_bytes).hexdigest(),
        hashlib.sha256(dev_bytes).hexdigest(),
    )
    assert archive_reads == [snli_diagnostic.SNLI_DEV_MEMBER]


def test_cli_is_plan_only_and_write_refuses_existing_outputs(tmp_path, monkeypatch, capsys) -> None:
    main = getattr(prepare_snli_diagnostic, "main", None)
    assert callable(main), "the plan-only SNLI preparation CLI is not implemented"

    bundle = snli_diagnostic.build_balanced_dataset(
        _ranked_fixture(), excluded_group_ids=frozenset(), group_count=2
    )
    candidate = snli_diagnostic.CandidateArtifacts(
        bundle,
        b"records\n",
        b"manifest\n",
        b"recipe\n",
    )
    calls: list[str] = []

    def fake_build(root: str | Path) -> snli_diagnostic.CandidateArtifacts:
        calls.append(str(root))
        return candidate

    monkeypatch.setattr(prepare_snli_diagnostic, "build_candidate", fake_build)
    repo_root = str(Path(__file__).resolve().parents[1])
    plan_dir = tmp_path / "plan-only"
    assert main(["--repo-root", repo_root, "--output-dir", str(plan_dir)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "plan_only"
    assert not plan_dir.exists()

    write_dir = tmp_path / "fresh-output"
    assert main(["--repo-root", repo_root, "--output-dir", str(write_dir), "--write"]) == 0
    assert (write_dir / "records.jsonl").read_bytes() == b"records\n"
    assert (write_dir / "manifest.json").read_bytes() == b"manifest\n"
    assert (write_dir / "recipe.json").read_bytes() == b"recipe\n"

    existing = tmp_path / "existing-output"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    call_count = len(calls)
    with pytest.raises(FileExistsError, match="must not already exist"):
        main(["--repo-root", repo_root, "--output-dir", str(existing), "--write"])
    assert len(calls) == call_count
    assert sentinel.read_text(encoding="utf-8") == "preserve"
