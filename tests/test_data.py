import json

import pytest

from reflex_decisions import data


def _partition_manifest(
    splits: tuple[tuple[str, data.SplitName], ...] = (
        ("train-set", "train"),
        ("development-set", "development"),
    ),
    *,
    group_partitioned_sources: tuple[str, ...] = ("shared-source",),
    held_out_families: tuple[str, ...] = (),
) -> data.SplitManifest:
    return data.SplitManifest(
        data_kind="benchmark",
        held_out_families=held_out_families,
        group_partitioned_sources=group_partitioned_sources,
        datasets=tuple(
            data.DatasetSpec(
                dataset_id=dataset_id,
                source_id="shared-source",
                task_family="family-a",
                split=split,
                source_uri=f"local://fixture/{split}",
                source_revision="fixture-v1",
                license="fixture license",
            )
            for dataset_id, split in splits
        ),
    )


def _partition_record(
    record_id: str,
    dataset_id: str,
    source_group_id: str,
    *,
    context: str = "Input",
    request: dict[str, object] | None = None,
) -> data.DecisionRecord:
    return data.DecisionRecord(
        record_id=record_id,
        dataset_id=dataset_id,
        source_group_id=source_group_id,
        request=request
        or {
            "context": context,
            "question": "Choose",
            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        },
        answer_id="a",
    )


def test_jsonl_records_load_as_immutable_decision_records(tmp_path) -> None:
    record = {
        "record_id": "case-1",
        "dataset_id": "fixture-train",
        "source_group_id": "conversation-1",
        "request": {
            "context": "A support question",
            "question": "Which category?",
            "options": [
                {"id": "billing", "label": "Billing"},
                {"id": "access", "label": "Access"},
            ],
        },
        "answer_id": "billing",
    }
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    loaded = data.load_records(path)

    assert len(loaded) == 1
    assert loaded[0].record_id == "case-1"
    assert loaded[0].answer_id in {option.id for option in loaded[0].request.options}


def test_jsonl_records_preserve_unicode_separators_inside_strings(tmp_path) -> None:
    contexts = ("before\u2028after", "before\u2029after", "before\u0085after")
    records = [
        {
            "record_id": f"case-{index}",
            "dataset_id": "fixture-train",
            "source_group_id": f"conversation-{index}",
            "request": {
                "context": context,
                "question": "Which category?",
                "options": [
                    {"id": "billing", "label": "Billing"},
                    {"id": "access", "label": "Access"},
                ],
            },
            "answer_id": "billing",
        }
        for index, context in enumerate(contexts)
    ]
    contents = (
        json.dumps(records[0], ensure_ascii=False)
        + "\n"
        + json.dumps(records[1], ensure_ascii=False)
        + "\r\n"
        + json.dumps(records[2], ensure_ascii=False)
        + "\n"
    )
    path = tmp_path / "records.jsonl"
    path.write_text(contents, encoding="utf-8")

    assert all(
        separator in path.read_text(encoding="utf-8")
        for separator in ("\u2028", "\u2029", "\u0085")
    )
    loaded = data.load_records(path)

    assert tuple(record.request.context for record in loaded) == contexts


def test_split_audit_rejects_exact_semantic_request_leakage() -> None:
    manifest = data.SplitManifest(
        data_kind="fixture",
        datasets=(
            data.DatasetSpec(
                dataset_id="train-set",
                source_id="source-train",
                task_family="family-a",
                split="train",
                source_uri="local://fixture/train",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
            data.DatasetSpec(
                dataset_id="test-set",
                source_id="source-test",
                task_family="family-b",
                split="test",
                source_uri="local://fixture/test",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )
    request = {
        "context": "A support request",
        "question": "Which category?",
        "options": [
            {"id": "billing", "label": "Billing"},
            {"id": "access", "label": "Access"},
        ],
    }
    records = (
        data.DecisionRecord(
            record_id="train-1",
            dataset_id="train-set",
            source_group_id="group-1",
            request=request,
            answer_id="billing",
        ),
        data.DecisionRecord(
            record_id="test-1",
            dataset_id="test-set",
            source_group_id="group-2",
            request=request,
            answer_id="billing",
        ),
    )

    try:
        data.audit_splits(manifest, records)
    except ValueError as error:
        assert "request" in str(error).lower()
    else:
        raise AssertionError("an exact semantic request duplicate must not cross splits")


def test_record_loader_rejects_duplicate_json_keys_without_echoing_payload(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"record_id":"safe","record_id":"private-value"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object key") as captured:
        data.load_records(path)

    assert "private-value" not in str(captured.value)
    assert f"{path}:1" in str(captured.value)


def test_split_audit_rejects_one_source_assigned_to_multiple_splits() -> None:
    manifest = data.SplitManifest(
        data_kind="fixture",
        datasets=(
            data.DatasetSpec(
                dataset_id="train-set",
                source_id="same-source",
                task_family="family-a",
                split="train",
                source_uri="local://fixture/train",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
            data.DatasetSpec(
                dataset_id="test-set",
                source_id="same-source",
                task_family="family-b",
                split="test",
                source_uri="local://fixture/test",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )

    with pytest.raises(ValueError, match="source_id.*crosses"):
        data.audit_splits(manifest, ())


def test_split_audit_allows_explicit_source_with_disjoint_groups() -> None:
    manifest = _partition_manifest()
    records = (
        _partition_record("train-1", "train-set", "train-group", context="First input"),
        _partition_record(
            "development-1", "development-set", "development-group", context="Different input"
        ),
    )

    summary = data.audit_splits(manifest, records)

    assert summary.record_count == 2
    assert dict(summary.split_counts)["train"] == 1
    assert dict(summary.split_counts)["development"] == 1
    serialized_manifest = manifest.model_dump(mode="json")
    assert serialized_manifest["group_partitioned_sources"] == ["shared-source"]
    assert data.SplitManifest.model_validate(serialized_manifest) == manifest


def test_split_policy_rejects_blank_or_duplicate_source_ids() -> None:
    dataset = data.DatasetSpec(
        dataset_id="train-set",
        source_id="shared-source",
        task_family="family-a",
        split="train",
        source_uri="local://fixture/train",
        source_revision="fixture-v1",
        license="fixture license",
    )

    with pytest.raises(ValueError, match="must not be blank"):
        data.SplitManifest(
            data_kind="benchmark", datasets=(dataset,), group_partitioned_sources=(" ",)
        )
    with pytest.raises(ValueError, match="group_partitioned_sources must be unique"):
        data.SplitManifest(
            data_kind="benchmark",
            datasets=(dataset,),
            group_partitioned_sources=("shared-source", "shared-source"),
        )


def test_split_policy_rejects_undeclared_source_ids() -> None:
    dataset = data.DatasetSpec(
        dataset_id="train-set",
        source_id="shared-source",
        task_family="family-a",
        split="train",
        source_uri="local://fixture/train",
        source_revision="fixture-v1",
        license="fixture license",
    )

    with pytest.raises(ValueError, match="must reference declared source IDs"):
        data.SplitManifest(
            data_kind="benchmark",
            datasets=(dataset,),
            group_partitioned_sources=("unknown-source",),
        )


def test_split_audit_rejects_shared_group_when_source_is_allowlisted() -> None:
    manifest = _partition_manifest()
    records = (
        _partition_record("train-1", "train-set", "same-group", context="First input"),
        _partition_record(
            "development-1", "development-set", "same-group", context="Different input"
        ),
    )

    with pytest.raises(ValueError, match="source group crosses"):
        data.audit_splits(manifest, records)


def test_split_audit_rejects_exact_request_when_source_is_allowlisted() -> None:
    manifest = _partition_manifest()
    request = {
        "context": "Same input",
        "question": "Choose",
        "options": [
            {"id": "a", "label": "A"},
            {"id": "b", "label": "B"},
        ],
    }
    records = (
        _partition_record("train-1", "train-set", "train-group", request=request),
        _partition_record("development-1", "development-set", "development-group", request=request),
    )

    with pytest.raises(ValueError, match="semantic request appears across"):
        data.audit_splits(manifest, records)


def test_split_audit_rejects_declared_held_out_family_in_development() -> None:
    manifest = _partition_manifest(
        splits=(("development-set", "development"), ("test-set", "test")),
        held_out_families=("family-a",),
    )

    with pytest.raises(ValueError, match="held-out task family"):
        data.audit_splits(manifest, ())
