import json

import pytest

from reflex_decisions import data


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


def test_split_audit_rejects_declared_held_out_family_in_development() -> None:
    manifest = data.SplitManifest(
        data_kind="fixture",
        held_out_families=("family-a",),
        datasets=(
            data.DatasetSpec(
                dataset_id="development-set",
                source_id="source-dev",
                task_family="family-a",
                split="development",
                source_uri="local://fixture/dev",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
            data.DatasetSpec(
                dataset_id="test-set",
                source_id="source-test",
                task_family="family-a",
                split="test",
                source_uri="local://fixture/test",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )

    with pytest.raises(ValueError, match="held-out task family"):
        data.audit_splits(manifest, ())
