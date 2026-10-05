import pytest

from experiments import broader_runner_core
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _record() -> DecisionRecord:
    return DecisionRecord(
        record_id="boolq-dev-pilot-v1-0",
        dataset_id="boolq-dev-pilot-v1",
        source_group_id="source-0",
        request=DecisionRequest(
            context="The sky is blue.",
            question="Is the sky blue?",
            options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
        ),
        answer_id="yes",
    )


def test_remote_serialization_omits_gold_labels() -> None:
    payload = broader_runner_core.serialize_remote_records((_record(),))

    assert payload == [
        {
            "record_id": "boolq-dev-pilot-v1-0",
            "request": _record().request.model_dump(mode="json"),
        }
    ]
    assert "answer_id" not in payload[0]
    assert "source_group_id" not in payload[0]


def test_permutations_include_all_six_orders_with_original_first() -> None:
    record = DecisionRecord(
        record_id="snli-dev-pilot-v1-0",
        dataset_id="snli-dev-pilot-v1",
        source_group_id="source-0",
        request=DecisionRequest(
            context="A person is outside.",
            question="Does the hypothesis follow?",
            options=(
                Option(id="entailment", label="Entailment"),
                Option(id="neutral", label="Neutral"),
                Option(id="contradiction", label="Contradiction"),
            ),
        ),
        answer_id="neutral",
    )
    payload = broader_runner_core.serialize_remote_records((record,))

    rows = broader_runner_core.presentations(payload)

    assert [row["option_ids"] for row in rows] == [
        ["entailment", "neutral", "contradiction"],
        ["entailment", "contradiction", "neutral"],
        ["neutral", "entailment", "contradiction"],
        ["neutral", "contradiction", "entailment"],
        ["contradiction", "entailment", "neutral"],
        ["contradiction", "neutral", "entailment"],
    ]
    assert [row["permutation_index"] for row in rows] == list(range(6))


def test_presentation_result_rejects_nonfinite_logits() -> None:
    record = _record()
    presented = broader_runner_core.presentations(
        broader_runner_core.serialize_remote_records((record,))
    )[0]
    scored = {
        "raw_logits": [0.0, float("nan")],
        "input_tokens": 4,
        "prompt_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="non-finite"):
        broader_runner_core.validate_presentation_result(presented, scored)


def _pilot_records() -> tuple[DecisionRecord, ...]:
    records = [
        DecisionRecord(
            record_id=f"boolq-dev-pilot-v1-{index}",
            dataset_id="boolq-dev-pilot-v1",
            source_group_id=f"boolq-group-{index}",
            request=DecisionRequest(
                context=f"Passage {index}.",
                question=f"Question {index}?",
                options=(Option(id="yes", label="Yes"), Option(id="no", label="No")),
            ),
            answer_id="yes",
        )
        for index in range(32)
    ]
    records.extend(
        DecisionRecord(
            record_id=f"snli-dev-pilot-v1-{index}",
            dataset_id="snli-dev-pilot-v1",
            source_group_id=f"snli-group-{index}",
            request=DecisionRequest(
                context=f"Premise {index}.",
                question=f"Does the hypothesis follow? Hypothesis {index}.",
                options=(
                    Option(id="entailment", label="Entailment"),
                    Option(id="neutral", label="Neutral"),
                    Option(id="contradiction", label="Contradiction"),
                ),
            ),
            answer_id="entailment",
        )
        for index in range(32)
    )
    return tuple(records)


def test_comparison_requires_complete_untampered_permutation_receipts() -> None:
    records = _pilot_records()
    remote_records = broader_runner_core.serialize_remote_records(records)
    expected_rows = broader_runner_core.presentations(remote_records)
    receipts = {
        name: {
            "model_name": name,
            "status": "passed",
            "presentations": [
                {
                    "record_id": row["record_id"],
                    "request_hash": row["request_hash"],
                    "permutation_index": row["permutation_index"],
                    "option_ids": row["option_ids"],
                    "raw_logits": [0.0] * len(row["option_ids"]),
                    "input_tokens": 5,
                    "prompt_sha256": "a" * 64,
                }
                for row in expected_rows
            ],
            "scored_presentation_count": 256,
            "auxiliary_forward_count": {"qwen": 0, "intern": 1, "kev": 2}[name],
            "total_forward_count": 256 + {"qwen": 0, "intern": 1, "kev": 2}[name],
        }
        for name in ("qwen", "intern", "kev")
    }

    assert broader_runner_core.comparison_passed(receipts, remote_records) is True
    receipts["kev"]["presentations"].pop()
    assert broader_runner_core.comparison_passed(receipts, remote_records) is False

    receipts["kev"]["presentations"] = receipts["qwen"]["presentations"].copy()
    receipts["kev"]["presentations"][1] = receipts["kev"]["presentations"][0]
    assert broader_runner_core.comparison_passed(receipts, remote_records) is False

    receipts["kev"]["presentations"] = receipts["qwen"]["presentations"].copy()
    receipts["kev"]["presentations"][1] = {
        **receipts["kev"]["presentations"][1],
        "request_hash": "0" * 64,
    }
    assert broader_runner_core.comparison_passed(receipts, remote_records) is False

    receipts["kev"]["presentations"] = receipts["qwen"]["presentations"].copy()
    receipts["kev"]["presentations"][1] = {
        **receipts["kev"]["presentations"][1],
        "permutation_index": True,
    }
    assert broader_runner_core.comparison_passed(receipts, remote_records) is False
