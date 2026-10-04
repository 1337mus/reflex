import json
import math

import pytest

from reflex_decisions import data, evaluation
from reflex_decisions.schema import DecisionRequest, Option


def test_jsonl_predictions_preserve_unicode_separators_inside_strings(tmp_path) -> None:
    revisions = ("before\u2028after", "before\u2029after", "before\u0085after")
    predictions = [
        {
            "record_id": f"prediction-{index}",
            "request_hash": "a" * 64,
            "logits": {"option": float(index)},
            "model_revision": revision,
        }
        for index, revision in enumerate(revisions)
    ]
    contents = (
        json.dumps(predictions[0], ensure_ascii=False)
        + "\n"
        + json.dumps(predictions[1], ensure_ascii=False)
        + "\r\n"
        + json.dumps(predictions[2], ensure_ascii=False)
        + "\n"
    )
    path = tmp_path / "predictions.jsonl"
    path.write_text(contents, encoding="utf-8")

    assert all(
        separator in path.read_text(encoding="utf-8")
        for separator in ("\u2028", "\u2029", "\u0085")
    )
    loaded = evaluation.load_predictions(path)

    assert tuple(prediction.model_revision for prediction in loaded) == revisions


def test_three_row_fixture_matches_accuracy_and_dataset_macro_f1() -> None:
    manifest = data.SplitManifest(
        data_kind="fixture",
        datasets=(
            data.DatasetSpec(
                dataset_id="d1",
                source_id="source-d1",
                task_family="family-a",
                split="test",
                source_uri="local://fixture/d1",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )
    rows = (
        ("r1", "A", (0.8, 0.2)),
        ("r2", "B", (0.2, 0.8)),
        ("r3", "B", (0.6, 0.4)),
    )
    records = tuple(
        data.DecisionRecord(
            record_id=record_id,
            dataset_id="d1",
            source_group_id=record_id,
            request=DecisionRequest(
                context=f"Request {record_id}",
                question="Choose one",
                options=(Option(id="A", label="Option A"), Option(id="B", label="Option B")),
            ),
            answer_id=answer,
        )
        for record_id, answer, _ in rows
    )
    request_hashes = {record.record_id: record.request.request_hash for record in records}
    predictions = tuple(
        reversed(
            tuple(
                evaluation.PredictionRecord(
                    record_id=record_id,
                    request_hash=request_hashes[record_id],
                    logits={"A": math.log(probabilities[0]), "B": math.log(probabilities[1])},
                    model_revision="fixture-v1",
                )
                for record_id, _, probabilities in rows
            )
        )
    )

    report = evaluation.evaluate(
        records,
        predictions,
        manifest,
        records_sha256="0" * 64,
        predictions_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )

    assert math.isclose(report.overall.micro_accuracy, 2 / 3)
    assert math.isclose(report.per_dataset[0].macro_f1, 2 / 3)
    assert math.isclose(report.per_dataset[0].nll, 0.454192611501)
    assert math.isclose(report.per_dataset[0].brier, 0.293333333333)
    assert math.isclose(report.per_dataset[0].ece, 1 / 3)

    accepted = evaluation.evaluate(
        records,
        predictions,
        manifest,
        records_sha256="0" * 64,
        predictions_sha256="1" * 64,
        manifest_sha256="2" * 64,
        min_confidence=0.8,
    )
    assert math.isclose(accepted.overall.coverage, 2 / 3)
    assert accepted.overall.accepted_risk == 0.0

    none_accepted = evaluation.evaluate(
        records,
        predictions,
        manifest,
        records_sha256="0" * 64,
        predictions_sha256="1" * 64,
        manifest_sha256="2" * 64,
        min_confidence=0.81,
    )
    assert none_accepted.overall.coverage == 0.0
    assert none_accepted.overall.accepted_risk is None


def test_risk_coverage_groups_equal_confidence_rows_and_nll_is_not_clipped() -> None:
    manifest = data.SplitManifest(
        data_kind="fixture",
        datasets=(
            data.DatasetSpec(
                dataset_id="d1",
                source_id="source-d1",
                task_family="family-a",
                split="test",
                source_uri="local://fixture/d1",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )
    rows = (
        ("r1", "A", (0.9, 0.1)),
        ("r2", "B", (0.9, 0.1)),
        ("r3", "A", (0.6, 0.4)),
        ("r4", "B", (0.6, 0.4)),
    )
    records = tuple(
        data.DecisionRecord(
            record_id=record_id,
            dataset_id="d1",
            source_group_id=record_id,
            request=DecisionRequest(
                context=f"Request {record_id}",
                question="Choose one",
                options=(Option(id="A", label="Option A"), Option(id="B", label="Option B")),
            ),
            answer_id=answer,
        )
        for record_id, answer, _ in rows
    )
    request_hashes = {record.record_id: record.request.request_hash for record in records}
    predictions = tuple(
        evaluation.PredictionRecord(
            record_id=record_id,
            request_hash=request_hashes[record_id],
            logits={"A": math.log(probabilities[0]), "B": math.log(probabilities[1])},
            model_revision="fixture-v1",
        )
        for record_id, _, probabilities in rows
    )
    report = evaluation.evaluate(
        records,
        predictions,
        manifest,
        records_sha256="0" * 64,
        predictions_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )
    curve = report.overall.risk_coverage_curve

    assert len(curve) == 3
    assert [point.coverage for point in curve] == [0.0, 0.5, 1.0]
    assert [point.risk for point in curve] == [None, 0.5, 0.5]
    assert math.isclose(report.overall.aurc, 0.5)

    reversed_report = evaluation.evaluate(
        tuple(reversed(records)),
        tuple(reversed(predictions)),
        manifest,
        records_sha256="0" * 64,
        predictions_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )
    assert reversed_report.overall.risk_coverage_curve == report.overall.risk_coverage_curve
    assert reversed_report.overall.aurc == report.overall.aurc

    extreme = evaluation.evaluate(
        (records[0].model_copy(update={"answer_id": "B"}),),
        (
            evaluation.PredictionRecord(
                record_id="r1",
                request_hash=records[0].request.request_hash,
                logits={"A": 1000.0, "B": 0.0},
                model_revision="fixture-v1",
            ),
        ),
        manifest,
        records_sha256="0" * 64,
        predictions_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )
    assert extreme.overall.pooled_nll == 1000.0


def test_macro_dataset_f1_keeps_unrelated_label_spaces_separate() -> None:
    first_spec = data.DatasetSpec(
        dataset_id="imbalanced",
        source_id="source-imbalanced",
        task_family="family-a",
        split="test",
        source_uri="local://fixture/imbalanced",
        source_revision="fixture-v1",
        license="synthetic fixture",
    )
    second_spec = data.DatasetSpec(
        dataset_id="balanced",
        source_id="source-balanced",
        task_family="family-b",
        split="test",
        source_uri="local://fixture/balanced",
        source_revision="fixture-v1",
        license="synthetic fixture",
    )
    manifest = data.SplitManifest(data_kind="fixture", datasets=(first_spec, second_spec))
    records: list[data.DecisionRecord] = []
    predictions: list[evaluation.PredictionRecord] = []
    for index in range(10):
        record_id = f"d1-{index}"
        answer = "A" if index < 9 else "B"
        records.append(
            data.DecisionRecord(
                record_id=record_id,
                dataset_id="imbalanced",
                source_group_id=record_id,
                request=DecisionRequest(
                    context=f"D1 request {index}",
                    question="Choose one",
                    options=(Option(id="A", label="A"), Option(id="B", label="B")),
                ),
                answer_id=answer,
            )
        )
        predictions.append(
            evaluation.PredictionRecord(
                record_id=record_id,
                request_hash=records[-1].request.request_hash,
                logits={"A": 3.0, "B": 0.0},
                model_revision="fixture-v1",
            )
        )
    for label in ("X", "Y", "Z"):
        record_id = f"d2-{label}"
        records.append(
            data.DecisionRecord(
                record_id=record_id,
                dataset_id="balanced",
                source_group_id=record_id,
                request=DecisionRequest(
                    context=f"D2 request {label}",
                    question="Choose one",
                    options=tuple(Option(id=value, label=value) for value in ("X", "Y", "Z")),
                ),
                answer_id=label,
            )
        )
        predictions.append(
            evaluation.PredictionRecord(
                record_id=record_id,
                request_hash=records[-1].request.request_hash,
                logits={"X": 0.0, "Y": 0.0, "Z": 0.0} | {label: 3.0},
                model_revision="fixture-v1",
            )
        )
    report = evaluation.evaluate(
        records,
        predictions,
        manifest,
        records_sha256="0" * 64,
        predictions_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )

    assert math.isclose(report.overall.macro_dataset_f1, (18 / 19 / 2 + 1) / 2)
    assert math.isclose(report.overall.macro_dataset_accuracy, 0.95)
    assert math.isclose(report.overall.micro_accuracy, 12 / 13)


def test_evaluation_rejects_mixed_splits_in_one_report() -> None:
    manifest = data.SplitManifest(
        data_kind="fixture",
        datasets=(
            data.DatasetSpec(
                dataset_id="train-set",
                source_id="source-train",
                task_family="family-train",
                split="train",
                source_uri="local://fixture/train",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
            data.DatasetSpec(
                dataset_id="test-set",
                source_id="source-test",
                task_family="family-test",
                split="test",
                source_uri="local://fixture/test",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )
    options = (Option(id="A", label="A"), Option(id="B", label="B"))
    records = tuple(
        data.DecisionRecord(
            record_id=record_id,
            dataset_id=dataset_id,
            source_group_id=record_id,
            request=DecisionRequest(context=record_id, question="Choose", options=options),
            answer_id="A",
        )
        for record_id, dataset_id in (("train-1", "train-set"), ("test-1", "test-set"))
    )
    predictions = tuple(
        evaluation.PredictionRecord(
            record_id=record_id,
            request_hash=next(
                record.request.request_hash for record in records if record.record_id == record_id
            ),
            logits={"A": 1.0, "B": 0.0},
            model_revision="fixture-v1",
        )
        for record_id in ("train-1", "test-1")
    )

    with pytest.raises(ValueError, match="one split"):
        evaluation.evaluate(
            records,
            predictions,
            manifest,
            records_sha256="0" * 64,
            predictions_sha256="1" * 64,
            manifest_sha256="2" * 64,
        )


def test_evaluation_rejects_prediction_from_a_different_request_hash() -> None:
    manifest = data.SplitManifest(
        data_kind="fixture",
        datasets=(
            data.DatasetSpec(
                dataset_id="test-set",
                source_id="source-test",
                task_family="family-test",
                split="test",
                source_uri="local://fixture/test",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )
    record = data.DecisionRecord(
        record_id="test-1",
        dataset_id="test-set",
        source_group_id="group-1",
        request=DecisionRequest(
            context="Current context",
            question="Choose",
            options=(Option(id="A", label="A"), Option(id="B", label="B")),
        ),
        answer_id="A",
    )

    class StalePrediction(evaluation.PredictionRecord):
        request_hash: str

    prediction = StalePrediction(
        record_id="test-1",
        logits={"A": 1.0, "B": 0.0},
        model_revision="fixture-v1",
        request_hash="0" * 64,
    )

    with pytest.raises(ValueError, match="request_hash"):
        evaluation.evaluate(
            (record,),
            (prediction,),
            manifest,
            records_sha256="0" * 64,
            predictions_sha256="1" * 64,
            manifest_sha256="2" * 64,
        )


def test_evaluation_rejects_missing_extra_duplicate_and_wrong_option_predictions() -> None:
    manifest = data.SplitManifest(
        data_kind="fixture",
        datasets=(
            data.DatasetSpec(
                dataset_id="test-set",
                source_id="source-test",
                task_family="family-test",
                split="test",
                source_uri="local://fixture/test",
                source_revision="fixture-v1",
                license="synthetic fixture",
            ),
        ),
    )
    record = data.DecisionRecord(
        record_id="test-1",
        dataset_id="test-set",
        source_group_id="group-1",
        request=DecisionRequest(
            context="Current context",
            question="Choose",
            options=(Option(id="A", label="A"), Option(id="B", label="B")),
        ),
        answer_id="A",
    )
    prediction = evaluation.PredictionRecord(
        record_id=record.record_id,
        request_hash=record.request.request_hash,
        logits={"A": 1.0, "B": 0.0},
        model_revision="fixture-v1",
    )
    extra = prediction.model_copy(update={"record_id": "extra-1"})
    wrong_options = prediction.model_copy(update={"logits": {"A": 1.0, "C": 0.0}})
    cases = (
        ((), "missing predictions"),
        ((extra,), "missing predictions"),
        ((prediction, prediction), "duplicate prediction"),
        ((wrong_options,), "wrong option IDs"),
    )

    for saved_predictions, message in cases:
        with pytest.raises(ValueError, match=message):
            evaluation.evaluate(
                (record,),
                saved_predictions,
                manifest,
                records_sha256="0" * 64,
                predictions_sha256="1" * 64,
                manifest_sha256="2" * 64,
            )
