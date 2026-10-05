from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pytest

from experiments import targeted_training_data as data
from reflex_decisions.data import DecisionRecord
from reflex_decisions.schema import DecisionRequest, Option


def _request(context: str, question: str) -> DecisionRequest:
    return DecisionRequest(
        context=context,
        question=question,
        options=(Option(id="entailment", label="yes"), Option(id="non-entailment", label="no")),
    )


def _candidate(
    split: str,
    source_id: str,
    subcase: str,
    context: str,
    *,
    question: str = "Does it follow?",
    options: tuple[Option, Option] | None = None,
    answer: str = "entailment",
) -> data._Candidate:
    if options is None:
        options = (
            Option(id="entailment", label="yes"),
            Option(id="non-entailment", label="no"),
        )
    return data._Candidate(
        split=split,
        raw_source_id=source_id,
        source_group_id=subcase,
        request=DecisionRequest(context=context, question=question, options=options),
        answer_id=answer,
        source_metadata={"subcase": subcase},
    )


def _record(
    dataset_id: str, record_id: str, context: str, answer: str = "option1"
) -> DecisionRecord:
    return DecisionRecord(
        dataset_id=dataset_id,
        record_id=record_id,
        source_group_id=f"group:{record_id}",
        request=DecisionRequest(
            context=context,
            question="Choose.",
            options=(Option(id="option1", label="A"), Option(id="option2", label="B")),
        ),
        answer_id=answer,
    )


def test_normalization_is_nfkc_casefold_and_collapsed_whitespace() -> None:
    assert data.normalize_text("  Ａ  Straße\n\tB  ") == "a strasse b"


def test_hans_uses_split_local_ids_but_excludes_eval_premises_globally() -> None:
    train = (
        _candidate("hans/train", "same-id", "subcase-a", "Train premise"),
        _candidate("hans/train", "overlap-id", "subcase-b", "Eval premise"),
        _candidate("hans/train", "train-ok", "subcase-b", "Unique train premise"),
    )
    evaluation = (_candidate("hans/evaluation", "same-id", "subcase-a", "Eval premise"),)

    selected, _reserved, audit = data.select_hans_candidates(
        train,
        evaluation,
        old_training=(),
        old_monitoring=(),
        train_quota=1,
        reserve_quota=0,
    )

    selected_ids = {record.record_id for record in selected}
    assert "targeted-reasoning-v1:hans/train/same-id" in selected_ids
    assert "targeted-reasoning-v1:hans/train/overlap-id" not in selected_ids
    assert audit["cross_split_raw_id_collisions"] == 1


def test_hans_excludes_snli_premise_and_hypothesis_rendered_as_a_pair() -> None:
    premise = "A musician plays a string instrument."
    hypothesis = "A musician plays music."
    snli_record = DecisionRecord(
        dataset_id="snli-training-v1",
        record_id="snli-training-v1:fixture",
        source_group_id="snli-group:fixture",
        request=DecisionRequest(
            context=f"Premise: {premise}\nHypothesis: {hypothesis}",
            question="Does the premise entail the hypothesis?",
            options=(
                Option(id="entailment", label="Entailment"),
                Option(id="neutral", label="Neutral"),
                Option(id="contradiction", label="Contradiction"),
            ),
        ),
        answer_id="entailment",
    )
    candidate = data._Candidate(
        split="hans/train",
        raw_source_id="same-pair",
        source_group_id="subcase-a",
        request=_request(
            premise,
            "Does the following statement necessarily follow from the context? " + hypothesis,
        ),
        answer_id="entailment",
        source_metadata={"subcase": "subcase-a"},
    )

    with pytest.raises(ValueError, match="quota"):
        data.select_hans_candidates(
            (candidate,),
            (),
            old_training=(snli_record,),
            old_monitoring=(),
            train_quota=1,
            reserve_quota=0,
        )


def test_hans_separation_extracts_balanced_snli_pair_from_evaluation_context() -> None:
    premise = "A musician plays a string instrument."
    hypothesis = "A musician plays music."
    balanced_record = DecisionRecord(
        dataset_id="snli-balanced-v1-development",
        record_id="snli-balanced:fixture",
        source_group_id="snli-balanced-group:fixture",
        request=DecisionRequest(
            context=f"Premise: {premise}\nHypothesis: {hypothesis}",
            question="Does the premise entail the hypothesis?",
            options=(
                Option(id="entailment", label="Entailment"),
                Option(id="neutral", label="Neutral"),
                Option(id="contradiction", label="Contradiction"),
            ),
        ),
        answer_id="entailment",
    )
    candidate = _candidate(
        "hans/train",
        "same-balanced-pair",
        "subcase-a",
        premise,
        question="Does the following statement necessarily follow from the context? " + hypothesis,
    )
    hans_training = data._hans_records((candidate,))

    with pytest.raises(ValueError, match="overlaps retention or evaluation"):
        data._validate_new_training_separation(hans_training, (), (balanced_record,))


def test_hans_separation_checks_every_approved_evaluation_request() -> None:
    candidate = _candidate(
        "hans/train",
        "same-evaluation-request",
        "subcase-a",
        "A musician plays a string instrument.",
        question="Does the following statement necessarily follow from the context? "
        "A musician plays music.",
    )
    unrelated_dataset_record = DecisionRecord(
        dataset_id="copa-dev-pilot-v1",
        record_id="copa:fixture",
        source_group_id="copa-group:fixture",
        request=candidate.request,
        answer_id="entailment",
    )

    with pytest.raises(ValueError, match="overlaps retention or evaluation"):
        data._validate_new_training_separation(
            data._hans_records((candidate,)), (), (unrelated_dataset_record,)
        )


def test_hans_processes_raw_subcase_names_before_hashed_group_ids() -> None:
    candidates = (
        _candidate("hans/train", "alpha", "zzz-hash", "Shared premise"),
        _candidate("hans/train", "shared", "aaa-hash", "Shared premise"),
        _candidate("hans/train", "unique", "aaa-hash", "Unique beta premise"),
    )
    candidates = tuple(
        data._Candidate(
            split=row.split,
            raw_source_id=row.raw_source_id,
            source_group_id=row.source_group_id,
            request=row.request,
            answer_id=row.answer_id,
            source_metadata={"subcase": "alpha" if row.raw_source_id == "alpha" else "beta"},
        )
        for row in candidates
    )

    selected, _reserved, _audit = data.select_hans_candidates(
        candidates,
        (),
        old_training=(),
        old_monitoring=(),
        train_quota=1,
        reserve_quota=0,
    )

    assert {record.record_id for record in selected} == {
        "targeted-reasoning-v1:hans/train/alpha",
        "targeted-reasoning-v1:hans/train/unique",
    }


def test_schedules_share_the_three_old_streams_and_emit_four_ordered_microbatches() -> None:
    pools = {
        name: tuple(
            _record(name, f"{name}-{index}", f"{name} example {index}") for index in range(4)
        )
        for name in ("real", "synthetic", "snli", "hans", "winogrande")
    }

    schedules = data.build_schedules(pools, rows_per_stream=4)

    assert set(schedules) == {"control", "treatment"}
    assert len(schedules["control"]) == len(schedules["treatment"]) == 16
    for update in range(4):
        for microbatch in range(3):
            position = update * 4 + microbatch
            control = dict(schedules["control"][position])
            treatment = dict(schedules["treatment"][position])
            control.pop("presentation_id")
            treatment.pop("presentation_id")
            assert control == treatment
        for role in ("control", "treatment"):
            row = schedules[role][update * 4 + 3]
            assert row["microbatch_index"] == 3
            assert row["update"] == update + 1
            assert row["gold_index"] in (0, 1)
            assert row["order_ids"][row["gold_index"]] == row["gold_option_id"]
            assert "answer_id" not in row
            assert set(row) == data.PRESENTATION_FIELDS | {
                "gold_option_id",
                "gold_index",
                "update",
                "microbatch_index",
            }
        assert schedules["control"][update * 4 + 3]["dataset_id"] in {
            "real",
            "synthetic",
            "snli",
        }
    assert [schedules["treatment"][update * 4 + 3]["dataset_id"] for update in range(4)] == [
        "hans",
        "winogrande",
        "hans",
        "winogrande",
    ]


def test_reserved_presentations_are_sorted_rotations_without_gold_fields() -> None:
    hans = _record("targeted-hans-reserved-v1", "hans:record", "HANS context", "option2")
    wino = _record("targeted-winogrande-reserved-v1", "wino:record", "Wino context", "option1")

    presentations = data.build_reserved_presentations((wino, hans))

    assert [row["dataset_id"] for row in presentations] == [
        "targeted-hans-reserved-v1",
        "targeted-hans-reserved-v1",
        "targeted-winogrande-reserved-v1",
        "targeted-winogrande-reserved-v1",
    ]
    assert presentations[0]["order_ids"] == ["option1", "option2"]
    assert presentations[1]["order_ids"] == ["option2", "option1"]
    assert presentations[0]["request_hash"] == presentations[1]["request_hash"]
    assert presentations[0]["presentation_id"].endswith(":order-0")
    assert all(set(row) == data.PRESENTATION_FIELDS for row in presentations)
    assert all("answer_id" not in row and "gold_option_id" not in row for row in presentations)


def test_winogrande_parser_aligns_labels_without_exposing_them_in_requests() -> None:
    source = b'{"qID":"q-1","sentence":"They left _.","option1":"early","option2":"late"}\n'
    candidates = data.parse_winogrande_rows(
        source,
        b"2\n",
        split="winogrande/train_xl",
    )

    assert len(candidates) == 1
    assert candidates[0].qualified_source_id == "winogrande/train_xl/q-1"
    assert candidates[0].answer_id == "option2"
    assert "answer" not in candidates[0].request.model_dump(mode="json")


def test_winogrande_parser_cross_checks_an_embedded_answer_against_aligned_labels() -> None:
    source = (
        b'{"qID":"q-embedded","sentence":"They left _.",'
        b'"option1":"early","option2":"late","answer":"2"}\n'
    )

    candidates = data.parse_winogrande_rows(source, b"2\n", split="winogrande/train_xl")

    assert candidates[0].answer_id == "option2"
    with pytest.raises(ValueError, match="conflicts"):
        data.parse_winogrande_rows(
            source.replace(b'"answer":"2"', b'"answer":"1"'),
            b"2\n",
            split="winogrande/train_xl",
        )


def test_pinned_reader_rejects_hash_mismatch_and_root_escape(tmp_path) -> None:
    safe = tmp_path / "safe.txt"
    safe.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="SHA-256"):
        data._read_pinned_file(tmp_path, "safe.txt", "0" * 64)

    outside = tmp_path.parent / "outside.txt"
    outside.write_bytes(b"outside")
    (tmp_path / "escape.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="pinned input"):
        data._read_pinned_file(tmp_path, "escape.txt", hashlib.sha256(b"outside").hexdigest())


def test_old_panel_filter_discards_unapproved_rows_before_validating_membership() -> None:
    record = _record("retention-fixture-v1", "keep:1", "Approved request")
    keep = data._presentation_row(
        record,
        record.request.options,
        0,
        identifier="keep:1:order-0",
    )
    unapproved = {"dataset_id": "sealed-test-v1", "payload": "not a presentation"}

    retained = data._filter_presentation_rows(
        (unapproved, keep), allowed_dataset_ids={"retention-fixture-v1"}
    )

    assert retained == (keep,)


def test_natural_reasoning_receipt_uses_pinned_schema_version_two() -> None:
    receipt = {
        "schema_version": 2,
        "experiment_id": "natural-reasoning-2026-10-04-r1",
        "status": "passed",
        "payloads": {
            "initialize": {},
            "synthetic_repeat": {},
            "snli_mix": {"phase": "train", "arm": "snli_mix"},
        },
        "results": {},
        "failure": None,
        "modal": {},
    }

    payload = data._receipt_payload(data._canonical_json(receipt), kind="natural-reasoning")

    assert payload == {"phase": "train", "arm": "snli_mix"}


def test_record_readers_accept_omitted_optional_option_descriptions() -> None:
    record = _record("records-fixture-v1", "record:omitted-description", "A fixture request")
    row = record.model_dump(mode="json")
    for option in row["request"]["options"]:
        option.pop("description")
    raw = data._canonical_json(row) + b"\n"

    receipt_records = data._records_from_rows([row], label="fixture receipt")
    jsonl_records = data._parse_jsonl_records(
        raw,
        "fixture.jsonl",
        allowed_dataset_ids={record.dataset_id},
        membership={record.record_id: record.dataset_id},
    )

    assert receipt_records == jsonl_records == (record,)


def test_record_readers_still_reject_extra_option_fields() -> None:
    record = _record("records-fixture-v1", "record:extra-field", "A fixture request")
    row = record.model_dump(mode="json")
    row["request"]["options"][0]["unexpected"] = "not approved"

    with pytest.raises(ValueError, match="malformed record"):
        data._records_from_rows([row], label="fixture receipt")
    with pytest.raises(ValueError, match="malformed"):
        data._parse_jsonl_records(
            data._canonical_json(row) + b"\n",
            "fixture.jsonl",
            allowed_dataset_ids={record.dataset_id},
            membership={record.record_id: record.dataset_id},
        )


def test_winogrande_overlap_key_accepts_multichoice_old_training_records() -> None:
    old_training = DecisionRecord(
        dataset_id="old-multichoice-v1",
        record_id="old:multichoice",
        source_group_id="old-group:multichoice",
        request=DecisionRequest(
            context="A person picked _. ",
            question="Which option fills the blank?",
            options=(
                Option(id="option1", label="oak"),
                Option(id="option2", label="pine"),
                Option(id="option3", label="birch"),
            ),
        ),
        answer_id="option1",
    )
    candidate = _candidate(
        "winogrande/train_xl",
        "wino:fixture",
        "",
        "A person picked _. ",
        question="Which option fills the blank?",
        options=(Option(id="option1", label="oak"), Option(id="option2", label="pine")),
        answer="option1",
    )

    selected, _reserved, _audit = data.select_winogrande_candidates(
        (candidate,),
        (),
        old_training=(old_training,),
        old_monitoring=(),
        train_quota=1,
        reserve_quota=0,
    )

    assert len(selected) == 1


def test_write_bundle_is_exclusive_and_manifest_hashes_every_data_file(tmp_path) -> None:
    pools = {
        name: (_record(f"{name}-fixture", f"{name}:1", f"{name} context"),)
        for name in ("real", "synthetic", "snli", "hans", "winogrande")
    }
    reserved = _record(
        data.HANS_RESERVED_DATASET,
        "reserved:1",
        "Reserved fixture context",
        "option2",
    )
    inputs = data.TargetedInputs(
        training_pools=pools,
        schedules={"control": (), "treatment": ()},
        evaluation_records=(reserved,),
        presentations=data.build_reserved_presentations((reserved,)),
        strata={data.HANS_RESERVED_DATASET: "reserved"},
        selection={"selection_id": "fixture"},
        file_sha256={"fixture": "0" * 64},
        audit={"fixture": True},
    )
    output_dir = tmp_path / "bundle"

    manifest = data.write_bundle(inputs, output_dir)

    data_paths = {
        "training-records.jsonl",
        "evaluation-records.jsonl",
        "presentations.json",
        "schedules.json",
        "audit.json",
    }
    assert set(manifest["bundle_file_sha256"]) == data_paths
    assert {path.name for path in output_dir.iterdir()} == data_paths | {"manifest.json"}
    assert all(
        hashlib.sha256((output_dir / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["bundle_file_sha256"].items()
    )
    presentations = json.loads((output_dir / "presentations.json").read_bytes())
    assert all("answer_id" not in row for row in presentations)
    with pytest.raises(FileExistsError):
        data.write_bundle(inputs, output_dir)


def test_wino_archive_reader_never_opens_official_test_members(monkeypatch) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        for index, name in enumerate(data.WINOGRANDE_MEMBERS):
            payload = b"x" * 5_000_001 if index == 0 else b"fixture"
            archive.writestr(name, payload)
        archive.writestr("winogrande_1.1/test.jsonl", b"must remain unopened")
        archive.writestr("winogrande_1.1/test-labels.lst", b"must remain unopened")
    opened: list[str] = []
    original_open = zipfile.ZipFile.open

    def track_open(self, name, *args, **kwargs):
        opened.append(name if isinstance(name, str) else name.filename)
        return original_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", track_open)
    members = data._read_wino_members(archive_bytes.getvalue())

    assert set(members) == set(data.WINOGRANDE_MEMBERS)
    assert len(members[data.WINOGRANDE_MEMBERS[0]]) == 5_000_001
    assert set(opened) == set(data.WINOGRANDE_MEMBERS)
    assert not any("test" in name for name in opened)


def test_winogrande_excludes_dev_blocks_and_sentences_and_conflict_blocks() -> None:
    train = (
        _candidate(
            "winogrande/train_xl",
            "block-twin",
            "",
            "A person picked _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="oak"), Option(id="option2", label="pine")),
            answer="option1",
        ),
        _candidate(
            "winogrande/train_xl",
            "sentence-twin",
            "",
            "Another person picked _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="red"), Option(id="option2", label="blue")),
            answer="option1",
        ),
        _candidate(
            "winogrande/train_xl",
            "conflict-train",
            "",
            "A conflicting person picked _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="sun"), Option(id="option2", label="moon")),
            answer="option1",
        ),
        _candidate(
            "winogrande/train_xl",
            "train-only",
            "",
            "Only the train person picked _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="east"), Option(id="option2", label="west")),
            answer="option2",
        ),
    )
    dev = (
        _candidate(
            "winogrande/dev",
            "block-twin",
            "",
            "Dev wording uses _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="pine"), Option(id="option2", label="oak")),
            answer="option2",
        ),
        _candidate(
            "winogrande/dev",
            "sentence-twin",
            "",
            "Another person picked _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="green"), Option(id="option2", label="gold")),
            answer="option1",
        ),
        _candidate(
            "winogrande/dev",
            "conflict-dev",
            "",
            "A conflicting person picked _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="moon"), Option(id="option2", label="sun")),
            answer="option1",
        ),
        _candidate(
            "winogrande/dev",
            "dev-only",
            "",
            "Only the dev person picked _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="left"), Option(id="option2", label="right")),
            answer="option2",
        ),
    )

    selected_train, selected_reserved, audit = data.select_winogrande_candidates(
        train,
        dev,
        old_training=(),
        old_monitoring=(),
        old_monitor_source_ids={"block-twin", "sentence-twin"},
        train_quota=1,
        reserve_quota=1,
    )

    assert [record.record_id for record in selected_train] == [
        "targeted-reasoning-v1:winogrande/train_xl/train-only"
    ]
    assert [record.record_id for record in selected_reserved] == [
        "targeted-reasoning-v1:winogrande/dev/dev-only"
    ]
    assert selected_train[0].answer_id == "option2"
    assert audit["conflicting_request_groups"] == 1


def test_winogrande_excludes_raw_question_id_collisions_across_splits() -> None:
    train = (
        _candidate(
            "winogrande/train_xl",
            "shared-id",
            "",
            "Training collision sentence _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="oak"), Option(id="option2", label="pine")),
            answer="option1",
        ),
        _candidate(
            "winogrande/train_xl",
            "train-only",
            "",
            "Training-only sentence _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="east"), Option(id="option2", label="west")),
            answer="option1",
        ),
    )
    development = (
        _candidate(
            "winogrande/dev",
            "shared-id",
            "",
            "Development collision sentence _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="sun"), Option(id="option2", label="moon")),
            answer="option1",
        ),
        _candidate(
            "winogrande/dev",
            "dev-only",
            "",
            "Development-only sentence _.",
            question="Which option fills the blank?",
            options=(Option(id="option1", label="left"), Option(id="option2", label="right")),
            answer="option1",
        ),
    )

    selected_train, selected_reserved, _audit = data.select_winogrande_candidates(
        train,
        development,
        old_training=(),
        old_monitoring=(),
        train_quota=1,
        reserve_quota=1,
    )

    assert [record.record_id for record in selected_train] == [
        "targeted-reasoning-v1:winogrande/train_xl/train-only"
    ]
    assert [record.record_id for record in selected_reserved] == [
        "targeted-reasoning-v1:winogrande/dev/dev-only"
    ]
