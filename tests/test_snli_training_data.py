import json

import pytest

from reflex_decisions import broader_data, snli_training_source


def test_training_selection_has_fixed_quotas_and_is_row_order_independent() -> None:
    import hashlib

    labels = broader_data.SNLI_LABELS
    items = tuple(
        broader_data.SnliItem(
            source_item_id=f"pair-{group_index:03}-{label}-{copy_index}",
            source_group_id=f"group-{group_index:03}",
            premise=f"Premise {group_index}.",
            hypothesis=f"Hypothesis {label} {group_index}.",
            label=label,
        )
        for group_index in range(510)
        for label in labels
        for copy_index in range(2)
    )

    from reflex_decisions import snli_training_data

    selected = snli_training_data.select_snli_training_items(items)
    reordered = snli_training_data.select_snli_training_items(tuple(reversed(items)))

    assert [(item.source_group_id, item.source_item_id, item.label) for item in selected] == [
        (item.source_group_id, item.source_item_id, item.label) for item in reordered
    ]
    assert len(selected) == 500
    assert len({item.source_group_id for item in selected}) == 500
    assert {label: sum(item.label == label for item in selected) for label in labels} == {
        "entailment": 167,
        "neutral": 167,
        "contradiction": 166,
    }

    group_rank = sorted(
        {item.source_group_id for item in items},
        key=lambda group_id: (
            hashlib.sha256(f"20261008:group:{group_id}".encode()).hexdigest(),
            group_id,
        ),
    )[:500]
    assert [item.source_group_id for item in selected] == group_rank
    label_rank = sorted(
        group_rank,
        key=lambda group_id: (
            hashlib.sha256(f"20261008:label:{group_id}".encode()).hexdigest(),
            group_id,
        ),
    )
    group_labels: dict[str, str] = {}
    offset = 0
    for label, quota in zip(labels, (167, 167, 166), strict=True):
        group_labels.update((group_id, label) for group_id in label_rank[offset : offset + quota])
        offset += quota
    by_group: dict[str, list[broader_data.SnliItem]] = {}
    for item in items:
        by_group.setdefault(item.source_group_id, []).append(item)
    expected = [
        min(
            (item for item in by_group[group_id] if item.label == group_labels[group_id]),
            key=lambda item: (
                hashlib.sha256(f"20261008:pair:{item.source_item_id}".encode()).hexdigest(),
                item.source_item_id,
            ),
        )
        for group_id in group_rank
    ]
    assert [(item.source_group_id, item.source_item_id, item.label) for item in selected] == [
        (item.source_group_id, item.source_item_id, item.label) for item in expected
    ]


def test_training_selection_rejects_fewer_than_500_eligible_groups() -> None:
    from reflex_decisions import snli_training_data

    items = tuple(
        broader_data.SnliItem(
            source_item_id=f"pair-{group_index}-{label}",
            source_group_id=f"group-{group_index}",
            premise=f"Premise {group_index}.",
            hypothesis=f"Hypothesis {label} {group_index}.",
            label=label,
        )
        for group_index in range(499)
        for label in broader_data.SNLI_LABELS
    )

    with pytest.raises(ValueError, match="fewer than 500 eligible"):
        snli_training_data.select_snli_training_items(items)


def test_cross_split_premise_match_excludes_whole_group_despite_different_group_ids() -> None:
    from reflex_decisions import snli_training_data

    train = snli_training_source._parse_snli_jsonl(
        [
            json.dumps(
                {
                    "captionID": "train-caption",
                    "pairID": "train-pair",
                    "sentence1": "Shared premise.",
                    "sentence2": "Train hypothesis.",
                    "gold_label": "entailment",
                }
            )
        ],
        include_unlabeled_anchors=True,
        conflict_policy="exclude",
    )
    dev = snli_training_source._parse_snli_jsonl(
        [
            json.dumps(
                {
                    "captionID": "dev-caption",
                    "pairID": "dev-pair",
                    "sentence1": " Shared   premise. ",
                    "sentence2": "Dev hypothesis.",
                    "gold_label": "neutral",
                }
            )
        ],
        include_unlabeled_anchors=True,
        conflict_policy="exclude",
    )

    assert train.items[0].source_group_id != dev.items[0].source_group_id
    result = snli_training_data.exclude_train_dev_overlaps(train, dev)

    assert result.shared_premise_anchor_count == 1
    assert result.shared_caption_anchor_count == 0
    assert result.exact_normalized_pair_overlap_count == 0
    assert train.items[0].source_group_id in result.excluded_group_ids
    assert result.remaining_items == ()


def test_cross_split_caption_match_excludes_whole_group_with_different_group_ids() -> None:
    from reflex_decisions import snli_training_data

    train = snli_training_source._parse_snli_jsonl(
        [
            json.dumps(
                {
                    "captionID": "shared-caption",
                    "pairID": "train-caption-pair",
                    "sentence1": "Train premise.",
                    "sentence2": "Train hypothesis.",
                    "gold_label": "entailment",
                }
            )
        ],
        include_unlabeled_anchors=True,
        conflict_policy="exclude",
    )
    dev = snli_training_source._parse_snli_jsonl(
        [
            json.dumps(
                {
                    "captionID": " shared-caption ",
                    "pairID": "dev-caption-pair",
                    "sentence1": "Dev premise.",
                    "sentence2": "Dev hypothesis.",
                    "gold_label": "neutral",
                }
            )
        ],
        include_unlabeled_anchors=True,
        conflict_policy="exclude",
    )

    assert train.items[0].source_group_id != dev.items[0].source_group_id
    result = snli_training_data.exclude_train_dev_overlaps(train, dev)

    assert result.shared_premise_anchor_count == 0
    assert result.shared_caption_anchor_count == 1
    assert result.remaining_items == ()


def test_exact_normalized_pair_overlap_is_counted_and_excluded() -> None:
    from reflex_decisions import snli_training_data

    train = snli_training_source._parse_snli_jsonl(
        [
            json.dumps(
                {
                    "captionID": "train-caption",
                    "pairID": "train-exact-pair",
                    "sentence1": "Same premise.",
                    "sentence2": " Same hypothesis. ",
                    "gold_label": "entailment",
                }
            )
        ],
        include_unlabeled_anchors=True,
        conflict_policy="exclude",
    )
    dev = snli_training_source._parse_snli_jsonl(
        [
            json.dumps(
                {
                    "captionID": "dev-caption",
                    "pairID": "dev-exact-pair",
                    "sentence1": " Same   premise. ",
                    "sentence2": "Same hypothesis.",
                    "gold_label": "neutral",
                }
            )
        ],
        include_unlabeled_anchors=True,
        conflict_policy="exclude",
    )

    result = snli_training_data.exclude_train_dev_overlaps(train, dev)

    assert result.exact_normalized_pair_overlap_count == 1
    assert result.excluded_candidate_pair_count == 1
    assert result.remaining_items == ()


def test_archive_reader_streams_only_train_and_dev_members(tmp_path, monkeypatch) -> None:
    import hashlib
    from zipfile import ZipFile as RealZipFile

    from reflex_decisions import snli_training_data

    def row(pair_id: str, label: str) -> bytes:
        return (
            json.dumps(
                {
                    "captionID": f"caption-{pair_id}",
                    "pairID": pair_id,
                    "sentence1": f"Premise {pair_id}.",
                    "sentence2": f"Hypothesis {pair_id}.",
                    "gold_label": label,
                }
            )
            + "\n"
        ).encode()

    train_member = "snli_1.0/snli_1.0_train.jsonl"
    dev_member = "snli_1.0/snli_1.0_dev.jsonl"
    test_member = "snli_1.0/snli_1.0_test.jsonl"
    train_bytes = row("train-1", "entailment")
    dev_bytes = row("dev-1", "neutral")
    archive_path = tmp_path / "snli.zip"
    with RealZipFile(archive_path, "w") as archive:
        archive.writestr(train_member, train_bytes)
        archive.writestr(dev_member, dev_bytes)
        archive.writestr(test_member, b"not JSON; must never be opened")

    monkeypatch.setattr(
        snli_training_data,
        "SNLI_ARCHIVE_SHA256",
        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(
        snli_training_data,
        "SNLI_TRAIN_MEMBER_SHA256",
        hashlib.sha256(train_bytes).hexdigest(),
        raising=False,
    )
    monkeypatch.setattr(
        snli_training_data,
        "SNLI_DEV_MEMBER_SHA256",
        hashlib.sha256(dev_bytes).hexdigest(),
        raising=False,
    )
    opened_members: list[str] = []

    class TrackingArchive:
        def __init__(self, path):
            self.inner = RealZipFile(path)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def infolist(self):
            return self.inner.infolist()

        def open(self, name, mode="r"):
            opened_members.append(name)
            assert name != test_member
            return self.inner.open(name, mode)

    monkeypatch.setattr(snli_training_data, "ZipFile", TrackingArchive, raising=False)

    train, dev, hashes = snli_training_data.read_snli_sources(archive_path)

    assert [item.source_item_id for item in train.items] == ["train-1"]
    assert [item.source_item_id for item in dev.items] == ["dev-1"]
    assert set(opened_members) == {train_member, dev_member}
    assert hashes["archive_sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()


def _synthetic_complete_sources():
    labels = broader_data.SNLI_LABELS
    train_rows = [
        {
            "captionID": f"caption-{group_index}",
            "pairID": f"pair-{group_index}-{label}",
            "sentence1": f"Private premise marker {group_index}.",
            "sentence2": f"Private hypothesis marker {label} {group_index}.",
            "gold_label": label,
        }
        for group_index in range(501)
        for label in labels
    ]
    dev_rows = [
        {
            "captionID": "separate-dev-caption",
            "pairID": "dev-overlap-pair",
            "sentence1": "Private premise marker 0.",
            "sentence2": "Development hypothesis.",
            "gold_label": "neutral",
        }
    ]
    train = snli_training_source._parse_snli_jsonl(
        (json.dumps(row) for row in train_rows),
        include_unlabeled_anchors=True,
        conflict_policy="exclude",
        source_name="SNLI train",
    )
    dev = snli_training_source._parse_snli_jsonl(
        (json.dumps(row) for row in dev_rows),
        include_unlabeled_anchors=True,
        conflict_policy="exclude",
        source_name="SNLI dev",
    )
    return train, dev


def test_candidate_build_has_500_train_records_and_metadata_only_recipe() -> None:
    import hashlib

    from reflex_decisions import snli_training_data

    train, dev = _synthetic_complete_sources()
    source_hashes = {
        "archive_sha256": "a" * 64,
        "train_member_sha256": "b" * 64,
        "development_member_sha256": "c" * 64,
    }

    candidate = snli_training_data.build_snli_training_candidate_from_parsed(
        train, dev, source_hashes
    )

    assert len(candidate.records) == 500
    assert len({record.source_group_id for record in candidate.records}) == 500
    assert {
        label: sum(record.answer_id == label for record in candidate.records)
        for label in broader_data.SNLI_LABELS
    } == {
        "entailment": 167,
        "neutral": 167,
        "contradiction": 166,
    }
    assert {record.dataset_id for record in candidate.records} == {
        snli_training_data.TRAIN_DATASET_ID
    }
    assert candidate.manifest.datasets[0].split == "train"
    assert candidate.manifest.held_out_families == ()
    assert [option.id for option in candidate.records[0].request.options] == list(
        broader_data.SNLI_LABELS
    )
    assert candidate.recipe["counts"]["cross_split"]["excluded_group_count"] == 1
    assert candidate.recipe["counts"]["cross_split"]["exact_normalized_pair_overlap_count"] == 0
    assert b"Private premise marker" in candidate.records_bytes
    assert b"Private premise marker" not in candidate.recipe_bytes
    assert (
        candidate.recipe["outputs"]["records_sha256"]
        == hashlib.sha256(candidate.records_bytes).hexdigest()
    )


def test_archive_reader_rejects_archive_hash_mismatch(tmp_path, monkeypatch) -> None:
    import hashlib
    from zipfile import ZipFile

    from reflex_decisions import snli_training_data

    archive_path = tmp_path / "wrong-snli.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(snli_training_data.TRAIN_MEMBER, b"")
        archive.writestr(snli_training_data.DEV_MEMBER, b"")
    actual_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert actual_hash != "0" * 64
    monkeypatch.setattr(snli_training_data, "SNLI_ARCHIVE_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        snli_training_data.read_snli_sources(archive_path)


def test_candidate_writer_refuses_existing_output_directory(tmp_path) -> None:
    from reflex_decisions import snli_training_data

    train, dev = _synthetic_complete_sources()
    candidate = snli_training_data.build_snli_training_candidate_from_parsed(
        train,
        dev,
        {
            "archive_sha256": "a" * 64,
            "train_member_sha256": "b" * 64,
            "development_member_sha256": "c" * 64,
        },
    )
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        snli_training_data.write_snli_training_candidate(candidate, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in output_dir.iterdir()) == ["keep.txt"]


def test_prepare_cli_is_plan_only_without_write_flag(tmp_path, monkeypatch, capsys) -> None:
    import json

    from experiments import prepare_snli_training
    from reflex_decisions import snli_training_data

    train, dev = _synthetic_complete_sources()
    candidate = snli_training_data.build_snli_training_candidate_from_parsed(
        train,
        dev,
        {
            "archive_sha256": "a" * 64,
            "train_member_sha256": "b" * 64,
            "development_member_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        prepare_snli_training, "build_snli_training_candidate", lambda archive_path: candidate
    )
    output_dir, evidence_path = tmp_path / "candidate", tmp_path / "evidence.json"

    assert (
        prepare_snli_training.main(
            [
                "--archive",
                str(tmp_path / "archive.zip"),
                "--output-dir",
                str(output_dir),
                "--evidence",
                str(evidence_path),
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["written"] is False
    assert printed["output_dir"] is None
    assert not output_dir.exists()
    assert not evidence_path.exists()


def test_archive_reader_rejects_train_member_hash_mismatch(tmp_path, monkeypatch) -> None:
    import hashlib
    from zipfile import ZipFile

    from reflex_decisions import snli_training_data

    train_bytes = (
        json.dumps(
            {
                "captionID": "train-caption",
                "pairID": "train-pair",
                "sentence1": "Train premise.",
                "sentence2": "Train hypothesis.",
                "gold_label": "entailment",
            }
        )
        + "\n"
    ).encode()
    dev_bytes = (
        json.dumps(
            {
                "captionID": "dev-caption",
                "pairID": "dev-pair",
                "sentence1": "Dev premise.",
                "sentence2": "Dev hypothesis.",
                "gold_label": "neutral",
            }
        )
        + "\n"
    ).encode()
    archive_path = tmp_path / "wrong-train-member.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(snli_training_data.TRAIN_MEMBER, train_bytes)
        archive.writestr(snli_training_data.DEV_MEMBER, dev_bytes)
    monkeypatch.setattr(
        snli_training_data,
        "SNLI_ARCHIVE_SHA256",
        hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        snli_training_data, "SNLI_DEV_MEMBER_SHA256", hashlib.sha256(dev_bytes).hexdigest()
    )
    monkeypatch.setattr(snli_training_data, "SNLI_TRAIN_MEMBER_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="train member SHA-256 mismatch"):
        snli_training_data.read_snli_sources(archive_path)


def test_training_record_recast_preserves_decision_contract() -> None:
    from reflex_decisions import snli_training_data

    item = broader_data.SnliItem(
        source_item_id="train-pair-1",
        source_group_id="train-group-1",
        premise="Premise.",
        hypothesis="Hypothesis.",
        label="entailment",
    )

    dev_record = broader_data.record_from_snli_item(item)
    training_record = snli_training_data.record_from_snli_training_item(item)

    assert training_record.dataset_id == snli_training_data.TRAIN_DATASET_ID
    assert training_record.record_id == dev_record.record_id.replace(
        f"{broader_data.SNLI_DATASET_ID}-", f"{snli_training_data.TRAIN_DATASET_ID}-", 1
    )
    assert training_record.source_group_id == dev_record.source_group_id
    assert training_record.request == dev_record.request
    assert training_record.answer_id == dev_record.answer_id
