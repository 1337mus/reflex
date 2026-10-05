from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments import adapter_transfer_contracts, runtime_rule_study_sources


def _write_source_tree(root: Path) -> dict[str, bytes]:
    contents = {
        path: f"self-authored:{path}\n".encode() for path in runtime_rule_study_sources.SOURCE_PATHS
    }
    for relative, content in contents.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return contents


def _commit_source_tree(root: Path) -> str:
    subprocess.run(["git", "init", "--quiet"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture Author"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
    subprocess.run(
        ["git", "add", "--", *runtime_rule_study_sources.SOURCE_PATHS],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture sources"],
        cwd=root,
        check=True,
    )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_source_packaging_helper_module_exists() -> None:
    module_path = Path(__file__).parents[1] / "experiments/runtime_rule_study_sources.py"

    assert module_path.is_file()


def test_source_paths_are_the_sorted_exact_reviewed_members() -> None:
    new_modules = {
        f"experiments/runtime_rule_study_{name}.py"
        for name in (
            "data",
            "inputs",
            "tokens",
            "cache",
            "statistics",
            "contracts",
            "bundle",
            "payloads",
            "payload_rows",
            "payload_metadata",
            "progress",
            "training_evidence",
            "provenance",
            "output_evidence",
            "results",
            "runtime_scoring",
            "runtime_training",
            "sources",
            "runtime",
            "launch",
            "provider",
            "authority",
            "receipts",
        )
    }
    expected = {
        *adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS,
        *new_modules,
        "docs/runtime-rule-study-protocol.md",
        "pyproject.toml",
        "uv.lock",
        "experiments/modal_runtime_rule_study.py",
    }

    assert len(adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS) == 60
    assert runtime_rule_study_sources.SOURCE_PATHS == tuple(sorted(expected))


def test_validate_source_map_returns_the_complete_map_in_path_order() -> None:
    expected = {
        path: f"{index:064x}"
        for index, path in enumerate(runtime_rule_study_sources.SOURCE_PATHS, start=1)
    }
    reversed_map = dict(reversed(tuple(expected.items())))

    assert runtime_rule_study_sources.validate_source_map(reversed_map) == expected


def test_validate_source_map_rejects_malformed_membership_and_digests() -> None:
    valid = {
        path: f"{index:064x}"
        for index, path in enumerate(runtime_rule_study_sources.SOURCE_PATHS, start=1)
    }
    malformed_maps = (
        {
            path: digest
            for path, digest in valid.items()
            if path != runtime_rule_study_sources.SOURCE_PATHS[0]
        },
        {**valid, "experiments/unlisted.py": "0" * 64},
        {**valid, 1: "0" * 64},
        {**valid, runtime_rule_study_sources.SOURCE_PATHS[0]: "A" * 64},
        {**valid, runtime_rule_study_sources.SOURCE_PATHS[0]: "0" * 63},
    )

    for malformed in malformed_maps:
        with pytest.raises(ValueError):
            runtime_rule_study_sources.validate_source_map(malformed)


def test_validate_source_map_applies_the_reviewed_path_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_rule_study_sources, "SOURCE_PATHS", ("data/private.txt",))

    with pytest.raises(ValueError, match="outside the fixed upload file classes"):
        runtime_rule_study_sources.validate_source_map({"data/private.txt": "a" * 64})


def test_measure_sources_hashes_only_the_explicit_files(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    contents = _write_source_tree(root)
    (root / "unlisted-neighbor.json").write_text("fixture only", encoding="utf-8")
    expected = {
        path: hashlib.sha256(contents[path]).hexdigest()
        for path in runtime_rule_study_sources.SOURCE_PATHS
    }

    measured = runtime_rule_study_sources.measure_sources(root)

    assert measured == expected
    assert tuple(measured) == runtime_rule_study_sources.SOURCE_PATHS


def test_measure_sources_rejects_missing_nonfile_and_symlink_paths(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    contents = _write_source_tree(root)
    target = root / "experiments/runtime_rule_study_sources.py"
    outside = tmp_path / "outside.py"
    outside.write_bytes(contents["experiments/runtime_rule_study_sources.py"])

    target.unlink()
    with pytest.raises(ValueError):
        runtime_rule_study_sources.measure_sources(root)

    target.mkdir()
    with pytest.raises(ValueError):
        runtime_rule_study_sources.measure_sources(root)
    target.rmdir()

    target.symlink_to(outside)
    with pytest.raises(ValueError):
        runtime_rule_study_sources.measure_sources(root)
    target.unlink()
    target.write_bytes(contents["experiments/runtime_rule_study_sources.py"])

    source_dir = root / "src/reflex_decisions"
    outside_dir = tmp_path / "outside-sources"
    outside_dir.mkdir()
    for relative, content in contents.items():
        if relative.startswith("src/reflex_decisions/"):
            (outside_dir / Path(relative).name).write_bytes(content)
    shutil.rmtree(source_dir)
    source_dir.symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(ValueError):
        runtime_rule_study_sources.measure_sources(root)


def test_freeze_sources_copies_only_the_selected_commit_blobs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    contents = _write_source_tree(root)
    (root / ".gitignore").write_text("ignored-neighbor.json\n", encoding="utf-8")
    commit = _commit_source_tree(root)
    notes = root / "notes.txt"
    notes.write_text("later tracked note", encoding="utf-8")
    subprocess.run(["git", "add", "--", "notes.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "non-source update"],
        cwd=root,
        check=True,
    )
    notes.write_text("working tree note", encoding="utf-8")
    (root / "ignored-neighbor.json").write_text("ignored fixture", encoding="utf-8")
    (root / "untracked-neighbor.json").write_text("untracked fixture", encoding="utf-8")
    (root / "experiments/unlisted_helper.py").write_text("local change", encoding="utf-8")
    destination = tmp_path / "frozen"
    expected = {path: hashlib.sha256(contents[path]).hexdigest() for path in contents}

    source_map = runtime_rule_study_sources.freeze_sources(root, commit, destination)

    copied_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert source_map == expected
    assert copied_paths == set(runtime_rule_study_sources.SOURCE_PATHS)
    assert not (destination / ".git").exists()
    assert all((destination / path).read_bytes() == content for path, content in contents.items())
    assert not (destination / "ignored-neighbor.json").exists()
    assert not (destination / "untracked-neighbor.json").exists()
    assert not (destination / "experiments/unlisted_helper.py").exists()
    assert not (destination / "notes.txt").exists()


def test_freeze_sources_rejects_a_symlinked_source_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_source_tree(root)
    commit = _commit_source_tree(root)
    root_alias = tmp_path / "repo-alias"
    root_alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match="source root"):
        runtime_rule_study_sources.freeze_sources(root_alias, commit, tmp_path / "frozen")


def test_freeze_sources_rejects_a_symlink_blob_in_the_selected_commit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_source_tree(root)
    _commit_source_tree(root)
    relative = "experiments/runtime_rule_study_sources.py"
    link_path = root / relative
    link_path.unlink()
    link_path.symlink_to("adapter_transfer_contracts.py")
    subprocess.run(["git", "add", "--", relative], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "symlink source"],
        cwd=root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    destination = tmp_path / "frozen"

    with pytest.raises(ValueError, match="not a normal tracked file"):
        runtime_rule_study_sources.freeze_sources(root, commit, destination)
    assert not destination.exists()


def test_freeze_sources_rejects_a_submodule_entry_in_the_selected_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _write_source_tree(root)
    _commit_source_tree(root)
    relative = runtime_rule_study_sources.SOURCE_PATHS[0]
    subprocess.run(["git", "rm", "--cached", "--", relative], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{'a' * 40},{relative}",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "gitlink source"],
        cwd=root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    destination = tmp_path / "frozen"

    with pytest.raises(ValueError, match="not a normal tracked file"):
        runtime_rule_study_sources.freeze_sources(root, commit, destination)
    assert not destination.exists()


def test_freeze_sources_rejects_invalid_or_missing_commit_members(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_source_tree(root)
    _commit_source_tree(root)
    destination = tmp_path / "frozen"

    with pytest.raises(ValueError, match="lowercase 40-character"):
        runtime_rule_study_sources.freeze_sources(root, "A" * 40, destination)
    assert not destination.exists()

    missing_path = runtime_rule_study_sources.SOURCE_PATHS[0]
    subprocess.run(["git", "rm", "--cached", "--", missing_path], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "missing member"],
        cwd=root,
        check=True,
    )
    missing_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with pytest.raises(ValueError, match="missing a tracked file"):
        runtime_rule_study_sources.freeze_sources(root, missing_commit, destination)
    assert not destination.exists()


def test_freeze_sources_rejects_local_source_drift(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_source_tree(root)
    commit = _commit_source_tree(root)
    changed_path = runtime_rule_study_sources.SOURCE_PATHS[0]
    (root / changed_path).write_text("drifted bytes", encoding="utf-8")
    destination = tmp_path / "frozen"

    with pytest.raises(ValueError, match="current source bytes differ"):
        runtime_rule_study_sources.freeze_sources(root, commit, destination)
    assert not destination.exists()


def test_freeze_sources_leaves_a_preexisting_destination_untouched(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write_source_tree(root)
    commit = _commit_source_tree(root)
    destination = tmp_path / "frozen"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("preexisting", encoding="utf-8")

    with pytest.raises(ValueError, match="must not already exist"):
        runtime_rule_study_sources.freeze_sources(root, commit, destination)

    assert sentinel.read_text(encoding="utf-8") == "preexisting"
    assert list(destination.iterdir()) == [sentinel]


def test_git_reads_disable_lazy_fetch_and_optional_locks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_environments: list[dict[str, str]] = []

    def fake_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs.get("env")
        if isinstance(environment, dict):
            captured_environments.append(environment)
        return subprocess.CompletedProcess(arguments, 0, stdout=b"fixture")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert runtime_rule_study_sources._run_git(tmp_path, "cat-file", "-t", "a" * 40) == b"fixture"
    assert len(captured_environments) == 1
    assert captured_environments[0].get("GIT_NO_LAZY_FETCH") == "1"
    assert captured_environments[0].get("GIT_OPTIONAL_LOCKS") == "0"
