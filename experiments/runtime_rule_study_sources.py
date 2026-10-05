"""Freeze an exact source bundle for the CPU-side runtime-rule study."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG

from experiments import adapter_transfer_contracts

_STUDY_MODULES = (
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
_STUDY_MODULE_PATHS = tuple(f"experiments/runtime_rule_study_{name}.py" for name in _STUDY_MODULES)
SOURCE_PATHS = tuple(
    sorted(
        {
            *adapter_transfer_contracts.SOURCE_FINGERPRINT_PATHS,
            *_STUDY_MODULE_PATHS,
            "docs/runtime-rule-study-protocol.md",
            "pyproject.toml",
            "uv.lock",
            "experiments/modal_runtime_rule_study.py",
        }
    )
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_PATHS = frozenset({"pyproject.toml", "uv.lock"})


def _validate_metadata_source_path(path: str) -> None:
    """Apply the reviewed upload path classes before exact-membership checks."""

    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(character in path for character in "*?[]")
        or any(part in {"", ".", ".."} or part.startswith(".") for part in parts)
    ):
        raise ValueError("source path is unsafe")
    if path in _CONFIG_PATHS:
        return
    if path.startswith("experiments/") and path.endswith(".py"):
        return
    if path.startswith("src/reflex_decisions/") and path.endswith(".py"):
        return
    if path.startswith("docs/") and path.endswith("protocol.md"):
        return
    raise ValueError("source path is outside the fixed upload file classes")


def validate_source_map(value: object) -> dict[str, str]:
    """Validate exact source membership and lowercase SHA-256 values."""

    if not isinstance(value, Mapping) or any(type(path) is not str for path in value):
        raise ValueError("source map must be a path-to-digest object with string keys")
    if set(value) != set(SOURCE_PATHS):
        raise ValueError("source map differs from the exact reviewed source paths")

    result: dict[str, str] = {}
    for path in SOURCE_PATHS:
        _validate_metadata_source_path(path)
        digest = value[path]
        if type(digest) is not str or not _SHA256.fullmatch(digest):
            raise ValueError(f"source SHA-256 for {path} must be lowercase hexadecimal")
        result[path] = digest
    return result


def measure_sources(root: str | Path) -> dict[str, str]:
    """Hash each exact member of the reviewed source set."""

    source_root = Path(root)
    try:
        root_stat = source_root.lstat()
        if S_ISLNK(root_stat.st_mode) or not S_ISDIR(root_stat.st_mode):
            raise ValueError("source root must be a real directory")
        resolved_root = source_root.resolve(strict=True)
        root_fd = os.open(
            resolved_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise ValueError("source root is unavailable") from exc

    try:
        if not S_ISDIR(os.fstat(root_fd).st_mode):
            raise ValueError("source root must be a real directory")
        hashes = {
            path: hashlib.sha256(_read_source_file(root_fd, path)).hexdigest()
            for path in SOURCE_PATHS
        }
    finally:
        os.close(root_fd)
    return hashes


def _read_source_file(root_fd: int, relative: str) -> bytes:
    """Read one regular file through no-follow directory descriptors."""

    parts = relative.split("/")
    parent_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            try:
                info = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
                if S_ISLNK(info.st_mode) or not S_ISDIR(info.st_mode):
                    raise ValueError(f"source path has a non-directory component: {relative}")
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise ValueError(f"source path component is unavailable: {relative}") from exc
            os.close(parent_fd)
            parent_fd = next_fd

        try:
            info = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if S_ISLNK(info.st_mode) or not S_ISREG(info.st_mode):
                raise ValueError(f"source path is not a regular file: {relative}")
            file_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            raise ValueError(f"source file is unavailable: {relative}") from exc
        try:
            if not S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError(f"source path is not a regular file: {relative}")
            chunks: list[bytes] = []
            while block := os.read(file_fd, 1024 * 1024):
                chunks.append(block)
            return b"".join(chunks)
        finally:
            os.close(file_fd)
    finally:
        os.close(parent_fd)


def _run_git(root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", *arguments],
            cwd=root,
            check=False,
            capture_output=True,
            env=environment,
        )
    except OSError as exc:
        raise ValueError("Git is unavailable for source freezing") from exc
    if result.returncode != 0:
        raise ValueError("Git could not read the selected source commit")
    return result.stdout


def _commit_blob(root: Path, source_commit: str, relative: str) -> bytes:
    listing = _run_git(
        root,
        "ls-tree",
        "-z",
        "--full-tree",
        source_commit,
        "--",
        f":(literal){relative}",
    )
    entries = listing.split(b"\0")
    if entries and entries[-1] == b"":
        entries.pop()
    if len(entries) != 1 or b"\t" not in entries[0]:
        raise ValueError(f"source commit is missing a tracked file: {relative}")
    header, entry_path = entries[0].split(b"\t", maxsplit=1)
    fields = header.split(b" ")
    if (
        len(fields) != 3
        or fields[0] not in {b"100644", b"100755"}
        or fields[1] != b"blob"
        or entry_path != relative.encode()
    ):
        raise ValueError(f"source commit path is not a normal tracked file: {relative}")
    return _run_git(root, "cat-file", "blob", fields[2].decode("ascii"))


def freeze_sources(
    root: str | Path,
    source_commit: str,
    destination: str | Path,
) -> dict[str, str]:
    """Copy exact source blobs from a commit into a new, verified directory."""

    source_root = Path(root)
    try:
        root_stat = source_root.lstat()
        if S_ISLNK(root_stat.st_mode) or not S_ISDIR(root_stat.st_mode):
            raise ValueError("source root must be a real directory")
        resolved_root = source_root.resolve(strict=True)
        top_level = Path(
            _run_git(resolved_root, "rev-parse", "--show-toplevel").decode().strip()
        ).resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ValueError("source root must be the root of a Git worktree") from exc
    if top_level != resolved_root:
        raise ValueError("source root must be the root of a Git worktree")
    if type(source_commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a lowercase 40-character Git SHA")

    destination_path = Path(destination)
    if not destination_path.name or os.path.lexists(destination_path):
        raise ValueError("source destination must not already exist")
    try:
        destination_parent = destination_path.parent.resolve(strict=True)
        if not destination_parent.is_dir():
            raise ValueError("source destination parent must be a directory")
    except OSError as exc:
        raise ValueError("source destination parent is unavailable") from exc
    frozen_root = destination_parent / destination_path.name
    if os.path.lexists(frozen_root):
        raise ValueError("source destination must not already exist")

    if _run_git(resolved_root, "cat-file", "-t", source_commit).strip() != b"commit":
        raise ValueError("source SHA does not identify a commit object")
    blobs = {path: _commit_blob(resolved_root, source_commit, path) for path in SOURCE_PATHS}
    expected = {path: hashlib.sha256(content).hexdigest() for path, content in blobs.items()}
    if measure_sources(resolved_root) != expected:
        raise ValueError("current source bytes differ from the selected Git commit")

    try:
        frozen_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ValueError("source destination must not already exist") from exc
    try:
        for relative, content in blobs.items():
            destination_file = frozen_root / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            with destination_file.open("xb") as output:
                output.write(content)
        copied = measure_sources(frozen_root)
        if copied != expected:
            raise ValueError("frozen source copy differs from the selected Git commit")
        return copied
    except BaseException:
        shutil.rmtree(frozen_root, ignore_errors=True)
        raise
