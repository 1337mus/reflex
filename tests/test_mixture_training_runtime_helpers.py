from __future__ import annotations

import hashlib
import json
import math
import struct

import pytest

from experiments import mixture_training_runtime_helpers as helpers
from experiments.mixture_training_runtime_helpers import (
    tensor_state_sha256,
    write_json_atomic,
    write_json_exclusive,
)


class _Array:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def tobytes(self, *, order: str) -> bytes:
        assert order == "C"
        return struct.pack("=" + "f" * len(self.values), *self.values)


class _Tensor:
    def __init__(
        self, values: tuple[float, ...], shape: tuple[int, ...], dtype: str = "float32"
    ) -> None:
        self.values = values
        self.shape = shape
        self.dtype = dtype

    def detach(self) -> _Tensor:
        return self

    def to(self, *, device: str) -> _Tensor:
        assert device == "cpu"
        return self

    def contiguous(self) -> _Tensor:
        return self

    def numpy(self) -> _Array:
        return _Array(self.values)


class _Torch:
    float32 = "float32"

    @staticmethod
    def isfinite(tensor: _Tensor) -> _FiniteMask:
        return _FiniteMask(all(math.isfinite(value) for value in tensor.values))


class _FiniteMask:
    def __init__(self, finite: bool) -> None:
        self.finite = finite

    def all(self) -> _FiniteMask:
        return self

    def item(self) -> bool:
        return self.finite


def test_tensor_state_digest_is_order_independent_and_binds_shape_dtype_and_bytes() -> None:
    left = {
        "z": _Tensor((3.0,), (1,)),
        "a": _Tensor((1.0, -2.0), (2,)),
    }
    reordered = {"a": left["a"], "z": left["z"]}

    digest = tensor_state_sha256(left, torch_module=_Torch)

    assert digest == tensor_state_sha256(reordered, torch_module=_Torch)
    assert digest != tensor_state_sha256(
        {"z": _Tensor((3.0,), (1,)), "a": _Tensor((1.0, -3.0), (2,))},
        torch_module=_Torch,
    )

    metadata = json.dumps(
        {"dtype": "float32", "name": "a", "shape": [2]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected = hashlib.sha256()
    raw = struct.pack("=" + "f" * 2, 1.0, -2.0)
    for chunk in (metadata, raw):
        expected.update(len(chunk).to_bytes(8, "big"))
        expected.update(chunk)
    metadata_z = json.dumps(
        {"dtype": "float32", "name": "z", "shape": [1]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    raw_z = struct.pack("=f", 3.0)
    for chunk in (metadata_z, raw_z):
        expected.update(len(chunk).to_bytes(8, "big"))
        expected.update(chunk)
    assert digest == expected.hexdigest()


def test_output_artifact_hash_covers_canonical_bytes_and_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "outputs.json"
    value = {"z": 1, "a": "µ"}

    digest = write_json_exclusive(path, value)

    stored = b'{"a":"\xc2\xb5","z":1}\n'
    assert path.read_bytes() == stored
    assert digest == hashlib.sha256(stored).hexdigest()
    try:
        write_json_exclusive(path, {"replacement": True})
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output artifact was overwritten")
    assert path.read_bytes() == stored


def test_atomic_json_write_creates_and_replaces_progress(tmp_path) -> None:
    path = tmp_path / "progress.json"

    first_digest = write_json_atomic(path, {"step": 1}, replace=True)
    first_bytes = b'{"step":1}\n'
    assert path.read_bytes() == first_bytes
    assert first_digest == hashlib.sha256(first_bytes).hexdigest()

    second_digest = write_json_atomic(path, {"step": 2}, replace=True)
    second_bytes = b'{"step":2}\n'
    assert path.read_bytes() == second_bytes
    assert second_digest == hashlib.sha256(second_bytes).hexdigest()


def test_exclusive_json_write_does_not_rely_on_hard_links(tmp_path, monkeypatch) -> None:
    path = tmp_path / "output.json"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("exclusive output creation must not require hard-link support")

    monkeypatch.setattr(helpers.os, "link", fail_if_called)

    write_json_exclusive(path, {"complete": True})

    assert path.read_bytes() == b'{"complete":true}\n'


def test_failed_progress_replace_preserves_existing_evidence(tmp_path, monkeypatch) -> None:
    path = tmp_path / "progress.json"
    existing = b'{"status":"passed"}\n'
    path.write_bytes(existing)

    def fail_replace(_source, _destination):
        raise OSError("volume replace failed")

    monkeypatch.setattr(helpers.os, "replace", fail_replace)
    with pytest.raises(OSError, match="volume replace failed"):
        write_json_atomic(path, {"status": "in_progress"}, replace=True)

    assert path.read_bytes() == existing
    assert not list(tmp_path.glob(".progress.json.*.tmp"))


def test_tensor_state_digest_rejects_non_fp32_tensors() -> None:
    with pytest.raises(ValueError, match="FP32"):
        tensor_state_sha256(
            {"adapter": _Tensor((1.0,), (1,), dtype="bfloat16")},
            torch_module=_Torch,
        )


def test_tensor_state_digest_rejects_non_finite_fp32_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        tensor_state_sha256(
            {"adapter": _Tensor((float("nan"),), (1,))},
            torch_module=_Torch,
        )
