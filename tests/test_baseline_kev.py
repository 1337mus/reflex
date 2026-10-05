import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import baseline_kev
from reflex_decisions.schema import DecisionRequest, Option


def test_kev_record_preserves_option_order_and_uses_text_not_semantic_ids() -> None:
    assert baseline_kev.AUXILIARY_FORWARD_COUNT == 2
    request = DecisionRequest(
        context="A duplicate charge is reported.",
        question="Which team should handle this?",
        options=(
            Option(id="billing", label="Billing", description="Charges and refunds"),
            Option(id="shipping", label="Shipping"),
        ),
    )

    record = baseline_kev._request_to_kev_record(request)

    assert record == {
        "state": "A duplicate charge is reported.",
        "questions": [
            {
                "instr": "Which team should handle this?",
                "options": ["Billing: Charges and refunds", "Shipping"],
                "label": 0,
            }
        ],
    }


def test_prompt_hash_is_deterministic_over_exact_encoded_token_ids() -> None:
    expected = hashlib.sha256(b"[1,2,3]").hexdigest()

    assert baseline_kev._prompt_sha256((1, 2, 3)) == expected
    assert baseline_kev._prompt_sha256((1, 2, 4)) != expected


def test_kev_source_fetch_is_pinned_to_two_sha_verified_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bytes = {name: f"pinned-{name}".encode() for name in ("model.py", "checkpoint.py")}
    expected_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in source_bytes.items()
    }
    monkeypatch.setattr(baseline_kev, "SOURCE_SHA256", expected_hashes)
    requested_urls: list[str] = []

    def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
        url = request.full_url  # type: ignore[attr-defined]
        requested_urls.append(url)
        return io.BytesIO(source_bytes[url.rsplit("/", maxsplit=1)[-1]])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    source_root = tmp_path / "official-source"

    baseline_kev._fetch_official_sources(source_root)

    expected_root = (
        f"https://raw.githubusercontent.com/jaredpalmer/kev/{baseline_kev.SOURCE_COMMIT}/kev"
    )
    assert requested_urls == [
        f"{expected_root}/model.py",
        f"{expected_root}/checkpoint.py",
    ]
    assert (source_root / "checkpoint-kev_model.py").read_bytes() == source_bytes["model.py"]
    assert (source_root / "checkpoint-kev_checkpoint.py").read_bytes() == source_bytes[
        "checkpoint.py"
    ]


def test_kev_source_fetch_rejects_bytes_before_writing_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(baseline_kev, "SOURCE_SHA256", {"model.py": "0" * 64})

    def fake_urlopen(_request: object, timeout: int) -> io.BytesIO:
        assert timeout == 30
        return io.BytesIO(b"unverified source")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    source_root = tmp_path / "official-source"

    with pytest.raises(RuntimeError, match="pinned Kev source hash mismatch"):
        baseline_kev._fetch_official_sources(source_root)

    assert list(source_root.iterdir()) == []


def test_base_loader_requests_safe_loading_and_rejects_partial_weights() -> None:
    received_options: dict[str, object] = {}

    class ModelClass:
        @classmethod
        def from_pretrained(cls, _name: str, **kwargs: object) -> tuple[object, object]:
            received_options.update(kwargs)
            return object(), {
                "missing_keys": ["layers.0.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    model_module = SimpleNamespace(AutoModelForCausalLM=ModelClass)
    loader_module = SimpleNamespace(LoadOptions=lambda **kwargs: kwargs)

    class Checkpoint:
        def load(self, _device: str, _options: object) -> tuple[str, object]:
            model, _ = model_module.AutoModelForCausalLM.from_pretrained("pinned-base")
            return "tokenizer", model

    original_method = ModelClass.__dict__["from_pretrained"]
    with pytest.raises(RuntimeError, match="loading diagnostics are not clean"):
        baseline_kev._load_official_checkpoint(
            Checkpoint(), loader_module, model_module, SimpleNamespace(float32="float32")
        )

    assert received_options == {
        "output_loading_info": True,
        "use_safetensors": True,
        "use_kernels": False,
        "trust_remote_code": False,
    }
    assert ModelClass.__dict__["from_pretrained"] is original_method


def test_effective_dtype_provenance_uses_loaded_parameter_metadata() -> None:
    class Device:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    class Parameter:
        dtype = "float32"
        device = Device()
        is_meta = False

        def numel(self) -> int:
            return 3

    class Model:
        def parameters(self) -> list[Parameter]:
            return [Parameter(), Parameter()]

        def buffers(self) -> list[object]:
            return []

    dtypes, count, effective_dtype, device = baseline_kev._assert_effective_fp32_cuda(
        Model(), SimpleNamespace(float32="float32")
    )

    assert dtypes == {"float32": 6}
    assert count == 6
    assert effective_dtype == "float32"
    assert device == "cuda:0"


def test_effective_dtype_check_rejects_unmaterialized_buffers() -> None:
    class Parameter:
        dtype = "float32"
        device = SimpleNamespace(type="cuda")
        is_meta = False

        def numel(self) -> int:
            return 1

    class Buffer:
        is_meta = True

    class Model:
        def parameters(self) -> list[Parameter]:
            return [Parameter()]

        def buffers(self) -> list[Buffer]:
            return [Buffer()]

    with pytest.raises(RuntimeError, match="unmaterialized meta"):
        baseline_kev._assert_effective_fp32_cuda(Model(), SimpleNamespace(float32="float32"))


def test_effective_dtype_check_rejects_non_float32_parameters() -> None:
    class Parameter:
        dtype = "bfloat16"
        device = SimpleNamespace(type="cuda")
        is_meta = False

        def numel(self) -> int:
            return 1

    class Model:
        def parameters(self) -> list[Parameter]:
            return [Parameter()]

        def buffers(self) -> list[object]:
            return []

    with pytest.raises(RuntimeError, match="expected effective float32"):
        baseline_kev._assert_effective_fp32_cuda(Model(), SimpleNamespace(float32="float32"))


def test_backbone_cache_configuration_is_disabled_before_forward() -> None:
    config = SimpleNamespace(use_cache=True)
    model = SimpleNamespace(lm=SimpleNamespace(config=config))

    effective_use_cache = baseline_kev._disable_backbone_cache(model)

    assert effective_use_cache is False
    assert config.use_cache is False


def test_kev_encoding_rejects_packed_input_over_limit() -> None:
    encoded = {"ids": list(range(2049)), "state_truncated": False}

    with pytest.raises(ValueError, match="exceeds 2048-token limit"):
        baseline_kev._validate_encoded(encoded)


def test_tensor_state_rejects_same_shape_but_different_values() -> None:
    class Tensor:
        def __init__(self, values: tuple[float, ...]) -> None:
            self.values = values
            self.shape = (len(values),)

        def cpu(self) -> "Tensor":
            return self

    class FiniteResult:
        def all(self) -> bool:
            return True

    class FakeTorch:
        @staticmethod
        def isfinite(_tensor: Tensor) -> FiniteResult:
            return FiniteResult()

        @staticmethod
        def equal(left: Tensor, right: Tensor) -> bool:
            return left.values == right.values

    with pytest.raises(RuntimeError, match="tensor values differ"):
        baseline_kev._assert_tensor_state(
            {"weight": Tensor((1.0, 2.0))},
            {"weight": Tensor((1.0, 9.0))},
            "Kev adapter",
            FakeTorch(),  # type: ignore[arg-type]
        )
