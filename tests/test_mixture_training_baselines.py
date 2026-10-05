from itertools import permutations

import pytest

from experiments import mixture_training_baselines as baselines
from experiments import real_pilot_core
from experiments.mixture_training_contracts import (
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_VERSION_PINS,
)
from reflex_decisions import pilot_data


def _base_identity(versions: dict[str, str]) -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_class": "Qwen3_5ForCausalLM",
        "config_class": "Qwen3_5TextConfig",
        "layer_count": 24,
        "tied_embeddings": True,
        "no_meta_parameters": True,
        "load_diagnostics": {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
        "tokenizer_file_sha256": {"tokenizer.json": "a" * 64},
        "effective_dtype": "torch.bfloat16",
        "attention_implementation": "eager",
        "use_kernels": False,
        "use_hub_kernels": "NO",
        "versions": versions,
    }


def test_historical_base_identity_accepts_its_eight_frozen_package_pins() -> None:
    versions = {name: version for name, version in RUNTIME_VERSION_PINS.items() if name != "Pillow"}

    identity = baselines._base_identity(_base_identity(versions), "historical fixture")

    assert identity["versions"] == versions


def test_initialization_runtime_still_requires_all_nine_package_pins() -> None:
    provenance = {"versions": dict(RUNTIME_VERSION_PINS)}
    baselines._runtime_versions(provenance, "initialization fixture")

    with pytest.raises(ValueError, match="nine frozen pins"):
        baselines._runtime_versions(
            {
                "versions": {
                    key: value for key, value in RUNTIME_VERSION_PINS.items() if key != "Pillow"
                }
            },
            "initialization fixture",
        )


def test_real_reuse_subset_drops_only_the_five_non_original_snli_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = ("gold", "wrong-a", "wrong-b")
    expected = {
        "presentation_id": "current-original",
        "record_id": "snli-record",
        "dataset_id": pilot_data.SNLI_DATASET_ID,
        "request_hash": "request-hash",
        "order_ids": list(order),
    }
    outputs = [
        {
            "presentation_id": f"saved-{index}",
            "record_id": "snli-record",
            "dataset_id": pilot_data.SNLI_DATASET_ID,
            "request_hash": "request-hash",
            "order_ids": list(permutation),
        }
        for index, permutation in enumerate(permutations(order))
    ]
    outputs[0] = {**outputs[0], "presentation_id": "current-original"}
    monkeypatch.setattr(real_pilot_core, "EVALUATION_PRESENTATION_COUNT", len(outputs))
    monkeypatch.setattr(
        real_pilot_core,
        "DATASET_RECORD_COUNTS",
        {pilot_data.SNLI_DATASET_ID: 1},
    )

    selected = baselines._select_real_reuse_subset(outputs, [expected])

    assert [row["presentation_id"] for row in selected] == ["current-original"]


def test_real_reuse_subset_rejects_unapproved_semantic_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = ("gold", "wrong-a", "wrong-b")
    expected = {
        "presentation_id": "current-original",
        "record_id": "snli-record",
        "dataset_id": pilot_data.SNLI_DATASET_ID,
        "request_hash": "request-hash",
        "order_ids": list(order),
    }
    outputs = [
        {
            "presentation_id": f"saved-{index}",
            "record_id": "snli-record",
            "dataset_id": pilot_data.SNLI_DATASET_ID,
            "request_hash": "request-hash",
            "order_ids": list(permutation),
        }
        for index, permutation in enumerate(permutations(order))
    ]
    outputs[0] = {**outputs[0], "presentation_id": "current-original"}
    outputs[1] = {**outputs[1], "request_hash": "different-request"}
    monkeypatch.setattr(real_pilot_core, "EVALUATION_PRESENTATION_COUNT", len(outputs))
    monkeypatch.setattr(
        real_pilot_core,
        "DATASET_RECORD_COUNTS",
        {pilot_data.SNLI_DATASET_ID: 1},
    )

    with pytest.raises(ValueError, match="not approved SNLI orders"):
        baselines._select_real_reuse_subset(outputs, [expected])
