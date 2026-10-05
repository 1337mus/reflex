from pathlib import Path

import pytest

from experiments import baseline_intern
from reflex_decisions.schema import DecisionRequest, Option


def test_temperature_matches_exact_published_hub_default():
    assert baseline_intern.TEMPERATURE == 2.747760550703


def test_engine_loader_rejects_published_temperature_drift():
    from types import SimpleNamespace

    class ModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object(), {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    class Engine:
        def __init__(self, **kwargs):
            ModelClass.from_pretrained("checkpoint")

    module = SimpleNamespace(
        DEFAULT_TEMPERATURE=1.0,
        Qwen3_5ForConditionalGeneration=ModelClass,
        DecisionEngine=Engine,
    )
    with pytest.raises(RuntimeError, match="published temperature"):
        baseline_intern._load_engine(module, Path("checkpoint"))


def test_engine_loader_disables_remote_code_and_optional_kernels():
    from types import SimpleNamespace

    observed = {}

    class ModelClass:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            observed.update(kwargs)
            return object(), {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    class Engine:
        def __init__(self, **kwargs):
            self.temperature = kwargs["temperature"]
            ModelClass.from_pretrained("checkpoint")

    module = SimpleNamespace(
        DEFAULT_TEMPERATURE=baseline_intern.TEMPERATURE,
        Qwen3_5ForConditionalGeneration=ModelClass,
        DecisionEngine=Engine,
    )
    engine, _ = baseline_intern._load_engine(module, Path("checkpoint"))

    assert engine.temperature == baseline_intern.TEMPERATURE
    assert observed == {
        "output_loading_info": True,
        "use_safetensors": True,
        "use_kernels": False,
        "trust_remote_code": False,
    }


def test_calibration_parity_provenance_changes_only_after_a_successful_check():
    provenance = {"calibration_parity_passed": False, "auxiliary_forward_count": 0}
    check = getattr(baseline_intern, "_record_calibration_parity", None)
    assert callable(check), "adapter must record calibration parity after verifying it"

    check(provenance, [0.7, 0.3], ["choice1", "choice2"], {"choice1": 0.7, "choice2": 0.3})
    assert provenance["calibration_parity_passed"] is True
    assert provenance["auxiliary_forward_count"] == 1

    failed = {"calibration_parity_passed": False, "auxiliary_forward_count": 0}
    with pytest.raises(RuntimeError, match="temperature check disagrees"):
        check(failed, [0.7, 0.3], ["choice1", "choice2"], {"choice1": 0.1, "choice2": 0.9})
    assert failed["calibration_parity_passed"] is False
    assert failed["auxiliary_forward_count"] == 0


def test_build_official_request_preserves_order_and_exposes_semantic_ids():
    request = DecisionRequest(
        context="Customer asks about a duplicate charge.",
        question="Which team should handle this?",
        options=(
            Option(id="billing", label="Billing", description="Charges and refunds"),
            Option(id="shipping", label="Shipping"),
        ),
    )

    builder = getattr(baseline_intern, "build_official_request", None)
    assert callable(builder), "adapter must build the official Intern-Decision request"
    row = builder(request)

    assert row == {
        "state": request.context,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": request.question,
                "criteria": {
                    "billing": "Billing: Charges and refunds",
                    "shipping": "Shipping",
                },
            }
        },
    }


def test_module_hash_check_rejects_modified_inference_source(tmp_path):
    source = tmp_path / "inference.py"
    source.write_text("changed official source", encoding="utf-8")

    verifier = getattr(baseline_intern, "verify_module_sha256", None)
    assert callable(verifier), "adapter must pin the official inference module"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verifier(source)


def test_loading_diagnostics_reject_missing_checkpoint_parameters():
    validator = getattr(baseline_intern, "validate_loading_info", None)
    assert callable(validator), "adapter must inspect Transformers loading diagnostics"

    with pytest.raises(RuntimeError, match="missing_keys"):
        validator(
            {
                "missing_keys": ["model.language_model.embed_tokens.weight"],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }
        )


def test_candidate_symbols_must_be_single_token_and_distinct():
    mapper = getattr(baseline_intern, "checked_candidate_token_ids", None)
    assert callable(mapper), "adapter must validate one-to-one symbol token mapping"

    class ToyTokenizer:
        def __init__(self, mapping):
            self.mapping = mapping

        def encode(self, symbol, *, add_special_tokens=False):
            assert add_special_tokens is False
            return self.mapping[symbol]

    assert mapper(ToyTokenizer({"A": [10], "B": [11]}), ("A", "B"), 2, 20) == [10, 11]
    with pytest.raises(ValueError, match="exactly one token"):
        mapper(ToyTokenizer({"A": [10], "B": [11, 12]}), ("A", "B"), 2, 20)
    with pytest.raises(ValueError, match="distinct"):
        mapper(ToyTokenizer({"A": [10], "B": [10]}), ("A", "B"), 2, 20)
