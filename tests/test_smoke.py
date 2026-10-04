import pytest

from reflex_decisions import smoke


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]


def test_synthetic_requests_are_fixed_and_include_reversed_semantics() -> None:
    requests = smoke.synthetic_requests()

    assert tuple(request.request_hash for request in requests) == tuple(
        request.request_hash for request in smoke.synthetic_requests()
    )
    assert len(requests) == 3
    assert tuple(option.id for option in requests[0].options) == ("hold", "close")
    assert tuple(option.id for option in requests[2].options) == ("close", "hold")
    assert len(requests[1].options) == 16
    compiled = [smoke.compile_smoke_request(request, CharacterTokenizer()) for request in requests]
    assert len({len(item.input_ids) for item in compiled[:2]}) == 2
    assert compiled[0].symbol_to_option_id == (("A", "hold"), ("B", "close"))


def test_left_padding_keeps_last_tokens_real_and_positions_masked() -> None:
    padded = smoke.left_pad_inputs(((11, 12, 13), (21, 22)), pad_token_id=0)

    assert padded.input_ids == ((11, 12, 13), (0, 21, 22))
    assert padded.attention_mask == ((1, 1, 1), (0, 1, 1))
    assert padded.position_ids == ((0, 1, 2), (0, 0, 1))
    assert all(mask[-1] == 1 for mask in padded.attention_mask)


def test_candidate_parity_reports_raw_differences_and_fixed_tolerances() -> None:
    request = smoke.synthetic_requests()[0]
    parity = smoke.compare_candidate_logits(request, (1.0, 0.0), (1.05, 0.0))

    assert parity.passed is True
    assert parity.logit_differences == pytest.approx((0.05, 0.0))
    assert parity.max_abs_logit_difference == pytest.approx(0.05)
    assert parity.max_abs_probability_difference < smoke.PROBABILITY_TOLERANCE
    assert parity.top_id_agreement is True
    assert parity.logit_tolerance == smoke.LOGIT_TOLERANCE


def test_candidate_parity_fails_probability_tolerance_and_nonfinite_logits() -> None:
    request = smoke.synthetic_requests()[0]
    probability_failure = smoke.compare_candidate_logits(request, (0.0, 0.0), (0.1, -0.1))
    nonfinite_failure = smoke.compare_candidate_logits(request, (float("nan"), 0.0), (0.0, 0.0))

    assert probability_failure.passed is False
    assert "probability_tolerance_exceeded" in probability_failure.failure_reasons
    assert probability_failure.max_abs_logit_difference <= smoke.LOGIT_TOLERANCE
    assert nonfinite_failure.passed is False
    assert "non_finite_or_invalid_logit" in nonfinite_failure.failure_reasons
    assert nonfinite_failure.logit_differences == (None, 0.0)


def test_artifact_writer_is_atomic_refuses_overwrite_and_rejects_nan(tmp_path) -> None:
    import json

    output = tmp_path / "result.json"
    with smoke.reserve_output(output) as reservation:
        smoke.write_json_artifact(reservation, {"schema_version": 1, "status": "passed"})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "status": "passed",
    }
    with pytest.raises(FileExistsError):
        with smoke.reserve_output(output):
            raise AssertionError("an existing destination must not be reserved")

    invalid_output = tmp_path / "invalid.json"
    with smoke.reserve_output(invalid_output) as reservation:
        with pytest.raises(ValueError):
            smoke.write_json_artifact(reservation, {"bad": float("nan")})
    assert not invalid_output.exists()
    assert not (tmp_path / ".invalid.json.reflex-smoke.lock").exists()


def test_remote_exception_message_keeps_safe_context_and_redacts_credentials() -> None:
    message = smoke.sanitize_exception_message(
        RuntimeError(
            "model load failed at https://alice:secret@hf.example/model?token=query#part; "
            "Authorization: Bearer very-long-secret abc123; hf_fakeToken456; "
            "api_key=top-secret; password='more secret'; safe-detail " + "x" * 2500
        )
    )

    assert "model load failed" in message
    assert "https://hf.example/model" in message
    assert "safe-detail" in message
    assert "alice:secret" not in message
    assert "?token=query" not in message
    assert "very-long-secret" not in message
    assert "hf_fakeToken456" not in message
    assert "top-secret" not in message
    assert "more secret" not in message
    assert len(message) <= smoke.MAX_FAILURE_MESSAGE_CHARS
