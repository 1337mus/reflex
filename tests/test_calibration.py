import math

import pytest

from reflex_decisions import calibration


def test_overconfident_mixed_correctness_improves_nll() -> None:
    examples = (
        calibration.CalibrationExample((4.0, 0.0), gold_index=0),
        calibration.CalibrationExample((4.0, 0.0), gold_index=1),
        calibration.CalibrationExample((4.0, 0.0), gold_index=0),
    )

    result = calibration.fit_temperature(examples)

    assert result.temperature > 1.0
    assert result.fitted_nll < result.raw_nll


def test_perfectly_correct_examples_select_lower_grid_boundary() -> None:
    result = calibration.fit_temperature(
        (
            calibration.CalibrationExample((0.5, 0.0), gold_index=0),
            calibration.CalibrationExample((0.5, 0.0), gold_index=0),
        )
    )

    assert result.temperature == result.grid_range[0]
    assert result.boundary_hit


def test_flat_logits_prefer_temperature_one_on_equal_nll() -> None:
    result = calibration.fit_temperature(
        (calibration.CalibrationExample((3.0, 3.0, 3.0), gold_index=2),)
    )

    assert result.temperature == 1.0
    assert result.fitted_nll == result.raw_nll
    assert result.grid_count == 82


@pytest.mark.parametrize("gold_index", (True, 1.0, -1, 2))
def test_rejects_invalid_gold_index(gold_index: object) -> None:
    example = calibration.CalibrationExample((1.0, 0.0), gold_index=gold_index)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gold_index"):
        calibration.fit_temperature((example,))


@pytest.mark.parametrize("weight", (0.0, -1.0, math.inf, math.nan, True, "1"))
def test_rejects_nonpositive_or_nonfinite_weight(weight: object) -> None:
    example = calibration.CalibrationExample(
        (1.0, 0.0),
        gold_index=0,
        weight=weight,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="weight"):
        calibration.fit_temperature((example,))


@pytest.mark.parametrize(
    "logits",
    (
        (),
        (1.0,),
        tuple(float(index) for index in range(17)),
        (math.nan, 0.0),
        (math.inf, 0.0),
        (True, 0.0),
        ("1", 0.0),
    ),
)
def test_rejects_invalid_logits(logits: tuple[object, ...]) -> None:
    example = calibration.CalibrationExample(logits=logits, gold_index=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="logit"):
        calibration.fit_temperature((example,))


def test_rejects_empty_examples_and_unrepresentable_logit_differences() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibration.fit_temperature(())
    extreme = calibration.CalibrationExample((1e308, -1e308), gold_index=0)
    with pytest.raises(ValueError, match="not representable"):
        calibration.fit_temperature((extreme,))


def test_common_logit_shift_and_candidate_permutation_preserve_fit() -> None:
    original = (
        calibration.CalibrationExample((4.0, 1.0, -1.0), gold_index=0, weight=2.0),
        calibration.CalibrationExample((0.0, 3.0, 1.0), gold_index=1, weight=0.5),
    )
    shifted_and_permuted = (
        calibration.CalibrationExample((999.0, 1001.0, 1004.0), gold_index=2, weight=2.0),
        calibration.CalibrationExample((1001.0, 1003.0, 1000.0), gold_index=1, weight=0.5),
    )

    original_result = calibration.fit_temperature(original)
    transformed_result = calibration.fit_temperature(shifted_and_permuted)

    assert transformed_result.temperature == original_result.temperature
    assert transformed_result.raw_nll == original_result.raw_nll
    assert transformed_result.fitted_nll == original_result.fitted_nll


def test_fit_keeps_input_sequences_unchanged_and_snapshots_logits() -> None:
    logits = [4.0, 0.0]
    example = calibration.CalibrationExample(logits, gold_index=0)  # type: ignore[arg-type]
    examples = [example]

    result = calibration.fit_temperature(examples)

    assert logits == [4.0, 0.0]
    assert example.logits == (4.0, 0.0)
    assert examples == [example]
    assert result.example_count == 1


def test_weighted_example_matches_equivalent_duplicate_rows() -> None:
    weighted = (
        calibration.CalibrationExample((3.0, 0.0), gold_index=0, weight=2.0),
        calibration.CalibrationExample((0.5, 0.0), gold_index=1, weight=1.0),
    )
    duplicated = (
        calibration.CalibrationExample((3.0, 0.0), gold_index=0, weight=1.0),
        calibration.CalibrationExample((3.0, 0.0), gold_index=0, weight=1.0),
        calibration.CalibrationExample((0.5, 0.0), gold_index=1, weight=1.0),
    )

    weighted_result = calibration.fit_temperature(weighted)
    duplicated_result = calibration.fit_temperature(duplicated)

    assert duplicated_result.temperature == weighted_result.temperature
    assert duplicated_result.raw_nll == weighted_result.raw_nll
    assert duplicated_result.fitted_nll == weighted_result.fitted_nll
