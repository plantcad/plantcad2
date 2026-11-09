import numpy as np
import pytest

from exp593_evals.readouts import normalize_dna, nucleotide_probabilities


def test_cases_are_combined_before_selecting_a_base():
    # a has the largest individual probability, but the C/c group wins.
    logits = np.log(np.array([0.01, 0.25, 0.01, 0.01, 0.30, 0.25, 0.10, 0.07]))
    result = nucleotide_probabilities(logits, collapse_case=True)
    np.testing.assert_allclose(result, [0.31, 0.50, 0.11, 0.08], atol=1e-6)
    assert result.argmax() == 1
    assert nucleotide_probabilities(logits, collapse_case=False).argmax() == 0


def test_case_grouping_is_not_an_equal_prior_over_cases():
    probabilities = np.array([0.80, 0.01, 0.01, 0.01, 0.01, 0.10, 0.03, 0.03])
    actual = nucleotide_probabilities(np.log(probabilities), collapse_case=True)
    np.testing.assert_allclose(actual, [0.81, 0.11, 0.04, 0.04], atol=1e-6)


def test_ambiguity_normalization_preserves_positions():
    assert normalize_dna("AaCGtNRy", case="lower") == "aacgtnnn"
    assert normalize_dna("AaCGtNRy", case="upper") == "AACGTNNN"
    with pytest.raises(ValueError, match="alphabet"):
        normalize_dna("AC-G", case="lower")
