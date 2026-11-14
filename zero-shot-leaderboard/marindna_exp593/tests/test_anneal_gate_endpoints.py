import numpy as np
import pytest

from exp593_anneal_evals.gate import (
    binary_auroc,
    binary_average_precision,
    endpoint_diagnostics,
)
from exp593_anneal_evals.scoring import sv_reduce


def test_auroc_matches_pairwise_wins_with_ties():
    assert binary_auroc([0, 1, 0, 1], [0, 0.5, 0.5, 1]) == 0.875
    assert binary_auroc([0, 1], [1, 1]) == 0.5
    assert binary_auroc([1, 1], [0, 1]) is None


def test_average_precision_treats_tied_scores_as_one_threshold():
    assert binary_average_precision([1, 0, 1, 0], [1, 0.5, 0.5, 0]) == pytest.approx(
        5 / 6
    )
    assert binary_average_precision([0, 1], [1, 1]) == 0.5


def test_sv_keeps_unknown_and_skipped_terms_in_the_original_ten_term_mean():
    sequence = "a" * 4093 + "n" + "a" * (8192 - 4094)
    rows = [{"RefSeq": "a" * 8192, "MutSeq": sequence, "left": 6, "right": 20}]
    probabilities = np.zeros((1, 2, 10, 4))
    probabilities[:, 0] = [0.1, 0.3, 0.3, 0.3]
    probabilities[:, 1] = [0.2, 0.3, 0.3, 0.2]
    result = sv_reduce(
        rows, {"forward": probabilities, "reverse_complement": probabilities}
    )
    # Ref position0 and the unknown ALT base each contribute zero. The other
    # eight terms are log(0.2/0.1); the denominator stays ten.
    assert result["forward__scores"][0] == pytest.approx(-0.8 * np.log(2))
    np.testing.assert_allclose(
        result["forward_rc_product__scores"], result["forward__scores"]
    )


def test_actual_motif_endpoint_aligns_both_positions_and_complements():
    p = np.array([[0.6, 0.2, 0.1, 0.1], [0.2, 0.6, 0.1, 0.1]])
    rc = p[::-1][:, [3, 2, 1, 0]]
    common = {
        "task": "motif_maize_test",
        "presentation": "fim",
        "example": 0,
        "task_kind": "motif",
        "label": None,
    }
    forward = ({**common, "strand": "fwd"}, p, p, np.array([0, 1]))
    reverse = ({**common, "strand": "rc"}, rc, rc, np.array([2, 3]))
    result = endpoint_diagnostics([forward, reverse])
    assert len(result) == 3
    assert all(r["native"] == r["deployed"] == 1 for r in result.values())
    with pytest.raises(ValueError, match="lacks a strand"):
        endpoint_diagnostics([forward])
