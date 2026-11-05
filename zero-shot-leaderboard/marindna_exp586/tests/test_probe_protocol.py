"""The hidden DNA region must never enter a probe's feature inputs."""

import numpy as np
import pytest

from exp586_evals.probe_protocol import (
    linear_features,
    reverse_complement,
    visible_sequences,
)


@pytest.mark.parametrize("positions", [[4095], [4095, 4096], [4094, 4095, 4096]])
def test_target_counterfactual_and_exact_boundaries(positions):
    sequence = "ACGT" * 2048
    spec = {"kind": "core", "positions": positions}
    changed = list(sequence)
    for position in positions:
        changed[position] = "T" if changed[position] != "T" else "A"
    original = visible_sequences(spec, {"sequence": sequence, "label": 0})
    assert original == visible_sequences(
        spec, {"sequence": "".join(changed), "label": 1}
    )
    assert original["fwd"] == sequence[: min(positions)]
    assert original["rc"] == reverse_complement(sequence[max(positions) + 1 :])
    assert sum(map(len, original.values())) == 8192 - len(positions)
    changed[min(positions) - 1] = "N"
    assert original != visible_sequences(spec, {"sequence": "".join(changed)})


def test_sv_visibility_is_allele_input_and_never_label():
    row = {"RefSeq": "A" * 8192, "MutSeq": "C" * 8192}
    spec = {"kind": "sv"}
    assert visible_sequences(spec, {**row, "label": 0}) == visible_sequences(
        spec, {**row, "label": 1}
    )
    assert visible_sequences(spec, row)["alt_rc"] == "G" * 8192


def test_af_feature_convention_is_preserved():
    arrays = {
        name: np.full((3, 4), i, dtype=np.float32)
        for i, name in enumerate(["ref_fwd", "ref_rc", "alt_fwd", "alt_rc"])
    }
    avg = linear_features(arrays, kind="sv", directional=False)
    np.testing.assert_array_equal(avg[:, :4], 0.5)
    np.testing.assert_array_equal(avg[:, 4:], 2)
    directional = linear_features(arrays, kind="sv", directional=True)
    np.testing.assert_array_equal(directional[0], [0] * 4 + [2] * 4 + [1] * 4 + [2] * 4)


def test_invalid_window_and_noncontiguous_targets_fail_closed():
    with pytest.raises(ValueError):
        visible_sequences(
            {"kind": "core", "positions": [4094, 4096]}, {"sequence": "A" * 8192}
        )
    with pytest.raises(ValueError):
        visible_sequences(
            {"kind": "conservation", "positions": [0]}, {"sequence": "A" * 8192}
        )


def test_iupac_ambiguity_is_preserved_and_complemented_without_replacement():
    assert reverse_complement("ACGTRYSWKMBDHVN") == "NBDHVKMWSRYACGT"
    assert reverse_complement(reverse_complement("ACGTRYSWKMBDHVN")) == "ACGTRYSWKMBDHVN"
    assert reverse_complement("awky") == "rmwt"
