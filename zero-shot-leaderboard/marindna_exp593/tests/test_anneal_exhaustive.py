import numpy as np
import pytest

from exp593_anneal_evals.exhaustive import (
    align_reverse,
    candidates,
    combine,
    expand,
    marginal_probabilities,
    observed_scores,
    predictions,
    reverse_complement,
)


@pytest.mark.parametrize("length,prefix_count", [(1, 1), (2, 5), (3, 21)])
def test_complete_prefix_tree_never_uses_an_observed_answer(length, prefix_count):
    calls = []

    def reader(requests):
        calls.extend(requests)
        return np.tile(np.log([0.1, 0.2, 0.3, 0.4]), (len(requests), 1))

    actual = expand([[99], [98]], length, (1, 2, 3, 4), reader)
    expected = np.array(
        [
            np.prod([(0.1, 0.2, 0.3, 0.4)["acgt".index(b)] for b in m])
            for m in candidates(length)
        ]
    )
    np.testing.assert_allclose(np.exp(actual), np.tile(expected, (2, 1)))
    assert len(calls) == 2 * prefix_count
    assert predictions(actual, length) == ["t" * length] * 2
    assert all(r[0] in (98, 99) and set(r[1:]) <= {1, 2, 3, 4} for r in calls)


def test_exhaustive_map_can_disagree_with_greedy():
    def reader(requests):
        result = []
        for row in requests:
            p = (
                [0.51, 0.47, 0.01, 0.01]
                if len(row) == 1
                else ([0.25] * 4 if row[-1] != 2 else [0.97, 0.01, 0.01, 0.01])
            )
            result.append(np.log(p))
        return np.array(result)

    actual = expand([[99]], 2, (1, 2, 3, 4), reader)
    assert predictions(actual, 2) == ["ca"]  # Greedy would choose aa.
    assert len(candidates(2)) == 16 and len(candidates(3)) == 64


@pytest.mark.parametrize("length", [1, 2, 3])
def test_reverse_alignment_and_sequence_geometric_mean(length):
    rng = np.random.default_rng(472)
    a = rng.dirichlet(np.ones(4**length), 3)
    b = rng.dirichlet(np.ones(4**length), 3)
    aligned = align_reverse(np.log(b), length)
    expected = np.sqrt(a * np.exp(aligned))
    expected /= expected.sum(-1, keepdims=True)
    np.testing.assert_allclose(np.exp(combine(np.log(a), np.log(b), length)), expected)
    np.testing.assert_allclose(align_reverse(aligned, length), np.log(b))
    assert all(
        reverse_complement(reverse_complement(m)) == m for m in candidates(length)
    )


def test_joint_and_marginal_scores_are_distinct_and_normalized():
    p = np.full((1, 16), 0.01)
    p[0, 0] = 0.35
    p[0, 1] = 0.25
    p /= p.sum(-1, keepdims=True)
    logs = np.log(p)
    marginals = marginal_probabilities(logs, 2)
    scores = observed_scores(logs, ["aa"])
    assert scores["joint"][0] == pytest.approx(p[0, 0])
    assert scores["mean_marginal"][0] == pytest.approx(
        (p[0, :4].sum() + p[0, ::4].sum()) / 2
    )
    assert scores["joint"][0] != scores["mean_marginal"][0]
    np.testing.assert_allclose(marginals.sum(-1), 1)
    assert observed_scores(logs, ["an"])["joint"][0] == 0


def test_uppercase_or_full_vocabulary_mass_cannot_enter_the_tree():
    with pytest.raises(ValueError, match="Normalize over lowercase"):
        expand(
            [[99]], 2, (1, 2, 3, 4), lambda rows: np.full((len(rows), 4), np.log(0.1))
        )
    with pytest.raises(ValueError):
        candidates(4)
