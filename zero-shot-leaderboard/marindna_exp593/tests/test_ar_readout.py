import numpy as np
import pytest

torch = pytest.importorskip("torch")

from exp593_evals.ar_readout import ARReadout


class ToyScorer:
    """Small independent full-vocabulary forward for readout-interface tests."""

    batch_size = 2
    sequence_length = 8192
    dna_ids = tuple(range(10))
    device = torch.device("cpu")
    full_head = torch.arange(30, dtype=torch.float32).reshape(10, 3) / 30

    def forward(self, ids):
        hidden = torch.ones((len(ids), len(ids[0]), 3)) / 3
        return (
            hidden @ self.full_head[:8].T,
            hidden,
            torch.tensor([len(r) for r in ids]),
        )

    def log_probabilities(self, logits, *, collapse_case):
        values = (
            torch.logaddexp(logits[..., :4], logits[..., 4:])
            if collapse_case
            else logits[..., 4:]
        )
        return torch.log_softmax(values, dim=-1)


def test_ar_case_collapse_and_full_vocab_normalizer_use_the_same_groups():
    scorer = ToyScorer()
    ids = [[4, 5, 8, 6, 7], [4, 5, 8, 6, 7]]
    actual = ARReadout(scorer, case="lower", collapse_case=True).evaluate(
        ids, [[1], [1]], [2, 2], allele_frequency=True
    )
    original = torch.softmax(
        torch.ones((1, 3)) / 3 @ scorer.full_head.T, dim=-1
    ).numpy()[0]
    grouped = original[:4] + original[4:8]
    np.testing.assert_allclose(
        actual["probabilities"][:, 0],
        np.tile(grouped / grouped.sum(), (2, 1)),
        atol=1e-7,
    )
    np.testing.assert_allclose(np.exp(actual["full_logs"][:, 0]), grouped[1], atol=1e-7)
    assert actual["valid"].tolist() == [[True, False, True, True]] * 2
    assert not actual["probabilities"][:, 1:].any()  # Original zero-query sentinel.


def test_lower_control_keeps_the_existing_lowercase_only_normalization():
    scorer = ToyScorer()
    actual = ARReadout(scorer, case="lower", collapse_case=False).evaluate(
        [[4, 5, 6]], [[1]], [1], allele_frequency=True
    )
    _, hidden, _ = scorer.forward([[4, 5, 6]])
    full = torch.log_softmax(hidden @ scorer.full_head.T, dim=-1)
    np.testing.assert_allclose(
        actual["full_logs"],
        full[:, :-1].gather(-1, torch.tensor([[[5], [6]]])).squeeze(-1),
        atol=1e-7,
    )
