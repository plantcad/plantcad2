import numpy as np
import pytest

torch = pytest.importorskip("torch")
from exp593_anneal_evals.scoring import LowercaseScorer


class Backbone:
    dna_ids = (32, 34, 38, 51, 64, 66, 70, 83, 45, 77)
    batch_size = 3
    tokenizer = None
    device = "cpu"

    def __init__(self, upper):
        self.upper = upper
        self.seen = []

    def forward(self, ids):
        self.seen.extend(ids)
        logits = torch.empty((len(ids), max(map(len, ids)), 8))
        logits[..., :4] = self.upper
        logits[..., 4:] = torch.tensor([-0.3, 0.2, 0.7, -0.5])
        return (
            logits,
            torch.zeros((*logits.shape[:2], 2)),
            torch.tensor(list(map(len, ids))),
        )


def test_extreme_uppercase_logits_cannot_affect_prediction_probabilities():
    a = LowercaseScorer(Backbone(1000))
    b = LowercaseScorer(Backbone(-1000))
    requests = [a.ids("ACGTN"), a.ids("TTAC")]
    actual = a.next_log_probs(requests)
    expected = b.next_log_probs(requests)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(np.exp(actual).sum(-1), 1, atol=2e-7)
    assert {token for row in a.scorer.seen for token in row} <= {64, 66, 70, 83, 77}


def test_teacher_forcing_uses_lowercase_answer_ids_and_the_same_normalizer():
    scorer = Backbone(1000)
    a = LowercaseScorer(scorer)
    probabilities = a.teacher([a.ids("AAAA")], ["acg"])
    assert scorer.seen == [[64] * 4 + [64, 66, 70]]
    np.testing.assert_allclose(
        probabilities[0],
        np.tile(np.exp(a.next_log_probs([[64]]))[0], (3, 1)),
        atol=1e-7,
    )


def test_causal_query_positions_and_padding_are_not_off_by_one():
    class PositionalBackbone(Backbone):
        def forward(self, ids):
            self.seen.extend(ids)
            logits = torch.zeros((len(ids), max(map(len, ids)), 8))
            for row, tokens in enumerate(ids):
                for position, token in enumerate(tokens):
                    logits[row, position, 4 + token % 4] = position + 1
            return logits, None, torch.tensor(list(map(len, ids)))

    backbone = PositionalBackbone(1000)
    adapter = LowercaseScorer(backbone)
    # The first prediction reads the final context token, then each previous
    # answer token. A target-position or pad-position shift changes the argmax.
    p = adapter.teacher([[65, 66], [67, 64, 65]], ["cg", "ta"])
    assert p.argmax(-1).tolist() == [[2, 2], [1, 3]]
    before = np.exp(adapter.next_log_probs([[65, 66], [67, 64, 65]]))
    np.testing.assert_allclose(p[:, 0], before, atol=1e-7)
