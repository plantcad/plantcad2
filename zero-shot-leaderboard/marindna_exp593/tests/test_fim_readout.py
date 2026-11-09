from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import pandas as pd

from exp593_dna.pipeline import load_encoder
from exp593_evals.fim_readout import af_site_readout, score_spans
from exp593_evals.readouts import canonical_ids


class ToyFIM:
    batch_size = 2
    sequence_length = 8192

    def __init__(self):
        self.tokenizer = load_encoder().tokenizer
        self.dna_ids = canonical_ids(self.tokenizer)
        generator = torch.Generator().manual_seed(593)
        self.full_head = torch.randn((128256, 3), generator=generator) * 0.01

    def forward(self, ids):
        x = torch.tensor(ids, dtype=torch.float32)
        visible = (x % 7).cumsum(1) / 100
        hidden = torch.stack((visible, visible * 0.5, torch.ones_like(visible)), dim=-1)
        self.last_hidden = hidden
        return (
            hidden @ self.full_head[list(self.dna_ids[:8])].T,
            hidden,
            torch.tensor([len(row) for row in ids]),
        )

    def log_probabilities(self, logits, *, collapse_case):
        assert collapse_case
        return torch.log_softmax(
            torch.logaddexp(logits[..., :4], logits[..., 4:]), dim=-1
        )


def test_changing_the_hidden_answer_does_not_change_its_prediction():
    scorer = ToyFIM()
    requests = [
        {
            "left": "aacgt",
            "right": "ttgca",
            "answer": answer,
            "metadata": {"species": "Zea mays"},
        }
        for answer in ["a", "t"]
    ]
    for presentation in ("minimal", "metadata"):
        actual = score_spans(scorer, requests, presentation=presentation)
        np.testing.assert_array_equal(actual[0], actual[1])


def test_fim_af_changes_candidates_but_never_conditions_on_af():
    scorer = ToyFIM()
    sequence = "A" * 32 + "C" + "G" * 31
    frame = pd.DataFrame(
        [
            {
                "row_id": 0,
                "sequence": sequence,
                "chrom": "10",
                "ref": "C",
                "alt": "T",
                "AF": 0.001,
            },
            {
                "row_id": 1,
                "sequence": sequence,
                "chrom": "10",
                "ref": "C",
                "alt": "T",
                "AF": 0.99,
            },
        ]
    )
    translation = str.maketrans("ACGTN", "TGCAN")
    protocol = SimpleNamespace(
        af=SimpleNamespace(
            reverse_complement=lambda value: value.upper().translate(translation)[::-1],
            variant_position=lambda size, strand: (
                size // 2 if strand == "fwd" else size - 1 - size // 2
            ),
        )
    )
    result = af_site_readout(
        scorer, frame, protocol, sequence_length=64, presentation="metadata"
    )
    for name, values in result.items():
        if name != "row_id":
            np.testing.assert_array_equal(values[0], values[1])
    np.testing.assert_allclose(
        result["llr_acgt_fwd"], result["llr_full_vocab_fwd"], atol=2e-6
    )
    # Independently normalize all original vocabulary columns for the RC site.
    full = torch.log_softmax(scorer.last_hidden[:, -2] @ scorer.full_head.T, dim=-1)
    ids = scorer.dna_ids
    expected = torch.logsumexp(full[:, [ids[0], ids[4]]], dim=-1) - torch.logsumexp(
        full[:, [ids[2], ids[6]]], dim=-1
    )
    np.testing.assert_allclose(result["llr_full_vocab_rc"], expected.numpy(), atol=2e-6)
