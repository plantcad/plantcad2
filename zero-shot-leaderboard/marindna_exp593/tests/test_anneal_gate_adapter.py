import numpy as np
import pytest

torch = pytest.importorskip("torch")
from exp593_anneal_evals.gate import adapter_diagnostics


class Scorer:
    dna_ids = (32, 34, 38, 51, 64, 66, 70, 83, 45, 77)
    batch_size = 64
    tokenizer = None
    device = "cpu"

    def forward(self, ids):
        logits = torch.zeros((len(ids), max(map(len, ids)), 8))
        logits[..., 4:] = torch.tensor([2.0, 0.5, -1.0, 0.0])
        return logits, None, torch.tensor(list(map(len, ids)))


def test_strict_adapter_comparison_uses_the_same_physical_requests():
    cases = {
        2: [
            (
                {"ids": [64, 64, 66, 70], "target_positions": [2, 3], "answer": "cg"},
                np.full((2, 4), 0.25),
            )
        ]
    }
    prefixes = [({"ids": [64, 66, 64]}, np.full(4, 0.25))]
    result = adapter_diagnostics(Scorer(), cases, prefixes)
    assert result["canonical_adapter_max_abs"] < 1e-6
    # The earlier, differently composed reference batch stays a diagnostic; it
    # cannot make a correct adapter fail the strict same-request comparison.
    assert result["regrouping_max_abs"] > 0.1
