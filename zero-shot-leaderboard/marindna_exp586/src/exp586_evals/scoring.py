"""Base-coordinate scoring for character DNA and historical BPE diagnostics.

Canonical BPE strings are scored after independent retokenization. This is a
likelihood-based scoring rule, not an exact marginal over all BPE segmentations.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from tokenizers import Tokenizer

TokenLogProbs = Callable[[list[list[int]]], list[np.ndarray]]
BASES = "acgt"


def reverse_complement(sequence: str) -> str:
    return sequence.lower().translate(str.maketrans("acgt", "tgca"))[::-1]


def shared_prefix_length(sequences: Sequence[Sequence[int]]) -> int:
    for index, values in enumerate(zip(*sequences)):
        if len(set(values)) != 1:
            return index
    return min(map(len, sequences))


@dataclass
class SequenceScorer:
    tokenizer: Tokenizer
    token_log_probs: TokenLogProbs
    max_tokens: int = 8192

    def encode(self, sequence: str) -> list[int]:
        if not sequence or not sequence.isascii() or not sequence.isalpha():
            raise ValueError("Expected a nonempty ASCII DNA sequence")
        ids = self.tokenizer.encode(sequence.lower(), add_special_tokens=False).ids
        if not 1 <= len(ids) <= self.max_tokens:
            raise ValueError(
                "Sequence outside the token limit; truncation is not allowed"
            )
        return ids

    def _divergent_scores(self, sequences: list[str]) -> np.ndarray:
        ids = [self.encode(sequence) for sequence in sequences]
        shared = shared_prefix_length(ids)
        if shared == 0:
            raise ValueError(
                "Candidates lack a shared first token; cannot score without a BOS distribution"
            )
        logs = self.token_log_probs(ids)
        if len(logs) != len(ids):
            raise ValueError("Backend returned a different batch size")
        scores = []
        for tokens, values in zip(ids, logs, strict=True):
            values = np.asarray(values)
            if values.shape != (len(tokens) - 1,) or not np.isfinite(values).all():
                raise ValueError(
                    "Backend must return one finite log probability for each next-token target"
                )
            scores.append(np.sum(values[shared - 1 :], dtype=np.float64))
        return np.asarray(scores)

    def base_probabilities(self, sequence: str, position: int) -> np.ndarray:
        """Score A/C/G/T after the left prefix; position is 0-based.

        Each candidate is retokenized through the queried base only, so no right
        flank is supplied. For a character tokenizer this is exactly the old
        four-logit softmax, including cancellation of the full-vocabulary normalizer.
        """
        if not 0 < position < len(sequence):
            raise ValueError("The queried base needs a nonempty left context")
        candidates = [sequence[:position] + base for base in BASES]
        return softmax(self._divergent_scores(candidates))

    def llr(self, reference: str, alternate: str) -> float:
        """Full-vocabulary ALT-minus-REF log likelihood on independently tokenized windows."""
        if reference.lower() == alternate.lower():
            return 0.0
        ref_score, alt_score = self._divergent_scores([reference, alternate])
        return float(alt_score - ref_score)

    def strand_average_llr(self, reference: str, alternate: str) -> float:
        return 0.5 * (
            self.llr(reference, alternate)
            + self.llr(reverse_complement(reference), reverse_complement(alternate))
        )


def embedding_features(
    tokenizer: Tokenizer, sequence: str, hidden: np.ndarray, variant_position: int
) -> tuple[np.ndarray, np.ndarray]:
    """Base-weighted whole-window mean and state of the token covering the variant.

    Hidden states are causal through each token's end. A BPE variant-token state
    can therefore include adjacent bases within that token; this is recorded as
    a tokenizer-dependent representation difference.
    """
    encoding = tokenizer.encode(sequence.lower(), add_special_tokens=False)
    offsets = np.asarray(encoding.offsets)
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[0] != len(encoding.ids):
        raise ValueError("One hidden vector per real token is required")
    if len(offsets) == 0 or offsets[0, 0] != 0 or offsets[-1, 1] != len(sequence):
        raise ValueError("Tokenizer offsets do not cover the window")
    if np.any(offsets[1:, 0] != offsets[:-1, 1]) or np.any(
        offsets[:, 1] <= offsets[:, 0]
    ):
        raise ValueError("Tokenizer offsets overlap or contain gaps")
    overlap = np.flatnonzero(
        (offsets[:, 0] <= variant_position) & (variant_position < offsets[:, 1])
    )
    if len(overlap) != 1:
        raise ValueError("Variant must be covered by exactly one token")
    weights = (offsets[:, 1] - offsets[:, 0]).astype(np.float32)
    mean = np.sum(hidden * weights[:, None], axis=0, dtype=np.float32) / len(sequence)
    return mean, hidden[overlap[0]]


def paired_probe_feature(
    ref: np.ndarray, alt: np.ndarray, ref_rc: np.ndarray, alt_rc: np.ndarray
) -> np.ndarray:
    """Historical FP32 strand averages followed by [REF, ALT minus REF]."""
    values = [np.asarray(x, dtype=np.float32) for x in (ref, alt, ref_rc, alt_rc)]
    if any(
        x.ndim != 1 or x.shape != values[0].shape or not np.isfinite(x).all()
        for x in values
    ):
        raise ValueError("Expected four finite matching embedding vectors")
    reference = (values[0] + values[2]) * np.float32(0.5)
    alternate = (values[1] + values[3]) * np.float32(0.5)
    return np.concatenate((reference, alternate - reference))
