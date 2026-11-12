"""Exact lowercase candidate distributions; no gold prefixes or greedy pruning."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np

BASES = "acgt"
COMPLEMENT = str.maketrans("acgt", "tgca")


def candidates(length: int) -> tuple[str, ...]:
    if length not in (1, 2, 3):
        raise ValueError("Exact benchmark decoding is restricted to 1–3 bases")
    return tuple("".join(p) for p in product(BASES, repeat=length))


def reverse_complement(sequence: str) -> str:
    if set(sequence) - set(BASES):
        raise ValueError("Candidate sequences must contain lowercase acgt only")
    return sequence.translate(COMPLEMENT)[::-1]


def normalize_log_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite example-by-candidate log scores")
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def expand(
    prompts: list[list[int]],
    length: int,
    base_ids: tuple[int, int, int, int],
    next_log_probs: Callable[[list[list[int]]], np.ndarray],
) -> np.ndarray:
    """Expand every prefix once; callback returns normalized lowercase log probs.

    No answer/label is an argument. Prefixes of depth0,1,2 are all evaluated,
    giving1,5,21 prefix states per example for gaps of length1,2,3 respectively.
    Returned columns follow lexicographic acgt candidate order.
    """
    candidates(length)
    if not prompts or any(not p for p in prompts) or len(set(base_ids)) != 4:
        raise ValueError("Expected nonempty prompts and distinct lowercase IDs")
    batch = len(prompts)
    scores = np.zeros((batch, 1), dtype=np.float64)
    for depth in range(length):
        prefixes = ("",) if depth == 0 else candidates(depth)
        requests = [
            prompt + [base_ids[BASES.index(c)] for c in prefix]
            for prompt in prompts
            for prefix in prefixes
        ]
        logs = np.asarray(next_log_probs(requests), dtype=np.float64)
        if logs.shape != (batch * len(prefixes), 4) or not np.isfinite(logs).all():
            raise ValueError(
                "Next-base readout has the wrong shape or nonfinite values"
            )
        if not np.allclose(np.exp(logs).sum(-1), 1, rtol=0, atol=2e-6):
            raise ValueError("Normalize over lowercase acgt before tree expansion")
        scores = (scores[..., None] + logs.reshape(batch, len(prefixes), 4)).reshape(
            batch, -1
        )
    # Only remove floating-point accumulation error; this is not beam truncation.
    if not np.allclose(np.exp(scores).sum(-1), 1, rtol=0, atol=6e-6):
        raise ValueError("The complete candidate tree lost probability mass")
    return normalize_log_scores(scores)


def combine(forward: np.ndarray, reverse: np.ndarray, length: int) -> np.ndarray:
    """Map whole RC answers to forward candidates, then normalize their geomean."""
    alphabet = candidates(length)
    if forward.shape != reverse.shape or forward.shape[-1] != len(alphabet):
        raise ValueError("Both strands must cover the same complete candidate set")
    lookup = {s: i for i, s in enumerate(alphabet)}
    mapping = [lookup[reverse_complement(s)] for s in alphabet]
    return normalize_log_scores(0.5 * (forward + reverse[:, mapping]))


def align_reverse(reverse: np.ndarray, length: int) -> np.ndarray:
    alphabet = candidates(length)
    lookup = {s: i for i, s in enumerate(alphabet)}
    return reverse[:, [lookup[reverse_complement(s)] for s in alphabet]]


def marginal_probabilities(log_scores: np.ndarray, length: int) -> np.ndarray:
    """Marginalize the full joint distribution without consulting a true answer."""
    alphabet = candidates(length)
    if log_scores.shape[-1] != len(alphabet):
        raise ValueError("Incomplete candidate set")
    probs = np.exp(normalize_log_scores(log_scores))
    result = np.zeros((len(probs), length, 4), dtype=np.float64)
    for index, motif in enumerate(alphabet):
        for position, base in enumerate(motif):
            result[:, position, BASES.index(base)] += probs[:, index]
    np.testing.assert_allclose(result.sum(-1), 1, rtol=0, atol=1e-12)
    return result


def predictions(log_scores: np.ndarray, length: int) -> list[str]:
    alphabet = candidates(length)
    if log_scores.shape[-1] != len(alphabet):
        raise ValueError("Incomplete candidate set")
    # Deterministic lexicographic acgt tie-breaking is fixed before any evaluation.
    return [alphabet[i] for i in np.argmax(log_scores, axis=-1)]


def observed_scores(
    log_scores: np.ndarray, answers: list[str]
) -> dict[str, np.ndarray]:
    """Labels are used only after inference to obtain joint/marginal task scores."""
    if (
        not answers
        or len({len(a) for a in answers}) != 1
        or len(log_scores) != len(answers)
    ):
        raise ValueError("Observed answers do not align")
    length = len(answers[0])
    alphabet = candidates(length)
    lookup = {m: i for i, m in enumerate(alphabet)}
    probs = np.exp(normalize_log_scores(log_scores))
    marginals = marginal_probabilities(log_scores, length)
    joint = np.array(
        [probs[i, lookup[a]] if a in lookup else 0.0 for i, a in enumerate(answers)]
    )
    mean = np.array(
        [
            sum(
                marginals[i, p, BASES.index(b)] if b in BASES else 0.0
                for p, b in enumerate(a)
            )
            / length
            for i, a in enumerate(answers)
        ]
    )
    return {"joint": joint, "mean_marginal": mean}
