"""Case-aware probability definitions, independent of inference providers."""

from __future__ import annotations

import numpy as np


def nucleotide_probabilities(logits: np.ndarray, *, collapse_case: bool) -> np.ndarray:
    """Condition on canonical bases; input columns are exactly ACGTacgt.

    Adding exponentiated logits implements summation under one common softmax
    normalizer. Summing already separately normalized case distributions would
    assign an arbitrary equal prior to uppercase and lowercase and is incorrect.
    """
    z = np.asarray(logits, dtype=np.float32)
    if z.shape[-1] != 8 or not np.all(np.isfinite(z)):
        raise ValueError("Expected eight finite ACGTacgt logits")
    if not collapse_case:
        z = z[..., 4:]
    weights = np.exp(z - z.max(axis=-1, keepdims=True))
    if collapse_case:
        weights = weights[..., :4] + weights[..., 4:]
    return weights / weights.sum(axis=-1, keepdims=True)


def canonical_ids(tokenizer: object) -> tuple[int, ...]:
    """Use isolated original character IDs; never a lowercasing export tokenizer."""
    result = []
    for base in "ACGTacgtNn":
        encoded = tokenizer.encode(base, add_special_tokens=False).ids
        if len(encoded) != 1:
            raise ValueError("Original tokenizer must encode each DNA character once")
        result.append(encoded[0])
    if len(set(result)) != len(result):
        raise ValueError("Original tokenizer collapsed distinct DNA characters")
    return tuple(result)


def normalize_dna(sequence: str, *, case: str) -> str:
    """Map IUPAC ambiguity to N one base at a time; reject other characters."""
    if case not in {"lower", "upper"}:
        raise ValueError("Case must be lower or upper")
    if set(sequence.upper()) - set("ACGTNRYSWKMBDHV"):
        raise ValueError("Unexpected genomic alphabet")
    result = "".join(base if base in "ACGTN" else "N" for base in sequence.upper())
    return result.lower() if case == "lower" else result
