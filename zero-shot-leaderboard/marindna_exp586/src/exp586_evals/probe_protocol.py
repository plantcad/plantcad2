"""Explicit input visibility and linear readouts for frozen classification probes."""

from typing import Any

import numpy as np

IUPAC = "ACGTRYSWKMBDHVN"
IUPAC_COMPLEMENT = "TGCAYRSWMKVHDBN"
COMPLEMENT = str.maketrans(
    IUPAC + IUPAC.lower(), IUPAC_COMPLEMENT + IUPAC_COMPLEMENT.lower()
)


def reverse_complement(sequence: str) -> str:
    """Reverse and complement a DNA sequence without adding tokens."""
    if set(sequence.upper()) - set(IUPAC):
        raise ValueError("Unexpected DNA characters")
    return sequence.translate(COMPLEMENT)[::-1]


def withheld_span(spec: dict[str, Any], length: int) -> tuple[int, int]:
    """Return the complete half-open withheld span in original base coordinates."""
    if spec["kind"] not in ("conservation", "core"):
        raise ValueError("Only conservation/core tasks use withheld flanks")
    positions = spec["positions"]
    begin, end = min(positions), max(positions) + 1
    if positions != list(range(begin, end)) or not 0 < begin < end < length:
        raise ValueError("Targets must form an interior contiguous span")
    return begin, end


def visible_sequences(spec: dict[str, Any], row: dict[str, Any]) -> dict[str, str]:
    """Build model inputs without accessing outcome labels.

    The entire designated region is absent from both centered-task passes.
    SV alleles are observed inputs; their conservation label is never read.
    """
    if spec["kind"] == "sv":
        ref, alt = row["RefSeq"].upper(), row["MutSeq"].upper()
        if len(ref) != 8192 or len(alt) != 8192 or ref == alt:
            raise ValueError("Expected the original distinct 8192-base SV pair")
        return {
            "ref_fwd": ref,
            "ref_rc": reverse_complement(ref),
            "alt_fwd": alt,
            "alt_rc": reverse_complement(alt),
        }
    sequence = row["sequence"].upper()
    if len(sequence) != 8192:
        raise ValueError("Expected the original 8192-base task window")
    begin, end = withheld_span(spec, len(sequence))
    return {
        "fwd": sequence[:begin],
        "rc": reverse_complement(sequence[end:]),
    }


def linear_features(
    arrays: dict[str, np.ndarray], *, kind: str, directional: bool
) -> np.ndarray:
    """Construct strand-averaged or directional features in the AF convention."""
    if kind == "sv":
        rf, rr, af, ar = (
            np.asarray(arrays[name], dtype=np.float32)
            for name in ("ref_fwd", "ref_rc", "alt_fwd", "alt_rc")
        )
        if directional:
            pieces = (rf, af - rf, rr, ar - rr)
        else:
            ref, alt = (rf + rr) * np.float32(0.5), (af + ar) * np.float32(0.5)
            pieces = (ref, alt - ref)
    elif kind in ("conservation", "core"):
        fwd, rc = (np.asarray(arrays[name], dtype=np.float32) for name in ("fwd", "rc"))
        pieces = (fwd, rc) if directional else ((fwd + rc) * np.float32(0.5),)
    else:
        raise ValueError("Unexpected probe outcome")
    if any(piece.ndim != 2 or not np.isfinite(piece).all() for piece in pieces):
        raise ValueError("Expected finite row-by-feature matrices")
    if len({piece.shape for piece in pieces}) != 1:
        raise ValueError("Strand/allele features do not align")
    return np.concatenate(pieces, axis=1)
