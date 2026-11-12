"""One training-supported no-metadata FIM form; lowercase DNA never uses BPE."""

from __future__ import annotations

from exp593_dna.pipeline import Span
from exp593_evals.prompts import encode_spans
from exp593_evals.readouts import normalize_dna


def fim_spans(left: str, right: str, length: int, answer: str = "") -> tuple[Span, ...]:
    if length < 1 or answer and len(answer) != length:
        raise ValueError("A FIM answer is absent or exactly the caller-known length")
    return (
        Span("[left] ", "text"),
        Span(left, "left"),
        Span("\n[right] ", "text"),
        Span(right, "right"),
        Span(f"\n[gap:{length}]\n[fill] ", "text"),
        Span(answer, "middle", True),
    )


def matched_flanks(
    tokenizer, left: str, right: str, length: int, *, budget: int = 8192
) -> tuple[str, str]:
    """Reserve only minimal structural overhead; use the same visible pair for RC.

    The historical AR baseline remains untouched. Both TF and exact FIM use
    these identical target-free flanks, with room reserved for all answer bases.
    """
    left, right = (normalize_dna(s, case="lower") for s in (left, right))
    if not left or not right:
        raise ValueError("FIM requires both flanks outside the complete target")
    overhead = len(encode_spans(tokenizer, fim_spans("a", "a", length))[0]) - 2
    available = budget - overhead - length
    if available < 2:
        raise ValueError("FIM structure leaves no two-sided context")
    excess = max(0, len(left) + len(right) - available)
    trim_left = min(len(left) - 1, (excess + 1) // 2)
    trim_right = excess - trim_left
    if trim_right >= len(right):
        shift = trim_right - len(right) + 1
        trim_left += shift
        trim_right -= shift
    if trim_left >= len(left):
        raise ValueError("No complete flank survives trimming")
    return left[trim_left:], right[: len(right) - trim_right]


def prompt_ids(tokenizer, left: str, right: str, length: int) -> list[int]:
    ids, targets = encode_spans(tokenizer, fim_spans(left, right, length))
    if targets:
        raise ValueError("A decoding prompt contains answer tokens")
    return ids
