"""Tokenize supported FIM presentations without merging DNA into text BPE."""

from __future__ import annotations

from typing import Any

from exp593_evals.readouts import canonical_ids, normalize_dna
from exp593_evals.rendering import fim_spans


def encode_spans(tokenizer: Any, spans: Any) -> tuple[list[int], list[int]]:
    lookup = dict(zip("ACGTacgtNn", canonical_ids(tokenizer), strict=True))
    tokens, positions = [], []
    for span in spans:
        ids = (
            tokenizer.encode(span.text, add_special_tokens=False).ids
            if span.role == "text"
            else [lookup[base] for base in span.text]
        )
        if span.score:
            positions.extend(range(len(tokens), len(tokens) + len(ids)))
        tokens.extend(ids)
    return tokens, positions


def matched_fim(
    tokenizer: Any,
    left: str,
    right: str,
    answer: str,
    metadata: dict[str, str],
    *,
    maximum_length: int = 8192,
) -> dict[str, dict[str, Any]]:
    """Reserve the richer prompt overhead once; keep exactly the same flanks.

    Trimming uses lengths only. No outcome or test score chooses the context.
    Answer token positions are explicit and must never occur inside the prompt.
    """
    left, right, answer = (
        normalize_dna(s, case="lower") for s in (left, right, answer)
    )
    if not answer:
        raise ValueError("A caller-known nonempty answer span is required")
    overhead = (
        len(
            encode_spans(
                tokenizer,
                fim_spans("a", "a", "", gap_length=len(answer), metadata=metadata),
            )[0]
        )
        - 2
    )
    available = maximum_length - overhead - len(answer)
    if available < 2:
        raise ValueError("Prompt and answer leave no two-sided context")
    nleft = min(len(left), available // 2)
    nright = min(len(right), available - nleft)
    nleft = min(len(left), available - nright)
    left, right = left[-nleft:], right[:nright]
    result = {}
    for name, fields in (("minimal", None), ("metadata", metadata)):
        spans = fim_spans(left, right, answer, gap_length=len(answer), metadata=fields)
        ids, positions = encode_spans(tokenizer, spans)
        prompt, _ = encode_spans(
            tokenizer,
            fim_spans(left, right, "", gap_length=len(answer), metadata=fields),
        )
        if (
            ids[: len(prompt)] != prompt
            or positions != list(range(len(prompt), len(ids)))
            or len(ids) > maximum_length
        ):
            raise ValueError("FIM answer exclusion or length contract failed")
        result[name] = {
            "ids": ids,
            "target_positions": positions,
            "prompt_length": len(prompt),
            "left": left,
            "right": right,
            "presentation": "".join(s.text for s in spans if s.role == "text"),
        }
    return result
