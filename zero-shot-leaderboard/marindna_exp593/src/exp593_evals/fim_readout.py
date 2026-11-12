"""Explicit trained-FIM scoring, preserving existing task outcomes and reducers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from exp593_evals.prompts import matched_fim


def score_spans(
    scorer: Any,
    requests: list[dict],
    *,
    presentation: str,
    stats: dict | None = None,
    log_outputs: list | None = None,
) -> np.ndarray:
    """Score whole withheld spans; labels and target bases never enter the prompt.

    Later bases in a multi-base answer retain the historical teacher-forcing
    convention. This is not an independently generated whole-span metric.
    """
    if presentation not in {"minimal", "metadata"} or not requests:
        raise ValueError("Expected a supported finite FIM request batch")
    if len({len(r["answer"]) for r in requests}) != 1:
        raise ValueError("A scientific chunk must share one target-span length")
    outputs = []
    for start in range(0, len(requests), scorer.batch_size):
        group = requests[start : start + scorer.batch_size]
        encoded = [
            matched_fim(
                scorer.tokenizer, r["left"], r["right"], r["answer"], r["metadata"]
            )[presentation]
            for r in group
        ]
        if stats is not None:
            stats["serialized_tokens"] = stats.get("serialized_tokens", 0) + sum(
                len(value["ids"]) for value in encoded
            )
            stats["compiled_tokens"] = (
                stats.get("compiled_tokens", 0)
                + scorer.batch_size * scorer.sequence_length
            )
        logits, hidden, _ = scorer.forward([value["ids"] for value in encoded])
        for row, value in enumerate(encoded):
            positions = np.asarray(value["target_positions"]) - 1
            logs = scorer.log_probabilities(logits[row, positions], collapse_case=True)
            outputs.append(logs.exp().cpu().numpy().copy())
            if log_outputs is not None:
                from exp593_evals.legacy_readout import _fp32_projection

                projected = _fp32_projection(
                    hidden[row, positions], scorer.full_head[list(scorer.dna_ids[:8])]
                )
                precise = scorer.log_probabilities(projected, collapse_case=True)
                log_outputs.append(
                    (logs.cpu().numpy().copy(), precise.cpu().numpy().copy())
                )
    return np.stack(outputs)


class FIMAdapter:
    """The existing standard-task function supplies the same target positions."""

    plantcad_native = True

    def __init__(
        self,
        scorer: Any,
        metadata: list[dict],
        *,
        presentation: str,
        sequence_length: int = 8192,
    ) -> None:
        self.scorer, self.metadata, self.presentation = scorer, metadata, presentation
        if sequence_length < 2 or sequence_length % 2 or sequence_length > 8192:
            raise ValueError("Expected an even biological window up to 8192 bases")
        self.sequence_length = sequence_length

    def score_positions(
        self,
        sequences: Any,
        positions: list[int],
        batch_size: int,
        progress: Any = None,
    ) -> tuple[np.ndarray, dict]:
        if batch_size != self.scorer.batch_size or len(sequences) != len(self.metadata):
            raise ValueError("FIM batch or metadata membership differs")
        begin, end = min(positions), max(positions) + 1
        if positions != list(range(begin, end)):
            raise ValueError("Standard-task FIM targets must be one contiguous gap")
        requests = []
        for sequence, metadata in zip(sequences, self.metadata, strict=True):
            if (
                len(sequence) < self.sequence_length
                or (len(sequence) - self.sequence_length) % 2
            ):
                raise ValueError("FIM source cannot be centered at this context length")
            crop = (len(sequence) - self.sequence_length) // 2
            lo, hi = begin - crop, end - crop
            sequence = sequence[crop : crop + self.sequence_length]
            if not 0 < lo < hi < len(sequence):
                raise ValueError("FIM needs both flanks outside the complete target")
            requests.append(
                {
                    "left": sequence[:lo],
                    "right": sequence[hi:],
                    "answer": sequence[lo:hi],
                    "metadata": metadata,
                }
            )
        started = time.monotonic()
        stats = {}
        result = score_spans(
            self.scorer, requests, presentation=self.presentation, stats=stats
        )
        elapsed = time.monotonic() - started
        if progress:
            progress(len(sequences), len(sequences))
        return result, {
            "sequences": len(sequences),
            "tokens": stats["serialized_tokens"],
            "wall_seconds": elapsed,
            "gpu_seconds": None,
            "sequences_per_second": len(sequences) / elapsed,
            "tokens_per_second": stats["serialized_tokens"] / elapsed,
            "compiled_tokens": stats["compiled_tokens"],
        }


def af_target_probabilities(
    scorer: Any,
    sequences: list[str],
    positions: list[int],
    metadata: list[dict],
    *,
    presentation: str,
) -> np.ndarray:
    """REF/ALT are candidate outcomes of the same target-free one-base prompt.

    This FIM readout is a site-only LLR, explicitly distinct from the historical
    AR suffix LLR. A full-vocabulary normalizer cancels in its REF/ALT ratio.
    """
    requests = []
    for sequence, position, fields in zip(sequences, positions, metadata, strict=True):
        requests.append(
            {
                "left": sequence[:position],
                "right": sequence[position + 1 :],
                "answer": sequence[position],
                "metadata": fields,
            }
        )
    return score_spans(scorer, requests, presentation=presentation)[:, 0]


def af_site_readout(
    scorer: Any, frame: Any, protocol: Any, *, sequence_length: int, presentation: str
) -> dict:
    """Paired site odds with native BF16 and FP32 head-projection contracts.

    The original-vocabulary normalizer cancels exactly in the common-prompt
    REF/ALT ratio. The FP32 ratio therefore needs only the eight original rows,
    rather than a 128256-column tensor. Neither value is an AR suffix score.
    """
    from exp593_evals.metadata import metadata_for

    result = {"row_id": frame.row_id.to_numpy(dtype=np.int64)}
    lookup = {base: index for index, base in enumerate("ACGT")}
    for strand in ("fwd", "rc"):
        sequences = []
        for value in frame.sequence:
            offset = (len(value) - sequence_length) // 2
            if len(value) < sequence_length or (len(value) - sequence_length) % 2:
                raise ValueError("AF FIM context is not an exact centered crop")
            sequence = value[offset : offset + sequence_length]
            sequences.append(
                protocol.af.reverse_complement(sequence) if strand == "rc" else sequence
            )
        position = protocol.af.variant_position(sequence_length, strand)
        requests = []
        references, alternatives = [], []
        for sequence, row in zip(sequences, frame.to_dict("records"), strict=True):
            ref, alt = row["ref"].upper(), row["alt"].upper()
            if strand == "rc":
                ref, alt = (
                    protocol.af.reverse_complement(ref),
                    protocol.af.reverse_complement(alt),
                )
            if (
                sequence[position].upper() != ref
                or ref not in lookup
                or alt not in lookup
            ):
                raise ValueError("AF target position or alleles differ from source")
            references.append(lookup[ref])
            alternatives.append(lookup[alt])
            requests.append(
                {
                    "left": sequence[:position],
                    "right": sequence[position + 1 :],
                    "answer": ref,
                    "metadata": metadata_for(
                        {"key": "maize-af", "kind": "af"},
                        row,
                        reverse_complement=strand == "rc",
                    ),
                }
            )
        log_outputs = []
        probabilities = score_spans(
            scorer, requests, presentation=presentation, log_outputs=log_outputs
        )[:, 0]
        canonical = np.stack([value[0][0] for value in log_outputs])
        precise = np.stack([value[1][0] for value in log_outputs])
        rows = np.arange(len(frame))
        result[f"llr_acgt_{strand}"] = (
            canonical[rows, alternatives] - canonical[rows, references]
        )
        result[f"llr_full_vocab_{strand}"] = (
            precise[rows, alternatives] - precise[rows, references]
        )
        result[f"non_acgt_targets_{strand}"] = np.zeros(len(frame), dtype=np.float32)
        result[f"target_probs_{strand}"] = probabilities
    return result


def sv_boundary_scores(
    scorer: Any, frame: Any, metadata: list[dict], *, presentation: str
) -> np.ndarray:
    """Masked single-base FIM at the original ten REF/ALT boundary positions.

    Keep the original mutation/conservation label and its exact boundary reducer.
    Each probability uses both flanks with that one target base withheld; this
    differs intentionally from the historical AR prefix-only condition.
    """
    from exp586_evals.protocol import original

    if len(frame) != len(metadata):
        raise ValueError("SV source and metadata memberships differ")
    requests, indices = [], []
    probabilities = [
        np.zeros((len(frame), 8192, 4), dtype=np.float32) for _ in range(2)
    ]
    for row_id, (row, fields) in enumerate(
        zip(frame.to_dict("records"), metadata, strict=True)
    ):
        if len(row["RefSeq"]) != 8192 or len(row["MutSeq"]) != 8192:
            raise ValueError("SV input windows differ from the original protocol")
        ref_points = [
            *range(int(row["left"]) - 6, int(row["left"]) - 1),
            *range(int(row["right"]), int(row["right"]) + 5),
        ]
        mut_points = list(range(4091, 4101))
        for allele, sequence, points in (
            (0, row["RefSeq"], ref_points),
            (1, row["MutSeq"], mut_points),
        ):
            for position in sorted(set(points)):
                if position <= 0:
                    continue  # The original boundary reducer also skips it.
                if position >= len(sequence) - 1:
                    raise ValueError("SV boundary does not have two visible flanks")
                requests.append(
                    {
                        "left": sequence[:position],
                        "right": sequence[position + 1 :],
                        "answer": sequence[position],
                        "metadata": fields,
                    }
                )
                indices.append((allele, row_id, position))
    values = score_spans(scorer, requests, presentation=presentation)
    for (allele, row_id, position), value in zip(indices, values, strict=True):
        probabilities[allele][row_id, position] = value[0]
    return original().tasks.EVAL._sv_llr_boundary(
        frame, probabilities[0], probabilities[1], flanking=5
    )
