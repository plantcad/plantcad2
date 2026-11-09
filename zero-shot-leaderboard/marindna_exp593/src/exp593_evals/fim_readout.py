"""Explicit trained-FIM scoring, preserving existing task outcomes and reducers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from exp593_evals.prompts import matched_fim


def score_spans(scorer: Any, requests: list[dict], *, presentation: str) -> np.ndarray:
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
        logits, _, _ = scorer.forward([value["ids"] for value in encoded])
        for row, value in enumerate(encoded):
            positions = np.asarray(value["target_positions"]) - 1
            outputs.append(
                scorer.probabilities(logits[row, positions], collapse_case=True)
                .cpu()
                .numpy()
                .copy()
            )
    return np.stack(outputs)


class FIMAdapter:
    """The existing standard-task function supplies the same target positions."""

    plantcad_native = True

    def __init__(self, scorer: Any, metadata: list[dict], *, presentation: str) -> None:
        self.scorer, self.metadata, self.presentation = scorer, metadata, presentation

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
            if not 0 < begin < end < len(sequence):
                raise ValueError("FIM needs both flanks outside the complete target")
            requests.append(
                {
                    "left": sequence[:begin],
                    "right": sequence[end:],
                    "answer": sequence[begin:end],
                    "metadata": metadata,
                }
            )
        started = time.monotonic()
        result = score_spans(self.scorer, requests, presentation=self.presentation)
        elapsed = time.monotonic() - started
        if progress:
            progress(len(sequences), len(sequences))
        return result, {
            "sequences": len(sequences),
            "tokens": None,
            "wall_seconds": elapsed,
            "gpu_seconds": None,
            "sequences_per_second": len(sequences) / elapsed,
            "tokens_per_second": None,
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
                if not 0 < position < len(sequence) - 1:
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
