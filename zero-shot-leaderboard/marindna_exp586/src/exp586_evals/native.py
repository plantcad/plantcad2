"""The MoE inference boundary for the existing PlantCAD2 evaluation protocol."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch

from exp586_evals.protocol import original


class NativeAdapter:
    plantcad_native = True

    def __init__(
        self, scorer: Any, tokenizer: Any, *, sequence_length: int = 8192
    ) -> None:
        self.scorer, self.tokenizer = scorer, tokenizer
        if sequence_length < 2 or sequence_length % 2 or sequence_length > 8192:
            raise ValueError("Evaluation context must be even and no larger than 8192")
        self.sequence_length = sequence_length
        ids = [tokenizer.encode(base, add_special_tokens=False).ids for base in "ACGT"]
        if any(len(x) != 1 for x in ids) or [x[0] for x in ids] != list(
            scorer.base_ids
        ):
            raise ValueError(
                "Nucleotide projection does not match actual character encoding"
            )
        if [x[0] for x in ids] not in ([3, 4, 5, 6], [64, 66, 70, 83]):
            raise ValueError("Unexpected exp586 nucleotide IDs")

    def score_positions(
        self,
        sequences: pd.Series,
        positions: list[int],
        batch_size: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if batch_size != self.scorer.batch_size:
            raise ValueError("Native collective batch size differs")
        start = time.monotonic()
        outputs = []
        for offset in range(0, len(sequences), batch_size):
            batch = sequences.iloc[offset : offset + batch_size].tolist()
            source_length = len(batch[0])
            if any(len(sequence) != source_length for sequence in batch):
                raise ValueError("Task batch contains unequal sequence lengths")
            if (
                source_length < self.sequence_length
                or (source_length - self.sequence_length) % 2
            ):
                raise ValueError("Task sequence cannot be center-cropped")
            crop_start = (source_length - self.sequence_length) // 2
            query_positions = [position - crop_start for position in positions]
            if any(
                position < 0 or position >= self.sequence_length
                for position in query_positions
            ):
                raise ValueError("Task target falls outside the centered context")
            batch = [
                sequence[crop_start : crop_start + self.sequence_length]
                for sequence in batch
            ]
            ids = [
                self.tokenizer.encode(x, add_special_tokens=False).ids for x in batch
            ]
            output = self.scorer.evaluate(
                ids,
                [query_positions] * len(batch),
                [0] * len(batch),
                allele_frequency=False,
            )
            outputs.append(output["probabilities"][:, : len(positions)])
            if progress is not None:
                progress(offset + len(batch), len(sequences))
        wall = time.monotonic() - start
        tokens = len(sequences) * self.sequence_length
        return np.concatenate(outputs), {
            "sequences": len(sequences),
            "tokens": tokens,
            "wall_seconds": wall,
            "gpu_seconds": None,
            "sequences_per_second": len(sequences) / wall,
            "tokens_per_second": tokens / wall,
        }

    def allele_frequency(
        self, frame: pd.DataFrame, *, include_strand_average: bool = True
    ) -> tuple[dict, dict]:
        """Use original suffix reduction and strand-averaged frozen representations."""
        protocol = original()
        predictions = {"row_id": frame.row_id.to_numpy(dtype=np.int64)}
        embeddings = {"row_id": predictions["row_id"]}
        strands = {}
        for strand in ("fwd", "rc"):
            position = protocol.af.variant_position(self.sequence_length, strand)
            values = {"ref": [], "alt": []}
            for offset in range(0, len(frame), self.scorer.batch_size):
                batch = frame.iloc[offset : offset + self.scorer.batch_size]
                refs = [
                    _center_crop(sequence, self.sequence_length)
                    for sequence in batch.sequence.tolist()
                ]
                center = self.sequence_length // 2
                alts = [
                    s[:center] + a.upper() + s[center + 1 :]
                    for s, a in zip(refs, batch.alt, strict=True)
                ]
                for allele, sequences in [("ref", refs), ("alt", alts)]:
                    if strand == "rc":
                        sequences = [
                            protocol.af.reverse_complement(s) for s in sequences
                        ]
                    ids = [
                        self.tokenizer.encode(s, add_special_tokens=False).ids
                        for s in sequences
                    ]
                    values[allele].append(
                        self.scorer.evaluate(
                            ids,
                            [[position]] * len(ids),
                            [position] * len(ids),
                            allele_frequency=True,
                        )
                    )
            merged = {
                allele: {key: np.concatenate([x[key] for x in rows]) for key in rows[0]}
                for allele, rows in values.items()
            }
            paired = {
                key: torch.from_numpy(
                    np.stack([merged[a][key] for a in ("ref", "alt")], axis=1)
                )
                for key in ("acgt_logs", "full_logs", "valid")
            }
            scores = protocol.af.reduce_uncached(
                paired["acgt_logs"], paired["full_logs"], paired["valid"], position
            ).numpy()
            for col, index in [
                ("llr_acgt", 0),
                ("llr_full_vocab", 1),
                ("non_acgt_targets", 2),
            ]:
                predictions[f"{col}_{strand}"] = scores[:, index]
            ref_probabilities = merged["ref"]["probabilities"][:, 0]
            alt_probabilities = merged["alt"]["probabilities"][:, 0]
            np.testing.assert_allclose(
                ref_probabilities, alt_probabilities, rtol=0, atol=2e-4
            )
            predictions[f"target_probs_{strand}"] = ref_probabilities
            strands[strand] = merged
        for representation, field in [
            ("whole_window", "mean"),
            ("variant_token", "variant"),
        ]:
            for allele in ("ref", "alt"):
                for strand in ("fwd", "rc"):
                    embeddings[f"{representation}_{allele}_{strand}"] = strands[strand][
                        allele
                    ][field].astype(np.float32)
                if include_strand_average:
                    embeddings[f"{representation}_{allele}"] = (
                        strands["fwd"][allele][field].astype(np.float32)
                        + strands["rc"][allele][field].astype(np.float32)
                    ) * np.float32(0.5)
        return predictions, embeddings


def _center_crop(sequence: str, length: int) -> str:
    if len(sequence) < length or (len(sequence) - length) % 2:
        raise ValueError("Sequence cannot be centered at the requested context length")
    start = (len(sequence) - length) // 2
    return sequence[start : start + length]
