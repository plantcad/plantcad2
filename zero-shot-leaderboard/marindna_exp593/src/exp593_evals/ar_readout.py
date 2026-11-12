"""Case-aware readouts exposing the unchanged exp586 protocol interface."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from exp593_evals.legacy_readout import _pool_hidden, streamed_full_log_probs


class DNATokenizer:
    """DNA alone is character-forced; this object never tokenizes prompt text."""

    def __init__(self, scorer: Any, case: str) -> None:
        self.scorer, self.case = scorer, case

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Any:
        if add_special_tokens:
            raise ValueError("The benchmark does not insert DNA special tokens")
        return SimpleNamespace(ids=self.scorer.dna_tokens(text, case=self.case))


class ARReadout:
    """One backbone can expose lower-control, lower-folded and upper-folded views."""

    def __init__(self, scorer: Any, *, case: str, collapse_case: bool) -> None:
        if case not in {"lower", "upper"}:
            raise ValueError("Unknown input case")
        self.scorer, self.case, self.collapse_case = scorer, case, collapse_case
        self.tokenizer = DNATokenizer(scorer, case)
        self.batch_size = scorer.batch_size
        self.sequence_length = scorer.sequence_length
        self.base_ids = scorer.dna_ids[4:8] if case == "lower" else scorer.dna_ids[:4]

    def evaluate(
        self,
        ids: list[list[int]],
        positions: list[list[int]],
        feature_positions: list[int],
        *,
        allele_frequency: bool,
    ) -> dict[str, np.ndarray]:
        import torch

        if not ids or len({len(row) for row in ids}) != 1:
            raise ValueError("The existing AR protocol requires equal real lengths")
        actual, length = len(ids), len(ids[0])
        if len(positions) != actual or len(feature_positions) != actual:
            raise ValueError("Input and query memberships differ")
        if any(not 0 <= p < length for row in positions for p in row) or any(
            not 0 <= p < length for p in feature_positions
        ):
            raise ValueError("Query falls outside actual input")
        queries = np.zeros((actual, max(10, max(map(len, positions)))), dtype=np.int64)
        for index, row in enumerate(positions):
            queries[index, : len(row)] = row
        with torch.inference_mode():
            logits, hidden, lengths = self.scorer.forward(ids)
            logs = self.scorer.log_probabilities(
                logits[:, : length - 1], collapse_case=self.collapse_case
            )
            probabilities = logs.exp()
            q = torch.tensor(queries, device=self.scorer.device)
            rows = torch.arange(actual, device=self.scorer.device)[:, None]
            selected = probabilities[rows, (q - 1).clamp(min=0)]
            selected = torch.where(q[..., None] > 0, selected, 0)
            targets = torch.tensor(ids, device=self.scorer.device)[:, 1:]
            matches = targets[..., None] == torch.tensor(
                self.base_ids, device=self.scorer.device
            )
            valid = matches.any(-1)
            acgt = logs.gather(-1, matches.long().argmax(-1)[..., None])[..., 0]
            full = torch.zeros_like(acgt)
            if allele_frequency:
                full = streamed_full_log_probs(
                    hidden=hidden[:, : length - 1],
                    targets=targets,
                    head=self.scorer.full_head,
                )
                if self.collapse_case:
                    # Case alternatives share one full-vocabulary normalizer.
                    other_ids = (
                        self.scorer.dna_ids[:4]
                        if self.case == "lower"
                        else self.scorer.dna_ids[4:8]
                    )
                    opposite = torch.tensor(other_ids, device=self.scorer.device)[
                        matches.long().argmax(-1)
                    ]
                    other = streamed_full_log_probs(
                        hidden=hidden[:, : length - 1],
                        targets=opposite,
                        head=self.scorer.full_head,
                    )
                    full = torch.where(valid, torch.logaddexp(full, other), full)
            values = {
                "probabilities": selected,
                "acgt_logs": acgt,
                "full_logs": full,
                "valid": valid,
                "mean": _pool_hidden(hidden, lengths, "mean"),
                "variant": hidden[
                    torch.arange(actual, device=self.scorer.device),
                    torch.tensor(feature_positions, device=self.scorer.device),
                ].float(),
            }
            result = {
                name: value.cpu().numpy().copy() for name, value in values.items()
            }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Nonfinite AR readout")
        return result


class ARAdapter:
    """Reuse original methods while permitting the explicitly selected case IDs."""

    plantcad_native = True

    def __init__(
        self,
        scorer: Any,
        *,
        case: str,
        collapse_case: bool,
        sequence_length: int = 8192,
    ) -> None:
        if sequence_length < 2 or sequence_length % 2 or sequence_length > 8192:
            raise ValueError(
                "AR context must be even and no longer than the model window"
            )
        self.scorer = ARReadout(scorer, case=case, collapse_case=collapse_case)
        self.tokenizer = self.scorer.tokenizer
        self.sequence_length = sequence_length

    def score_positions(self, *args: Any, **kwargs: Any) -> Any:
        from exp586_evals.native import NativeAdapter

        return NativeAdapter.score_positions(self, *args, **kwargs)

    def allele_frequency(self, *args: Any, **kwargs: Any) -> Any:
        from exp586_evals.native import NativeAdapter

        return NativeAdapter.allele_frequency(self, *args, **kwargs)
