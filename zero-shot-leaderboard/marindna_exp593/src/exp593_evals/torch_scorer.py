"""Case-aware Fast-FA3 adapter; attention, experts and router are inherited."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from exp593_evals.legacy_readout import (
    _configure_environment,
    _pool_hidden,
    _readout_model_class,
)
from exp593_evals.readouts import canonical_ids, normalize_dna


class FastScorer:
    """Original-ID full embedding with a selected BF16 eight-row DNA head."""

    def __init__(
        self, model_dir: Path, *, batch_size: int = 64, sequence_length: int = 8192
    ) -> None:
        _configure_environment("fast-fa3")
        import torch
        from tokenizers import Tokenizer

        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or "H100" not in torch.cuda.get_device_name(0)
        ):
            raise RuntimeError("Expose exactly one H100 to each scorer process")
        if sequence_length != 8192 or batch_size < 1:
            raise ValueError("Use a positive fixed batch and 8192-token compiler shape")
        self.batch_size, self.sequence_length = batch_size, sequence_length
        self.device = torch.device("cuda:0")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "original-tokenizer.json"))
        receipt = json.loads((model_dir / "native-receipt.json").read_text())
        self.checkpoint = receipt["checkpoint"]
        if (
            hashlib.sha256(self.tokenizer.to_str().encode()).hexdigest()
            != self.checkpoint["bindings"]["tokenizer"]
        ):
            raise ValueError("Scorer tokenizer differs from its training checkpoint")
        self.dna_ids = canonical_ids(self.tokenizer)
        self.model = _readout_model_class("fast-fa3")(
            model_dir,
            self.device,
            dtype=torch.bfloat16,
            input_vocab=None,
            output_vocab=None,
            attention_provider="vllm",
            flash_attention_version=3,
            moe_backend="vllm",
        )
        self.full_head = self.model._weight("lm_head.weight")
        buffer = self.model._weight_names["lm_head.weight"]
        self.model.register_buffer(
            buffer,
            self.full_head[list(self.dna_ids[:8])].contiguous(),
            persistent=False,
        )
        self._forward = torch.compile(
            self.model, fullgraph=True, dynamic=False, mode="reduce-overhead"
        )

    def dna_tokens(self, sequence: str, *, case: str) -> list[int]:
        text = normalize_dna(sequence, case=case)
        lookup = {c: token for c, token in zip("ACGTacgtNn", self.dna_ids, strict=True)}
        return [lookup[base] for base in text]

    def forward(self, ids: list[list[int]]) -> tuple[Any, Any, Any]:
        """Right-pad independent documents; never pack or truncate examples."""
        import torch

        if (
            not ids
            or len(ids) > self.batch_size
            or any(not row or len(row) > self.sequence_length for row in ids)
        ):
            raise ValueError("Batch exceeds its document or sequence shape")
        lengths = torch.tensor(
            [len(row) for row in ids], device=self.device, dtype=torch.long
        )
        padded = torch.zeros(
            (self.batch_size, self.sequence_length),
            device=self.device,
            dtype=torch.long,
        )
        for index, row in enumerate(ids):
            padded[index, : len(row)] = torch.tensor(
                row, device=self.device, dtype=torch.long
            )
        # Repeat real rows in the last physical batch, matching the established
        # fullgraph workload and excluding repeated rows from scientific outputs.
        for index in range(len(ids), self.batch_size):
            padded[index] = padded[index % len(ids)]
        with torch.inference_mode():
            logits, hidden = self._forward(padded)
            return logits[: len(ids)].clone(), hidden[: len(ids)].clone(), lengths

    def probabilities(self, logits: Any, *, collapse_case: bool) -> Any:
        return self.log_probabilities(logits, collapse_case=collapse_case).exp()

    def log_probabilities(self, logits: Any, *, collapse_case: bool) -> Any:
        import torch

        values = logits.float()
        if collapse_case:
            values = torch.logaddexp(values[..., :4], values[..., 4:])
        else:
            values = values[..., 4:]
        return torch.log_softmax(values, dim=-1)

    def features(self, ids: list[list[int]], *, pooling: str) -> np.ndarray:
        import torch

        with torch.inference_mode():
            _, hidden, lengths = self.forward(ids)
            return _pool_hidden(hidden, lengths, pooling).cpu().numpy().copy()
