"""Evaluation-only readouts over the frozen deployment implementations.

The model adapter returns the original BF16 four-way logits and final hidden
states. Primary A/C/G/T scores apply FP32 softmax to those canonical logits,
preserving deployment head-precision differences. The additional full-vocabulary
AF readout uses FP32 projection/normalization, like PilotScorer's native readout.
It is streamed; no batch-by-length-by-vocabulary tensor is built.

Torch is deliberately imported lazily: Fast's freezing setting must precede
its first import. Run each backend in a fresh process. The owner of this
experiment validates the supplied local HF snapshot and runs GPU/JAX gates.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


CONFIGURATIONS = ("torch-fa2", "vllm-fa3", "fast-fa3")
INPUT_ALPHABET = "ACGTNRYSWKMBDHV"
INPUT_NORMALIZATION = "IUPAC ambiguity letters to N, one base per token, v1"


def dna_tokenizer(source: Any) -> Any:
    """Keep the five trained DNA IDs and normalize ambiguity letters to N."""
    from exp586_deployment.tokenizer import reduced_dna_unigram_tokenizer
    from tokenizers import Regex, normalizers

    tokenizer = reduced_dna_unigram_tokenizer(source)
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.Lowercase(), normalizers.Replace(Regex("[^acgtn]"), "n")]
    )
    return tokenizer


def _configure_environment(configuration: str) -> None:
    """Select freezing before Torch import; never switch a live compiler stack."""
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown deployment configuration: {configuration}")
    freezing = configuration == "fast-fa3"
    already_freezing = os.environ.get("TORCHINDUCTOR_FREEZING") == "1"
    if "torch" in sys.modules and freezing != already_freezing:
        raise RuntimeError(
            "Select TORCHINDUCTOR_FREEZING before importing Torch; "
            "run each configuration in a fresh process"
        )
    if freezing:
        os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    else:
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)


def _readout_model_class(configuration: str) -> type[Any]:
    """Subclass the actual backend, changing only the forward's return value."""
    import torch.nn.functional as F
    from exp586_deployment import FastMarinDNAMoE, MarinDNAMoE
    from exp586_deployment.model import rms_norm

    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown deployment configuration: {configuration}")
    base = FastMarinDNAMoE if configuration == "fast-fa3" else MarinDNAMoE

    class FinalHiddenModel(base):
        def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Keep this arithmetic identical to MarinDNAMoE.forward. Inherited
            # methods retain Fast's router and shared-projection overrides.
            cfg = self.config
            hidden = F.embedding(token_ids, self._weight("model.embed_tokens.weight"))
            hidden = self._gated_norm(
                "model.embed_gated_norm",
                rms_norm(
                    hidden, self._weight("model.embed_norm.weight"), cfg.layer_norm_eps
                ),
            )
            for layer in range(cfg.num_layers):
                prefix = f"model.layers.{layer}"
                is_global = (
                    layer + 1
                ) % cfg.global_every == 0 or layer == cfg.num_layers - 1
                attn_input = self._gated_norm(
                    f"{prefix}.attn_gated_norm",
                    rms_norm(
                        hidden,
                        self._weight(f"{prefix}.input_layernorm.weight"),
                        cfg.layer_norm_eps,
                    ),
                )
                attention = self._attention(layer, attn_input, is_global=is_global)
                hidden = hidden + self.short_conv(
                    attention,
                    self._weight(f"{prefix}.sconv_attn.weight"),
                )
                mlp_input = self._gated_norm(
                    f"{prefix}.mlp_gated_norm",
                    rms_norm(
                        hidden,
                        self._weight(f"{prefix}.post_attention_layernorm.weight"),
                        cfg.layer_norm_eps,
                    ),
                )
                mlp = self._moe(layer, mlp_input) + self._shared_experts(
                    layer, mlp_input
                )
                hidden = hidden + self.short_conv(
                    mlp,
                    self._weight(f"{prefix}.sconv_mlp.weight"),
                )
            hidden = self._gated_norm(
                "model.final_gated_norm",
                rms_norm(hidden, self._weight("model.norm.weight"), cfg.layer_norm_eps),
            )
            return F.linear(hidden, self._weight("lm_head.weight")), hidden

    return FinalHiddenModel


def _fp32_projection(hidden: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Accumulate and return FP32, rather than rounding a BF16 projection result."""
    import torch
    import torch.nn.functional as F

    if head.ndim != 2 or hidden.ndim < 1 or hidden.shape[-1] != head.shape[1]:
        raise ValueError("Hidden/head dimensions differ")
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        if (
            hidden.device.type == "cuda"
            and hidden.dtype == head.dtype
            and hidden.dtype in (torch.bfloat16, torch.float16)
        ):
            projected = torch.mm(
                hidden.reshape(-1, hidden.shape[-1]),
                head.t(),
                out_dtype=torch.float32,
            )
            return projected.reshape(*hidden.shape[:-1], head.shape[0])
        return F.linear(hidden.float(), head.float())


def streamed_full_log_probs(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    head: torch.Tensor,
    *,
    token_chunk_size: int = 1024,
    vocabulary_chunk_size: int = 4096,
) -> torch.Tensor:
    """Return original-vocabulary target log probabilities with bounded workspace.

    ``head`` has the original [vocabulary, hidden] orientation and may reside
    on CPU. Only a vocabulary slice and a token slice need to be projected at
    once. Selected logits come from the same FP32 blocks as the normalizer.
    CUDA BF16 operands use Torch's explicit FP32-output GEMM, matching the
    native readout's accumulation/output type without a BF16 rounding step.
    """
    import torch

    if token_chunk_size < 1 or vocabulary_chunk_size < 1:
        raise ValueError("Projection chunk sizes must be positive")
    if (
        head.ndim != 2
        or head.shape[0] < 1
        or hidden.shape[-1] != head.shape[1]
        or tuple(targets.shape) != tuple(hidden.shape[:-1])
        or targets.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Invalid full-head projection shapes or target dtype")
    if targets.numel() and (
        targets.min().item() < 0 or targets.max().item() >= head.shape[0]
    ):
        raise ValueError("Target ID is outside the original vocabulary")
    flat = hidden.reshape(-1, hidden.shape[-1])
    target_ids = targets.reshape(-1).to(device=hidden.device, dtype=torch.long)
    result = torch.empty(target_ids.shape, dtype=torch.float32, device=hidden.device)
    for start in range(0, len(flat), token_chunk_size):
        stop = min(start + token_chunk_size, len(flat))
        tokens = flat[start:stop]
        labels = target_ids[start:stop]
        normalizer = torch.full(
            (stop - start,), -torch.inf, dtype=torch.float32, device=hidden.device
        )
        selected = torch.zeros_like(normalizer)
        for vocabulary_start in range(0, head.shape[0], vocabulary_chunk_size):
            vocabulary_stop = min(
                vocabulary_start + vocabulary_chunk_size, head.shape[0]
            )
            weights = head[vocabulary_start:vocabulary_stop].to(hidden.device)
            logits = _fp32_projection(tokens, weights)
            normalizer = torch.logaddexp(normalizer, torch.logsumexp(logits, dim=-1))
            local = labels - vocabulary_start
            belongs = (local >= 0) & (local < logits.shape[-1])
            picked = logits.gather(1, local.clamp(0, logits.shape[-1] - 1)[:, None])[
                :, 0
            ]
            selected = torch.where(belongs, picked, selected)
        result[start:stop] = selected - normalizer
    return result.reshape(targets.shape)


@dataclass(frozen=True)
class _Batch:
    """A static model batch, with real examples and lengths retained separately."""

    compact_ids: np.ndarray
    original_ids: np.ndarray
    lengths: np.ndarray
    actual: int


def _prepare_batch(
    ids: list[list[int]],
    input_ids: np.ndarray,
    *,
    batch_size: int,
    sequence_length: int,
    minimum_length: int = 1,
) -> _Batch:
    """Right-pad with N and repeat the first real row to fill a partial batch."""
    if not ids or not 1 <= len(ids) <= batch_size:
        raise ValueError("Expected one nonempty batch no larger than batch_size")
    mapping = {int(original): compact for compact, original in enumerate(input_ids)}
    if len(mapping) != 5:
        raise ValueError("Need five distinct original A/C/G/T/N IDs")
    actual = len(ids)
    repeated = ids + [ids[0]] * (batch_size - actual)
    compact = np.full((batch_size, sequence_length), 4, dtype=np.int64)
    original = np.full_like(compact, int(input_ids[4]))
    lengths = np.empty(batch_size, dtype=np.int64)
    for row, tokens in enumerate(repeated):
        if not minimum_length <= len(tokens) <= sequence_length:
            raise ValueError("Real tokens must fit the static scoring length")
        if any(
            not isinstance(token, (int, np.integer)) or int(token) not in mapping
            for token in tokens
        ):
            raise ValueError("Input must contain original A/C/G/T/N token IDs only")
        lengths[row] = len(tokens)
        compact[row, : len(tokens)] = [mapping[int(token)] for token in tokens]
        original[row, : len(tokens)] = tokens
    return _Batch(compact, original, lengths, actual)


def _queries_and_features(
    positions: list[list[int]], feature_positions: list[int], batch: _Batch
) -> tuple[np.ndarray, np.ndarray]:
    """Preserve PilotScorer's query-zero sentinel and at least ten query columns."""
    actual = batch.actual
    if len(positions) != actual or len(feature_positions) != actual:
        raise ValueError("Request metadata count differs")
    for points, feature, length in zip(
        positions, feature_positions, batch.lengths[:actual], strict=True
    ):
        if (
            not 1 <= len(points) <= length
            or any(
                not isinstance(point, (int, np.integer)) or not 0 <= point < length
                for point in points
            )
            or not isinstance(feature, (int, np.integer))
            or not 0 <= feature < length
        ):
            raise ValueError("Invalid query or feature positions")
    queries = np.zeros((actual, max(10, max(map(len, positions)))), dtype=np.int64)
    for row, points in enumerate(positions):
        queries[row, : len(points)] = points
    return queries, np.asarray(feature_positions, dtype=np.int64)


def _pool_hidden(
    hidden: torch.Tensor, lengths: torch.Tensor, pooling: str
) -> torch.Tensor:
    """Pool real-token final states only, with no decoder targets as inputs."""
    import torch

    rows = torch.arange(hidden.shape[0], device=hidden.device)
    if pooling == "last":
        return hidden[rows, lengths - 1].float()
    if pooling != "mean":
        raise ValueError("Pooling must be last or mean")
    real = (
        torch.arange(hidden.shape[1], device=hidden.device)[None, :] < lengths[:, None]
    )
    masked = torch.where(real[..., None], hidden.float(), 0.0)
    return masked.sum(dim=1) / lengths[:, None].float()


def _initialize_scorer(
    scorer: Any,
    model_dir: Path,
    configuration: str,
    batch_size: int,
    sequence_length: int,
) -> None:
    """Set the common original-ID/device contract before either model is loaded."""
    if batch_size < 1 or sequence_length != 8192:
        raise ValueError("Use a positive batch size and the fixed 8192-token shape")
    _configure_environment(configuration)
    import torch
    from exp586_deployment.contract import token_contract
    from tokenizers import Tokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one H100 CUDA GPU per scorer process")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("These configurations were validated only on H100")
    source = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    contract = token_contract(source)
    scorer.base_ids = np.asarray(
        [contract.output_ids[base] for base in "ACGT"], dtype=np.int32
    )
    scorer.input_ids = np.asarray(
        [contract.input_ids[base] for base in "ACGTN"], dtype=np.int32
    )
    scorer.tokenizer = dna_tokenizer(source)
    scorer.batch_size, scorer.sequence_length = batch_size, sequence_length
    scorer.configuration, scorer.model_dir = configuration, Path(model_dir)
    scorer.device = torch.device("cuda:0")


def _compiler_settings(configuration: str) -> dict[str, Any]:
    """Keep the published compiler profile explicit for either readout boundary."""
    compiler: dict[str, Any] = {"fullgraph": True}
    if configuration == "fast-fa3":
        compiler.update(dynamic=False, mode="reduce-overhead")
    return compiler


def _model_settings(configuration: str) -> dict[str, Any]:
    """Spell out the pinned public API choices for each requested backend."""
    import torch

    if configuration not in CONFIGURATIONS:
        raise ValueError(f"Unknown deployment configuration: {configuration}")
    settings: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "input_vocab": "ACGTN",
        "output_vocab": "ACGT",
        "attention_provider": "flash_attn" if configuration == "torch-fa2" else "vllm",
        "flash_attention_version": 2 if configuration == "torch-fa2" else 3,
        "moe_backend": "torch" if configuration == "torch-fa2" else "vllm",
    }
    if configuration != "fast-fa3":
        settings.update(sconv_backend="direct", cache_rope=True)
    return settings


class CanonicalScorer:
    """Probability-only scorer over the literal unchanged deployment model.

    NativeAdapter.score_positions reads only ``probabilities``. This boundary
    deliberately exposes no features/AF placeholders and rejects AF requests.
    Run it in a different process from BackendScorer to avoid loading two
    backbones or changing the original graph's output signature.
    """

    def __init__(
        self,
        model_dir: Path,
        configuration: str,
        batch_size: int = 64,
        sequence_length: int = 8192,
        *,
        compile_model: bool = True,
    ) -> None:
        _initialize_scorer(self, model_dir, configuration, batch_size, sequence_length)
        import torch
        from exp586_deployment import FastMarinDNAMoE, MarinDNAMoE

        cls = FastMarinDNAMoE if configuration == "fast-fa3" else MarinDNAMoE
        self.model = cls(
            model_dir,
            self.device,
            **_model_settings(configuration),
        )
        self._forward = (
            torch.compile(self.model, **_compiler_settings(configuration))
            if compile_model
            else self.model
        )

    def evaluate(
        self,
        ids: list[list[int]],
        positions: list[list[int]],
        feature_positions: list[int],
        *,
        allele_frequency: bool,
    ) -> dict[str, np.ndarray]:
        """Return only canonical-head probabilities; never run a hidden forward."""
        if allele_frequency:
            raise ValueError(
                "CanonicalScorer is probability-only; AF requires BackendScorer"
            )
        if len({len(tokens) for tokens in ids}) != 1:
            raise ValueError("Scoring requires one batch of equal-length windows")
        import torch

        batch = _prepare_batch(
            ids,
            self.input_ids,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            minimum_length=2,
        )
        queries, _ = _queries_and_features(positions, feature_positions, batch)
        with torch.inference_mode():
            logits = self._forward(
                torch.as_tensor(batch.compact_ids, device=self.device)
            )
            logs = torch.log_softmax(
                logits[: batch.actual, : int(batch.lengths[0]) - 1].float(), dim=-1
            )
            query_indices = torch.as_tensor(queries, device=self.device)
            rows = torch.arange(batch.actual, device=self.device)[:, None]
            probabilities = logs[rows, (query_indices - 1).clamp(min=0)].exp()
            probabilities = torch.where(
                query_indices[..., None] > 0, probabilities, 0.0
            )
            result = probabilities.cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite canonical probabilities")
        return {"probabilities": result}


class BackendScorer:
    """PilotScorer-compatible outputs using an unchanged deployment backbone.

    ``model`` and compiled ``_forward`` return (canonical_bf16_logits, hidden),
    allowing an independent original-logit gate. ``evaluate`` uses those BF16
    logits with FP32 softmax for primary probabilities; only the additional
    full-vocabulary readout projects hidden states in FP32. Returned arrays are CPU
    copies, safe across later CUDA-graph invocations. ``full_logs`` is zero when
    allele_frequency is false; other readouts keep their real-data shapes.
    """

    def __init__(
        self,
        model_dir: Path,
        configuration: str,
        batch_size: int = 64,
        sequence_length: int = 8192,
        *,
        compile_model: bool = True,
    ) -> None:
        _initialize_scorer(self, model_dir, configuration, batch_size, sequence_length)
        import torch

        self.model = _readout_model_class(configuration)(
            model_dir,
            self.device,
            **_model_settings(configuration),
        )
        self._dna_head = self.model._weight("lm_head.weight")
        self._full_head: torch.Tensor | None = None
        self._forward = (
            torch.compile(self.model, **_compiler_settings(configuration))
            if compile_model
            else self.model
        )

    def _load_full_head(self) -> torch.Tensor:
        """Load the pinned original head, rather than the four-row model buffer."""
        from safetensors import safe_open

        if self._full_head is None:
            index = json.loads(
                (self.model_dir / "model.safetensors.index.json").read_text()
            )
            shard = index["weight_map"]["lm_head.weight"]
            with safe_open(
                self.model_dir / shard, framework="pt", device="cpu"
            ) as handle:
                head = handle.get_tensor("lm_head.weight")
            expected = (self.model.config.vocab_size, self.model.config.hidden_dim)
            if tuple(head.shape) != expected:
                raise ValueError(f"Original full-head shape differs from {expected}")
            # About 188 MiB for the pinned BF16 head: retaining it avoids copying
            # the whole vocabulary from CPU for every token chunk.
            self._full_head = head.to(self.device)
        return self._full_head

    def evaluate(
        self,
        ids: list[list[int]],
        positions: list[list[int]],
        feature_positions: list[int],
        *,
        allele_frequency: bool,
    ) -> dict[str, np.ndarray]:
        """Score one equal-length batch; query p predicts token p from state p-1."""
        import torch

        if len({len(tokens) for tokens in ids}) != 1:
            raise ValueError("Scoring requires one batch of equal-length windows")
        batch = _prepare_batch(
            ids,
            self.input_ids,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            minimum_length=2,
        )
        queries, features = _queries_and_features(positions, feature_positions, batch)
        actual, real_length = batch.actual, int(batch.lengths[0])
        with torch.inference_mode():
            compact = torch.as_tensor(batch.compact_ids, device=self.device)
            canonical_logits, hidden = self._forward(compact)
            hidden = hidden[:actual]
            dna_logs = torch.log_softmax(
                canonical_logits[:actual, : real_length - 1].float(), dim=-1
            )
            query_indices = torch.as_tensor(queries, device=self.device)
            rows = torch.arange(actual, device=self.device)[:, None]
            probabilities = dna_logs[rows, (query_indices - 1).clamp(min=0)].exp()
            probabilities = torch.where(
                query_indices[..., None] > 0, probabilities, 0.0
            )
            targets = torch.as_tensor(
                batch.original_ids[:actual, 1:real_length], device=self.device
            )
            base_ids = torch.as_tensor(self.base_ids, device=self.device)
            matches = targets[..., None] == base_ids
            valid = matches.any(dim=-1)
            acgt_logs = dna_logs.gather(-1, matches.long().argmax(dim=-1)[..., None])[
                ..., 0
            ]
            full_logs = (
                streamed_full_log_probs(
                    hidden[:, : real_length - 1], targets, self._load_full_head()
                )
                if allele_frequency
                else torch.zeros_like(acgt_logs)
            )
            lengths = torch.as_tensor(batch.lengths[:actual], device=self.device)
            mean = _pool_hidden(hidden, lengths, "mean")
            variant = hidden[
                torch.arange(actual, device=self.device),
                torch.as_tensor(features, device=self.device),
            ].float()
            result = {
                name: value.cpu().numpy().copy()
                for name, value in {
                    "probabilities": probabilities,
                    "acgt_logs": acgt_logs,
                    "full_logs": full_logs,
                    "valid": valid,
                    "mean": mean,
                    "variant": variant,
                }.items()
            }
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Nonfinite deployment evaluation readouts")
        return result

    def features(self, sequences: list[str], pooling: str) -> np.ndarray:
        """Return last/mean real-token hidden states, accepting duplicate windows."""
        import torch

        if pooling not in ("last", "mean"):
            raise ValueError("Pooling must be last or mean")
        if not sequences:
            return np.empty((0, self.model.config.hidden_dim), dtype=np.float32)
        if any(
            not sequence or set(sequence.upper()) - set(INPUT_ALPHABET)
            for sequence in sequences
        ):
            raise ValueError("Feature sequences must be nonempty IUPAC DNA strings")
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(sequences), self.batch_size):
                group = sequences[start : start + self.batch_size]
                ids = [
                    self.tokenizer.encode(sequence, add_special_tokens=False).ids
                    for sequence in group
                ]
                if any(
                    len(tokens) != len(sequence)
                    for tokens, sequence in zip(ids, group, strict=True)
                ):
                    raise ValueError(
                        "Tokenizer must emit one original ID per nucleotide"
                    )
                batch = _prepare_batch(
                    ids,
                    self.input_ids,
                    batch_size=self.batch_size,
                    sequence_length=self.sequence_length,
                )
                _, hidden = self._forward(
                    torch.as_tensor(batch.compact_ids, device=self.device)
                )
                pooled = _pool_hidden(
                    hidden[: batch.actual],
                    torch.as_tensor(batch.lengths[: batch.actual], device=self.device),
                    pooling,
                )
                outputs.append(pooled.cpu().numpy().copy())
        result = np.concatenate(outputs)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite frozen features")
        return result
