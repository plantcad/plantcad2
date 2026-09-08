"""Final-layer REF/ALT representations using the zero-shot runner's prefix cache."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from scoring import COMPLEMENT, _repeat_cache, _tokenize, extract_window, variant_position


def _backbone(model: Any) -> Any:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("Expected an HF causal LM with a `.model` base transformer")
    return backbone


def embed_cached(model: Any, reference_ids: torch.Tensor, alternate_ids: torch.Tensor, var_pos: int) -> dict[str, torch.Tensor]:
    """Return REF/ALT whole-window means and variant-token states in FP32."""
    batch, length = reference_ids.shape
    prefix_ids = reference_ids[:, :var_pos].contiguous()
    reference_suffix = reference_ids[:, var_pos:].contiguous()
    alternate_suffix = torch.cat([alternate_ids.unsqueeze(-1), reference_suffix[:, 1:]], dim=-1)
    suffixes = rearrange(torch.stack([reference_suffix, alternate_suffix], dim=1), "B A L -> (B A) L").contiguous()

    backbone = _backbone(model)
    prefix = backbone(prefix_ids, use_cache=True)
    prefix_sum = prefix.last_hidden_state.sum(dim=1, dtype=torch.float32)
    cache = _repeat_cache(prefix.past_key_values, 2)
    suffix = backbone(suffixes, past_key_values=cache, use_cache=False).last_hidden_state
    suffix = rearrange(suffix, "(B A) L D -> B A L D", B=batch, A=2)
    suffix_sum = suffix.sum(dim=2, dtype=torch.float32)
    whole = (prefix_sum[:, None, :] + suffix_sum) / length
    variant = suffix[:, :, 0, :].float()
    return {"whole_window": whole, "variant_token": variant}


def embed_uncached(model: Any, reference_ids: torch.Tensor, alternate_ids: torch.Tensor, var_pos: int) -> dict[str, torch.Tensor]:
    """Uncached parity reference for the cached representation kernel."""
    batch, length = reference_ids.shape
    alternate = torch.cat([reference_ids[:, :var_pos], alternate_ids.unsqueeze(-1), reference_ids[:, var_pos + 1 :]], dim=-1)
    pairs = rearrange(torch.stack([reference_ids, alternate], dim=1), "B A L -> (B A) L").contiguous()
    hidden = _backbone(model)(pairs, use_cache=False).last_hidden_state
    hidden = rearrange(hidden, "(B A) L D -> B A L D", B=batch, A=2)
    return {
        "whole_window": hidden.sum(dim=2, dtype=torch.float32) / length,
        "variant_token": hidden[:, :, var_pos, :].float(),
    }


def _batch_inputs(tokenizer: Any, fasta: Any, frame: pd.DataFrame, window_size: int, strand: Literal["fwd", "rc"]) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = [extract_window(fasta, str(row.chrom), int(row.pos), str(row.ref).upper(), window_size, strand) for row in frame.itertuples(index=False)]
    reference_ids = _tokenize(tokenizer, sequences, "cuda:0")
    alt_bases = [str(row.alt).upper() if strand == "fwd" else str(row.alt).upper().translate(COMPLEMENT) for row in frame.itertuples(index=False)]
    alternate_ids = torch.tensor([tokenizer.encode(base, add_special_tokens=False)[0] for base in alt_bases], device="cuda:0")
    return reference_ids, alternate_ids


def embed_rows(
    model: Any,
    tokenizer: Any,
    fasta: Any,
    frame: pd.DataFrame,
    *,
    window_size: int,
    batch_size: int,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Extract FP32 FWD/RC-averaged REF/ALT representations for a frame."""
    by_strand: dict[str, dict[str, list[np.ndarray]]] = {
        strand: {name: [] for name in ("whole_window", "variant_token")} for strand in ("fwd", "rc")
    }
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for strand in ("fwd", "rc"):
            for start in range(0, len(frame), batch_size):
                batch = frame.iloc[start : start + batch_size]
                reference_ids, alternate_ids = _batch_inputs(tokenizer, fasta, batch, window_size, strand)
                result = embed_cached(model, reference_ids, alternate_ids, variant_position(window_size, strand))
                for name, value in result.items():
                    by_strand[strand][name].append(value.cpu().numpy())
                if progress and (start == 0 or start + len(batch) == len(frame) or (start // batch_size + 1) % 10 == 0):
                    progress(strand, start + len(batch), len(frame))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output: dict[str, np.ndarray] = {}
    for name in ("whole_window", "variant_token"):
        fwd = np.concatenate(by_strand["fwd"][name])
        rc = np.concatenate(by_strand["rc"][name])
        averaged = (fwd.astype(np.float32) + rc.astype(np.float32)) / np.float32(2)
        output[f"{name}_ref"] = averaged[:, 0]
        output[f"{name}_alt"] = averaged[:, 1]
    return output, {"wall_seconds": elapsed, "variants": len(frame), "strand_forwards": 2 * len(frame), "variants_per_second": len(frame) / elapsed}


def parity_check(model: Any, tokenizer: Any, fasta: Any, frame: pd.DataFrame, *, window_size: int) -> dict[str, Any]:
    """Compare cached and paired-uncached hidden-state reductions on a small batch."""
    rows: dict[str, Any] = {}
    with torch.inference_mode():
        for strand in ("fwd", "rc"):
            reference_ids, alternate_ids = _batch_inputs(tokenizer, fasta, frame, window_size, strand)
            pos = variant_position(window_size, strand)
            cached = embed_cached(model, reference_ids, alternate_ids, pos)
            uncached = embed_uncached(model, reference_ids, alternate_ids, pos)
            rows[strand] = {
                name: {
                    "max_abs": float((cached[name] - uncached[name]).abs().max()),
                    "mean_abs": float((cached[name] - uncached[name]).abs().mean()),
                    "cosine_min": float(torch.nn.functional.cosine_similarity(cached[name].flatten(0, 1), uncached[name].flatten(0, 1), dim=-1).min()),
                }
                for name in cached
            }
    return rows


def profile_embeddings(model: Any, tokenizer: Any, fasta: Any, frame: pd.DataFrame, *, window_size: int) -> dict[str, Any]:
    reference_ids, alternate_ids = _batch_inputs(tokenizer, fasta, frame.iloc[:1], window_size, "fwd")
    pos = variant_position(window_size, "fwd")
    with torch.inference_mode():
        embed_cached(model, reference_ids, alternate_ids, pos)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.inference_mode():
            embed_cached(model, reference_ids, alternate_ids, pos)
        torch.cuda.synchronize()
    events = sorted({event.key for event in profiler.key_averages()})
    flash = [event for event in events if "flash" in event.lower()]
    external = [event for event in flash if "flash_attn" in event.lower() or "flash::" in event.lower()]
    return {
        "resolved_attention": getattr(model.config, "_attn_implementation", None),
        "flash_events": flash,
        "external_fa2_events": external,
        "external_flash_attention_2_verified": bool(external),
    }
