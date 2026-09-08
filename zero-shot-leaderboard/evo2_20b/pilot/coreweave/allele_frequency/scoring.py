"""Full-sequence SNV LLR scoring with BF16 FA2 and FP32 accumulation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import DynamicCache


DNA = "ACGT"
COMPLEMENT = str.maketrans("ACGTRYMKBDHVN", "TGCAYRKMVHDBN")


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def variant_position(window_size: int, strand: Literal["fwd", "rc"]) -> int:
    return window_size // 2 if strand == "fwd" else window_size - 1 - window_size // 2


def extract_window(fasta: Any, chrom: str, pos: int, ref: str, window_size: int, strand: Literal["fwd", "rc"]) -> str:
    """Extract an even centered window using the newer MarinDNA VCF convention."""
    center_index = pos - 1
    fwd_position = window_size // 2
    start = center_index - fwd_position
    stop = start + window_size
    chrom_size = len(fasta[chrom])
    sequence = "N" * max(0, -start) + str(fasta[chrom][max(0, start) : min(chrom_size, stop)]) + "N" * max(0, stop - chrom_size)
    sequence = sequence.upper()
    if len(sequence) != window_size:
        raise RuntimeError(f"Wrong window length for {chrom}:{pos}: {len(sequence)}")
    if strand == "rc":
        sequence = reverse_complement(sequence)
        expected = ref.translate(COMPLEMENT)
    else:
        expected = ref
    if sequence[variant_position(window_size, strand)] != expected:
        raise RuntimeError(f"Reference mismatch at {chrom}:{pos} ({strand})")
    return sequence


def validate_center_crop(fasta: Any, row: Any, window_size: int, full_window_size: int = 8192) -> dict[str, int | bool]:
    """Prove that a smaller window is the centered crop and preserves the mirrored variant index."""
    if window_size > full_window_size or window_size % 2 or full_window_size % 2:
        raise ValueError("Context lengths must be even and no larger than the full window")
    chrom, pos, ref = str(row.chrom), int(row.pos), str(row.ref).upper()
    full = extract_window(fasta, chrom, pos, ref, full_window_size, "fwd")
    cropped = extract_window(fasta, chrom, pos, ref, window_size, "fwd")
    crop_start = full_window_size // 2 - window_size // 2
    if cropped != full[crop_start : crop_start + window_size]:
        raise RuntimeError(f"Window is not a centered crop at {chrom}:{pos}")
    reverse = extract_window(fasta, chrom, pos, ref, window_size, "rc")
    if reverse != reverse_complement(cropped):
        raise RuntimeError(f"Reverse-complement crop mismatch at {chrom}:{pos}")
    fwd_index = variant_position(window_size, "fwd")
    rc_index = variant_position(window_size, "rc")
    if cropped[fwd_index] != ref or reverse[rc_index] != ref.translate(COMPLEMENT):
        raise RuntimeError(f"Variant index shifted during crop at {chrom}:{pos}")
    return {
        "full_window_size": full_window_size,
        "window_size": window_size,
        "full_crop_start": crop_start,
        "forward_variant_index_zero_based": fwd_index,
        "reverse_complement_variant_index_zero_based": rc_index,
        "entire_window_matches_center_crop": True,
        "reverse_complement_exact": True,
    }


def nucleotide_token_ids(tokenizer: Any, device: torch.device | str) -> torch.Tensor:
    ids = [tokenizer.encode(base, add_special_tokens=False)[0] for base in DNA]
    if len(set(ids)) != 4:
        raise RuntimeError(f"Tokenizer does not have four distinct nucleotide IDs: {ids}")
    return torch.tensor(ids, dtype=torch.long, device=device)


def _tokenize(tokenizer: Any, sequences: list[str], device: torch.device | str) -> torch.Tensor:
    ids = tokenizer(
        sequences,
        add_special_tokens=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )["input_ids"]
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1 or ids.shape != (len(sequences), next(iter(lengths))):
        raise RuntimeError(f"Expected character-level tokenization, got {tuple(ids.shape)}")
    return ids.to(device)


def _token_to_nucleotide_index(targets: torch.Tensor, nucleotide_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matches = targets.unsqueeze(-1) == nucleotide_ids
    valid = matches.any(dim=-1)
    return matches.to(torch.int64).argmax(dim=-1), valid


def _repeat_cache(cache: Any, repeats: int) -> DynamicCache:
    if hasattr(cache, "batch_repeat_interleave"):
        cache.batch_repeat_interleave(repeats)
        return cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for index in range(len(cache.key_cache)):
            cache.key_cache[index] = cache.key_cache[index].repeat_interleave(repeats, dim=0)
            cache.value_cache[index] = cache.value_cache[index].repeat_interleave(repeats, dim=0)
        return cache
    result = DynamicCache()
    for layer, (key, value) in enumerate(cache):
        result.update(key.repeat_interleave(repeats, dim=0), value.repeat_interleave(repeats, dim=0), layer_idx=layer)
    return result


def _selected_log_probability(logits: torch.Tensor, targets: torch.Tensor, nucleotide_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 A/C/G/T conditional log-probability and an A/C/G/T target mask."""
    indices, valid = _token_to_nucleotide_index(targets, nucleotide_ids)
    log_probability = F.log_softmax(logits.index_select(-1, nucleotide_ids).float(), dim=-1)
    selected = log_probability.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    return selected, valid


def _full_log_probability(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def score_cached(model: Any, reference_ids: torch.Tensor, alternate_ids: torch.Tensor, var_pos: int, nucleotide_ids: torch.Tensor) -> torch.Tensor:
    """Prefix-shared exact-sequence LLR atoms; returns [ACGT, full-vocab, n-ambiguous]."""
    batch, length = reference_ids.shape
    prefix = reference_ids[:, :var_pos].contiguous()
    reference_suffix = reference_ids[:, var_pos:].contiguous()
    alternate_suffix = torch.cat([alternate_ids.unsqueeze(-1), reference_suffix[:, 1:]], dim=-1)
    suffixes = torch.stack([reference_suffix, alternate_suffix], dim=1)
    suffixes_flat = rearrange(suffixes, "B V L -> (B V) L").contiguous()

    prefix_output = model(prefix, use_cache=True, logits_to_keep=1)
    prefix_logits = prefix_output.logits[:, -1]
    cache = _repeat_cache(prefix_output.past_key_values, 2)
    suffix_logits = model(suffixes_flat, past_key_values=cache, use_cache=False).logits
    suffix_logits = rearrange(suffix_logits, "(B V) L C -> B V L C", B=batch)

    reference_at_variant = reference_ids[:, var_pos]
    acgt_prefix_ref, prefix_ref_valid = _selected_log_probability(prefix_logits, reference_at_variant, nucleotide_ids)
    acgt_prefix_alt, prefix_alt_valid = _selected_log_probability(prefix_logits, alternate_ids, nucleotide_ids)
    if not (prefix_ref_valid & prefix_alt_valid).all():
        raise RuntimeError("REF/ALT alleles must be A/C/G/T")
    full_prefix_ref = _full_log_probability(prefix_logits, reference_at_variant)
    full_prefix_alt = _full_log_probability(prefix_logits, alternate_ids)

    targets = reference_ids[:, var_pos + 1 :]
    reference_logits = suffix_logits[:, 0, :-1]
    alternate_logits = suffix_logits[:, 1, :-1]
    acgt_ref, valid = _selected_log_probability(reference_logits, targets, nucleotide_ids)
    acgt_alt, valid_alt = _selected_log_probability(alternate_logits, targets, nucleotide_ids)
    if not torch.equal(valid, valid_alt):
        raise RuntimeError("REF/ALT downstream target masks differ")
    acgt_downstream = torch.where(valid, acgt_alt - acgt_ref, 0.0).sum(dim=-1, dtype=torch.float32)
    full_downstream = (_full_log_probability(alternate_logits, targets) - _full_log_probability(reference_logits, targets)).sum(dim=-1, dtype=torch.float32)
    return torch.stack(
        [
            acgt_prefix_alt - acgt_prefix_ref + acgt_downstream,
            full_prefix_alt - full_prefix_ref + full_downstream,
            (~valid).sum(dim=-1).float(),
        ],
        dim=-1,
    )


def score_uncached(model: Any, reference_ids: torch.Tensor, alternate_ids: torch.Tensor, var_pos: int, nucleotide_ids: torch.Tensor) -> torch.Tensor:
    """Paired uncached full forwards, retaining only the non-cancelling suffix."""
    batch = reference_ids.shape[0]
    alternate = torch.cat([reference_ids[:, :var_pos], alternate_ids.unsqueeze(-1), reference_ids[:, var_pos + 1 :]], dim=-1)
    pairs = torch.stack([reference_ids, alternate], dim=1)
    flat = rearrange(pairs, "B V L -> (B V) L").contiguous()
    logits = model(flat, use_cache=False).logits[:, :-1]
    targets = flat[:, 1:]
    acgt, valid = _selected_log_probability(logits, targets, nucleotide_ids)
    full = _full_log_probability(logits, targets)
    acgt = rearrange(acgt, "(B V) L -> B V L", B=batch)
    valid = rearrange(valid, "(B V) L -> B V L", B=batch)
    full = rearrange(full, "(B V) L -> B V L", B=batch)
    if not torch.equal(valid[:, 0], valid[:, 1]):
        raise RuntimeError("REF/ALT target masks differ")
    start = var_pos - 1
    acgt_llr = torch.where(valid[:, 0, start:], acgt[:, 1, start:] - acgt[:, 0, start:], 0.0).sum(dim=-1, dtype=torch.float32)
    full_llr = (full[:, 1, start:] - full[:, 0, start:]).sum(dim=-1, dtype=torch.float32)
    return torch.stack([acgt_llr, full_llr, (~valid[:, 0, start:]).sum(dim=-1).float()], dim=-1)


def score_rows(
    model: Any,
    tokenizer: Any,
    fasta: Any,
    frame: pd.DataFrame,
    *,
    window_size: int,
    batch_size: int,
    method: Literal["cached", "uncached"],
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    scorer = score_cached if method == "cached" else score_uncached
    nucleotide_ids = nucleotide_token_ids(tokenizer, "cuda:0")
    outputs = {strand: [] for strand in ("fwd", "rc")}
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for strand in ("fwd", "rc"):
            for start in range(0, len(frame), batch_size):
                batch = frame.iloc[start : start + batch_size]
                sequences = [extract_window(fasta, str(row.chrom), int(row.pos), str(row.ref).upper(), window_size, strand) for row in batch.itertuples(index=False)]
                reference_ids = _tokenize(tokenizer, sequences, "cuda:0")
                alt_bases = [str(row.alt).upper() if strand == "fwd" else str(row.alt).upper().translate(COMPLEMENT) for row in batch.itertuples(index=False)]
                alt_ids = torch.tensor([tokenizer.encode(base, add_special_tokens=False)[0] for base in alt_bases], device="cuda:0")
                result = scorer(model, reference_ids, alt_ids, variant_position(window_size, strand), nucleotide_ids)
                outputs[strand].append(result.cpu().numpy())
                if progress and (start == 0 or start + len(batch) == len(frame) or (start // batch_size + 1) % 10 == 0):
                    progress(strand, start + len(batch), len(frame))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    arrays = {strand: np.concatenate(parts) for strand, parts in outputs.items()}
    return arrays, {"wall_seconds": elapsed, "variants": len(frame), "strand_forwards": 2 * len(frame), "variants_per_second": len(frame) / elapsed}


def profile_scoring(model: Any, tokenizer: Any, fasta: Any, frame: pd.DataFrame, *, window_size: int, method: Literal["cached", "uncached"]) -> dict[str, Any]:
    sequence = extract_window(fasta, str(frame.iloc[0]["chrom"]), int(frame.iloc[0]["pos"]), str(frame.iloc[0]["ref"]).upper(), window_size, "fwd")
    reference_ids = _tokenize(tokenizer, [sequence], "cuda:0")
    alt_id = torch.tensor([tokenizer.encode(str(frame.iloc[0]["alt"]).upper(), add_special_tokens=False)[0]], device="cuda:0")
    nucleotide_ids = nucleotide_token_ids(tokenizer, "cuda:0")
    scorer = score_cached if method == "cached" else score_uncached
    with torch.inference_mode():
        scorer(model, reference_ids, alt_id, variant_position(window_size, "fwd"), nucleotide_ids)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.inference_mode():
            scorer(model, reference_ids, alt_id, variant_position(window_size, "fwd"), nucleotide_ids)
        torch.cuda.synchronize()
    events = sorted({event.key for event in profiler.key_averages()})
    flash = [event for event in events if "flash" in event.lower()]
    external = [event for event in flash if "flash_attn" in event.lower() or "flash::" in event.lower()]
    return {"method": method, "resolved_attention": getattr(model.config, "_attn_implementation", None), "flash_events": flash, "external_fa2_events": external, "external_flash_attention_2_verified": bool(external)}
