#!/usr/bin/env python3
"""
Interactive throughput / memory probe for the causal zero-shot evals.

Drives the real code path from zero-shot-eval-causal.py (same model loader, same
_causal_probs_at_positions inner loop) on synthetic sequences, so the measured
seconds/sequence include the softmax and the device->host copy -- not just the
bare forward pass.

Typical use on TACC Vista:

    idev -p gh -N 1 -n 1 -t 02:00:00
    source slurms/config.sh            # modules + venv + MODEL/HF_HOME
    python bench_evo2_speed.py --batch_sizes 1,2,4,8,16 --project

Notes
-----
* The first call also times model load; --warmup batches are discarded.
* --seq_len 4097 measures the "truncate to max(mask_idx)+1" variant, which is
  mathematically identical to 8192 under causal masking for motif_acc/evo_cons/
  core_noncore (logits at index p-1 depend only on tokens 0..p-1).
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Full test-split sizes, from the Carbon leaderboard run summaries.
TASK_SIZES = {
    "recovery": {
        "tis_recovery/test_maize": 39035,
        "tis_recovery/test_tomato": 35484,
        "tts_recovery/test_maize": 39035,
        "tts_recovery/test_tomato": 35483,
        "donor_recovery/test_maize": 153869,
        "donor_recovery/test_tomato": 140456,
        "acceptor_recovery/test_maize": 153869,
        "acceptor_recovery/test_tomato": 140455,
    },
    "core_noncore": {
        "tis_core_noncore/test_maize": 36409,
        "tis_core_noncore/test_tomato": 35478,
        "tts_core_noncore/test_maize": 36409,
        "tts_core_noncore/test_tomato": 35477,
        "donor_core_noncore/test_maize": 144550,
        "donor_core_noncore/test_tomato": 140451,
        "acceptor_core_noncore/test_maize": 144550,
        "acceptor_core_noncore/test_tomato": 140451,
    },
    "conservation": {
        "within_poaceae_tis/test": 36662,
        "within_poaceae_non_tis/test": 183685,
        "within_andropogoneae/test": 38060,
    },
    "sv_effect": {
        "structural_variant_effect_prediction/test": 18075,  # x2 passes (ref+mut)
    },
}


def load_eval_module():
    """Import zero-shot-eval-causal.py (dashes in the name block a plain import)."""
    path = Path(__file__).resolve().with_name("zero-shot-eval-causal.py")
    if not path.exists():
        sys.exit(f"Cannot find {path}")
    spec = importlib.util.spec_from_file_location("zs_causal", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def random_seqs(n, seq_len, seed=0):
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    return ["".join(bases[rng.integers(0, 4, seq_len)]) for _ in range(n)]


def bench_one(zs, model, tok, device, batch_size, seq_len, positions, n_batches, warmup):
    """Time the inner loop of _causal_probs_at_positions for one (batch_size, seq_len)."""
    idxs = [tok.get_vocab()[n] for n in zs.NUCLEOTIDES_LOWER]
    seqs = random_seqs(batch_size, seq_len)

    torch.cuda.reset_peak_memory_stats()
    times = []
    for it in range(warmup + n_batches):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        enc = tok(
            seqs,
            truncation=False,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"].to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits
        if it == 0:
            # Catch shape/unwrapping surprises before reporting a throughput number.
            assert logits.ndim == 3, f"expected [B, L, vocab] logits, got {type(logits)} {getattr(logits, 'shape', None)}"
            assert logits.shape[:2] == (batch_size, seq_len), f"got {tuple(logits.shape)}"
            assert logits.shape[2] > max(idxs), f"vocab {logits.shape[2]} too small for {max(idxs)}"
        # Mirror _causal_probs_at_positions exactly, including the .cpu() sync.
        next_logits = logits[:, :-1, :][:, :, idxs]
        next_probs = torch.softmax(next_logits.float(), dim=-1).cpu().numpy()
        _ = [next_probs[b, p - 1, :] for b in range(next_probs.shape[0]) for p in positions]

        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if it >= warmup:
            times.append(dt)

    times = np.array(times)
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "sec_per_batch_median": float(np.median(times)),
        "sec_per_batch_min": float(times.min()),
        "sec_per_seq": float(np.median(times)) / batch_size,
        "seq_per_sec": batch_size / float(np.median(times)),
        "tokens_per_sec": batch_size * seq_len / float(np.median(times)),
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


def project(seq_per_sec, n_context_modes, wall_hours):
    print(f"\n=== Projection at {seq_per_sec:.3f} seq/s, {n_context_modes} context mode(s) ===")
    print(f"{'job':14} {'worst element':44} {'fwd passes':>12} {'hours':>9}  fits {wall_hours}h?")
    for job, sizes in TASK_SIZES.items():
        modes = 1 if job == "sv_effect" else n_context_modes
        mult = 2 if job == "sv_effect" else 1  # ref + mut
        worst_k, worst_n = max(sizes.items(), key=lambda kv: kv[1])
        passes = worst_n * modes * mult
        hours = passes / seq_per_sec / 3600
        print(f"{job:14} {worst_k:44} {passes:>12,} {hours:>9.1f}  {'YES' if hours <= wall_hours else 'NO'}")
        total = sum(sizes.values()) * modes * mult
        print(f"{'':14} {'(all elements combined)':44} {total:>12,} {total / seq_per_sec / 3600:>9.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="evo2_20b", help="Evo2 name or HF repo id")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--use_kernels",
        action="store_true",
        help="Enable Vortex's opt-in Triton HC{S,M,L} kernels (needs vtx>=1.1.0). "
        "Worth timing both ways; validate outputs before using in a real run.",
    )
    ap.add_argument("--batch_sizes", default="1,2,4,8,16")
    ap.add_argument("--seq_lens", default="8192", help="comma-separated; try 8192,4097")
    ap.add_argument("--mask_idx", default="4094,4095,4096")
    ap.add_argument("--n_batches", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--project", action="store_true", help="extrapolate to full job sizes")
    ap.add_argument("--context_modes", type=int, default=4)
    ap.add_argument("--wall_hours", type=float, default=12.0)
    ap.add_argument("--json_out", default="")
    args = ap.parse_args()

    zs = load_eval_module()
    device = zs._require_cuda(args.device)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name()}  total={total / 1024**3:.1f} GiB  free={free / 1024**3:.1f} GiB")

    t0 = time.perf_counter()
    model, tok = zs._load_model(args.model, device, use_kernels=args.use_kernels)
    print(f"Model load: {time.perf_counter() - t0:.1f} s")
    free_after, _ = torch.cuda.mem_get_info()
    print(f"Weights resident: ~{(free - free_after) / 1024**3:.1f} GiB\n")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    positions = [int(x) for x in args.mask_idx.split(",")]

    header = f"{'B':>4} {'L':>6} {'s/batch':>9} {'s/seq':>8} {'seq/s':>8} {'tok/s':>10} {'peak GiB':>9}"
    print(header)
    print("-" * len(header))

    results = []
    best_by_len = {}
    for seq_len in seq_lens:
        pos = [p for p in positions if p < seq_len] or [seq_len - 1]
        for bs in batch_sizes:
            try:
                r = bench_one(zs, model, tok, device, bs, seq_len, pos, args.n_batches, args.warmup)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # B*L past INT_MAX trips vortex's FIR conv with a canUse32BitIndexMath
                # error, which is a plain RuntimeError rather than an OOM. Either way the
                # larger batch sizes are unusable at this length -- record and move on.
                torch.cuda.empty_cache()
                reason = "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else str(exc).split("\n")[0][:60]
                print(f"{bs:>4} {seq_len:>6}   FAILED ({reason}) -- stopping sweep at this length")
                break
            results.append(r)
            print(
                f"{r['batch_size']:>4} {r['seq_len']:>6} {r['sec_per_batch_median']:>9.3f} "
                f"{r['sec_per_seq']:>8.3f} {r['seq_per_sec']:>8.2f} {r['tokens_per_sec']:>10,.0f} "
                f"{r['peak_reserved_gb']:>9.1f}"
            )
            cur = best_by_len.get(seq_len)
            if cur is None or r["seq_per_sec"] > cur["seq_per_sec"]:
                best_by_len[seq_len] = r

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"model": args.model, "results": results}, indent=2))
        print(f"\nWrote {args.json_out}")

    if args.project:
        for seq_len in sorted(best_by_len, reverse=True):
            b = best_by_len[seq_len]
            print(f"\nBest at L={seq_len}: B={b['batch_size']} -> {b['seq_per_sec']:.3f} seq/s")
            project(b["seq_per_sec"], args.context_modes, args.wall_hours)


if __name__ == "__main__":
    main()
