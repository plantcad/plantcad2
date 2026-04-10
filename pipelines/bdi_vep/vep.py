"""
Variant Effect Prediction (VEP) using a fine-tuned PlantCAD2 binary classifier.

Scores each variant by computing:
    LLR = log P(alt | binary classifier) - log P(ref | binary classifier)

where each allele is scored as: log_prob_positive - log_prob_negative
(i.e. the log-odds from the softmax output of the classification head).

Usage:
    python vep.py \
        --vcf variants.vcf.gz \
        --fasta genome.fa \
        --checkpoint /path/to/lora_checkpoint \
        --model_name /path/to/base_model \
        --output output.tsv

    # Load adapter from Hugging Face Hub (base model resolved from adapter config):
    python vep.py \
        --vcf variants.vcf.gz \
        --fasta genome.fa \
        --checkpoint org/model-adapter \
        --output output.tsv

    # Multi-node with srun (one GPU per node, e.g. 4 nodes):
    srun --ntasks=4 --ntasks-per-node=1 --gpus-per-task=1 \
        python vep.py \
            --vcf variants.vcf.gz \
            --fasta genome.fa \
            --checkpoint /path/to/lora_checkpoint \
            --model_name /path/to/base_model \
            --output output.tsv \
            --is_distributed=True
    # Produces output.rank0.tsv ... output.rank3.tsv; merge with:
    #   head -1 output.rank0.tsv > merged.tsv
    #   tail -n +2 -q output.rank*.tsv >> merged.tsv
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import fire
import numpy as np
import torch
import torch.nn.functional as F
from cyvcf2 import VCF
from pyfaidx import Fasta
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)

# Complement table for reverse-complementing sequences
_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

VALID_DNA_BASES = {"A", "C", "G", "T"}


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _extract_window(fasta: Fasta, chrom: str, pos: int, half: int) -> str:
    """Return a (2*half)-bp window centred on *pos* (1-based, inclusive).

    The variant sits at position *half* (0-based) inside the returned string.
    Missing flanking bases at sequence boundaries are padded with 'N'.
    """
    chrom_len = len(fasta[chrom])
    start = pos - half  # 1-based, inclusive
    end = pos + half - 1  # 1-based, inclusive

    # Clamp to chromosome bounds; track how much padding is needed
    left_pad = max(0, 1 - start)
    right_pad = max(0, end - chrom_len)
    fetch_start = max(1, start)
    fetch_end = min(chrom_len, end)

    seq = str(fasta[chrom][fetch_start - 1 : fetch_end]).upper()
    return "N" * left_pad + seq + "N" * right_pad


def _make_allele_seq(window: str, ref: str, alt: str, half: int) -> Optional[str]:
    """Substitute the central base(s) with *alt*, handling indels by recentering.

    Returns None when the variant is too complex to handle (e.g., symbolic ALT).
    Only SNPs and short indels whose total window fits in 2*half are supported.
    """
    center = half  # 0-based index of the first ref base in the window
    ref_len = len(ref)
    alt_len = len(alt)

    if ref_len == alt_len:
        # SNP or MNP: simple substitution
        new_seq = window[:center] + alt + window[center + ref_len :]
        # For SNPs/MNPs the window may shift; trim or pad to keep length constant
        target_len = 2 * half
        if len(new_seq) != target_len:
            # Shouldn't happen for MNPs of same length, but guard anyway
            return None
        return new_seq
    else:
        # Indel: rebuild a window of the correct length around the variant
        left = window[:center]
        right = window[center + ref_len :]
        new_seq = left + alt + right
        target_len = 2 * half
        if len(new_seq) >= target_len:
            # Deletion (new_seq shorter) or trim long seq to target
            # Centre the window on the variant site
            mid = center
            half = target_len // 2
            pad_new = ("N" * max(0, half - mid)) + new_seq
            start = max(0, mid - half)
            result = pad_new[start : start + target_len]
            if len(result) < target_len:
                result = result + "N" * (target_len - len(result))
            return result
        else:
            # Insertion made sequence longer; trim symmetrically
            mid = center
            half = target_len // 2
            start = max(0, mid - half)
            result = new_seq[start : start + target_len]
            if len(result) < target_len:
                result = result + "N" * (target_len - len(result))
            return result


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _check_ranks() -> Tuple[int, int, int]:
    """Detect rank/world_size from srun (SLURM) or torchrun env vars.

    Returns (global_rank, world_size).  local_rank is not returned because
    each node is assumed to have exactly one GPU (always cuda:0).
    """
    # torchrun sets RANK / WORLD_SIZE; srun sets SLURM_PROCID / SLURM_NTASKS
    global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "-1")))
    world_size  = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "-1")))
    if global_rank < 0 or world_size < 0:
        raise ValueError(
            "Could not detect distributed rank. "
            "Make sure RANK+WORLD_SIZE (torchrun) or SLURM_PROCID+SLURM_NTASKS (srun) are set."
        )
    return global_rank, world_size


# ---------------------------------------------------------------------------
# Model loading  (mirrors lora_fine_tune.py patterns)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(
    checkpoint_dir: str,
    model_name: Optional[str],
):
    """Load the LoRA binary classifier and its tokenizer.

    If *checkpoint_dir* is a local path, *model_name* must point to the base
    model.  If it is a HF repo id, the base model is resolved from the
    adapter config (model_name is ignored).

    The base model's forward is patched to accept only input_ids, mirroring
    the training setup in dev.py (Caduceus does not support attention_mask).
    """
    from transformers import AutoModelForSequenceClassification

    checkpoint_path = Path(checkpoint_dir)

    if checkpoint_path.exists():
        if model_name is None:
            raise ValueError(
                "--model_name is required when loading a local checkpoint directory"
            )
        logger.info("Loading base model from %s", model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
        )
        logger.info("Loading local LoRA adapter from %s", checkpoint_dir)
        tokenizer_source = model_name
    else:
        logger.info("Loading LoRA adapter from Hugging Face: %s", checkpoint_dir)
        config = PeftConfig.from_pretrained(checkpoint_dir)
        base_model_name = config.base_model_name_or_path
        logger.info("Resolved base model: %s", base_model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
        )
        tokenizer_source = base_model_name

    # Patch forward to strip unsupported kwargs (e.g. attention_mask),
    # matching the pattern used in dev.py load_base_model().
    original_forward = base_model.forward
    def _forward_compat(*args, **kwargs):
        return original_forward(input_ids=kwargs.get("input_ids", args[0] if args else None))
    base_model.forward = _forward_compat

    model = PeftModel.from_pretrained(base_model, checkpoint_dir, device_map={"": "cpu"})

    logger.info("Loading tokenizer from %s", tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    return model, tokenizer


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_sequences(
    sequences: list[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (log_odds, pos_probs) for each sequence.

    log_odds : log P(pos) - log P(neg)
    pos_probs: P(positive class)
    """
    log_odds = []
    pos_probs = []
    model.eval()

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        enc = tokenizer(
            batch_seqs,
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits  # (B, 2)

        probs = F.softmax(logits.float(), dim=-1)          # (B, 2)
        log_probs = torch.log(probs)
        batch_log_odds = (log_probs[:, 1] - log_probs[:, 0]).cpu().numpy()
        batch_pos_probs = probs[:, 1].cpu().numpy()

        log_odds.append(batch_log_odds)
        pos_probs.append(batch_pos_probs)

    return np.concatenate(log_odds), np.concatenate(pos_probs)


# ---------------------------------------------------------------------------
# Main VEP function
# ---------------------------------------------------------------------------

def vep(
    vcf: str,
    fasta: str,
    checkpoint: str,
    output: str,
    model_name: Optional[str] = None,
    batch_size: int = 32,
    half_window: int = 300,
    bf16: bool = True,
    is_distributed: bool = False,
) -> None:
    """Score variants in *vcf* against a fine-tuned PlantCAD2 binary classifier.

    Parameters
    ----------
    vcf:             Path to input VCF/BCF (plain or bgzipped).
    fasta:           Path to reference genome FASTA (must have .fai index or be indexable).
    checkpoint:      Local directory or HF repo id of the LoRA adapter checkpoint.
    output:          Path for the output TSV (CHROM, POS, REF, ALT, REF_PROB, ALT_PROB, LLR).
    model_name:      Base model path/id; required when checkpoint is a local directory.
    batch_size:      Number of sequences per GPU batch (default 32).
    half_window:     Flanking bases on each side of the variant (default 300 → 600 bp total).
    bf16:            Use bfloat16 on CUDA (default True).
    is_distributed:  Run in distributed mode via srun (default False).
                     In distributed mode each rank writes output.rank{N}.tsv.
                     Merge afterwards with:
                       head -1 output.rank0.tsv > merged.tsv
                       tail -n +2 -q output.rank*.tsv >> merged.tsv
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    # ---- distributed setup --------------------------------------------------
    global_rank = 0
    world_size = 1

    if is_distributed:
        logger.info("Running in distributed mode")
        global_rank, world_size = _check_ranks()
        logger.info("global_rank=%d  world_size=%d", global_rank, world_size)

    # ---- device setup -------------------------------------------------------
    # One GPU per node: always cuda:0 on each node (or cpu as fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- load model ---------------------------------------------------------
    model, tokenizer = _load_model_and_tokenizer(checkpoint, model_name)
    model.eval()
    if bf16 and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device)
    logger.info("Model loaded and moved to %s", device)

    # ---- open genome --------------------------------------------------------
    genome = Fasta(fasta, as_raw=False, sequence_always_upper=True)

    # ---- output path (per-rank when distributed) ----------------------------
    output_path = Path(output)
    if is_distributed:
        output_path = output_path.with_name(
            output_path.stem + f".rank{global_rank}" + output_path.suffix
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- iterate VCF and build batches --------------------------------------
    vcf_reader = VCF(vcf)
    records = []
    ref_seqs = []
    alt_seqs = []

    skipped = 0
    processed = 0
    variant_index = -1  # counts all accepted (chrom, pos, ref, alt) tuples before rank filtering

    with open(output_path, "w") as out_fh:
        out_fh.write("CHROM\tPOS\tREF\tALT\tREF_PROB\tALT_PROB\tLLR\n")

        def _flush_batch():
            nonlocal processed
            if not records:
                return
            ref_log_odds, ref_probs = _score_sequences(ref_seqs, model, tokenizer, device, batch_size, 2 * half_window)
            alt_log_odds, alt_probs = _score_sequences(alt_seqs, model, tokenizer, device, batch_size, 2 * half_window)
            llrs = alt_log_odds - ref_log_odds
            for (chrom, pos, ref, alt), ref_p, alt_p, llr in zip(records, ref_probs, alt_probs, llrs):
                out_fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{ref_p:.6f}\t{alt_p:.6f}\t{llr:.6f}\n")
            processed += len(records)
            records.clear()
            ref_seqs.clear()
            alt_seqs.clear()

        rank_label = f"rank {global_rank}/{world_size}" if is_distributed else "variants"
        pbar = tqdm(desc=rank_label, unit="var")
        for variant in vcf_reader:
            chrom = variant.CHROM
            pos = variant.POS  # 1-based in cyvcf2
            ref = variant.REF

            # cyvcf2 always gives ALT as a list
            alts = variant.ALT
            if alts is None:
                skipped += 1
                continue

            for alt in alts:
                alt = (alt or "").upper()
                # Keep only non-empty A/C/G/T alleles; drop all other ALT forms.
                if not alt or any(base not in VALID_DNA_BASES for base in alt):
                    skipped += 1
                    continue

                try:
                    window = _extract_window(genome, chrom, pos, half=half_window)
                except KeyError:
                    logger.warning("Skipping variant %s:%d — chromosome not found in FASTA", chrom, pos)
                    skipped += 1
                    continue

                alt_window = _make_allele_seq(window, ref, alt, half=half_window)
                if alt_window is None:
                    logger.warning(
                        "Skipping variant %s:%d %s>%s — could not construct alt window",
                        chrom, pos, ref, alt,
                    )
                    skipped += 1
                    continue

                variant_index += 1

                # Distribute work across ranks
                if is_distributed and (variant_index % world_size != global_rank):
                    continue

                records.append((chrom, pos, ref, alt))
                ref_seqs.append(window)
                alt_seqs.append(alt_window)
                pbar.update(1)

                if len(records) >= batch_size * 4:
                    _flush_batch()

        # flush remaining
        _flush_batch()
        pbar.close()

    vcf_reader.close()

    logger.info(
        "[rank %d] Done. Scored %d variant-allele pairs; skipped %d. Output: %s",
        global_rank, processed, skipped, output_path,
    )


if __name__ == "__main__":
    fire.Fire(vep)