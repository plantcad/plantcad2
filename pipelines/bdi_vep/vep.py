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
"""

import logging
import sys
from pathlib import Path
from typing import Optional

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

HALF_WINDOW = 300  # flanking bases on each side -> total window = 600 bp
VALID_DNA_BASES = {"A", "C", "G", "T"}


# ---------------------------------------------------------------------------
# Sequence helpers
# ---------------------------------------------------------------------------

def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _extract_window(fasta: Fasta, chrom: str, pos: int, half: int = HALF_WINDOW) -> str:
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


def _make_allele_seq(window: str, ref: str, alt: str) -> Optional[str]:
    """Substitute the central base(s) with *alt*, handling indels by recentering.

    Returns None when the variant is too complex to handle (e.g., symbolic ALT).
    Only SNPs and short indels whose total window fits in 2*HALF_WINDOW are supported.
    """
    center = HALF_WINDOW  # 0-based index of the first ref base in the window
    ref_len = len(ref)
    alt_len = len(alt)

    if ref_len == alt_len:
        # SNP or MNP: simple substitution
        new_seq = window[:center] + alt + window[center + ref_len :]
        # For SNPs/MNPs the window may shift; trim or pad to keep length constant
        target_len = 2 * HALF_WINDOW
        if len(new_seq) != target_len:
            # Shouldn't happen for MNPs of same length, but guard anyway
            return None
        return new_seq
    else:
        # Indel: rebuild a window of the correct length around the variant
        left = window[:center]
        right = window[center + ref_len :]
        new_seq = left + alt + right
        target_len = 2 * HALF_WINDOW
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
    """
    checkpoint_path = Path(checkpoint_dir)

    if checkpoint_path.exists():
        if model_name is None:
            raise ValueError(
                "--model_name is required when loading a local checkpoint directory"
            )
        logger.info("Loading base model from %s", model_name)
        from transformers import AutoModelForSequenceClassification

        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
        )
        logger.info("Loading local LoRA adapter from %s", checkpoint_dir)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        tokenizer_source = model_name
    else:
        logger.info("Loading LoRA adapter from Hugging Face: %s", checkpoint_dir)
        config = PeftConfig.from_pretrained(checkpoint_dir)
        base_model_name = config.base_model_name_or_path
        logger.info("Resolved base model: %s", base_model_name)
        from transformers import AutoModelForSequenceClassification

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1},
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        tokenizer_source = base_model_name

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
) -> np.ndarray:
    """Return log-odds scores (log P_pos - log P_neg) for each sequence."""
    log_odds = []
    model.eval()

    n_batches = (len(sequences) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(sequences), batch_size), total=n_batches, desc="scoring", leave=False):
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
            # Caduceus does not accept attention_mask; pass only input_ids
            logits = model.base_model(input_ids=input_ids).logits  # (B, 2)

        log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, 2)
        # log-odds: log P(pos) - log P(neg)
        batch_log_odds = (log_probs[:, 1] - log_probs[:, 0]).cpu().numpy()
        log_odds.append(batch_log_odds)

    return np.concatenate(log_odds)


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
    sequence_length: int = 600,
    bf16: bool = True,
) -> None:
    """Score variants in *vcf* against a fine-tuned PlantCAD2 binary classifier.

    Parameters
    ----------
    vcf:             Path to input VCF/BCF (plain or bgzipped).
    fasta:           Path to reference genome FASTA (must have .fai index or be indexable).
    checkpoint:      Local directory or HF repo id of the LoRA adapter checkpoint.
    output:          Path for the output TSV (CHROM, POS, REF, ALT, LLR).
    model_name:      Base model path/id; required when checkpoint is a local directory.
    batch_size:      Number of sequences per GPU batch (default 32).
    sequence_length: Length of the tokenised window (default 600).
    bf16:            Use bfloat16 on CUDA (default True).
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    # ---- device setup -------------------------------------------------------
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

    # ---- iterate VCF and build batches --------------------------------------
    vcf_reader = VCF(vcf)
    records = []       # (chrom, pos, ref, alt) tuples for the current batch
    ref_seqs = []
    alt_seqs = []

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    processed = 0

    with open(output_path, "w") as out_fh:
        out_fh.write("CHROM\tPOS\tREF\tALT\tLLR\n")

        def _flush_batch():
            nonlocal processed
            if not records:
                return
            ref_scores = _score_sequences(ref_seqs, model, tokenizer, device, batch_size, sequence_length)
            alt_scores = _score_sequences(alt_seqs, model, tokenizer, device, batch_size, sequence_length)
            llrs = alt_scores - ref_scores
            for (chrom, pos, ref, alt), llr in zip(records, llrs):
                out_fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{llr:.6f}\n")
            processed += len(records)
            records.clear()
            ref_seqs.clear()
            alt_seqs.clear()

        for variant in tqdm(vcf_reader, desc="variants", unit="var"):
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
                    window = _extract_window(genome, chrom, pos, half=HALF_WINDOW)
                except KeyError:
                    logger.warning("Skipping variant %s:%d — chromosome not found in FASTA", chrom, pos)
                    skipped += 1
                    continue

                alt_window = _make_allele_seq(window, ref, alt)
                if alt_window is None:
                    logger.warning(
                        "Skipping variant %s:%d %s>%s — could not construct alt window",
                        chrom, pos, ref, alt,
                    )
                    skipped += 1
                    continue

                records.append((chrom, pos, ref, alt))
                ref_seqs.append(window)
                alt_seqs.append(alt_window)

                if len(records) >= batch_size * 4:
                    _flush_batch()

        # flush remaining
        _flush_batch()

    vcf_reader.close()

    logger.info("Done. Scored %d variant-allele pairs; skipped %d.", processed, skipped)
    logger.info("Output written to %s", output_path)


if __name__ == "__main__":
    fire.Fire(vep)
