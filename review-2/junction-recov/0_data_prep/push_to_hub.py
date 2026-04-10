#!/usr/bin/env python3
"""
Push junction-recovery datasets to HuggingFace Hub.

Dataset structure:
  config_name  split   file
  -----------  ------  ----
  tis          human   GCF_000001405.26_GRCh38_start_sites.tsv
  tis          mouse   GCF_000001635.27_GRCm39_start_sites.tsv
  tts          human   GCF_000001405.26_GRCh38_stop_sites.tsv
  tts          mouse   GCF_000001635.27_GRCm39_stop_sites.tsv
  donor        human   GCF_000001405.26_GRCh38_donor.tsv
  donor        mouse   GCF_000001635.27_GRCm39_donor.tsv
  acceptor     human   GCF_000001405.26_GRCh38_acceptor.tsv
  acceptor     mouse   GCF_000001635.27_GRCm39_acceptor.tsv

Usage:
  python push_to_hub.py --repo_id kuleshov-group/<dataset-name>

Loading after upload:
  from datasets import load_dataset
  ds = load_dataset("kuleshov-group/...", "tis")
  df_human = ds["human"].to_pandas()
  df_mouse = ds["mouse"].to_pandas()
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset

RESULTS_DIR = Path(__file__).parent.parent / "results"

# (config_name, split) -> tsv path
CONFIGS = {
    ("tis",      "human"): RESULTS_DIR / "GCF_000001405.26_GRCh38_start_sites.tsv",
    ("tis",      "mouse"): RESULTS_DIR / "GCF_000001635.27_GRCm39_start_sites.tsv",
    ("tts",      "human"): RESULTS_DIR / "GCF_000001405.26_GRCh38_stop_sites.tsv",
    ("tts",      "mouse"): RESULTS_DIR / "GCF_000001635.27_GRCm39_stop_sites.tsv",
    ("donor",    "human"): RESULTS_DIR / "GCF_000001405.26_GRCh38_donor.tsv",
    ("donor",    "mouse"): RESULTS_DIR / "GCF_000001635.27_GRCm39_donor.tsv",
    ("acceptor", "human"): RESULTS_DIR / "GCF_000001405.26_GRCh38_acceptor.tsv",
    ("acceptor", "mouse"): RESULTS_DIR / "GCF_000001635.27_GRCm39_acceptor.tsv",
}


def push_config(repo_id: str, config_name: str, split: str, tsv_path: Path) -> None:
    print(f"Loading {config_name}/{split} from {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t")
    ds = Dataset.from_pandas(df, preserve_index=False)
    print(f"  {len(ds)} rows, columns: {ds.column_names}")
    ds.push_to_hub(repo_id, config_name=config_name, split=split)
    print(f"  Pushed {config_name}/{split} to {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push junction datasets to HuggingFace Hub")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo, e.g. kuleshov-group/vert-junction-recovery")
    parser.add_argument("--configs", nargs="+", default=["tis", "tts", "donor", "acceptor"],
                        choices=["tis", "tts", "donor", "acceptor"],
                        help="Which task configs to upload (default: all)")
    parser.add_argument("--species", nargs="+", default=["human", "mouse"], choices=["human", "mouse"],
                        help="Which species splits to upload (default: all)")
    args = parser.parse_args()

    for config_name in args.configs:
        for split in args.species:
            tsv_path = CONFIGS[(config_name, split)]
            if not tsv_path.exists():
                print(f"WARNING: {tsv_path} not found, skipping {config_name}/{split}")
                continue
            push_config(args.repo_id, config_name, split, tsv_path)

    print("Done.")


if __name__ == "__main__":
    main()