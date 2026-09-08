#!/usr/bin/env python3
"""Convert one local exp472 Levanter checkpoint and optionally upload its HF form."""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.main.export_lm_to_hf import main as export_lm_to_hf
from levanter.trainer import TrainerConfig


def _load_exp472_common(marin_dna: Path):
    common_path = marin_dna / "experiments/exp472_plantcad2_baseline/common.py"
    spec = importlib.util.spec_from_file_location("exp472_common", common_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marin-dna", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--path-in-repo", default="")
    parser.add_argument("--max-shard-size", type=int, default=2_000_000_000)
    args = parser.parse_args()
    if bool(args.repo_id) != bool(args.path_in_repo):
        raise ValueError("repo-id and path-in-repo must be provided together")

    common = _load_exp472_common(args.marin_dna)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = TrainerConfig(per_device_parallelism=1)
    config = ConvertLmConfig(
        trainer=trainer,
        checkpoint_path=str(args.checkpoint),
        output_dir=str(args.output_dir),
        checkpoint_subpath="model",
        max_shard_size=args.max_shard_size,
        model=common.MODEL_CONFIG,
        save_tokenizer=True,
        tokenizer=common.TOKENIZER,
        override_vocab_size=common.VOCAB_SIZE,
    )
    # JAX 0.11 requires jax.set_mesh (via Levanter's compatibility wrapper);
    # entering a Mesh object alone no longer establishes the ambient mesh used by
    # Haliax while the exporter constructs the checkpoint-shaped model.
    with trainer.use_device_mesh():
        export_lm_to_hf(config)

    if args.repo_id:
        token = os.environ.get("HF_TOKEN") or os.environ["HUGGING_FACE_HUB_TOKEN"]
        api = HfApi(token=token)
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=args.output_dir,
            path_in_repo=args.path_in_repo,
            commit_message=f"Export exp472 checkpoint to {args.path_in_repo}",
        )
        print({"commit_url": commit.commit_url, "path_in_repo": args.path_in_repo})


if __name__ == "__main__":
    main()
