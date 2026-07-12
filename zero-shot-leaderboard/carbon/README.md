# Carbon zero-shot evaluation for PlantCAD2

This folder contains Carbon-native evaluators for the PlantCAD2 zero-shot
leaderboard tasks in `plantcad/PlantCAD2_zero_shot_tasks`.

## Files

- `eval_conservation.py`: conservation AUROC/AUPRC scoring with center 6-mer
  and full mean log-likelihood modes.
- `eval_motif_tasks.py`: motif recovery accuracy and core/noncore AUROC/AUPRC
  scoring for TIS, TTS, donor, and acceptor tasks.
- `eval_structural_variant.py`: structural variant effect prediction using
  reference-vs-mutant full mean log-likelihood delta.
- `run_*.sh`: default launchers for each task family.

## Environment

Use a Python environment with the packages in `requirements.txt`, plus a CUDA
build of PyTorch suitable for the target machine. The scripts load Carbon from
Hugging Face with `trust_remote_code=True` and expect exactly one visible CUDA
device.

The launchers respect these environment variables:

- `CUDA_VISIBLE_DEVICES`: defaults to `0`.
- `HF_HOME`: defaults to `<plantcad2>/.cache/huggingface`.
- `PYTHON`: defaults to `python`.

## Examples

Run from anywhere:

```bash
plantcad2/zero-shot-leaderboard/carbon/run_recovery.sh
plantcad2/zero-shot-leaderboard/carbon/run_core_noncore.sh
plantcad2/zero-shot-leaderboard/carbon/run_conservation.sh
plantcad2/zero-shot-leaderboard/carbon/run_structural_variant.sh
```

Override model or batching by calling the Python scripts directly:

```bash
python plantcad2/zero-shot-leaderboard/carbon/eval_motif_tasks.py \
  --mode recovery \
  --tasks tis_recovery tts_recovery donor_recovery acceptor_recovery \
  --splits test_maize test_tomato \
  --model HuggingFaceBio/Carbon-3B \
  --output_dir plantcad2/zero-shot-leaderboard/carbon/results/recovery
```

Outputs are written under `zero-shot-leaderboard/carbon/results/` by the
launchers. That directory is ignored by the repository.
