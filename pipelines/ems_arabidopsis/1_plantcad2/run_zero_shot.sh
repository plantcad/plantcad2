#!/usr/bin/env bash
# Zero-shot masked-LM scoring of the EMS/Kono 2018 SNVs with PlantCAD2 (8,192 bp windows).
# Usage: conda activate transformers && [BATCH_SIZE=16] bash run_zero_shot.sh [model] [gpu]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
MODEL="${1:-kuleshov-group/PlantCAD2-Large-l48-d1536}"
GPU="${2:-0}"
DATA="$REPO_ROOT/results/ems/ems_training_snvs.tsv"
OUT="$REPO_ROOT/results/ems/$(basename "$MODEL")"
mkdir -p "$OUT"

# SNV sits at 1-based index 4096 of the 8,192 bp window -> token_idx 4095.
CUDA_VISIBLE_DEVICES="$GPU" python "$REPO_ROOT/review/8_tiny_model_eval/zero-shot-eval.py" evo_cons \
    --input_tsv "$DATA" \
    --model "$MODEL" \
    --device cuda:0 \
    --token_idx 4095 \
    --batch_size "${BATCH_SIZE:-16}" \
    --save_logits "$OUT/logits.tsv" \
    --metrics_json "$OUT/metrics.json"

python "$(dirname "$0")/summarize.py" --data "$DATA" --logits "$OUT/logits.tsv" --out-dir "$OUT"
