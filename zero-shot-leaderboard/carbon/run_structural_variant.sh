#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${SCRIPT_DIR}/eval_structural_variant.py" \
  --split test \
  --batch_size 2 \
  --logit_chunk_size 64 \
  --progress_interval 600 \
  --output_dir "${SCRIPT_DIR}/results/structural_variant"
