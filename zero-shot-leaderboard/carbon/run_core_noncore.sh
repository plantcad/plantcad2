#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" "${SCRIPT_DIR}/eval_motif_tasks.py" \
  --mode core_noncore \
  --tasks tis_core_noncore_classification tts_core_noncore_classification donor_core_noncore_classification acceptor_core_noncore_classification \
  --splits test_maize test_tomato \
  --batch_size 64 \
  --micro_batch_size 128 \
  --progress_interval 600 \
  --output_dir "${SCRIPT_DIR}/results/core_noncore"
