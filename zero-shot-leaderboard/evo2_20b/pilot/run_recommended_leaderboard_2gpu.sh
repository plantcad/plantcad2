#!/usr/bin/env bash
# Lambda VM entrypoint; CoreWeave/Iris orchestration is isolated in coreweave/.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:?Set MODEL to a local Hugging Face checkpoint directory}"
MODEL_LABEL="${MODEL_LABEL:-marindna-exp472}"
SAMPLES="${SAMPLES:-10000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SV_BATCH_SIZE="${SV_BATCH_SIZE:-32}"
SAMPLE_DIR="${SAMPLE_DIR:-${SCRIPT_DIR}/samples-seed${SEED}-n${SAMPLES}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results-seed${SEED}-n${SAMPLES}}"

if [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')" -lt 2 ]]; then
  echo "This runner requires at least two visible GPUs." >&2
  exit 1
fi

mkdir -p "${SAMPLE_DIR}" "${OUTPUT_DIR}/workers"
if [[ ! -f "${SAMPLE_DIR}/manifest.json" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/sample_tasks.py" --output-dir "${SAMPLE_DIR}" --samples "${SAMPLES}" --seed "${SEED}" --task-set leaderboard --shuffle-buffer-size 200000 --shared-seed --no-balance-binary
fi

pids=()
for shard in 0 1; do
  worker_dir="${OUTPUT_DIR}/workers/gpu${shard}"
  mkdir -p "${worker_dir}"
  CUDA_VISIBLE_DEVICES="${shard}" "${PYTHON}" "${SCRIPT_DIR}/sensitivity/run_condition.py" --model "${MODEL}" --sample-dir "${SAMPLE_DIR}" --output-dir "${worker_dir}" --condition "${MODEL_LABEL}-recommended-gpu${shard}" --dtype bf16 --attention flash_attention_2 --batch-size "${BATCH_SIZE}" --sv-batch-size "${SV_BATCH_SIZE}" --softmax-dtype fp32 --tf32 --no-use-cache --require-flash --task-set leaderboard --task-shard-count 2 --task-shard-index "${shard}" >"${worker_dir}/run.log" 2>&1 &
  pids+=("$!")
done

for _ in 0 1; do
  if ! wait -n; then
    for pid in "${pids[@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait || true
    echo "An evaluation worker failed; the sibling was stopped. Inspect ${OUTPUT_DIR}/workers/gpu*/run.log" >&2
    exit 1
  fi
done

"${PYTHON}" "${SCRIPT_DIR}/merge_recommended.py" --shard "${OUTPUT_DIR}/workers/gpu0/result.json" --shard "${OUTPUT_DIR}/workers/gpu1/result.json" --sample-manifest "${SAMPLE_DIR}/manifest.json" --output-dir "${OUTPUT_DIR}"
