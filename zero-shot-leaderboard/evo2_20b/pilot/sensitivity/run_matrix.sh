#!/usr/bin/env bash
set -u

CURRENT_PYTHON="${CURRENT_PYTHON:-$HOME/plantcad2/.venv-current/bin/python}"
TRANSFORMERS4_PYTHON="${TRANSFORMERS4_PYTHON:-$HOME/plantcad2/.venv-transformers4/bin/python}"
MODEL="${MODEL:-$HOME/artifacts/marindna-exp472/exp472-plantcad2-angiosperm-lr0p0001-wd0p2-v2/hf/step-75046}"
SAMPLE_DIR="${SAMPLE_DIR:-$HOME/sensitivity/samples-seed0-n5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/sensitivity/matrix}"
RUNNER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_condition.py"
STATUS="${OUTPUT_ROOT}/status.tsv"

mkdir -p "${OUTPUT_ROOT}"
touch "${STATUS}"

run_one() {
  local name="$1"
  local python="$2"
  shift 2
  local out="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out}"
  if [[ -f "${out}/result.json" ]]; then
    printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${name}" "already-complete" | tee -a "${STATUS}"
    return 0
  fi
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "${name}" "started" | tee -a "${STATUS}"
  local started=$SECONDS
  if timeout 7200 "${python}" "${RUNNER}" \
    --model "${MODEL}" \
    --sample-dir "${SAMPLE_DIR}" \
    --output-dir "${out}" \
    --condition "${name}" "$@" >"${out}/run.log" 2>&1; then
    printf '%s\t%s\tcomplete\t%d\n' "$(date -u +%FT%TZ)" "${name}" "$((SECONDS - started))" | tee -a "${STATUS}"
  else
    local code=$?
    printf '%s\t%s\tfailed-%d\t%d\n' "$(date -u +%FT%TZ)" "${name}" "${code}" "$((SECONDS - started))" | tee -a "${STATUS}"
    tail -30 "${out}/run.log"
  fi
}

# Production-relevant conditions: all require profiler-confirmed flash kernels.
run_one bf16-fa2-b4 "${CURRENT_PYTHON}" \
  --dtype bf16 --attention flash_attention_2 --batch-size 4 --sv-batch-size 2
run_one bf16-fa2-b4-repeat "${CURRENT_PYTHON}" \
  --dtype bf16 --attention flash_attention_2 --batch-size 4 --sv-batch-size 2 --max-samples 1000
run_one fp16-fa2-b4 "${CURRENT_PYTHON}" \
  --dtype fp16 --attention flash_attention_2 --batch-size 4 --sv-batch-size 2
run_one transformers4-bf16-fa2-b4 "${TRANSFORMERS4_PYTHON}" \
  --dtype bf16 --attention flash_attention_2 --batch-size 4 --sv-batch-size 2
run_one bf16-sdpa-flash-b4 "${CURRENT_PYTHON}" \
  --dtype bf16 --attention sdpa --batch-size 4 --sv-batch-size 2
run_one bf16-fa2-b1 "${CURRENT_PYTHON}" \
  --dtype bf16 --attention flash_attention_2 --batch-size 1 --sv-batch-size 1
run_one bf16-fa2-native-softmax-b4 "${CURRENT_PYTHON}" \
  --dtype bf16 --attention flash_attention_2 --batch-size 4 --sv-batch-size 2 --softmax-dtype native
run_one bf16-fa2-no-tf32-b4 "${CURRENT_PYTHON}" \
  --dtype bf16 --attention flash_attention_2 --batch-size 4 --sv-batch-size 2 --no-tf32 --max-samples 1000
run_one bf16-fa2-no-cache-b32 "${CURRENT_PYTHON}" \
  --dtype bf16 --attention flash_attention_2 --batch-size 32 --sv-batch-size 32 --no-use-cache

# Correctness diagnostics. FlashAttention does not support true FP32 inputs, so these
# intentionally use eager attention and explicitly waive the flash-kernel assertion.
run_one bf16-eager-b1 "${CURRENT_PYTHON}" \
  --dtype bf16 --attention eager --batch-size 1 --sv-batch-size 1 --max-samples 256 --no-require-flash
run_one fp32-eager-tf32-b1 "${CURRENT_PYTHON}" \
  --dtype fp32 --attention eager --batch-size 1 --sv-batch-size 1 --max-samples 256 --no-require-flash
run_one fp32-eager-no-tf32-b1 "${CURRENT_PYTHON}" \
  --dtype fp32 --attention eager --batch-size 1 --sv-batch-size 1 --max-samples 256 --no-require-flash --no-tf32

printf '%s\t%s\n' "$(date -u +%FT%TZ)" "matrix-finished" | tee -a "${STATUS}"
