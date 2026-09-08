#!/usr/bin/env bash
set -u

PYTHON="${PYTHON:-$HOME/plantcad2/.venv-current/bin/python}"
MODEL="${MODEL:-$HOME/artifacts/marindna-exp472/exp472-plantcad2-angiosperm-lr0p0001-wd0p2-v2/hf/step-75046}"
SAMPLE_DIR="${SAMPLE_DIR:-$HOME/sensitivity/samples-seed0-n5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/sensitivity/batch-sweep}"
RUNNER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_condition.py"

run_one() {
  local name="$1"
  local batch="$2"
  local cache_flag="$3"
  local out="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out}"
  if [[ ! -f "${out}/result.json" ]] && ! "${PYTHON}" "${RUNNER}" \
    --model "${MODEL}" \
    --sample-dir "${SAMPLE_DIR}" \
    --output-dir "${out}" \
    --condition "${name}" \
    --dtype bf16 \
    --attention flash_attention_2 \
    --batch-size "${batch}" \
    --sv-batch-size "${batch}" \
    "${cache_flag}" \
    --max-samples 32 >"${out}/run.log" 2>&1; then
    echo "${name} failed"
    tail -20 "${out}/run.log"
    return 1
  fi
  "${PYTHON}" -c '
import json, sys
r = json.load(open(sys.argv[1]))
peak = max(c["peak_reserved_gib"] for t in r["tasks"].values() for c in t["contexts"].values())
print(r["condition"], r["aggregate_timing"]["sequences_per_second"], peak,
      r["tasks"]["sv_impact"]["contexts"]["left"]["timing"]["sequences_per_second"], sep="\t")
' "${out}/result.json"
}

for batch in 1 4 8 16 32; do
  run_one "cache-b${batch}" "${batch}" --use-cache
done
for batch in 4 8 16 32 64; do
  run_one "no-cache-b${batch}" "${batch}" --no-use-cache
done
