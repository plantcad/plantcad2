#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/../zero-shot-eval-causal.py"
SUMMARIZE_SCRIPT="${SCRIPT_DIR}/../summarize.py"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:?Set MODEL to a local Hugging Face checkpoint directory}"
MODEL_LABEL="${MODEL_LABEL:-marindna-exp472}"
SAMPLES="${SAMPLES:-2800}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SV_BATCH_SIZE="${SV_BATCH_SIZE:-2}"
SAMPLE_DIR="${SAMPLE_DIR:-${SCRIPT_DIR}/samples-seed${SEED}-n${SAMPLES}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results-seed${SEED}-n${SAMPLES}}"
CONTEXT_MODES="${CONTEXT_MODES:-left right_reverse_complement}"
EXPECTED_CONTEXT_MODES="${EXPECTED_CONTEXT_MODES:-left,right_reverse_complement}"

mkdir -p "${SAMPLE_DIR}" "${OUTPUT_DIR}"
if [[ ! -f "${SAMPLE_DIR}/manifest.json" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/sample_tasks.py" \
    --output-dir "${SAMPLE_DIR}" \
    --samples "${SAMPLES}" \
    --seed "${SEED}" \
    --task-set leaderboard \
    --shuffle-buffer-size 200000 \
    --shared-seed \
    --no-balance-binary
fi

run_context_pair() {
  local category="$1"
  local task="$2"
  local split="$3"
  local positions="$4"
  local motif_len="$5"
  local input_tsv="${SAMPLE_DIR}/${task}__${split}.tsv"
  local outdir="${OUTPUT_DIR}/${category}/${task}_${split}"
  local manifests=""
  mkdir -p "${outdir}"

  for context_mode in ${CONTEXT_MODES}; do
    local common=(
      --input_tsv "${input_tsv}"
      --task "${task}"
      --split "${split}"
      --model "${MODEL}"
      --device cuda:0
      --batch_size "${BATCH_SIZE}"
      --context_mode "${context_mode}"
      --metrics_json "${outdir}/${context_mode}.metrics.json"
      --manifest_json "${outdir}/${context_mode}.manifest.json"
    )
    case "${category}" in
      evo_cons)
        "${PYTHON}" "${EVAL_SCRIPT}" evo_cons "${common[@]}" --token_idx "${positions}"
        ;;
      motif_acc)
        "${PYTHON}" "${EVAL_SCRIPT}" motif_acc "${common[@]}" \
          --mask_idx="${positions}" --motif_len "${motif_len}" --decode_mode independent
        ;;
      core_noncore)
        "${PYTHON}" "${EVAL_SCRIPT}" core_noncore "${common[@]}" \
          --mask_idx="${positions}" --motif_len "${motif_len}" --label_column label
        ;;
      *)
        echo "Unknown category: ${category}" >&2
        return 1
        ;;
    esac
    manifests="${manifests:+${manifests},}${outdir}/${context_mode}.manifest.json"
  done

  "${PYTHON}" "${EVAL_SCRIPT}" aggregate_context_max \
    --manifest_json "${manifests}" \
    --expected_context_modes "${EXPECTED_CONTEXT_MODES}" \
    --strict_expected_modes True \
    --output_json "${outdir}/aggregate_context_max.json"
}

run_context_pair evo_cons conservation_within_andropogoneae test 4095 1
run_context_pair evo_cons conservation_within_poaceae_non_tis test 4095 1
run_context_pair evo_cons conservation_within_poaceae_tis test 4095 1

for split in test_maize test_tomato; do
  run_context_pair motif_acc tis_recovery "${split}" 4094,4095,4096 3
  run_context_pair motif_acc tts_recovery "${split}" 4094,4095,4096 3
  run_context_pair motif_acc donor_recovery "${split}" 4095,4096 2
  run_context_pair motif_acc acceptor_recovery "${split}" 4095,4096 2
done

for split in test_maize test_tomato; do
  run_context_pair core_noncore tis_core_noncore_classification "${split}" 4094,4095,4096 3
  run_context_pair core_noncore tts_core_noncore_classification "${split}" 4094,4095,4096 3
  run_context_pair core_noncore donor_core_noncore_classification "${split}" 4095,4096 2
  run_context_pair core_noncore acceptor_core_noncore_classification "${split}" 4095,4096 2
done

sv_task="structural_variant_effect_prediction"
sv_split="test"
sv_outdir="${OUTPUT_DIR}/sv_effect/${sv_task}_${sv_split}"
mkdir -p "${sv_outdir}"
"${PYTHON}" "${EVAL_SCRIPT}" sv_effect \
  --input_tsv "${SAMPLE_DIR}/${sv_task}__${sv_split}.tsv" \
  --task "${sv_task}" \
  --split "${sv_split}" \
  --model "${MODEL}" \
  --device cuda:0 \
  --batch_size "${SV_BATCH_SIZE}" \
  --flanking 5 \
  --output "${sv_outdir}/scores.tsv" \
  --manifest_json "${sv_outdir}/left.manifest.json"

for category in evo_cons motif_acc core_noncore sv_effect; do
  summary_args=(
    --results_dir "${OUTPUT_DIR}"
    --subcommand "${category}"
    --model "${MODEL_LABEL}"
    --output "${OUTPUT_DIR}/${category}.md"
  )
  if [[ "${category}" == "sv_effect" ]]; then
    summary_args+=(--modes left)
  fi
  "${PYTHON}" "${SUMMARIZE_SCRIPT}" "${summary_args[@]}"
done
