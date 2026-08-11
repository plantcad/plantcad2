#!/bin/bash
# Shared environment + configuration for PlantCAD2 zero-shot *causal* evals on TACC.
# Sourced by each *.sub AFTER its #SBATCH directives.
#
# The eval script (zero-shot-eval-causal.py) loads the model via HuggingFace
# AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True), so MODEL
# must be an HF repo id or a local path in HF format.

# ---- TACC modules / toolchain (mirrors review/1_benchmark_evo2/evo2/*.sub) ----
module load gcc/13.2.0 cuda/12.4
module load python3/3.11.8
module load nvidia_math/12.4
export CUDA_HOME=/opt/apps/cuda/12.4
export CUDA_PATH=$CUDA_HOME
export GCC_HOME=/opt/apps/gcc/13.2.0
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas
export TORCH_CUDA_ARCH_LIST="9.0"

# Python env with: torch, transformers, datasets, fire, scikit-learn, pandas, tqdm.
# Reuse the evo2 env or point at a dedicated one.
source $WORK/envs/evo2/bin/activate

# ---------------------------- EDIT THESE ----------------------------
# Evo2 checkpoint name (loaded via the `evo2` package): evo2_20b / evo2_7b / evo2_40b ...
# Also accepts a plain HF causal repo id / local path (loaded via AutoModelForCausalLM).
export MODEL="${MODEL:-evo2_20b}"
export REVISION="${REVISION:-main}"  # ignored for Evo2

export REPO_ID="${REPO_ID:-plantcad/PlantCAD2_zero_shot_tasks}"
export EVAL_SCRIPT="${EVAL_SCRIPT:-$SCRATCH/utils/plantcad2/zero-shot-leaderboard/evo2_20b/zero-shot-eval-causal.py}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/utils/plantcad2/results/zero-shot-leaderboard/evo2_20b}"
# HF_HOME is taken from your account-level setting (e.g. /scratch/10373/jzhai/hugging_face); not overridden here.
# --------------------------------------------------------------------

export DEVICE="${DEVICE:-cuda:0}"
# Measured on one GH200 (evo2_20b, L=8192): 1.49 seq/s at B=1 vs 1.59 at B=8 -- the
# model is already compute-bound at batch 1, so larger batches buy ~nothing and only
# cost memory (43.4 GiB weights + ~4.7 GiB/seq; B=8 peaked at 91/95 GiB, B=16 fails).
export BATCH_SIZE="${BATCH_SIZE:-4}"        # conservation / motif / core-noncore
export SV_BATCH_SIZE="${SV_BATCH_SIZE:-4}"  # sv_effect (two full-length passes/example)

# Nodes per array element for the sharded evals (1 GPU/node, 1 rank/node).
# At 1.59 seq/s the worst array element is ~128 GPU-h, so N=8 -> ~16 h wall.
export NODES="${NODES:-8}"

# Context modes to sweep; aggregate_context_max picks the best per (task,split).
# `left` and `right_reverse_complement` are the two real strands (forward and reverse,
# both 5'->3'); `left_complement` and `right_reverse` are controls, so including them
# doubles the cost and lets a control win the max. Override from the submitting shell
# to cut to the real strands -- both must change together or --strict_expected_modes
# aborts the aggregation step:
#   export CONTEXT_MODES="left right_reverse_complement"
#   export EXPECTED_CONTEXT_MODES="left,right_reverse_complement"
export CONTEXT_MODES="${CONTEXT_MODES:-left left_complement right_reverse right_reverse_complement}"
export EXPECTED_CONTEXT_MODES="${EXPECTED_CONTEXT_MODES:-left,left_complement,right_reverse,right_reverse_complement}"

# Motif positions are shared by *_recovery and *_core_noncore_classification tasks
# (single-nucleotide tokens, 8192 bp sequences, 0-based center = 4095/4096).
motif_spec() {  # sets POS and MLEN from a task name prefix
  case "$1" in
    tis_*|tts_*)        POS="4094,4095,4096"; MLEN=3 ;;
    donor_*|acceptor_*) POS="4095,4096";      MLEN=2 ;;
    *) echo "Unknown motif task: $1" >&2; return 1 ;;
  esac
}
