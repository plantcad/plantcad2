#!/bin/bash
#SBATCH -J ntv3_evo_cons
#SBATCH -o logs/%x-%A_%a.log
#SBATCH -e logs/%x-%A_%a.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH -a 0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jz963@cornell.edu

# Submit separately for each model with appropriate --time:
#   sbatch --time=3:00:00 --export=ALL,MODEL=InstaDeepAI/NTv3_650M_pre,BATCH=8,TAG=ntv3_650M  run_evo_cons.sh
#   sbatch --time=1:00:00 --export=ALL,MODEL=InstaDeepAI/NTv3_100M_pre,BATCH=32,TAG=ntv3_100M run_evo_cons.sh

module load gcc/13.2.0 cuda/12.4
module load python3/3.11.8

source $WORK/envs/transformers_440/bin/activate
source ~/.bashrc

export TRITON_PTXAS_PATH=/opt/apps/cuda/12.4/bin/ptxas

: "${MODEL:?MODEL env var not set}"
: "${BATCH:?BATCH env var not set}"
: "${TAG:?TAG env var not set}"

TASKS=(
    "conservation_within_poaceae_tis"
    "conservation_within_andropogoneae"
    "conservation_within_poaceae_non_tis"
)

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

REPO=plantcad/PlantCAD2_zero_shot_tasks
SCRIPT=ntv3-zero-shot-eval.py
OUT=results

echo "Model=$TAG  task=$TASK"

python $SCRIPT evo_cons \
    --repo_id $REPO \
    --task "$TASK" \
    --split test \
    --model "$MODEL" \
    --device cuda:0 \
    --token_idx 4095 \
    --batch_size $BATCH \
    --metrics_json ${OUT}/evo_cons_${TASK}_${TAG}.json \
    2>&1 | tee ${OUT}/evo_cons_${TASK}_${TAG}.log

echo "Done"
