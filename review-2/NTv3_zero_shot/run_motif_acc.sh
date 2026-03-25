#!/bin/bash
#SBATCH -J ntv3_motif_acc
#SBATCH -o logs/%x-%A_%a.log
#SBATCH -e logs/%x-%A_%a.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH -a 0-7
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jz963@cornell.edu

# Submit separately for each model with appropriate --time:
#   sbatch --time=6:00:00 --export=ALL,MODEL=InstaDeepAI/NTv3_650M_pre,BATCH=8,TAG=ntv3_650M  run_motif_acc.sh
#   sbatch --time=2:00:00 --export=ALL,MODEL=InstaDeepAI/NTv3_100M_pre,BATCH=32,TAG=ntv3_100M run_motif_acc.sh

module load gcc/13.2.0 cuda/12.4
module load python3/3.11.8

source $WORK/envs/transformers_440/bin/activate
source ~/.bashrc

export TRITON_PTXAS_PATH=/opt/apps/cuda/12.4/bin/ptxas

# Require MODEL, BATCH, TAG to be set via --export
: "${MODEL:?MODEL env var not set}"
: "${BATCH:?BATCH env var not set}"
: "${TAG:?TAG env var not set}"

# Array index selects task × split combination (uniform runtime)
TASKS=(
    "tis_recovery      4094,4095,4096  3  test_maize"
    "tis_recovery      4094,4095,4096  3  test_tomato"
    "acceptor_recovery 4095,4096       2  test_maize"
    "acceptor_recovery 4095,4096       2  test_tomato"
    "donor_recovery    4095,4096       2  test_maize"
    "donor_recovery    4095,4096       2  test_tomato"
    "tts_recovery      4094,4095,4096  3  test_maize"
    "tts_recovery      4094,4095,4096  3  test_tomato"
)

read -r TASK MASK_IDX MOTIF_LEN SPLIT <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

REPO=plantcad/PlantCAD2_zero_shot_tasks
SCRIPT=ntv3-zero-shot-eval.py
OUT=results

echo "Model=$TAG  task=$TASK  split=$SPLIT  mask_idx=$MASK_IDX  motif_len=$MOTIF_LEN"

python $SCRIPT motif_acc \
    --repo_id $REPO \
    --task "$TASK" \
    --split "$SPLIT" \
    --model "$MODEL" \
    --device cuda:0 \
    --mask_idx "$MASK_IDX" \
    --motif_len $MOTIF_LEN \
    --batch_size $BATCH \
    --metrics_json ${OUT}/motif_acc_${TASK}_${SPLIT}_${TAG}.json \
    2>&1 | tee ${OUT}/motif_acc_${TASK}_${SPLIT}_${TAG}.log

echo "Done"
