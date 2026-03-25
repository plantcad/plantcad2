#!/bin/bash
#SBATCH -J ntv3_sv_effect
#SBATCH -o logs/%x-%A_%a.log
#SBATCH -e logs/%x-%A_%a.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jz963@cornell.edu

# No array needed (single task). Submit separately per model:
#   sbatch --time=4:00:00 --export=ALL,MODEL=InstaDeepAI/NTv3_650M_pre,BATCH=8,TAG=ntv3_650M  run_sv_effect.sh
#   sbatch --time=1:00:00 --export=ALL,MODEL=InstaDeepAI/NTv3_100M_pre,BATCH=32,TAG=ntv3_100M run_sv_effect.sh

module load gcc/13.2.0 cuda/12.4
module load python3/3.11.8

source $WORK/envs/transformers_440/bin/activate
source ~/.bashrc

export TRITON_PTXAS_PATH=/opt/apps/cuda/12.4/bin/ptxas

: "${MODEL:?MODEL env var not set}"
: "${BATCH:?BATCH env var not set}"
: "${TAG:?TAG env var not set}"

REPO=plantcad/PlantCAD2_zero_shot_tasks
SCRIPT=ntv3-zero-shot-eval.py
OUT=results

echo "Model=$TAG"

python $SCRIPT sv_effect \
    --repo_id $REPO \
    --task structural_variant_effect_prediction \
    --split test \
    --model "$MODEL" \
    --device cuda:0 \
    --batch_size $BATCH \
    --flanking 5 \
    --output ${OUT}/sv_effect_${TAG}.tsv \
    2>&1 | tee ${OUT}/sv_effect_${TAG}.log

echo "Done"
