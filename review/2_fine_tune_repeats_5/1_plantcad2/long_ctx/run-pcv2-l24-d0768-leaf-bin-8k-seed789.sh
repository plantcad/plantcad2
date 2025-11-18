#!/bin/bash

#SBATCH -J pcv2-l24-d0768-8k-seed789
#SBATCH -o logs/%x-%j.log
#SBATCH -e logs/%x-%j.log
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jz963@cornell.edu

# Function to display usage
function usage() {
    echo "Usage: $0 <train|evaluate>"
    exit 1
}

# Argument parsing
if [ $# -lt 1 ]; then
    usage
fi
MODE=$1
if [[ "$MODE" != "train" && "$MODE" != "evaluate" ]]; then
    usage
fi

# Load required modules
module load gcc/13.2.0 cuda/12.4
module load python3/3.11.8

export JOB_ID=${SLURM_JOB_ID:-manual}
export TRITON_CACHE_DIR="/tmp/$USER/triton-$JOB_ID"
export XDG_CACHE_HOME="$TRITON_CACHE_DIR"
export TRITON_AUTOTUNE_CACHE_DIR="$TRITON_CACHE_DIR"
export TRITON_CACHE_PATHS="$TRITON_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR"


# Environment variables
export TRITON_PTXAS_PATH=/opt/apps/cuda/12.4/bin/ptxas

export DATA_DIR=${cad2}/results/PlantCAD2_tasks/exp-leaf-bin
export OUTPUT_DIR="${DATA_DIR}/pcv2-l24-d0768-8k-checkpoints-lr-1e-4-seed789"


source $WORK/envs/PlantCAD/bin/activate
source ~/.bashrc

# Distributed training variables
if [ -z "$PMIX_RANK" ] || [ -z "$SLURM_NNODES" ] || [ -z "$SLURM_JOB_NODELIST" ] || [ -z "$SLURM_JOB_ID" ]; then
    echo "Required SLURM environment variables are not set"
    exit 1
fi

export RANK=${PMIX_RANK}
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NNODES
export LOCAL_WORLD_SIZE=1

echo "[INFO] RANK=$RANK WORLD_SIZE=$WORLD_SIZE NODE ID=$SLURM_NODEID"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"


if [ "$MODE" == "train" ]; then
    NUM_EPOCHS=${NUM_EPOCHS:-3}
    CHECKPOINT_DIR=${CHECKPOINT_DIR:-None}
    echo "[INFO] Starting training on $(hostname) with RANK=$RANK NODE_RANK=$SLURM_NODEID"
    echo "Using TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

    torchrun \
        --node_rank=$RANK \
        --nproc_per_node=1 \
        --nnodes=$SLURM_NNODES \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        ../dev.py train \
        --train_dir=$DATA_DIR/train_8k.parquet \
        --valid_dir=$DATA_DIR/valid_8k.parquet \
        --output_dir=$OUTPUT_DIR \
        --model_name "kuleshov-group/PlantCAD2-Small-l24-d0768" \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --max_steps -1 \
        --seed 789 \
        --use_wandb True \
        --wandb_project PlantCAD2 \
        --wandb_run_name "exp-leaf-bin_pcv2-l24-d0768-8k-seed789" \
        --learning_rate 1e-4 \
        --warmup_steps 50 \
        --lr_scheduler_type "linear" \
        --gradient_accumulation_steps 8 \
        --bf16 True \
        --num_train_epochs 1 \
        --weight_decay 0.01 \
        --eval_strategy "steps" \
        --eval_steps 700 \
        --save_strategy "steps" \
        --save_steps 700 \
        --logging_steps 700 \
        --remove_unused_columns False \
        --is_distributed True
elif [ "$MODE" == "evaluate" ]; then
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "CHECKPOINT_DIR is not set"
        exit 1
    fi

    torchrun \
        --nproc_per_node=1 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        dev.py evaluate \
        --checkpoint_dir=$CHECKPOINT_DIR \
        --data_dir=$DATA_DIR \
        --model_name=$MODEL_NAME \
        --use_wandb=True \
        --is_distributed=True
fi
