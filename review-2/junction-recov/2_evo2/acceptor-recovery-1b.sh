#!/bin/bash
#SBATCH -J evo2_acceptor_recovery_1b
#SBATCH -o logs/%x-%A_%a.log
#SBATCH -e logs/%x-%A_%a.log
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 6:00:00
#SBATCH -a 0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jz963@cornell.edu

module load gcc/13.2.0 cuda/12.4
module load python3/3.11.8
module load nvidia_math/12.4
export CUDA_HOME=/opt/apps/cuda/12.4
export CUDA_PATH=$CUDA_HOME
export GCC_HOME=/opt/apps/gcc/13.2.0
export TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas
export TORCH_CUDA_ARCH_LIST="9.0"

source $WORK/envs/evo2/bin/activate

PREFIXES=(GCF_000001405.26_GRCh38 GCF_000001635.27_GRCm39)
PREFIX=${PREFIXES[$SLURM_ARRAY_TASK_ID]}

# forward: upstream 4095 bp, predict AG (2 tokens)
srun python evo2_generate.py \
    --tsv ../results/${PREFIX}_acceptor.tsv \
    --length 4095 \
    --output ../results/${PREFIX}_acceptor_evo2_1b_base_ntokens_2 \
    --batch-size 2 --n-tokens 2 --model evo2_1b_base

# reverse complement
srun python evo2_generate.py \
    --tsv ../results/${PREFIX}_acceptor.tsv \
    --length 4095 \
    --output ../results/${PREFIX}_acceptor_evo2_1b_base_rc_ntokens_2 \
    --batch-size 2 --n-tokens 2 --reverse --model evo2_1b_base
