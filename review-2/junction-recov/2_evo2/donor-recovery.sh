#!/bin/bash
#SBATCH -J evo2_donor_recovery
#SBATCH -o logs/%x-%A_%a.log
#SBATCH -e logs/%x-%A_%a.log
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
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

# forward: downstream 4095 bp, predict GT (2 tokens)
srun python evo2_generate.py \
    --tsv ../results/${PREFIX}_donor.tsv \
    --length 4095 \
    --output ../results/${PREFIX}_donor_evo2_7b_ntokens_2 \
    --batch-size 2 --n-tokens 2

# reverse complement
srun python evo2_generate.py \
    --tsv ../results/${PREFIX}_donor.tsv \
    --length 4095 \
    --output ../results/${PREFIX}_donor_evo2_7b_rc_ntokens_2 \
    --batch-size 2 --n-tokens 2 --reverse
