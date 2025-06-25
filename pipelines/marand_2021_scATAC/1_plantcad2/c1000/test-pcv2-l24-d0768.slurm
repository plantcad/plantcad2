#!/bin/bash

#SBATCH -J pcv2-l24-d0768-scATAC
#SBATCH -o logs/%x-%j.log
#SBATCH -e logs/%x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jz963@cornell.edu

module load gcc/13.2.0 cuda/12.4
module load python3/3.11.8

# Environment variables
export TRITON_PTXAS_PATH=/opt/apps/cuda/12.4/bin/ptxas
export DATA_DIR=/work/10373/jzhai/vista/utils/plantcad2/results/PlantCAD2_tasks/maize_cell_type_accessible_c1000
export OUTPUT_DIR="${DATA_DIR}/pcv2-l24-d0768-checkpoints-lr-1e-4"

source $WORK/envs/PlantCAD/bin/activate
source ~/.bashrc

python ../main.py predict \
    --checkpoint_dir=${OUTPUT_DIR}/checkpoint-11807 \
    --num_labels 92 \
    --data_dir=$DATA_DIR/test.parquet \
    --output_file=${OUTPUT_DIR}/test_predictions.tsv \
    --model_name "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000" \
    --batch_size 64