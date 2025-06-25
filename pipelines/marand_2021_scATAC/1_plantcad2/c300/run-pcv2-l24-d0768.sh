#!/bin/bash

DATA_DIR=../../../results/PlantCAD2_tasks/maize_cell_type_accessible_c300
OUTPUT_DIR=${DATA_DIR}/pcv2-l24-d0768-checkpoints-lr-1e-4

CUDA_VISIBLE_DEVICES=1 python main.py train \
    --data_dir=$DATA_DIR \
    --output_dir=${OUTPUT_DIR} \
    --model_name "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000" \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --max_steps -1 \
    --seed 42 \
    --use_wandb True \
    --wandb_project PlantCAD2 \
    --wandb_run_name "maize_cell_type_accessible_c300_pcv2-l24-d0768" \
    --learning_rate 1e-4 \
    --warmup_steps 50 \
    --lr_scheduler_type "linear" \
    --gradient_accumulation_steps 1 \
    --bf16 True \
    --num_train_epochs 1 \
    --weight_decay 0.01 \
    --eval_strategy "steps" \
    --eval_steps 4000 \
    --save_strategy "steps" \
    --save_steps 4000 \
    --logging_steps 4000 \
    --remove_unused_columns False --num_labels 92

# predict test
CUDA_VISIBLE_DEVICES=1 python main.py predict \
    --checkpoint_dir=${OUTPUT_DIR}/checkpoint-39034 \
    --num_labels 92 \
    --data_dir=$DATA_DIR/test.parquet \
    --output_file=${OUTPUT_DIR}/test_predictions.tsv \
    --model_name "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000" \
    --batch_size 128