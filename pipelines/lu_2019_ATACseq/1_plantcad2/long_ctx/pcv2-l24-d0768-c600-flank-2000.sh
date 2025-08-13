
export DATA_DIR=${cad2}/results/PlantCAD2_tasks/accessible_angiosperm_c600
export OUTPUT_DIR="${DATA_DIR}/models/flank-2000-pcv2-l48-c1536-checkpoints-lr-1e-4"

# Train model
# CUDA_VISIBLE_DEVICES=1 python ../main.py train \
#     --train_dir=$DATA_DIR/data/train_flank_2000.parquet \
#     --valid_dir=$DATA_DIR/data/valid_flank_2000.parquet \
#     --output_dir=$OUTPUT_DIR \
#     --model_name "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000" \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --max_steps -1 \
#     --seed 42 \
#     --use_wandb True \
#     --wandb_project PlantCAD2 \
#     --wandb_run_name "accessible_angiosperm_c600_flank_2000_pcv2-l48-c1536-lr-1e-4" \
#     --learning_rate 1e-4 \
#     --warmup_steps 50 \
#     --lr_scheduler_type "linear" \
#     --gradient_accumulation_steps 16 \
#     --bf16 True \
#     --num_train_epochs 2 \
#     --weight_decay 0.01 \
#     --eval_strategy "steps" \
#     --eval_steps 150 \
#     --save_strategy "steps" \
#     --save_steps 150 \
#     --logging_steps 150 \
#     --remove_unused_columns False

CUDA_VISIBLE_DEVICES=1 python ../main.py predict \
    --checkpoint_dir ${OUTPUT_DIR}/checkpoint-2310 \
    --data_dir ${DATA_DIR}/data/test_setaria_viridis_flank_2000.parquet \
    --output_file ${OUTPUT_DIR}/test_setaria_viridis_flank_2000_scores.tsv \
    --model_name 'kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000' \
    --batch_size 128