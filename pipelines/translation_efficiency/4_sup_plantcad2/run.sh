DATA_DIR=../../../results/translation_efficiency/training_data/1_absolute
OUTPUT_DIR=${DATA_DIR}/models/sup_pcv2-l24-d0768

CUDA_VISIBLE_DEVICES=0 python main.py train \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_name "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000" \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --max_steps -1 \
    --seed 42 \
    --use_wandb True \
    --wandb_project PlantCAD2 \
    --wandb_run_name "ath_leaf_te_absolute_sup_pcv2-l24-d0768" \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --lr_scheduler_type "linear" \
    --gradient_accumulation_steps 2 \
    --bf16 True \
    --num_train_epochs 1 \
    --weight_decay 0.01 \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 20 \
    --logging_steps 20 \
    --remove_unused_columns False \
    --task_type "regression"

CUDA_VISIBLE_DEVICES=0 python main.py predict \
    --checkpoint_dir ${OUTPUT_DIR}/checkpoint-171 \
    --data_dir $DATA_DIR/test.parquet \
    --output_file ${OUTPUT_DIR}/predictions.tsv \
    --task_type "regression"

CUDA_VISIBLE_DEVICES=0 python main.py predict \
    --checkpoint_dir ${OUTPUT_DIR}/checkpoint-171 \
    --data_dir $DATA_DIR/valid.parquet \
    --output_file ${OUTPUT_DIR}/valid_predictions.tsv \
    --task_type "regression"

DATA_DIR=../../../results/translation_efficiency/training_data/2_bin
OUTPUT_DIR=${DATA_DIR}/models/sup_pcv2-l24-d0768
CUDA_VISIBLE_DEVICES=1 python main.py train \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_name "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000" \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --max_steps -1 \
    --seed 42 \
    --use_wandb True \
    --wandb_project PlantCAD2 \
    --wandb_run_name "ath_leaf_te_bin_sup_pcv2-l24-d0768" \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --lr_scheduler_type "linear" \
    --gradient_accumulation_steps 2 \
    --bf16 True \
    --num_train_epochs 1 \
    --weight_decay 0.01 \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 20 \
    --logging_steps 20 \
    --remove_unused_columns False 

CUDA_VISIBLE_DEVICES=1 python main.py predict \
    --checkpoint_dir ${OUTPUT_DIR}/checkpoint-171 \
    --data_dir $DATA_DIR/test.parquet \
    --output_file ${OUTPUT_DIR}/predictions.tsv 

CUDA_VISIBLE_DEVICES=1 python main.py predict \
    --checkpoint_dir ${OUTPUT_DIR}/checkpoint-171 \
    --data_dir $DATA_DIR/valid.parquet \
    --output_file ${OUTPUT_DIR}/valid_predictions.tsv 