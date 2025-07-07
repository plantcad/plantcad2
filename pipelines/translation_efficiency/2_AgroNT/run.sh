DATA_DIR=../../../results/translation_efficiency/training_data/1_absolute
OUTPUT_DIR=${DATA_DIR}/models/agront

# Train model
CUDA_VISIBLE_DEVICES=0 python main.py train \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --max_steps -1 \
    --seed 42 \
    --use_wandb True \
    --wandb_project PlantCAD2 \
    --wandb_run_name "ath_leaf_te_absolute_agront" \
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
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --data_dir $DATA_DIR/test.agront.parquet \
    --output_file ${OUTPUT_DIR}/predictions.tsv \
    --task_type "regression"

DATA_DIR=../../../results/translation_efficiency/training_data/2_bin
OUTPUT_DIR=${DATA_DIR}/models/agront

# Train model
CUDA_VISIBLE_DEVICES=0 python main.py train \
    --data_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --max_steps -1 \
    --seed 42 \
    --use_wandb True \
    --wandb_project PlantCAD2 \
    --wandb_run_name "ath_leaf_te_bin_agront" \
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

CUDA_VISIBLE_DEVICES=0 python main.py predict \
   --checkpoint_dir ${OUTPUT_DIR}/checkpoint-171 \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --data_dir $DATA_DIR/test.agront.parquet \
    --output_file ${OUTPUT_DIR}/predictions.tsv 

CUDA_VISIBLE_DEVICES=0 python main.py predict \
   --checkpoint_dir ${OUTPUT_DIR}/checkpoint-171 \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --data_dir $DATA_DIR/valid.agront.parquet \
    --output_file ${OUTPUT_DIR}/valid_predictions.tsv 