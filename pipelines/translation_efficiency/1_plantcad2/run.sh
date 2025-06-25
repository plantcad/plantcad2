DATA_DIR=../../../results/translation_efficiency/arabidopsis/
output_dir=${DATA_DIR}/models/pcv2-l24-d0768
CUDA_VISIBLE_DEVICES=0 python main.py train \
    --data_dir=$DATA_DIR \
    --output_dir=${output_dir} \
    --model_name "kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000" \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --max_steps -1 \
    --seed 42 \
    --use_wandb True \
    --wandb_project PlantCAD2 \
    --wandb_run_name "ath_translation_efficiency_pcv2-l24-d0768" \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --lr_scheduler_type "linear" \
    --gradient_accumulation_steps 2 \
    --bf16 True \
    --num_train_epochs 5 \
    --weight_decay 0.01 \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 20 \
    --logging_steps 20 \
    --remove_unused_columns False



DATA_DIR=../../../results/translation_efficiency/arabidopsis/
output_dir=${DATA_DIR}/models/pcv2-l48-d1024
CUDA_VISIBLE_DEVICES=0 python main.py train \
    --data_dir=$DATA_DIR \
    --output_dir=${output_dir} \
    --model_name "maize-genetics/pcv2-l48-d1024" \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --max_steps -1 \
    --seed 42 \
    --use_wandb True \
    --wandb_project PlantCAD2 \
    --wandb_run_name "ath_translation_efficiency_pcv2-l48-d1024" \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --lr_scheduler_type "linear" \
    --gradient_accumulation_steps 4 \
    --bf16 True \
    --num_train_epochs 5 \
    --weight_decay 0.01 \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 20 \
    --logging_steps 20 \
    --remove_unused_columns False


DATA_DIR=../../../results/translation_efficiency/arabidopsis/
output_dir=${DATA_DIR}/models/pcv2-l48-d1536
CUDA_VISIBLE_DEVICES=0 python main.py train \
    --data_dir=$DATA_DIR \
    --output_dir=${output_dir} \
    --model_name "maize-genetics/pcv2-l48-d1536" \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --max_steps -1 \
    --seed 42 \
    --use_wandb True \
    --wandb_project PlantCAD2 \
    --wandb_run_name "ath_translation_efficiency_pcv2-l48-d1536" \
    --learning_rate 1e-4 \
    --warmup_steps 10 \
    --lr_scheduler_type "linear" \
    --gradient_accumulation_steps 8 \
    --bf16 True \
    --num_train_epochs 5 \
    --weight_decay 0.01 \
    --eval_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 20 \
    --logging_steps 20 \
    --remove_unused_columns False