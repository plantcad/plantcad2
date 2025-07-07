DATA_DIR=../../../results/translation_efficiency/training_data/1_absolute
OUT_DIR=${DATA_DIR}/cnn_lstm
CUDA_VISIBLE_DEVICES=0 python main.py train --train ${DATA_DIR}/train.tsv \
    --valid ${DATA_DIR}/valid.tsv \
    --output_dir ${OUT_DIR} \
    --run_id ath_leaf_te_absolute_cnn-lstm \
    --learning_rate 0.001 \
    --wandb_project PlantCAD2 \
    --task_type regression

CUDA_VISIBLE_DEVICES=0 python main.py predict --model_path ${OUT_DIR}/bestmodel.ckpt \
    --test ${DATA_DIR}/test.tsv \
    --output_file ${OUT_DIR}/predictions.tsv \
    --device cuda:0 \
    --batch_size 2048

CUDA_VISIBLE_DEVICES=0 python main.py predict --model_path ${OUT_DIR}/bestmodel.ckpt \
    --test ${DATA_DIR}/valid.tsv \
    --output_file ${OUT_DIR}/valid_predictions.tsv \
    --device cuda:0 \
    --batch_size 2048

DATA_DIR=../../../results/translation_efficiency/training_data/2_bin
OUT_DIR=${DATA_DIR}/cnn_lstm
CUDA_VISIBLE_DEVICES=0 python main.py train --train ${DATA_DIR}/train.tsv \
    --valid ${DATA_DIR}/valid.tsv \
    --output_dir ${OUT_DIR} \
    --run_id ath_leaf_te_bin_cnn-lstm \
    --learning_rate 0.001 \
    --wandb_project PlantCAD2 \
    --task_type classification

CUDA_VISIBLE_DEVICES=0 python main.py predict --model_path ${OUT_DIR}/bestmodel.ckpt \
    --test ${DATA_DIR}/test.tsv \
    --output_file ${OUT_DIR}/predictions.tsv \
    --device cuda:0 \
    --batch_size 2048

CUDA_VISIBLE_DEVICES=0 python main.py predict --model_path ${OUT_DIR}/bestmodel.ckpt \
    --test ${DATA_DIR}/valid.tsv \
    --output_file ${OUT_DIR}/valid_predictions.tsv \
    --device cuda:0 \
    --batch_size 2048