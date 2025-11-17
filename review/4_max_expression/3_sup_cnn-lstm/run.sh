DATA_DIR=../../../results/PlantCAD2_tasks/exp-max
OUT_DIR=${DATA_DIR}/cnn_lstm
CUDA_VISIBLE_DEVICES=1 python main.py train --train ${DATA_DIR}/train.tsv \
    --valid ${DATA_DIR}/valid.tsv \
    --output_dir ${OUT_DIR} \
    --run_id exp-max_cnn-lstm \
    --learning_rate 0.001 \
    --wandb_project PlantCAD2 \
    --task_type regression

python main.py predict --model_path ${OUT_DIR}/bestmodel.ckpt \
    --test ${DATA_DIR}/test.tsv \
    --output_file ${OUT_DIR}/test_predictions.tsv \
    --device cuda:1 \
    --batch_size 2048

python main.py predict --model_path ${OUT_DIR}/bestmodel.ckpt \
    --test ${DATA_DIR}/valid.tsv \
    --output_file ${OUT_DIR}/valid_predictions.tsv \
    --device cuda:1 \
    --batch_size 2048