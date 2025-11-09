DATA_DIR=../../../results/PlantCAD2_tasks/exp-leaf-abs
OUT_DIR=${DATA_DIR}/cnn_lstm_seed789
CUDA_VISIBLE_DEVICES=1 python main.py train --train ${DATA_DIR}/train.tsv \
    --valid ${DATA_DIR}/valid.tsv \
    --output_dir ${OUT_DIR} \
    --run_id exp-leaf-abs_cnn-lstm-seed789 \
    --learning_rate 0.001 \
    --wandb_project PlantCAD2 \
    --task_type regression \
    --seed 789

python main.py predict --model_path ${OUT_DIR}/bestmodel.ckpt \
    --test ${DATA_DIR}/test.tsv \
    --output_file ${OUT_DIR}/predictions.csv \
    --device cuda:1 \
    --batch_size 2048

python main.py predict --model_path ${OUT_DIR}/bestmodel.ckpt \
    --test ${DATA_DIR}/valid.tsv \
    --output_file ${OUT_DIR}/valid_predictions.csv \
    --device cuda:1 \
    --batch_size 2048
