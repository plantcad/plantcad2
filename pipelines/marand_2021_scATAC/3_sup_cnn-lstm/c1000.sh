DATA_DIR=/local/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/maize_cell_type_accessible_c1000
CKPT_DIR=${DATA_DIR}/cnn-lstm-imbalance-lr-1e-3
CUDA_VISIBLE_DEVICES=1 python main.py train  \
    --train $DATA_DIR/train.tsv \
    --valid $DATA_DIR/valid.tsv \
    --output_dir ${CKPT_DIR} \
    --run_id maize_cell_type_accessible_c1000_cnn-lstm-imbalance-lr-1e-3 \
    --learning_rate 0.001 \
    --wandb_project PlantCAD2 \
    --balance False \
    --num_labels 92

# predict
DATA_DIR=/local/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/maize_cell_type_accessible_c1000
CKPT_DIR=${DATA_DIR}/cnn-lstm-imbalance-lr-1e-3
for species in test
do
    CUDA_VISIBLE_DEVICES=1 python main.py predict \
        --test ${DATA_DIR}/${species}.tsv \
        --model_path ${CKPT_DIR}/bestmodel.ckpt \
        --output_file ${CKPT_DIR}/${species}_predictions.tsv
done