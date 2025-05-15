DATA_DIR=/local/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/accessible_angiosperm_c600/data_leave_out_zma_hvu
CKPT_DIR=${DATA_DIR}/cnn-lstm-imbalance-lr-1e-3
CUDA_VISIBLE_DEVICES=1 python main.py train  \
    --train $DATA_DIR/train.tsv \
    --valid $DATA_DIR/valid.tsv \
    --output_dir ${CKPT_DIR} \
    --run_id accessible_angiosperm_c600_leave_zma_hvu_cnn-lstm-imbalance-lr-1e-3 \
    --learning_rate 0.001 \
    --wandb_project PlantCAD2 \
    --balance False