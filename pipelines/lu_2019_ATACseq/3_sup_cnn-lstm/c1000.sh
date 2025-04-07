DATA_DIR=/local/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/accessible_angiosperm_c1000
CKPT_DIR=${DATA_DIR}/cnn-lstm-imbalance-lr-1e-3
CUDA_VISIBLE_DEVICES=1 python main.py train  \
    --train $DATA_DIR/train.tsv \
    --valid $DATA_DIR/valid.tsv \
    --output_dir ${CKPT_DIR} \
    --run_id accessible_angiosperm_c1000_cnn-lstm-imbalance-lr-1e-3 \
    --learning_rate 0.001 \
    --wandb_project PlantCAD2 \
    --balance False

DATA_DIR=/local/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/accessible_angiosperm_c1000
CKPT_DIR=${DATA_DIR}/cnn-lstm-balance-lr-1e-3
CUDA_VISIBLE_DEVICES=1 python main.py train  \
    --train $DATA_DIR/train.tsv \
    --valid $DATA_DIR/valid.tsv \
    --output_dir ${CKPT_DIR} \
    --run_id accessible_angiosperm_c1000_cnn-lstm-balance-lr-1e-3 \
    --learning_rate 0.001 \
    --wandb_project PlantCAD2 \
    --balance True


# predict
DATA_DIR=/local/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/accessible_angiosperm_c1000
CKPT_DIR=${DATA_DIR}/models/cnn-lstm-imbalance-lr-1e-3
for species in test_brachypodium_distachyon test_eutrema_salsugineum test_glycine_max test_hordeum_vulgare test_oryza_sativa test_phaseolus_vulgaris test_populus_trichocarpa test_setaria_viridis test_sorghum_bicolor test_zea_mays
do
    CUDA_VISIBLE_DEVICES=1 python main.py predict \
        --test ${DATA_DIR}/data/${species}.tsv \
        --model_path ${CKPT_DIR}/bestmodel.ckpt \
        --output_file ${CKPT_DIR}/${species}_scores.tsv
done


DATA_DIR=/local/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/accessible_angiosperm_c1000
CKPT_DIR=${DATA_DIR}/models/cnn-lstm-balance-lr-1e-3
for species in test_brachypodium_distachyon test_eutrema_salsugineum test_glycine_max test_hordeum_vulgare test_oryza_sativa test_phaseolus_vulgaris test_populus_trichocarpa test_setaria_viridis test_sorghum_bicolor test_zea_mays
do
    CUDA_VISIBLE_DEVICES=1 python main.py predict \
        --test ${DATA_DIR}/data/${species}.tsv \
        --model_path ${CKPT_DIR}/bestmodel.ckpt \
        --output_file ${CKPT_DIR}/${species}_scores.tsv
done