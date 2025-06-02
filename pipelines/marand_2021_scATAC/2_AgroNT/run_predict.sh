DATA_DIR="../../../results/PlantCAD2_tasks/maize_cell_type_accessible_c300/"
CUDA_VISIBLE_DEVICES=0 python main.py predict \
    --data_dir ${DATA_DIR}/test.agront.parquet \
    --num_labels 92 \
    --checkpoint_dir ${DATA_DIR}/agront-checkpoints-lr-1e-4/checkpoint-39034 \
    --output_file ${DATA_DIR}/agront-checkpoints-lr-1e-4/test_predictions.tsv \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --batch_size 512

DATA_DIR="../../../results/PlantCAD2_tasks/maize_cell_type_accessible_c600/"
CUDA_VISIBLE_DEVICES=1 python main.py predict \
    --data_dir ${DATA_DIR}/test.agront.parquet \
    --num_labels 92 \
    --checkpoint_dir ${DATA_DIR}/agront-checkpoints-lr-1e-4/checkpoint-19666 \
    --output_file ${DATA_DIR}/agront-checkpoints-lr-1e-4/test_predictions.tsv \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --batch_size 512

DATA_DIR="../../../results/PlantCAD2_tasks/maize_cell_type_accessible_c1000/"
CUDA_VISIBLE_DEVICES=0 python main.py predict \
    --data_dir ${DATA_DIR}/test.agront.parquet \
    --num_labels 92 \
    --checkpoint_dir ${DATA_DIR}/agront-checkpoints-lr-1e-4/checkpoint-11807 \
    --output_file ${DATA_DIR}/agront-checkpoints-lr-1e-4/test_predictions.tsv \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --batch_size 512