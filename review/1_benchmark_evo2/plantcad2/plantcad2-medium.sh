INPUT_DIR=/workdir/jz963/datasets/PlantCAD2_tasks/fine-tuning/cross_species_acr_train_on_arabidopsis
OUTPUT_DIR=/local/workdir/jz963/utils/plantcad2/results/review/1_benchmark_evo2/plantcad2
MODEL='kuleshov-group/PlantCAD2-Medium-l48-d1024'

# train xgboost
python plantcad2.py train \
    --train-embeddings ${OUTPUT_DIR}/train_medium.pkl \
    --val-embeddings ${OUTPUT_DIR}/valid_medium.pkl \
    --pooling mean \
    --output ${OUTPUT_DIR}/xgb_plantcad2_medium_mean.pkl

python plantcad2.py train \
    --train-embeddings ${OUTPUT_DIR}/train_medium.pkl \
    --val-embeddings ${OUTPUT_DIR}/valid_medium.pkl \
    --pooling max \
    --output ${OUTPUT_DIR}/xgb_plantcad2_medium_max.pkl

# train neural network
python plantcad2.py train-nn \
    --train-embeddings ${OUTPUT_DIR}/train_medium.pkl \
    --val-embeddings ${OUTPUT_DIR}/valid_medium.pkl \
    --pooling mean \
    --num-hidden-layers 1 \
    --dropout 0.1 \
    --batch-size 64 \
    --epochs 200 \
    --patience 20 \
    --learning-rate 0.001 \
    --weight-decay 0.0 \
    --output ${OUTPUT_DIR}/nn_plantcad2_medium_mean.pkl

python plantcad2.py train-nn \
    --train-embeddings ${OUTPUT_DIR}/train_medium.pkl \
    --val-embeddings ${OUTPUT_DIR}/valid_medium.pkl \
    --pooling max \
    --num-hidden-layers 1 \
    --dropout 0.1 \
    --batch-size 64 \
    --epochs 200 \
    --patience 20 \
    --learning-rate 0.001 \
    --weight-decay 0.0 \
    --output ${OUTPUT_DIR}/nn_plantcad2_medium_max.pkl

# evaluate xgboost
python plantcad2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/valid_medium.pkl \
    --model ${OUTPUT_DIR}/xgb_plantcad2_medium_mean.pkl \
    --output ${OUTPUT_DIR}/xgb_plantcad2_medium_mean_results.tsv

python plantcad2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/valid_medium.pkl \
    --model ${OUTPUT_DIR}/xgb_plantcad2_medium_max.pkl \
    --output ${OUTPUT_DIR}/xgb_plantcad2_medium_max_results.tsv

# evaluate neural network
python plantcad2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/valid_medium.pkl \
    --model ${OUTPUT_DIR}/nn_plantcad2_medium_mean.pkl \
    --output ${OUTPUT_DIR}/nn_plantcad2_medium_mean_results.tsv

python plantcad2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/valid_medium.pkl \
    --model ${OUTPUT_DIR}/nn_plantcad2_medium_max.pkl \
    --output ${OUTPUT_DIR}/nn_plantcad2_medium_max_results.tsv
