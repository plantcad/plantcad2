INPUT_DIR=/workdir/jz963/datasets/PlantCAD2_tasks/fine-tuning/cross_species_acr_train_on_arabidopsis
OUTPUT_DIR=/local/workdir/jz963/utils/plantcad2/results/review/1_benchmark_evo2/evo2
CUDA_VISIBLE_DEVICES=0 python evo2.py fetch \
    --input ${INPUT_DIR}/train.tsv \
    --output ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --batch-size 4

CUDA_VISIBLE_DEVICES=0 python evo2.py fetch \
    --input ${INPUT_DIR}/valid.tsv \
    --output ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --batch-size 4

# train xgboost
python evo2.py train \
    --train-embeddings ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --val-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --strategy forward \
    --output ${OUTPUT_DIR}/xgb_evo2_forward.pkl 

python evo2.py train \
    --train-embeddings ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --val-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --strategy reverse \
    --output ${OUTPUT_DIR}/xgb_evo2_reverse.pkl &

python evo2.py train \
    --train-embeddings ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --val-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --strategy average \
    --output ${OUTPUT_DIR}/xgb_evo2_average.pkl &

python evo2.py train \
    --train-embeddings ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --val-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --strategy concatenate \
    --output ${OUTPUT_DIR}/xgb_evo2_concatenate.pkl &

# train neural network
CUDA_VISIBLE_DEVICES=1 python evo2.py train-nn \
    --train-embeddings ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --val-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --strategy forward \
    --num-hidden-layers 1 \
    --dropout 0.1 \
    --batch-size 64 \
    --epochs 200 \
    --patience 20 \
    --learning-rate 0.001 \
    --weight-decay 0.0 \
    --output ${OUTPUT_DIR}/nn_evo2_forward.pkl &

python evo2.py train-nn \
    --train-embeddings ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --val-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --strategy reverse \
    --num-hidden-layers 1 \
    --dropout 0.1 \
    --batch-size 64 \
    --epochs 200 \
    --patience 20 \
    --learning-rate 0.001 \
    --weight-decay 0.0 \
    --output ${OUTPUT_DIR}/nn_evo2_reverse.pkl &

CUDA_VISIBLE_DEVICES=1 python evo2.py train-nn \
    --train-embeddings ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --val-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --strategy average \
    --num-hidden-layers 1 \
    --dropout 0.1 \
    --batch-size 64 \
    --epochs 200 \
    --patience 20 --learning-rate 0.001 \
    --weight-decay 0.0 \
    --output ${OUTPUT_DIR}/nn_evo2_average.pkl &

CUDA_VISIBLE_DEVICES=1 python evo2.py train-nn \
    --train-embeddings ${OUTPUT_DIR}/evo2_train_embeddings.pkl \
    --val-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --strategy concatenate \
    --num-hidden-layers 1 \
    --dropout 0.1 \
    --batch-size 64 \
    --epochs 200 \
    --patience 20 \
    --learning-rate 0.001 \
    --weight-decay 0.0 \
    --output ${OUTPUT_DIR}/nn_evo2_concatenate.pkl &

# evaluate xgboost
python evo2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --model ${OUTPUT_DIR}/xgb_evo2_forward.pkl \
    --output ${OUTPUT_DIR}/xgb_evo2_forward_results.tsv

python evo2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --model ${OUTPUT_DIR}/xgb_evo2_reverse.pkl \
    --output ${OUTPUT_DIR}/xgb_evo2_reverse_results.tsv 

python evo2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --model ${OUTPUT_DIR}/xgb_evo2_average.pkl \
    --output ${OUTPUT_DIR}/xgb_evo2_average_results.tsv 

python evo2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --model ${OUTPUT_DIR}/xgb_evo2_concatenate.pkl \
    --output ${OUTPUT_DIR}/xgb_evo2_concatenate_results.tsv 

# evaluate neural network
python evo2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --model ${OUTPUT_DIR}/nn_evo2_forward.pkl \
    --output ${OUTPUT_DIR}/nn_evo2_forward_results.tsv  

python evo2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --model ${OUTPUT_DIR}/nn_evo2_reverse.pkl \ 
    --output ${OUTPUT_DIR}/nn_evo2_reverse_results.tsv

python evo2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --model ${OUTPUT_DIR}/nn_evo2_average.pkl \
    --output ${OUTPUT_DIR}/nn_evo2_average_results.tsv

python evo2.py evaluate \
    --test-embeddings ${OUTPUT_DIR}/evo2_valid_embeddings.pkl \
    --model ${OUTPUT_DIR}/nn_evo2_concatenate.pkl \
    --output ${OUTPUT_DIR}/nn_evo2_concatenate_results.tsv  