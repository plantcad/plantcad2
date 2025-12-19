OUTPUT_DIR=/local/workdir/jz963/utils/plantcad2/results/review/1_benchmark_evo2/evo2
TEST_EMBEDDINGS=${OUTPUT_DIR}/test_zea_mays_evo2.pkl

# evaluate xgboost
python evo2_embedding.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/xgb_evo2_forward.pkl \
    --output ${OUTPUT_DIR}/xgb_evo2_forward_zea_mays_results.tsv

python evo2_embedding.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/xgb_evo2_reverse.pkl \
    --output ${OUTPUT_DIR}/xgb_evo2_reverse_zea_mays_results.tsv

python evo2_embedding.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/xgb_evo2_average.pkl \
    --output ${OUTPUT_DIR}/xgb_evo2_average_zea_mays_results.tsv

python evo2_embedding.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/xgb_evo2_concatenate.pkl \
    --output ${OUTPUT_DIR}/xgb_evo2_concatenate_zea_mays_results.tsv

# evaluate neural network
python evo2_embedding.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/nn_evo2_forward.pkl \
    --output ${OUTPUT_DIR}/nn_evo2_forward_zea_mays_results.tsv

python evo2_embedding.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/nn_evo2_reverse.pkl \
    --output ${OUTPUT_DIR}/nn_evo2_reverse_zea_mays_results.tsv

python evo2_embedding.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/nn_evo2_average.pkl \
    --output ${OUTPUT_DIR}/nn_evo2_average_zea_mays_results.tsv

python evo2_embedding.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/nn_evo2_concatenate.pkl \
    --output ${OUTPUT_DIR}/nn_evo2_concatenate_zea_mays_results.tsv
