OUTPUT_DIR=/local/workdir/jz963/utils/plantcad2/results/review/1_benchmark_evo2/plantcad2
TEST_EMBEDDINGS=${OUTPUT_DIR}/test_zea_mays_large.pkl

# evaluate xgboost
python plantcad2.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/xgb_plantcad2_large_mean.pkl \
    --output ${OUTPUT_DIR}/xgb_plantcad2_large_mean_zea_mays_results.tsv

python plantcad2.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/xgb_plantcad2_large_max.pkl \
    --output ${OUTPUT_DIR}/xgb_plantcad2_large_max_zea_mays_results.tsv

# evaluate neural network
python plantcad2.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/nn_plantcad2_large_mean.pkl \
    --output ${OUTPUT_DIR}/nn_plantcad2_large_mean_zea_mays_results.tsv

python plantcad2.py evaluate \
    --test-embeddings ${TEST_EMBEDDINGS} \
    --model ${OUTPUT_DIR}/nn_plantcad2_large_max.pkl \
    --output ${OUTPUT_DIR}/nn_plantcad2_large_max_zea_mays_results.tsv
