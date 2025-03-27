DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene/outputs/"
TAXA="B73"
CHECKPOINT="kuleshov-group/compo-cad2-l48-d1536-dna-chtk-c8k-1t-v1-b2-lr4e4-NzqiLr"
PREFIX="pcv2-l48-d1536"
BATCH_SIZE=16
DEVICE="1"

CUDA_VISIBLE_DEVICES=${DEVICE} python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_start_sites_filtered.fasta \
    -output ${OUTPUT}${TAXA}_start_sites_mask_ATG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

CUDA_VISIBLE_DEVICES=${DEVICE} python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_stop_sites_filtered.fasta \
    -output ${OUTPUT}${TAXA}_stop_sites_mask_TAG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

CUDA_VISIBLE_DEVICES=${DEVICE} python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_donor_filtered.fasta \
    -output ${OUTPUT}${TAXA}_donor_mask_GT_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4095,4096

CUDA_VISIBLE_DEVICES=${DEVICE} python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_acceptor_filtered.fasta \
    -output ${OUTPUT}${TAXA}_acceptor_mask_AG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4095,4096
