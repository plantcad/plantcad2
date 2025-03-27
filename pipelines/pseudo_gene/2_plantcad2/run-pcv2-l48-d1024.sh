DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene/outputs/"
TAXA="B73"
CHECKPOINT="kuleshov-group/compo-cad2-l48-dna-chtk-c8k-1t-v2d-b2-lr4e4-fcpffa-ba476838"
PREFIX="pcv2-l48-d1024"
BATCH_SIZE=16
DEVICE="cuda:0"

python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_start_sites_filtered.fasta \
    -output ${OUTPUT}${TAXA}_start_sites_mask_ATG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -device ${DEVICE} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_stop_sites_filtered.fasta \
    -output ${OUTPUT}${TAXA}_stop_sites_mask_TAG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -device ${DEVICE} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_donor_filtered.fasta \
    -output ${OUTPUT}${TAXA}_donor_mask_GT_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -device ${DEVICE} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4095,4096

python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_acceptor_filtered.fasta \
    -output ${OUTPUT}${TAXA}_acceptor_mask_AG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -device ${DEVICE} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4095,4096
