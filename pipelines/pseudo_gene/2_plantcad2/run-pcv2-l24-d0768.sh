DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene/outputs/"
TAXA="B73"
CHECKPOINT="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000"
PREFIX="pcv2-l24-d0768"
BATCH_SIZE=16
DEVICE="cuda:0"

python ../main.py -input ${DATA_DIR}${TAXA}_start_sites_filtered.fasta \
    -output ${OUTPUT}${TAXA}_start_sites_mask_ATG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -device ${DEVICE} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

python ../main.py -input ${DATA_DIR}${TAXA}_stop_sites_filtered.fasta \
    -output ${OUTPUT}${TAXA}_stop_sites_mask_TAG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -device ${DEVICE} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

python ../main.py -input ${DATA_DIR}${TAXA}_donor_filtered.fasta \
    -output ${OUTPUT}${TAXA}_donor_mask_GT_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -device ${DEVICE} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4095,4096

python ../main.py -input ${DATA_DIR}${TAXA}_acceptor_filtered.fasta \
    -output ${OUTPUT}${TAXA}_acceptor_mask_AG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -device ${DEVICE} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4095,4096
