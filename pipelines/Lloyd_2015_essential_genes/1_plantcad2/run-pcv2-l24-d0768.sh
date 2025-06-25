DATA_DIR="/workdir/jz963/utils/plantcad2/results/Lloyd_2015_essential_genes/zero_shot_input/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/Lloyd_2015_essential_genes/plantcad2_logits/"
TAXA="Ath"
CHECKPOINT="kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000"
PREFIX="pcv2-l24-d0768"
BATCH_SIZE=16

CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene/main.py -input ${DATA_DIR}${TAXA}_filtered_tis.fasta \
    -output ${OUTPUT}${TAXA}_tis_mask_ATG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene/main.py -input ${DATA_DIR}${TAXA}_filtered_tss.fasta \
    -output ${OUTPUT}${TAXA}_stop_sites_mask_TAG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene/main.py -input ${DATA_DIR}${TAXA}_filtered_donor.fasta \
    -output ${OUTPUT}${TAXA}_donor_mask_GT_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4095,4096

CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene/main.py -input ${DATA_DIR}${TAXA}_filtered_acceptor.fasta \
    -output ${OUTPUT}${TAXA}_acceptor_mask_AG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4095,4096