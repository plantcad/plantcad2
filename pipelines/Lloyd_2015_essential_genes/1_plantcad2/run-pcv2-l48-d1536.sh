DATA_DIR="/workdir/jz963/utils/plantcad2/results/Lloyd_2015_essential_genes/zero_shot_input/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/Lloyd_2015_essential_genes/plantcad2_logits/"
TAXA="Ath"
CHECKPOINT="kuleshov-group/compo-cad2-l48-d1536-dna-chtk-c8k-1t-v1-b2-lr4e4-NzqiLr"
PREFIX="pcv2-l48-d1536"
BATCH_SIZE=16

CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene/main.py -input ${DATA_DIR}${TAXA}_filtered_tis.fasta \
    -output ${OUTPUT}${TAXA}_tis_mask_ATG_${PREFIX}.h5 \
    -model ${CHECKPOINT} \
    -batchSize ${BATCH_SIZE} -tokenIdx 4094,4095,4096

CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene/main.py -input ${DATA_DIR}${TAXA}_filtered_tts.fasta \
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
