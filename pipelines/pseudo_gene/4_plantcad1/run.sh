DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene/outputs/"
TAXA="B73"

python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_start_sites_filtered.fasta \
 -output ${OUTPUT}${TAXA}_start_sites_mask_ATG_PlantCaduceus_l32.h5 \
 -model 'kuleshov-group/PlantCaduceus_l32' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 254,255,256 \
 -short_sequence_length 512

python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_stop_sites_filtered.fasta \
 -output ${OUTPUT}${TAXA}_stop_sites_mask_TAG_PlantCaduceus_l32.h5 \
 -model 'kuleshov-group/PlantCaduceus_l32' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 254,255,256 \
 -short_sequence_length 512

python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_donor_filtered.fasta \
 -output ${OUTPUT}${TAXA}_donor_mask_GT_PlantCaduceus_l32.h5 \
 -model 'kuleshov-group/PlantCaduceus_l32' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 255,256 \
 -short_sequence_length 512

python ../../../masked_token_accuracy.py -input ${DATA_DIR}${TAXA}_acceptor_filtered.fasta \
 -output ${OUTPUT}${TAXA}_acceptor_mask_AG_PlantCaduceus_l32.h5 \
 -model 'kuleshov-group/PlantCaduceus_l32' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 255,256 \
 -short_sequence_length 512