DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene/outputs/"
TAXA="B73"

python main.py -input ${DATA_DIR}${TAXA}_start_sites_filtered.fasta \
 -outLogit ${OUTPUT}${TAXA}_start_sites_mask_ATG_gpn.tsv \
 -model 'songlab/gpn-brassicales' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 254,255,256 

python main.py -input ${DATA_DIR}${TAXA}_stop_sites_filtered.fasta \
 -outLogit ${OUTPUT}${TAXA}_stop_sites_mask_TAG_gpn.tsv \
 -model 'songlab/gpn-brassicales' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 254,255,256

python main.py -input ${DATA_DIR}${TAXA}_donor_filtered.fasta \
 -outLogit ${OUTPUT}${TAXA}_donor_mask_GT_gpn.tsv \
 -model 'songlab/gpn-brassicales' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 255,256

python main.py -input ${DATA_DIR}${TAXA}_acceptor_filtered.fasta \
 -outLogit ${OUTPUT}${TAXA}_acceptor_mask_AG_gpn.tsv \
 -model 'songlab/gpn-brassicales' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 255,256