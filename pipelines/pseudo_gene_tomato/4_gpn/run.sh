DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene_tomato/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene_tomato/outputs/"
TAXA="SL3.0_ITAG3.2"


python main.py -input ${DATA_DIR}${TAXA}_start_sites_filtered_rmless8192.fasta \
 -outLogit ${OUTPUT}${TAXA}_start_sites_mask_ATG_gpn.tsv \
 -model 'songlab/gpn-brassicales' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 254,255,256 

python main.py -input ${DATA_DIR}${TAXA}_stop_sites_filtered_rmless8192.fasta \
 -outLogit ${OUTPUT}${TAXA}_stop_sites_mask_TAG_gpn.tsv \
 -model 'songlab/gpn-brassicales' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 254,255,256

python main.py -input ${DATA_DIR}${TAXA}_donor_sites_filtered_rmless8192.fasta \
 -outLogit ${OUTPUT}${TAXA}_donor_mask_GT_gpn.tsv \
 -model 'songlab/gpn-brassicales' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 255,256

python main.py -input ${DATA_DIR}${TAXA}_acceptor_sites_filtered_rmless8192.fasta \
 -outLogit ${OUTPUT}${TAXA}_acceptor_mask_AG_gpn.tsv \
 -model 'songlab/gpn-brassicales' \
 -device 'cuda:1' \
 -batchSize 512 -tokenIdx 255,256