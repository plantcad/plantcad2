DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene_tomato/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene_tomato/outputs/"
TAXA="SL3.0_ITAG3.2"

# using the upstream 4094 base pairs as prompts, no start codon provided, predict ATG
CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene_maize/2_evo2/evo2_generate.py \
    --fasta ${DATA_DIR}${TAXA}_start_sites_filtered_rmless8192.fasta \
    --length 4094 \
    --output ${OUTPUT_DIR}/${TAXA}_start_sites_evo2_7b_ntokens_3 \
    --batch-size 2 --n-tokens 3

# using the upstream 4094 base pairs as prompts, no stop codon provided, predict stop codon
CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene_maize/2_evo2/evo2_generate.py \
    --fasta ${DATA_DIR}${TAXA}_stop_sites_filtered_rmless8192.fasta \
    --length 4094 \
    --output ${OUTPUT_DIR}/${TAXA}_stop_sites_evo2_7b_ntokens_3 \
    --batch-size 2 --n-tokens 3


# splicing donor site, using the downstream 4095 base pairs as prompts, predict GT
CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene_maize/2_evo2/evo2_generate.py \
    --fasta ${DATA_DIR}/${TAXA}_donor_sites_filtered_rmless8192.fasta \
    --length 4095 \
    --output ${OUTPUT_DIR}/${TAXA}_donor_filtered_evo2_7b_ntokens_2 \
    --batch-size 2 --n-tokens 2


# splicing acceptor site, using the upstream 4095 base pairs as prompts, predict AG
CUDA_VISIBLE_DEVICES=1 python ../../pseudo_gene_maize/2_evo2/evo2_generate.py \
    --fasta ${DATA_DIR}/${TAXA}_acceptor_filtered_rmless8192.fasta \
    --length 4095 \
    --output ${OUTPUT_DIR}/${TAXA}_acceptor_filtered_evo2_7b_ntokens_2 \
    --batch-size 2 --n-tokens 2





#########################Input reverse complement sequences#######################

CUDA_VISIBLE_DEVICES=0 python evo2_generate.py \
    --fasta ${DATA_DIR}${TAXA}_start_sites_filtered_rmless8192.fasta \
    --length 4095 \
    --output ${OUTPUT}${TAXA}_start_sites_evo2_7b_rc_ntokens_3 \
    --batch-size 2 \
    --n-tokens 3 \
    --reverse

CUDA_VISIBLE_DEVICES=1 python evo2_generate.py \
    --fasta ${DATA_DIR}${TAXA}_stop_sites_filtered_rmless8192.fasta \
    --length 4095 \
    --output ${OUTPUT}${TAXA}_stop_sites_evo2_7b_rc_ntokens_3 \
    --batch-size 2 \
    --n-tokens 3 \
    --reverse

CUDA_VISIBLE_DEVICES=0 python evo2_generate.py \
    --fasta ${DATA_DIR}${TAXA}_donor_sites_filtered_rmless8192.fasta \
    --length 4095 \
    --output ${OUTPUT}${TAXA}_donor_evo2_7b_rc_ntokens_2 \
    --batch-size 2 \
    --n-tokens 2 \
    --reverse

CUDA_VISIBLE_DEVICES=1 python evo2_generate.py \
    --fasta ${DATA_DIR}${TAXA}_acceptor_sites_filtered_rmless8192.fasta \
    --length 4095 \
    --output ${OUTPUT}${TAXA}_acceptor_evo2_7b_rc_ntokens_2 \
    --batch-size 2 \
    --n-tokens 2 \
    --reverse
