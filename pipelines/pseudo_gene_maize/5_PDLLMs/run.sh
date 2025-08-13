DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene/outputs/"
TAXA="B73"

# using the upstream 4094 base pairs as prompts, no start codon provided, predict ATG
CUDA_VISIBLE_DEVICES=1 python main.py --fasta ${DATA_DIR}${TAXA}_start_sites_filtered.fasta \
    --length 254 \
    --output ${OUTPUT}${TAXA}_start_sites_pdllm_ntokens_3 \
    --batch-size 128 --n-tokens 3

CUDA_VISIBLE_DEVICES=1 python main.py --fasta ${DATA_DIR}${TAXA}_stop_sites_filtered.fasta \
    --length 254 \
    --output ${OUTPUT}${TAXA}_stop_sites_pdllm_ntokens_3 \
    --batch-size 128 --n-tokens 3


# splicing donor site, using the downstream 4095 base pairs as prompts, predict GT
CUDA_VISIBLE_DEVICES=1 python main.py --fasta ${DATA_DIR}${TAXA}_donor_filtered.fasta \
    --length 255 \
    --output ${OUTPUT}${TAXA}_donor_pdllm_ntokens_2 \
    --batch-size 128 --n-tokens 2

# splicing acceptor site, using the upstream 4095 base pairs as prompts, predict AG
CUDA_VISIBLE_DEVICES=1 python main.py --fasta ${DATA_DIR}${TAXA}_acceptor_filtered.fasta \
    --length 255 \
    --output ${OUTPUT}${TAXA}_acceptor_pdllm_ntokens_2 \
    --batch-size 128 --n-tokens 2




#########################Input reverse complement sequences#######################

CUDA_VISIBLE_DEVICES=0 python main.py --fasta ${DATA_DIR}${TAXA}_start_sites_filtered.fasta \
    --length 255 \
    --output ${OUTPUT}${TAXA}_start_sites_pdllm_rc_ntokens_3 \
    --batch-size 128 \
    --n-tokens 3 \
    --reverse

CUDA_VISIBLE_DEVICES=0 python main.py --fasta ${DATA_DIR}${TAXA}_stop_sites_filtered.fasta \
    --length 255 \
    --output ${OUTPUT}${TAXA}_stop_sites_pdllm_rc_ntokens_3 \
    --batch-size 128 \
    --n-tokens 3 \
    --reverse

CUDA_VISIBLE_DEVICES=0 python main.py --fasta ${DATA_DIR}${TAXA}_donor_filtered.fasta \
    --length 255 \
    --output ${OUTPUT}${TAXA}_donor_pdllm_rc_ntokens_2 \
    --batch-size 128 \
    --n-tokens 2 \
    --reverse

CUDA_VISIBLE_DEVICES=0 python main.py --fasta ${DATA_DIR}${TAXA}_acceptor_filtered.fasta \
    --length 255 \
    --output ${OUTPUT}${TAXA}_acceptor_pdllm_rc_ntokens_2 \
    --batch-size 128 \
    --n-tokens 2 \
    --reverse
