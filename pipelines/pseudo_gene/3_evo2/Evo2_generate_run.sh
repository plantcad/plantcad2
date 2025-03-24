# using the upstream 4094 base pairs as prompts, no start codon provided, predict ATG
CUDA_VISIBLE_DEVICES=0 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_start_sites_filtered.fasta --length 4094 --output splice_junctions/B73_start_sites_evo2_7b_ntokens_3 --batch-size 2 --n-tokens 3

# using the downstream 4095 base pairs as prompts, A is provided, predict TG
CUDA_VISIBLE_DEVICES=1 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_start_sites_filtered.fasta --length 4095 --output splice_junctions/B73_start_sites_evo2_7b_ntokens_2 --batch-size 2 --n-tokens 2

# using the downstream 4096 base pairs as prompts, AT is provided, try to predict G
CUDA_VISIBLE_DEVICES=0 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_start_sites_filtered.fasta --length 4096 --output splice_junctions/B73_start_sites_evo2_7b_ntokens_1 --batch-size 2 --n-tokens 1

# splicing donor site, using the downstream 4095 base pairs as prompts, predict GT
CUDA_VISIBLE_DEVICES=1 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_donor_filtered.fasta --length 4095 --output splice_junctions/B73_donor_filtered_evo2_7b_ntokens_2 --batch-size 2 --n-tokens 2
# splicing donor site, using the downstream 4096 base pairs as prompts, G is provided, try to predict T
CUDA_VISIBLE_DEVICES=0 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_donor_filtered.fasta --length 4096 --output splice_junctions/B73_donor_filtered_evo2_7b_ntokens_1 --batch-size 2 --n-tokens 1

# splicing acceptor site, using the upstream 4095 base pairs as prompts, predict AG
CUDA_VISIBLE_DEVICES=1 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_acceptor_filtered.fasta --length 4095 --output splice_junctions/B73_acceptor_filtered_evo2_7b_ntokens_2 --batch-size 2 --n-tokens 2

# splicing acceptor site, using the upstream 4096 base pairs as prompts, A is provided, try to predict G
CUDA_VISIBLE_DEVICES=0 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_acceptor_filtered.fasta --length 4096 --output splice_junctions/B73_acceptor_filtered_evo2_7b_ntokens_1 --batch-size 2 --n-tokens 1

# using the upstream 4094 base pairs as prompts, no stop codon provided, predict stop codon
CUDA_VISIBLE_DEVICES=1 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_stop_sites_filtered.fasta --length 4094 --output splice_junctions/B73_stop_sites_evo2_7b_ntokens_3 --batch-size 2 --n-tokens 3

# using the downstream 4095 base pairs as prompts, T is provided, predict TAA, TAG, TGA
CUDA_VISIBLE_DEVICES=0 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_stop_sites_filtered.fasta --length 4095 --output splice_junctions/B73_stop_sites_evo2_7b_ntokens_2 --batch-size 2 --n-tokens 2

# using the downstream 4096 base pairs as prompts, T[AG] is provided, try to predict [AG]
CUDA_VISIBLE_DEVICES=1 python ../../tasks/pseudo_gene/evo2_generate.py --fasta splice_junctions/B73_stop_sites_filtered.fasta --length 4096 --output splice_junctions/B73_stop_sites_evo2_7b_ntokens_1 --batch-size 2 --n-tokens 1
