#!/bin/bash
python ../../pipelines/expression_panand/1_plantcad2/main.py tokenize \
    --data_dir ../../results/review/4_gene_exp_more_species/osa_leaf_bin.tsv \
    --model_name 'kuleshov-group/PlantCaduceus_l32' \
    --sequence_length 2048

python ../../pipelines/expression_panand/1_plantcad2/main.py tokenize \
    --data_dir ../../results/review/4_gene_exp_more_species/osa_leaf_exp.tsv \
    --model_name 'kuleshov-group/PlantCaduceus_l32' \
    --sequence_length 2048

python ../../pipelines/expression_panand/1_plantcad2/main.py tokenize \
    --data_dir ../../results/review/4_gene_exp_more_species/sly_leaf_bin.tsv \
    --model_name 'kuleshov-group/PlantCaduceus_l32' \
    --sequence_length 2048

python ../../pipelines/expression_panand/1_plantcad2/main.py tokenize \
    --data_dir ../../results/review/4_gene_exp_more_species/sly_leaf_exp.tsv \
    --model_name 'kuleshov-group/PlantCaduceus_l32' \
    --sequence_length 2048

python ../../pipelines/expression_panand/2_AgroNT/main.py tokenize \
    --data_dir ../../results/review/4_gene_exp_more_species/osa_leaf_bin.tsv \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --sequence_length 2048

python ../../pipelines/expression_panand/2_AgroNT/main.py tokenize \
    --data_dir ../../results/review/4_gene_exp_more_species/sly_leaf_bin.tsv \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --sequence_length 2048

python ../../pipelines/expression_panand/2_AgroNT/main.py tokenize \
    --data_dir ../../results/review/4_gene_exp_more_species/osa_leaf_exp.tsv \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --sequence_length 2048

python ../../pipelines/expression_panand/2_AgroNT/main.py tokenize \
    --data_dir ../../results/review/4_gene_exp_more_species/sly_leaf_exp.tsv \
    --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
    --sequence_length 2048
