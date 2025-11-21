#!/bin/bash
# for binary expression prediction
## for plantcad2 small, medium, large

run_predict () {
    species=$1
    gpu=$2

    # Small
    CUDA_VISIBLE_DEVICES=$gpu python ../../pipelines/expression_panand/1_plantcad2/main.py predict \
        --checkpoint_dir "plantcad/cross_species_leaf_on_off_expression_plantcad2_small" \
        --data_dir ../../results/review/4_gene_exp_more_species/${species}_leaf_bin.parquet \
        --task_type "classification" \
        --output_file ../../results/review/4_gene_exp_more_species/predictions/${species}_leaf_bin_plantcad2-small.tsv \
        --model_name 'kuleshov-group/PlantCAD2-Small-l24-d0768' \
        --batch_size 64

    # Medium
    CUDA_VISIBLE_DEVICES=$gpu python ../../pipelines/expression_panand/1_plantcad2/main.py predict \
        --checkpoint_dir "plantcad/cross_species_leaf_on_off_expression_plantcad2_medium" \
        --data_dir ../../results/review/4_gene_exp_more_species/${species}_leaf_bin.parquet \
        --task_type "classification" \
        --output_file ../../results/review/4_gene_exp_more_species/predictions/${species}_leaf_bin_plantcad2-medium.tsv \
        --model_name 'kuleshov-group/PlantCAD2-Medium-l48-d1024' \
        --batch_size 64

    # Large
    CUDA_VISIBLE_DEVICES=$gpu python ../../pipelines/expression_panand/1_plantcad2/main.py predict \
        --checkpoint_dir "plantcad/cross_species_leaf_on_off_expression_plantcad2_large" \
        --data_dir ../../results/review/4_gene_exp_more_species/${species}_leaf_bin.parquet \
        --task_type "classification" \
        --output_file ../../results/review/4_gene_exp_more_species/predictions/${species}_leaf_bin_plantcad2-large.tsv \
        --model_name 'kuleshov-group/PlantCAD2-Large-l48-d1536' \
        --batch_size 64
}

run_predict osa 0 & run_predict sly 1 & wait


## for AgroNT
for species in osa sly
do
    CUDA_VISIBLE_DEVICES=0 python ../../pipelines/expression_panand/2_AgroNT/main.py predict \
        --data_dir ../../results/review/4_gene_exp_more_species/${species}_leaf_bin.agront.parquet \
        --checkpoint_dir ../../results/PlantCAD2_tasks/exp-leaf-bin/agront-checkpoints-lr-1e-4/checkpoint-7975 \
        --output_file --output_file ../../results/review/4_gene_exp_more_species/predictions/${species}_leaf_bin_agront.tsv \
        --model_name "InstaDeepAI/agro-nucleotide-transformer-1b" \
        --batch_size 64
done &

## supervised plantcad2
for species in osa sly
do
    CUDA_VISIBLE_DEVICES=1 python main.py predict \
    --data_dir ../../results/review/4_gene_exp_more_species/${species}_leaf_bin.parquet \
    --checkpoint_dir ../../results/PlantCAD2_tasks/exp-leaf-bin/sup_pcv2-l24-d0768-checkpoints-lr-1e-4/checkpoint-7975 \
    --output_file ../../results/review/4_gene_exp_more_species/predictions/${species}_leaf_bin_sup-plantcad2.tsv
done 

## supervised cnn-lstm
for species in osa sly
    python ../../pipelines/expression_panand/3_sup_cnn-lstm/main.py predict \
        --model_path ../../results/PlantCAD2_tasks/exp-leaf-abs/cnn_lstm/bestmodel.ckpt \
        --test ../../results/review/4_gene_exp_more_species/${species}_leaf_exp.tsv \
        --output_file ../../results/review/4_gene_exp_more_species/predictions/${species}_leaf_bin_cnn-lstm.tsv \
        --device cuda:0 \
        --batch_size 2048
done