#!/bin/bash
DATA_DIR=../../results/review/9_ap2_tf_family/
INPUT_BED=${DATA_DIR}/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events.bed

# Split the input file into two parts
split -d -n l/2 ${INPUT_BED} ${DATA_DIR}/split_temp_

# Run the first part on cuda:0
CUDA_VISIBLE_DEVICES=0 python zero_shot_score.py \
    -input-bed ${DATA_DIR}/split_temp_00 \
    -input-fasta ../../results/SV_effect/Arabidopsis_thaliana.TAIR10.dna.toplevel.noMtPt.fa \
    -output ${DATA_DIR}/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_pcv2_large_part0.tsv \
    -model 'kuleshov-group/PlantCAD2-Large-l48-d1536' \
    -device 'cuda:0' \
    -step-size 1 \
    -aggregation max \
    -use-masking \
    -output-raw-prob \
    -contextSize 8192 \
    -batchSize 16 &

# Run the second part on cuda:1
CUDA_VISIBLE_DEVICES=1 python /workdir/jz963/utils/plantcad/src/zero_shot_score.py \
    -input-bed ${DATA_DIR}/split_temp_01 \
    -input-fasta ../../results/SV_effect/Arabidopsis_thaliana.TAIR10.dna.toplevel.noMtPt.fa \
    -output ${DATA_DIR}/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_pcv2_large_part1.tsv \
    -model 'kuleshov-group/PlantCAD2-Large-l48-d1536' \
    -device 'cuda:0' \
    -step-size 1 \
    -aggregation max \
    -use-masking \
    -output-raw-prob \
    -contextSize 8192 \
    -batchSize 16 &

# Wait for both background processes to finish
wait

# Merge the results
cat ${DATA_DIR}/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_pcv2_large_part0.tsv ${DATA_DIR}/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_pcv2_large_part1.tsv > ${DATA_DIR}/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_pcv2_large.tsv

# Clean up temporary files
rm ${DATA_DIR}/split_temp_00 ${DATA_DIR}/split_temp_01 ${DATA_DIR}/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_pcv2_large_part0.tsv ${DATA_DIR}/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_pcv2_large_part1.tsv
