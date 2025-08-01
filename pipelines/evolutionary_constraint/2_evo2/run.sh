OUTPUT_DIR=../../../results/evolutionary_constraint/valid_8192_evo2_logits_new.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/evolutionary_constraint/valid_8192.tsv \
    --batch-size 16 --tokenIdx 4095