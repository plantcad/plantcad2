export DATA_DIR=${cad2}/results/evolutionary_constraint/


# the output is the replace .tsv with _pcv1_logits.tsv
OUTPUT_DIR=${DATA_DIR}valid_pdllm_logits.tsv

CUDA_VISIBLE_DEVICES=0 python main.py \
    -input 'valid' \
    -outLogit ${OUTPUT_DIR} \
    -model 'zhangtaolab/plant-dnamamba-singlebase' \
    -device 'cuda:0'