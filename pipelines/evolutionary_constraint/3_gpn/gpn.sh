export DATA_DIR=${cad2}/results/evolutionary_constraint/


# the output is the replace .tsv with _pcv1_logits.tsv
OUTPUT_DIR=${DATA_DIR}valid_gpn_logits.tsv

CUDA_VISIBLE_DEVICES=0 python main.py \
    -input 'valid' \
    -outLogit ${OUTPUT_DIR} \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'