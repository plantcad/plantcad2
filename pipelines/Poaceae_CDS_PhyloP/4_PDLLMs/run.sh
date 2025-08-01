OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_downsampled_pdllm_logits.tsv
CUDA_VISIBLE_DEVICES=0 python main.py \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_downsampled.tsv \
    -outLogit ${OUTPUT_DIR} \
    -model 'zhangtaolab/plant-dnamamba-singlebase' \
    -device 'cuda:0'


OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_downsampled_pdllm_logits.tsv
CUDA_VISIBLE_DEVICES=0 python main.py \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_downsampled.tsv \
    -outLogit ${OUTPUT_DIR} \
    -model 'zhangtaolab/plant-dnamamba-singlebase' \
    -device 'cuda:0'