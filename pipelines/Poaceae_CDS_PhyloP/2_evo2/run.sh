OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8192/neutral_8k_downsampled_evo2_logits.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8192/neutral_8k_downsampled.tsv \
    --batch-size 16 --tokenIdx 4095



OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8192/conserved_8k_downsampled_evo2_logits.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8192/conserved_8k_downsampled.tsv \
    --batch-size 16 --tokenIdx 4095