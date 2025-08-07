OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_downsampled_evo2_logits_new.tsv
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_downsampled.tsv \
    --batch-size 16 --tokenIdx 4095



OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_downsampled_evo2_logits_new.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_downsampled.tsv \
    --batch-size 16 --tokenIdx 4095


# for boundary CDS
## conserved
OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_1_evo2.tsv
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_1.tsv \
    --batch-size 16 --tokenIdx 4095

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_2_evo2.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_2.tsv \
    --batch-size 16 --tokenIdx 4095

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_3_evo2.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_3.tsv \
    --batch-size 16 --tokenIdx 4095

## neutral
OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_1_evo2.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_1.tsv \
    --batch-size 16 --tokenIdx 4095

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_2_evo2.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_2.tsv \
    --batch-size 16 --tokenIdx 4095

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_3_evo2.tsv
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output ${OUTPUT_DIR} \
    --input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_3.tsv \
    --batch-size 16 --tokenIdx 4095