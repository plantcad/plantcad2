OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_downsampled_pdllm_logits.tsv
CUDA_VISIBLE_DEVICES=0 python main.py \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_downsampled.tsv \
    -outLogit ${OUTPUT_DIR} \
    -model 'zhangtaolab/plant-dnamamba-singlebase' \
    -device 'cuda:1'


OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_downsampled_pdllm_logits.tsv
CUDA_VISIBLE_DEVICES=0 python main.py \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_downsampled.tsv \
    -outLogit ${OUTPUT_DIR} \
    -model 'zhangtaolab/plant-dnamamba-singlebase' \
    -device 'cuda:1'



OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_1_pdllm.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_1.tsv \
    -model 'zhangtaolab/plant-dnamamba-singlebase'  \
    -device 'cuda:1'

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_2_pdllm.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_2.tsv \
    -model 'zhangtaolab/plant-dnamamba-singlebase'  \
    -device 'cuda:1'

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_3_pdllm.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_3.tsv \
    -model 'zhangtaolab/plant-dnamamba-singlebase'  \
    -device 'cuda:1'

## neutral
OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_1_pdllm.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_1.tsv \
    -model 'zhangtaolab/plant-dnamamba-singlebase'  \
    -device 'cuda:1'

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_2_pdllm.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_2.tsv \
    -model 'zhangtaolab/plant-dnamamba-singlebase'  \
    -device 'cuda:1'

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_3_pdllm.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_3.tsv \
    -model 'zhangtaolab/plant-dnamamba-singlebase'  \
    -device 'cuda:1'