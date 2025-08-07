OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_downsampled_gpn_logits.tsv
CUDA_VISIBLE_DEVICES=0 python main.py \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_downsampled.tsv \
    -outLogit ${OUTPUT_DIR} \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'



OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_downsampled_gpn_logits.tsv
CUDA_VISIBLE_DEVICES=0 python main.py \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_downsampled.tsv \
    -outLogit ${OUTPUT_DIR} \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'

# for boundary CDS
## conserved
OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_1_gpn.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_1.tsv \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_2_gpn.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_2.tsv \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_3_gpn.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/conserved_8k_TIS_3.tsv \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'

## neutral
OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_1_gpn.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_1.tsv \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_2_gpn.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_2.tsv \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'

OUTPUT_DIR=../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_3_gpn.tsv
python main.py \
    -outLogit ${OUTPUT_DIR} \
    -input ../../../results/Poaceae_CDS_PhyloP/8k/neutral_8k_TIS_3.tsv \
    -model 'songlab/gpn-brassicales' \
    -device 'cuda:0'