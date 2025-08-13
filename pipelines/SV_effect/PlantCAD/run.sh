#!/bin/bash
mainDIC=../../../results/SV_effect/

python main.py \
    --input ${mainDIC}/inputs/Ath_Simulated_DEL_Len_1-50.tsv \
    --model 'kuleshov-group/PlantCaduceus_l32' \
    --batch_size 256 \
    --output ${mainDIC}/outputs/Ath_Simulated_DEL_Len_1-50_pcv1_Ref.npz \
    --device 'cuda:1' \
    --label 'RefSeq' \
    --extract_unmasked_logits

python main.py \
    --input ${mainDIC}/inputs/Ath_Simulated_DEL_Len_1-50.tsv \
    --model 'kuleshov-group/PlantCaduceus_l32' \
    --batch_size 256 \
    --output ${mainDIC}/outputs/Ath_Simulated_DEL_Len_1-50_pcv1_Mut.npz \
    --device 'cuda:1' \
    --label 'MutSeq' \
    --extract_unmasked_logits


for shift in 0 5 10 15 20
do
    python 4_parseLogits.py \
        --input ${mainDIC}/inputs/Ath_Simulated_DEL_Len_1-50.tsv \
        --ref_logits ${mainDIC}/outputs/Ath_Simulated_DEL_Len_1-50_pcv1_Ref.npz \
        --mut_logits ${mainDIC}/outputs/Ath_Simulated_DEL_Len_1-50_pcv1_Mut.npz \
        --output ${mainDIC}/outputs/Ath_Simulated_DEL_Len_1-50_pcv1_shift_${shift}.tsv \
        --shift ${shift}
done