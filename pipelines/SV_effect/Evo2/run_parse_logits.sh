for shift in 0 5 10 15 20
do
    python 4_parseLogits.py \
        --input ../../../results/SV_effect/inputs/Ath_Simulated_DEL_Len_1-50.tsv \
        --ref_logits ../../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_Ref_Evo2.npz \
        --mut_logits ../../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_Mut_Evo2.npz \
        --output ../../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_Evo2_shift_${shift}.tsv \
        --shift ${shift}
done
