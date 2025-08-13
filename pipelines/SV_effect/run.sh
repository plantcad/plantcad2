for shift in 0 5 10 15 20
do
    python 4_parseLogits.py \
        --input ../../results/SV_effect/inputs/Ath_Simulated_DEL_Len_1-50.tsv \
        --ref_logits ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-l24-d0768-2nd-run_Ref.npz \
        --mut_logits ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-l24-d0768-2nd-run_Mut.npz \
        --output ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-small-2nd_shift_${shift}.tsv \
        --shift ${shift} &
done

for shift in 0 5 10 15 20
do
    python 4_parseLogits.py \
        --input ../../results/SV_effect/inputs/Ath_Simulated_DEL_Len_1-50.tsv \
        --ref_logits ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-l48-d1024-2nd-run_Ref.npz \
        --mut_logits ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-l48-d1024-2nd-run_Mut.npz \
        --output ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-medium-2nd_shift_${shift}.tsv \
        --shift ${shift} &
done


for shift in 0 5 10 15 20
do
    python 4_parseLogits.py \
        --input ../../results/SV_effect/inputs/Ath_Simulated_DEL_Len_1-50.tsv \
        --ref_logits ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-l48-d1536-2nd-run_Ref.npz \
        --mut_logits ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-l48-d1536-2nd-run_Mut.npz \
        --output ../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_pcv2-large-2nd_shift_${shift}.tsv \
        --shift ${shift} &
done
