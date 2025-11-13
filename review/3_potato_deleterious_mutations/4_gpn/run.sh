for i in {01..12}
do
python main.py evo_cons --input_tsv ../../../results/review/3_potato_deleterious_mutations/processed_v2/chr${i}_downsampled.tsv --save_logits ../../../results/review/3_potato_deleterious_mutations/logits_v2/${i}_downsampled_gpn.tsv \
  --model songlab/gpn-brassicales \
  --device cuda:1 \
  --token_idx 255 \
  --batch_size 64 --crop_length 512
done
