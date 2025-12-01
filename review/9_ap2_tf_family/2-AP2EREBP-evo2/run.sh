CUDA_VISIBLE_DEVICES=1 python main.py \
    --input-bed ../../../results/review/9_ap2_tf_family/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events.bed \
    --input-fasta ../../../results/SV_effect/Arabidopsis_thaliana.TAIR10.dna.toplevel.noMtPt.fa \
    --output ../../../results/review/9_ap2_tf_family/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_evo2_context-1k.tsv \
    --contextSize 1024 --batch-size 16 --tokenIdx 511

CUDA_VISIBLE_DEVICES=1 python main.py \
    --input-bed ../../../results/review/9_ap2_tf_family/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events.bed \
    --input-fasta ../../../results/SV_effect/Arabidopsis_thaliana.TAIR10.dna.toplevel.noMtPt.fa \
    --output ../../../results/review/9_ap2_tf_family/AP2EREBP_tnt-DREB2_colamp_a-chr1-5_GEM_events_evo2_context-2k.tsv \
    --contextSize 2048 --batch-size 16 --tokenIdx 1023