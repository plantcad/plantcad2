MODEL="/local/workdir/jz963/models/rebuttal-exp-1-ep16-ba240000"
OUTPUT_DIR="/local/workdir/jz963/utils/plantcad2/results/review/8_tiny_model_eval"
mkdir -p $OUTPUT_DIR


# GPU 0 tasks
(
    export CUDA_VISIBLE_DEVICES=0
    echo "Starting GPU 0 tasks..."
    for task in conservation_within_poaceae_tis conservation_within_andropogoneae conservation_within_poaceae_non_tis
    do
        python zero-shot-eval.py evo_cons \
        --repo_id plantcad/PlantCAD2_zero_shot_tasks \
        --task $task \
        --split test \
        --model $MODEL \
        --device cuda:0 \
        --token_idx 4095 \
        --batch_size 128 \
        --save_logits $OUTPUT_DIR/$task_logits.tsv \
        --metrics_json $OUTPUT_DIR/$task_metrics.json
    done

    for task in tis_recovery tts_recovery
    do
        for split in test_maize test_tomato
        do
            python zero-shot-eval.py motif_acc \
            --repo_id plantcad/PlantCAD2_zero_shot_tasks \
            --task $task \
            --split $split \
            --model $MODEL \
            --device cuda:0 \
            --mask_idx 4094,4095,4096 \
            --motif_len 3 \
            --batch_size 128 \
            --save_logits $OUTPUT_DIR/${task}_${split}_logits.tsv \
            --metrics_json $OUTPUT_DIR/${task}_${split}_metrics.json
        done
    done

    for task in acceptor_recovery donor_recovery
    do
        for split in test_maize test_tomato
        do
            python zero-shot-eval.py motif_acc \
            --repo_id plantcad/PlantCAD2_zero_shot_tasks \
            --task $task \
            --split $split \
            --model $MODEL \
            --device cuda:0 \
            --mask_idx 4095,4096 \
            --motif_len 2 \
            --batch_size 128 \
            --save_logits $OUTPUT_DIR/${task}_${split}_logits.tsv \
            --metrics_json $OUTPUT_DIR/${task}_${split}_metrics.json
        done
    done
    echo "GPU 0 tasks finished."
) &

# GPU 1 tasks
(
    export CUDA_VISIBLE_DEVICES=1
    echo "Starting GPU 1 tasks..."
    for task in tis_core_noncore_classification tts_core_noncore_classification
    do
        for split in test_maize test_tomato
        do
            python zero-shot-eval.py core_noncore \
            --repo_id plantcad/PlantCAD2_zero_shot_tasks \
            --task $task \
            --split $split \
            --model $MODEL \
            --device cuda:0 \
            --mask_idx 4094,4095,4096 \
            --motif_len 3 \
            --batch_size 128 \
            --save_logits $OUTPUT_DIR/${task}_${split}_logits.tsv \
            --metrics_json $OUTPUT_DIR/${task}_${split}_metrics.json
        done
    done

    for task in acceptor_core_noncore_classification donor_core_noncore_classification
    do
        for split in test_maize test_tomato
        do
            python zero-shot-eval.py core_noncore \
            --repo_id plantcad/PlantCAD2_zero_shot_tasks \
            --task $task \
            --split $split \
            --model $MODEL \
            --device cuda:0 \
            --mask_idx 4095,4096 \
            --motif_len 2 \
            --batch_size 128 \
            --save_logits $OUTPUT_DIR/${task}_${split}_logits.tsv \
            --metrics_json $OUTPUT_DIR/${task}_${split}_metrics.json
        done
    done

    python zero-shot-eval.py sv_effect \
      --repo_id plantcad/PlantCAD2_zero_shot_tasks \
      --task structural_variant_effect_prediction \
      --split test \
      --model $MODEL \
      --device cuda:0 \
      --batch_size 64 \
      --flanking 5 \
      --output $OUTPUT_DIR/sv_effect_scored.tsv
    echo "GPU 1 tasks finished."
) &

wait
echo "All tasks finished."