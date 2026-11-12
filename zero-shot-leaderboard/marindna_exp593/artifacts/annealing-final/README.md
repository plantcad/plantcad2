# Final annealing task scores

[scores.csv](scores.csv) contains individual-task zero-shot and frozen linear-probe scores for completed final evaluations. It is updated as the remaining final models finish. Missing models are omitted rather than represented by empty or zero scores. No training-monitor values, task-family averages, shuffled-label controls or majority-predictor baselines are mixed into this file.

Each row identifies a model, task, readout, strand strategy and metric. Scores retain their native scale, without percentage scaling or rounding. `training_input_tokens` includes the source DNA-training stage and the additional annealing inputs, including structural FIM text. `context_length` is the 8192-token budget; prompt overhead and withholding the target reduce visible DNA context.

- `ar_historical_teacher_forced` preserves the original per-task strand selection. Other zero-shot readouts explicitly identify forward, RC or probability fusion.
- Exhaustive core readouts distinguish `_joint` and `_marginal`. Motif `accuracy` is whole-motif exact-match accuracy. Single-base conservation has identical joint and marginal definitions, so those are not duplicated.
- AF suffix readouts distinguish full-vocabulary and lowercase-four normalization. FIM AF uses single-site odds, with separately identified BF16 and FP32 head projections; it does not use a full-vocabulary denominator. `forward_rc_score_average` averages strand-specific log-odds, whereas `forward_rc_fusion` combines aligned base probabilities.
- Probe readouts identify boundary, SV REF/ALT-delta, variant-site or whole-window features. Strand strategies distinguish feature concatenation from feature averaging. Probe scores use held-out memberships; zero-shot scores use the original full benchmark memberships.
- `n_examples` is the metric's scored row count. Motif zero-shot counts exclude ambiguous target motifs using the same target-validity rule as the evaluator; probe counts are the actual held-out counts. AUROC/AP retain the evaluator's original unknown-target treatment. AF rows contain the overall result, not consequence-subgroup results.

The immutable, checksum-bound sources are recorded in [scores.sources.json](scores.sources.json). The exporter reads complete scientific-result snapshots and the original motif memberships; it performs no inference or probe fitting. Reproduce from this project directory with CoreWeave credentials:

```bash
uv run python -m exp593_anneal_evals.csv_scores \
  --manifest artifacts/annealing-final/scores.sources.json \
  --cache /tmp/exp593-final-score-inputs \
  --output artifacts/annealing-final/scores.csv \
  --refresh-credentials
```
