# MarinMoE checkpoint trajectories

Five evaluated checkpoints from each text-pretrained d768 run, with 216B-token and 1T-token DNA training horizons.
`post.md` is the source for the checkpoint comparison comment on PlantCAD2 PR #22.

- `figures/checkpoint-tokens.*`: paired zero-shot and frozen-probe trajectories against DNA tokens.
- `figures/checkpoint-progress.*`: identical score values against recorded W&B `run_progress`.
- `figures/checkpoint-changes.*`: changes between successive checkpoints in original score units.
- `all-category-scores.csv`: 140 scores; each line figure uses 120 in twelve panels, excluding the composite without SV.
- `original-category-scores.csv`: the 110 scores preceding motif probes, preserved unchanged.
- `motif-probe-summary.json`: eight whole-motif probe scores per unique checkpoint, controls, sample sizes and immutable result receipts.
- `motif-fitting-environment.json`: actual CoreWeave fitting versions verified from W&B, separate from the local development lockfile.
- `run-progress.csv`: actual checkpoint progress values, not estimates from token fractions.
- `*-provenance.json`: source hashes, fixed readouts, cohort definitions and plotting metadata.

Both line figures arrange paired tasks as Composite/Conservation/Motifs above Core/non-core/SV/AF, with zero-shot above linear probe inside each pair.
`add_motif_scores.py` adds verified motif probes and matching composites while asserting that every original score is unchanged.
`plot_checkpoint_pairs.py` renders the token and recorded-progress figures; `plot_checkpoint_changes.py` renders checkpoint changes.
Motif and composite probe results are included in the heatmap as well.
Zero-shot and probe cohorts can differ; the plotted levels are not a paired comparison.
