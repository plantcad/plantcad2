# MarinMoE checkpoint trajectories

Five evaluated checkpoints from each text-pretrained d768 run, with 216B-token and 1T-token DNA training horizons.
`post.md` is the source for the checkpoint comparison comment on PlantCAD2 PR #22.
`draft.md` adds the requested eight-model overview first; it has not replaced the live comment.

- `figures/checkpoint-tokens.*`: paired zero-shot and frozen-probe trajectories against DNA tokens.
- `figures/checkpoint-progress.*`: identical score values against recorded W&B `run_progress`.
- `figures/checkpoint-changes.*`: changes between successive checkpoints in original score units.
- `figures/benchmark-comparison.*`: full-benchmark zero-shot scores for final 216B/1T MoEs, dense MarinDNA at 562.022055936B tokens, and five archived baseline models, followed by independent Conservation and Composite rankings.
- `figures/checkpoint-changes-primary.*`: draft heatmap with only the primary four-category Composite.
- `benchmark-comparison-data.json` and `benchmark-comparison-provenance.json`: all forty overview values, original colors, source hashes and the exact dense token-count check.
- `all-category-scores.csv`: 140 scores; each line figure uses 120 in twelve panels, excluding the composite without SV.
- `original-category-scores.csv`: the 110 scores preceding motif probes, preserved unchanged.
- `motif-probe-summary.json`: eight whole-motif probe scores per unique checkpoint, controls, sample sizes and immutable result receipts.
- `motif-fitting-environment.json`: actual CoreWeave fitting versions verified from W&B, separate from the local development lockfile.
- `run-progress.csv`: actual checkpoint progress values, not estimates from token fractions.
- `*-provenance.json`: source hashes, fixed readouts, cohort definitions and plotting metadata.

Both line figures arrange paired tasks as Composite/Conservation/Motifs above Core/non-core/SV/AF, with zero-shot above linear probe inside each pair.
`add_motif_scores.py` adds verified motif probes and matching composites while asserting that every original score is unchanged.
`plot_checkpoint_pairs.py` renders the token and recorded-progress figures; `plot_checkpoint_changes.py` renders draft checkpoint changes with only the primary Composite.
`plot_benchmark_comparison.py` consumes the saved short/long trajectory `results.json` files via `--short-results` and `--long-results`; it runs no inference or fitting.
`uv run --no-project --with markdown-it-py --with mdit-py-plugins python build_draft_preview.py --output <directory>` copies the four draft figures and creates a self-contained HTML preview, including the GitHub-style reference footnote.
Motif and composite probe results are included in the heatmap as well.
Zero-shot and probe cohorts can differ; the plotted levels are not a paired comparison.
