# MarinMoE checkpoint trajectories

Five evaluated checkpoints from each text-pretrained d768 run, with 216B-token and 1T-token DNA training horizons.
`post.md` is the source for the checkpoint comparison comment on PlantCAD2 PR #22.

- `figures/checkpoint-tokens.*`: paired zero-shot and frozen-probe trajectories against DNA tokens.
- `figures/checkpoint-progress.*`: identical score values against recorded W&B `run_progress`.
- `figures/checkpoint-changes.*`: changes between successive checkpoints in original score units.
- `all-category-scores.csv`: all 110 source scores; the line figures use 100, excluding the composite without SV.
- `run-progress.csv`: actual checkpoint progress values, not estimates from token fractions.
- `*-provenance.json`: source hashes, fixed readouts, cohort definitions and plotting metadata.

The PNG and SVG files preserve the approved figures.
Zero-shot and probe cohorts can differ; the plotted levels are not a paired comparison.
