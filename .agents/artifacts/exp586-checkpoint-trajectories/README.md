# MarinMoE checkpoint trajectories

Five evaluated checkpoints from each text-pretrained d768 run, with 216B-token and 1T-token DNA training horizons.
`post.md` is the source for the checkpoint comparison comment on PlantCAD2 PR #22.
`draft.md` adds the requested eight-model overview first; it has not replaced the live comment.

- `figures/checkpoint-tokens.*`: paired zero-shot and frozen-probe trajectories against DNA tokens.
- `figures/checkpoint-progress.*`: identical score values against recorded W&B `run_progress`.
- `figures/checkpoint-changes.*`: changes between successive checkpoints in original score units.
- `figures/benchmark-comparison.*`: full-benchmark zero-shot scores for final 216B/1T MoEs, dense MarinDNA at 562.022055936B tokens, and five archived baseline models, followed by independent Conservation, SV impact and Composite rankings.
  The horizontal rankings use marked axis breaks and score ranges of 0.62–0.82, 0.72–0.89 and 0.67–0.78, respectively, while retaining a short zero-origin segment.
- `figures/checkpoint-changes-primary.*`: draft heatmap with only the primary four-category Composite.
- `figures/loss-comparison.*`: raw and exponentially smoothed W&B training loss, unsmoothed W&B validation loss, and matched-sample final validation loss versus Composite/Conservation for the 216B and 1T text-pretrained runs.
  Loss curves start at 1B DNA input tokens and use logarithmic token axes; training smoothing has a 1B-token half-life, initialized at the first included observation.
  Every unique logged training value is rendered, retaining the latest W&B record at replayed token coordinates; null metric rows are absent observations, not zero loss.
  The scatter contains four points, two descriptive connecting segments and three dense-checkpoint rugs, with identical numeric limits on both score axes and no fitted trend or uncertainty estimate.
  Scatter and rug losses come from the same 65,536-window sample in PR #22 comment 6604041302; they intentionally differ from the 512-window W&B curve endpoints.
  Orange rugs reuse dense losses at 216B, 389B and 562B DNA tokens, with opacity increasing linearly from 0.35 to 0.95 across those token totals; parameter counts are identical.
- `loss-figure-data.json` and `loss-figure-provenance.json`: complete plotted validation values, compact exact-smoothed training samples, final scatter values, source hashes and history counts.
  Full source history archives remain outside Git and are referenced by immutable CoreWeave storage receipts in `loss-history-receipts.json`.
- `benchmark-comparison-data.json` and `benchmark-comparison-provenance.json`: all forty overview values, model colors, source hashes and the exact dense token-count check.
- `figure-palette.json`: shared model colors across the overview, loss comparison and both checkpoint line figures: dense MarinDNA orange, 216B MoE saturated deep gold, 1T MoE cherry red, Evo 2 emerald green.
  The warm Marin family is separated from the original PlantCAD blues and Evo green; the heatmap retains its independent increase/decrease colors.
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
`fetch_loss_history.py --model {216b,1t} --output <history-dir>` reads complete, unsampled W&B histories using the configured account; it makes no changes to the runs.
The 216B archive uses W&B SDK 0.30.0; the 1T archive uses `--reader paged`, based on the [0.22.3 paged API](https://github.com/wandb/wandb/blob/v0.22.3/wandb/apis/public/history.py), after 0.30.0 stalled while initializing its bulk export.
That fallback runs in an isolated `uv run --no-project --with wandb==0.22.3` environment without changing training dependencies.
It requests up to 10,000 rows for disjoint windows containing at most 5,000 W&B steps, avoiding the older scanner's overlapping inclusive page bounds and implicit downsampling at the row limit.
The renderer checks gap-free logged W&B training steps before normalizing replayed token coordinates; it records training updates lacking a distinct loss observation rather than fabricating them.
`plot_loss_comparison.py --history-dir <history-dir>` validates those archives against their metadata and renders the loss figure without training or evaluation.
`uv run --no-project --with markdown-it-py --with mdit-py-plugins python build_draft_preview.py --output <directory>` copies the five draft figures and creates a self-contained HTML preview, including the GitHub-style reference footnote.
The printed preview filename includes a content hash so updated drafts open at a fresh path; `preview.html` remains an alias for the latest draft.
Motif and composite probe results are included in the heatmap as well.
Zero-shot and probe cohorts can differ; the plotted levels are not a paired comparison.
