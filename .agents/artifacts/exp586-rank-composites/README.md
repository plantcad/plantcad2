# Baseline ranks and Marin model summaries

These artifacts reproduce [the updated checkpoint results post on PlantCAD2 PR #22](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065).
The reviewed draft was published to that existing comment; no new comment or PR-description edit was made.
`post.md` contains the exact published text with commit-pinned figure URLs; `publication.json` records remote image hashes and verification of the re-fetched comment and rendered HTML.
The original post, including the author's manual introduction edits, is archived in `live-post-source.json` and `live-post-source.md`.
The opening in `draft.md` is identical in wording, with CRLF normalized to LF.
The author subsequently edited the live comment; the historical draft is not a replacement for that newer text.
Figure-only updates must fetch the current comment and replace only the intended image URL, preserving all other content exactly.
Existing figures and draft files under `../exp586-checkpoint-trajectories/` remain unchanged.

## Scoring contract

`rank_scores.py` implements Avg Rank using a fixed-reference task percentile: `100 × (lower + ties / 2) / N`.
Averages are taken first within task families and then equally across families.
All scores are higher-is-better.
Reference membership, tie handling and source task values are frozen in `frozen-references.json`.

- Avg Rank: 11 published baseline models, four equally weighted families, no AF.
- Family-balanced win rate: compare only the two final probe checkpoints per task, award half credit for ties, average within each family, then weight the five families equally.
- Pooled win rate: use the same per-task wins, but weight the 21 tasks equally without balancing families.
- Avg normalized score: min–max normalize each saved native family mean over all checkpoints from both runs, separately for zero-shot and probes, then equally average the five normalized families.
- The shared pre-DNA checkpoint appears once in the normalization population, despite being displayed in both trajectories.
- Normalization bounds describe the nine checkpoints shown and can change if that population changes; they do not use checkpoint percentiles.
- There is no reranking of family averages and no averaging of native scores across task families.
- Conservation AUROC, motif accuracy, core/non-core AUROC, SV average precision and AF correlation stay unchanged.
- Classification/motif probes retain their existing selected readouts and cohorts; AF uses the existing directional probe and original plotted zero-shot score.
- The old Composite without SV is not shown.
- No inference, probe fitting, sampling, smoothing choices or losses were recomputed.

`native-task-scores.json` stores the 9 × 2 × 21 task observations, checkpoint identities, source hashes and receipts.
`marin_summaries.py` and `marin-summaries.json` record the family min/max bounds, each normalized value and each final-model task win.
The old checkpoint-rank snapshots in `all-category-scores.csv`, `task-percentiles.csv`, `trajectory-ranks.json` and the legacy section of `frozen-references.json` are retained only for audit; current Marin plots do not use those rank values.
`benchmark-ranks.json` records the 20-task breakdown for every model in the overview.
`normalized-category-scores.csv` contains the same 100 native family observations and 20 new average normalized scores.
The original `all-category-scores.csv` supplies native family means; its old rank rows are ignored.
`prepare_scores.py` reproduces the original source snapshots and verifies exact hashes of both saved trajectory result files, the historical archive and the original family-score CSV before extraction.

## Presentation

The primary grouped bars show all five readouts together on a single `Metric score (×100) or percentile rank` axis, with a labeled value on every bar.
For display only, the four task-family scores are multiplied by 100; Avg Rank already uses a 0–100 percentile scale.
The shared vertical axis has a marked break omitting 5–45.
Underlying scores, definitions and family aggregation are unchanged; the horizontal rankings and other figures retain their native units.
Avg Rank is also retained in the third sorted horizontal panel.
The second figure is titled `Linear probe PlantCAD2 benchmark`.
Its shared model key uses large regular-weight labels and solid color swatches in a white bordered box between the score and loss rows.
Both dot-and-line figures use the same key style and wording: `MoE training horizon`, `216B tokens` and `1T tokens`.
Its first row has three facets: family-average scores, task win-rate bars and a task-winner map.
The first compares the final 216B and 1T MoE models on native family-average scores for Conservation, Motifs, Core/non-core, SV and AF.
The second has paired bars for family-balanced and pooled win rates, explicitly labeled with equal family and equal task weighting.
The third shows one colored tile per task: gold for a higher 216B score, plum for a higher 1T score, gray for an exact tie.
Motifs and core/non-core share explicit Maize/Tomato row labels and start codon, stop codon, splice donor and splice acceptor columns.
Conservation separates the sequence species (sorghum or Pharus) from the conservation clade (Andropogoneae or Poaceae); the SV and AF tiles identify Arabidopsis and maize, respectively.
The two eight-task grids take more room than the three-task conservation strip and the single SV/AF tiles.
SV and AF have separate left-aligned rows, with the same tile position and single-line family labels.
The map is solely a detailed task-level breakdown; both overall win rates appear in the middle facet.
`loss-figure-data.json` records pooled win points, their denominator of 21 tasks and both percentage summaries.
The grouped score bars display native metrics ×100 with two-decimal annotations; tile colors indicate winners without encoding the margin.
This row keeps the full score range because AF correlations lie below the first figure's 45-point axis break.
A collapsed scorecard displays all 21 per-task scores in native units, grouped by family, with both model scores and Δ = 1T − 216B.
Only the difference column is colored by winning model; five decimal places avoid false visible ties.
`plot_probe_scorecard.py`, `probe-scorecard-data.json` and `probe-scorecard.csv` preserve the task identities, source scores, differences and checks against the family means and exact win counts.
The map reuses the scorecard's validated native rows and records all 21 task/species/winner assignments in `loss-figure-data.json` and its provenance.
The training/validation loss curves and final-loss scatter occupy the second row.
The scatter explicitly labels family-balanced probe win rate on the left axis and native probe Conservation AUROC on the right; matched losses, W&B histories and dense rugs remain unchanged.
Checkpoint plots and the change heatmap use Avg normalized score on a 0–1 scale; the five native family panels are unchanged.
Both dot-and-line figures use white cards and axes, retaining thin gray borders around the zero-shot/probe pairs.
The heatmap reports normalized-score changes in 0–1 units and native changes in their original units; colors encode sign only.
All six PNGs and SVGs were inspected, with programmatic overlap/boundary checks on labels and paired-card layouts.

## Reproduction and checks

Use the existing analysis environment:

```bash
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python prepare_scores.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python marin_summaries.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_benchmark_comparison.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_checkpoint_pairs.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_checkpoint_changes.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_probe_scorecard.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_loss_comparison.py --history-dir /home/exedev/repos/marin-dna-br/codex-exp586-probes/scratch/loss-figure-wandb
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python -m pytest test_rank_scores.py test_marin_summaries.py -q
python build_draft.py
uv run --no-project --with markdown-it-py --with mdit-py-plugins python build_draft_preview.py --output /home/exedev/repos/marin-dna-br/codex-exp586-probes/.agents/artifacts/exp586-rank-draft-preview
```

The tests check published-reference percentiles against SciPy, min–max normalization of family means, equal family weighting, half-credit task ties, final-only Win Rate comparisons, rejection of missing/constant families and preservation of all native family values.
Training/validation arrays and dense loss rugs were compared directly with the old saved figure data and are identical.
The HTML builder embeds PNGs into a self-contained, content-hashed preview and verifies six correctly placed figures, six closed dropdowns and unchanged methods/control tables.
The live comment was checked before publication to preserve the author's edits, then re-fetched to verify the exact replacement text, six rendered images, six closed dropdowns and two tables.
