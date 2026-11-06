# Draft: frozen-reference rank summaries

This is an unpublished replacement draft for [the checkpoint results post on PlantCAD2 PR #22](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065).
No live comment or PR description was edited.
The original post, including the author's manual introduction edits, is archived in `live-post-source.json` and `live-post-source.md`.
The opening in `draft.md` is identical in wording, with CRLF normalized to LF.
Existing figures and draft files under `../exp586-checkpoint-trajectories/` remain unchanged.

## Scoring contract

`rank_scores.py` implements a fixed-reference task percentile: `100 × (lower + ties / 2) / N`.
Averages are taken first within task families and then equally across families.
All scores are higher-is-better.
Reference membership, tie handling and source task values are frozen in `frozen-references.json`.

- Avg Rank: 11 published baseline models, four equally weighted families, no AF.
- Marin Avg Rank: nine unique language-initialized MoE checkpoints, five equally weighted families including AF, with separate references for zero-shot and linear probes.
- The shared pre-DNA checkpoint appears once in the reference, despite being displayed in both trajectories.
- There is no reranking of family averages and no averaging of native scores across task families.
- Conservation AUROC, motif accuracy, core/non-core AUROC, SV average precision and AF correlation stay unchanged.
- Classification/motif probes retain their existing selected readouts and cohorts; AF uses the existing directional probe and original plotted zero-shot score.
- The old Composite without SV is not shown.
- No inference, probe fitting, sampling, smoothing choices or losses were recomputed.

`native-task-scores.json` stores the 9 × 2 × 21 task observations, checkpoint identities, source hashes and receipts.
`task-percentiles.csv` makes the task → family → overall calculation inspectable, with the shared starting checkpoint displayed in both trajectories.
`benchmark-ranks.json` records the 20-task breakdown for every model in the overview.
`all-category-scores.csv` contains 100 unchanged native family observations and 20 newly computed Marin Avg Rank values.
`prepare_scores.py` verifies exact hashes of both saved trajectory result files, the historical archive and the original family-score CSV before extraction.

## Presentation

The primary grouped bars retain four native task-family scores and add a labeled value to every bar.
The vertical axis has a marked break omitting 0.05–0.50.
Avg Rank is shown on its own percentile axis in the third sorted horizontal panel, avoiding a mixture of 0–1 native metrics and 0–100 ranks on one scale.
The loss scatter replaces Composite with Avg Rank and retains Conservation AUROC on the explicitly labeled secondary axis.
Checkpoint plots and the change heatmap replace Composite with Marin Avg Rank; the five native family panels are unchanged.
The heatmap reports rank changes in percentile points and native changes in their original units; colors encode sign only.
All five PNGs and SVGs were inspected, with programmatic overlap/boundary checks on labels and paired-card layouts.

## Reproduction and checks

Use the existing analysis environment:

```bash
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python prepare_scores.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_benchmark_comparison.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_checkpoint_pairs.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_checkpoint_changes.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python plot_loss_comparison.py --history-dir /home/exedev/repos/marin-dna-br/codex-exp586-probes/scratch/loss-figure-wandb
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python -m pytest test_rank_scores.py -q
python build_draft.py
uv run --no-project --with markdown-it-py --with mdit-py-plugins python build_draft_preview.py --output /home/exedev/repos/marin-dna-br/codex-exp586-probes/.agents/artifacts/exp586-rank-draft-preview
```

The tests independently check percentiles against SciPy, equal family weighting, reference invariance, missing-task rejection and exact preservation of all native family values.
Training/validation arrays and dense loss rugs were compared directly with the old saved figure data and are identical.
The HTML builder embeds PNGs into a self-contained, content-hashed preview and verifies five correctly placed figures, five closed dropdowns and unchanged methods/control tables.
The live comment was re-fetched after drafting to confirm no edits were lost or published.
