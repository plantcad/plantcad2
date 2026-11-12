# Draft: equal task weighting for Avg Rank

This is an unpublished revision of [the exp586 results comment](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065).
`live-source.json` is the current author-edited comment captured when drafting; the historical draft is not used as source text.
No issue, comment or PR description was edited.

## Scope

- Avg Rank averages the twenty existing task percentiles equally: 5% per task.
- Family weights are therefore Conservation 15%, Motifs 40%, Core/non-core 40% and SV impact 5%.
- The fixed eleven-baseline reference and existing midpoint-percentile convention remain unchanged.
- These are the dashboard's task weights, while the percentile convention still differs by 4.545 points for baseline members; a short footnote states that distinction and links both implementations.
- All 220 baseline values match the dashboard CSV at `f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29` exactly.
- Native task-family averages, probes, win rates, checkpoint normalized scores and losses are unchanged.
- The first figure is rendered again with the new Avg Rank bars and ordering; the other five figures are reused unchanged.
- The author's introduction, wording and takeaways are preserved outside the explicit source/weighting edits.

The main-text addition names the baseline source and per-task/family weights.
Two brief footnotes point to the raw scores, task definitions, dashboard aggregation and fixed-reference percentile implementation.
The old Avg Rank definition and first image alt text are updated to agree with the figure.

## Reproduction and checks

From the repository root, with `live-source.json` already captured:

```bash
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python .agents/artifacts/exp586-dashboard-weighting-draft/prepare_draft.py
uv run --no-project --with markdown-it-py --with mdit-py-plugins python .agents/artifacts/exp586-dashboard-weighting-draft/build_preview.py --output /home/exedev/repos/marin-dna-br/codex-exp586-probes/.agents/artifacts/exp586-dashboard-weighting-preview
```

The preparation script recomputes all per-task percentiles using the existing scoring code and checks that they match the saved task ranks exactly.
Only their averaging weights change.
It checks every native family mean and reverses all allowed text edits to recover the captured live body exactly, after LF normalization.
The existing bar renderer checks all value-label overlaps, boundary placement and panel separation.
The resulting PNG was visually inspected.
The self-contained HTML preview embeds six figures and checks six closed details sections, two tables and three footnotes.
