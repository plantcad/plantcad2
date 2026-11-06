# Unpublished draft: two average-rank summaries

Draft destination: [the existing results comment on PlantCAD2 PR #22](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065).
Do not publish until the user approves. No GitHub post or PR metadata has been changed for this draft.

## Scope

- Unweighted Avg Rank averages task ranks within each family, then averages the four family means equally: Conservation, Motifs, Core/non-core and SV impact contribute 25% each. Per-task weights are respectively 8⅓%, 3.125%, 3.125% and 25%.
- Weighted Avg Rank matches the validated PlantCAD2 leaderboard calculation: weight family means by task count (3/8/8/1, or 15%/40%/40%/5%), equivalently twenty tasks at 5% each.
- Both use the unchanged one-based percentile convention and fixed eleven-baseline reference; AF is excluded.
- Only the first figure adds Weighted Avg Rank. It has six vertical groups and four sorted horizontal panels: Conservation, SV impact, Unweighted Avg Rank and Weighted Avg Rank.
- Elsewhere in the draft, Avg Rank means the unweighted definition. Existing probe win rates, checkpoint normalized scores and native metrics keep their definitions.
- The other five figures, manual introduction, takeaways and unrelated text are unchanged. The phrase PlantCAD2 leaderboard replaces dashboard references in the draft.

`prepare_draft.py` adds the family-unweighted result to the current validated score file, preserving its task-weighted result and all native scores. It checks each displayed model two ways, rechecks all eleven baseline averages, and verifies every text edit is reversible to the freshly captured source in `live-source.json`.
`plot_benchmark_comparison.py` checks all 80 value labels, panel separation, top label boundaries and the legend. Vertical-chart value labels are rotated 90 degrees; horizontal-chart value labels remain horizontal, with compact model labels and enough numeric headroom.

## Reproduction

From the repository root:

```bash
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python .agents/artifacts/exp586-two-ranks-draft/prepare_draft.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python .agents/artifacts/exp586-two-ranks-draft/plot_benchmark_comparison.py
uv run --no-project --with markdown-it-py --with mdit-py-plugins python .agents/artifacts/exp586-two-ranks-draft/build_preview.py --output /home/exedev/repos/marin-dna-br/codex-exp586-probes/.agents/artifacts/exp586-two-ranks-preview
```

The preview embeds all six PNGs and preserves six collapsed details sections, two tables and three footnotes. Local PNG/SVG copies and a manifest accompany it. `draft.diff` shows the complete text changes for review.
