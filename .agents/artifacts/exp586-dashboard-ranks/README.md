# Dashboard-compatible Avg Rank for the exp586 results post

This revision updates only the Avg Rank results and definition in [the existing results comment](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065).
The current author-edited comment was fetched into `live-source.json`; the earlier draft was not used as publication source.
The update is published. `post.md` is the published text with its immutable figure URL; `publication.json` and `published-readback.json` record verification of the exact comment body and rendered figure/footnotes.

## Calculation and reference

- Each of the 20 zero-shot tasks contributes 5%, matching the unfiltered dashboard: Conservation 15%, Motifs 40%, Core/non-core 40%, SV impact 5%. AF is excluded.
- Per-task baseline percentiles use average one-based ranks divided by 11, including exact ties, as in pandas `rank(pct=True)`.
- The reference remains the dashboard's 11 baseline models at revision `f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29`. Marin results are scored against that fixed reference with the SciPy `percentileofscore(kind="rank")` extension; they do not change baseline ranks. Unmatched external scores use their empirical percentile, so this is not a blanket 4.545-point addition to Marin scores.
- `dashboard-source/` contains the pinned upstream CSVs and dashboard source snapshot, with source links in the prepared post. The Python source is stored as `.py.txt` to distinguish archived third-party input from executable project code.
- The audit executes the dashboard's actual `rank_within_task` function in isolation, then its equal-task mean, and compares all 220 task ranks and all 11 model averages. Only floating-point rounding at approximately 1e-14 differs.
- Native task-family scores, probe win rates, checkpoint normalized scores, training losses and the other five figures remain unchanged.
- The first figure's rank bars and sorted rank panel are regenerated. Its upper limit gains room for the higher percentile labels; automated overlap and boundary checks remain enforced.

## Reproduction

From the repository root:

```bash
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python .agents/artifacts/exp586-dashboard-ranks/prepare_post.py
/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/.venv/bin/python -m pytest .agents/artifacts/exp586-dashboard-ranks/test_dashboard_ranks.py -q
```

`baseline-validation.csv` contains all 11 reproduced dashboard averages; `validation.json` records per-task checks. The figure is saved as PNG and SVG.
`text-edits.json` records the small reversible patch applied to the live comment. Before publication, fetch the live comment again and verify it still matches the source; if it has changed, rebase these edits onto the fresh body instead of overwriting author changes.

`prepared.md` uses a local figure path for review; the published body must use the immutable raw GitHub URL of the committed figure. Publication receipts and a fresh post-publication readback are saved separately.
