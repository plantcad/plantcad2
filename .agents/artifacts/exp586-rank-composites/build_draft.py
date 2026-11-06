"""Preserve the author's live wording and replace only aggregate readouts."""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    live = json.loads((HERE / "live-post-source.json").read_text())
    source = live["body"].replace("\r\n", "\n")
    opening = source.split('<p align="center">', 1)[0]
    body = re.sub(r'https://raw\.githubusercontent\.com/plantcad/plantcad2/[^"\s]+/figures/([a-z-]+\.png)', r'figures/\1', source)
    original_images = re.findall(r'<img src="([^"]+)" alt="([^"]+)" />', body)
    alts = {
        "benchmark-comparison": "Zero-shot comparison of eight models. Top: Conservation, Motifs, Core/non-core, SV impact and Avg Rank are grouped together on one Metric score (×100) or percentile rank axis with a marked break and a value on every bar. Task-family AUROC, accuracy and average precision are multiplied by 100 for display; Avg Rank is already a percentile. Bottom: sorted Conservation AUROC, SV average precision and Avg Rank retain their existing native scales and marked horizontal breaks. Avg Rank averages per-task percentiles within four equally weighted families against 11 fixed published baseline models.",
        "loss-comparison": "Linear probe PlantCAD2 benchmark. Top: paired bars for only the final 216B and 1T MoE models, comparing Conservation, Motifs, Core/non-core, SV impact, Allele frequency and Win Rate. Native probe metrics are multiplied by 100 for display; Win Rate is the family-balanced percentage of task wins against the other final model, counting ties as half. Bottom: unchanged W&B training and validation loss curves versus DNA tokens, followed by matched final loss versus probe Win Rate (left, solid circles) and probe Conservation AUROC (right, dashed diamonds). Dense MarinDNA 1B rugs retain their matched losses, directly labeled with training tokens.",
        "checkpoint-tokens": "Six paired zero-shot and linear-probe metrics across DNA training. Top: Avg normalized score, Conservation, Motifs; bottom: Core/non-core, SV impact, Allele frequency. Avg normalized score first min–max scales each task family's native mean over all checkpoints from both runs, separately for each readout, then averages the five normalized families including AF. Other panels retain native scores.",
        "checkpoint-progress": "The same six paired metrics versus actual W&B run progress. Avg normalized score equally averages five min–max normalized task-family means, including AF; individual families retain native units.",
        "checkpoint-changes-primary": "Changes between successive checkpoints for zero-shot and linear probes. Family changes retain native units; the final row shows changes in Avg normalized score on a 0–1 scale. Blue means increase, orange decrease, gray no change; color encodes sign only.",
    }
    assert len(original_images) == len(alts) == 5
    for image, old_alt in original_images:
        name = Path(image).stem
        body = body.replace(f'alt="{old_alt}"', f'alt="{alts[name]}"')
    body = body.replace(
        f'  <img src="figures/benchmark-comparison.png" alt="{alts["benchmark-comparison"]}" />\n</p>',
        f'  <img src="figures/benchmark-comparison.png" alt="{alts["benchmark-comparison"]}" />\n</p>\n\n_Top: AUROC, accuracy and average precision ×100; Avg Rank is a percentile. Higher is better; rank definitions are below._')
    body = body.replace(
        'Curves use W&B losses; the scatter uses the [matched 65,536-window sample](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6604041302), with darker coral rugs for later dense checkpoints.',
        'Top: native probe metrics ×100 and family-balanced Win Rate against the other final model. Below: W&B loss curves and [matched final loss](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6604041302) versus probe scores; darker coral rugs mark later dense checkpoints.')
    body = body.replace(
        f'  <img src="figures/checkpoint-tokens.png" alt="{alts["checkpoint-tokens"]}" />\n</p>',
        f'  <img src="figures/checkpoint-tokens.png" alt="{alts["checkpoint-tokens"]}" />\n</p>\n\n_Avg normalized score gives each of five task families equal weight after scaling its observed checkpoint range to 0–1. Other panels retain native scores._')
    old = '''For both zero-shot and probes, Composite equally averages conservation, motifs, core/non-core and SV; AF is separate.
The motif score averages eight recovery tasks: start/stop codons and splice donor/acceptor sites in maize and tomato._'''
    new = '''The motif score averages eight recovery tasks: start/stop codons and splice donor/acceptor sites in maize and tomato._

### Aggregate score definitions

Only summaries **across task families** change. Conservation/core AUROC, motif accuracy, SV average precision and AF Spearman correlation retain their definitions and existing means across tasks within each family. The primary bar chart displays task-family scores ×100; other panels retain native units.

- **Avg Rank** remains unchanged for the zero-shot baseline comparison: rank each task against the same **11 published baseline models**, frozen at the [archived leaderboard revision](https://huggingface.co/spaces/plantcad/plantcad2-zeroshot-leaderboard/blob/f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29/data/results.csv), then average task percentiles within families and give conservation, motifs, core/non-core and SV 25% each. Per-task percentile is `100 × (reference scores below the model + half the ties) / reference count`, using exact stored values. AF is unavailable for these baselines and is excluded.
- **Win Rate** compares only the two final probe models. For each task, award 1 for a higher score, ½ for an exact tie and 0 for a lower score. Average task wins within each family, then give all five families—including AF—20% each. Display the result as a percentage. This is a family-balanced fraction of task wins, not a fraction of correctly predicted examples; win margins do not affect it.
- **Avg normalized score** describes checkpoint progress. First compute each family's existing mean task score, then scale it as `(score − minimum) / (maximum − minimum)` over all nine unique checkpoints from both runs. Use separate bounds for zero-shot and probes, shared across both runs within each readout. Average the five normalized family scores equally, including AF, to obtain a **0–1** score.

The 11-model reference comprises PlantCAD (512 bp), PlantCAD2 Small/Medium/Large, PlantCAD2.5 Large, Evo 2 7B/20B, GPN, Carbon-3B and NTv3 100M/650M; it includes models not drawn in the overview.
The normalization population contains the shared pre-DNA model once, four checkpoints from the 216B run (2.16B, 30.88B, 123.52B, 216.16B tokens), and four from the 1T run (10B, 181.82B, 545.46B, 1T tokens).

Higher is better for all three summaries, but they have different meanings and should not be compared numerically with each other. The normalized average is descriptive of the checkpoint set shown; adding checkpoints can change its bounds. Its levels are not comparable between zero-shot and probes. Native family panels retain the absolute scores.'''
    assert old in body
    body = body.replace(old, new)
    body = body.replace('_Changes are in the original score units; colors show direction, not magnitude._',
                        '_Family changes use native units; Avg normalized score changes use normalized 0–1 units. Colors show direction, not magnitude._')
    assert body.split('<p align="center">', 1)[0] == opening
    assert "Composite" not in body and 'github.com/eric-czech/plantcad2' not in body
    assert "Marin Avg Rank" not in body
    assert body.count('<details>') == body.count('</details>') == 5 and '<details open' not in body
    assert re.findall(r'\|.*\|', body[body.index('<summary>Probe inputs'):]) == re.findall(r'\|.*\|', source[source.index('<summary>Probe inputs'):])
    (HERE / "draft.md").write_text(body)
    (HERE / "draft-text-provenance.json").write_text(json.dumps({
        "source_url": "https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065",
        "source_updated_at": live["updated_at"], "source_body_sha256": hashlib.sha256(live["body"].encode()).hexdigest(),
        "preserved_opening_sha256_LF": hashlib.sha256(opening.encode()).hexdigest(), "opening_unchanged": True,
        "probe_and_control_tables_unchanged": True, "published": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
