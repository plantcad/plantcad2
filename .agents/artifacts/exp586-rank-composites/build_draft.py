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
        "benchmark-comparison": "Zero-shot comparison of eight models. Top: native mean Conservation AUROC, motif accuracy, core/non-core AUROC and SV average precision, with a marked broken vertical scale and a value on every bar. Bottom: sorted Conservation AUROC, SV average precision and Avg Rank, with marked broken horizontal scales. Avg Rank averages per-task percentiles within four equally weighted families against 11 fixed published baseline models.",
        "loss-comparison": "Unchanged W&B training and validation loss curves versus DNA tokens. Right: matched-sample final loss versus Avg Rank on the left percentile axis (solid circles) and native Conservation AUROC on the right axis (dashed diamonds). Colors identify 216B and 1T MoE models. Dense MarinDNA 1B checkpoint rugs retain the matched losses, with direct training-token labels and darker rugs for later checkpoints.",
        "checkpoint-tokens": "Six paired zero-shot and linear-probe metrics across DNA training. Top: Marin Avg Rank, Conservation, Motifs; bottom: Core/non-core, SV impact, Allele frequency. Only Marin Avg Rank is percentile-scaled; the other families keep their native scores. References are the same nine unique checkpoints, separately for each readout, with five equally weighted families including AF.",
        "checkpoint-progress": "The same six paired metrics versus actual W&B run progress. Marin Avg Rank includes all five task families; individual families retain native units.",
        "checkpoint-changes-primary": "Changes between successive checkpoints for zero-shot and linear probes. Family-score changes retain native units; Marin Avg Rank changes are in percentile points. Blue means increase, orange decrease, gray no change; color encodes sign only.",
    }
    assert len(original_images) == len(alts) == 5
    for image, old_alt in original_images:
        name = Path(image).stem
        body = body.replace(f'alt="{old_alt}"', f'alt="{alts[name]}"')
    body = body.replace(
        f'  <img src="figures/benchmark-comparison.png" alt="{alts["benchmark-comparison"]}" />\n</p>',
        f'  <img src="figures/benchmark-comparison.png" alt="{alts["benchmark-comparison"]}" />\n</p>\n\n_Native task-family scores are unchanged; Avg Rank averages task percentiles within four equally weighted families. Higher is better; rank definitions are below._')
    body = body.replace(
        'with darker coral rugs for later dense checkpoints.',
        'with darker coral rugs for later dense checkpoints. The scatter now uses Avg Rank (left axis) and Conservation AUROC (right axis).')
    body = body.replace(
        f'  <img src="figures/checkpoint-tokens.png" alt="{alts["checkpoint-tokens"]}" />\n</p>',
        f'  <img src="figures/checkpoint-tokens.png" alt="{alts["checkpoint-tokens"]}" />\n</p>\n\n_Marin Avg Rank includes AF and uses nine fixed checkpoint references, separately for zero-shot and probes. All other panels retain native scores._')
    old = '''For both zero-shot and probes, Composite equally averages conservation, motifs, core/non-core and SV; AF is separate.
The motif score averages eight recovery tasks: start/stop codons and splice donor/acceptor sites in maize and tomato._'''
    new = '''The motif score averages eight recovery tasks: start/stop codons and splice donor/acceptor sites in maize and tomato._

### Rank definitions

Only summaries **across task families** change. Conservation/core AUROC, motif accuracy, SV average precision and AF Spearman correlation retain their native values, including the existing mean across tasks within each family.

For each task, the percentile is `100 × (reference scores below the model + half the ties) / reference count`, using exact stored values. Average these task percentiles within each family, then average the families equally. Both rank summaries run from **0 to 100, higher is better**; they measure relative standing, not accuracy or effect size.

- **Avg Rank:** conservation, motifs, core/non-core and SV receive 25% each. Each task is ranked against the same **11 published baseline models**, frozen at the [archived leaderboard revision](https://huggingface.co/spaces/plantcad/plantcad2-zeroshot-leaderboard/blob/f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29/data/results.csv). AF is unavailable for these baselines and is excluded.
- **Marin Avg Rank:** those four families plus AF receive 20% each. Each task is ranked against the **nine unique language-initialized MarinMoE checkpoints**, using separate zero-shot and probe reference scores. No published zero-shot values enter probe ranks.

The 11-model reference comprises PlantCAD (512 bp), PlantCAD2 Small/Medium/Large, PlantCAD2.5 Large, Evo 2 7B/20B, GPN, Carbon-3B and NTv3 100M/650M; it includes models not drawn in the overview.
The Marin reference contains the shared pre-DNA model once, four checkpoints from the 216B run (2.16B, 30.88B, 123.52B, 216.16B tokens), and four from the 1T run (10B, 181.82B, 545.46B, 1T tokens).

Reference membership stays fixed when adding future models. Finite reference sets make ranks coarse and can hide improvements within a rank interval; the native family panels retain that information. Compare rank values only within the same definition and readout. This follows the leaderboard's per-task percentile idea but gives families equal weight and freezes the reference sets.'''
    assert old in body
    body = body.replace(old, new)
    body = body.replace('_Changes are in the original score units; colors show direction, not magnitude._',
                        '_Family changes use native units; Marin Avg Rank changes use percentile points. Colors show direction, not magnitude._')
    assert body.split('<p align="center">', 1)[0] == opening
    assert "Composite" not in body and 'github.com/eric-czech/plantcad2' not in body
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
