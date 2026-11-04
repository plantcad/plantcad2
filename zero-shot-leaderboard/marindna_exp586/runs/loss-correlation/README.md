# Validation loss versus downstream evaluations

This report updates the existing results comment on plantcad/plantcad2 PR22 using eight completed DNA-trained checkpoints: three exp472 dense token budgets, both exp582 d1536 models, both exp586 0.22T d768 models, and the final 1T language-initialized d768 model.
The main figure excludes the two 481M-active MoEs, leaving six models; the second figure includes all eight.
All points use measured final-checkpoint losses and downstream scores.
Color distinguishes Dense from MoE; marker shape distinguishes scratch from language-pretrained initialization.
Dotted ordinary-least-squares lines are fitted separately to Dense and MoE over their observed loss ranges.

`models.json` identifies runs, original final losses, checkpoint URIs and metadata hashes.
`inputs.json` identifies the immutable samples, including the first 65,536 validation windows of 8,192 bp (536,805,376 next-base targets).
The entire common sample was checked against the historical dense cache.
Dense losses are retained after a D216 restoration/sampling replay; each MoE replays its original 512 windows and then scores the common sample.
`results-manifest.json` records durable S3 locations and hashes for all six raw execution outputs under `validation/`.
Their batch sufficient statistics were independently summed, and all common-sample evaluations require zero dropped expert assignments.

The final 1T common-sample loss is reused from its completed evaluation, with exactly the same input hashes as the original analysis.
Its archived downstream outputs and AF probe are under `downstream/m61-l-1t/`; `provenance.json` records file hashes and the checkpoint-bound execution receipt verified directly from CoreWeave S3.
No new inference or fitting is required.

Every model uses the same historical 8K Natural convention and the same 20 tasks.
Composite is the equal-weight mean of Conservation, Masked motif, Core/non-core and Structural variant; maize AF is excluded.
Natural retains the historical better causal direction per task.
Zero-shot AF uses strand-averaged full causal-suffix LLR on 94,075 variants.
Probe AF uses the fixed 8K target-position, strand-averaged ridge readout on the identical 28,378-variant test split, including for the 1T model.
The original seven models' scores and losses are unchanged.

From this package, regenerate the analysis, both PNG/SVG figures and `draft.md`:

```bash
.venv/bin/python -m exp586_evals.loss_correlation .
.venv/bin/pytest -q tests/test_loss_correlation.py
```

`analysis.json` contains the compact plotting data and per-checkpoint vocabulary/readout/context sensitivity values.
Inspect both figures at presentation size before publication.
The draft uses relative image and data links; publication replaces them with commit-pinned links in plantcad/plantcad2.

The maintained preparation and GPU worker sources are `src/exp586_evals/validation_loss_prepare.py` and `validation_loss_worker.py`.
The original loss analysis runs in the compatible Marin project at `/home/exedev/repos/marin-dna-br/codex-exp586-loss-analysis/experiments/exp586_plantcad2_moe`.
The final 1T native scatter worker and its request are separately hash-pinned in that model's loss artifact; common sample, target count, full-vocabulary loss definition and replay requirements remain identical.
