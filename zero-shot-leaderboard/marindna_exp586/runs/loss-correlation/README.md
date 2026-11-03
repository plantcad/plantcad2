# Validation-loss correlation draft

This analysis is for a new results comment on plantcad/plantcad2 PR22. Nothing in this directory is authorized for automatic publication.

The main figure excludes the two 481M-active MoEs and recalculates all seven plotted correlations on the remaining five models. The original seven-model figure is retained unchanged in a dropdown immediately after it. The table and existing readout/sensitivity checks still cover the full seven-model collection; the five-model correlations are stored separately under `without_481m` in `analysis.json`.

Only the five-model Composite panel includes a fit equation and a conditional projection for the 1T-token d768 language-initialized run at an assumed final loss of 0.88. This projection is not a measured model or an input to the fit/correlations. Coefficients, observed loss range and projection provenance are stored under `without_481m.composite_linear_fit`. Both AF panels are explicitly labeled as excluded from Composite.

The analysis includes seven DNA-trained checkpoints: three exp472 dense token budgets, both exp582 final d1536 models, and both exp586 final d768 1× models. The unadapted language model and reference models are excluded. Every checkpoint is an individual model in the correlation; no paired or model-selection analysis is performed.

`models.json` identifies original W&B runs, the metric selected from final history, exact checkpoints and metadata hashes. All seven original runs were resolved, including the five now under plantcad/marin-dna. `inputs.json` identifies the two immutable validation samples. The entire shared sample was checked against the historical dense cache, not just spot-checked.

`original-final-history.json` preserves the selected final W&B history records. `results-manifest.json` records durable S3 locations and hashes for the five raw execution outputs under `validation/`. Their batch sufficient statistics were independently summed to verify the saved losses and counts. `analysis.json` is the compact plotting table with correlations and sensitivity results; it points back to those raw files rather than duplicating the batch statistics.

The common native loss uses the historical dense sample: the first 65,536 8,192-base validation windows, or 536,805,376 next-token targets. The original dense native losses are retained after a D216 replay verifies restoration and sampling. Each MoE first reproduces its original 512 evenly spaced-window loss and then scores the shared sample. Loss uses the native full vocabulary. The A/C/G/T-conditional sensitivity uses only the four MoEs, so the other dense checkpoints do not need another forward pass.

The maintained preparation and GPU worker sources are `src/exp586_evals/validation_loss_prepare.py` and `validation_loss_worker.py`. They run in the isolated compatible Marin project at `/home/exedev/repos/marin-dna-br/codex-exp586-loss-analysis/experiments/exp586_plantcad2_moe`. The worker file is copied byte-for-byte into the `exp586_moe` package and its SHA-256 is pinned by the uploaded request. The tracked execution operations document in that checkout and the CoreWeave sweep skill own fleet policy and SQLite recovery; this directory owns the scientific inputs and outputs.

Downstream metrics are reused from the checked-in historical artifacts and exp586 final results. All four task groups are weighted equally in Composite; AF is never included. Natural retains the historical better causal direction per task, not strand averaging. Zero-shot AF separately uses strand-averaged causal LLR on all 94,075 variants. Probe AF uses the fixed 8K target-position, strand-averaged ridge readout on the identical 28,378-variant test split.

With verified GPU results under `validation/{d216,m61-s,m61-l,m481-s,m481-l}.json`, run from this package:

```bash
.venv/bin/python -m exp586_evals.loss_correlation .
.venv/bin/pytest -q tests/test_loss_correlation.py
```

This writes `analysis.json`, both `loss-correlation` and `loss-correlation-without-481m` figures (PNG/SVG), and the unpublished `draft.md`. Inspect the final figures visually and check all point labels, legend spacing and axis ranges before review. The draft intentionally uses local image links; replace them with durable image URLs only when publication is authorized.
