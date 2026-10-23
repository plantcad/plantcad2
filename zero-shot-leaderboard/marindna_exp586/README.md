# MarinDNA exp586 evaluations

Evaluate the scratch-initialized and language-pretrained d768 1× MoE checkpoints from [MarinDNA #586](https://github.com/Open-Athena/marin-dna/issues/586) with the same PlantCAD2 protocol used for exp582.

The evaluation covers all 20 original tasks at 8,192 bases, 19 centered tasks at 2,048 bases, maize allele-frequency scoring at both contexts, frozen probes, and task-independent bidirectional methods.
Historical dense, PlantCAD2, PlantCAD2.5, and Evo results are read from archived computations and are never rerun.
The task catalog, genomic memberships, reference assembly, and evaluation formulas remain pinned to the exp582 protocol.

```bash
uv run --locked --extra cpu pytest
uv run --locked --extra cpu python -m exp586_evals.historical .
uv run --locked --extra cpu python -m exp586_evals.visual_report \
  --catalog runs/exp586-final/catalog.json \
  --historical historical \
  --output runs/exp586-final/assets
```

The report catalog controls which current and archived models appear in each figure.
The catalog selects the two final exp586 checkpoints and the report views in which they appear.

The scratch model uses the historical eight-entry DNA tokenizer. The pretrained model retains its 128,256-entry language vocabulary but applies the same character-forcing pretokenizer used during DNA adaptation. Both therefore consume one nucleotide per token; A/C/G/T scoring projects their four corresponding output logits directly.
Whole-window and target-position frozen features are shape-dynamic, so the same fitting code supports d768 and later exp586 checkpoints without a width assumption.

The implementation reuses the evaluator from [PlantCAD2 PR20 at `475fba5`](https://github.com/plantcad/plantcad2/tree/475fba58f81b79efa234d0018eb0776d1a960bfd/zero-shot-leaderboard/evo2_20b).
`protocol.py` loads that implementation and `native.py` adapts native Marin MoE predictions to its established task, allele-frequency, and probe interfaces.
`protocol-provenance.json` records the reused source hashes and the small integration changes.

GPU workers use dropless Sonic inference after comparison against the backend recorded in each checkpoint.
They emit the original task, allele-frequency, and frozen-embedding schemas plus exact checkpoint, tokenizer, input, implementation, and wheel bindings.
All CoreWeave jobs use H100s at batch priority under Iris user `eczech`.

The full evaluation uses batches of eight and the proven 16-worker exp582 file partition.
Postprocessing verifies complete chunk coverage and hashes before applying the original metric reducers, genomic-block bootstrap, and grouped ridge-probe fitting.
Saved probe estimators are reloaded and checked against their persisted predictions.
