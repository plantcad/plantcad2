# Augmented MarinMoE checkpoint evaluations

Evaluate the two augmented d768 models from [MarinDNA experiment #593](https://github.com/Open-Athena/marin-dna/issues/593) at quarter-budget permanent checkpoints.
The evaluator reuses the PlantCAD2 protocol in `../marindna_exp586/` and the fast-fa3 inference implementation from [PR #590](https://github.com/Open-Athena/marin-dna/pull/590).
Historical experiment code and results remain unchanged.

Run `uv sync --locked` and `uv run --locked pytest` for CPU contracts.
The separate validated H100 provider environment owns Torch, CUDA, FA3, vLLM and Triton; do not install the scientific CPU environment over it.
The vendored deployment and training presentation sources are pinned in `src/exp593_evals/sources.json`.

## Scientific comparisons

The five zero-shot variants compare historical lowercase scoring, case-collapsed lowercase scoring, case-collapsed uppercase input, minimal FIM, and metadata-rich FIM.
FIM uses the training renderer's supported templates and original case-sensitive DNA IDs, with separately BPE-tokenized text.
Case-pair probabilities are combined before conditioning on four canonical bases.
Keep exact target spans absent from FIM flanks; never include labels, AF values, target motifs or answer-derived annotations in metadata.
FIM examples share DNA context across presentations and fit within 8192 total tokens.
Existing probe definitions and chromosome-held-out splits are reused, with heads refitted per checkpoint.

## Execution

Read [operations.md](operations.md) before dispatch or recovery.
The checkpoint observer is read-only and never submits jobs or makes recovery decisions.
Native committed metadata determines availability; W&B supplies progress context.
All GPU jobs use whole H100 nodes, batch priority and Iris user eczech.
The single results comment on this experiment's PlantCAD2 PR is updated in place as verified results become available.
