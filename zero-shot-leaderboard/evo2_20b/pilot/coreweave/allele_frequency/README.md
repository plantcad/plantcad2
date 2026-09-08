# Maize allele-frequency evaluation on CoreWeave

This runner evaluates the three pinned MarinDNA LR5e-4 / WD0.1 checkpoints against allele frequency (AF) in `plantcad/maize-allele-frequency`. It runs on Iris-managed H100x8 nodes and retains row-level predictions in `plantcad/marindna-exp472`.

## Protocol

- Reproduce the published consequence-balanced 10k and 20k test samples exactly with Polars 1.34.0 and seed 42, then generate 50k and 100k by the same method.
- Score the 94,953-row union once per checkpoint. The nominal 10k, 20k, 50k, and 100k samples contain 9,998, 19,997, 48,625, and 94,075 retained rows.
- Exclude `2:234358020 A>T`: its 8,192-bp reference window contains 771 non-ACGT bases that become causal-suffix targets on the reverse-complement strand. The exclusion is applied only after reproducing the published samples and is recorded in every manifest.
- Compute `log P(ALT sequence) - log P(REF sequence)` over the variant and complete causal suffix in FP32 A/C/G/T log-softmax space. Average forward and reverse-complement LLRs, then correlate the result directly with AF using Pearson and Spearman correlation.
- Load BF16 weights with explicit external FlashAttention-2 and TF32 enabled. A profiler trace must contain an external FA2 kernel before scoring begins.
- Prefix sharing is active only for the common prefix forward. The divergent suffix forward explicitly disables cache. A 128-row H100 smoke benchmark found this path 1.25x faster than paired uncached forwards with identical Spearman correlation and a maximum ACGT LLR difference of `3.04e-5`.

## Run

The submission wrapper enforces batch priority, the approved H100 clusters, and supported node counts. It builds the CUDA environment only on the remote allocation.

```bash
source ~/oa.env
source ~/.zshrc
export IRIS_BIN=/path/to/iris

CLUSTER=cw-us-east-02a NODES=1 CHECKPOINT=056t METHOD=cached JOB_NAME=maize-af-smoke \
  bash zero-shot-leaderboard/evo2_20b/pilot/coreweave/allele_frequency/submit.sh --smoke --smoke-rows 128

CLUSTER=cw-us-east-02a NODES=2 CHECKPOINT=056t METHOD=cached JOB_NAME=maize-af-056t \
  bash zero-shot-leaderboard/evo2_20b/pilot/coreweave/allele_frequency/submit.sh

CLUSTER=cw-us-east-02a NODES=1 CHECKPOINT=056t SAMPLE_SIZE=100000 CONTEXT_LENGTH=512 METHOD=cached JOB_NAME=maize-af-056t-context-512 \
  bash zero-shot-leaderboard/evo2_20b/pilot/coreweave/allele_frequency/submit.sh
```

`CHECKPOINT` is one of `022t`, `039t`, or `056t`; `NODES` may be changed independently. `SAMPLE_SIZE=100000` scores only the retained 94,075-row sample instead of the union of all sample sizes. `CONTEXT_LENGTH` is one of 128, 256, 512, 1,024, 2,048, 4,096, or 8,192. Every worker verifies that the selected sequence is the exact centered crop of the 8,192-bp window and that the variant remains at zero-based index `length / 2` forward and `length / 2 - 1` after reverse complementation. Use `iris task exec <job>/0 -- ...` to inspect live task files or processes without replacing the allocation.

Each completed run uploads metrics, provenance, exact input and execution manifests, worker environments, and `predictions.parquet` to the checkpoint's `results/step-*/coreweave/<job-name>/` path in the Hugging Face repository. Raw chunk archives remain in CWS3.
