# exp472 step 75046 numerical and runtime sensitivity

This is an execution-sensitivity pilot for a very early-training checkpoint, not a
leaderboard submission. Four representative PlantCAD2 rows use 5,000 random,
unstratified examples each with shared seed 0: Poaceae non-TIS conservation, tomato
splice-acceptor recovery, maize TIS core/non-core, and structural-variant impact.
Full-sample conditions therefore score 20,000 examples and execute 40,000 8,192-token
forwards after strand contexts or ref/mut SV pairs.

- W&B run ID: `exp472-plantcad2-angiosperm-lr0p0001-wd0p2-v2`
- Levanter/HF checkpoint: `step-75046`
- Model: Qwen3, 973,178,880 parameters, vocabulary size 7
- Hardware: one Lambda H100 SXM5 80 GB at $4.29/hour
- Baseline runtime: PyTorch 2.7.0, CUDA 12.8, Transformers 5.15.1,
  FlashAttention 2.8.3.post1
- Dataset revision: `d340debe0c8402c84f0696cd2002f87c2f7ba6db`
- Private artifact prefix:
  `eczech/marindna-exp472/exp472-plantcad2-angiosperm-lr0p0001-wd0p2-v2/results/step-75046/sensitivity-seed0-n5000`

Every production-relevant condition profiler-verified an actual flash forward kernel
on a real 8,192-token input. FlashAttention-2 runs dispatched
`flash::flash_fwd_kernel`; PyTorch SDPA dispatched
`pytorch_flash::flash_fwd_kernel`. True FP32 does not support the required
FlashAttention-2 path, so FP32/eager rows below are correctness diagnostics only.

## Main findings

- BF16/FlashAttention-2 repeated byte-for-byte on 1,000 examples. Transformers 4.57.6
  and 5.15.1 were also byte-identical on all 5,000 examples per task.
- FP16 was 2.6% slower than BF16, changed 306/60,000 non-SV base calls, and moved the
  largest strand-specific metric by 0.0016. No task changed its selected strand.
- SDPA used PyTorch's flash kernel, was 1.8% slower than FlashAttention-2, changed
  285/60,000 calls, and moved the largest strand metric by 0.0008.
- Batch 1 was 7.9% slower than batch 4. It changed 52/60,000 calls and moved the
  largest strand metric by 0.0004.
- Native BF16 versus FP32 A/C/G/T softmax had no speed effect; its largest selected
  task-metric change was 0.00005 and largest strand-specific change was 0.0004.
- TF32 on/off was exactly identical for BF16, as expected.
- Disabling the unused generation KV cache and using batch 32 was 3.7% faster than the
  baseline. Non-SV probabilities were byte-identical; SV AUPRC moved -0.000019.
- Disabling TF32 for true FP32 eager execution changed no base calls and moved the
  largest metric by 0.000283, but cut throughput from 3.01 to 1.36 sequences/second.
- No production condition changed the winning strand/context for any task.

The recommended evaluation path is BF16, explicit FlashAttention-2, FP32 A/C/G/T
softmax, `use_cache=False`, and batch 32 on this 1B checkpoint/H100 combination. It
projects to about 56.9 H100-hours for all 1,727,943 current leaderboard rows, versus
58.5 hours for the conservative cache-on/batch-4 baseline. The estimate weights the
measured non-SV and SV rates by their actual full-leaderboard row counts and excludes
provisioning, checkpoint conversion, and remote dataset materialization.

## Production-relevant conditions

`|Δ metric|` is the largest absolute primary-metric change in either strand/context
against the named paired control. Call changes cover saved A/C/G/T argmaxes for the
three non-SV tasks; SV saves boundary LLR scores instead.

| Condition | Paired control | n/task | Flash verified | Seq/s | Full hours | Peak GiB | Max \|Δ metric\| | Call changes |
| :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| BF16 · FA2 · b4 | self | 5,000 | yes | 16.34 | 58.54 | 5.84 | 0 | 0/60,000 |
| Exact repeat | BF16 · FA2 · b4 | 1,000 | yes | 16.36 | 58.61 | 5.84 | 0 | 0/12,000 |
| Transformers 4.57.6 | Transformers 5.15.1 | 5,000 | yes | 16.35 | 58.47 | 5.84 | 0 | 0/60,000 |
| FP16 · FA2 · b4 | BF16 · FA2 · b4 | 5,000 | yes | 15.91 | 60.13 | 5.84 | 0.001600 | 306/60,000 |
| BF16 · SDPA-flash · b4 | BF16 · FA2 · b4 | 5,000 | yes | 16.04 | 59.56 | 5.84 | 0.000800 | 285/60,000 |
| BF16 · FA2 · b1 | BF16 · FA2 · b4 | 5,000 | yes | 15.04 | 64.00 | 2.84 | 0.000400 | 52/60,000 |
| BF16 · native softmax | FP32 softmax | 5,000 | yes | 16.35 | 58.46 | 5.84 | 0.000400 | 74/60,000 |
| BF16 · TF32 off | TF32 on | 1,000 | yes | 16.34 | 58.62 | 5.84 | 0 | 0/12,000 |
| BF16 · no cache · b32 | BF16 · cache · b4 | 5,000 | yes | 16.94 | 56.86 | 21.59 | 0.000019 | 0/60,000 |

The exact-repeat and Transformers-version full-hour differences are timing noise; their
saved probabilities and scores are byte-identical.

## Baseline task results by strand

| Task | Metric | Left | Reverse complement | Selected |
| :--- | :--- | ---: | ---: | ---: |
| Conservation · Poaceae non-TIS | AUROC | 0.658547 | 0.656862 | left |
| Masked motif · Tomato acceptor | accuracy | 0.204400 | 0.511800 | reverse complement |
| Core/non-core · Maize TIS | AUROC | 0.380504 | 0.568236 | reverse complement |
| Structural variant · impact | AUPRC | 0.738904 | — | left/ref-mut |

## Cache and batch smoke test

This 32-example diagnostic explains the cache setting. The model does not consume KV
states during scoring, but returning them changes memory and kernel behavior.

| Batch | Cache | Seq/s | Peak GiB |
| ---: | :---: | ---: | ---: |
| 1 | on | 15.04 | 2.84 |
| 4 | on | 16.53 | 5.84 |
| 8 | on | 8.43 | 9.59 |
| 16 | on | 8.44 | 18.34 |
| 32 | on | 8.52 | 31.84 |
| 4 | off | 16.75 | 4.84 |
| 8 | off | 16.92 | 7.34 |
| 16 | off | 16.95 | 12.09 |
| 32 | off | 17.01 | 21.59 |
| 64 | off | 17.01 | 48.58 |

## Eager and FP32 diagnostics

| Condition | Paired control | n/task | Seq/s | Full hours | Max \|Δ metric\| | Call changes |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| BF16 eager | BF16 FA2 b1 | 256 | 3.01 | 319.39 | 0.003906 | 26/3,072 |
| FP32 eager, TF32 on | BF16 eager | 256 | 3.01 | 320.20 | 0.003906 | 32/3,072 |
| FP32 eager, TF32 off | FP32 eager, TF32 on | 256 | 1.36 | 706.51 | 0.000283 | 0/3,072 |

The HF checkpoint stores all 179 tensors as FP32. The eager comparison therefore uses
real FP32 weights, but it necessarily changes attention backend relative to the
production path. BF16 eager versus BF16 FA2 isolates that backend effect; FP32 eager
versus BF16 eager then isolates arithmetic precision under the same backend.

## Sampling portability

The dataset repo had not changed since September 2025, but regenerating four seed-0
samples under the current data stack reproduced only two prior samples exactly. For
the first 2,800 rows, conservation had 38 rows in common and tomato acceptor had 58;
maize TIS core/non-core and SV matched all 2,800 rows in order. Seed plus dataset
revision is therefore not a sufficient portable fixture for the streaming shuffle.
The materialized TSVs are retained, and their manifest records the resolved revision,
`datasets`/pandas versions, and SHA-256 file hashes.

Artifact commit: `cb1a0423772fd96b0465ae342ce8b0488dc97490` (96 files,
212,949,805 bytes; verified against the local artifact tree before teardown)
