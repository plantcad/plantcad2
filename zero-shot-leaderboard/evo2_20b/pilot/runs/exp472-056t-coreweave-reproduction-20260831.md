# MarinDNA 1B 0.56T — CoreWeave/Iris reproduction

The LR5e-4/WD0.1 final checkpoint reproduced the prior Lambda evaluation exactly on two CoreWeave H100x8 nodes. All 20 selected task scores, all 77 recorded per-context metrics, and all 77 saved raw prediction arrays (3,830,000 values) agree; an independent local check confirmed matching array shapes, dtypes, and bytes. No forward/RC selections changed. Composite remains **0.6975179962197113**.

This establishes reproducibility for this checkpoint, retained sample set, H100 hardware, and pinned inference settings; it is not evidence that arbitrary environments or changed precision settings are interchangeable.

## Results

| Group | Lambda | CoreWeave | Δ |
| :--- | ---: | ---: | ---: |
| Conservation | 0.670884 | 0.670884 | 0 |
| Masked motif | 0.569908 | 0.569908 | 0 |
| Core/non-core | 0.699672 | 0.699672 | 0 |
| Structural variant | 0.849608 | 0.849608 | 0 |
| Composite | 0.697518 | 0.697518 | 0 |

Composite is the unweighted mean of the four task-group scores, not the mean of the 20 task rows.

<details>
<summary>All 20 paired task results</summary>

| Task | Lambda | CoreWeave | Δ |
| :--- | ---: | ---: | ---: |
| Conservation — Andropogoneae, genome-wide | 0.721996 | 0.721996 | 0.000000 |
| Conservation — Poaceae, non-TIS CDS | 0.699142 | 0.699142 | 0.000000 |
| Conservation — Poaceae, TIS CDS | 0.591513 | 0.591513 | 0.000000 |
| Masked motif — Maize TIS (start) | 0.481800 | 0.481800 | 0.000000 |
| Masked motif — Maize TTS (stop) | 0.384400 | 0.384400 | 0.000000 |
| Masked motif — Maize splice donor | 0.816700 | 0.816700 | 0.000000 |
| Masked motif — Maize splice acceptor | 0.816500 | 0.816500 | 0.000000 |
| Masked motif — Tomato TIS (start) | 0.401680 | 0.401680 | 0.000000 |
| Masked motif — Tomato TTS (stop) | 0.220910 | 0.220910 | 0.000000 |
| Masked motif — Tomato splice donor | 0.725273 | 0.725273 | 0.000000 |
| Masked motif — Tomato splice acceptor | 0.712000 | 0.712000 | 0.000000 |
| Core/non-core — Maize TIS (start) | 0.614869 | 0.614869 | 0.000000 |
| Core/non-core — Maize TTS (stop) | 0.644980 | 0.644980 | 0.000000 |
| Core/non-core — Maize splice donor | 0.771584 | 0.771584 | 0.000000 |
| Core/non-core — Maize splice acceptor | 0.796035 | 0.796035 | 0.000000 |
| Core/non-core — Tomato TIS (start) | 0.611907 | 0.611907 | 0.000000 |
| Core/non-core — Tomato TTS (stop) | 0.635943 | 0.635943 | 0.000000 |
| Core/non-core — Tomato splice donor | 0.752078 | 0.752078 | 0.000000 |
| Core/non-core — Tomato splice acceptor | 0.769981 | 0.769981 | 0.000000 |
| Structural variant — Impact prediction | 0.849608 | 0.849608 | 0.000000 |

</details>

## Execution and throughput

| Platform | H100s | Evaluation wall time | Aggregate forwards/s | Active-worker forwards/s/GPU | Full-split estimate on that allocation |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Lambda, one VM | 2 | 3:21:43 | 33.05 | 16.54 | 29.02 h |
| CoreWeave, two nodes | 16 | 0:27:19 | 244.08 | 16.37 | 3.93 h |

This is a **7.38× wall-clock speedup using 8× as many GPUs**, not a faster per-GPU inference path. Active-worker throughput was about 1.0% lower; wall-clock scaling also includes startup skew and sample-chunk scheduling/I/O. Both runs scored 400,000 sequences / 3,276,800,000 tokens. Peak reserved GPU memory was identical at 21.587890625 GiB. CoreWeave wall time is the first worker's evaluation start through the last worker's finish, including chunk I/O but excluding model setup and final reduction/upload. Active-worker throughput divides total forwards by summed worker evaluation durations. The full-split estimates cover 1,727,943 rows and are extrapolations, not full evaluations.

Submission through verified upload and terminal job state took 31m18s. The two task containers ran for 29m38s and 27m42s; both exited normally, releasing their allocations. The job recorded zero failures and zero preemptions. All submissions, including the environment probe and smoke, explicitly used `--priority batch --user eczech`.

## Environment differences

| Setting | Lambda | CoreWeave |
| :--- | :--- | :--- |
| Python | 3.12.3 | 3.11.12 |
| Torch version string | 2.7.0 | 2.7.0+cu128 |
| CUDA runtime | 12.8 | 12.8 |
| cuDNN | 9.8.0 | 9.7.1 |
| GPU | NVIDIA H100 80GB HBM3 | NVIDIA H100 80GB HBM3 |
| Torch C++ ABI | Not captured | CXX11 enabled |

Transformers 5.15.1, FlashAttention 2.8.3.post1, NumPy 1.26.4, sklearn 1.5.2, BF16 model execution, FP32 A/C/G/T softmax, batch32, TF32, and disabled KV cache were unchanged. All 16 CoreWeave workers independently profiler-verified the external FlashAttention-2 forward kernel. Tokenizer parallelism was disabled on both paths; CoreWeave additionally capped OpenMP/MKL at two threads per process for eight workers per node. The CoreWeave driver was 595.71.05; the Lambda driver and C++ ABI were not retained in the reference metadata, so they cannot be claimed to match.

The isolated image was `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`, digest `sha256:e97058f7b9b583517643477cc9a0433e54594038efd85ec4abd7836233d626f3`. The remote venv reused image Torch/CUDA and installed the ABI-matched prebuilt FA2 wheel. No CUDA/NVIDIA dependencies or checkpoint weights were installed/downloaded locally.

## Scoring and partition verification

The original causal scorer and `sensitivity/run_condition.py` were unchanged from the Lambda run. A deterministic plan assigned 200 contiguous, batch-aligned chunks across 16 persistent model workers. The reducer verified original row coverage and array hashes, concatenated predictions in original order, and computed full-task metrics. It did not average chunk AUROCs/AUPRCs or take per-example strand maxima. Motif ambiguous-base validity masks and forward-only SV scoring were preserved.

Three local partition tests checked exact minibatch identity, coverage, and balancing for 1/2/4/8 nodes. Four remote reducer tests checked array order, full-task AUROC, motif validity masks, and strand selection. Before production inference, the reducer also reproduced all saved Lambda per-context metrics directly from the reference arrays. Model and fixture hashes were checked on both nodes and again in the small local verification.

## Provenance and artifacts

- Job: `/eczech/plantcad2-cw-056t-lr5e4-n2-20260831-v1`, `cw-us-east-02a`, 2026-08-31. Each replica requested H100x8, 32 CPUs, 256 GiB RAM, and 160 GiB ephemeral disk.
- Submitted evaluation implementation: `3cd251ef57527c64fbe21abf8974fbf41de7076c`. Later monitor/test documentation and future-attempt safety guards did not alter the measured run.
- Model HF revision: `2972ca5abb575ccb9878d2525aafd396e6b73d7c`, `exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1/hf/step-535985`.
- Reference HF revision: `dfd4d41f0fc70f95cc22081de28a1546412f1bdc`. All 20 fixtures are the retained 10,000-row random-unstratified seed-0 samples.
- Model safetensors SHA-256: `71e0c9587b2887f1376095439137223bcb94bf6f338a5264988c95a6b5199d98`.
- Sample manifest SHA-256: `6549414b71fe76e8d1f5e0f10b0559a07b35e064607fae0de20b81fc9dfc1345`.
- Shared scoring SHA-256: `af302e42ae740cd01af1d3ca556dcfbef0d525ef385ce72a9697dcf59501599c`; causal scorer SHA-256: `fb99363dd4c7c14070317f5cbf44cbf07e5862ee58506460613a1bb4477f9a4f`.
- [Verified private HF artifact](https://huggingface.co/eczech/marindna-exp472/tree/d83121d1b2fda75c9619047c7f10f8e7924173a6/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1/results/step-535985/coreweave/plantcad2-cw-056t-lr5e4-n2-20260831-v1): 445 files / 27,852,997 bytes, containing raw chunks, merged arrays, task/context comparisons, environment records, plan, and provenance. Source TSVs are referenced by immutable revision rather than duplicated.
- Local verification downloaded only 45 files / 13,797,751 bytes. Local disk stayed at about 63% used with 155 GiB available; no related local GPU cache or venv needed pruning.

See the [CoreWeave runbook](../coreweave/README.md), [Lambda entrypoints](../lambda/README.md), and [machine-readable comparison](exp472-056t-coreweave-reproduction-20260831.json). The PR result update was prepared as a local draft only; no result comment or OP was posted/edited.
