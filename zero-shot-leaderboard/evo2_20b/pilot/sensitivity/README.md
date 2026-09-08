# Numerical and runtime sensitivity checks

`run_condition.py` evaluates the same seed-0 sample under one explicitly named
precision/runtime condition. It records paired scores, primary metrics, GPU and wall
throughput, peak memory, package versions, and attention-kernel profiler events.

The reference environments reuse the Lambda image's CUDA-enabled PyTorch. Install
FlashAttention with `--no-deps`; otherwise uv may replace the image's PyTorch/CUDA
stack while resolving FlashAttention's unconstrained Torch dependency:

```bash
uv venv --system-site-packages ~/plantcad2/.venv-current
uv pip install --python ~/plantcad2/.venv-current/bin/python \
  -r ../requirements.txt packaging ninja wheel einops
MAX_JOBS=8 uv pip install --python ~/plantcad2/.venv-current/bin/python \
  flash-attn==2.8.3.post1 --no-build-isolation --no-deps
```

Verify `torch.__version__` and `torch.version.cuda` before and after installation.
The August 2026 comparison used Transformers 5.15.1 and 4.57.6 with the same
PyTorch 2.7.0, CUDA 12.8, and FlashAttention 2.8.3.post1 runtime.

Create the shared sample once:

```bash
python ../sample_tasks.py \
  --output-dir samples-seed0-n5000 \
  --samples 5000 --seed 0 --task-set sensitivity \
  --shuffle-buffer-size 200000 --shared-seed --no-balance-binary
```

Treat the resulting TSVs—not the seed alone—as the comparison fixture. Streaming
shuffle order can vary with the data-library/runtime path. The manifest therefore
records the resolved dataset revision, library versions, and a SHA-256 digest for
each TSV; retain and reuse those exact files across checkpoint and environment runs.

Example production-relevant run (FlashAttention verification is mandatory by default):

```bash
python run_condition.py \
  --model /path/to/hf/checkpoint \
  --sample-dir samples-seed0-n5000 \
  --output-dir results/bf16-fa2-b4 \
  --condition bf16-fa2-b4 \
  --dtype bf16 --attention flash_attention_2 \
  --batch-size 4 --sv-batch-size 2
```

True FP32 and eager-attention checks must pass `--no-require-flash`; FlashAttention
kernels do not support FP32 inputs. These are correctness diagnostics, not recommended
production configurations.

`run_matrix.sh` is the exact condition matrix. `analyze.py` performs paired comparisons
against the BF16/FlashAttention-2 baseline and extrapolates the measured sequence rate
to all 1,727,943 current leaderboard rows (3,455,886 full-sequence forwards after the
two strand contexts per non-SV row and ref/mut forwards per SV row). It weights
non-SV and SV throughput by their actual full-leaderboard row counts rather than the
equal task mix used by the sensitivity sample.

```bash
python analyze.py --matrix-dir results --sample-dir samples-seed0-n5000 \
  --output results/summary.json
uv run --with matplotlib python plot_summary.py --summary results/summary.json \
  --output sensitivity-summary.png
```

The FP32 diagnostics use matched eager-attention controls, and the FP32 TF32-off
diagnostic uses FP32 TF32-on as its control. `plot_summary.py` refuses to plot any
production condition whose profiler record did not verify a known flash forward
operation or CUDA kernel.

`run_batch_sweep.sh` is a short 32-example cache/batch diagnostic. It checks batches
1–32 with the generation KV cache enabled and batches 4–64 with it disabled; causal
scoring does not consume the cache, so disabling it is expected to save memory and
avoid batch-size throughput cliffs without changing scores.
