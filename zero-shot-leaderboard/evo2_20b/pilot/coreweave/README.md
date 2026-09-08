# CoreWeave / Iris execution

This folder isolates CoreWeave provisioning, environment setup, node orchestration, and object-store I/O. The existing [Lambda route](../lambda/README.md) remains supported. Both platforms call the same `sensitivity/run_condition.py` and `../../zero-shot-eval-causal.py` inference/scoring functions. Full-split workers skip chunk-local metrics, which may be undefined for single-class chunks; the reducer computes the original metrics over complete tasks. Inference, score definitions, and existing sampled defaults are unchanged. `../partition.py` is platform-neutral sample scheduling.

## Hard submission constraints

Every `iris job run` must explicitly specify `--priority batch --user eczech`, including smoke and debugging jobs. `submit.sh` always supplies both and never submits child jobs. Only `cw-us-east-02a` (default) and `cw-rno2a` are allowed. Each replica requests a full `H100x8` node; `NODES=2` is the default, with 1, 4, 8, etc. supported. No GB200 or partial-node allocations. Live debugging uses `iris task exec`, not a second interactive-priority allocation.

```bash
# Reuse an existing Iris CLI; do not uv-sync a GPU project on the laptop.
export IRIS_BIN=/path/to/existing/.venv/bin/iris
source ~/oa.env
source ~/.zshrc  # supplies HUGGING_FACE_HUB_TOKEN; never print the token

python zero-shot-leaderboard/evo2_20b/pilot/coreweave/capacity.py --iris-bin "$IRIS_BIN"
NODES=2 CLUSTER=cw-us-east-02a JOB_NAME=plantcad2-cw-UNIQUE-RUN-NAME \
  bash zero-shot-leaderboard/evo2_20b/pilot/coreweave/submit.sh \
  zero-shot-leaderboard/evo2_20b/pilot/coreweave/node.py
```

Use `NODES=1` with `node.py --smoke` to validate artifact access, all saved reference metrics, and four representative GPU tasks at 32 examples each. `probe.py` is a still smaller environment/storage-access check. Neither smoke mode runs or overwrites the complete evaluation.

`capacity.py` follows the H100 accounting and freshness checks in [MarinFold's utilization utility](https://github.com/Open-Athena/MarinFold/blob/main/.agents/skills/run-training-sweep-cw/scripts/utilization.py). On 2026-08-31 Iris reports availability schema version 3 while that utility requires version 2; this adapter accepts either, checks the same H100 free+held=total accounting, rejects stale observations, and ignores the ineligible GB200 cluster. Free node equivalents are advisory, not a gang-admission guarantee.

## Full splits and multiple checkpoints

`EVAL_MODE=sampled` remains the default and reuses the exact retained 10,000-row seed-0 fixtures. `EVAL_MODE=full` evaluates all 1,727,943 rows across all 20 tasks at dataset revision `d340debe0c8402c84f0696cd2002f87c2f7ba6db`. No random selection, row cap, class balancing, filtering, or shuffling is applied. Rows retain sorted HF shard/original row order. All Parquet files are size/SHA-256 checked, row counts come from Parquet metadata, and every planned row range must be accounted for exactly once. Workers read only intersecting row groups, avoiding whole-split copies in every GPU process. Binary-task reduction reads labels only; motif reduction uses the unchanged ambiguous-base mask.

`CHECKPOINT=022t|039t|056t` selects immutable post-cooldown exports in `checkpoints.py`; all three use LR5e-4/WD0.1. The default remains `056t`. The .22T entry is NOT the LR2e-4 model in older result posts. Each result path includes its actual run ID and final step. Sampled previews reuse the common retained fixtures for any checkpoint, but the optional paired Lambda-reproduction comparison is only emitted for the matching 0.56T checkpoint; it must not misrepresent a cross-checkpoint difference as a platform difference. To launch the three full runs concurrently, invoke the following once for each checkpoint with a unique job name (the launcher detaches):

```bash
CHECKPOINT=022t EVAL_MODE=full NODES=4 CLUSTER=cw-us-east-02a JOB_TIMEOUT=21600 JOB_NAME=plantcad2-full-022t-UNIQUE \
  bash zero-shot-leaderboard/evo2_20b/pilot/coreweave/submit.sh \
  zero-shot-leaderboard/evo2_20b/pilot/coreweave/node.py
# Repeat with CHECKPOINT=039t and CHECKPOINT=056t, using distinct JOB_NAME values.
```

This is 4 H100x8 nodes per checkpoint, 12 nodes / 96 GPUs total when all three run concurrently. The sampled numerical regression smoke also runs the reducer and Parquet-slicing tests. Add `--validate-full-inputs` to `node.py --smoke` in sampled mode to download/verify the complete dataset and exercise the last partial slice of every task. The 0.56T smoke requires its saved first-batch predictions to match the previous Lambda reference exactly. Each production worker independently profiler-verifies external FA2 before scoring.

Full artifacts contain all 77 merged prediction arrays, all context metrics, the exact input manifest, chunk manifest, execution plan, and node/worker provenance. Raw per-chunk arrays remain in CWS3; HF stores merged arrays instead of thousands of duplicate chunk files. Full runs report measured full-evaluation throughput, not downsampled extrapolations. Complete jobs release allocations automatically.

For an independent final check, `validation_targets.py` extracts compact labels and target bases from the already-staged full inputs on a live allocation. It computes RC targets directly as the complement of reversed original targets, without calling the evaluator's transforms or metrics. `verify_full.py --artifact ARTIFACT_DIR --targets-dir TARGETS_DIR --output validation.json` runs locally with only NumPy/sklearn, checks all raw-array shapes and probabilities, reconstructs every non-SV score, recomputes all 77 context metrics, and verifies every selected strand. `test_verify_full.py` checks both correct fixtures and intentional corruption. The target artifact is under 0.5 MB; no complete input sequences need to be downloaded locally. An Iris `task exec` RPC may time out before a longer CPU diagnostic finishes even with a larger command timeout; inspect its output files before retrying, or launch long diagnostics detached on the existing allocation.

`report_full.py` builds group/species/context tables and a compact post from independently verified downloads and the published leaderboard CSV. Review its explicitly marked interpretation placeholder before posting. `../plot_family_comparison.py` detects full results and removes sampled hatching/labels while retaining the old preview presentation for sampled inputs.

For a direct full-versus-preview numerical check, run `sample_overlap.py --root RUN_ROOT --s3-prefix RESULT_PREFIX` on an already-live allocation. It hash-matches the exact retained preview sequences, binary labels, and SV coordinates onto the full inputs, writes a compact index map, and removes only its own temporary fixture download. Identical duplicate rows map to the first equivalent occurrence. `verify_overlap.py --full FULL_ARTIFACT --sampled PREVIEW_ARTIFACT --mapping INDEX_DIR --sample-manifest PREVIEW_MANIFEST --targets-dir TARGETS_DIR --output overlap.json` then compares all 77 arrays on matching rows and recomputes task metrics on that same subset. This separates any numerical effect of changed batch composition from the effect of evaluating more examples; it does not rerun inference. Its CPU-only regression test checks row reordering and intentional corruption.

## Environment and reproducibility

The pinned image is `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel`, digest `sha256:e97058f7b9b583517643477cc9a0433e54594038efd85ec4abd7836233d626f3`. The remote-only bootstrap creates a system-site-packages venv, reuses image Torch/CUDA, installs `requirements.txt`, and installs the prebuilt FlashAttention 2.8.3.post1 wheel matching Python/Torch/C++ ABI without dependencies. It asserts eight H100s. No NVIDIA packages or model weights are downloaded to the submitter. The observed image uses Python 3.11.12 and cuDNN 9.7.1, whereas the Lambda reference used Python 3.12.3 and cuDNN 9.8.0; both use Torch 2.7.0/CUDA 12.8, Transformers 5.15.1, NumPy 1.26.4, sklearn 1.5.2, BF16, FA2, FP32 A/C/G/T softmax, batch32, TF32, and no KV cache. The image reports Torch's explicit `+cu128` build suffix. Every GPU worker must profiler-confirm the external FA2 kernel before scoring. Tokenizer parallelism is disabled on both paths; this eight-worker-per-node launcher additionally caps OpenMP/MKL at two threads per process to avoid CPU oversubscription.

`checkpoints.py` pins the three model exports; `common.py` retains the original 0.56T sampled reference artifacts (`dfd4d41f0fc70f95cc22081de28a1546412f1bdc`). Each node downloads the selected model and either retained TSV fixtures/reference arrays or full Parquet inputs directly from HF. Model sizes and LFS SHA-256, the manifest hash, and fixture/source SHA-256s are verified. Local staging is rejected if projected disk usage reaches 90%. Credentials are forwarded in the task environment, never in committed code; S3 credentials are ambient in CoreWeave pods.

The original unstratified seed-0 samples are not resampled. Each 10,000-row task is split into contiguous 1,024-row chunks whose boundaries preserve every original 32-example minibatch, including its final partial batch. A deterministic greedy plan balances chunks across `8 × NODES` persistent model processes. Chunk size can be changed with `--chunk-size` but must be a multiple of 32. Completed chunks publish raw predictions before their completion marker. The reducer rejects gaps, overlaps, incorrect worker assignments, and array hash mismatches, then reconstructs original row order and computes metrics over the complete task. It never averages chunk AUROCs/AUPRCs. Motif validity masks use the original helpers. Forward/RC selection remains the maximum of the two completed task-level metrics; SV is unchanged forward reference-versus-mutant scoring. Reported per-chunk metrics are diagnostics only.

## Progress, outputs, and cleanup

```bash
"$IRIS_BIN" --cluster marin job summary /eczech/JOB_NAME
"$IRIS_BIN" --cluster marin job logs /eczech/JOB_NAME --tail --max-lines 30
"$IRIS_BIN" --cluster marin task exec /eczech/JOB_NAME/0 --timeout 60 -- \
  bash -c 'tail -n 5 /tmp/plantcad2-JOB_NAME/gpu000.log'

# Global counts across both nodes (after their environments are ready):
"$IRIS_BIN" --cluster marin task exec /eczech/JOB_NAME/0 --timeout 60 -- \
  /tmp/plantcad2-cw-venv/bin/python \
  /app/zero-shot-leaderboard/evo2_20b/pilot/coreweave/monitor.py JOB_NAME
```

Each node logs progress every 20 seconds; each GPU updates its progress after 25 batches and on chunk boundaries. `workers/gpuNNN/progress.json` is also published to S3. Jobs write only their unique prefix under `s3://marin-us-east-02a/MarinDNA/plantcad2-evals/<run-id>/step-<step>/coreweave/<job-name>/`. Node 0 collects and validates all chunks, computes full-task metrics (plus raw-score reference differences in sampled mode), and uploads a separate verified HF result artifact at `<run-id>/results/step-<step>/coreweave/<job-name>/`. Input manifests and immutable source revisions are retained instead of duplicating source datasets in each artifact. `monitor.py` reads the run's actual total row/forward counts, supporting either mode.

GPU workers are stopped if a sibling fails; failures are marked in S3 and stop peer workers and the reducer. Jobs have no failure retries and a four-hour default safety timeout, configurable with `JOB_TIMEOUT` in seconds. The three parallel four-node full runs use a six-hour safety timeout each. Iris separately defaults to preemption retries: `--max-retries 0` does not disable those. To prevent mixed-attempt artifacts, this runner rejects any task attempt other than `:0` before reading or writing results; after preemption, use a fresh job name rather than resuming. Always verify zero preemptions in the measured job's terminal status. Successful tasks exit, releasing allocations automatically. Task-local staging disappears with the pod. Do not prune shared CoreWeave node caches, and do not remove unrelated local uv environments. This runner never posts to GitHub; publish result posts separately only when the user has authorized publication.

## Verification commands

`test_partition.py` is standard-library-only and runs locally with any existing Python. It checks original minibatch identity, complete coverage, duplicate/gap rejection, and balanced assignments for 1/2/4/8 nodes. `test_aggregation.py` runs in the remote environment and checks array ordering, global rather than averaged chunk AUROC, ambiguous-base motif validity, and full-task strand selection. Before production scoring, node 0 also recomputes all 20 saved Lambda tasks and every recorded forward/RC metric from the pinned reference arrays and requires agreement within `1e-14`.

The 2026-08-31 environment probe `/eczech/plantcad2-cw-env-probe-20260831-v1` and inference smoke `/eczech/plantcad2-cw-smoke-20260831-v1` both succeeded. Both used full H100 nodes at explicit batch priority/user eczech, and exited after validation. The initial pre-submission attempt exceeded Iris's 25-MB bundle limit because the repository contains unrelated large tracked files; the launcher now includes only the evaluator subtree (about 0.1 MB) and excludes historical result assets.

The completed production job `/eczech/plantcad2-cw-056t-lr5e4-n2-20260831-v1` used two H100x8 nodes and reproduced all 20 Lambda task scores plus all 77 raw prediction arrays byte-for-byte. Evaluation took 27m19s, or 31m18s from submission through verified upload and terminal state. Both tasks succeeded with zero failures/preemptions and released their allocations. See the [run report and verified artifact links](../runs/exp472-056t-coreweave-reproduction-20260831.md) and [approved Result 6 post](https://github.com/eric-czech/plantcad2/pull/1#issuecomment-5481049205). Only about 14 MB of final results/provenance was fetched locally; no local CUDA environment was created.

The subsequent three **full-split** LR5e-4/WD0.1 runs used four H100x8 nodes each concurrently (96 GPUs total). All three finished successfully and released all allocations. The .22T/.39T/.56T Composites were 0.684271/0.692732/0.696641; each stage improved 19/20 task rows. Measured scoring took about 1h51–1h53 per checkpoint, at 510–519 forwards/s per 32-GPU job. Independent checks reproduced every recorded metric exactly. Full-versus-preview raw differences were confined to five examples per repeated checkpoint in final batches of five or six rows, with maximum same-subset task-score change below 5.5e-7. See the [full run report, compact comparison, and immutable HF artifacts](../runs/exp472-lr5e4-full-coreweave-20260831.md) and [published Result 7](https://github.com/eric-czech/plantcad2/pull/1#issuecomment-5483172876). The default sampled path remains available; full mode does not overwrite or resample its fixtures.
