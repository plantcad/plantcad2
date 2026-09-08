# Lambda execution notes

This is the short path used for the exp472 Qwen3 evaluation. It deliberately contains no credentials. The local API key is `LAMBDA_KEY` in `~/oa.env`; never print it or commit it. Lambda's API currently limits launches to one per 12 seconds and five per minute.

The durable CWS3 → Hugging Face transfer procedure is in [`CHECKPOINT_TRANSFER.md`](CHECKPOINT_TRANSFER.md). Check that runbook before locating or moving another checkpoint; in particular, inventory the actual final artifact label instead of deriving it from W&B `num_train_steps`.

## Provision safely

Use `https://cloud.lambda.ai/api/v1` with `Authorization: Bearer
${LAMBDA_KEY}`. Before launch, query `/instance-types`, `/regions`, and
`/ssh-keys`; do not assume capacity or key names. A minimal launch body is:

```json
{
  "region_name": "us-southeast-1",
  "instance_type_name": "gpu_1x_h100_sxm5",
  "ssh_key_names": ["eczech-laptop"],
  "file_system_names": [],
  "quantity": 1,
  "name": "exp472-plantcad2-<purpose>"
}
```

Treat `regions_with_capacity` as advisory, not authoritative. On 2026-08-24 both the API field and a filtered all-SKU query reported no GPU capacity while the console showed the 2×H100 SKU; one direct launch in the previously successful `us-southeast-1` region immediately succeeded. Before any direct attempt, list `/instances` and reject an existing instance with the exact evaluation name plus `eczech-laptop` key. After an accepted launch response, record its returned ID and do not retry merely because it has not appeared in `/instances` yet.

POST it to `/instance-operations/launch`, then poll `/instances` until the
returned ID is `active` and has an IP. The full instance response contains a
Jupyter token, so filter status output rather than logging the response:

```bash
curl -fsS -H "Authorization: Bearer ${LAMBDA_KEY}" \
  https://cloud.lambda.ai/api/v1/instances |
  jq '[.data[] | {id,name,ip,status,ssh_key_names,region:.region.name,type:.instance_type.name}]'
```

Always identify our rental by exact instance ID, descriptive name, and the
`eczech-laptop` SSH key. Other keys, such as `tim-id_rsa`, indicate someone
else's instance and must not be touched.

## Bootstrap and run

Connect with `~/.ssh/id_ed25519`, copy the checkout with `rsync`, and create a
venv that reuses the Lambda image's CUDA-enabled PyTorch:

```bash
python3 -m venv --system-site-packages ~/plantcad2/.venv
uv pip install --python ~/plantcad2/.venv/bin/python \
  -r ~/plantcad2/zero-shot-leaderboard/evo2_20b/pilot/requirements.txt
```

The requirement caps on NumPy, SciPy, and scikit-learn are intentional: newer
versions were ABI-incompatible with the image's system PyTorch stack. Validate
one finite CUDA forward pass before the long run.

For FlashAttention, first install `packaging`, `ninja`, `wheel`, and `einops`, then
install `flash-attn` with both `--no-build-isolation` and `--no-deps`. Without
`--no-deps`, uv can silently install a different Torch/CUDA stack inside the venv.
Print and compare `torch.__version__` and `torch.version.cuda` before and after. A
Transformers attention setting is not sufficient verification: profile one real
8,192-token forward and require a CUDA event such as `flash_fwd_kernel` or
`aten::_scaled_dot_product_flash_attention`.

Download the converted checkpoint from the public
`plantcad/marindna-exp472` repo, then run:

```bash
PYTHON=~/plantcad2/.venv/bin/python \
MODEL=<local-hf-checkpoint> \
MODEL_LABEL=<checkpoint-label> \
SAMPLE_DIR=<persistent-seed0-samples> \
OUTPUT_DIR=<checkpoint-specific-results> \
~/plantcad2/zero-shot-leaderboard/evo2_20b/pilot/run_sampled_leaderboard.sh
```

For the recommended two-H100, 10,000-example-per-row path, use `run_recommended_leaderboard_2gpu.sh`. It runs one persistent model process per GPU, divides whole leaderboard rows between the workers, and requires a profiler-confirmed FlashAttention kernel on both GPUs before scoring.

Each worker atomically refreshes `workers/gpuN/progress.json` every 25 batches, about every 800 sequences or 47 seconds for this model. Poll a live two-GPU run without parsing worker logs:

```bash
python zero-shot-leaderboard/evo2_20b/pilot/monitor_recommended.py --compact <output-dir>
```

The compact line reports overall estimated completion, ETA, current task and phase, within-phase counts, and completed-task counts for both GPUs. Omit `--compact` to get JSON including all completed task scores. Progress fractions are estimates until the merged result is written; validation must still use the final `result.json`.

Keep `SAMPLE_DIR` unchanged when comparing later checkpoints. For this roughly
1B model, batch 4 for causal tasks and batch 2 for SV sustained about 16.7
sequences/second; 2,800 examples for each of 20 rows finished in about two
hours on one H100 SXM5.

## Preserve, verify, terminate

Before teardown:

1. Upload samples, manifests, summaries, and comparisons to the checkpoint's
   `results/` path in the HF repo.
2. Verify the remote file count and total bytes with `HfApi.list_repo_tree`.
3. `rsync` the result directory back to local gitignored `scratch/`.
4. Re-list Lambda instances and resolve the exact evaluation instance ID.
5. POST only that ID to `/instance-operations/terminate`:

```json
{"instance_ids": ["<exact-evaluation-instance-id>"]}
```

Filter the termination response to non-secret fields such as `id`, `name`, `status`, and `instance_type.name`; like the instance-list response, it can contain a Jupyter token. Finally poll `/instances` until that ID disappears, while confirming unrelated instances remain active. Billing ends when the instance is terminated.

The 2026-08-24 final-checkpoint run used `gpu_2x_h100_sxm5` in `us-southeast-1`, scored 10,000 examples for each of 20 rows in 3:17:39, and sustained 33.75 aggregate sequences/s. Provisioning, sampling, checkpoint download, smoke validation, evaluation, artifact upload, and verification kept the rental active for about 3 hours 41 minutes. The exact evaluation instance was then terminated and the unrelated account instance was explicitly rechecked.

The 2026-08-27 MarinDNA 1B 20E run reused the exact materialized seed-0 fixtures, finished the same 400,000 forwards in 3:17:34, and sustained 33.76 aggregate sequences/s. Its complete 42-file / 1,734,923,623-byte artifact was uploaded and verified at Hugging Face commit `2090e4c6c52393b5b8626ea9be70205c79935d53` before exact instance `8b7e96189c8644c2acae5efa6d9b8dd4` was terminated; its disappearance from `/instances` was verified and the account was empty.

Official API reference: <https://docs.lambda.ai/public-cloud/cloud-api/>

The 2026-08-29/30 stage-s02 comparison evaluated two different final checkpoints sequentially on the same 2×H100 VM. Keep a distinct run-specific root when checkpoint step numbers coincide: these both ended at step535985. Reuse the venv and hardlink the already-verified fixtures into the second artifact tree; download only the second HF checkpoint. Start the second GPU run only after the first merged result completes. The two runs took 3:21:52 and 3:21:43 at 33.02 and 33.05 aggregate forwards/s. Both complete artifact trees were exactly verified on HF and copied locally before instance `91964762e12a4bc68ecd9abc72be5418` was terminated at about 04:25 EDT; its disappearance and an empty account were verified. The rental lasted about seven hours. See [the paired result note](runs/exp472-train-s02-step-535985-seed0-n10000.md).
