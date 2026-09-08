# Lambda execution notes

This is the short path used for the exp472 Qwen3 evaluation. It deliberately
contains no credentials. The local API key is `LAMBDA_KEY` in `~/.lambda.env`;
never print it or commit it. Lambda's API currently limits launches to one per
12 seconds and five per minute.

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

Download the converted checkpoint from the private
`eczech/marindna-exp472` repo, then run:

```bash
PYTHON=~/plantcad2/.venv/bin/python \
MODEL=<local-hf-checkpoint> \
MODEL_LABEL=<checkpoint-label> \
SAMPLE_DIR=<persistent-seed0-samples> \
OUTPUT_DIR=<checkpoint-specific-results> \
~/plantcad2/zero-shot-leaderboard/evo2_20b/pilot/run_sampled_leaderboard.sh
```

Keep `SAMPLE_DIR` unchanged when comparing later checkpoints. For this roughly
1B model, batch 4 for causal tasks and batch 2 for SV sustained about 16.7
sequences/second; 2,800 examples for each of 20 rows finished in about two
hours on one H100 SXM5.

## Preserve, verify, terminate

Before teardown:

1. Upload samples, manifests, summaries, and comparisons to the checkpoint's
   `results/` path in the private HF repo.
2. Verify the remote file count and total bytes with `HfApi.list_repo_tree`.
3. `rsync` the result directory back to local gitignored `scratch/`.
4. Re-list Lambda instances and resolve the exact evaluation instance ID.
5. POST only that ID to `/instance-operations/terminate`:

```json
{"instance_ids": ["<exact-evaluation-instance-id>"]}
```

Finally poll `/instances` until that ID disappears, while confirming unrelated
instances remain active. Billing ends when the instance is terminated.

Official API reference: <https://docs.lambda.ai/public-cloud/cloud-api/>
