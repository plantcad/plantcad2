# CWS3 to Hugging Face checkpoint transfer

This is the durable runbook for moving later exp472 checkpoints without routing multi-gigabyte weights through a laptop or a disk-constrained development VM. The proven route is a CPU-only Iris job on CoreWeave: CWS3 → temporary in-cluster disk → private Hugging Face repo → automatic temporary cleanup.

## Do not infer the final checkpoint label

Inspect both `<run>/<date>/hf/` and `<run>/<date>/checkpoints/` from an in-cluster job before copying. Levanter labels the last artifact with the zero-based completed step; a W&B `num_train_steps` value of 206145 corresponded to `step-206144`, not `step-206145`.

The first verified exp472 example was:

```text
s3://marin-us-east-02a/MarinDNA/exp472_plantcad2_baseline/checkpoints/exp472-plantcad2-angiosperm-lr0p0002-wd0p1-v2/2026.08.20/hf/step-206144
→ eczech/marindna-exp472/exp472-plantcad2-angiosperm-lr0p0002-wd0p1-v2/hf/step-206144
```

Copy the existing HF export when one is present. Do not download the much larger native Levanter checkpoint and reconvert it unnecessarily. In this example, the HF export was 4 files / 3,892,739,156 bytes; the native checkpoint was 96 files / 11,678,737,034 bytes.

## Proven job shape

Use the MarinDNA experiment checkout locally so Iris captures the transfer driver with the workspace. Load `HUGGING_FACE_HUB_TOKEN` from `~/.zshrc`; load CoreWeave/Lambda API credentials from `~/oa.env`, but do not forward the CoreWeave object keys. An in-cluster Iris task receives the working S3 credentials through its ambient environment.

```bash
source ~/.zshrc
cd /Users/eczech/tmp/repos/marin-dna-exp472
uv run iris --cluster marin job run --enable-extra-resources --target-cluster cw-us-east-02a --priority batch --user eczech --job-name exp472-copy-final-hf-step206144 --cpu 4 --memory 16GB --disk 32GB -e MARIN_PREFIX s3://marin-us-east-02a/marin -e HUGGING_FACE_HUB_TOKEN "$HUGGING_FACE_HUB_TOKEN" -- python experiments/exp472_plantcad2_baseline/exp472_copy_final_hf.py
```

The transfer driver should enforce all of the following:

- Confirm the authenticated Hugging Face account is `eczech` and `eczech/marindna-exp472` is private.
- Refuse to overwrite a non-empty destination prefix.
- Enumerate the source recursively and require `config.json` plus recognized model weights.
- Stage only one HF checkpoint under `tempfile.TemporaryDirectory`.
- Reject a transfer projected to cross 90% disk usage, leaving an explicit safety margin.
- Download objects concurrently and verify every local byte count against CWS3 metadata.
- Set `HF_XET_HIGH_PERFORMANCE=1`, upload the directory to `<wandb-run-id>/hf/<step>`, then compare the exact destination file set and total bytes.
- Let the temporary directory delete itself whether the upload succeeds or fails.

The successful job was `/eczech/exp472-copy-final-hf-step206144`. It took about four minutes end to end, including roughly two minutes to download from CWS3 and two minutes to upload to Hugging Face. Staged disk usage peaked at 14.14%. The verified Hugging Face commit is <https://huggingface.co/eczech/marindna-exp472/commit/fe3b167ff53d4d5d0fefbc92652acbef3b801831>.

The later post-cooldown transfer used the same route and established the current Iris CLI shape. Inventory job `/eczech/exp472-inspect-lr0p0005-s01-final-v3` found matching final native and HF artifacts at `step-371065`; transfer job `/eczech/exp472-copy-final-hf-step371065` copied the 4-file / 3,892,739,156-byte HF tree in about 70 seconds and verified commit <https://huggingface.co/eczech/marindna-exp472/commit/e56696e49dbc4c5d904507983df901fbe9d6d32d>. The exact route was:

```text
s3://marin-us-east-02a/MarinDNA/exp472_plantcad2_baseline/checkpoints/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1/2026.08.25/hf/step-371065
→ eczech/marindna-exp472/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1/hf/step-371065
```

Current Iris requires `--enable-extra-resources` for `--cpu`, `--memory`, and `--disk`. Do not pass the obsolete `--extra cpu`; current `--extra` values are accelerator-oriented. The reusable command is:

```bash
source ~/oa.env
source ~/.zshrc
cd /Users/eczech/tmp/repos/marin-dna-exp472
export EXP472_RUN_ID="RUN_ID_HERE"
export EXP472_RUN_DATE="YYYY.MM.DD"
export EXP472_STEP="STEP_NUMBER"
uv run iris --cluster marin job run --enable-extra-resources --target-cluster cw-us-east-02a --priority batch --user eczech --job-name "exp472-copy-final-hf-step${EXP472_STEP}" --cpu 4 --memory 16GB --disk 32GB -e MARIN_PREFIX s3://marin-us-east-02a/marin -e HUGGING_FACE_HUB_TOKEN "$HUGGING_FACE_HUB_TOKEN" -e EXP472_RUN_ID "$EXP472_RUN_ID" -e EXP472_RUN_DATE "$EXP472_RUN_DATE" -e EXP472_STEP "$EXP472_STEP" -- python experiments/exp472_plantcad2_baseline/exp472_copy_final_hf.py
```

The two stage-s02 transfers on 2026-08-29 reused this route. Both final exports were at `<run-id>/2026.08.28/hf/step-535985`, each 4 files / 3,892,739,156 bytes. `lr0p0001-wd0p2-train-s02-v1` completed in 33.72 seconds at HF revision `dcc95e936a05322c6312589670bf21944cd7ea65`; `lr0p0005-wd0p1-train-s02-v1` completed in 27.34 seconds at revision `2972ca5abb575ccb9878d2525aafd396e6b73d7c`. No weights passed through the laptop or a development VM. Include the run/hyperparameters in the Iris job name and local/remote evaluation directories when multiple runs share the same final step, to avoid collisions. `iris job summary <job>` confirms terminal state; there is no `iris job status` subcommand. A quiet federated log stream does not imply the job stalled: verify its summary and the uploaded HF inventory.

## Verification checklist

The matching LR5e-4/WD0.1 0.22T export was copied on 2026-08-31 by CPU-only batch job `/eczech/exp472-copy-lr5e4-022t-final-hf-20260831`. Source: `s3://marin-us-east-02a/MarinDNA/exp472_plantcad2_baseline/checkpoints/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2/2026.08.20/hf/step-206144`. Destination: `eczech/marindna-exp472/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2/hf/step-206144`, verified [HF commit 4c71ba8](https://huggingface.co/eczech/marindna-exp472/commit/4c71ba81544b93b8a0a0f878b44ac51d1ebb186f), 4 files / 3,892,739,156 bytes. The same transfer script ran in the lightweight `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` image with `--no-sync` and `uv run --no-project --with s3fs --with huggingface-hub`; workspace bundling included only the experiment folder. Download took about 8 seconds, upload/verification about 2m15s, and temporary staging was cleaned automatically. No local dependencies or model downloads were needed.

1. Inventory the actual CWS3 artifact names and sizes in-cluster.
2. Confirm the exact source and destination before launch when the checkpoint is ambiguous.
3. Require source file count and bytes to equal staged file count and bytes.
4. Require the uploaded destination file set and total bytes to equal the staged tree.
5. Confirm the Iris job is terminal and the temporary staging directory was removed.
6. Record the source, destination, Iris job name, file count, byte total, and Hugging Face commit in the evaluation result.

If direct laptop access to `https://cwobject.com` returns `InvalidAccessKeyId`, do not cycle through other credentials or route the data through the laptop. Use the in-cluster Iris path above.
