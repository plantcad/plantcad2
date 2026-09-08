#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
IRIS_BIN="${IRIS_BIN:-iris}"
CLUSTER="${CLUSTER:-cw-us-east-02a}"
NODES="${NODES:-4}"
CHECKPOINT="${CHECKPOINT:-056t}"
JOB_TIMEOUT="${JOB_TIMEOUT:-43200}"
JOB_NAME="${JOB_NAME:?Set a unique JOB_NAME}"
case "$CLUSTER" in cw-us-east-02a|cw-rno2a) ;; *) echo 'Only approved H100 clusters are allowed.' >&2; exit 1;; esac
case "$NODES" in 1|2|4|8|16|32) ;; *) echo 'NODES must be a supported power of two.' >&2; exit 1;; esac
case "$CHECKPOINT" in 022t|039t|056t) ;; *) echo 'Unknown checkpoint.' >&2; exit 1;; esac
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN without printing it}"
cd "$REPO_DIR"
exec "$IRIS_BIN" --cluster marin job run --target-cluster "$CLUSTER" --priority batch --user eczech \
  --enable-extra-resources --gpu H100x8 --replicas "$NODES" --cpu 32 --memory 256GB --disk 80GB \
  --job-name "$JOB_NAME" --max-retries 0 --timeout "$JOB_TIMEOUT" --no-wait --no-sync \
  --task-image 'pytorch/pytorch@sha256:e97058f7b9b583517643477cc9a0433e54594038efd85ec4abd7836233d626f3' \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin -e HUGGING_FACE_HUB_TOKEN "$HUGGING_FACE_HUB_TOKEN" \
  -e PLANTCAD_CW_CLUSTER "$CLUSTER" -e PLANTCAD_CW_NODES "$NODES" -e PLANTCAD_CHECKPOINT "$CHECKPOINT" \
  -e PLANTCAD_EVAL_MODE sampled -e PLANTCAD_AF_PROBE_BATCH_SIZE 32 \
  -- bash zero-shot-leaderboard/evo2_20b/pilot/coreweave/bootstrap.sh zero-shot-leaderboard/evo2_20b/pilot/coreweave/allele_frequency_probe/node.py "$@"
