#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
IRIS_BIN="${IRIS_BIN:-iris}"
CLUSTER="${CLUSTER:-cw-us-east-02a}"
NODES="${NODES:-2}"
CHECKPOINT="${CHECKPOINT:-056t}"
EVAL_MODE="${EVAL_MODE:-sampled}"
JOB_TIMEOUT="${JOB_TIMEOUT:-14400}"
JOB_NAME="${JOB_NAME:?Set a unique JOB_NAME}"
case "$CLUSTER" in cw-us-east-02a|cw-rno2a) ;; *) echo 'Only the two approved H100 clusters are allowed.' >&2; exit 1;; esac
case "$NODES" in 1|2|4|8|16|32) ;; *) echo 'NODES must be a supported power of two.' >&2; exit 1;; esac
case "$CHECKPOINT" in 022t|039t|056t) ;; *) echo 'Unknown checkpoint.' >&2; exit 1;; esac
case "$EVAL_MODE" in sampled|full) ;; *) echo 'EVAL_MODE must be sampled or full.' >&2; exit 1;; esac
: "${HUGGING_FACE_HUB_TOKEN:?Set HUGGING_FACE_HUB_TOKEN without printing it}"
cd "$REPO_DIR"
# Explicit on EVERY root submission. No child jobs or inherited-priority defaults.
exec "$IRIS_BIN" --cluster marin job run --target-cluster "$CLUSTER" --priority batch --user eczech \
  --enable-extra-resources --gpu H100x8 --replicas "$NODES" --cpu 32 --memory 256GB --disk 160GB \
  --job-name "$JOB_NAME" --max-retries 0 --timeout "$JOB_TIMEOUT" --no-wait --no-terminate-on-exit --no-sync \
  --task-image 'pytorch/pytorch@sha256:e97058f7b9b583517643477cc9a0433e54594038efd85ec4abd7836233d626f3' \
  --exclude '^(?!zero-shot-leaderboard/evo2_20b/)' --exclude '/runs/' \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin -e HUGGING_FACE_HUB_TOKEN "$HUGGING_FACE_HUB_TOKEN" \
  -e PLANTCAD_CW_CLUSTER "$CLUSTER" -e PLANTCAD_CW_NODES "$NODES" \
  -e PLANTCAD_CHECKPOINT "$CHECKPOINT" -e PLANTCAD_EVAL_MODE "$EVAL_MODE" \
  -- bash zero-shot-leaderboard/evo2_20b/pilot/coreweave/bootstrap.sh "$@"
