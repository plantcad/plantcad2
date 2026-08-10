#!/usr/bin/env bash
# Submit every PlantCAD2 zero-shot causal eval job. Edit config.sh (MODEL, paths) first.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# The three sweeps are sharded across NODES (default 8) ranks per array element via
# srun; rank 0 gathers the per-rank score shards and computes the metrics. sv_effect
# is ~6 h on one node and is not sharded.
sbatch run_conservation.sub    # array 0-2  : evo_cons      (3 tasks x 4 ctx)   8 nodes, 24 h
sbatch run_recovery.sub        # array 0-7  : motif_acc     (4x2 splits x 4 ctx) 8 nodes, 24 h
sbatch run_core_noncore.sub    # array 0-7  : core_noncore  (4x2 splits x 4 ctx) 8 nodes, 24 h
sbatch run_sv_effect.sub       # single job : sv_effect     (1 task, no ctx sweep) 1 node, 12 h
