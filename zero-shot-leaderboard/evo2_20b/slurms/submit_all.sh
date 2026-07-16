#!/usr/bin/env bash
# Submit every PlantCAD2 zero-shot causal eval job. Edit config.sh (MODEL, paths) first.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

sbatch run_conservation.sub    # array 0-2  : evo_cons          (3 tasks x 4 ctx)
sbatch run_recovery.sub        # array 0-7  : motif_acc         (4 tasks x 2 splits x 4 ctx)
sbatch run_core_noncore.sub    # array 0-7  : core_noncore      (4 tasks x 2 splits x 4 ctx)
sbatch run_sv_effect.sub       # single job : sv_effect         (1 task, no ctx sweep)
