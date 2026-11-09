# exp593 evaluation operations

Read this file before each execution pass.
Use the CoreWeave training-sweep skill's persistence, utilization, exact-job dispatch and recovery contracts.
The experiment definitions live in the package and its pinned source manifest, not in scheduling helpers.

## Invariants

Training jobs and source checkpoints are read-only and remain owned by the aug-train session.
Stop if required coordination or direct CoreWeave S3 access is unavailable.
Evaluate only quarter-budget permanent milestones for the two approved run identities.
No historical baseline model reruns and no silent changes to original labels, splits, genomic geometry, scoring semantics or numerical gates.
Preserve fast-fa3 BF16 and fullgraph=True; failures cannot justify falling back to eager execution or weakening validation.
Only this experiment's new PlantCAD2 PR and its single results comment may be updated.
Read live public text before updating it and preserve manual user edits.

## Placement and observation

Use only H100s in whole eight-GPU nodes, Iris user eczech, and batch priority on every root, inherited by all children.
Begin with one node for validation; use up to eight independent nodes for finite productive extraction when validated and supported by fresh available capacity.
No GPUs are retained while waiting for checkpoints.
Use the documented v3 utilization helper immediately before placement.
W&B is authoritative for execution progress; Iris is used for exact dispatch reconciliation, capacity, liveness and concrete startup failures.
Keep one writer per output identity and persist intent before submitting an exact unique Iris name.
Use 30-minute agent heartbeat passes, with earlier startup checks for concrete failures; consider reslicing after one hour without progress and investigate a restart after three hours.
An Iris timeout blocks changes until that exact submission is reconciled.
Iris preemption alone does not justify a duplicate dispatch.

| Cluster | GPU | Nodes | GPUs | State | Reason |
| --- | --- | ---: | ---: | --- | --- |
| cw-rno2a | H100 | 1 | 8 | eligible | Prior pinned provider runtime validated on this hardware; new checkpoint/readouts still require gates |
| cw-us-east-02a | H100 | 1 | 8 | eligible | Same H100 deployment profile; first task requires bounded startup validation |

## State and completion

Local state and the untracked approved plan are in `scratch/exp593-evals/` at the evaluation checkout root.
Use the sweep helper's SQLite transactions; backup a consistent copy before external dispatch and after every mutation.
Immutable backup owner is `s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/evaluations/plantcad2-v1/operations/`.
Upload new UTC/UUID-named backups and verify size and SHA-256 by direct readback.
Recover the newest integrity-checked backup if the local working copy is lost.
An observation helper may list and validate checkpoints but may not autonomously dispatch or recover jobs.
No hosted scheduling tool is exposed in this session, so maintain 30-minute agent decision passes with bounded active-session waits and report any interruption honestly.
Stop and release owned resources after all eight checkpoint evaluations have verified complete results.
There is no authorization to continue beyond those finite evaluations or to hold resources pending unrelated work.
