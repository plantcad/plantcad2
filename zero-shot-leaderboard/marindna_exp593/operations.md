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
For a fresh final checkpoint, prefer the full approved fleet once earlier GPU work drains. Choose its source-bound execution layout explicitly; never repartition a running checkpoint. Verify identical job coverage against the original manifest before using a new layout, preserving inference code, microbatches and scientific definitions. The agent chooses the layout; the finite preparation, dispatch and receipt-verification helpers in `scratch/exp593-evals/` materialize and verify it. Their immutable source archive and real-job audit are recorded in SQLite.
Finite CPU-only reductions and probe fits may use batch/eczech Iris tasks without reserving GPUs; keep their environments separate from GPU providers.
The current CoreWeave peer does not admit CPU-only tasks on its GPU-tainted nodes, so finite CPU phases use already-owned H100 containers with separate CPU environments.
Check their W&B identities before releasing an extraction allocation: a completed GPU controller may still host a necessary active probe fit for another checkpoint.
New launchers keep the foreground process alive for registered finite CPU controllers; asynchronous side processes alone do not survive successful parent completion.
Launch CPU phases only on allocations with that foreground lifetime guard, and retain them only until their finite CPU work finishes, without reserving extra GPUs or waiting for future checkpoints.
Prefer younger extraction allocations for the largest fits.
Use the documented v3 utilization helper immediately before placement.
For each independently placed single eight-H100 validation or finite extraction task, accept its documented `--max-age-seconds 300` bound when the federation cache repeatedly exceeds the default 90 seconds, only with peer contact within 30 seconds and at least four times the requested GPU headroom; preserve both actual ages in the dispatch receipt.
This bounded admission exception does not change batch priority, count occupied GPUs as free, or assert that aggregate headroom guarantees a whole-node fit.
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
The active user-requested goal includes continued monitoring through all eight evaluations; a progress reply does not end this assignment.
There is no authorization to continue beyond those finite evaluations or to hold resources pending unrelated work.

## Practical validation before full evaluation

The initial per-window gate and tiny FIM panel do not establish scientific significance. The user has reviewed this and explicitly authorized a larger real-task validation, including ordinary vllm-fa3 and torch-fa3 controls. Finite validation GPU dispatch is authorized; full evaluation remains blocked until the agent reviews evidence that input/target contracts and scientific readouts agree. Preserve the original failed artifacts. Use per-target differences with uncertainty, motif-answer disagreements and task-score differences; do not confuse a tiny rounded-loss difference with meaningful degradation. Keep the backend compiler/provider profiles fixed. Do not silently accept a failed or incomplete comparison as clearance.

## Reviewed text-Q1 validation

The expanded practical review accepts the text25% checkpoint for fixed Fast scoring. It uses completed four-backend real-task predictions,1024 AF rows, suffix-length diagnostics and80 exact-input full-FIM-adapter replay checks. Preserve the earlier failed tiny gate; the new acceptance has its own immutable review receipt, source/panel hashes and caveats in the durable validation owner. AR suffix AF has genuine per-variant numerical sensitivity across Torch backends. Keep its historical definition, report that limitation and do not claim exact native equivalence or interpret small AF improvements as established. Other checkpoints need their own native/export and practical sanity checks before full evaluation.

The user prioritizes practical zero-shot agreement and has directed continued evaluation. Separate the historical zero-shot AF suffix sensitivity from frozen-activation probes; do not treat it as proof that probe activations are invalid. Do not repeat expanded AF follow-up suites at later milestones. Use native export parity and the existing real-task spot checks for each checkpoint, with an explicit agent review of meaningful differences and no new arbitrary acceptance threshold.
