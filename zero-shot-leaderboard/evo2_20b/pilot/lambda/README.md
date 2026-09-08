# Lambda execution

The original pilot, sensitivity tests, and two-GPU leaderboard evaluations were developed for Lambda Labs VMs. Their detailed provisioning, environment reuse, monitoring, artifact verification, and termination instructions remain in [LAMBDA.md](../LAMBDA.md).

The Lambda-specific two-GPU launcher is [run_recommended_leaderboard_2gpu.sh](../run_recommended_leaderboard_2gpu.sh); the older single-GPU entrypoints are [run_sampled_leaderboard.sh](../run_sampled_leaderboard.sh) and [run_pilot.sh](../run_pilot.sh). Their established paths are retained so previous commands remain reproducible. These launchers assume a preconfigured local CUDA environment and local model/sample paths; they do not provision Iris jobs.

The scoring implementation, sampling, metric merging, plotting, and HF artifact upload code in the parent folder are shared utilities. CoreWeave-specific environments and orchestration are isolated in [coreweave/](../coreweave/README.md), which calls the same scoring functions but distributes batch-aligned sample chunks across full H100 nodes.
