
## Overview
This pipeline assesses the [accessible regions prediction task](../../tasks/lu_2019_ATACseq/README.md). It is based on Eric's implementation, which can be found in the [Open-Athena GitHub repository](https://github.com/Open-Athena/oa-cornell-dna/tree/main/caduceus/experiments/pipelines/finetuning_example).

For this task, experiments were conducted using three learning rates:
- 1e-3
- 1e-4
- 1e-5

These experiments utilized three different PlantCAD2 checkpoints:
- [pcv2-l24-d0768](https://huggingface.co/kuleshov-group/compo-cad2-l24-dna-chtk-c8192-v2-b2-NpnkD-ba240000)
- [pcv2-l24-d1024](https://huggingface.co/kuleshov-group/compo-cad2-l48-dna-chtk-c8k-1t-v2d-b2-lr4e4-fcpffa-ba476838)
- [pcv2-l24-d1536](https://huggingface.co/kuleshov-group/compo-cad2-l48-d1536-dna-chtk-c8k-1t-v1-b2-lr4e4-NzqiLr)

For detailed results, refer to the WANDB project: [PlantCAD2](https://wandb.ai/jingjingzhai/plantCAD2)
