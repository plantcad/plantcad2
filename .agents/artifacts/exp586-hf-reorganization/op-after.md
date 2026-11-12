Evaluates the MarinMoE checkpoints from [MarinDNA #586](https://github.com/Open-Athena/marin-dna/issues/586) with the same PlantCAD2 protocol used for exp582.

The workflow covers the original 20 tasks, centered 2K context, maize allele frequency, frozen probes, and inference-time bidirectional readouts. It preserves the existing dataset splits and historical comparisons.

## Results

<!-- results-toc-start -->

- [Result 1 — Scratch versus language-pretrained at 216B DNA tokens](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6366361125)
- [Result 2 — Validation loss versus downstream evaluations](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6604041302)
- [Result 3 — 216B versus 1T: benchmarks and checkpoint probes](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6647191065)

<!-- results-toc-end -->

Models evaluated in the results posts below:

| Model (Hugging Face) | Training run | Initialization and tokenizer | Total parameters | Active parameters | DNA training | Approx. training FLOPs |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| [MarinMoE 1.5B-60M 0.22T (scratch, 1× LR)](https://huggingface.co/marin-dna/marin-dna-exp586/tree/main/scratch-lrm1-seed0) | [W&B](https://forge.coreweave.com/wandb/plantcad/marin-dna/runs/exp586-plantcad2-d768-scratch-lrm1-seed0-v1) | Random; 8-token DNA vocabulary | 1.395B | 60.555M | 216.159B tokens | 1.36×10²⁰ |
| [MarinMoE 1.5B-60M 0.22T (language-pretrained, 1× LR)](https://huggingface.co/plantcad/marin-dna-exp586-1.5B-60M-216B) | [W&B](https://forge.coreweave.com/wandb/plantcad/marin-dna/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-v1) | d768 language checkpoint; 128,256-token vocabulary with character-forced DNA input | 1.592B | 60.555M | 216.159B tokens | 2.64×10²⁰ |
| [MarinMoE 1.5B-60M 1T (language-pretrained, 1× LR)](https://huggingface.co/plantcad/marin-dna-exp586-1.5B-60M) | [W&B](https://forge.coreweave.com/wandb/plantcad/marin-dna/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1) | d768 language checkpoint; 128,256-token vocabulary with character-forced DNA input | 1.592B | 60.555M | 1,000.000B tokens | 1.22×10²¹ |
