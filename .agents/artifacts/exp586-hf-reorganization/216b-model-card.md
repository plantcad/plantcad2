---
license: apache-2.0
tags:
- biology
- genomics
- dna
- moe
- safetensors
- levanter
---

# MarinDNA exp586 1.5B-60M — 216B tokens

BF16 export of the language-pretrained d768 MoE after 216B DNA tokens: 1.59B total parameters and approximately 60.6M active parameters per token.

- Training input tokens: `216,158,699,520` (10 DNA epochs)
- Final checkpoint: `step-412290`
- Context length: `8192`
- Vocabulary: the original 128,256-token language vocabulary with character-forced DNA inputs
- [Training run](https://forge.coreweave.com/wandb/plantcad/marin-dna/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-v1)

See [MarinDNA issue #586](https://github.com/Open-Athena/marin-dna/issues/586) and the [PlantCAD2 evaluation results](https://github.com/plantcad/plantcad2/pull/22).

The export is copied unchanged from the [research archive](https://huggingface.co/marin-dna/marin-dna-exp586/tree/4b8537591261ad14204b28a9c830fa7c6e3f7fbd/pretrained-lrm1-seed0).
The [1T-token model](https://huggingface.co/plantcad/marin-dna-exp586-1.5B-60M) is available separately.
