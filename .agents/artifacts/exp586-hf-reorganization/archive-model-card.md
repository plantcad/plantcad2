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

# MarinDNA exp586 archive

BF16 endpoint exports from experiment 586:

| Directory | Initialization and LR | DNA training tokens | Checkpoint | Standalone model |
| --- | --- | ---: | --- | --- |
| `pretrained-lrm0p5-seed0/` | Language-pretrained, 0.5× | 216,158,699,520 | `step-412290` | — |
| `pretrained-lrm1-seed0/` | Language-pretrained, 1× | 216,158,699,520 | `step-412290` | [216B](https://huggingface.co/plantcad/marin-dna-exp586-1.5B-60M-216B) |
| `pretrained-lrm2-seed0/` | Language-pretrained, 2× | 216,158,699,520 | `step-412290` | — |
| `scratch-lrm1-seed0/` | Random, 1× | 216,158,699,520 | `step-412290` | — |
| `pretrained-lrm1-seed0-1t/` | Language-pretrained, 1× | 1,000,000,192,512 | `step-1907349` | [1T](https://huggingface.co/plantcad/marin-dna-exp586-1.5B-60M) |

The 216B-token endpoints have trained for 10 DNA epochs; the 1T-token endpoint has trained for approximately 46.3.
The 1T export is an unchanged copy of the standalone model; per-model receipts preserve the source checkpoints and file identities.

[Experiment issue](https://github.com/Open-Athena/marin-dna/issues/586) · [Evaluation results](https://github.com/plantcad/plantcad2/pull/22)
