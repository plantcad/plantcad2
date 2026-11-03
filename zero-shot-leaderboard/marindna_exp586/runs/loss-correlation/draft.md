## DNA validation loss versus downstream evaluations

Among the three dense checkpoints and two 60.6M-active MoEs, lower validation loss tracks higher **Composite** performance: **Pearson r = -0.984; Spearman ρ = -0.900** (five models). Maize allele frequency is shown separately and is excluded from Composite.

Lower loss tracks better conservation, masked-motif and structural-variant scores; it does not consistently predict better core/non-core or maize allele-frequency results.

![Validation loss versus Composite and six task groups, excluding the 481M-active MoEs](loss-correlation-without-481m.png)

<details>
<summary>Including the 481M-active MoEs (all seven models)</summary>

![Validation loss versus Composite and six task groups, all seven models](loss-correlation.png)

With all seven models: Composite **Pearson r = -0.987; Spearman ρ = -0.929**.

</details>

At an assumed loss of **0.88**, the [1T-token d768 language-initialized run](https://wandb.ai/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1) projects to **Composite ≈ 0.742** (five-model fit: Composite ≈ 1.23262 − 0.55722 × loss; extrapolated, not evaluated).

| ID | MarinDNA model (total / active parameters) | Initialization | DNA tokens | Vocabulary | Validation loss | Training run |
| --- | --- | --- | ---: | ---: | ---: | --- |
| D216 | Dense 0.97B / 0.97B | Scratch | 216.16B | 7 | 0.981251 | [W&B](https://wandb.ai/plantcad/marin-dna/runs/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2) |
| D389 | Dense 0.97B / 0.97B | Scratch | 389.09B | 7 | 0.970131 | [W&B](https://wandb.ai/plantcad/marin-dna/runs/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1) |
| D562 | Dense 0.97B / 0.97B | Scratch | 562.02B | 7 | 0.965427 | [W&B](https://wandb.ai/plantcad/marin-dna/runs/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1) |
| M481-L | MoE 11.53B / 481.3M | Language → DNA | 11.24B | 128,256 | 1.034050 | [W&B](https://wandb.ai/plantcad/marin-dna/runs/exp582-plantcad2-d1536-pretrained-lrm1-seed0-v1) |
| M481-S | MoE 11.14B / 481.3M | Scratch | 11.24B | 8 | 1.040699 | [W&B](https://wandb.ai/plantcad/marin-dna/runs/exp582-plantcad2-d1536-random-lrm2-seed0-v1) |
| M61-L | MoE 1.59B / 60.6M | Language → DNA | 216.16B | 128,256 | 0.946216 | [W&B](https://wandb.ai/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-v1) |
| M61-S | MoE 1.39B / 60.6M | Scratch | 216.16B | 8 | 0.948175 | [W&B](https://wandb.ai/eric-czech/marin/runs/exp586-plantcad2-d768-scratch-lrm1-seed0-v1) |

MoE active counts exclude input/output embeddings; total counts include them. S = scratch; L = language-initialized; D = dense. The numbers in dense IDs indicate billions of DNA training tokens.

The analysis uses the historical **8K Natural** scoring convention. For target predictions, Natural uses one flank, reporting the better forward/reverse-complement direction per task; **PoE** combines both flanks’ predictions with a normalized geometric mean.

Across the four MoEs, 8K beats 2K Natural by **+0.0084** on the matched three-group mean. Switching from Natural to PoE at 8K changes it by **-0.0010**: conservation and motifs improve, while core/non-core declines.

<details>
<summary>Score definitions and readout choices</summary>

Composite is the equal-weight mean of four task-group averages: Conservation (3 AUROCs), Masked motif (8 whole-motif accuracies), Core/non-core (8 AUROCs) and Structural variant (1 AUPRC). Each group averages its tasks equally. This reconstructs the historical four-group definition; the three-group PoE composites from later posts are not substituted.

Both forward and reverse-complement passes process the window, but causal masking makes each target prediction depend on only its preceding flank. Natural retains the larger task-level metric from either direction where both are available; structural variants have one causal scoring form. This direction selection uses evaluation labels, as in the historical dense results. PoE instead combines the aligned target probabilities as $p_{\mathrm{PoE}}(b) \propto \sqrt{p_{\mathrm{left}}(b)p_{\mathrm{right}}(b)}$, normalized over A/C/G/T—equivalently, averaging log probabilities. At 8K, PoE minus Natural averaged over the four MoEs is Conservation: **+0.0202**; Masked motif: **+0.0088**; Core/non-core: **-0.0319**. Structural-variant PoE was not computed, so the matched readout comparison above uses three groups only.

For Natural, 8K minus 2K averaged over the four MoEs is Conservation: **+0.0013**; Masked motif: **+0.0154**; Core/non-core: **+0.0086**. The same 8K reporting convention is used for every model; context and Natural/PoE settings are not selected per model or task.

Per-model PoE minus Natural at 8K, for the matched three-group mean: M481-L: +0.0021; M481-S: +0.0026; M61-L: -0.0039; M61-S: -0.0047.

Maize AF zero-shot is Spearman correlation between the full causal-suffix allele log-likelihood ratio (averaged across strands) and observed AF on **94,075 variants**. Linear-probe AF is Spearman correlation on **28,378 held-out variants**, using a ridge readout of frozen target-position embeddings averaged across strands. The same genomic-block split and overlap purge are used for every model. These are different evaluation sets; their correlations are separate endpoints, not two interchangeable estimates.

</details>

<details>
<summary>Original losses, replay checks and shared-sample losses</summary>

The common sample is the first **512 logical batches × 128 examples × 8,192 bp = 65,536 windows**, or **536,805,376 next-base targets**. Every raw window was checked against the original dense token cache. This preserves the dense sample; it is not a new genome-balanced sample of the full validation split. No examples are packed, reverse-complemented or given EOS. Loss is mean cross-entropy in nats over positions 2–8,192, including N targets, with the model's full native vocabulary.

Dense runs already used this sample and batch size, so their original native losses are retained. D216 is replayed as a restoration/sampling check. The MoEs originally scored only 512 evenly spaced validation windows; each first replays that sample and then scores the common sample. The physical MoE forward microbatch is 16; sums and target counts are accumulated to the same logical sample, with zero dropped expert assignments required.

- **D216** — original/common 0.98125064; exact-sample replay 0.98125068; `eval/plantcad/loss`.
- **D389** — original/common 0.97013146; `eval/plantcad/loss`.
- **D562** — original/common 0.96542716; `eval/plantcad/loss`.
- **M481-L** — original 1.03529936; original-sample replay 1.03530520; common-sample 1.03405043 (common − original -0.001249); `eval/loss`.
- **M481-S** — original 1.04119466; original-sample replay 1.04120194; common-sample 1.04069893 (common − original -0.000496); `eval/loss`.
- **M61-L** — original 0.94750627; original-sample replay 0.94750318; common-sample 0.94621596 (common − original -0.001290); `eval/loss`.
- **M61-S** — original 0.94914071; original-sample replay 0.94913985; common-sample 0.94817529 (common − original -0.000965); `eval/loss`.

Using the original model-specific losses throughout gives Composite r = -0.989, ρ = -0.929, before matching the MoE sample to the dense sample.

A stale `eval/inputs/plantcad-angiosperm-validation-path-only/loss` value on D216 is excluded. Its active `eval/plantcad/loss` agrees with the final reported dense loss. MoE losses use `eval/loss`. Final metric history is checked against the checkpoint's physical update counter; resumed W&B history steps need not equal that counter.

</details>

<details>
<summary>Sensitivity checks: vocabulary, readout and context</summary>

Input tokenization is one token per nucleotide for every model. Dense models have 7 vocabulary entries, scratch MoEs 8, and language-initialized MoEs retain 128,256. The main figure uses full-vocabulary loss and marks these differences by shape.

The four-base check conditions probabilities on A/C/G/T and uses only canonical-base targets. To avoid rerunning the other dense checkpoints, its correlation comparison is restricted to the **same four MoEs** in both versions: Composite/native r = -0.996, ρ = -0.600; Composite/four-base r = -0.996, ρ = -0.600. This is not a seven-model, vocabulary-controlled estimate.

- **M481-L** — full vocabulary/all targets 1.034050; full vocabulary/A-C-G-T targets 1.035599; four-base conditional/A-C-G-T targets 1.035553.
- **M481-S** — full vocabulary/all targets 1.040699; full vocabulary/A-C-G-T targets 1.042258; four-base conditional/A-C-G-T targets 1.042211.
- **M61-L** — full vocabulary/all targets 0.946216; full vocabulary/A-C-G-T targets 0.947630; four-base conditional/A-C-G-T targets 0.947583.
- **M61-S** — full vocabulary/all targets 0.948175; full vocabulary/A-C-G-T targets 0.949595; four-base conditional/A-C-G-T targets 0.949548.

Replacing the three eligible MoE task groups with PoE while retaining Natural structural-variant scores gives Composite r = -0.982, ρ = -0.964. This hybrid is a sensitivity check, not the primary score.

Using 2K instead of 8K Natural zero-shot AF for the MoEs increases their mean AF Spearman by +0.0076; the seven-model loss/AF relationship becomes r = -0.918, ρ = -0.571. Dense AF retains its historical 8K setting, so this mixed-context result is secondary.

The Composite projection for the [1T-token run in #586](https://github.com/Open-Athena/marin-dna/issues/586#issuecomment-6544696578) uses a least-squares fit to the five measured checkpoints, excluding the 481M-active MoEs. Their losses span **0.946–0.981**. The assumed loss of **0.88** lies outside that range and is not included in the fit or correlations.

</details>

<details>
<summary>Checkpoint and execution provenance</summary>

Historical downstream results are copied from [exp472](https://github.com/plantcad/plantcad2/pull/20), the [exp582 archive](https://github.com/plantcad/plantcad2/tree/351e50a114c26635d5d921c5dd2e88dd9bccdfcf/zero-shot-leaderboard/marindna_exp586/historical/exp582-final) and [exp586](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6366361125). All task scores and probe fits are reused; only held-out language-model loss is evaluated here. Parameter counts are approximate and follow the earlier non-embedding active-parameter convention for MoEs.

**D216** — `plantcad/marin-dna/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2`, metric `eval/plantcad/loss`; checkpoint:

`s3://marin-us-east-02a/MarinDNA/exp472_plantcad2_baseline/checkpoints/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2/2026.08.20/checkpoints/step-206144`

[Loss replay/common-sample execution](https://wandb.ai/eric-czech/marin/runs/exp586-loss-corr-d216-final-eval-v1); checkpoint metadata SHA-256 `88f6bb4316a2d8e2302bfa0b0f33aea65347fa4670629e988809fbdc02fa322a`.

**D389** — `plantcad/marin-dna/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1`, metric `eval/plantcad/loss`; checkpoint:

`s3://marin-us-east-02a/MarinDNA/exp472_plantcad2_baseline/checkpoints/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1/2026.08.25/checkpoints/step-371065`

**D562** — `plantcad/marin-dna/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1`, metric `eval/plantcad/loss`; checkpoint:

`s3://marin-us-east-02a/MarinDNA/exp472_plantcad2_baseline/checkpoints/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1/2026.08.28/checkpoints/step-535985`

**M481-L** — `plantcad/marin-dna/exp582-plantcad2-d1536-pretrained-lrm1-seed0-v1`, metric `eval/loss`; checkpoint:

`s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/checkpoints/exp582-plantcad2-d1536-pretrained-lrm1-seed0-v1/2026.10.06/checkpoints/short-linear-v2/step-21440`

[Loss replay/common-sample execution](https://wandb.ai/eric-czech/marin/runs/exp586-loss-corr-m481-l-final-eval-v1); checkpoint metadata SHA-256 `c882f2cedb62266fc33f5dc60d0b83741b72fb506892baee99a70c7cd7efa4d6`.

**M481-S** — `plantcad/marin-dna/exp582-plantcad2-d1536-random-lrm2-seed0-v1`, metric `eval/loss`; checkpoint:

`s3://marin-us-east-02a/MarinDNA/exp582_plantcad2_moe/checkpoints/exp582-plantcad2-d1536-random-lrm2-seed0-v1/2026.10.06/checkpoints/short-linear-v2/step-21440`

[Loss replay/common-sample execution](https://wandb.ai/eric-czech/marin/runs/exp586-loss-corr-m481-s-final-eval-v1); checkpoint metadata SHA-256 `4a1ddbd00f0e204a662c20b431ec68dc72d398ce77022318755758b1df98c347`.

**M61-L** — `eric-czech/marin/exp586-plantcad2-d768-pretrained-lrm1-seed0-v1`, metric `eval/loss`; checkpoint:

`s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/exp586-plantcad2-d768-pretrained-lrm1-seed0-v1/2026.10.08/checkpoints/step-412290`

[Loss replay/common-sample execution](https://wandb.ai/eric-czech/marin/runs/exp586-loss-corr-m61-l-final-eval-v1); checkpoint metadata SHA-256 `275e68e8a521333a4dd5c7bf2aa8f9c79c11517677a8af47ce7abc1aa5004a89`.

**M61-S** — `eric-czech/marin/exp586-plantcad2-d768-scratch-lrm1-seed0-v1`, metric `eval/loss`; checkpoint:

`s3://marin-us-east-02a/MarinDNA/exp586_plantcad2_moe/checkpoints/exp586-plantcad2-d768-scratch-lrm1-seed0-v1/2026.10.08/checkpoints/step-412290`

[Loss replay/common-sample execution](https://wandb.ai/eric-czech/marin/runs/exp586-loss-corr-m61-s-final-eval-v1); checkpoint metadata SHA-256 `737fa99f7998a31ca2a27aacf82ac3f292d9ff62a49742847e4037ad3cd372d9`.

Evaluation used one eight-H100 node per job on CoreWeave, Iris batch priority and user eczech. The runtime is pinned to Marin 0.2.224.dev40027069267, the newest compatible release retaining the checkpoint implementation's ArrayStacked API. Checkpoints are read without updates; native MoE backends and pending router-bias updates are preserved. The saved JSON records input hashes, worker hash, runtime versions, batch sufficient statistics and storage locations.

</details>
