## Result 11 — Maize allele-frequency frozen-embedding probes

The three MarinDNA primary-lineage checkpoints were compared using the same deterministic held-out variants at 8,192-bp context. A single global ridge probe was fitted separately for each checkpoint on FP32 forward/reverse-complement-averaged final-layer embeddings. The planned primary representation is the whole-window mean used by the MarinDNA probe path; a variant-token representation was retained as an exploratory comparison because single-position probes are also common.

![Maize allele-frequency zero-shot and frozen-embedding comparison](../blob/8b8dde178ffba9ae567dcfff871cbd7d1ce55e51/zero-shot-leaderboard/evo2_20b/pilot/runs/assets/marindna-lr5e4-maize-af-probe-20260904.png?raw=true)

**Main takeaway:** probing helps noncoding variants disproportionately. Every noncoding consequence improves over zero-shot with both probe readouts at all three checkpoints, while coding variants gain little from the variant-token probe and start/stop and missense scores decline with whole-window pooling. This suggests that coding constraint is already comparatively accessible through the zero-shot likelihood, whereas substantial noncoding AF signal exists in the hidden states but is not expressed well by that generative readout; probing changes signal accessibility rather than improving all variant classes uniformly.

This view compares the training-scale trajectory within each consequence without subtracting the zero-shot score. The side rail distinguishes coding (ochre) from noncoding (teal) consequences in both figures.

![Raw maize allele-frequency Spearman scores by consequence, readout, and training scale](../blob/8b8dde178ffba9ae567dcfff871cbd7d1ce55e51/zero-shot-leaderboard/evo2_20b/pilot/runs/assets/marindna-lr5e4-maize-af-probe-raw-by-consequence-20260905.png?raw=true)

The zero-shot values here are recomputed on the exact held-out test rows and should not be compared numerically to the 94,075-row pooled values in Result 8.

<details>
<summary>Overall scores and bootstrap intervals</summary>

| Checkpoint | Zero-shot ρ [95% CI] | Whole-window probe ρ [95% CI] | Whole-window Δρ vs zero-shot [95% CI] | Variant-token probe ρ [95% CI] | Variant-token Δρ vs zero-shot [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MarinDNA 1B 0.22T | 0.1430 [0.1301, 0.1551] | 0.1801 [0.1679, 0.1920] | +0.0371 [+0.0238, +0.0507] | 0.1912 [0.1796, 0.2029] | +0.0482 [+0.0335, +0.0619] |
| MarinDNA 1B 0.39T | 0.1394 [0.1269, 0.1515] | 0.1801 [0.1691, 0.1913] | +0.0407 [+0.0283, +0.0525] | 0.1916 [0.1801, 0.2022] | +0.0521 [+0.0382, +0.0643] |
| MarinDNA 1B 0.56T | 0.1415 [0.1285, 0.1545] | 0.1803 [0.1687, 0.1913] | +0.0387 [+0.0256, +0.0510] | 0.1950 [0.1831, 0.2056] | +0.0535 [+0.0403, +0.0671] |

| Readout | 0.22T→0.56T Δρ | Paired block-bootstrap 95% CI |
| :--- | ---: | :--- |
| Zero-shot LLR | -0.0015 | [-0.0087, +0.0058] |
| Whole-window probe | +0.0002 | [-0.0076, +0.0080] |
| Variant-token probe | +0.0038 | [-0.0017, +0.0098] |

</details>

<details>
<summary>Held-out scores by consequence</summary>

| Consequence | Test n | 0.22T zero-shot | 0.22T whole | 0.22T variant | 0.39T zero-shot | 0.39T whole | 0.39T variant | 0.56T zero-shot | 0.56T whole | 0.56T variant |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Start/stop | 950 | 0.3183 | 0.2630 | 0.3200 | 0.3060 | 0.2622 | 0.3162 | 0.3051 | 0.2820 | 0.3161 |
| Missense | 2,696 | 0.2591 | 0.2291 | 0.2583 | 0.2629 | 0.2210 | 0.2537 | 0.2646 | 0.2181 | 0.2557 |
| Synonymous | 2,710 | 0.1805 | 0.1864 | 0.1906 | 0.1674 | 0.1905 | 0.1891 | 0.1734 | 0.1824 | 0.1960 |
| Non-coding exon | 2,731 | 0.1747 | 0.1875 | 0.2006 | 0.1641 | 0.1938 | 0.2066 | 0.1659 | 0.1992 | 0.2148 |
| 5′ UTR | 2,776 | 0.1415 | 0.1665 | 0.1742 | 0.1356 | 0.1664 | 0.1740 | 0.1386 | 0.1721 | 0.1786 |
| Downstream | 2,764 | 0.1183 | 0.1658 | 0.1715 | 0.1017 | 0.1637 | 0.1775 | 0.1053 | 0.1643 | 0.1762 |
| Intergenic | 2,839 | 0.0952 | 0.1861 | 0.2019 | 0.1067 | 0.1926 | 0.2044 | 0.1054 | 0.1930 | 0.2219 |
| Splice region | 2,709 | 0.0867 | 0.1552 | 0.1368 | 0.1057 | 0.1499 | 0.1391 | 0.1054 | 0.1404 | 0.1511 |
| Upstream | 2,742 | 0.0988 | 0.1872 | 0.2144 | 0.0960 | 0.1816 | 0.1991 | 0.1004 | 0.1784 | 0.2034 |
| Intron | 2,669 | 0.0783 | 0.1564 | 0.1335 | 0.0697 | 0.1675 | 0.1447 | 0.0617 | 0.1586 | 0.1318 |
| 3′ UTR | 2,792 | 0.0584 | 0.1260 | 0.1303 | 0.0612 | 0.1181 | 0.1293 | 0.0693 | 0.1290 | 0.1255 |

These are within-consequence correlations from one global probe per checkpoint; no consequence-specific probes were fitted. Per-cell bootstrap intervals are retained in each `result.json` artifact.

</details>

<details>
<summary>Protocol, verification, and artifacts</summary>

- The fixed 94,075-variant sample was split with seed 0 by 1 Mb genomic blocks, stratified jointly by consequence and within-consequence AF decile. The split has 65,565 train, 28,378 test, and 132 excluded training rows whose 8,192-bp windows overlap a test window. The split hash is `c37e713eb0449af48f99a14de5e3c86fab8cb5504c4872ccfd24e8988f0b89a0` for all checkpoints.
- Each frozen-model feature is `[emb_ref, emb_alt − emb_ref]`. Whole-window embeddings mean-pool all 8,192 DNA-token final states; variant-token embeddings use the allele state at forward index 4,096 / reverse-complement index 4,095. REF/ALT and forward/reverse-complement embeddings are averaged in FP32.
- `StandardScaler → Ridge` predicts raw AF. Ridge α was selected independently per checkpoint/representation from `10^-4…10^8` by five-fold train-only stratified genomic-block CV maximizing Spearman correlation. No optimum showed truncation risk. Test rows were used only for the reported evaluation.
- Intervals use 1,000 paired seed-0 genomic-block bootstrap replicates. Row-level independent recomputation exactly reproduced all reported correlations; all prediction arrays are finite and all input/split inventories match.
- BF16 inference used profiler-verified external FlashAttention-2. Prefix-cached and uncached embeddings agreed within `4.8e-7` maximum absolute difference for whole-window means and exactly at variant-token states in the live parity check.
- Three batch-priority jobs ran in `cw-us-east-02a`, each using four H100x8 nodes. Wall times were 20m00s, 19m15s, and 18m33s for 0.22T, 0.39T, and 0.56T. All allocations are released.

Artifacts contain the split, fitted probes, row-level zero-shot/probe predictions, bootstrap metrics, environment/FA2 verification, and provenance:

- [MarinDNA 1B 0.22T](https://huggingface.co/plantcad/marindna-exp472/tree/69656cc98d0ad0d17203e1d18b213c47a9d8187e/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-v2/results/step-206144/coreweave/plantcad2-maize-af-probe-022t-20260904-v1)
- [MarinDNA 1B 0.39T](https://huggingface.co/plantcad/marindna-exp472/tree/69656cc98d0ad0d17203e1d18b213c47a9d8187e/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s01-v1/results/step-371065/coreweave/plantcad2-maize-af-probe-039t-20260904-v1)
- [MarinDNA 1B 0.56T](https://huggingface.co/plantcad/marindna-exp472/tree/69656cc98d0ad0d17203e1d18b213c47a9d8187e/exp472-plantcad2-angiosperm-lr0p0005-wd0p1-train-s02-v1/results/step-535985/coreweave/plantcad2-maize-af-probe-056t-20260904-v1)

</details>
