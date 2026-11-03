"""Join archived downstream scores to checkpoint-bound held-out losses.

No downstream inference or probe fitting takes place here. A complete result
requires every requested checkpoint and rejects partial validation artifacts.
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

from exp586_evals.historical import dense_results, verify_exp582_results

GROUPS = ("Conservation", "Masked motif", "Core/non-core", "Structural variant")
AF_ZERO = "Maize AF · zero-shot"
AF_PROBE = "Maize AF · linear probe"
METRICS = ("Composite", *GROUPS, AF_ZERO, AF_PROBE)
SPLIT = "c37e713eb0449af48f99a14de5e3c86fab8cb5504c4872ccfd24e8988f0b89a0"
COLORS = {"dense": "#7355A3", "small": "#008B87", "large": "#D77C28"}
MARKERS = {7: "s", 8: "^", 128256: "o"}
PROJECTED_LOSS = 0.90
PROJECTED_RUN = "https://wandb.ai/eric-czech/marin/runs/exp586-plantcad2-d768-pretrained-lrm1-seed0-1t-v1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def correlation(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) != len(y) or len(x) < 3 or not np.isfinite([x, y]).all():
        raise ValueError("Correlation needs at least three finite paired values")
    return {
        "n": len(x),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_rho": float(spearmanr(x, y).statistic),
    }


def composite_fit(rows: list[dict]) -> dict:
    """Fit only observed Composite scores; the assumed loss is not a data point."""
    if len(rows) != 5 or any(r["plot_family"] == "large" for r in rows):
        raise ValueError(
            "Composite projection requires the five models without 481M MoEs"
        )
    losses = np.asarray([r["loss"] for r in rows], float)
    scores = np.asarray([r["scores"]["Composite"] for r in rows], float)
    slope, intercept = np.polyfit(losses, scores, 1)
    return {
        "method": "ordinary least squares: Composite = intercept + slope * validation_loss",
        "model_codes": [r["code"] for r in rows],
        "slope": float(slope),
        "intercept": float(intercept),
        "observed_loss_range": [float(losses.min()), float(losses.max())],
        "projection": {
            "training_run": PROJECTED_RUN,
            "experiment": "https://github.com/Open-Athena/marin-dna/issues/586#issuecomment-6544696578",
            "assumed_final_loss": PROJECTED_LOSS,
            "composite": float(intercept + slope * PROJECTED_LOSS),
            "is_extrapolation": bool(
                PROJECTED_LOSS < losses.min() or PROJECTED_LOSS > losses.max()
            ),
            "included_in_fit_or_correlations": False,
        },
    }


def downstream(root: Path, records: list[dict]) -> list[dict]:
    dense = dense_results(root)
    verify_exp582_results(root)
    reference_tasks = {r["key"]: r for r in dense["022t"]["tasks"]}
    output = []
    for record in records:
        row = dict(record)
        if record["rung"]:
            saved = dense[record["rung"]]
            scores = {g: saved["groups"][g] for g in GROUPS}
            scores["Composite"] = mean(scores.values())
            scores[AF_ZERO] = saved["allele_frequency"]["100000"]["llr_acgt_avg"][
                "spearman"
            ]
            scores[AF_PROBE] = saved["probes"]["variant_token"]["overall"]["probe"][
                "value"
            ]
            row["readout_sensitivity"] = None
        else:
            directory = root / record["results"]
            provenance = read(directory / "provenance.json")
            for relative, expected in provenance["selected_files"].items():
                payload = (directory / relative).read_bytes()
                if (
                    len(payload) != expected["bytes"]
                    or hashlib.sha256(payload).hexdigest() != expected["sha256"]
                ):
                    raise ValueError("Archived downstream artifact changed")
            if (directory / "model-provenance.json").exists():
                model_provenance = read(directory / "model-provenance.json")
                binding = model_provenance["checkpoint"]
                if (
                    binding["uri"] != record["checkpoint"]
                    or binding["canonical_metadata_digest"]
                    != record["canonical_metadata_sha256"]
                ):
                    raise ValueError("Downstream scores refer to another checkpoint")
                row["total_parameters_approx"] = model_provenance["model"][
                    "total_parameters"
                ]
                row["active_parameters_approx"] = model_provenance["model"][
                    "active_parameters_hero_convention"
                ]
                row["parameter_count_source"] = str(directory / "model-provenance.json")
            saved = read(directory / "result.json")
            eight = saved["profiles"]["context-8192"]
            two = saved["profiles"]["context-2048"]
            natural = eight["tasks"]["groups"]["natural"]
            tasks = eight["tasks"]["rows"]
            if len(tasks) != 20 or {r["key"] for r in tasks} != set(reference_tasks):
                raise ValueError("MoE task inventory differs from historical dense")
            for task in tasks:
                reference = reference_tasks[task["key"]]
                if any(
                    task[k] != reference[k] for k in ("category", "metric", "samples")
                ):
                    raise ValueError("Task metric or membership size differs")
                if task["natural"] != max(
                    v
                    for v in (task["left"], task["reverse_complement"])
                    if v is not None
                ):
                    raise ValueError("Historical best-direction convention differs")
            scores = {
                g: mean(r["natural"] for r in tasks if r["category"] == g)
                for g in GROUPS
            }
            if any(abs(scores[g] - natural[g]) > 1e-12 for g in GROUPS):
                raise ValueError("Saved task-group average differs")
            scores["Composite"] = mean(scores.values())
            if abs(scores["Composite"] - natural["Composite"]) > 1e-12:
                raise ValueError("Saved four-group composite differs")
            if eight["allele_frequency"]["rows"] != 94075:
                raise ValueError("AF zero-shot sample differs")
            scores[AF_ZERO] = eight["allele_frequency"]["scores"]["llr_acgt_avg"][
                "overall"
            ]["spearman"]
            probe = read(directory / "context-8192/probes/result.json")
            if (
                probe["split"]["historical_split_sha256"] != SPLIT
                or probe["split"]["test"] != 28378
            ):
                raise ValueError("AF probe split differs")
            scores[AF_PROBE] = probe["metrics"]["variant_token/strand_average"][
                "overall"
            ]["probe"]["value"]
            poe = eight["tasks"]["groups"]["bidirectional_poe"]
            delta = {g: poe[g] - natural[g] for g in GROUPS[:3]}
            row["readout_sensitivity"] = {
                "poe_minus_natural_8k": delta,
                "natural_8k_minus_2k": {
                    g: natural[g] - two["tasks"]["groups"]["natural"][g]
                    for g in GROUPS[:3]
                },
                "poe_minus_natural_matched_three_group": mean(delta.values()),
                "af_zero_2k_minus_8k": two["allele_frequency"]["scores"][
                    "llr_acgt_avg"
                ]["overall"]["spearman"]
                - scores[AF_ZERO],
                "scores_poe_with_natural_sv": {
                    **scores,
                    **{g: poe[g] for g in GROUPS[:3]},
                    "Composite": mean(
                        [poe[g] for g in GROUPS[:3]] + [natural[GROUPS[3]]]
                    ),
                },
                "af_zero_2k": two["allele_frequency"]["scores"]["llr_acgt_avg"][
                    "overall"
                ]["spearman"],
            }
        row["scores"] = scores
        row["plot_family"] = (
            "dense"
            if row["rung"]
            else ("small" if row["model_config"]["hidden_dim"] == 768 else "large")
        )
        output.append(row)
    if len(output) != 7 or len({r["code"] for r in output}) != 7:
        raise ValueError("Expected exactly seven distinct checkpoint models")
    return output


def validation(row: dict, path: Path) -> dict:
    result = read(path)
    if (
        result["checkpoint"] != row["checkpoint"]
        or result["checkpoint_digest"] != row["canonical_metadata_sha256"]
    ):
        raise ValueError(f"Wrong checkpoint in {path}")
    sample = result["samples"]["common-65536"]
    if sample["examples"] != 65536 or sample["targets"] != 65536 * 8191:
        raise ValueError(f"Incomplete common sample: {path}")
    if sample["dropped_assignments"] != 0 or not np.isfinite(sample["native_loss"]):
        raise ValueError("Nonfinite inference or token drops")
    if not row["rung"]:
        replay = result["samples"]["original-moe-512"]
        if (
            replay["examples"] != 512
            or replay["targets"] != 512 * 8191
            or abs(replay["native_loss"] - row["original_loss"]) > 2e-4
        ):
            raise ValueError("Original loss replay failed")
    return result


def figure(rows: list[dict], output: Path, *, dpi: int = 180) -> None:
    count = len(rows)
    count_label = {5: "Five", 7: "Seven"}.get(count, str(count))
    projected_fit = composite_fit(rows) if count == 5 else None
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#333C46",
            "text.color": "#24313C",
            "svg.hashsalt": "marindna-validation-loss-correlation-v1",
        }
    )
    fig = plt.figure(figsize=(16, 9.6), facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.65, 1, 1],
        left=0.07,
        right=0.975,
        bottom=0.175,
        top=0.77,
        wspace=0.30,
        hspace=0.60 if count == 5 else 0.42,
    )
    axes = [fig.add_subplot(grid[:, 0])]
    axes += [fig.add_subplot(grid[i // 2, i % 2 + 1]) for i in range(6)]
    x = np.asarray([r["loss"] for r in rows])
    offsets = {
        "D216": (10, 2),
        "D389": (10, -9),
        "D562": (10, 4),
        "M481-L": (-12, -14),
        "M481-S": (-12, 12),
        "M61-L": (10, -20),
        "M61-S": (10, 12),
    }
    for index, (ax, metric) in enumerate(zip(axes, METRICS, strict=True)):
        y = np.asarray([r["scores"][metric] for r in rows])
        stats = correlation(x, y)
        fit = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            xx, np.polyval(fit, xx), color="#ADB5BE", lw=1.3, linestyle="--", zorder=1
        )
        for row, yy in zip(rows, y, strict=True):
            ax.scatter(
                row["loss"],
                yy,
                s=100 if index == 0 else 60,
                marker=MARKERS[row["vocab_size"]],
                color=COLORS[row["plot_family"]],
                edgecolors="white",
                linewidths=0.75,
                zorder=3,
            )
            if index == 0:
                dx, dy = offsets[row["code"]]
                ax.annotate(
                    row["code"],
                    (row["loss"], yy),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx > 0 else "right",
                    fontsize=11,
                    fontweight="normal",
                    arrowprops={"arrowstyle": "-", "color": "#8D969F", "lw": 0.7}
                    if row["code"].startswith("M")
                    else None,
                )
        left_margin, right_margin = (
            (0.013, 0.014) if count == 7 else (0.14 * np.ptp(x), 0.15 * np.ptp(x))
        )
        ax.set_xlim(x.min() - left_margin, x.max() + right_margin)
        ax.margins(y=0.23 if index == 0 else 0.18)
        ax.grid(color="#E8ECF0", linewidth=0.8)
        ax.set_axisbelow(True)
        panel_title = metric
        if count == 5:
            if metric == "Composite":
                panel_title = "Composite (excludes AF)"
            elif metric in (AF_ZERO, AF_PROBE):
                panel_title += "\nExcluded from Composite"
        ax.set_title(
            panel_title,
            loc="left",
            fontsize=16
            if index == 0
            else (12 if count == 5 and metric in (AF_ZERO, AF_PROBE) else 13),
            fontweight="bold",
            pad=6 if count == 5 and metric in (AF_ZERO, AF_PROBE) else 10,
        )
        stats_at_top = count == 5 and metric in ("Core/non-core", AF_ZERO)
        stats_text = f"Pearson r = {stats['pearson_r']:+.3f}\nSpearman ρ = {stats['spearman_rho']:+.3f}"
        if index == 0 and projected_fit:
            sign = "−" if projected_fit["slope"] < 0 else "+"
            stats_text += f"\nComposite = {projected_fit['intercept']:.5f} {sign} {abs(projected_fit['slope']):.5f} × loss"
            ax.text(
                0.97,
                0.97,
                f"Extrapolation · 1T d768\nAssumed loss {PROJECTED_LOSS:.2f}\nProjected Composite ≈ {projected_fit['projection']['composite']:.3f}",
                transform=ax.transAxes,
                ha="right",
                multialignment="left",
                va="top",
                fontsize=11.5,
                bbox={"facecolor": "#F5F8FA", "edgecolor": "#CBD5DE", "pad": 6},
            )
        ax.text(
            0.03,
            0.965 if stats_at_top else 0.035,
            stats_text,
            transform=ax.transAxes,
            va="top" if stats_at_top else "bottom",
            fontsize=13 if index == 0 else 10.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 4},
        )
        if index == 0 or index >= 5:
            ax.set_xlabel("Validation loss (nats / base)", labelpad=10, fontsize=12)
        if index == 0:
            ax.set_ylabel("Mean of four task-group scores ↑", labelpad=10)
        else:
            ax.set_ylabel(
                "Spearman ρ ↑"
                if metric in (AF_ZERO, AF_PROBE)
                else "Task-group score ↑",
                fontsize=11,
            )
        ax.tick_params(labelsize=11 if index == 0 else 10.5)
    fig.text(
        0.07,
        0.955,
        "DNA validation loss versus downstream performance",
        fontsize=22,
        weight="bold",
    )
    fig.text(
        0.07,
        0.916,
        f"{count_label} MarinDNA checkpoints · identical 8,192-base held-out windows · fixed 8K downstream readout",
        fontsize=12,
    )
    color_handles = [
        Line2D([], [], color=COLORS[family], lw=4, label=label)
        for family, label in [
            ("dense", "Dense · 0.97B active"),
            ("small", "MoE · 60.6M active (excl. embeddings)"),
            ("large", "MoE · 481M active (excl. embeddings)"),
        ]
        if any(row["plot_family"] == family for row in rows)
    ]
    vocab_handles = [
        Line2D(
            [],
            [],
            marker=m,
            linestyle="none",
            color="#53606B",
            markersize=8,
            label=label,
        )
        for m, label in [
            ("s", "7-token DNA vocab"),
            ("^", "8-token DNA vocab"),
            ("o", "128,256-token language vocab"),
        ]
    ]
    fig.legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(0.064, 0.885),
        frameon=False,
        ncol=3,
        fontsize=10,
    )
    fig.legend(
        handles=vocab_handles,
        loc="upper left",
        bbox_to_anchor=(0.064, 0.851),
        frameon=False,
        ncol=3,
        fontsize=10,
    )
    fig.text(
        0.07,
        0.075,
        "Composite = mean of Conservation, Masked motif, Core/non-core and Structural variant. Both maize AF readouts are excluded."
        if count == 5
        else "Composite = equal-weight mean of Conservation, Masked motif, Core/non-core and Structural variant. Maize AF is separate.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.045,
        f"Loss uses each model’s full native vocabulary; all inputs are single nucleotides. Dashed lines are descriptive linear fits (n = {count}).",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        target = output.with_suffix("." + suffix)
        fig.savefig(
            target,
            dpi=dpi,
            facecolor="white",
            metadata={"Date": None} if suffix == "svg" else None,
        )
        if suffix == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def assemble(root: Path, directory: Path) -> dict:
    records = read(directory / "models.json")["models"]
    rows = downstream(root, records)
    inputs = read(directory / "inputs.json")
    for row in rows:
        # The dense runs already scored this exact sample. Replay one checkpoint
        # to qualify restoration and sampling, then reuse all three native losses.
        if row["rung"] and row["code"] != "D216":
            row["validation"] = None
            row["loss"] = row["original_loss"]
            row["acgt_loss"] = None
            continue
        result_path = directory / "validation" / (row["code"].lower() + ".json")
        result = validation(row, result_path)
        for name, sample in result["samples"].items():
            if sample["input_sha256"] != inputs["samples"][name]["sha256"]:
                raise ValueError("Loss artifact used a different validation sample")
        row["validation_result_file"] = str(result_path.relative_to(directory))
        row["validation_result_sha256"] = hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest()
        # Raw batch sufficient statistics remain in their source artifact; do
        # not duplicate them in the plotting table.
        row["validation"] = {
            **result,
            "samples": {
                name: {k: v for k, v in sample.items() if k != "batch_statistics"}
                for name, sample in result["samples"].items()
            },
        }
        sample = result["samples"]["common-65536"]
        # Retain historical dense native losses only after the exact-sample replay agrees.
        if row["rung"] and abs(sample["native_loss"] - row["original_loss"]) > 2e-4:
            raise ValueError("Dense replay does not reproduce its original native loss")
        row["loss"] = row["original_loss"] if row["rung"] else sample["native_loss"]
        row["acgt_loss"] = sample["acgt_conditional_loss"]
    stats = {
        metric: correlation(
            [r["loss"] for r in rows], [r["scores"][metric] for r in rows]
        )
        for metric in METRICS
    }
    focused_rows = [r for r in rows if r["plot_family"] != "large"]
    without_481m = {
        "model_codes": [r["code"] for r in focused_rows],
        "composite_linear_fit": composite_fit(focused_rows),
        "correlations": {
            metric: correlation(
                [r["loss"] for r in focused_rows],
                [r["scores"][metric] for r in focused_rows],
            )
            for metric in METRICS
        },
    }
    sensitivity = {
        "original_reported_losses": correlation(
            [r["original_loss"] for r in rows], [r["scores"]["Composite"] for r in rows]
        ),
        "acgt_conditional_moes_only": {
            metric: correlation(
                [r["acgt_loss"] for r in rows if not r["rung"]],
                [r["scores"][metric] for r in rows if not r["rung"]],
            )
            for metric in METRICS
        },
        "native_moes_only": {
            metric: correlation(
                [r["loss"] for r in rows if not r["rung"]],
                [r["scores"][metric] for r in rows if not r["rung"]],
            )
            for metric in METRICS
        },
    }
    sensitivity["poe_with_natural_sv"] = correlation(
        [r["loss"] for r in rows],
        [
            (
                r["readout_sensitivity"]["scores_poe_with_natural_sv"]["Composite"]
                if r["readout_sensitivity"]
                else r["scores"]["Composite"]
            )
            for r in rows
        ],
    )
    sensitivity["af_2k_for_moes"] = correlation(
        [r["loss"] for r in rows],
        [
            (
                r["readout_sensitivity"]["af_zero_2k"]
                if r["readout_sensitivity"]
                else r["scores"][AF_ZERO]
            )
            for r in rows
        ],
    )
    return {
        "schema": 1,
        "rows": rows,
        "correlations": stats,
        "without_481m": without_481m,
        "sensitivity": sensitivity,
    }


def draft(result: dict, directory: Path) -> None:
    rows = result["rows"]
    stats = result["without_481m"]["correlations"]["Composite"]
    full_stats = result["correlations"]["Composite"]
    fit = result["without_481m"]["composite_linear_fit"]
    moes = [r for r in rows if not r["rung"]]
    changes = {
        g: mean(r["readout_sensitivity"]["poe_minus_natural_8k"][g] for r in moes)
        for g in GROUPS[:3]
    }
    avg = mean(changes.values())
    context_changes = {
        g: mean(r["readout_sensitivity"]["natural_8k_minus_2k"][g] for r in moes)
        for g in GROUPS[:3]
    }
    lines = [
        "## DNA validation loss versus downstream evaluations",
        "",
        f"Among the three dense checkpoints and two 60.6M-active MoEs, lower validation loss tracks higher **Composite** performance: **Pearson r = {stats['pearson_r']:+.3f}; Spearman ρ = {stats['spearman_rho']:+.3f}** (five models). Maize allele frequency is shown separately and is excluded from Composite.",
        "",
        "Lower loss tracks better conservation, masked-motif and structural-variant scores; it does not consistently predict better core/non-core or maize allele-frequency results.",
        "",
        "![Validation loss versus Composite and six task groups, excluding the 481M-active MoEs](loss-correlation-without-481m.png)",
        "",
        "<details>",
        "<summary>Including the 481M-active MoEs (all seven models)</summary>",
        "",
        "![Validation loss versus Composite and six task groups, all seven models](loss-correlation.png)",
        "",
        f"With all seven models: Composite **Pearson r = {full_stats['pearson_r']:+.3f}; Spearman ρ = {full_stats['spearman_rho']:+.3f}**.",
        "",
        "</details>",
        "",
        f"At an assumed loss of **{PROJECTED_LOSS:.2f}**, the [1T-token d768 language-initialized run]({PROJECTED_RUN}) projects to **Composite ≈ {fit['projection']['composite']:.3f}** (five-model fit: Composite ≈ {fit['intercept']:.5f} − {abs(fit['slope']):.5f} × loss; extrapolated, not evaluated).",
        "",
        "| ID | MarinDNA model (total / active parameters) | Initialization | DNA tokens | Vocabulary | Validation loss | Training run |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        name = (
            "Dense 0.97B / 0.97B"
            if row["rung"]
            else f"MoE {row['total_parameters_approx'] / 1e9:.2f}B / {row['active_parameters_approx'] / 1e6:.1f}M"
        )
        init = (
            "Language → DNA"
            if row["initialization"] == "language-pretrained"
            else "Scratch"
        )
        lines.append(
            f"| {row['code']} | {name} | {init} | {row['dna_tokens'] / 1e9:.2f}B | {row['vocab_size']:,} | {row['loss']:.6f} | [W&B]({row['wandb_url']}) |"
        )
    lines += [
        "",
        "MoE active counts exclude input/output embeddings; total counts include them. S = scratch; L = language-initialized; D = dense. The numbers in dense IDs indicate billions of DNA training tokens.",
        "",
        "The analysis uses the historical **8K Natural** scoring convention. For target predictions, Natural uses one flank, reporting the better forward/reverse-complement direction per task; **PoE** combines both flanks’ predictions with a normalized geometric mean.",
        "",
        f"Across the four MoEs, 8K beats 2K Natural by **{mean(context_changes.values()):+.4f}** on the matched three-group mean. Switching from Natural to PoE at 8K changes it by **{avg:+.4f}**: conservation and motifs improve, while core/non-core declines.",
        "",
        "<details>",
        "<summary>Score definitions and readout choices</summary>",
        "",
        "Composite is the equal-weight mean of four task-group averages: Conservation (3 AUROCs), Masked motif (8 whole-motif accuracies), Core/non-core (8 AUROCs) and Structural variant (1 AUPRC). Each group averages its tasks equally. This reconstructs the historical four-group definition; the three-group PoE composites from later posts are not substituted.",
        "",
        "Both forward and reverse-complement passes process the window, but causal masking makes each target prediction depend on only its preceding flank. Natural retains the larger task-level metric from either direction where both are available; structural variants have one causal scoring form. This direction selection uses evaluation labels, as in the historical dense results. PoE instead combines the aligned target probabilities as $p_{\\mathrm{PoE}}(b) \\propto \\sqrt{p_{\\mathrm{left}}(b)p_{\\mathrm{right}}(b)}$, normalized over A/C/G/T—equivalently, averaging log probabilities. At 8K, PoE minus Natural averaged over the four MoEs is "
        + "; ".join(f"{g}: **{v:+.4f}**" for g, v in changes.items())
        + ". Structural-variant PoE was not computed, so the matched readout comparison above uses three groups only.",
        "",
        "For Natural, 8K minus 2K averaged over the four MoEs is "
        + "; ".join(f"{g}: **{v:+.4f}**" for g, v in context_changes.items())
        + ". The same 8K reporting convention is used for every model; context and Natural/PoE settings are not selected per model or task.",
        "",
        "Per-model PoE minus Natural at 8K, for the matched three-group mean: "
        + "; ".join(
            f"{r['code']}: {r['readout_sensitivity']['poe_minus_natural_matched_three_group']:+.4f}"
            for r in moes
        )
        + ".",
        "",
        "Maize AF zero-shot is Spearman correlation between the full causal-suffix allele log-likelihood ratio (averaged across strands) and observed AF on **94,075 variants**. Linear-probe AF is Spearman correlation on **28,378 held-out variants**, using a ridge readout of frozen target-position embeddings averaged across strands. The same genomic-block split and overlap purge are used for every model. These are different evaluation sets; their correlations are separate endpoints, not two interchangeable estimates.",
        "",
        "</details>",
        "",
        "<details>",
        "<summary>Original losses, replay checks and shared-sample losses</summary>",
        "",
        "The common sample is the first **512 logical batches × 128 examples × 8,192 bp = 65,536 windows**, or **536,805,376 next-base targets**. Every raw window was checked against the original dense token cache. This preserves the dense sample; it is not a new genome-balanced sample of the full validation split. No examples are packed, reverse-complemented or given EOS. Loss is mean cross-entropy in nats over positions 2–8,192, including N targets, with the model's full native vocabulary.",
        "",
        "Dense runs already used this sample and batch size, so their original native losses are retained. D216 is replayed as a restoration/sampling check. The MoEs originally scored only 512 evenly spaced validation windows; each first replays that sample and then scores the common sample. The physical MoE forward microbatch is 16; sums and target counts are accumulated to the same logical sample, with zero dropped expert assignments required.",
        "",
    ]
    for row in rows:
        v = row["validation"]
        if row["rung"]:
            replay = (
                f"; exact-sample replay {v['samples']['common-65536']['native_loss']:.8f}"
                if v
                else ""
            )
            lines.append(
                f"- **{row['code']}** — original/common {row['original_loss']:.8f}{replay}; `{row['original_metric']}`."
            )
        else:
            s = v["samples"]
            lines.append(
                f"- **{row['code']}** — original {row['original_loss']:.8f}; original-sample replay {s['original-moe-512']['native_loss']:.8f}; common-sample {row['loss']:.8f} (common − original {row['loss'] - row['original_loss']:+.6f}); `{row['original_metric']}`."
            )
    lines += [
        "",
        f"Using the original model-specific losses throughout gives Composite r = {result['sensitivity']['original_reported_losses']['pearson_r']:+.3f}, ρ = {result['sensitivity']['original_reported_losses']['spearman_rho']:+.3f}, before matching the MoE sample to the dense sample.",
        "",
        "A stale `eval/inputs/plantcad-angiosperm-validation-path-only/loss` value on D216 is excluded. Its active `eval/plantcad/loss` agrees with the final reported dense loss. MoE losses use `eval/loss`. Final metric history is checked against the checkpoint's physical update counter; resumed W&B history steps need not equal that counter.",
        "",
        "</details>",
        "",
        "<details>",
        "<summary>Sensitivity checks: vocabulary, readout and context</summary>",
        "",
        "Input tokenization is one token per nucleotide for every model. Dense models have 7 vocabulary entries, scratch MoEs 8, and language-initialized MoEs retain 128,256. The main figure uses full-vocabulary loss and marks these differences by shape.",
        "",
        "The four-base check conditions probabilities on A/C/G/T and uses only canonical-base targets. To avoid rerunning the other dense checkpoints, its correlation comparison is restricted to the **same four MoEs** in both versions: "
        + f"Composite/native r = {result['sensitivity']['native_moes_only']['Composite']['pearson_r']:+.3f}, ρ = {result['sensitivity']['native_moes_only']['Composite']['spearman_rho']:+.3f}; "
        + f"Composite/four-base r = {result['sensitivity']['acgt_conditional_moes_only']['Composite']['pearson_r']:+.3f}, ρ = {result['sensitivity']['acgt_conditional_moes_only']['Composite']['spearman_rho']:+.3f}. This is not a seven-model, vocabulary-controlled estimate.",
        "",
    ]
    for row in moes:
        sample = row["validation"]["samples"]["common-65536"]
        lines.append(
            f"- **{row['code']}** — full vocabulary/all targets {row['loss']:.6f}; full vocabulary/A-C-G-T targets {sample['canonical_native_loss']:.6f}; four-base conditional/A-C-G-T targets {row['acgt_loss']:.6f}."
        )
    readout_stats = result["sensitivity"]["poe_with_natural_sv"]
    af = result["sensitivity"]["af_2k_for_moes"]
    lines += [
        "",
        f"Replacing the three eligible MoE task groups with PoE while retaining Natural structural-variant scores gives Composite r = {readout_stats['pearson_r']:+.3f}, ρ = {readout_stats['spearman_rho']:+.3f}. This hybrid is a sensitivity check, not the primary score.",
        "",
        f"Using 2K instead of 8K Natural zero-shot AF for the MoEs increases their mean AF Spearman by {mean(r['readout_sensitivity']['af_zero_2k_minus_8k'] for r in moes):+.4f}; the seven-model loss/AF relationship becomes r = {af['pearson_r']:+.3f}, ρ = {af['spearman_rho']:+.3f}. Dense AF retains its historical 8K setting, so this mixed-context result is secondary.",
        "",
        f"The Composite projection for the [1T-token run in #586](https://github.com/Open-Athena/marin-dna/issues/586#issuecomment-6544696578) uses a least-squares fit to the five measured checkpoints, excluding the 481M-active MoEs. Their losses span **{fit['observed_loss_range'][0]:.3f}–{fit['observed_loss_range'][1]:.3f}**. The assumed loss of **{PROJECTED_LOSS:.2f}** lies outside that range and is not included in the fit or correlations.",
        "",
        "</details>",
        "",
        "<details>",
        "<summary>Checkpoint and execution provenance</summary>",
        "",
        "Historical downstream results are copied from [exp472](https://github.com/plantcad/plantcad2/pull/20), the [exp582 archive](https://github.com/plantcad/plantcad2/tree/351e50a114c26635d5d921c5dd2e88dd9bccdfcf/zero-shot-leaderboard/marindna_exp586/historical/exp582-final) and [exp586](https://github.com/plantcad/plantcad2/pull/22#issuecomment-6366361125). All task scores and probe fits are reused; only held-out language-model loss is evaluated here. Parameter counts are approximate and follow the earlier non-embedding active-parameter convention for MoEs.",
        "",
    ]
    for row in rows:
        lines += [
            f"**{row['code']}** — `{row['wandb_path']}`, metric `{row['original_metric']}`; checkpoint:",
            "",
            f"`{row['checkpoint']}`",
            "",
        ]
        if row["validation"]:
            lines += [
                f"[Loss replay/common-sample execution]({row['validation']['validation_wandb']}); checkpoint metadata SHA-256 `{row['canonical_metadata_sha256']}`.",
                "",
            ]
    lines += [
        "Evaluation used one eight-H100 node per job on CoreWeave, Iris batch priority and user eczech. The runtime is pinned to Marin 0.2.224.dev40027069267, the newest compatible release retaining the checkpoint implementation's ArrayStacked API. Checkpoints are read without updates; native MoE backends and pending router-bias updates are preserved. The saved JSON records input hashes, worker hash, runtime versions, batch sufficient statistics and storage locations.",
        "",
        "</details>",
        "",
    ]
    (directory / "draft.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--downstream-only", action="store_true")
    args = parser.parse_args()
    directory = args.root / "runs/loss-correlation"
    if args.downstream_only:
        rows = downstream(args.root, read(directory / "models.json")["models"])
        (directory / "downstream.json").write_text(json.dumps(rows, indent=2) + "\n")
        return
    result = assemble(args.root, directory)
    (directory / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    figure(result["rows"], directory / "loss-correlation")
    figure(
        [
            r
            for r in result["rows"]
            if r["code"] in result["without_481m"]["model_codes"]
        ],
        directory / "loss-correlation-without-481m",
    )
    draft(result, directory)


if __name__ == "__main__":
    main()
