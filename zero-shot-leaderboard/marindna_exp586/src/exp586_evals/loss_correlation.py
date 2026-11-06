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

from exp586_evals.historical import dense_results, verify_exp582_results

GROUPS = ("Conservation", "Masked motif", "Core/non-core", "Structural variant")
AF_ZERO = "Maize AF · zero-shot"
AF_PROBE = "Maize AF · linear probe"
METRICS = ("Composite", *GROUPS, AF_ZERO, AF_PROBE)
SPLIT = "c37e713eb0449af48f99a14de5e3c86fab8cb5504c4872ccfd24e8988f0b89a0"
COLORS = {"dense": "#7355A3", "moe": "#008B87"}
MARKERS = {"scratch": "^", "language-pretrained": "o"}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


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
            if "execution" in provenance:
                binding = provenance["execution"]["binding"]
                receipt = read(directory / "analysis-receipt.json")
                checkpoint = read(directory / "checkpoint-binding.json")
                if (
                    binding["checkpoint_digest"] != record["canonical_metadata_sha256"]
                    or binding["tokenizer_sha256"] != record["tokenizer_sha256"]
                    or binding["requests_sha256"] != receipt["requests_sha256"]
                    or binding["checkpoint_role"] != "dna-trained"
                    or receipt["source_artifacts"]
                    != provenance["execution"]["uri"].split("/workers/")[0]
                    or checkpoint["checkpoint"] != record["checkpoint"]
                    or checkpoint["canonical_metadata_sha256"]
                    != record["canonical_metadata_sha256"]
                    or checkpoint["metadata"]["exp586_clock"]["input_tokens"]
                    != record["dna_tokens"]
                ):
                    raise ValueError(
                        "Downstream execution refers to another checkpoint"
                    )
                for relative in ("result.json", "context-8192/probes/result.json"):
                    if (
                        provenance["selected_files"][relative]
                        != receipt["files"][relative]
                    ):
                        raise ValueError(
                            "Downstream file differs from analysis receipt"
                        )
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
    if len(output) != len({r["code"] for r in output}) or len(output) < 3:
        raise ValueError("Expected at least three distinct checkpoint models")
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
    count_label = {6: "Six", 8: "Eight"}.get(count, str(count))
    focused = all(row["plot_family"] != "large" for row in rows)
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
        top=0.80,
        wspace=0.30,
        hspace=0.60,
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
        "M61-L-1T": (12, -18),
    }
    if not focused:
        offsets["M61-L"] = (-12, -20)
    for index, (ax, metric) in enumerate(zip(axes, METRICS, strict=True)):
        y = np.asarray([r["scores"][metric] for r in rows])
        for family, color in COLORS.items():
            members = [r for r in rows if bool(r["rung"]) == (family == "dense")]
            losses = np.asarray([r["loss"] for r in members])
            scores = np.asarray([r["scores"][metric] for r in members])
            fit = np.polyfit(losses, scores, 1)
            xx = np.linspace(losses.min(), losses.max(), 100)
            ax.plot(
                xx,
                np.polyval(fit, xx),
                color=color,
                lw=1.8,
                linestyle=":",
                zorder=1,
            )
        for row, yy in zip(rows, y, strict=True):
            ax.scatter(
                row["loss"],
                yy,
                s=100 if index == 0 else 60,
                marker=MARKERS[row["initialization"]],
                color=COLORS["dense" if row["rung"] else "moe"],
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
            (0.013, 0.014) if not focused else (0.14 * np.ptp(x), 0.15 * np.ptp(x))
        )
        ax.set_xlim(x.min() - left_margin, x.max() + right_margin)
        ax.margins(y=0.23 if index == 0 else 0.18)
        ax.grid(color="#E8ECF0", linewidth=0.8)
        ax.set_axisbelow(True)
        panel_title = metric
        if metric == "Composite":
            panel_title = "Composite (excludes AF)"
        elif metric in (AF_ZERO, AF_PROBE):
            panel_title += "\nExcluded from Composite"
        ax.set_title(
            panel_title,
            loc="left",
            fontsize=16
            if index == 0
            else (12 if metric in (AF_ZERO, AF_PROBE) else 13),
            fontweight="bold",
            pad=6 if metric in (AF_ZERO, AF_PROBE) else 10,
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
        Line2D([], [], color=COLORS[family], lw=2, linestyle=":", label=label)
        for family, label in [("dense", "Dense"), ("moe", "MoE")]
    ]
    initialization_handles = [
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
            (MARKERS["scratch"], "From scratch"),
            (MARKERS["language-pretrained"], "Language-pretrained"),
        ]
    ]
    fig.legend(
        handles=[*color_handles, *initialization_handles],
        loc="upper left",
        bbox_to_anchor=(0.064, 0.885),
        frameon=False,
        ncol=4,
        fontsize=11,
        columnspacing=2.4,
    )
    fig.text(
        0.07,
        0.075,
        "Composite = mean of Conservation, Masked motif, Core/non-core and Structural variant. Both maize AF readouts are excluded.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.045,
        f"Dotted lines: separate Dense (n = {sum(bool(r['rung']) for r in rows)}) and MoE (n = {sum(not r['rung'] for r in rows)}) fits. Full-vocabulary loss; one token per nucleotide.",
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
    return {
        "schema": 2,
        "rows": rows,
        "without_481m": {
            "model_codes": [r["code"] for r in rows if r["plot_family"] != "large"],
        },
    }


def draft(result: dict, directory: Path) -> None:
    rows = result["rows"]
    new = next(r for r in rows if r["code"] == "M61-L-1T")
    old = next(r for r in rows if r["code"] == "M61-L")
    lines = [
        "## DNA validation loss versus downstream evaluations",
        "",
        "**The finished 1T model has the lowest validation loss, but a lower four-group Composite than its 0.22T counterpart.** Conservation and motif scores improve; structural-variant and core/non-core scores decline. Lower next-base loss does not guarantee better performance across these tasks.",
        "",
        f"On the identical validation sample, loss falls from **{old['loss']:.6f} to {new['loss']:.6f}**, while Composite changes from **{old['scores']['Composite']:.4f} to {new['scores']['Composite']:.4f}**. Maize allele frequency (AF) is separate and excluded from Composite.",
        "",
        "![Validation loss versus Composite and six task groups, excluding the 481M-active MoEs](loss-correlation-without-481m.png)",
        "",
        "<details>",
        f"<summary>Including the 481M-active MoEs (all {len(rows)} models)</summary>",
        "",
        "![Validation loss versus Composite and six task groups, all models](loss-correlation.png)",
        "",
        "</details>",
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
        "MoE active counts exclude input/output embeddings; total counts include them. S = scratch; L = language-initialized; D = dense. Dense IDs indicate billions of DNA training tokens; M61-L and M61-S use 216.16B tokens, and M61-L-1T uses 1T.",
        "",
        "<details>",
        "<summary>Identical downstream metrics and readouts</summary>",
        "",
        "All models use the historical **8K Natural** convention. Composite equally weights four task-group means: Conservation (3 AUROCs), Masked motif (8 whole-motif accuracies), Core/non-core (8 AUROCs), and Structural variant (1 AUPRC). The same 20 tasks and sample counts are used throughout. Natural reports the better forward/reverse-complement task metric where both exist; this direction selection uses evaluation labels, matching the historical dense results. SV retains its original causal scoring form. No PoE or context selection is substituted.",
        "",
        "Maize AF zero-shot is Spearman correlation of the full causal-suffix allele log-likelihood ratio, averaged across strands, on **94,075 variants**. The frozen linear probe uses **8K target-position embeddings averaged across strands**, with the same genomic-block split and **28,378 held-out variants**. These are separate endpoints on different evaluation sets. The 1T point uses these exact readouts, rather than a different probe or its best-performing inference setting.",
        "",
        "</details>",
        "",
        "<details>",
        "<summary>Matched validation loss and replay checks</summary>",
        "",
        "Loss is measured on the same first **65,536 validation windows of 8,192 bp**, totaling **536,805,376 next-base targets**. The window data and order were verified against the original dense token cache. Inputs use one token per nucleotide, without packing, reverse complementation, or EOS. Cross-entropy is in nats over positions 2–8,192, includes N targets, and uses each model's full native vocabulary.",
        "",
        "Dense runs already used this sample; their original losses are retained after a D216 replay check. Every MoE, including 1T, first reproduces its original 512-window validation loss, then scores this common sample with forward microbatches of 16 and zero dropped expert assignments. The 1T sample hash exactly matches the earlier analysis.",
        "",
        "| ID | Original logged loss | Original-sample replay | Common-sample loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        replay = "—"
        if row["validation"]:
            sample = "common-65536" if row["rung"] else "original-moe-512"
            replay = f"{row['validation']['samples'][sample]['native_loss']:.8f}"
        lines.append(
            f"| {row['code']} | {row['original_loss']:.8f} | {replay} | {row['loss']:.8f} |"
        )
    lines += [
        "",
        "Color distinguishes Dense from MoE; marker shape distinguishes scratch from language-pretrained initialization. Dotted regression lines are fitted separately to Dense and MoE checkpoints over their observed loss ranges. Loss uses each model's full native vocabulary; the A/C/G/T sensitivity check remains in the saved analysis.",
        "",
        "</details>",
        "",
        "<details>",
        "<summary>Data and checkpoint provenance</summary>",
        "",
        "All points are measured results from completed runs. Downstream evaluations and probe fits are reused from their saved outputs. Only the common-sample language-model losses were separately evaluated; no new model inference or probe fitting was needed for this update.",
        "",
        "[Models, final checkpoints, and training-run bindings](models.json) · [Immutable validation samples](inputs.json) · [Loss artifacts and hashes](results-manifest.json) · [1T downstream provenance](downstream/m61-l-1t/provenance.json) · [Scores and validation losses](analysis.json)",
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
