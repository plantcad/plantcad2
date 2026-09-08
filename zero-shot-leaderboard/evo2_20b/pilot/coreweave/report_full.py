"""Generate compact full-checkpoint comparison tables/post from verified artifacts."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from merge_recommended import LABELS
from plot_recommended_comparison import BASELINE_MODELS, TASK_IDS, category_summary


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| :--- |" + " ---: |" * (len(headers) - 1), *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]])


def details(title, body):
    return f"<details>\n<summary><strong>{title}</strong></summary>\n\n{body}\n\n</details>"


def clock(seconds):
    value = round(seconds)
    return f"{value // 3600}:{value // 60 % 60:02d}:{value % 60:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-ref", default="IMAGE_REFERENCE")
    parser.add_argument("--report-url", default="RUN_REPORT_URL")
    args = parser.parse_args()
    records = {key: json.loads((args.run_root / f"{key}-download.json").read_text()) for key in ("022t", "039t", "056t")}
    results = {key: json.loads((Path(record["local_artifact"]) / "result.json").read_text()) for key, record in records.items()}
    validations = {key: json.loads((args.run_root / f"{key}-validation.json").read_text()) for key in results}
    for key, result in results.items():
        assert result["sampling"]["method"] == "full" and validations[key]["verified"]
        assert [row["key"] for row in result["rows"]] == list(LABELS)
        assert len(result["rows"]) == 20
    values = {key: [row["selected_value"] for row in result["rows"]] for key, result in results.items()}
    groups = {key: category_summary(vector) for key, vector in values.items()}
    published_rows = list(csv.DictReader(args.published.open()))
    published = {}
    for label, (model, context) in BASELINE_MODELS.items():
        selected = {row["task_id"]: float(row["value"]) for row in published_rows if row["model"] == model and row["context_bp"] == context}
        published[label] = [selected[key] for key in TASK_IDS]
    published_groups = {label: category_summary(vector) for label, vector in published.items()}
    latest_deltas = [new - old for new, old in zip(values["056t"], values["039t"], strict=True)]
    previous_deltas = [new - old for new, old in zip(values["039t"], values["022t"], strict=True)]
    labels = {"022t": "MarinDNA 1B 0.22T", "039t": "MarinDNA 1B 0.39T", "056t": "MarinDNA 1B 0.56T"}
    sampled_paths = {"039t": args.run_root.parent / "step371065/leaderboard-seed0-n10000/result.json", "056t": args.run_root.parent / "s02-lr5e4-step535985/leaderboard-seed0-n10000/result.json"}
    sampled = {key: json.loads(path.read_text()) for key, path in sampled_paths.items()}
    sampled_groups = {key: category_summary([row["selected_value"] for row in result["rows"]]) for key, result in sampled.items()}
    overlaps = {key: json.loads((args.run_root / f"{key}-overlap.json").read_text()) for key in sampled}
    summary = {"labels": labels, "groups": groups, "published_groups": published_groups, "sampled_groups": sampled_groups, "results": results, "artifacts": records, "validation": validations, "overlap_validation": overlaps, "latest_task_deltas": dict(zip(LABELS, latest_deltas, strict=True)), "previous_task_deltas": dict(zip(LABELS, previous_deltas, strict=True)), "published_revision": "f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29", "protocol": "Full splits; same LR5e-4/WD0.1 lineage; BF16/FA2/FP32 ACGT softmax/no cache/b32; non-SV max task-level fwd/RC, SV forward; Composite unweighted mean of four task-group means"}
    group_rows = [[labels[key], *[f"{v:.4f}" for v in groups[key]]] for key in ("056t", "039t", "022t")]
    group_rows += [[label, *[f"{v:.4f}" for v in vector]] for label, vector in published_groups.items()]
    task_rows, baseline_rows, context_rows = [], [], []
    for index, (key, (category, label)) in enumerate(LABELS.items()):
        name = category + " — " + label
        metric = {"auroc": "AUROC", "auprc": "AUPRC", "motif_accuracy": "Accuracy"}[results["056t"]["tasks"][key]["primary_metric"]]
        count = results["056t"]["tasks"][key]["samples"]
        assert all(result["tasks"][key]["samples"] == count for result in results.values())
        task_rows.append([name, f"{count:,}", metric, *[f"{values[scale][index]:.4f}" for scale in ("022t", "039t", "056t")], f"{previous_deltas[index]:+.4f}", f"{latest_deltas[index]:+.4f}"])
        baseline_rows.append([name, metric, *[f"{vector[index]:.4f}" for vector in published.values()]])
        context_values = []
        for scale in ("022t", "039t", "056t"):
            task = results[scale]["tasks"][key]
            for context in ("left", "right_reverse_complement"):
                if context not in task["contexts"]:
                    context_values.append("—")
                else:
                    value = f"{task['contexts'][context]['metrics'][task['primary_metric']]:.6f}"
                    context_values.append(f"**{value}**" if context == task["best_context"] else value)
        context_rows.append([name, *context_values])
    runtime_rows, checkpoint_rows, artifact_links = [], [], []
    for key in ("022t", "039t", "056t"):
        result = results[key]
        config, runtime = result["checkpoint"], result["runtime"]
        peak = max(context["peak_reserved_gib"] for task in result["tasks"].values() for context in task["contexts"].values())
        runtime_rows.append([labels[key], clock(runtime["measured_parallel_wall_seconds"]), f"{runtime['cluster_sequences_per_second']:.2f}", f"{runtime['h100_sequences_per_second']:.2f}", f"{peak:.2f} GiB"])
        checkpoint_rows.append([f"[{labels[key]}](https://wandb.ai/eric-czech/marin/runs/{config['run_id']})", f"{config['step']:,}", f"{config['tokens']:,}"])
        artifact_links.append(f"- {labels[key]}: [checkpoint](https://huggingface.co/eczech/marindna-exp472/tree/{config['revision']}/{config['model_prefix']}) · [verified full results]({records[key]['url']}).")
    sampled_rows = []
    for key in ("039t", "056t"):
        diffs = [a - b["selected_value"] for a, b in zip(values[key], sampled[key]["rows"], strict=True)]
        sampled_rows.append([labels[key], f"{sampled_groups[key][-1]:.4f}", f"{groups[key][-1]:.4f}", f"{groups[key][-1] - sampled_groups[key][-1]:+.4f}", f"{max(abs(value) for value in diffs):.4f}"])
    overlap_rows = [[labels[key], f"{overlaps[key]['different_elements']:,} / {overlaps[key]['elements']:,}", f"{overlaps[key]['max_abs_delta']:.3g}", f"{overlaps[key]['subset_metrics']['max_selected_metric_delta']:.3g}", overlaps[key]["subset_metrics"]["changed_strand_selections"]] for key in ("039t", "056t")]
    parts = [
        "## Result 7 — Full evaluation of MarinDNA 1B 0.22T / 0.39T / 0.56T",
        "**Full splits, no downsampling:** 1,727,943 input rows across all 20 tasks, for each checkpoint. All three are post-cooldown checkpoints from the same **LR5e-4 / WD0.1** lineage. The .22T checkpoint replaces the different LR2e-4 model shown in earlier posts.",
        f"![Full MarinDNA training-scale comparison]({args.image_ref})",
        f"**Composite: {groups['022t'][-1]:.4f} → {groups['039t'][-1]:.4f} → {groups['056t'][-1]:.4f}** at .22T → .39T → .56T. The .39T checkpoint improves {sum(value > 0 for value in previous_deltas)}/20 task rows over .22T; .56T improves {sum(value > 0 for value in latest_deltas)}/20 over .39T. ANALYSIS_TO_REVIEW",
        "Composite gives each of the four task groups equal weight (25% each), not each of the 20 rows. Scoring is unchanged: select the larger completed forward/RC **task-level metric** for non-SV tasks; SV remains forward reference-versus-mutant. The figure uses model-family colors and annotates .56T and evo2_20b.",
        details("Group scores and published-model comparison", table(["Model", "Conservation", "Masked motif", "Core/non-core", "SV", "Composite"], group_rows) + "\n\nPublished values use the same leaderboard snapshot (`f3c4ddb1978b78ca56fed5d40378f0aa5f19ec29`), still current when this run began. PlantCAD uses its published 512-bp context; the other models use 8,192 bp. All comparisons here use full splits."),
        details("All 20 full MarinDNA results by species/task", table(["Species / task", "Rows", "Metric", ".22T", ".39T", ".56T", "Δ .39T−.22T", "Δ .56T−.39T"], task_rows) + "\n\nDeltas use unrounded values. All available rows are scored; the established ambiguous-base validity mask remains in motif-accuracy denominators. These are point scores on the complete benchmark, not statistical significance tests."),
        details("Published baselines by species/task", table(["Species / task", "Metric", *published], baseline_rows)),
        details("Forward / reverse-complement metrics", table(["Species / task", ".22T fwd", ".22T RC", ".39T fwd", ".39T RC", ".56T fwd", ".56T RC"], context_rows) + "\n\nBold selects the reported task score. No per-example strand maximum or averaging of chunk AUROCs/AUPRCs."),
        details("Full versus previous 10,000-row previews", table(["Same checkpoint", "Sampled Composite", "Full Composite", "Δ full−sampled", "Largest absolute task Δ"], sampled_rows) + "\n\nThe previous .39T/.56T numbers are precomputed on the retained seed-0 10,000-row fixtures. Full runs use all rows in original dataset order. The old .22T LR2e-4 result is deliberately excluded from this same-checkpoint comparison.\n\nA direct overlap check matched all 200,000 retained examples to the full inputs, then compared saved predictions and metrics on those same examples (including duplicate-equivalent rows). This isolates numerical/batch-composition effects from the larger sample size:\n\n" + table(["Same checkpoint", "Different / total raw values", "Max abs raw Δ", "Max abs same-subset task Δ", "Changed strand selections"], overlap_rows)),
        details("Execution, checkpoint provenance, and verified artifacts", "Three jobs ran concurrently, each on **4 H100x8 nodes / 32 GPUs** in `cw-us-east-02a`, at batch priority: **12 nodes / 96 GPUs total**. All allocations were released after successful completion.\n\n" + table(["Model", "Scoring wall time", "Aggregate forwards/s", "Active-worker forwards/s/GPU", "Peak reserved/GPU"], runtime_rows) + "\n\nEach run scores 3,455,886 sequence forwards / 28,310,618,112 tokens. These are measured scoring times, including worker input/output but excluding provisioning, input staging, and final reduction/upload; not extrapolations. BF16, external FA2, FP32 A/C/G/T softmax, no KV cache, batch32, TF32; Torch2.7.0/CUDA12.8, Transformers5.15.1, FA2 2.8.3.post1. FA2 was profiler-verified on all 96 workers.\n\n" + table(["Model / W&B run", "Final step", "Post-cooldown training tokens"], checkpoint_rows) + "\n\nLater stages resume the matching pre-cooldown checkpoints (steps164920/329840), then undergo their own cooldown. Dataset revision `d340debe0c8402c84f0696cd2002f87c2f7ba6db`; identical input hashes across all workers/checkpoints. The sampled execution path remains available for previews.\n\nIndependent CPU validation recomputed all 77 context metrics per model from saved predictions and directly extracted targets, checked every selected strand, and reproduced non-SV scores exactly from saved probabilities. Complete row coverage, chunk hashes, model hashes, and all 77 prediction arrays were verified per model.\n\n" + f"[Run report]({args.report_url}) · [Published leaderboard](https://huggingface.co/spaces/plantcad/plantcad2-zeroshot-leaderboard)\n\n" + "\n".join(artifact_links)),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "post.md").write_text("\n\n".join(parts) + "\n")
    print(json.dumps({"groups": groups, "improved_039_over_022": sum(value > 0 for value in previous_deltas), "improved_056_over_039": sum(value > 0 for value in latest_deltas), "largest_latest_deltas": sorted(zip(LABELS, latest_deltas, strict=True), key=lambda pair: pair[1]), "runtime": {key: result["runtime"] for key, result in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
