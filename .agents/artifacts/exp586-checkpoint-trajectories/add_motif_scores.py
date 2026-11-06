"""Add independently verified motif probes and matching category composites."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CHECKPOINTS = {
    "216B-token run": {
        "dna-000b": "base-000b", "dna-002b": "short-002b",
        "dna-031b": "short-031b", "dna-124b": "short-124b", "dna-216b": "short-216b",
    },
    "1T-token run": {
        "dna-000b": "base-000b", "dna-010b": "long-010b",
        "dna-182b": "long-182b", "dna-545b": "long-545b", "dna-1000b": "long-1000b",
    },
}
TASKS = [f"motif_{species}_{task}" for species in ["maize", "tomato"]
         for task in ["tis", "tts", "donor", "acceptor"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-results", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.verified_results.read_text())
    assert len(results) == 9 and all(set(tasks) == set(TASKS) for tasks in results.values())
    source = HERE / "original-category-scores.csv"
    with source.open() as stream:
        original = list(csv.DictReader(stream))
    assert len(original) == 110
    rows = original.copy()
    provenance = {}
    for run, checkpoints in CHECKPOINTS.items():
        provenance[run] = {}
        for checkpoint, label in checkpoints.items():
            values = []
            for task in TASKS:
                record = results[label][task]
                assert record["all_file_hashes_verified"] and record["independent_heldout_metrics_exact"]
                values.append(record["result"]["test_score"])
            score = float(np.mean(values))
            template = next(row for row in original if row["run"] == run
                            and row["checkpoint"] == checkpoint)
            rows.append({**template, "metric": "Motifs", "readout": "Linear probe", "score": score})
            for name, categories in [
                ("Composite +SV", ["Conservation", "Motifs", "Core/non-core", "SV impact"]),
                ("Composite −SV", ["Conservation", "Motifs", "Core/non-core"]),
            ]:
                selected = [float(next(row["score"] for row in rows
                                       if row["run"] == run and row["checkpoint"] == checkpoint
                                       and row["metric"] == category and row["readout"] == "Linear probe"))
                            for category in categories]
                rows.append({**template, "metric": name, "readout": "Linear probe",
                             "score": float(np.mean(selected))})
            provenance[run][checkpoint] = {"model_label": label, "motif_category_accuracy": score,
                                           "task_accuracies": dict(zip(TASKS, values, strict=True))}
    assert len(rows) == 140
    keys = [(row["run"], row["checkpoint"], row["metric"], row["readout"]) for row in rows]
    assert len(set(keys)) == 140
    for row in original:
        assert next(point for point in rows if all(point[key] == row[key]
                    for key in ["run", "checkpoint", "metric", "readout"])) == row
    for run, checkpoints in CHECKPOINTS.items():
        for checkpoint in checkpoints:
            for readout in ["Zero-shot likelihood", "Linear probe"]:
                selection = {row["metric"]: float(row["score"]) for row in rows
                             if row["run"] == run and row["checkpoint"] == checkpoint
                             and row["readout"] == readout}
                expected = np.mean([selection[key] for key in ["Conservation", "Motifs", "Core/non-core", "SV impact"]])
                assert abs(selection["Composite +SV"] - expected) < 1e-15
    with (HERE / "all-category-scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(original[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {label: {task: {
        **{key: value["result"][key] for key in ["test_score", "majority_baseline", "shuffled_labels", "train_rows", "test_rows"]},
        "selected_C": value["result"]["development"]["C"],
        "development_accuracy": value["result"]["development"]["development_score"],
        "receipt_sha256": value["receipt_sha256"], "uri": value["uri"],
    } for task, value in tasks.items()} for label, tasks in results.items()}
    (HERE / "motif-probe-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    aggregation = {
        "original_scores_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "verified_results_sha256": hashlib.sha256(args.verified_results.read_bytes()).hexdigest(),
        "output_scores_sha256": hashlib.sha256((HERE / "all-category-scores.csv").read_bytes()).hexdigest(),
        "original_scores_unchanged": 110, "new_scores": 30,
        "unique_checkpoint_task_heads": 72, "displayed_checkpoint_task_heads": 80,
        "motif_aggregation": "Equal-weight mean of eight per-task whole-motif exact-match accuracies",
        "composite": "Equal-weight mean of conservation, motifs, core/non-core and SV, separately for each readout; AF excluded",
        "checkpoints": provenance,
    }
    (HERE / "motif-aggregation-provenance.json").write_text(json.dumps(aggregation, indent=2) + "\n")
    previous = HERE / "original-category-provenance.json"
    if not previous.exists():
        previous.write_bytes((HERE / "all-category-provenance.json").read_bytes())
    current = json.loads(previous.read_text())
    current.update({"plotted_scores": 140, "motif_aggregation": aggregation,
                    "current_scores_sha256": aggregation["output_scores_sha256"]})
    current["declines"] = {}
    for run in CHECKPOINTS:
        current["declines"][run] = {}
        for metric in current["metrics"]:
            for readout in ["Zero-shot likelihood", "Linear probe"]:
                points = sorted((row for row in rows if row["run"] == run
                                 and row["metric"] == metric and row["readout"] == readout),
                                key=lambda row: float(row["DNA_tokens_B"]))
                current["declines"][run][metric + "/" + readout] = [
                    {"from_B": float(left["DNA_tokens_B"]), "to_B": float(right["DNA_tokens_B"]),
                     "delta": float(right["score"]) - float(left["score"])}
                    for left, right in zip(points, points[1:]) if float(right["score"]) < float(left["score"])
                ]
    (HERE / "all-category-provenance.json").write_text(json.dumps(current, indent=2) + "\n")
    print("Preserved all 110 existing scores and added 30 independently checked motif/composite probe scores.")


if __name__ == "__main__":
    main()
