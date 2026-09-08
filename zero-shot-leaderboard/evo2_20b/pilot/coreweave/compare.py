"""Paired metric and raw-prediction comparison with the pinned Lambda reference."""

import json
from pathlib import Path

import numpy as np

from common import write_json


def group_scores(result: dict) -> dict[str, float]:
    categories = ("Conservation", "Masked motif", "Core/non-core", "Structural variant")
    groups = {category: float(np.mean([row["selected_value"] for row in result["rows"] if row["category"] == category])) for category in categories}
    groups["Composite"] = float(np.mean(list(groups.values())))
    return groups


def compare(root: Path, reference: Path) -> dict:
    result = json.loads((root / "result.json").read_text())
    baseline = json.loads((reference / "result.json").read_text())
    reference_arrays = {}
    for path in sorted((reference / "workers").glob("gpu*/scores.npz")):
        with np.load(path, allow_pickle=False) as archive:
            overlap = set(reference_arrays) & set(archive.files)
            if overlap:
                raise ValueError(f"Duplicate reference arrays: {overlap}")
            reference_arrays.update({key: archive[key] for key in archive.files})
    with np.load(root / "scores.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(reference_arrays):
            raise ValueError("CoreWeave/reference array sets differ")
        array_rows = []
        for key in archive.files:
            actual, expected = archive[key], reference_arrays[key]
            if actual.shape != expected.shape:
                raise ValueError(f"Prediction shapes differ: {key}")
            delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            array_rows.append({"key": key, "shape": list(actual.shape), "elements": actual.size, "changed_elements": int(np.count_nonzero(actual != expected)), "max_absolute_difference": float(delta.max()), "mean_absolute_difference": float(delta.mean())})
    by_key = {row["key"]: row for row in baseline["rows"]}
    rows = [{"key": row["key"], "task": row["category"] + " — " + row["task"], "lambda": by_key[row["key"]]["selected_value"], "coreweave": row["selected_value"], "delta": row["selected_value"] - by_key[row["key"]]["selected_value"], "lambda_context": by_key[row["key"]]["selected_context"], "coreweave_context": row["selected_context"]} for row in result["rows"]]
    contexts = []
    for key, task in result["tasks"].items():
        for context, record in task["contexts"].items():
            for metric, value in record["metrics"].items():
                previous = baseline["tasks"][key]["contexts"][context]["metrics"][metric]
                contexts.append({"task": key, "context": context, "metric": metric, "lambda": previous, "coreweave": value, "delta": value - previous})
    environments = [{"worker": index, "differences": {key: {"lambda": baseline["environments"][0].get(key), "coreweave": value} for key, value in environment.items() if baseline["environments"][0].get(key) != value}} for index, environment in enumerate(result["environments"])]
    report = {"rows": rows, "all_context_metrics": contexts, "arrays": array_rows, "groups": {"lambda": group_scores(baseline), "coreweave": group_scores(result)}, "environment_differences": environments, "runtime": {"lambda": baseline["runtime"], "coreweave": result["runtime"]}, "max_selected_metric_delta": max(abs(row["delta"]) for row in rows), "max_any_context_metric_delta": max(abs(row["delta"]) for row in contexts), "changed_selected_contexts": [row["key"] for row in rows if row["lambda_context"] != row["coreweave_context"]], "raw_predictions_exact": all(row["changed_elements"] == 0 for row in array_rows)}
    write_json(root / "comparison.json", report)
    table = ["| Task | Lambda | CoreWeave | Δ |", "| :--- | ---: | ---: | ---: |"]
    table.extend(f"| {row['task']} | {row['lambda']:.6f} | {row['coreweave']:.6f} | {row['delta']:+.6f} |" for row in rows)
    (root / "comparison.md").write_text("\n".join(table) + "\n")
    return report
