"""Family-balanced summaries without ranks over correlated checkpoints."""

import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
FAMILIES = ["Conservation", "Motifs", "Core/non-core", "SV impact", "Allele frequency"]
METRIC = "Avg normalized score"


def normalized_average(family_means: dict[str, dict[str, float]]) -> dict:
    """Min–max scale family means over all checkpoints, then average families."""
    if not family_means:
        raise ValueError("A checkpoint population is required")
    expected = set(next(iter(family_means.values())))
    if not expected or any(set(scores) != expected for scores in family_means.values()):
        raise ValueError("Every checkpoint must cover the same nonempty families")
    if not all(isfinite(value) for scores in family_means.values() for value in scores.values()):
        raise ValueError("Family means must be finite")
    bounds = {family: {"min": min(scores[family] for scores in family_means.values()),
                       "max": max(scores[family] for scores in family_means.values())}
              for family in sorted(expected)}
    if any(bound["min"] == bound["max"] for bound in bounds.values()):
        raise ValueError("A constant family cannot be min–max normalized")
    result = {}
    for checkpoint, scores in family_means.items():
        scaled = {family: (scores[family] - bounds[family]["min"]) /
                         (bounds[family]["max"] - bounds[family]["min"]) for family in sorted(expected)}
        result[checkpoint] = {"score": mean(scaled.values()), "families": scaled}
    return {"bounds": bounds, "checkpoints": result}


def pairwise_win_rate(models: dict[str, dict[str, float]], families: dict[str, list[str]]) -> dict:
    """Half-credit exact ties; average task wins within families, then families."""
    if len(models) != 2:
        raise ValueError("Win Rate compares exactly two models")
    expected = {task for tasks in families.values() for task in tasks}
    if not expected or any(not tasks for tasks in families.values()) or sum(map(len, families.values())) != len(expected):
        raise ValueError("Tasks must belong to exactly one nonempty family")
    if any(set(scores) != expected for scores in models.values()):
        raise ValueError("Both models must cover every task")
    if not all(isfinite(value) for scores in models.values() for value in scores.values()):
        raise ValueError("Task scores must be finite")
    result = {}
    for model, scores in models.items():
        opponent = next(other for other in models if other != model)
        wins = {task: float(scores[task] > models[opponent][task]) + .5 * float(scores[task] == models[opponent][task])
                for task in sorted(expected)}
        family_wins = {family: mean(wins[task] for task in tasks) for family, tasks in families.items()}
        result[model] = {"score": 100 * mean(family_wins.values()),
                         "families": {family: 100 * value for family, value in family_wins.items()}, "tasks": wins}
    return result


def main() -> None:
    source = HERE / "all-category-scores.csv"
    with source.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row["metric"] in FAMILIES]
    assert len(rows) == 100
    native_path = HERE / "native-task-scores.json"
    native = json.loads(native_path.read_text())["checkpoints"]
    task_families = {**json.loads((HERE / "frozen-references.json").read_text())["Avg Rank"]["families"],
                     "Allele frequency": ["af"]}
    identities = {}
    for row in rows:
        run, checkpoint = row["run"], row["checkpoint"]
        identities[run, checkpoint] = "base-000b" if checkpoint == "dna-000b" else checkpoint.replace(
            "dna", "short" if run == "216B-token run" else "long", 1)
    assert len(identities) == 10 and len(set(identities.values())) == 9
    normalized = {}
    for readout in ["Zero-shot", "Linear probe"]:
        unique = {}
        for (run, checkpoint), identity in identities.items():
            means = {row["metric"]: float(row["score"]) for row in rows
                     if row["run"] == run and row["checkpoint"] == checkpoint and row["readout"] == readout}
            assert set(means) == set(FAMILIES)
            if identity in unique:
                assert unique[identity] == means
            unique[identity] = means
        normalized[readout] = normalized_average(unique)
        for (run, checkpoint), identity in identities.items():
            original = next(row for row in rows if row["run"] == run and row["checkpoint"] == checkpoint)
            rows.append({"run": run, "checkpoint": checkpoint, "DNA_tokens_B": original["DNA_tokens_B"],
                         "metric": METRIC, "readout": readout,
                         "score": normalized[readout]["checkpoints"][identity]["score"]})
    assert len(rows) == 120
    with (HERE / "normalized-category-scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    final = {key: native[key]["values"]["Linear probe"] for key in ["short-216b", "long-1000b"]}
    wins = pairwise_win_rate(final, task_families)
    output = {"normalization": normalized, "probe_win_rates": wins,
              "normalization_definition": "Normalize each saved family mean to its observed 0–1 range across both runs' nine unique checkpoints, separately by readout; average five families equally",
              "win_rate_definition": "Compare the two final probes per task: win=1, exact tie=0.5, loss=0; average within families, then five families equally; display ×100",
              "source_native_csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "source_task_scores_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
              "families": task_families,
              "constant_family_policy": "Reject instead of silently choosing a normalization",
              "population": {identity: {key: value for key, value in native[identity].items() if key != "values"}
                             for identity in sorted(set(identities.values()))}}
    (HERE / "marin-summaries.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"final_probe_win_rates": {model: value["score"] for model, value in wins.items()},
                      "final_normalized": {readout: {key: data["checkpoints"][key]["score"] for key in final}
                                           for readout, data in normalized.items()}}, indent=2))


if __name__ == "__main__":
    main()
