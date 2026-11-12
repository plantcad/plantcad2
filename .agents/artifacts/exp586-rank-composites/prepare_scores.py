"""Recompute cross-family summaries without changing saved native scores."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np
from rank_scores import aggregate

HERE = Path(__file__).resolve().parent
OLD = HERE.with_name("exp586-checkpoint-trajectories")
REPO = HERE.parents[2]
RUNS = Path("/home/exedev/repos/plantcad2-br/codex-exp586-probes/zero-shot-leaderboard/marindna_exp586/runs")
FAMILIES = {
    "Conservation": ["conservation_andropogoneae", "conservation_poaceae_non_tis", "conservation_poaceae_tis"],
    "Motifs": [f"motif_{species}_{kind}" for species in ["maize", "tomato"] for kind in ["tis", "tts", "donor", "acceptor"]],
    "Core/non-core": [f"core_{species}_{kind}" for species in ["maize", "tomato"] for kind in ["tis", "tts", "donor", "acceptor"]],
    "SV impact": ["sv_impact"],
}
WITH_AF = {**FAMILIES, "Allele frequency": ["af"]}
READOUTS = ["Zero-shot", "Linear probe"]
SOURCE_READOUTS = {"Zero-shot": "Zero-shot likelihood", "Linear probe": "Linear probe"}
SOURCE_SHA = {
    "216B-token run": "1da421c0bb53052a705deb1bf523c83f14a18e8d2bcc3c288dfe24edc3ee0b1e",
    "1T-token run": "e4769f8a6f6d6a4a77993d04062e8ecf7413aa18374180e284fd513b775cbe06",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, data: dict) -> None:
    (HERE / name).write_text(json.dumps(data, indent=2) + "\n")


def published_id(task: str) -> str:
    if task.startswith("conservation_"):
        return task.replace("conservation_", "cons_").replace("non_tis", "nontis")
    if task.startswith("motif_"):
        return task.replace("_tis", "_start_sites").replace("_tts", "_stop_sites")
    if task.startswith("core_"):
        return task.replace("core_", "coreclass_", 1)
    return "sv_impact_auprc"


def main() -> None:
    original_rows = list(csv.DictReader((OLD / "all-category-scores.csv").open()))
    assert sha(OLD / "all-category-scores.csv") == "3f350bd6668ba71046f618a4dd44cab8cd86114927b32d551a8e32634d01a6ca"
    motif_path = OLD / "motif-probe-summary.json"
    motifs = json.loads(motif_path.read_text())
    observations, unique, sources = {}, {}, {}
    for run, folder, prefix in [("216B-token run", "exp586-checkpoint-trajectory-022t", "short"),
                                ("1T-token run", "exp586-checkpoint-trajectory", "long")]:
        path = RUNS / folder / "results.json"
        assert sha(path) == SOURCE_SHA[run]
        raw = json.loads(path.read_text())
        metadata = json.loads(path.with_name("provenance.json").read_text())["catalog"]["models"]
        sources[run] = {"results_sha256": sha(path), "catalog_sha256": sha(path.with_name("provenance.json")), "path": str(path)}
        for checkpoint, data in raw.items():
            identity = "base-000b" if checkpoint == "dna-000b" else checkpoint.replace("dna", prefix, 1)
            tasks = data["zero_shot"]["profiles"]["context-8192"]["tasks"]["rows"]
            assert len(tasks) == 20 and {row["key"] for row in tasks} == {task for tasks in FAMILIES.values() for task in tasks}
            values = {"Zero-shot": {row["key"]: row["natural"] for row in tasks}, "Linear probe": {}}
            for family, task_ids in FAMILIES.items():
                for task in task_ids:
                    values["Linear probe"][task] = (motifs[identity][task]["test_score"] if family == "Motifs"
                        else data["classification"]["tasks"][task]["readouts"]["strand_average" if family == "SV impact" else "directional"]["test_score"])
            for readout in READOUTS:
                af = next(row for row in original_rows if row["run"] == run and row["checkpoint"] == checkpoint
                          and row["metric"] == "Allele frequency" and row["readout"] == SOURCE_READOUTS[readout])
                values[readout]["af"] = float(af["score"])
                if readout == "Linear probe":
                    assert values[readout]["af"] == data["AF_probes"]["context-8192"]["metrics"]["variant_token/directional"]["overall"]["probe"]["value"]
                for family, ids in WITH_AF.items():
                    original = next(float(row["score"]) for row in original_rows if row["run"] == run and row["checkpoint"] == checkpoint
                                    and row["metric"] == family and row["readout"] == SOURCE_READOUTS[readout])
                    assert abs(original - mean(values[readout][task] for task in ids)) < 1e-14
            observation = {"id": identity, "checkpoint": metadata[checkpoint]["checkpoint"],
                           "DNA_tokens": metadata[checkpoint]["DNA_tokens"], "values": values}
            if identity in unique:
                assert unique[identity] == observation, "Shared starting checkpoint differs"
            unique[identity] = observation
            observations[run, checkpoint] = observation
    assert len(observations) == 10 and len(unique) == 9

    historical_path = REPO / "zero-shot-leaderboard/marindna_exp586/historical/exp472-lr5e4-full-coreweave-20260831.json"
    assert sha(historical_path) == "76e5bb907568e7b800e5803216f72495933d584e95bf2a03cbae597d7cc6391b"
    historical = json.loads(historical_path.read_text())
    baseline = {}
    for row in historical["published_rows"]:
        task = next(task for tasks in FAMILIES.values() for task in tasks if published_id(task) == row["task_id"])
        member = baseline.setdefault(row["model"], {})
        assert task not in member
        member[task] = float(row["value"])
    assert len(baseline) == 11 and sum(map(len, baseline.values())) == 220
    marin_reference = {readout: {key: row["values"][readout] for key, row in sorted(unique.items())} for readout in READOUTS}
    frozen = {"policy": "100 × (strictly lower reference scores + half the tied reference scores) / reference count; no rounding before ties",
              "aggregation": "Percentile per task → arithmetic mean per family → equally weighted mean of families",
              "Avg Rank": {"reference": baseline, "families": FAMILIES, "readout": "Zero-shot", "reference_revision": historical["published_revision"]},
              "Marin Avg Rank": {"reference": marin_reference, "families": WITH_AF, "members": {key: {k: v for k, v in row.items() if k != "values"} for key, row in sorted(unique.items())}},
              "fixed": True, "units": "Percentile points; 0–100; higher is better"}
    write("frozen-references.json", frozen)
    write("native-task-scores.json", {"checkpoints": unique, "sources": sources, "motif_summary_sha256": sha(motif_path)})

    rows = [{**row, "readout": next(key for key, value in SOURCE_READOUTS.items() if value == row["readout"])}
            for row in original_rows if row["metric"] in WITH_AF]
    assert len(rows) == 100
    task_rows, summary = [], {}
    for (run, checkpoint), observation in observations.items():
        summary.setdefault(run, {})[checkpoint] = {}
        for readout in READOUTS:
            result = aggregate(observation["values"][readout], marin_reference[readout], WITH_AF)
            summary[run][checkpoint][readout] = result
            rows.append({"run": run, "checkpoint": checkpoint, "DNA_tokens_B": observation["DNA_tokens"] / 1e9,
                         "metric": "Marin Avg Rank", "readout": readout, "score": result["score"]})
            for family, ids in WITH_AF.items():
                for task in ids:
                    task_rows.append({"run": run, "checkpoint": checkpoint, "readout": readout, "task": task,
                                      "family": family, "native_score": observation["values"][readout][task],
                                      "percentile": result["tasks"][task], "family_percentile": result["families"][family],
                                      "Marin Avg Rank": result["score"]})
    for filename, content in [("all-category-scores.csv", rows), ("task-percentiles.csv", task_rows)]:
        with (HERE / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(content[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(content)
    write("trajectory-ranks.json", summary)
    old_benchmark = json.loads((OLD / "benchmark-comparison-data.json").read_text())
    baseline_ids = {"PlantCAD2.5-Large": "PlantCAD2.5-L", "PlantCAD2-Large": "PlantCAD2-L", "PlantCAD2-Medium": "PlantCAD2-M",
                    "PlantCAD2-Small": "PlantCAD2-S", "Evo 2 20B": "evo2_20b"}
    rank_results, benchmark = {}, {"categories": [*FAMILIES, "Avg Rank"], "models": []}
    for index, model in enumerate(old_benchmark["models"]):
        if index == 0:
            scores = {row["key"]: row["selected_value"] for row in historical["results"]["056t"]["rows"]}
        elif index in [1, 2]:
            scores = {key: value for key, value in unique["short-216b" if index == 1 else "long-1000b"]["values"]["Zero-shot"].items() if key != "af"}
        else:
            scores = baseline[baseline_ids[model["label"]]]
        result = aggregate(scores, baseline, FAMILIES)
        np.testing.assert_allclose([mean(scores[task] for task in ids) for ids in FAMILIES.values()], model["scores"][:4], rtol=0, atol=1e-14)
        rank_results[model["label"]] = result
        benchmark["models"].append({**model, "scores": [*model["scores"][:4], result["score"]]})
    write("benchmark-comparison-data.json", benchmark)
    write("benchmark-ranks.json", rank_results)
    write("benchmark-comparison-provenance.json", {"frozen_reference_sha256": sha(HERE / "frozen-references.json"),
          "historical_archive_sha256": sha(historical_path), "original_benchmark_sha256": sha(OLD / "benchmark-comparison-data.json"),
          "native_categories_preserved": True, "fresh_model_evaluations": False, "live_post_changed": False,
          "aggregate": "Avg Rank against 11 frozen published models, equal weights for four task families; no AF"})
    print(json.dumps({"benchmark_Avg_Rank": {key: value["score"] for key, value in rank_results.items()},
                      "Marin_Avg_Rank": {run: {cp: {r: v["score"] for r, v in rr.items()} for cp, rr in vv.items()} for run, vv in summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
