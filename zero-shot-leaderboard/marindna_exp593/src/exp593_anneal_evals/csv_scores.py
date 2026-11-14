"""Export task scores from immutable, complete final-evaluation snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from exp593_anneal_evals import TRIALS
from exp593_evals.artifacts import fetch

COLUMNS = (
    "model",
    "run_id",
    "checkpoint_step",
    "training_input_tokens",
    "task_family",
    "task",
    "species",
    "context_length",
    "evaluation_type",
    "readout",
    "strand_strategy",
    "metric",
    "score",
    "n_examples",
)
LABELS = {
    "181b": "398B FIM",
    "364b": "532B FIM",
    "545b": "665B FIM",
    "control": "665B AR control",
}
FAMILIES = {
    "conservation": "Conservation",
    "motif": "Motifs",
    "core": "Core/non-core",
    "sv": "SV impact",
    "af": "Allele frequency",
}
STRANDS = {
    "forward": "forward",
    "reverse_complement": "reverse_complement",
    "forward_rc_product": "forward_rc_fusion",
}
VARIANTS = {
    "ar-lower-control": "ar_teacher_forced",
    "fim-minimal-control": "fim_teacher_forced",
    "ar-lower-exact": "ar_exhaustive",
    "fim-minimal-exact": "fim_exhaustive",
}


def species(task: str) -> str:
    if task == "conservation_andropogoneae":
        return "Sorghum bicolor"
    if task.startswith("conservation_poaceae_"):
        return "Pharus latifolius"
    if task == "sv_impact":
        return "Arabidopsis thaliana"
    if "maize" in task:
        return "Zea mays"
    if "tomato" in task:
        return "Solanum lycopersicum"
    raise ValueError(f"Unknown task species: {task}")


def model_rows(key: str, observation: dict, motif_counts: dict) -> list[dict]:
    trial = TRIALS[key]
    checkpoint = observation["checkpoint"]
    if not observation["complete"] or not checkpoint["final"]:
        raise ValueError("Only complete, final-checkpoint evaluations can be exported")
    if (
        checkpoint["run_id"] != trial.run_id
        or checkpoint["input_tokens"] != trial.stage_tokens
    ):
        raise ValueError(
            "Snapshot does not match the selected training identity/horizon"
        )
    if len(observation["jobs"]) != 27 or not all(
        p["complete"] for p in observation["phases"].values()
    ):
        raise ValueError("Final evaluation is missing expected analysis results")
    rows = []

    def add(task, kind, evaluation_type, readout, strand, metric, score, n):
        if not math.isfinite(score) or not isinstance(n, int) or n <= 0:
            raise ValueError("Scores must be finite and counts positive integers")
        rows.append(
            dict(
                zip(
                    COLUMNS,
                    (
                        LABELS[key],
                        trial.run_id,
                        checkpoint["step"],
                        trial.source_tokens + checkpoint["input_tokens"],
                        FAMILIES[kind],
                        task,
                        species(task),
                        8192,
                        evaluation_type,
                        readout,
                        strand,
                        metric,
                        score,
                        n,
                    ),
                    strict=True,
                )
            )
        )

    for name, job in sorted(observation["jobs"].items()):
        result = job["result"]
        profile, variant, kind, *task_parts = name.split("/")
        if kind == "tasks":
            assert profile == "context-8192" and variant in VARIANTS
            for row in result["rows"]:
                task = row["key"]
                family = task.split("_", 1)[0]
                n = row["samples"]
                if family == "motif":
                    counts = motif_counts[task]
                    assert n == counts["source_rows"]
                    n = counts["scored_rows"]
                for orientation, metrics in row["orientations"].items():
                    for original_metric, score in metrics.items():
                        metric = original_metric
                        readout = VARIANTS[variant]
                        if family == "conservation" and metric.endswith("_joint"):
                            continue
                        for suffix in ("_mean_marginal", "_joint"):
                            if metric.endswith(suffix):
                                metric = metric.removesuffix(suffix)
                                # A one-base target has identical joint/marginal scores.
                                if family != "conservation":
                                    readout += (
                                        "_marginal"
                                        if suffix == "_mean_marginal"
                                        else "_joint"
                                    )
                        add(
                            task,
                            family,
                            "zero_shot",
                            readout,
                            STRANDS[orientation],
                            metric,
                            score,
                            n,
                        )
                if variant == "ar-lower-control":
                    h = row["historical"]
                    metric = {
                        "motif_accuracy": "accuracy",
                        "auprc": "average_precision",
                    }.get(h["primary_metric"], h["primary_metric"])
                    strand = {
                        "left": "forward",
                        "right_reverse_complement": "reverse_complement",
                    }[h["context"]]
                    add(
                        task,
                        family,
                        "zero_shot",
                        "ar_historical_teacher_forced",
                        strand,
                        metric,
                        h["value"],
                        n,
                    )
        elif kind == "af":
            assert profile == "context-8192" and variant in (
                "ar-lower-control",
                "fim-minimal-control",
            )
            readouts = (
                {
                    "llr_full_vocab_avg": (
                        "ar_suffix_llr_full_vocab",
                        "forward_rc_score_average",
                    ),
                    "llr_acgt_avg": (
                        "ar_suffix_llr_lower_four",
                        "forward_rc_score_average",
                    ),
                    "llr_target_poe": ("ar_target_site_odds", "forward_rc_fusion"),
                }
                if variant.startswith("ar-")
                else {
                    "llr_full_vocab_avg": (
                        "fim_site_odds_fp32",
                        "forward_rc_score_average",
                    ),
                    "llr_acgt_avg": ("fim_site_odds_bf16", "forward_rc_score_average"),
                    "llr_target_poe": ("fim_site_odds_bf16", "forward_rc_fusion"),
                }
            )
            for column, (readout, strand) in readouts.items():
                for metric, score in result["scores"][column]["overall"].items():
                    add(
                        "maize_allele_frequency",
                        "af",
                        "zero_shot",
                        readout,
                        strand,
                        metric,
                        score,
                        result["rows"],
                    )
        elif kind == "probe":
            assert (
                profile == "frozen-probes"
                and variant == "ar-lower-features"
                and len(task_parts) == 1
            )
            task = task_parts[0]
            family = task.split("_", 1)[0]
            if family == "motif":
                add(
                    task,
                    family,
                    "linear_probe",
                    "frozen_boundary_classifier",
                    "forward_rc_concatenation",
                    "accuracy",
                    result["test_score"],
                    result["test_rows"],
                )
            else:
                for contract, strand in (
                    ("directional", "forward_rc_concatenation"),
                    ("strand_average", "forward_rc_feature_average"),
                ):
                    r = result["readouts"][contract]
                    add(
                        task,
                        family,
                        "linear_probe",
                        "frozen_sv_ref_delta_classifier"
                        if family == "sv"
                        else "frozen_boundary_classifier",
                        strand,
                        "average_precision" if family == "sv" else "auroc",
                        r["test_score"],
                        r["test_rows"],
                    )
        elif kind == "af-probe":
            for contract, metrics in result["metrics"].items():
                feature, strand = contract.split("/")
                add(
                    "maize_allele_frequency",
                    "af",
                    "linear_probe",
                    "frozen_variant_site_ridge"
                    if feature == "variant_token"
                    else "frozen_whole_window_ridge",
                    "forward_rc_concatenation"
                    if strand == "directional"
                    else "forward_rc_feature_average",
                    "spearman",
                    metrics["overall"]["probe"]["value"],
                    result["fits"][contract]["test_rows"],
                )
        else:
            raise ValueError(f"Unknown analysis kind: {name}")
    identities = [
        tuple(r[k] for k in COLUMNS if k not in ("score", "n_examples")) for r in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate task/model/readout/strand/metric rows")
    return sorted(
        rows,
        key=lambda r: tuple(
            str(r[k])
            for k in (
                "evaluation_type",
                "task_family",
                "task",
                "readout",
                "strand_strategy",
                "metric",
            )
        ),
    )


def main() -> None:
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--refresh-credentials", action="store_true")
    args = p.parse_args()
    if args.refresh_credentials:
        from exp593_evals.storage import client

        client(refresh=True)
    manifest = json.loads(args.manifest.read_text())
    counts = {}
    for task, record in manifest["motif_memberships"].items():
        path = fetch(record, args.cache / record["sha256"])
        targets = pq.read_table(path, columns=["target_motif"])[
            "target_motif"
        ].to_pylist()
        counts[task] = {
            "source_rows": len(targets),
            "scored_rows": sum(
                bool(s) and all(b in "ACGT" for b in s.upper()) for s in targets
            ),
        }
    rows = []
    for key, record in manifest["score_snapshots"].items():
        observation = json.loads(
            fetch(record, args.cache / record["sha256"]).read_text()
        )
        rows.extend(model_rows(key, observation, counts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "models": list(manifest["score_snapshots"]),
                "motif_counts": counts,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
