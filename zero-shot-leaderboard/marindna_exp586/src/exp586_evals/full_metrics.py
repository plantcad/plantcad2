"""Reduce complete exp586 chunks with the original and bidirectional metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from exp586_evals.bidirectional import (
    target_log_likelihood_ratio,
    two_sided_probabilities,
)
from exp586_evals.protocol import original

DNA_INDEX = {base: index for index, base in enumerate("ACGT")}


def records(directory: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def merge(
    plan: list[dict], directory: Path, rows: int, *, task: str | None = None
) -> dict[str, np.ndarray]:
    protocol = original()
    expected = [piece for piece in plan if task is None or piece["task"] == task]
    observed = records(directory)
    if task is not None:
        observed = [piece for piece in observed if piece["task"] == task]
    expected_keys = {piece["key"] for piece in expected}
    observed_keys = {piece["key"] for piece in observed}
    if expected_keys != observed_keys or len(observed_keys) != len(observed):
        raise ValueError(
            f"Chunk coverage differs: missing={sorted(expected_keys - observed_keys)}, "
            f"extra={sorted(observed_keys - expected_keys)}"
        )
    return protocol.aggregate.merge_arrays(observed, directory, rows)


def bidirectional_task(
    spec: dict, frame: pd.DataFrame, arrays: dict[str, np.ndarray]
) -> dict:
    protocol = original()
    if spec["kind"] == "sv":
        raise ValueError("The historical SV metric has no centered bidirectional form")
    probabilities = two_sided_probabilities(
        arrays["left__probs"], arrays["right_reverse_complement__probs"]
    )
    flattened = probabilities.reshape(-1, 4)
    true = protocol.tasks.EVAL._compute_true_tokens_from_seq(
        frame["sequence"], spec["positions"]
    )
    if spec["kind"] == "conservation":
        scores = protocol.tasks.EVAL._refprob_scores(
            frame, flattened, spec["positions"][0], "sequence"
        )
    elif spec["kind"] == "motif":
        predicted = np.asarray(protocol.tasks.EVAL.NUCLEOTIDES)[
            flattened.argmax(axis=1)
        ].reshape(len(frame), -1)
        scores = np.all(predicted == true.reshape(len(frame), -1), axis=1).astype(
            np.float32
        )
    else:
        scores = protocol.tasks.EVAL._avg_trueprob_scores(
            flattened, true, len(spec["positions"])
        )
    if spec["kind"] == "motif":
        metrics = {
            "token_accuracy": protocol.tasks.EVAL._metric_token_accuracy(
                flattened, true
            ),
            "motif_accuracy": protocol.tasks.EVAL._metric_motif_accuracy(
                flattened, true, len(spec["positions"])
            ),
        }
        primary = "motif_accuracy"
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics = {
            "auroc": float(roc_auc_score(frame["label"].astype(int), scores)),
            "auprc": float(average_precision_score(frame["label"].astype(int), scores)),
        }
        primary = "auroc"
    return {
        "method": "equal-weight product of forward and aligned reverse-complement A/C/G/T probabilities",
        "samples": len(frame),
        "primary_metric": primary,
        "value": metrics[primary],
        "metrics": metrics,
    }


def group_scores(rows: list[dict], field: str) -> dict[str, float]:
    categories = (
        "Conservation",
        "Masked motif",
        "Core/non-core",
        "Structural variant",
    )
    result = {
        category: float(
            np.mean([row[field] for row in rows if row["category"] == category])
        )
        for category in categories
        if any(
            row["category"] == category and row.get(field) is not None for row in rows
        )
    }
    result["Composite"] = float(np.mean(list(result.values())))
    return result


def task_metrics(preparation: dict, profile_root: Path, sequence_length: int) -> dict:
    protocol = original()
    specs = {spec["key"]: spec for spec in protocol.tasks.LEADERBOARD_TASKS}
    tasks, rows = {}, []
    for key, spec in specs.items():
        if sequence_length < 8192 and spec["kind"] == "sv":
            continue
        frame = protocol.inputs.read_frame(preparation, spec)
        arrays = merge(
            preparation["plan"], profile_root / "tasks/chunks", len(frame), task=key
        )
        natural = protocol.aggregate.recompute_task(spec, frame, arrays)
        poe = None if spec["kind"] == "sv" else bidirectional_task(spec, frame, arrays)
        tasks[key] = {"natural": natural, "bidirectional_poe": poe}
        category, label = protocol.aggregate.LABELS[key]
        rows.append(
            {
                "key": key,
                "category": category,
                "task": label,
                "metric": natural["primary_metric"],
                "samples": len(frame),
                "natural": natural["best_value"],
                "natural_context": natural["best_context"],
                "left": natural["contexts"]["left"]["metrics"][
                    natural["primary_metric"]
                ],
                "reverse_complement": natural["contexts"]
                .get("right_reverse_complement", {})
                .get("metrics", {})
                .get(natural["primary_metric"]),
                "bidirectional_poe": None if poe is None else poe["value"],
            }
        )
    return {
        "sequence_length": sequence_length,
        "rows": rows,
        "tasks": tasks,
        "groups": {
            "natural": group_scores(rows, "natural"),
            "bidirectional_poe": group_scores(rows, "bidirectional_poe"),
        },
    }


def correlation(frame: pd.DataFrame, column: str) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(frame["AF"], frame[column]).statistic),
        "spearman": float(spearmanr(frame["AF"], frame[column]).statistic),
    }


def af_metrics(preparation: dict, profile_root: Path, output: Path) -> dict:
    protocol = original()
    spec = {"file": "maize-af.parquet"}
    frame = protocol.inputs.read_frame(preparation, spec)
    arrays = merge(preparation["af_plan"], profile_root / "af/chunks", len(frame))
    if not np.array_equal(arrays["row_id"], frame["row_id"].to_numpy()):
        raise ValueError("AF predictions do not align with immutable membership")
    for name, values in arrays.items():
        if name != "row_id" and values.ndim == 1:
            frame[name] = values
    frame["llr_acgt_avg"] = (frame["llr_acgt_fwd"] + frame["llr_acgt_rc"]) / 2
    frame["llr_full_vocab_avg"] = (
        frame["llr_full_vocab_fwd"] + frame["llr_full_vocab_rc"]
    ) / 2
    reference = np.asarray([DNA_INDEX[base.upper()] for base in frame["ref"]])
    alternate = np.asarray([DNA_INDEX[base.upper()] for base in frame["alt"]])
    frame["llr_target_poe"] = target_log_likelihood_ratio(
        arrays["target_probs_fwd"],
        arrays["target_probs_rc"],
        reference,
        alternate,
    )
    score_columns = ("llr_acgt_avg", "llr_full_vocab_avg", "llr_target_poe")
    metrics = {
        column: {
            "overall": correlation(frame, column),
            "by_consequence": {
                str(name): correlation(group, column)
                for name, group in frame.groupby("consequence", sort=True)
            },
        }
        for column in score_columns
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {
        "rows": len(frame),
        "scores": metrics,
        "predictions": output.name,
        "natural_method": "mean forward/reverse-complement full causal suffix LLR",
        "bidirectional_method": "equal-weight product at the variant target",
    }


def reduce(preparation_path: Path, directory: Path, output: Path) -> None:
    preparation = json.loads(preparation_path.read_text())
    preparation["samples"] = str(preparation_path.parent.resolve())
    output.mkdir(parents=True, exist_ok=True)
    profiles = {}
    for name, length in (("context-8192", 8192), ("context-2048", 2048)):
        profile_output = output / name
        profile_output.mkdir(exist_ok=True)
        tasks = task_metrics(preparation, directory / name, length)
        af = af_metrics(
            preparation, directory / name, profile_output / "af-predictions.parquet"
        )
        profiles[name] = {"tasks": tasks, "allele_frequency": af}
        pd.DataFrame(tasks["rows"]).to_csv(
            profile_output / "tasks.tsv", sep="\t", index=False
        )
    result = {
        "schema": 1,
        "preparation": str(preparation_path),
        "profiles": profiles,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reduce(**vars(args))


if __name__ == "__main__":
    main()
