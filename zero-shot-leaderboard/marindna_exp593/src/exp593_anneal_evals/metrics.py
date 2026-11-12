"""Native-scale task reducers; retain historical TF metrics and exact alternatives."""

from __future__ import annotations

import numpy as np

from exp593_anneal_evals.scoring import exact_task_scores


def complete_tf_arrays(protocol, spec, frame, arrays):
    """Derive legacy score columns from saved probabilities with its own helpers.

    FIM workers intentionally return probabilities. The historical reducer also
    requires per-row score columns, even when reducing motif accuracy. Deriving
    these here preserves the immutable inference output and original N handling.
    """
    result = dict(arrays)
    for context in spec["contexts"]:
        key = context + "__scores"
        if key in result:
            continue
        evaluation = protocol.tasks.EVAL
        sequences, positions = evaluation._transform_sequences_and_positions(
            frame.sequence, spec["positions"], context
        )
        probabilities = result[context + "__probs"].reshape(-1, 4)
        true = evaluation._compute_true_tokens_from_seq(sequences, positions)
        if spec["kind"] == "conservation":
            work = frame.copy()
            work["sequence"] = sequences
            values = evaluation._refprob_scores(
                work, probabilities, positions[0], "sequence"
            )
        elif spec["kind"] == "motif":
            predicted = np.asarray(evaluation.NUCLEOTIDES)[
                probabilities.argmax(axis=-1)
            ].reshape(len(frame), -1)
            values = np.all(predicted == true.reshape(len(frame), -1), axis=-1).astype(
                np.float32
            )
        else:
            values = evaluation._avg_trueprob_scores(
                probabilities, true, len(positions)
            )
        result[key] = values
    return result


def exact_motif_accuracy(matches, answers):
    """Retain the historical whole-motif ambiguity mask for exhaustive MAPs."""
    valid = np.asarray([all(base in "acgt" for base in a) for a in answers])
    return float(np.asarray(matches)[valid].mean()) if valid.any() else 0.0


def task_metrics(preparation, profile_root, sequence_length, variant):
    from exp586_evals.full_metrics import bidirectional_task, merge
    from exp586_evals.protocol import original
    from sklearn.metrics import average_precision_score, roc_auc_score

    protocol = original()
    rows = []
    details = {}
    for spec in protocol.tasks.LEADERBOARD_TASKS:
        if spec["kind"] == "sv" and variant.endswith("-exact"):
            continue
        frame = protocol.inputs.read_frame(preparation, spec)
        arrays = merge(
            preparation["plan"],
            profile_root / "tasks/chunks",
            len(frame),
            task=spec["key"],
        )
        if not np.array_equal(arrays["row_id"], np.arange(len(frame))):
            raise ValueError("Predictions do not preserve source membership")
        orientations = {}
        historical = None
        metric = (
            "accuracy"
            if spec["kind"] == "motif"
            else "average_precision"
            if spec["kind"] == "sv"
            else "auroc"
        )
        if variant.endswith("-exact"):
            answers = [
                "".join(s[p].lower() for p in spec["positions"]) for s in frame.sequence
            ]
            scores = exact_task_scores(arrays, answers)
            for name, values in scores.items():
                if spec["kind"] == "motif":
                    orientations[name] = {
                        "accuracy": exact_motif_accuracy(values["exact_match"], answers)
                    }
                else:
                    orientations[name] = {
                        "auroc_mean_marginal": float(
                            roc_auc_score(frame.label, values["mean_marginal"])
                        ),
                        "auroc_joint": float(
                            roc_auc_score(frame.label, values["joint"])
                        ),
                        "average_precision_mean_marginal": float(
                            average_precision_score(
                                frame.label, values["mean_marginal"]
                            )
                        ),
                        "average_precision_joint": float(
                            average_precision_score(frame.label, values["joint"])
                        ),
                    }
            primary = "accuracy" if spec["kind"] == "motif" else "auroc_mean_marginal"
        else:
            if spec["kind"] != "sv":
                arrays = complete_tf_arrays(protocol, spec, frame, arrays)
            saved = protocol.aggregate.recompute_task(spec, frame, arrays)
            historical = {
                "value": saved["best_value"],
                "context": saved["best_context"],
                "primary_metric": saved["primary_metric"],
                "rule": "Original #586 reducer and historical best-strand convention; keep separately from fixed strand fusion",
            }
            for context, value in saved["contexts"].items():
                name = "forward" if context == "left" else "reverse_complement"
                orientations[name] = {
                    metric: float(value["metrics"][saved["primary_metric"]])
                }
            if spec["kind"] != "sv":
                orientations["forward_rc_product"] = {
                    metric: float(bidirectional_task(spec, frame, arrays)["value"])
                }
            elif variant.startswith("fim-"):
                for name in ["forward", "reverse_complement", "forward_rc_product"]:
                    orientations[name] = {
                        "average_precision": float(
                            average_precision_score(
                                frame.label, arrays[name + "__scores"]
                            )
                        )
                    }
            primary = metric
        category, label = protocol.aggregate.LABELS[spec["key"]]
        row = {
            "key": spec["key"],
            "category": category,
            "task": label,
            "samples": len(frame),
            "primary_metric": primary,
            "orientations": orientations,
            "historical": historical,
        }
        rows.append(row)
        details[spec["key"]] = row
    groups = {}
    for orientation in ["forward", "reverse_complement", "forward_rc_product"]:
        families = {}
        for category in dict.fromkeys(r["category"] for r in rows):
            selected = [
                r
                for r in rows
                if r["category"] == category and orientation in r["orientations"]
            ]
            if selected:
                keys = set.intersection(
                    *(set(r["orientations"][orientation]) for r in selected)
                )
                families[category] = {
                    k: float(
                        np.mean([r["orientations"][orientation][k] for r in selected])
                    )
                    for k in keys
                }
        groups[orientation] = families
    return {
        "sequence_length": sequence_length,
        "variant": variant,
        "rows": rows,
        "tasks": details,
        "groups": groups,
        "output_contract": "lowercase a/c/g/t only; no case folding",
        "decoding": "complete candidate enumeration"
        if variant.endswith("-exact")
        else "historical teacher-forced per-position scoring",
    }
