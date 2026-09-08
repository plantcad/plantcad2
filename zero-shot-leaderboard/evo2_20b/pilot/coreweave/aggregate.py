"""Reassemble original row order, then apply full-task metrics and strand selection."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from common import PILOT, write_json
from checkpoints import FULL_ROWS
from inputs import read_frame

sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(PILOT / "sensitivity"))
import run_condition as scoring
from merge_recommended import LABELS
from partition import validate_task_coverage


def recompute_task(spec: dict, frame: pd.DataFrame, arrays: dict[str, np.ndarray]) -> dict:
    """Use the exact existing metric helpers, including motif ambiguous-base masks."""
    contexts = {}
    for context in spec["contexts"]:
        scores = arrays[f"{context}__scores"]
        if len(scores) != len(frame):
            raise ValueError("Score and fixture lengths differ")
        if spec["kind"] == "sv":
            metrics = {"auprc": float(average_precision_score(frame["label"].astype(int), scores))}
        elif spec["kind"] == "motif":
            sequences, positions = scoring.EVAL._transform_sequences_and_positions(frame["sequence"], spec["positions"], context)
            true = scoring.EVAL._compute_true_tokens_from_seq(sequences, positions)
            probs = arrays[f"{context}__probs"].reshape(-1, 4)
            metrics = {"token_accuracy": scoring.EVAL._metric_token_accuracy(probs, true), "motif_accuracy": scoring.EVAL._metric_motif_accuracy(probs, true, len(positions))}
        else:
            metrics = {"auroc": float(roc_auc_score(frame["label"].astype(int), scores)), "auprc": float(average_precision_score(frame["label"].astype(int), scores))}
        contexts[context] = {"metrics": metrics}
    primary = "auprc" if spec["kind"] == "sv" else "motif_accuracy" if spec["kind"] == "motif" else "auroc"
    best = max(spec["contexts"], key=lambda context: contexts[context]["metrics"][primary])
    return {"kind": spec["kind"], "samples": len(frame), "primary_metric": primary, "best_context": best, "best_value": contexts[best]["metrics"][primary], "contexts": contexts}


def merge_arrays(pieces: list[dict], chunk_dir: Path, expected_rows: int) -> dict[str, np.ndarray]:
    ordered = validate_task_coverage(pieces, expected_rows)
    parts = []
    for piece in ordered:
        with np.load(chunk_dir / f"{piece['key']}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        if any(len(array) != piece["stop"] - piece["start"] for array in arrays.values()):
            raise ValueError("Chunk array length mismatch")
        if parts and set(arrays) != set(parts[0]):
            raise ValueError("Inconsistent chunk array names")
        parts.append(arrays)
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def aggregate(root: Path, preparation: dict, worker_records: list[dict], nodes: int) -> dict:
    pieces = [json.loads(path.read_text()) for path in (root / "chunks").glob("*.json")]
    tasks = {}
    archive = {}
    for spec in scoring.LEADERBOARD_TASKS:
        task_pieces = [piece for piece in pieces if piece["task"] == spec["key"]]
        frame = read_frame(preparation, spec, columns=None if spec["kind"] == "motif" else ["label"])
        arrays = merge_arrays(task_pieces, root / "chunks", len(frame))
        task = recompute_task(spec, frame, arrays)
        for context in spec["contexts"]:
            source = [piece["result"]["contexts"][context] for piece in task_pieces]
            task["contexts"][context].update(timing=scoring._merge_timing([item["timing"] for item in source]), peak_reserved_gib=max(item["peak_reserved_gib"] for item in source), peak_allocated_gib=max(item["peak_allocated_gib"] for item in source))
        tasks[spec["key"]] = task
        archive.update({f"{spec['key']}__{key}": value for key, value in arrays.items()})
        print(json.dumps({"phase": "task_result", "task": spec["key"], "samples": len(frame), "value": task["best_value"], "context": task["best_context"]}), flush=True)
    rows = []
    for key, (category, label) in LABELS.items():
        task = tasks[key]
        primary = task["primary_metric"]
        rows.append({"key": key, "category": category, "task": label, "metric": primary, "samples": task["samples"], "selected_context": task["best_context"], "selected_value": task["best_value"], "left": task["contexts"]["left"]["metrics"][primary], "reverse_complement": task["contexts"].get("right_reverse_complement", {}).get("metrics", {}).get(primary)})
    timings = [context["timing"] for task in tasks.values() for context in task["contexts"].values()]
    started = min(worker["started_epoch"] for worker in worker_records)
    finished = max(worker["finished_epoch"] for worker in worker_records)
    wall = finished - started
    forwards = sum(timing["sequences"] for timing in timings)
    total_rows = sum(task["samples"] for task in tasks.values())
    if total_rows != preparation["sampling"]["total_rows"] or forwards != 2 * total_rows:
        raise ValueError(f"Incorrect full coverage/forward count: {total_rows} rows, {forwards} forwards")
    result = {"model": preparation["model"], "checkpoint": preparation["checkpoint"], "condition": "bf16-fa2-fp32-acgt-softmax-no-cache-b32", "platform": "coreweave-iris", "sampling": {**preparation["sampling"], "manifest_sha256": preparation["manifest_sha256"]}, "runtime": {"nodes": nodes, "gpus": nodes * 8, "measured_parallel_wall_seconds": wall, "scored_sequences": forwards, "scored_tokens": sum(timing["tokens"] for timing in timings), "cluster_sequences_per_second": forwards / wall, "h100_sequences_per_second": forwards / sum(worker["run_wall_seconds"] for worker in worker_records), "summed_forward_wall_seconds": sum(timing["wall_seconds"] for timing in timings), "full_leaderboard_rows": FULL_ROWS, "full_estimated_same_allocation_hours": wall * FULL_ROWS / total_rows / 3600}, "environments": [worker["environment"] for worker in worker_records], "flash_verification": [worker["flash_verification"] for worker in worker_records], "rows": rows, "tasks": tasks}
    write_json(root / "result.json", result)
    np.savez_compressed(root / "scores.npz", **archive)
    pd.DataFrame(rows).to_csv(root / "leaderboard.tsv", sep="\t", index=False)
    return result
