"""Frozen-reference percentiles, with equal weights for task families."""

from math import isfinite
from statistics import mean


def percentile(score: float, reference: list[float]) -> float:
    """Higher is better; ties receive half credit, including reference members."""
    if not reference or not all(isfinite(x) for x in [score, *reference]):
        raise ValueError("Scores and a nonempty reference must be finite")
    return 100 * (sum(x < score for x in reference) + .5 * sum(x == score for x in reference)) / len(reference)


def aggregate(scores: dict[str, float], reference: dict[str, dict[str, float]],
              families: dict[str, list[str]]) -> dict:
    """Rank tasks first, average within families, then weight families equally."""
    expected = {task for tasks in families.values() for task in tasks}
    if sum(map(len, families.values())) != len(expected) or any(not tasks for tasks in families.values()):
        raise ValueError("Every task must belong to exactly one nonempty family")
    if set(scores) != expected or not reference:
        raise ValueError("Candidate tasks must match the complete task manifest")
    if any(set(row) != expected for row in reference.values()):
        raise ValueError("Every reference member must cover every task")
    task_ranks = {task: percentile(scores[task], [row[task] for row in reference.values()]) for task in sorted(expected)}
    family_ranks = {family: mean(task_ranks[task] for task in tasks) for family, tasks in families.items()}
    return {"score": mean(family_ranks.values()), "families": family_ranks, "tasks": task_ranks}
