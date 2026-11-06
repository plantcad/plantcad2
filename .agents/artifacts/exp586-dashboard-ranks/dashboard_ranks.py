"""Dashboard-compatible percentiles against an unchanged baseline reference."""

from math import isfinite
from statistics import mean


def percentile(score: float, reference: list[float]) -> float:
    """Average one-based ranks for matches; empirical percentile between matches.

    Matches pandas Series.rank(pct=True) for every reference member and
    scipy.stats.percentileofscore(kind="rank") for external model scores.
    External models never enter the reference or change baseline ranks.
    """
    if not reference or not all(isfinite(x) for x in [score, *reference]):
        raise ValueError("Scores and a nonempty reference must be finite")
    below = sum(x < score for x in reference)
    equal = sum(x == score for x in reference)
    return 100 * (below + (equal + 1) / 2 if equal else below) / len(reference)


def aggregate(scores: dict[str, float], reference: dict[str, dict[str, float]],
              families: dict[str, list[str]]) -> dict:
    """Give every task equal weight, as in the unfiltered dashboard."""
    tasks = [task for group in families.values() for task in group]
    if not tasks or len(tasks) != len(set(tasks)) or any(not group for group in families.values()):
        raise ValueError("Every task must belong to exactly one nonempty family")
    if not reference or set(scores) != set(tasks) or any(set(row) != set(tasks) for row in reference.values()):
        raise ValueError("Every model must cover the complete task manifest")
    ranks = {task: percentile(scores[task], [row[task] for row in reference.values()]) for task in sorted(tasks)}
    return {"score": mean(ranks.values()), "tasks": ranks,
            "families": {family: mean(ranks[task] for task in group) for family, group in families.items()}}
