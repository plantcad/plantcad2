#!/usr/bin/env python3
"""
Summarise causal zero-shot eval runs into a markdown table.

Reads the per-context-mode manifests written by zero-shot-eval-causal.py and
reports max-over-context, restricted by default to the two real strands
(`left` = forward, `right_reverse_complement` = reverse). Missing runs are
reported rather than silently skipped, so a partial array shows up as such.

    python summarize.py --results_dir ../../results/zero-shot-leaderboard/evo2_20b
    python summarize.py --subcommand motif_acc --modes left,right_reverse_complement
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

# metric columns per subcommand directory
METRICS = {
    "motif_acc": [("token_accuracy", "Token Acc"), ("motif_accuracy", "Motif Acc")],
    "evo_cons": [("auroc", "AUROC"), ("auprc", "AUPRC")],
    "core_noncore": [("auroc", "AUROC"), ("auprc", "AUPRC")],
    "sv_effect": [("auprc", "AUPRC")],
}
# metric that decides max-over-context
PRIMARY = {
    "motif_acc": "motif_accuracy",
    "evo_cons": "auroc",
    "core_noncore": "auroc",
    "sv_effect": "auprc",
}


def load_runs(results_dir: Path, subcommand: str, modes: list[str]):
    """-> ({(task, split): {mode: metrics}}, modes seen on disk, dirs with no output).

    Output directories are created by the .sub before inference, so a directory
    holding no manifests is an array element that produced nothing -- report it
    rather than let it vanish from the table.
    """
    runs: dict = defaultdict(dict)
    seen_modes = set()
    empty_dirs = []
    root = results_dir / subcommand
    if not root.is_dir():
        return runs, seen_modes, empty_dirs
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        manifests = sorted(d.glob("*.manifest.json"))
        if not manifests:
            empty_dirs.append(d.name)
            continue
        for manifest in manifests:
            m = json.loads(manifest.read_text())
            mode = m.get("context_mode")
            seen_modes.add(mode)
            if mode not in modes:
                continue
            runs[(m.get("task"), m.get("split"))][mode] = m.get("metrics", {})
    return runs, seen_modes, empty_dirs


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "--"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results/zero-shot-leaderboard/evo2_20b")
    ap.add_argument("--subcommand", default="motif_acc", choices=sorted(METRICS))
    ap.add_argument("--modes", default="left,right_reverse_complement",
                    help="comma-separated context modes to consider for the max")
    ap.add_argument("--model", default="evo2_20b")
    ap.add_argument("--output", default="", help="write markdown here as well as stdout")
    args = ap.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    results_dir = Path(args.results_dir)
    runs, seen, empty_dirs = load_runs(results_dir, args.subcommand, modes)
    if not runs:
        raise SystemExit(f"No manifests under {results_dir / args.subcommand}")

    cols = METRICS[args.subcommand]
    primary = PRIMARY[args.subcommand]
    lines = [
        f"# {args.model} -- {args.subcommand}",
        "",
        f"Max over context modes: {', '.join(modes)}. "
        f"Modes present on disk: {', '.join(sorted(m for m in seen if m))}.",
        "",
        "| Task | Split | " + " | ".join(c[1] for c in cols) + " | Best strand |",
        "| :--- | :--- | " + " | ".join(":---" for _ in cols) + " | :--- |",
    ]

    incomplete = []
    per_mode_rows = []
    last_task = None
    for (task, split) in sorted(runs, key=lambda k: (k[0] or "", k[1] or "")):
        by_mode = runs[(task, split)]
        missing = [m for m in modes if m not in by_mode]
        if missing:
            incomplete.append((task, split, missing))
        if not by_mode:
            continue
        best_mode = max(by_mode, key=lambda m: by_mode[m].get(primary, float("-inf")))
        best = by_mode[best_mode]
        task_col = f"**{task}**" if task != last_task else ""
        last_task = task
        lines.append(
            f"| {task_col} | {split} | "
            + " | ".join(fmt(best.get(k)) for k, _ in cols)
            + f" | {'fwd' if best_mode == 'left' else 'rc'} |"
        )
        for m in modes:
            if m in by_mode:
                per_mode_rows.append((task, split, m, by_mode[m]))

    lines += ["", "## Per context mode", "",
              "| Task | Split | Mode | " + " | ".join(c[1] for c in cols) + " |",
              "| :--- | :--- | :--- | " + " | ".join(":---" for _ in cols) + " |"]
    for task, split, mode, met in per_mode_rows:
        lines.append(f"| {task} | {split} | {mode} | " + " | ".join(fmt(met.get(k)) for k, _ in cols) + " |")

    if incomplete or empty_dirs:
        lines += ["", "## Incomplete", ""]
        for task, split, missing in incomplete:
            lines.append(f"- `{task}` / `{split}`: missing {', '.join(missing)}")
        for name in empty_dirs:
            lines.append(f"- `{name}`: no output at all (array element produced nothing)")

    md = "\n".join(lines) + "\n"
    print(md)
    if args.output:
        Path(args.output).write_text(md)
        print(f"(wrote {args.output})")


if __name__ == "__main__":
    main()
