#!/usr/bin/env python3
"""
Compare local causal zero-shot results against the public PlantCAD2 leaderboard.

Pulls data/results.csv from the leaderboard Space and joins it to the local
per-context-mode manifests, taking max over the two real strands locally (the
leaderboard's own Evo2 row uses the same best=fwd/rc protocol, per its `note`
column). The leaderboard's "Masked motif" metric is motif accuracy -- verified
against the NTv3-650M and Carbon-3B rows, whose local runs live in this repo.

    python compare_leaderboard.py --results_dir results/zero-shot-leaderboard/evo2_20b
"""

import argparse
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

SPACE = "plantcad/plantcad2-zeroshot-leaderboard"
RESULTS_URL = f"https://huggingface.co/spaces/{SPACE}/resolve/main/data/results.csv"

# local (task, split) -> leaderboard task_id, for the Masked motif category
TASK_MAP = {
    ("tis_recovery", "test_maize"): "motif_maize_start_sites",
    ("tts_recovery", "test_maize"): "motif_maize_stop_sites",
    ("donor_recovery", "test_maize"): "motif_maize_donor",
    ("acceptor_recovery", "test_maize"): "motif_maize_acceptor",
    ("tis_recovery", "test_tomato"): "motif_tomato_start_sites",
    ("tts_recovery", "test_tomato"): "motif_tomato_stop_sites",
    ("donor_recovery", "test_tomato"): "motif_tomato_donor",
    ("acceptor_recovery", "test_tomato"): "motif_tomato_acceptor",
}
DEFAULT_MODES = ["left", "right_reverse_complement"]


def load_local(results_dir: Path, modes):
    """-> {leaderboard_task_id: (motif_acc, best_mode)} using max over `modes`."""
    out = {}
    for d in sorted((results_dir / "motif_acc").iterdir()):
        if not d.is_dir():
            continue
        by_mode = {}
        for mf in d.glob("*.manifest.json"):
            m = json.loads(mf.read_text())
            if m.get("context_mode") in modes:
                by_mode[m["context_mode"]] = m.get("metrics", {}).get("motif_accuracy")
        by_mode = {k: v for k, v in by_mode.items() if v is not None}
        if not by_mode:
            continue
        mf0 = json.loads(next(iter(d.glob("*.manifest.json"))).read_text())
        key = TASK_MAP.get((mf0.get("task"), mf0.get("split")))
        if key is None:
            continue
        best = max(by_mode, key=by_mode.get)
        out[key] = (by_mode[best], "rc" if best != "left" else "fwd", len(by_mode))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results/zero-shot-leaderboard/evo2_20b")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--label", default="evo2_20b (ours)")
    ap.add_argument("--compare", default="Evo2,PlantCAD2-L,PlantCAD2.5-L",
                    help="comma-separated leaderboard model ids")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    lb = pd.read_csv(io.StringIO(urllib.request.urlopen(RESULTS_URL).read().decode()))
    local = load_local(Path(args.results_dir), modes)
    compare = [c.strip() for c in args.compare.split(",")]

    rows = []
    for (task, split), tid in TASK_MAP.items():
        row = {"task": task.replace("_recovery", ""), "split": split.replace("test_", "")}
        for model in compare:
            hit = lb[(lb.model == model) & (lb.task_id == tid)]
            row[model] = float(hit.value.iloc[0]) if len(hit) else None
        got = local.get(tid)
        row[args.label] = got[0] if got else None
        row["strand"] = got[1] if got else ""
        row["n_modes"] = got[2] if got else 0
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["split", "task"])

    def f(v, n_modes=2):
        if v is None or pd.isna(v):
            return "--"
        return f"{v:.3f}" + ("*" if n_modes < 2 else "")

    lines = [
        "# evo2_20b vs PlantCAD2 leaderboard -- masked-motif recovery",
        "",
        "Metric: motif accuracy (exact match over all motif positions).",
        f"Local column is max over {', '.join(modes)}; the leaderboard's own Evo2 row",
        "(evo2_7b) uses the same best=fwd/rc protocol. `*` = only one strand available.",
        "",
        "| Task | Split | " + " | ".join(compare) + f" | {args.label} | best strand | vs PC2.5-L |",
        "| :--- | :--- | " + " | ".join(":---" for _ in compare) + " | :--- | :--- | :--- |",
    ]
    for _, r in df.iterrows():
        ours, ref = r[args.label], r.get("PlantCAD2.5-L")
        delta = "--" if ours is None or ref is None or pd.isna(ours) or pd.isna(ref) else f"{ours - ref:+.3f}"
        lines.append(
            f"| {r.task} | {r.split} | "
            + " | ".join(f(r[m]) for m in compare)
            + f" | {f(ours, r.n_modes)} | {r.strand or '--'} | {delta} |"
        )

    md = "\n".join(lines) + "\n"
    print(md)
    if args.output:
        Path(args.output).write_text(md)
        print(f"(wrote {args.output})")


if __name__ == "__main__":
    main()
