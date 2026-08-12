#!/usr/bin/env python3
"""
Compare local causal zero-shot results against the public PlantCAD2 leaderboard.

Pulls data/results.csv from the leaderboard Space and joins it to the local
per-context-mode manifests, taking max over the two real strands locally (the
leaderboard's own Evo2 row uses the same best=fwd/rc protocol for motif
recovery, per its `note` column).

Metric per category, verified against leaderboard rows whose local runs live in
this repo (NTv3-650M, Carbon-3B) and against the leaderboard's Evo2 row:
  motif_acc     motif accuracy (exact match over all motif positions)
  evo_cons      AUROC          core_noncore  AUROC          sv_effect  AUPRC

Only the 8192 bp leaderboard rows are used -- results.csv also carries a 512 bp
copy of every task, which is not what we ran.

    python compare_leaderboard.py --results_dir results/zero-shot-leaderboard/evo2_20b
    python compare_leaderboard.py --category evo_cons
"""

import argparse
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd

SPACE = "plantcad/plantcad2-zeroshot-leaderboard"
RESULTS_URL = f"https://huggingface.co/spaces/{SPACE}/resolve/main/data/results.csv"
CONTEXT_BP = 8192

# local (task, split) -> leaderboard task_id, per results subcommand directory
TASK_MAP = {
    "motif_acc": {
        ("tis_recovery", "test_maize"): "motif_maize_start_sites",
        ("tts_recovery", "test_maize"): "motif_maize_stop_sites",
        ("donor_recovery", "test_maize"): "motif_maize_donor",
        ("acceptor_recovery", "test_maize"): "motif_maize_acceptor",
        ("tis_recovery", "test_tomato"): "motif_tomato_start_sites",
        ("tts_recovery", "test_tomato"): "motif_tomato_stop_sites",
        ("donor_recovery", "test_tomato"): "motif_tomato_donor",
        ("acceptor_recovery", "test_tomato"): "motif_tomato_acceptor",
    },
    "evo_cons": {
        ("conservation_within_andropogoneae", "test"): "cons_andropogoneae",
        ("conservation_within_poaceae_tis", "test"): "cons_poaceae_tis",
        ("conservation_within_poaceae_non_tis", "test"): "cons_poaceae_nontis",
    },
    "core_noncore": {
        (f"{m}_core_noncore_classification", f"test_{sp}"): f"coreclass_{sp}_{m}"
        for sp in ("maize", "tomato")
        for m in ("tis", "tts", "donor", "acceptor")
    },
    "sv_effect": {
        ("structural_variant_effect_prediction", "test"): "sv_impact_auprc",
    },
}
METRIC = {
    "motif_acc": "motif_accuracy",
    "evo_cons": "auroc",
    "core_noncore": "auroc",
    "sv_effect": "auprc",
}
TITLE = {
    "motif_acc": ("Masked-motif recovery", "motif accuracy"),
    "evo_cons": ("Evolutionary conservation", "AUROC"),
    "core_noncore": ("Core vs non-core site classification", "AUROC"),
    "sv_effect": ("Structural-variant effect prediction", "AUPRC"),
}
DEFAULT_MODES = ["left", "right_reverse_complement"]
# Column headers only; lookups still use the leaderboard's own model ids. The
# leaderboard's "Evo2" row is evo2_7b, worth spelling out next to evo2_20b.
DISPLAY_NAMES = {"Evo2": "evo2_7b"}


def load_local(results_dir: Path, category: str, modes):
    """-> {leaderboard_task_id: (value, strand, n_modes)} using max over `modes`."""
    metric, task_map = METRIC[category], TASK_MAP[category]
    root = results_dir / category
    out = {}
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        by_mode, key = {}, None
        for mf in sorted(d.glob("*.manifest.json")):
            m = json.loads(mf.read_text())
            key = key or task_map.get((m.get("task"), m.get("split")))
            if m.get("context_mode") in modes:
                v = m.get("metrics", {}).get(metric)
                if v is not None:
                    by_mode[m["context_mode"]] = v
        if key is None or not by_mode:
            continue
        best = max(by_mode, key=by_mode.get)
        out[key] = (by_mode[best], "fwd" if best == "left" else "rc", len(by_mode))
    return out


def short_name(task: str) -> str:
    for suffix in ("_recovery", "_core_noncore_classification"):
        task = task.replace(suffix, "")
    return task.replace("conservation_within_", "").replace(
        "structural_variant_effect_prediction", "sv_impact"
    )


def build_table(lb, local, category, compare, label, modes, ours):
    rows = []
    for (task, split), tid in TASK_MAP[category].items():
        row = {"task": short_name(task), "split": split.replace("test_", "") or "test"}
        for model in compare:
            hit = lb[(lb.model == model) & (lb.task_id == tid)]
            row[model] = float(hit.value.iloc[0]) if len(hit) else None
        got = local.get(tid)
        row[label] = got[0] if got else None
        row["strand"] = got[1] if got else ""
        row["n_modes"] = got[2] if got else 0
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["split", "task"])

    def f(v, n_modes=2):
        if v is None or pd.isna(v):
            return "--"
        return f"{v:.3f}" + ("*" if n_modes < 2 else "")

    # `ours` is a PlantCAD model on the leaderboard; the local column is the
    # external model being evaluated, so the delta reads ours - theirs.
    headers = [
        f"{DISPLAY_NAMES.get(m, m)}" + (" (ours)" if m == ours else "") for m in compare
    ]
    heading, metric_name = TITLE[category]
    lines = [
        f"## {heading}",
        "",
        f"Metric: {metric_name}. Leaderboard values are the {CONTEXT_BP} bp context rows.",
        f"The {label} column is run locally, max over {', '.join(modes)}. "
        "`*` = only one strand available; `--` = not run locally.",
        "",
        "| Task | Split | " + " | ".join(headers) + f" | {label} | strand | {ours} - {label} |",
        "| :--- | :--- | " + " | ".join(":---" for _ in compare) + " | :--- | :--- | :--- |",
    ]
    for _, r in df.iterrows():
        theirs, ref = r[label], r.get(ours)
        delta = "--" if theirs is None or ref is None or pd.isna(theirs) or pd.isna(ref) else f"{ref - theirs:+.3f}"
        lines.append(
            f"| {r.task} | {r.split} | "
            + " | ".join(f(r[m]) for m in compare)
            + f" | {f(theirs, r.n_modes)} | {r.strand or '--'} | {delta} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results/zero-shot-leaderboard/evo2_20b")
    ap.add_argument("--category", default="all",
                    help="motif_acc / evo_cons / core_noncore / sv_effect / all")
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--label", default="evo2_20b",
                    help="name for the locally-evaluated (external) model column")
    ap.add_argument("--compare", default="Evo2,PlantCAD2-L,PlantCAD2.5-L",
                    help="comma-separated leaderboard model ids")
    ap.add_argument("--ours", default="PlantCAD2.5-L",
                    help="our leaderboard model; marked (ours) and used as the delta reference")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    compare = [c.strip() for c in args.compare.split(",")]
    categories = list(TASK_MAP) if args.category == "all" else [args.category]

    lb = pd.read_csv(io.StringIO(urllib.request.urlopen(RESULTS_URL).read().decode()))
    lb = lb[lb.context_bp == CONTEXT_BP]

    parts = [
        f"# {args.label} vs PlantCAD2 zero-shot leaderboard",
        "",
        f"{args.label} is an external model evaluated here with our causal harness; "
        f"{args.ours} and the other leaderboard columns are the published numbers.",
        "",
    ]
    for cat in categories:
        local = load_local(Path(args.results_dir), cat, modes)
        parts += [build_table(lb, local, cat, compare, args.label, modes, args.ours), ""]

    md = "\n".join(parts)
    print(md)
    if args.output:
        Path(args.output).write_text(md)
        print(f"(wrote {args.output})")


if __name__ == "__main__":
    main()
