"""Explain backend differences on real targets and unchanged native-scale task metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp593_evals.readouts import nucleotide_probabilities

BACKENDS = ["native", "torch-fa3", "vllm-fa3", "fast-fa3"]
VARIANTS = {
    "ar-lower": ["ar-lower-control", "ar-lower-folded"],
    "ar-upper": ["ar-upper-folded"],
    "fim-minimal": ["fim-minimal-folded"],
    "fim-metadata": ["fim-metadata-folded"],
}


def load_arrays(root: Path, backend: str, panel_hash: str, expected: set[int]):
    archives, owners, identity = [], {}, None
    for receipt_path in sorted(root.glob(backend + "-*.json")):
        if receipt_path.name.endswith("-progress.json"):
            continue
        receipt = json.loads(receipt_path.read_text())
        archive = receipt_path.with_suffix(".npz")
        with archive.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != receipt["array_sha256"]:
                raise ValueError("Prediction archive differs")
        if receipt["panel_sha256"] != panel_hash:
            raise ValueError("Panel bindings differ")
        if identity is not None and receipt["checkpoint"] != identity:
            raise ValueError("Mixed checkpoints")
        identity = receipt["checkpoint"]
        data = np.load(archive, allow_pickle=False)
        archives.append(data)
        for index in receipt["cases"]:
            if index in owners:
                raise ValueError("Duplicate case")
            owners[index] = data
    if set(owners) != expected:
        raise ValueError(
            f"{backend}: incomplete case coverage {len(owners)}/{len(expected)}"
        )
    return owners, identity


def variants(case, logits):
    return {
        name: nucleotide_probabilities(logits, collapse_case=name != "ar-lower-control")
        for name in VARIANTS[case["readout"]]
    }


def science(panel, owners, protocol):

    score_parts = defaultdict(dict)
    sv_probs = {}
    af_parts = {}
    for c in panel["cases"]:
        data = owners[c["index"]]
        logits = data[f"{c['index']}__logits"]
        for variant, p in variants(c, logits).items():
            key = (variant, c["task"], c["context"])
            truth = np.asarray(c["target_bases"])
            valid = truth >= 0
            chosen = np.zeros(len(truth))
            chosen[valid] = p[np.flatnonzero(valid), truth[valid]]
            if c["kind"] == "sv":
                arrays = sv_probs.setdefault(
                    variant,
                    {
                        a: np.zeros(
                            (len(panel["tasks"][c["task"]]["rows"]), 8192, 4),
                            dtype=np.float32,
                        )
                        for a in ["ref", "alt"]
                    },
                )
                arrays[c["allele"]][c["row"], c["original_positions"]] = p
            elif c["kind"] == "af":
                if c["readout"].startswith("ar-"):
                    full = data[f"{c['index']}__full_log_probs"]
                    if variant == "ar-lower-control":
                        canonical = full[:, 4:8]
                    else:
                        canonical = np.logaddexp(full[:, :4], full[:, 4:8])
                    original_n = 9 if c["readout"] == "ar-lower" else 8
                    logs = full[:, original_n].copy()
                    logs[valid] = canonical[np.flatnonzero(valid), truth[valid]]
                    af_parts[(variant, c["context"], c["row"], c["allele"])] = float(
                        logs.sum(dtype=np.float32)
                    )
                else:
                    row = panel["tasks"]["maize-af"]["rows"][c["row"]]
                    ref, alt = row["ref"].upper(), row["alt"].upper()
                    if c["context"] == "rc":
                        ref, alt = (
                            protocol.af.reverse_complement(ref),
                            protocol.af.reverse_complement(alt),
                        )
                    # Match the existing FIM AF FP32 eight-row head readout.
                    precise = data[f"{c['index']}__precise_logits"]
                    grouped = np.logaddexp(precise[:, :4], precise[:, 4:8])[0]
                    odds = float(
                        grouped["ACGT".index(alt)] - grouped["ACGT".index(ref)]
                    )
                    af_parts[(variant, c["context"], c["row"], "site")] = odds
            elif c["kind"] == "motif":
                score_parts[key][c["row"]] = float(np.all(p.argmax(-1) == truth))
            else:
                # Identical to the original ref-probability / mean-true-probability reducers.
                score_parts[key][c["row"]] = float(chosen.mean())
    for variant, p in sv_probs.items():
        import pandas as pd

        frame = pd.DataFrame(panel["tasks"]["sv_impact"]["rows"])
        scores = protocol.tasks.EVAL._sv_llr_boundary(
            frame, p["ref"], p["alt"], flanking=5
        )
        score_parts[(variant, "sv_impact", "left")] = dict(enumerate(scores))
    for variant in {k[0] for k in af_parts}:
        values = []
        for i in range(len(panel["tasks"]["maize-af"]["rows"])):
            strands = []
            for strand in ["fwd", "rc"]:
                if variant.startswith("ar-"):
                    strands.append(
                        af_parts[(variant, strand, i, "alt")]
                        - af_parts[(variant, strand, i, "ref")]
                    )
                else:
                    strands.append(af_parts[(variant, strand, i, "site")])
            values.append(float(np.mean(strands)))
        score_parts[(variant, "maize-af", "forward-RC mean")] = dict(enumerate(values))
    result = {}
    for key, value in score_parts.items():
        n = len(panel["tasks"][key[1]]["rows"])
        if set(value) != set(range(n)):
            raise ValueError("Incomplete scientific row coverage")
        result[key] = np.asarray([value[i] for i in range(n)])
    return result


def metric(kind, truth, value):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    if kind == "motif":
        return float(np.mean(value))
    if kind == "sv":
        return float(average_precision_score(truth, value))
    if kind == "af":
        return float(spearmanr(truth, value).statistic)
    return float(roc_auc_score(truth, value))


def interval(values):
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def compare(panel_path: Path, root: Path, scientific: Path, output: Path):
    sys.path.insert(0, str(scientific))
    from exp586_evals.protocol import original
    from scipy.stats import binomtest

    protocol = original()
    panel = json.loads(panel_path.read_text())
    panel_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    owners = {}
    checkpoint = None
    for backend in BACKENDS:
        owners[backend], identity = load_arrays(
            root, backend, panel_hash, {c["index"] for c in panel["cases"]}
        )
        if checkpoint is not None and identity != checkpoint:
            raise ValueError("Backend checkpoints differ")
        checkpoint = identity
    all_scores = {b: science(panel, owners[b], protocol) for b in BACKENDS}
    rows = []
    rng = np.random.default_rng(593)
    for backend in BACKENDS[1:]:
        for key, prediction in sorted(all_scores[backend].items()):
            variant, task, context = key
            entry = panel["tasks"][task]
            kind = entry["spec"]["kind"]
            native = all_scores["native"][key]
            truth = np.asarray(
                [
                    r.get("AF", 0) if kind == "af" else r.get("label", 0)
                    for r in entry["rows"]
                ]
            )
            score, reference = (
                metric(kind, truth, prediction),
                metric(kind, truth, native),
            )
            bs, delta = [], []
            for _ in range(500):
                take = rng.integers(len(truth), size=len(truth))
                if kind in ["core", "conservation", "sv"] and len(set(truth[take])) < 2:
                    continue
                a, b = (
                    metric(kind, truth[take], prediction[take]),
                    metric(kind, truth[take], native[take]),
                )
                if np.isfinite(a) and np.isfinite(b):
                    bs.append(b)
                    delta.append(a - b)
            row = {
                "backend": backend,
                "variant": variant,
                "task": task,
                "kind": kind,
                "context": context,
                "samples": len(truth),
                "native_score": reference,
                "score": score,
                "difference": score - reference,
                "native_sampling_95ci": interval(bs),
                "paired_difference_95ci": interval(delta),
                "prediction_mean_abs_difference": float(
                    np.abs(prediction - native).mean()
                ),
            }
            if kind == "motif":
                worse = int(np.sum((native == 1) & (prediction == 0)))
                better = int(np.sum((native == 0) & (prediction == 1)))
                row.update(
                    native_correct_backend_wrong=worse,
                    native_wrong_backend_correct=better,
                    paired_accuracy_pvalue=float(
                        binomtest(better, better + worse).pvalue
                    )
                    if worse + better
                    else 1.0,
                )
            rows.append(row)
    distributions = []
    for backend in BACKENDS[1:]:
        groups = defaultdict(list)
        for c in panel["cases"]:
            index = c["index"]
            actual = owners[backend][index][f"{index}__logits"]
            reference = owners["native"][index][f"{index}__logits"]
            for variant, p in variants(c, actual).items():
                q = variants(c, reference)[variant]
                truth = np.asarray(c["target_bases"])
                valid = truth >= 0
                if not valid.any():
                    continue
                pt = p[valid]
                qt = q[valid]
                t = truth[valid]
                loss = -np.log(np.maximum(pt[np.arange(len(t)), t], 1e-30)) + np.log(
                    np.maximum(qt[np.arange(len(t)), t], 1e-30)
                )
                flips = pt.argmax(-1) != qt.argmax(-1)
                ordered = np.sort(qt, axis=-1)
                groups[(variant, c["kind"])].append(
                    {
                        "loss": loss,
                        "prob": np.abs(pt - qt).reshape(-1),
                        "flips": flips,
                        "margin": (ordered[:, -1] - ordered[:, -2])[flips],
                        "case_mean_loss": float(loss.mean()),
                    }
                )
        for (variant, kind), parts in groups.items():
            losses = np.concatenate([p["loss"] for p in parts])
            errors = np.concatenate([p["prob"] for p in parts])
            flips = np.concatenate([p["flips"] for p in parts])
            margins = np.concatenate([p["margin"] for p in parts])
            case_means = np.asarray([p["case_mean_loss"] for p in parts])
            boot = np.mean(
                case_means[rng.integers(len(parts), size=(1000, len(parts)))], axis=1
            )
            distributions.append(
                {
                    "backend": backend,
                    "variant": variant,
                    "kind": kind,
                    "targets": len(losses),
                    "mean_nll_difference": float(losses.mean()),
                    "mean_case_nll_difference_95ci": interval(boot),
                    "mean_abs_target_nll_difference": float(np.abs(losses).mean()),
                    "mean_abs_probability_difference": float(errors.mean()),
                    "pooled_probability_p99": float(np.quantile(errors, 0.99)),
                    "argmax_disagreements": int(flips.sum()),
                    "argmax_disagreement_fraction": float(flips.mean()),
                    "median_native_top2_margin_on_disagreements": float(
                        np.median(margins)
                    )
                    if len(margins)
                    else None,
                }
            )
    # A direct ordinary-versus-fast comparison isolates the Fast-specific choices.
    controls = []
    for control in ["torch-fa3", "vllm-fa3"]:
        for key, fast in sorted(all_scores["fast-fa3"].items()):
            variant, task, context = key
            entry = panel["tasks"][task]
            kind = entry["spec"]["kind"]
            truth = np.asarray(
                [
                    r.get("AF", 0) if kind == "af" else r.get("label", 0)
                    for r in entry["rows"]
                ]
            )
            controls.append(
                {
                    "control": control,
                    "variant": variant,
                    "task": task,
                    "context": context,
                    "fast_minus_control_score": metric(kind, truth, fast)
                    - metric(kind, truth, all_scores[control][key]),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "panel_sha256": panel_hash,
                "status": "complete-requires-agent-interpretation",
                "notes": [
                    "These are fixed validation subsets, not final benchmark scores.",
                    "Classification subsets are label-balanced for runtime validation; SV AUPRC is not a population estimate.",
                    "Confidence intervals are finite-subset diagnostics; related genomic rows may be correlated.",
                    "NLL/bootstrap values do not automatically accept/reject a backend.",
                    "No thresholds or task definitions were adjusted to select a winning backend.",
                ],
                "task_scores": rows,
                "target_differences": distributions,
                "fast_controls": controls,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "report": str(output),
                "task_score_comparisons": len(rows),
                "target_distributions": len(distributions),
            }
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["panel", "predictions", "scientific-code", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    compare(a.panel, a.predictions, a.scientific_code, a.output)


if __name__ == "__main__":
    main()
