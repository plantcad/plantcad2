"""Pooled native parity and real-task diagnostics for the lower-only integration.

Retain the published32-window aggregate AR thresholds. FIM answer and exact
candidate differences are reported by task for agent review, without inventing
per-example NLL thresholds for two/three-base answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from exp593_anneal_evals.exhaustive import combine, expand, predictions
from exp593_evals.readouts import nucleotide_probabilities


def binary_auroc(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores)
    positives = labels.sum()
    if positives == 0 or positives == len(labels):
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    return float(
        (ranks[inverse][labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * (len(labels) - positives))
    )


def binary_average_precision(labels, scores):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores)
    if not labels.sum():
        return None
    order = np.argsort(-scores, kind="stable")
    y, s = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    positives = np.cumsum(y)[ends]
    precision = positives / (ends + 1)
    recall = positives / y.sum()
    return float(np.sum(precision * np.diff(np.r_[0, recall])))


def sv_endpoint_diagnostics(rows, values):
    from exp593_anneal_evals.scoring import sv_reduce

    result = {}
    for presentation, models in values.items():
        native, actual = (
            sv_reduce(rows, models[key]) for key in ("native", "deployed")
        )
        for orientation in native:
            a = binary_average_precision(
                [row["label"] for row in rows], native[orientation]
            )
            b = binary_average_precision(
                [row["label"] for row in rows], actual[orientation]
            )
            result[presentation + "/" + orientation.removesuffix("__scores")] = {
                "examples": len(rows),
                "metric": "average_precision",
                "native": a,
                "deployed": b,
                "delta": None if a is None else b - a,
                "score_max_abs": float(
                    np.abs(native[orientation] - actual[orientation]).max()
                ),
            }
    return result


def endpoint_diagnostics(cases):
    """Actual TF task metrics on paired native/deployed predictions, including fusion."""
    grouped = defaultdict(dict)
    for case, native, actual, indexes in cases:
        grouped[(case["task"], case["presentation"], case["example"])][
            case["strand"]
        ] = (case, native, actual, indexes)
    values = defaultdict(list)
    for (task, presentation, _), views in grouped.items():
        if set(views) != {"fwd", "rc"}:
            raise ValueError("A paired task endpoint lacks a strand")
        f, q, p, indexes = views["fwd"]
        _, qr, pr, _ = views["rc"]
        aligned_q, aligned_p = qr[::-1, [3, 2, 1, 0]], pr[::-1, [3, 2, 1, 0]]
        # Advanced indexing above preserves the target-position axis.
        fused_q, fused_p = np.sqrt(q * aligned_q), np.sqrt(p * aligned_p)
        fused_q /= fused_q.sum(-1, keepdims=True)
        fused_p /= fused_p.sum(-1, keepdims=True)
        for orientation, a, b in [
            ("forward", q, p),
            ("reverse_complement", aligned_q, aligned_p),
            ("forward_rc_product", fused_q, fused_p),
        ]:
            if f["task_kind"] == "motif":
                native_score = float(np.array_equal(a.argmax(-1), indexes))
                torch_score = float(np.array_equal(b.argmax(-1), indexes))
            else:
                valid = indexes >= 0
                columns = indexes.clip(min=0)
                native_score = float(
                    (a[np.arange(len(indexes)), columns] * valid).mean()
                )
                torch_score = float(
                    (b[np.arange(len(indexes)), columns] * valid).mean()
                )
            values[(task, presentation, orientation, f["task_kind"])].append(
                (f.get("label"), native_score, torch_score)
            )
    result = {}
    for key, rows in values.items():
        task, presentation, orientation, kind = key
        label, native, actual = zip(*rows, strict=True)
        a = float(np.mean(native)) if kind == "motif" else binary_auroc(label, native)
        b = float(np.mean(actual)) if kind == "motif" else binary_auroc(label, actual)
        result["/".join(key[:3])] = {
            "task_kind": kind,
            "examples": len(rows),
            "metric": "accuracy" if kind == "motif" else "auroc",
            "native": a,
            "deployed": b,
            "delta": None if a is None else b - a,
        }
    return result


def adapter_diagnostics(scorer, adapter_cases, prefix_cases):
    """Compare integration using identical physical batches, not regrouped rows."""
    import torch

    from exp593_anneal_evals.scoring import LowercaseScorer

    adapter = LowercaseScorer(scorer)
    integration, batching = [], []
    for selected in adapter_cases.values():
        prompts = [c["ids"][: c["target_positions"][0]] for c, _ in selected]
        answers = [c["answer"] for c, _ in selected]
        requests = [p + adapter.ids(a) for p, a in zip(prompts, answers, strict=True)]
        with torch.inference_mode():
            logits, _, _ = scorer.forward(requests)
            queries = torch.tensor(
                [
                    [len(p) - 1 + j for j in range(len(a))]
                    for p, a in zip(prompts, answers, strict=True)
                ],
                device=scorer.device,
            )
            rows = torch.arange(len(requests), device=scorer.device)[:, None]
            canonical = (
                torch.softmax(logits[rows, queries, 4:8].float(), -1).cpu().numpy()
            )
        actual = adapter.teacher(prompts, answers)
        integration.append(float(np.abs(actual - canonical).max()))
        batching.append(
            float(np.abs(canonical - np.asarray([p for _, p in selected])).max())
        )
    requests = [c["ids"][:-1] for c, _ in prefix_cases]
    with torch.inference_mode():
        logits, _, lengths = scorer.forward(requests)
        rows = torch.arange(len(requests), device=scorer.device)
        canonical = (
            torch.softmax(logits[rows, lengths - 1, 4:8].float(), -1).cpu().numpy()
        )
    actual = np.exp(adapter.next_log_probs(requests))
    integration.append(float(np.abs(actual - canonical).max()))
    batching.append(
        float(np.abs(canonical - np.asarray([p for _, p in prefix_cases])).max())
    )
    return {
        "canonical_adapter_max_abs": max(integration),
        "regrouping_max_abs": max(batching),
    }


def check(model: Path, fixtures: Path, output: Path):
    import torch

    from exp593_anneal_evals.scoring import exact_task_scores
    from exp593_evals.torch_scorer import FastScorer

    receipt = json.loads((model / "native-receipt.json").read_text())
    if (
        hashlib.sha256(fixtures.read_bytes()).hexdigest() != receipt["fixtures_sha256"]
        or hashlib.sha256((model / "native-reference.npz").read_bytes()).hexdigest()
        != receipt["reference_sha256"]
    ):
        raise ValueError("Native fixture/reference identity differs")
    fixture = json.loads(fixtures.read_text())
    cases = fixture["cases"]
    scorer = FastScorer(model)
    native = np.load(model / "native-reference.npz", allow_pickle=False)
    errors = defaultdict(list)
    ar_native = []
    ar_actual = []
    ar_target = []
    tf = defaultdict(list)
    prefix = defaultdict(dict)
    hidden_errors = []
    adapter_cases = defaultdict(list)
    prefix_adapter_cases = []
    endpoint_cases = []
    sv_values = {}
    for start in range(0, len(cases), scorer.batch_size):
        group = cases[start : start + scorer.batch_size]
        with torch.inference_mode():
            logits, hidden, lengths = scorer.forward([c["ids"] for c in group])
        for local, case in enumerate(group):
            i = start + local
            positions = native[f"{i}__logit_positions"]
            actual = (
                logits[local, torch.tensor(positions, device=scorer.device)]
                .float()
                .cpu()
                .numpy()
            )
            expected = native[f"{i}__logits"]
            p = nucleotide_probabilities(actual, collapse_case=False)
            q = nucleotide_probabilities(expected, collapse_case=False)
            errors[case["kind"]].append(np.abs(p - q).reshape(-1))
            target_positions = np.asarray(case["target_positions"]) - 1
            where = np.searchsorted(positions, target_positions)
            if not np.array_equal(positions[where], target_positions):
                raise ValueError("Native query positions shifted")
            targets = np.asarray(case["ids"])[np.asarray(case["target_positions"])]
            if case.get("sv_query"):
                request = case["sv_query"]
                presentation = request["presentation"]
                if presentation not in sv_values:
                    sv_values[presentation] = {
                        name: {} for name in ("native", "deployed")
                    }
                strand = (
                    "forward" if request["strand"] == "fwd" else "reverse_complement"
                )
                for name, probabilities in [
                    ("native", q[where]),
                    ("deployed", p[where]),
                ]:
                    if strand not in sv_values[presentation][name]:
                        sv_values[presentation][name][strand] = np.zeros(
                            (len(fixture["sv_rows"]), 2, 10, 4)
                        )
                    for boundary, probability in zip(
                        request["boundaries"], probabilities, strict=True
                    ):
                        sv_values[presentation][name][strand][
                            request["row"], request["allele"], boundary
                        ] = (
                            probability
                            if strand == "forward"
                            else probability[[3, 2, 1, 0]]
                        )
            indexes = np.full(len(targets), -1)
            for b, token in enumerate(scorer.dna_ids[4:8]):
                indexes[targets == token] = b
            if case["kind"] == "ar-window":
                valid = indexes >= 0
                ar_actual.append(p[where[valid]])
                ar_native.append(q[where[valid]])
                ar_target.append(indexes[valid])
            elif case["kind"] == "tf":
                valid = indexes >= 0
                actual_match = bool(
                    valid.all() and np.array_equal(p[where].argmax(-1), indexes)
                )
                native_match = bool(
                    valid.all() and np.array_equal(q[where].argmax(-1), indexes)
                )
                tf[(case["task"], case["presentation"], case["strand"])].append(
                    (native_match, actual_match)
                )
                if (
                    case["presentation"] == "fim"
                    and len(adapter_cases[len(targets)]) < 16
                ):
                    adapter_cases[len(targets)].append((case, p[where].copy()))
                if case.get("task_kind"):
                    endpoint_cases.append(
                        (case, q[where].copy(), p[where].copy(), indexes.copy())
                    )
            elif case["kind"] == "candidate-prefix":
                key = (
                    case["task"],
                    case["presentation"],
                    case["strand"],
                    case["example"],
                    case["gap_length"],
                )
                prefix[key][case["prefix"]] = (
                    np.log(q[0].clip(1e-300)),
                    np.log(p[0].clip(1e-300)),
                )
                if len(prefix_adapter_cases) < 32:
                    prefix_adapter_cases.append((case, p[0].copy()))
            length = int(lengths[local])
            a = hidden[local, length - 1].float().cpu().numpy()
            b = native[f"{i}__last_hidden"]
            hidden_errors.append(
                float(
                    np.sqrt(np.mean((a - b) ** 2)) / max(np.sqrt(np.mean(b * b)), 1e-12)
                )
            )
        print(
            json.dumps(
                {
                    "phase": "lower-only-native-parity",
                    "completed": start + len(group),
                    "total": len(cases),
                }
            ),
            flush=True,
        )
    # Exercise the actual evaluation adapter against the direct canonical
    # readout too. This catches query shifts or accidental normalization changes
    # separately from the native-vs-Torch numerical differences above.
    adapter_report = adapter_diagnostics(scorer, adapter_cases, prefix_adapter_cases)
    adapter_report["canonical_adapter_passed"] = (
        adapter_report["canonical_adapter_max_abs"] <= 1e-6
    )
    stats = {}
    for kind, parts in errors.items():
        e = np.concatenate(parts)
        stats[kind] = {
            "values": len(e),
            "mean_abs": float(e.mean()),
            "p99_abs": float(np.quantile(e, 0.99)),
            "max_abs": float(e.max()),
        }
    p, q, t = (
        np.concatenate(ar_actual),
        np.concatenate(ar_native),
        np.concatenate(ar_target),
    )
    rows = np.arange(len(t))
    nll_delta = float(
        np.mean(-np.log(p[rows, t].clip(1e-30)))
        - np.mean(-np.log(q[rows, t].clip(1e-30)))
    )
    accuracy_delta = float(np.mean(p.argmax(-1) == t) - np.mean(q.argmax(-1) == t))
    published = {
        "windows": sum(c["kind"] == "ar-window" for c in cases),
        "targets": len(t),
        "nll_delta": nll_delta,
        "accuracy_delta": accuracy_delta,
        "mean_abs": stats["ar-window"]["mean_abs"],
        "p99_abs": stats["ar-window"]["p99_abs"],
    }
    published["passed"] = bool(
        published["windows"] == 32
        and published["mean_abs"] <= 0.003
        and published["p99_abs"] <= 0.03
        and abs(nll_delta) <= 0.001
        and abs(accuracy_delta) <= 0.001
    )
    tf_stats = {}
    for key, values in tf.items():
        a = np.asarray(values, dtype=float)
        delta = a[:, 1] - a[:, 0]
        tf_stats["/".join(key)] = {
            "examples": len(a),
            "native_exact_match": float(a[:, 0].mean()),
            "torch_exact_match": float(a[:, 1].mean()),
            "paired_delta": float(delta.mean()),
            "prediction_correctness_disagreements": int(np.count_nonzero(delta)),
        }
    exact = {}
    joint = defaultdict(dict)
    for key, table in prefix.items():
        task, presentation, strand, example, length = key

        def reader(which, table=table):
            return lambda requests: np.array(
                [
                    table["".join("acgt"[token] for token in request[1:])][which]
                    for request in requests
                ]
            )

        native_joint = expand([[99]], length, (0, 1, 2, 3), reader(0))
        torch_joint = expand([[99]], length, (0, 1, 2, 3), reader(1))
        joint[(task, presentation, example, length)][strand] = (
            native_joint,
            torch_joint,
        )
        exact["/".join(map(str, key))] = {
            "native_map": predictions(native_joint, length)[0],
            "torch_map": predictions(torch_joint, length)[0],
            "joint_probability_max_abs": float(
                np.abs(np.exp(native_joint) - np.exp(torch_joint)).max()
            ),
        }
    paired = []
    exact_endpoint_values = defaultdict(list)
    metadata = {
        (c["task"], c["presentation"], c["example"]): c
        for c in cases
        if c["kind"] == "tf" and c["strand"] == "fwd" and c.get("task_kind")
    }
    for key, views in joint.items():
        if set(views) != {"fwd", "rc"}:
            raise ValueError("Exact native fixture lacks a strand")
        task, presentation, example, length = key
        a = combine(views["fwd"][0], views["rc"][0], length)
        b = combine(views["fwd"][1], views["rc"][1], length)
        paired.append(
            {
                "task": task,
                "presentation": presentation,
                "example": example,
                "native_map": predictions(a, length)[0],
                "torch_map": predictions(b, length)[0],
                "joint_probability_max_abs": float(np.abs(np.exp(a) - np.exp(b)).max()),
            }
        )
        case = metadata.get((task, presentation, example))
        if case:
            native_values, actual_values = (
                exact_task_scores(
                    {
                        "forward__candidate_log_probs": views["fwd"][i],
                        "rc__candidate_log_probs": views["rc"][i],
                    },
                    [case["answer"]],
                )
                for i in (0, 1)
            )
            for orientation in native_values:
                for mode in (
                    ["exact_match"]
                    if case["task_kind"] == "motif"
                    else ["joint", "mean_marginal"]
                ):
                    exact_endpoint_values[
                        (task, presentation, orientation, mode, case["task_kind"])
                    ].append(
                        (
                            case.get("label"),
                            float(native_values[orientation][mode][0]),
                            float(actual_values[orientation][mode][0]),
                        )
                    )
    exact_endpoints = {}
    for key, rows in exact_endpoint_values.items():
        label, native, actual = zip(*rows, strict=True)
        a = (
            float(np.mean(native))
            if key[-1] == "motif"
            else binary_auroc(label, native)
        )
        b = (
            float(np.mean(actual))
            if key[-1] == "motif"
            else binary_auroc(label, actual)
        )
        exact_endpoints["/".join(key[:-1])] = {
            "examples": len(rows),
            "task_kind": key[-1],
            "metric": "accuracy" if key[-1] == "motif" else "auroc",
            "native": a,
            "deployed": b,
            "delta": None if a is None else b - a,
        }
    result = {
        "checkpoint": receipt["checkpoint"],
        "fixtures_sha256": receipt["fixtures_sha256"],
        "native_reference_sha256": receipt["reference_sha256"],
        "configuration": "fast-fa3",
        "fullgraph": True,
        "output_contract": "lowercase a/c/g/t only; no case folding",
        "published_ar_gate": published,
        "probability_errors": stats,
        "hidden_relative_rmse_max": max(hidden_errors),
        **adapter_report,
        "teacher_forced_task_diagnostics": tf_stats,
        "paired_task_endpoints": endpoint_diagnostics(endpoint_cases)
        if endpoint_cases
        else {},
        "sv_task_endpoints": sv_endpoint_diagnostics(
            fixture.get("sv_rows", []), sv_values
        )
        if sv_values
        else {},
        "exact_candidate_diagnostics": exact,
        "paired_exact_candidate_diagnostics": paired,
        "exact_task_endpoints": exact_endpoints,
        "native_base_gate_passed": published["passed"],
        "requires_agent_practical_review": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "published_ar_gate": published,
                "probability_errors": stats,
                "joint_map_disagreements": sum(
                    r["native_map"] != r["torch_map"] for r in paired
                ),
                "agent_review_required": True,
            }
        ),
        flush=True,
    )
    if not adapter_report["canonical_adapter_passed"]:
        raise ValueError(
            f"The evaluation adapter differs from its canonical readout: {adapter_report}"
        )
    if not published["passed"]:
        raise ValueError(
            "Published aggregate lower-only native gate failed; do not weaken it"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["model-dir", "fixtures", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    check(a.model_dir, a.fixtures, a.output)


if __name__ == "__main__":
    main()
