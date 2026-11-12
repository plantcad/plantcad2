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


def check(model: Path, fixtures: Path, output: Path):
    import torch

    from exp593_anneal_evals.scoring import LowercaseScorer
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
                if len(adapter_cases[len(targets)]) < 16:
                    adapter_cases[len(targets)].append((case, p[where].copy()))
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
    adapter = LowercaseScorer(scorer)
    adapter_errors = []
    for length, selected in adapter_cases.items():
        prompts = [c["ids"][: c["target_positions"][0]] for c, _ in selected]
        answers = [c["answer"] for c, _ in selected]
        actual = adapter.teacher(prompts, answers)
        expected = np.asarray([p for _, p in selected])
        adapter_errors.append(float(np.abs(actual - expected).max()))
    actual = np.exp(
        adapter.next_log_probs([c["ids"][:-1] for c, _ in prefix_adapter_cases])
    )
    expected = np.asarray([p for _, p in prefix_adapter_cases])
    adapter_errors.append(float(np.abs(actual - expected).max()))
    if max(adapter_errors) > 1e-6:
        raise ValueError("The evaluation adapter differs from its canonical readout")
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
        "canonical_adapter_max_abs": max(adapter_errors),
        "teacher_forced_task_diagnostics": tf_stats,
        "exact_candidate_diagnostics": exact,
        "paired_exact_candidate_diagnostics": paired,
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
