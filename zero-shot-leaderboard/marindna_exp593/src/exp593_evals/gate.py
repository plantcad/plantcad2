"""Finite fast-fa3 parity checks against the checkpoint's native-JAX reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exp593_evals.readouts import nucleotide_probabilities


def check(model: Path, fixtures: Path, output: Path) -> None:
    from exp593_evals.legacy_readout import (
        _configure_environment,
        streamed_full_log_probs,
    )
    from exp593_evals.torch_scorer import FastScorer

    _configure_environment("fast-fa3")
    import torch

    native_receipt = json.loads((model / "native-receipt.json").read_text())
    if (
        hashlib.sha256(fixtures.read_bytes()).hexdigest()
        != native_receipt["fixtures_sha256"]
        or hashlib.sha256((model / "native-reference.npz").read_bytes()).hexdigest()
        != native_receipt["reference_sha256"]
    ):
        raise ValueError("Native reference or fixtures differ from their receipt")
    scorer = FastScorer(model)
    references = np.load(model / "native-reference.npz", allow_pickle=False)
    cases = json.loads(fixtures.read_text())["cases"]
    rows = []
    for index, case in enumerate(cases):
        logits, hidden, lengths = scorer.forward([case["ids"]])
        length = int(lengths[0])
        actual = logits[0, :length].float().cpu().numpy().copy()
        expected = references[f"{index}__logits"]
        probability = nucleotide_probabilities(actual, collapse_case=True)
        reference = nucleotide_probabilities(expected, collapse_case=True)
        error = np.abs(probability - reference)
        tokens = np.asarray(case["ids"])
        targets = np.full(len(tokens) - 1, -1, dtype=np.int32)
        for base, token in enumerate(scorer.dna_ids[:8]):
            targets[tokens[1:] == token] = base % 4
        requested = np.zeros(len(tokens), dtype=bool)
        requested[case["target_positions"]] = True
        valid = (targets >= 0) & requested[1:]
        valid_positions = np.flatnonzero(valid)
        if not valid.any():
            raise ValueError("Reference case has no canonical DNA targets")
        p = probability[valid_positions, targets[valid]]
        q = reference[valid_positions, targets[valid]]
        nll_delta = float(
            np.mean(-np.log(np.maximum(p, 1e-30)))
            - np.mean(-np.log(np.maximum(q, 1e-30)))
        )
        accuracy_delta = float(
            np.mean(probability[valid_positions].argmax(-1) == targets[valid])
            - np.mean(reference[valid_positions].argmax(-1) == targets[valid])
        )
        last = hidden[0, length - 1].float().cpu().numpy().copy()
        expected_last = references[f"{index}__last_hidden"]
        relative_rmse = float(
            np.sqrt(np.mean((last - expected_last) ** 2))
            / max(float(np.sqrt(np.mean(expected_last**2))), 1e-12)
        )
        queries = references[f"{index}__full_queries"]
        selected_hidden = hidden[0, queries].repeat_interleave(8, dim=0)
        labels = torch.tensor(scorer.dna_ids[:8], device=scorer.device).repeat(
            len(queries)
        )
        with torch.inference_mode():
            full = (
                streamed_full_log_probs(
                    hidden=selected_hidden, targets=labels, head=scorer.full_head
                )
                .reshape(len(queries), 8)
                .cpu()
                .numpy()
            )
        expected_full = references[f"{index}__full_log_probs"]
        full_probability_error = float(
            np.max(np.abs(np.exp(full) - np.exp(expected_full)))
        )
        actual_mass = float(np.exp(full).sum(-1).mean())
        accepted = bool(
            error.mean() <= 0.003
            and np.quantile(error, 0.99) <= 0.03
            and abs(nll_delta) <= 0.001
            and abs(accuracy_delta) <= 0.001
            and relative_rmse <= 0.15
            and full_probability_error <= 0.03
        )
        row = {
            "name": case["name"],
            "mean_probability_abs": float(error.mean()),
            "p99_probability_abs": float(np.quantile(error, 0.99)),
            "nll_delta": nll_delta,
            "accuracy_delta": accuracy_delta,
            "last_hidden_relative_rmse": relative_rmse,
            "full_vocab_selected_probability_max_abs": full_probability_error,
            "full_vocab_canonical_mass": actual_mass,
            "accepted": accepted,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "checkpoint": native_receipt["checkpoint"],
        "fixtures_sha256": native_receipt["fixtures_sha256"],
        "native_reference_sha256": native_receipt["reference_sha256"],
        "accepted": all(r["accepted"] for r in rows),
        "fullgraph": True,
        "configuration": "fast-fa3",
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["accepted"]:
        raise RuntimeError(
            "Native-JAX parity gate failed; do not loosen its thresholds"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.model_dir, args.fixtures, args.output)


if __name__ == "__main__":
    main()
