"""Synthetic end-to-end fixtures; these produce no benchmark results."""

import hashlib
import json
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

from exp586_evals.scoring import (
    BASES,
    SequenceScorer,
    paired_probe_feature,
    reverse_complement,
)


def tokenizer_digest(tokenizer: Tokenizer) -> str:
    return hashlib.sha256(tokenizer.to_str().encode()).hexdigest()


def synthetic_requests(tokenizer: Tokenizer, *, bases: int = 8192) -> dict[str, Any]:
    """Collect every sequence needed by scoring and 32 frozen-feature fixtures.

    Requests remain separate windows. CPU scoring can consume the saved GPU
    outputs without requiring private-repository credentials on GPU workers.
    """
    if bases < 64 or bases > 8192:
        raise ValueError("Synthetic windows require 64..8192 bases")
    rng = np.random.default_rng(582)
    sequences = ["".join(rng.choice(list(BASES), size=bases)) for _ in range(32)]
    position = bases // 2
    requests: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    def add(sequence: str, variant_position: int | None = None) -> int:
        variant_position = (
            min(position, len(sequence) - 1)
            if variant_position is None
            else variant_position
        )
        if (
            sequence in seen
            and requests[seen[sequence]]["variant_position"] != variant_position
        ):
            raise ValueError(
                "One sequence was requested at conflicting variant positions"
            )
        if sequence not in seen:
            encoding = tokenizer.encode(sequence, add_special_tokens=False)
            if not 2 <= len(encoding.ids) <= 8192:
                raise ValueError("Fixture is outside the supported token range")
            seen[sequence] = len(requests)
            requests.append(
                {
                    "sequence": sequence,
                    "ids": encoding.ids,
                    "variant_position": variant_position,
                }
            )
        return seen[sequence]

    probe_indices = [add(sequence) for sequence in sequences]
    llr_cases = []
    for ref in sequences[:4]:
        alternative = BASES[(BASES.index(ref[position]) + 1) % 4]
        alt = ref[:position] + alternative + ref[position + 1 :]
        llr_cases.append({"reference": ref, "alternate": alt, "position": position})
        for sequence in (ref, alt):
            add(sequence)
        for sequence in (reverse_complement(ref), reverse_complement(alt)):
            add(sequence, bases - 1 - position)
    base_cases = []
    for sequence in sequences[:2]:
        base_cases.append({"sequence": sequence, "position": position})
        for base in BASES:
            add(sequence[:position] + base)
    return {
        "schema": 1,
        "synthetic": True,
        "seed": 582,
        "bases": bases,
        "tokenizer_sha256": tokenizer_digest(tokenizer),
        "requests": requests,
        "probe_indices": probe_indices,
        "llr_cases": llr_cases,
        "base_cases": base_cases,
    }


def request_bytes(requests: dict[str, Any]) -> bytes:
    return json.dumps(requests, sort_keys=True, separators=(",", ":")).encode()


def validate_outputs(
    tokenizer: Tokenizer,
    requests: dict[str, Any],
    token_logs: list[np.ndarray],
    mean_features: np.ndarray,
    variant_features: np.ndarray,
) -> dict[str, float | int]:
    """Exercise variant scores and a frozen-feature probe on synthetic labels.

    The fixed-alpha toy probe verifies data flow and train-only scaling, not the
    real benchmark's genomic-block cross-validation or scientific performance.
    """
    if not requests.get("synthetic") or requests.get("schema") != 1:
        raise ValueError("Only synthetic schema-1 validation fixtures are allowed")
    if requests["tokenizer_sha256"] != tokenizer_digest(tokenizer):
        raise ValueError("Tokenizer differs from the request manifest")
    items = requests["requests"]
    if len(token_logs) != len(items):
        raise ValueError("Inference request/output counts differ")
    cache = {}
    for item, logs in zip(items, token_logs, strict=True):
        ids = tokenizer.encode(item["sequence"], add_special_tokens=False).ids
        if ids != item["ids"]:
            raise ValueError("Saved token IDs differ from canonical tokenization")
        logs = np.asarray(logs, dtype=np.float64)
        if logs.shape != (len(ids) - 1,) or not np.isfinite(logs).all():
            raise ValueError("Missing or nonfinite real-token log probabilities")
        cache[tuple(ids)] = logs
    model = SequenceScorer(tokenizer, lambda batch: [cache[tuple(x)] for x in batch])
    metrics: dict[str, float | int] = {"requests": len(items)}
    for i, case in enumerate(requests["base_cases"]):
        sequence, position = case["sequence"], case["position"]
        probabilities = model.base_probabilities(sequence, position)
        np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-12)
        # Replacing the right flank must leave the actual GPU requests unchanged.
        np.testing.assert_array_equal(
            probabilities,
            model.base_probabilities(sequence[:position] + "t" * 17, position),
        )
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite base probabilities")
        metrics[f"base_case_{i}/max_probability"] = float(probabilities.max())
    for i, case in enumerate(requests["llr_cases"]):
        ref, alt = case["reference"], case["alternate"]
        score = model.strand_average_llr(ref, alt)
        np.testing.assert_allclose(
            score, -model.strand_average_llr(alt, ref), atol=1e-12
        )
        np.testing.assert_allclose(
            score,
            model.strand_average_llr(reverse_complement(ref), reverse_complement(alt)),
            atol=1e-12,
        )
        metrics[f"llr_case_{i}"] = score
    indices = np.asarray(requests["probe_indices"])
    if indices.shape != (32,) or len(set(indices)) != 32:
        raise ValueError("Expected 32 independent synthetic probe windows")
    target = np.asarray(
        [
            (items[i]["sequence"].count("g") + items[i]["sequence"].count("c"))
            / len(items[i]["sequence"])
            for i in indices
        ]
    )
    for name, values in (("whole", mean_features), ("variant", variant_features)):
        values = np.asarray(values, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[0] != len(items)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Missing or nonfinite frozen features")
        features = values[indices]
        # Six blocks of four for fitting; two disjoint blocks for the toy check.
        probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        probe.fit(features[:24], target[:24])
        np.testing.assert_allclose(probe[0].mean_, features[:24].mean(axis=0))
        predictions = probe.predict(features[24:])
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite frozen-probe predictions")
        metrics[f"synthetic_probe/{name}/mse"] = float(
            np.mean((predictions - target[24:]) ** 2)
        )
        lookup = {item["sequence"]: i for i, item in enumerate(items)}
        paired = []
        for case in requests["llr_cases"]:
            ref, alt, position = case["reference"], case["alternate"], case["position"]
            sequences = (ref, alt, reverse_complement(ref), reverse_complement(alt))
            rows = [lookup[sequence] for sequence in sequences]
            expected_positions = (
                position,
                position,
                len(ref) - 1 - position,
                len(alt) - 1 - position,
            )
            if any(
                items[row]["variant_position"] != expected
                for row, expected in zip(rows, expected_positions, strict=True)
            ):
                raise ValueError(
                    "Probe variant position does not follow the strand transform"
                )
            vectors = [values[row] for row in rows]
            feature = paired_probe_feature(*vectors)
            np.testing.assert_array_equal(
                feature,
                paired_probe_feature(vectors[2], vectors[3], vectors[0], vectors[1]),
            )
            paired.append(feature)
        paired = np.asarray(paired)
        # Four synthetic variant pairs exercise the historical feature contract;
        # arbitrary labels and a 3/1 split have no scientific interpretation.
        if paired.shape != (4, 2 * values.shape[1]):
            raise ValueError("Unexpected paired probe dimensions")
        paired_probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        paired_probe.fit(paired[:3], np.asarray([0.1, 0.2, 0.4]))
        np.testing.assert_allclose(
            paired_probe[0].mean_, paired[:3].mean(axis=0, dtype=np.float64)
        )
        prediction = paired_probe.predict(paired[3:])
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite paired-probe prediction")
        metrics[f"synthetic_probe/{name}/paired_prediction"] = float(prediction[0])
    return metrics
