"""Exercise new saved-probability outputs through the real historical reducers."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
sklearn_metrics = pytest.importorskip("sklearn.metrics")
average_precision_score = sklearn_metrics.average_precision_score
roc_auc_score = sklearn_metrics.roc_auc_score

from exp593_anneal_evals.metrics import complete_tf_arrays, exact_motif_accuracy

ROOT = Path(__file__).resolve().parents[2] / "evo2_20b"


def functions(path, names, scope):
    # Load unmodified pure functions without importing a GPU model/runtime.
    tree = ast.parse(path.read_text())
    selected = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in selected} == set(names)
    prefix = ast.parse("from __future__ import annotations").body
    exec(  # noqa: S102 - execute only named pure functions from the tracked reference.
        compile(ast.Module(body=prefix + selected, type_ignores=[]), str(path), "exec"),
        scope,
    )
    return SimpleNamespace(**{name: scope[name] for name in names})


@pytest.fixture
def legacy():
    scope = {
        "np": np,
        "pd": pd,
        "NUCLEOTIDES": tuple("ACGT"),
        "COMPLEMENT": dict(zip("ACGTN", "TGCAN")),
    }
    evaluation = functions(
        ROOT / "zero-shot-eval-causal.py",
        [
            "_reverse_complement",
            "_complement_same_order",
            "_transform_sequences_and_positions",
            "_compute_true_tokens_from_seq",
            "_metric_token_accuracy",
            "_metric_motif_accuracy",
            "_refprob_scores",
            "_avg_trueprob_scores",
        ],
        scope,
    )
    evaluation.NUCLEOTIDES = tuple("ACGT")
    return evaluation


@pytest.mark.parametrize(
    "kind,positions", [("conservation", [2]), ("core", [2, 3]), ("motif", [1, 2, 3])]
)
def test_fim_probabilities_reduce_like_historical_outputs(legacy, kind, positions):
    frame = pd.DataFrame(
        {"sequence": ["aacgtn", "ttnact", "nacgta"], "label": [0, 1, 0]}
    )
    spec = {
        "kind": kind,
        "positions": positions,
        "contexts": ["left", "right_reverse_complement"],
    }
    rng = np.random.default_rng(593)
    arrays = {}
    for context in spec["contexts"]:
        p = rng.random((len(frame), len(positions), 4))
        arrays[context + "__probs"] = p / p.sum(-1, keepdims=True)
    calls = iter(spec["contexts"])

    def model_probabilities(*args):
        return arrays[next(calls) + "__probs"], {}

    scoring = functions(
        ROOT / "pilot/sensitivity/run_condition.py",
        ["_score_standard_task"],
        {
            "np": np,
            "EVAL": legacy,
            "_reset_memory": lambda model: None,
            "_score_positions": model_probabilities,
            "_peak_memory": lambda *args: None,
        },
    )
    _, reference = scoring._score_standard_task(
        spec,
        frame,
        None,
        None,
        SimpleNamespace(batch_size=64, softmax_dtype="fp32", use_cache=False),
        compute_metrics=False,
    )
    enriched = complete_tf_arrays(
        SimpleNamespace(tasks=SimpleNamespace(EVAL=legacy)), spec, frame, arrays
    )
    for key, value in reference.items():
        np.testing.assert_array_equal(enriched[key], value)
    assert not any(k.endswith("__scores") for k in arrays)
    aggregate = functions(
        ROOT / "pilot/coreweave/aggregate.py",
        ["recompute_task"],
        {
            "np": np,
            "roc_auc_score": roc_auc_score,
            "average_precision_score": average_precision_score,
            "scoring": SimpleNamespace(EVAL=legacy),
        },
    )
    assert aggregate.recompute_task(spec, frame, enriched) == aggregate.recompute_task(
        spec, frame, reference
    )
    unchanged = complete_tf_arrays(SimpleNamespace(tasks=None), spec, frame, reference)
    assert all(unchanged[k] is v for k, v in reference.items())


def test_exhaustive_motif_mask_matches_historical_accuracy(legacy):
    answers = ["acg", "ttt", "nnn"]
    predictions = ["acg", "aaa", "aaa"]
    probabilities = np.eye(4)[["acgt".index(b) for s in predictions for b in s]]
    expected = legacy._metric_motif_accuracy(
        probabilities, np.array(list("".join(answers).upper())), 3
    )
    assert exact_motif_accuracy([True, False, False], answers) == expected == 0.5
    assert exact_motif_accuracy([False], ["nnn"]) == 0.0


def test_completed_phase_requires_pinned_receipt_and_every_verified_job(tmp_path):
    import hashlib
    import io
    import json

    from exp593_anneal_evals.analysis_bootstrap import completed_phase
    from exp593_evals.analysis_resume import job_name

    owner = "s3://bucket/eval/model"
    phase = "probes-p1"
    job = {
        "kind": "probe",
        "profile": "frozen-probes",
        "variant": "ar-lower-features",
        "task": "motif_maize_tis",
    }
    finalized = (
        owner + "/analysis/probes-p1/attempts/old/results/head/analysis-receipt.json"
    )
    body = {
        "phase": phase,
        "error": None,
        "finished": [{"job": job, "receipt": finalized}],
    }
    raw = json.dumps(body).encode()
    record = {
        "uri": owner + "/analysis/" + phase + "/analysis-receipt.json",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (tmp_path / "recovery-plan.json").write_text(
        json.dumps({"phase_receipts": {phase: record}})
    )

    class Storage:
        def get_object(self, *, Bucket, Key):
            assert (
                Bucket == "bucket"
                and Key == "eval/model/analysis/probes-p1/analysis-receipt.json"
            )
            return {"Body": io.BytesIO(raw)}

    recovered = {job_name(job): {"receipt": finalized}}
    assert (
        completed_phase(tmp_path, phase, [job], recovered, owner, Storage()) == record
    )
    with pytest.raises(ValueError, match="incomplete"):
        completed_phase(tmp_path, phase, [job], {}, owner, Storage())
    assert (
        completed_phase(tmp_path, "probes-p2", [job], recovered, owner, Storage())
        is None
    )
    raw = b"{}"  # An immutable receipt cannot be replaced silently.
    with pytest.raises(ValueError, match="differs"):
        completed_phase(tmp_path, phase, [job], recovered, owner, Storage())
