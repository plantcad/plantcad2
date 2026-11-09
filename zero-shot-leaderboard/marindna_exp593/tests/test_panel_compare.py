import numpy as np

from exp593_evals.panel_compare import science


def test_scientific_panel_uses_whole_motif_accuracy_and_case_probability_sums():
    logits = np.array(
        [
            [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
        ]
    )
    panel = {
        "cases": [
            {
                "index": 0,
                "readout": "ar-lower",
                "kind": "motif",
                "task": "motif_maize_donor",
                "context": "left",
                "row": 0,
                "target_bases": [0, 1],
            }
        ],
        "tasks": {"motif_maize_donor": {"rows": [{}]}},
    }
    values = science(panel, {0: {"0__logits": logits}}, None)
    assert values[("ar-lower-control", "motif_maize_donor", "left")].tolist() == [1.0]
    assert values[("ar-lower-folded", "motif_maize_donor", "left")].tolist() == [1.0]
    panel["cases"][0]["target_bases"] = [0, 2]
    values = science(panel, {0: {"0__logits": logits}}, None)
    assert values[("ar-lower-folded", "motif_maize_donor", "left")].tolist() == [0.0]


def test_bootstrap_auc_matches_sklearn_with_ties():
    import pytest

    from exp593_evals.panel_compare import bootstrap_metric

    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(593)
    for _ in range(20):
        truth = np.r_[np.zeros(32), np.ones(32)]
        value = rng.integers(0, 10, 64)
        assert bootstrap_metric("core", truth, value) == pytest.approx(
            sklearn.roc_auc_score(truth, value)
        )


def test_af_matches_original_subtract_then_sum_reduction():
    import pytest

    torch = pytest.importorskip("torch")
    cases = []
    data = {}
    reference = np.array([-12.0, -15.0, -19.0], dtype=np.float32)
    alternate = reference + np.array([0.001, -0.002, 0.004], dtype=np.float32)
    for strand in ["fwd", "rc"]:
        for allele, values in [("ref", reference), ("alt", alternate)]:
            i = len(cases)
            cases.append(
                {
                    "index": i,
                    "readout": "ar-lower",
                    "kind": "af",
                    "task": "maize-af",
                    "context": strand,
                    "row": 0,
                    "allele": allele,
                    "target_bases": [0, 1, 2],
                }
            )
            full = np.full((3, 8), -30.0, dtype=np.float32)
            full[np.arange(3), np.arange(3) + 4] = values
            data[f"{i}__full_log_probs"] = full
            data[f"{i}__logits"] = np.zeros((3, 8), dtype=np.float32)
    panel = {"cases": cases, "tasks": {"maize-af": {"rows": [{}]}}}
    result = science(panel, {i: data for i in range(4)}, None)
    expected = float(torch.from_numpy(alternate - reference).sum(dtype=torch.float32))
    assert result[("ar-lower-control", "maize-af", "forward-RC mean")].tolist() == [
        expected
    ]
