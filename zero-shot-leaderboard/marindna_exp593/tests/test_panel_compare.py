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
