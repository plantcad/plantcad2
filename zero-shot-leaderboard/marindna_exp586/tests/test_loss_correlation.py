import copy
import json
from pathlib import Path
from statistics import mean

import pytest

from exp586_evals.loss_correlation import (
    AF_PROBE,
    AF_ZERO,
    GROUPS,
    assemble,
    downstream,
    validation,
)

ROOT = Path(__file__).parents[1]
MODELS = json.loads((ROOT / "runs/loss-correlation/models.json").read_text())["models"]


def test_archived_score_join_uses_four_group_composite_and_fixed_probe_readout():
    rows = downstream(ROOT, MODELS)
    assert len(rows) == 8
    assert all("language_base" not in row["results"] for row in rows)
    for row in rows:
        assert row["scores"]["Composite"] == pytest.approx(
            mean(row["scores"][g] for g in GROUPS), abs=1e-14
        )
        assert AF_ZERO in row["scores"] and AF_PROBE in row["scores"]
    # These values identify the exact previously fitted readout, rather than
    # whole-window probes, another context, or the smaller zero-shot test split.
    by_code = {r["code"]: r for r in rows}
    assert by_code["M61-L"]["scores"][AF_PROBE] == pytest.approx(0.19298377789700477)
    assert by_code["D216"]["scores"][AF_ZERO] == pytest.approx(0.14401223283262485)
    assert by_code["M481-L"]["scores"]["Composite"] == pytest.approx(0.6212519128718621)
    assert by_code["M61-L-1T"]["scores"]["Composite"] == pytest.approx(
        0.693604777105964
    )
    assert by_code["M61-L-1T"]["scores"][AF_PROBE] == pytest.approx(0.19394317458721153)
    assert by_code["M61-L-1T"]["scores"][AF_ZERO] == pytest.approx(0.12785375362741644)


def test_final_1t_loss_uses_the_same_immutable_sample_and_changes_the_conclusion():
    result = assemble(ROOT, ROOT / "runs/loss-correlation")
    rows = {r["code"]: r for r in result["rows"]}
    old, new = rows["M61-L"], rows["M61-L-1T"]
    assert new["loss"] == pytest.approx(0.8873021401353057)
    assert (
        new["validation"]["samples"]["common-65536"]["input_sha256"]
        == old["validation"]["samples"]["common-65536"]["input_sha256"]
    )
    assert new["loss"] < old["loss"]
    assert new["scores"]["Composite"] < old["scores"]["Composite"]
    assert new["scores"]["Masked motif"] > old["scores"]["Masked motif"]
    assert new["scores"]["Structural variant"] < old["scores"]["Structural variant"]
    assert len(result["without_481m"]["model_codes"]) == 6
    assert "pearson_r" not in json.dumps(result)
    assert "spearman_rho" not in json.dumps(result)
    assert "projection" not in json.dumps(result).lower()


def test_final_downstream_scores_reject_another_checkpoint():
    records = copy.deepcopy(MODELS)
    records[-1]["canonical_metadata_sha256"] = "wrong-checkpoint"
    with pytest.raises(ValueError, match="another checkpoint"):
        downstream(ROOT, records)


def test_validation_join_rejects_wrong_checkpoint_partial_sample_and_failed_replay(
    tmp_path,
):
    row = next(r for r in MODELS if r["code"] == "M61-S")
    result = {
        "checkpoint": row["checkpoint"],
        "checkpoint_digest": row["canonical_metadata_sha256"],
        "samples": {
            "common-65536": {
                "examples": 65536,
                "targets": 65536 * 8191,
                "dropped_assignments": 0,
                "native_loss": 0.95,
            },
            "original-moe-512": {
                "examples": 512,
                "targets": 512 * 8191,
                "native_loss": row["original_loss"],
            },
        },
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result))
    assert validation(row, path) == result
    cases = []
    bad = copy.deepcopy(result)
    bad["checkpoint"] += "-wrong"
    cases.append(bad)
    bad = copy.deepcopy(result)
    bad["samples"]["common-65536"]["examples"] = 64
    cases.append(bad)
    bad = copy.deepcopy(result)
    bad["samples"]["common-65536"]["dropped_assignments"] = 1
    cases.append(bad)
    bad = copy.deepcopy(result)
    bad["samples"]["original-moe-512"]["native_loss"] += 0.01
    cases.append(bad)
    for bad in cases:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            validation(row, path)
