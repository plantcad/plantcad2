import copy
import json
from pathlib import Path
from statistics import mean

import pytest

from exp586_evals.loss_correlation import (
    AF_PROBE,
    AF_ZERO,
    GROUPS,
    downstream,
    validation,
)

ROOT = Path(__file__).parents[1]
MODELS = json.loads((ROOT / "runs/loss-correlation/models.json").read_text())["models"]


def test_archived_score_join_uses_four_group_composite_and_fixed_probe_readout():
    rows = downstream(ROOT, MODELS)
    assert len(rows) == 7
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
