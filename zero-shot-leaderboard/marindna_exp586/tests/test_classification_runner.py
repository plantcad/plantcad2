"""Frozen feature collection rejects partial, mixed, or misaligned inference."""

import hashlib
import json

import numpy as np
import pytest

from exp586_evals.classification_runner import collect_features


def fixture(tmp_path):
    root = tmp_path / "remote"
    (root / "chunks").mkdir(parents=True)
    plan = []
    for start in (0, 2):
        piece = {
            "task": "example",
            "start": start,
            "stop": start + 2,
            "worker": start,
            "key": f"rows-{start}",
        }
        path = root / "chunks" / (piece["key"] + ".npz")
        np.savez(
            path,
            row_id=np.arange(start, start + 2),
            fwd=np.full((2, 768), start, np.float32),
            rc=np.full((2, 768), -start, np.float32),
        )
        receipt = {
            **piece,
            "binding": {
                "requests_sha256": "request",
                "model_label": "model",
                "checkpoint_metadata_raw_sha256": "checkpoint",
                "implementation_sha256": "implementation",
            },
            "array_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(".json").write_text(json.dumps(receipt))
        plan.append(piece)
    requests = {
        "plan": plan,
        "models": {"model": {"raw_metadata_sha256": "checkpoint"}},
        "tasks": [{"key": "example", "file": "source"}],
        "inputs": {"source": {"rows": 4}},
    }
    return requests, root


def test_collection_is_complete_and_uses_global_row_ids(tmp_path):
    requests, root = fixture(tmp_path)
    arrays = collect_features(
        requests, "request", "model", str(root), "example", tmp_path / "collected"
    )
    np.testing.assert_array_equal(arrays["row_id"], [0, 1, 2, 3])
    np.testing.assert_array_equal(arrays["fwd"][:, 0], [0, 0, 2, 2])
    requests["plan"].pop()
    with pytest.raises(ValueError, match="complete task"):
        collect_features(
            requests, "request", "model", str(root), "example", tmp_path / "partial"
        )


def test_collection_rejects_mixed_backend_implementations(tmp_path):
    requests, root = fixture(tmp_path)
    path = root / "chunks/rows-2.json"
    receipt = json.loads(path.read_text())
    receipt["binding"]["implementation_sha256"] = "changed"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="mixes extraction"):
        collect_features(
            requests, "request", "model", str(root), "example", tmp_path / "mixed"
        )
