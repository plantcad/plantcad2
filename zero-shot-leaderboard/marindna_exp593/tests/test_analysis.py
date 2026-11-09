import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from exp593_evals.analysis import aligned_features, chunks


def test_alignment_keeps_the_original_noncontiguous_membership():
    arrays = {"row_id": np.arange(6), "fwd": np.arange(12).reshape(6, 2)}
    result = aligned_features(arrays, np.array([1, 4]))
    np.testing.assert_array_equal(result["row_id"], [1, 4])
    np.testing.assert_array_equal(result["fwd"], [[2, 3], [8, 9]])
    with pytest.raises(ValueError, match="no extracted feature"):
        aligned_features(arrays, np.array([1, 6]))
    with pytest.raises(ValueError, match="unique and ordered"):
        aligned_features(arrays, np.array([4, 1]))


def test_af_embedding_receipts_keep_the_paired_lowercase_scorer_binding(
    tmp_path, monkeypatch
):
    from exp593_evals import analysis

    piece = {
        "key": "maize-af--000000-000001",
        "task": "maize-af",
        "start": 0,
        "stop": 1,
    }
    binding = {"checkpoint": "new-model"}
    archive = tmp_path / "original.npz"
    np.savez_compressed(archive, row_id=np.array([0]), fwd=np.ones((1, 2)))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = {
        **piece,
        "binding": {
            **binding,
            "job_profile": {
                "profile": "context-8192",
                "variant": "ar-lower-control",
                "kind": "af",
                "length": 8192,
            },
        },
        "array_sha256": sha,
        "artifact": {
            "uri": "s3://owner/arrays/" + sha + ".npz",
            "sha256": sha,
            "bytes": archive.stat().st_size,
        },
    }

    class Storage:
        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return self

        def paginate(self, **kwargs):
            assert kwargs["Prefix"].endswith("/ar-lower-features/embeddings/chunks/")
            return [{"Contents": [{"Key": piece["key"] + ".json"}]}]

        def get_object(self, **kwargs):
            assert kwargs["Key"] == piece["key"] + ".json"
            return {"Body": io.BytesIO(json.dumps(record).encode())}

    monkeypatch.setattr(analysis, "client", Storage)
    monkeypatch.setattr(analysis, "fetch", lambda *args: archive)
    directory = chunks(
        "s3://owner/evaluation",
        [piece],
        tmp_path / "download",
        binding,
        profile="context-8192",
        variant="ar-lower-features",
        kind="embeddings",
    )
    assert (directory / (piece["key"] + ".npz")).resolve() == archive
    with pytest.raises(ValueError, match="Mixed model"):
        chunks(
            "s3://owner/evaluation",
            [piece],
            tmp_path / "other",
            {"checkpoint": "another-model"},
            profile="context-8192",
            variant="ar-lower-features",
            kind="embeddings",
        )


def test_probe_fitting_files_remain_identical_to_the_original_pin():
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((project / "src/exp593_evals/sources.json").read_text())
    assert (
        manifest["probe_fitting"]["commit"]
        == "cc13e527b9b180177802a342fbda5c6feedcf3a3"
    )
    for relative, expected in manifest["probe_fitting"]["files"].items():
        assert hashlib.sha256((project / relative).read_bytes()).hexdigest() == expected
