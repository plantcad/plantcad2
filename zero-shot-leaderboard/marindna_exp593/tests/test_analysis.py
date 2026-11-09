import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import types
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


def test_normalizing_ambiguity_cannot_hide_a_probe_split_overlap(monkeypatch):
    pd = pytest.importorskip("pandas")
    from exp593_evals.analysis import audit_visible_inputs

    project = Path(__file__).resolve().parents[1]
    path = project / "provenance/probe-fit-code/exp586_evals/probe_protocol.py"
    spec = importlib.util.spec_from_file_location("saved_probe_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = types.ModuleType("exp586_evals")
    package.__path__ = [str(path.parent)]
    monkeypatch.setitem(sys.modules, "exp586_evals", package)
    monkeypatch.setitem(sys.modules, "exp586_evals.probe_protocol", module)
    task = {"kind": "motif", "positions": [4096]}
    source = pd.DataFrame({"sequence": ["R" + "A" * 8191, "N" + "A" * 8191]})
    membership = pd.DataFrame({"row_id": [0, 1], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="normalized visible inputs cross"):
        audit_visible_inputs(task, source, membership)
    source.loc[1, "sequence"] = "C" + "A" * 8191
    report = audit_visible_inputs(task, source, membership)
    assert report["normalized_train_test_duplicates"] == 0
    assert report["rows_with_ambiguous_visible_bases"] == 1
    # Exclusions belong to the immutable original protocol, not new split rows.
    excluded_source = pd.concat(
        [source, source.iloc[[0]], source.iloc[[0]]], ignore_index=True
    )
    excluded_membership = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "split": ["train", "test", "unmapped", "invalid_target"],
        }
    )
    report = audit_visible_inputs(task, excluded_source, excluded_membership)
    assert report["audited_rows"] == 2
    assert report["original_excluded_rows"] == {"unmapped": 1, "invalid_target": 1}
    excluded_membership.loc[2, "split"] = "trani"
    with pytest.raises(ValueError, match="Unexpected membership"):
        audit_visible_inputs(task, excluded_source, excluded_membership)


def test_cv_children_can_import_the_pinned_fitter_namespace(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    project = Path(__file__).resolve().parents[1]
    scientific = tmp_path / "scientific-code" / "exp586_evals"
    scientific.mkdir(parents=True)
    (scientific / "__init__.py").write_text('"""Original scientific wheel."""\n')
    (scientific / "original_marker.py").write_text("ORIGINAL = True\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project / "provenance/probe-fit-code"), str(scientific.parent)]
    )
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    code = """
import numpy as np
from exp586_evals.original_marker import ORIGINAL
from exp586_evals.classification_probe import fit_head
assert ORIGINAL
x=np.random.default_rng(593).normal(size=(120,4)).astype(np.float32)
y=np.arange(120)%2
x[:,0]+=y*2
model,metadata=fit_head(x,y,np.repeat(np.arange(6),20),kind='conservation',jobs=2)
assert np.isfinite(model.decision_function(x)).all()
print('probe-cv-ready')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "probe-cv-ready" in completed.stdout


def test_finite_cpu_parts_cover_each_probe_once_and_keep_af_separate():
    from exp593_evals.analysis_bootstrap import jobs

    catalog = {"probe_tasks": {f"task-{i:02}": {} for i in range(20)}}
    parts = [jobs(f"probes-p{i}", catalog) for i in range(4)]
    assert [len(part) for part in parts] == [5] * 4
    selected = [job["task"] for part in parts for job in part]
    assert sorted(selected) == sorted(catalog["probe_tasks"])
    assert len(set(selected)) == 20
    for length in (8192, 2048):
        assert jobs(f"af-probe{length}", catalog) == [
            {
                "kind": "af-probe",
                "profile": f"context-{length}",
                "variant": "ar-lower-features",
            }
        ]


def test_reused_af_predictions_reject_a_different_checkpoint(tmp_path, monkeypatch):
    from exp593_evals import analysis_bootstrap as module

    job = {"kind": "af", "profile": "context-8192", "variant": "ar-lower-control"}
    values = iter(
        [
            (
                {
                    "error": None,
                    "finished": [{"job": job, "receipt": "s3://wrong/receipt"}],
                },
                b"phase",
            ),
            ({"binding": {"checkpoint": "other"}, **job, "artifacts": []}, b"receipt"),
        ]
    )
    monkeypatch.setattr(module, "client", lambda: object())
    monkeypatch.setattr(module, "read_json", lambda *args: next(values))
    with pytest.raises(ValueError, match="different checkpoint"):
        module.prior_af_predictions(
            "s3://owner/model", "context-8192", {"checkpoint": "current"}, tmp_path
        )
