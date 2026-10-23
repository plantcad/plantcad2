import hashlib
import io
import json

import numpy as np
import pytest

from exp586_evals.artifacts import collect, sha256


def test_original_chunk_collection_is_bound_to_inputs_and_checkpoint(tmp_path):
    preparation = tmp_path / "preparation.json"
    chunk = {
        "key": "test--000000-000001",
        "start": 0,
        "stop": 1,
        "task": "test",
        "worker": 0,
    }
    preparation.write_text(json.dumps({"plan": [chunk], "af_plan": []}))
    binding = {
        "requests_sha256": sha256(preparation),
        "condition": "random",
        "checkpoint_digest": "x",
    }
    stream = io.BytesIO()
    np.savez(stream, left__scores=np.array([0.5]))
    blob = stream.getvalue()
    receipt = {
        **chunk,
        "binding": binding,
        "array_sha256": hashlib.sha256(blob).hexdigest(),
    }
    report = {"binding": binding, "chunks": [{"kind": "tasks", "receipt": receipt}]}
    files = {
        "root/manifest.json": json.dumps(report).encode(),
        "root/tasks/chunks/test--000000-000001.json": json.dumps(receipt).encode(),
        "root/tasks/chunks/test--000000-000001.npz": blob,
    }
    collect("root", tmp_path / "out", preparation, "random", files.__getitem__, "x")
    with pytest.raises(ValueError, match="condition mismatch"):
        collect(
            "root", tmp_path / "out", preparation, "pretrained", files.__getitem__, "x"
        )
    files["root/tasks/chunks/test--000000-000001.npz"] = b"bad"
    with pytest.raises(ValueError, match="checksum"):
        collect("root", tmp_path / "out", preparation, "random", files.__getitem__, "x")

    with pytest.raises(ValueError, match="condition mismatch"):
        collect(
            "root",
            tmp_path / "out",
            preparation,
            "random",
            files.__getitem__,
            "wrong-checkpoint",
        )
