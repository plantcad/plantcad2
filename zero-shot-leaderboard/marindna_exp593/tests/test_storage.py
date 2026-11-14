import hashlib
import io

import pytest

from exp593_evals.storage import upload_verified


class StoredObject:
    def __init__(self, payload, metadata):
        self.payload = payload
        self.metadata = metadata

    def head_object(self, **kwargs):
        return {"ContentLength": len(self.payload), "Metadata": self.metadata}

    def get_object(self, **kwargs):
        from botocore.response import StreamingBody

        return {"Body": StreamingBody(io.BytesIO(self.payload), len(self.payload))}


@pytest.mark.parametrize("header", ["sha256", "Sha256", "SHA256"])
def test_cached_object_metadata_case_keeps_full_byte_verification(tmp_path, header):
    payload = b"identical scientific artifact"
    sha = hashlib.sha256(payload).hexdigest()
    p = tmp_path / "artifact"
    p.write_bytes(payload)
    assert upload_verified(StoredObject(payload, {header: sha}), p, "s3://b/k") == {
        "uri": "s3://b/k",
        "sha256": sha,
        "bytes": len(payload),
    }


def test_case_normalization_does_not_allow_corrupt_cached_bytes(tmp_path):
    p = tmp_path / "artifact"
    p.write_bytes(b"expected")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="readback"):
        upload_verified(StoredObject(b"corrupt!", {"Sha256": sha}), p, "s3://b/k")


def test_case_normalization_does_not_allow_replacing_other_digest(tmp_path):
    p = tmp_path / "artifact"
    p.write_bytes(b"expected")
    with pytest.raises(ValueError, match="immutable"):
        upload_verified(StoredObject(b"expected", {"Sha256": "0" * 64}), p, "s3://b/k")
