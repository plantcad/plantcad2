"""Read CoreWeave artifacts with the operator's documented credentials."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def client(*, refresh: bool = False) -> Any:
    import boto3
    from botocore.config import Config

    if refresh:
        for variable, secret in (
            ("CW_KEY_ID", "cw-object-storage-key-id"),
            ("CW_KEY_SECRET", "cw-object-storage-key-secret"),
        ):
            os.environ[variable] = subprocess.check_output(
                [
                    "gcloud",
                    "secrets",
                    "versions",
                    "access",
                    "latest",
                    "--project=hai-gcp-models",
                    "--secret=" + secret,
                ],
                text=True,
                timeout=45,
            ).strip()
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("EXP593_S3_ENDPOINT", "https://cwobject.com"),
        region_name="US-EAST-02A",
        aws_access_key_id=os.environ["CW_KEY_ID"],
        aws_secret_access_key=os.environ["CW_KEY_SECRET"],
        config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 5}),
    )


def read_json(storage: Any, uri: str) -> tuple[dict[str, Any], bytes]:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    payload = storage.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(payload), payload


def upload_verified(storage: Any, path: Path, uri: str) -> dict[str, Any]:
    """Upload immutable content and verify by a streaming direct readback."""
    from botocore.exceptions import ClientError

    bucket, key = uri.removeprefix("s3://").split("/", 1)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    try:
        previous = storage.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchKey"}:
            raise
        storage.upload_file(
            str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": digest}}
        )
    else:
        if (
            previous["ContentLength"] != path.stat().st_size
            or {
                key.lower(): value
                for key, value in previous.get("Metadata", {}).items()
            }.get("sha256")
            != digest
        ):
            raise ValueError("Refusing to replace an immutable artifact")
    remote = storage.get_object(Bucket=bucket, Key=key)["Body"]
    actual, size = hashlib.sha256(), 0
    for chunk in remote.iter_chunks(chunk_size=8 << 20):
        actual.update(chunk)
        size += len(chunk)
    if actual.hexdigest() != digest or size != path.stat().st_size:
        raise ValueError("CoreWeave direct readback differs from upload")
    return {"uri": uri, "sha256": digest, "bytes": size}
