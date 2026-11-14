"""Bounded per-worker fixture access; preserve the immutable panel's exact cases."""

import gc
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source: Path, output: Path):
    panel = json.loads(source.read_text())
    header = {
        "source_sha256": digest(source),
        "tokenizer_sha256": panel["tokenizer_sha256"],
        "cases": len(panel["cases"]),
    }
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(header) + "\n")
        for index, case in enumerate(panel["cases"]):
            if case["index"] != index:
                raise ValueError("Source cases are not contiguous")
            stream.write(
                str(index) + "\t" + json.dumps(case, separators=(",", ":")) + "\n"
            )
    temporary.replace(output)
    del panel
    gc.collect()
    return header


def header(path: Path):
    with path.open() as stream:
        return json.loads(stream.readline())


def cases(path: Path, worker: int, workers: int):
    with path.open() as stream:
        info = json.loads(stream.readline())
        count = 0
        for count, line in enumerate(stream, 1):
            index, record = line.split("\t", 1)
            if int(index) != count - 1:
                raise ValueError("Stream case ordering differs")
            if int(index) % workers == worker:
                value = json.loads(record)
                if value["index"] != int(index):
                    raise ValueError("Stream record identity differs")
                yield value
        if count != info["cases"]:
            raise ValueError("Truncated fixture stream")
