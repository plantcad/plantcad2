import json

import pytest

from exp593_evals.panel_stream import cases, digest, header, prepare


def test_bounded_fixture_partitions_preserve_all_original_cases(tmp_path):
    source = tmp_path / "source.json"
    values = [
        {
            "index": i,
            "ids": [32, 64 + i, 77],
            "target_positions": [1, 2],
            "target_bases": [i % 4, -1],
        }
        for i in range(11)
    ]
    source.write_text(
        json.dumps(
            {
                "tokenizer_sha256": "pinned-tokenizer",
                "cases": values,
                "tasks": {"not_needed_by_inference": True},
            }
        )
    )
    stream = tmp_path / "cases.jsonl"
    prepare(source, stream)
    assert header(stream) == {
        "source_sha256": digest(source),
        "tokenizer_sha256": "pinned-tokenizer",
        "cases": 11,
    }
    partitions = [list(cases(stream, worker, 3)) for worker in range(3)]
    assert (
        sorted([c for group in partitions for c in group], key=lambda c: c["index"])
        == values
    )
    assert not any(
        {c["index"] for c in partitions[i]} & {c["index"] for c in partitions[j]}
        for i in range(3)
        for j in range(i)
    )
    stream.write_text("\n".join(stream.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Truncated"):
        list(cases(stream, 0, 1))
