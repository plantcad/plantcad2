from copy import deepcopy

import pytest

from exp593_evals import RUNS, STAGE_INPUT_TOKENS
from exp593_evals.checkpoints import select_milestones


def record(tokens, step=156559, temporary=False):
    return {
        "checkpoint": "s3://bucket/step-" + str(step),
        "metadata_sha256": "a" * 64,
        "metadata": {
            "exp593_schema": 1,
            "is_temporary": temporary,
            "step": step,
            "exp593_bindings": {
                "run_id": RUNS["text"],
                "schedule": {"training_tokens": STAGE_INPUT_TOKENS},
            },
            "exp593_clock": {
                "updates": step,
                "examples": step * 64,
                "input_tokens": tokens,
            },
        },
    }


def test_only_quarter_permanent_saves():
    target = STAGE_INPUT_TOKENS // 4
    values = [
        record(2161419856, 6316),
        record(49243032600, 142700),
        record(target + 279131),
        record(target + 17, 160000, True),
    ]
    actual = select_milestones(values, RUNS["text"])
    assert [(r["quarter"], r["step"]) for r in actual] == [(1, 156559)]


def test_wrong_scientific_identity_rejected():
    r = record(STAGE_INPUT_TOKENS // 4)
    r["metadata"]["exp593_bindings"]["run_id"] = RUNS["dna216b"]
    with pytest.raises(ValueError, match="identity"):
        select_milestones([r], RUNS["text"])


def test_metadata_is_not_mutated_and_clock_is_checked():
    r = record(STAGE_INPUT_TOKENS)
    before = deepcopy(r)
    assert select_milestones([r], RUNS["text"])[0]["quarter"] == 4
    assert before == r
    r["metadata"]["exp593_clock"]["examples"] += 1
    with pytest.raises(ValueError, match="cursor"):
        select_milestones([r], RUNS["text"])
