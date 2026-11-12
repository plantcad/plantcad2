"""Exercise native inference through the original task/AF entry points."""

import argparse
import itertools
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from exp586_evals.full_data import assign_task_files, task_plan
from exp586_evals.native import NativeAdapter, _center_crop
from exp586_evals.protocol import original
from exp586_evals.real_metrics import legacy_af_chunks


def tokenizer():
    t = Tokenizer(
        models.WordLevel(
            {"?": 0, "a": 3, "c": 4, "g": 5, "t": 6, "A": 7}, unk_token="?"
        )
    )
    t.normalizer = normalizers.Lowercase()
    t.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    return t


class FakeScorer:
    base_ids = (3, 4, 5, 6)
    batch_size = 8

    def evaluate(self, ids, positions, feature_positions, *, allele_frequency):
        assert not allele_frequency
        p = np.array([0.6, 0.1, 0.2, 0.1], np.float32)
        return {"probabilities": np.tile(p, (len(ids), len(positions[0]), 1))}


def test_original_task_scoring_accepts_native_leaf():
    protocol = original()
    adapter = NativeAdapter(FakeScorer(), tokenizer(), sequence_length=4)
    spec = {
        "kind": "conservation",
        "positions": [1],
        "contexts": ["left", "right_reverse_complement"],
    }
    frame = pd.DataFrame(
        {"sequence": ["AAAA", "ACAA", "AGAA", "ATAA"], "label": [1, 0, 1, 0]}
    )
    args = argparse.Namespace(batch_size=8, softmax_dtype="fp32", use_cache=False)
    result, arrays = protocol.tasks._score_standard_task(
        spec, frame, adapter, None, args, compute_metrics=False
    )
    assert result["contexts"]["left"]["peak_allocated_gib"] is None
    assert result["contexts"]["left"]["timing"]["gpu_seconds"] is None
    np.testing.assert_array_equal(
        arrays["left__scores"], np.array([0.6, 0.1, 0.2, 0.1], np.float32)
    )
    assert protocol.aggregate.recompute_task(spec, frame, arrays)["best_value"] == 1.0


def test_token_ids_follow_normalizer_instead_of_uppercase_vocab():
    NativeAdapter(FakeScorer(), tokenizer())
    with pytest.raises(ValueError, match="character encoding"):
        NativeAdapter(SimpleNamespace(base_ids=[7, 4, 5, 6]), tokenizer())


def test_center_crop_preserves_the_middle_base_and_rejects_asymmetry():
    assert _center_crop("AAAACCCC", 4) == "AACC"
    with pytest.raises(ValueError, match="centered"):
        _center_crop("AAAAAAA", 4)


def test_standard_task_center_crop_shifts_query_positions():
    class RecordingScorer(FakeScorer):
        def evaluate(self, ids, positions, feature_positions, *, allele_frequency):
            assert len(ids[0]) == 4
            assert positions == [[1], [1]]
            return super().evaluate(
                ids, positions, feature_positions, allele_frequency=allele_frequency
            )

    adapter = NativeAdapter(RecordingScorer(), tokenizer(), sequence_length=4)
    probabilities, timing = adapter.score_positions(
        pd.Series(["AACCGGTT", "TTAACCGG"]), [3], 8
    )
    assert probabilities.shape == (2, 1, 4)
    assert timing["tokens"] == 8


def test_original_af_reduction_suffix_and_ambiguity():
    p = original()
    ref = torch.tensor([[-100.0, -100.0, -100.0, -100.0]])
    alt = ref + torch.tensor([[50.0, 1.0, 2.0, 3.0]])
    valid = torch.ones((1, 2, 4), dtype=torch.bool)
    valid[:, :, 2] = False
    acgt = torch.stack([ref, alt], dim=1)
    actual = p.af.reduce_uncached(acgt, acgt, valid, 2)
    # Earlier prefix cancels, ambiguous targets excluded only from ACGT score.
    np.testing.assert_array_equal(actual.numpy(), [[4.0, 6.0, 1.0]])


def test_legacy_af_chunks_project_richer_native_schema(tmp_path):
    columns = ("llr_acgt_fwd", "llr_acgt_rc")
    source = tmp_path / "native"
    source.mkdir()
    np.savez_compressed(
        source / "chunk.npz",
        row_id=np.array([3, 4]),
        llr_acgt_fwd=np.array([0.1, 0.2]),
        llr_acgt_rc=np.array([0.3, 0.4]),
        target_probs_fwd=np.full((2, 4), 0.25),
        target_probs_rc=np.full((2, 4), 0.25),
    )
    chunks = legacy_af_chunks(
        [{"key": "chunk", "start": 0, "stop": 2, "worker": 0}],
        source,
        tmp_path / "legacy",
        columns,
    )
    assert len(chunks) == 1
    with np.load(chunks[0][1], allow_pickle=False) as archive:
        assert set(archive.files) == {"row_id", *columns}
        np.testing.assert_array_equal(archive["row_id"], [3, 4])


def test_full_plan_preserves_task_locality_and_exact_coverage():
    tasks = [
        {"key": "large", "file": "large.tsv"},
        {"key": "medium", "file": "medium.tsv"},
    ]
    inputs = {
        "large.tsv": {
            "files": [
                {"path": "large-0.parquet", "rows": 4104},
                {"path": "large-1.parquet", "rows": 2056},
            ]
        },
        "medium.tsv": {"files": [{"path": "medium-0.parquet", "rows": 1025}]},
    }
    assignment = assign_task_files(tasks, inputs, workers=2)
    plan = task_plan(tasks, inputs, workers=2, chunk_size=2048)
    sizes = {"large": 6160, "medium": 1025}
    for task, size in sizes.items():
        pieces = sorted(
            (piece for piece in plan if piece["task"] == task),
            key=lambda piece: piece["start"],
        )
        assert pieces[0]["start"] == 0 and pieces[-1]["stop"] == size
        assert all(
            left["stop"] == right["start"] for left, right in itertools.pairwise(pieces)
        )
        for piece in pieces:
            assert piece["worker"] == assignment[(task, piece["source_path"])]
    for path in {piece["source_path"] for piece in plan}:
        assert (
            len({piece["worker"] for piece in plan if piece["source_path"] == path})
            == 1
        )
