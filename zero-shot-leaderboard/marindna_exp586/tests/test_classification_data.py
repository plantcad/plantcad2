"""Holdout boundaries are determined by source genomes, never task outcomes."""

from types import SimpleNamespace

import pandas as pd

from exp586_evals.classification_data import assign_membership, recover_location


def test_whole_chromosome_holdout_and_visible_duplicate_purge():
    frame = pd.DataFrame(
        {
            "chrom": ["1", "2", "2", ""],
            "label": [0, 1, 0, 1],
            "visible_sha256": ["same", "same", "different", "unknown"],
        }
    )
    result = assign_membership(frame, "maize")
    assert result.split.tolist() == ["test", "dropped_duplicate", "train", "unmapped"]
    frame["label"] = 1 - frame.label
    assert assign_membership(frame, "maize").split.tolist() == result.split.tolist()


def test_reverse_mapped_window_expands_query_ends_in_reference_orientation():
    hit = SimpleNamespace(
        is_primary=True,
        mapq=60,
        mlen=2048,
        blen=2048,
        q_st=0,
        q_en=2048,
        r_st=10_000,
        r_en=12_048,
        strand=-1,
        ctg="2",
    )
    aligner = SimpleNamespace(map=lambda sequence: [hit])
    result = recover_location(aligner, "A" * 8192)
    assert result["window_start"] == 10_000 - 3072
    assert result["window_end"] == 12_048 + 3072
