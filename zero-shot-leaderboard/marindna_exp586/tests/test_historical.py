from pathlib import Path

from exp586_evals.historical import dense_results, verify_exp582_results


def test_archived_dense_results_are_complete_and_reproduce_published_aggregates():
    results = dense_results(Path(__file__).resolve().parents[1])
    assert [results[r]["tokens"] for r in ("022t", "039t", "056t")] == [
        216158699520,
        389090902016,
        562022055936,
    ]
    for result in results.values():
        assert len(result["tasks"]) == 20
        assert result["allele_frequency"]["100000"]["actual_rows"] == 94075
        assert set(result["probes"]) == {"whole_window", "variant_token"}
        assert result["probe_split"]["rows"] == {
            "train": 65565,
            "test": 28378,
            "dropped_overlap": 132,
        }


def test_archived_exp582_results_match_their_source_manifest():
    root = Path(__file__).resolve().parents[1]
    manifest = verify_exp582_results(root)
    assert manifest["original_source_commit"] == "92365ad509f56be730b67923c32285e8262a20ee"
    assert len(manifest["files"]) == 19
