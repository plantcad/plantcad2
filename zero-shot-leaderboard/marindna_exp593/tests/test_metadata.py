import pytest

from exp593_evals.metadata import metadata_for


def test_outcomes_do_not_change_any_rendered_metadata():
    spec = {"key": "maize-af", "kind": "af"}
    a = {
        "chrom": "10",
        "AF": 0.001,
        "alt": "A",
        "ref": "G",
        "consequence": "synonymous",
    }
    b = {**a, "AF": 0.99, "alt": "T", "ref": "C", "consequence": "missense"}
    assert metadata_for(spec, a) == metadata_for(spec, b)
    assert set(metadata_for(spec, a)) == {
        "species",
        "assembly",
        "taxonomy",
        "chrom",
        "orientation",
    }


def test_unverified_target_containing_mappings_are_rejected():
    with pytest.raises(ValueError, match="target-free"):
        metadata_for(
            {"key": "motif_maize_tis", "kind": "motif"},
            {},
            verified_mapping={"chrom": "1", "strand": "+"},
        )


def test_unknown_training_assembly_codes_are_not_invented():
    for key, kind in [
        ("motif_tomato_tis", "motif"),
        ("conservation_poaceae_tis", "conservation"),
    ]:
        fields = metadata_for({"key": key, "kind": kind}, {})
        assert (
            "species" in fields
            and "assembly" not in fields
            and "taxonomy" not in fields
        )
