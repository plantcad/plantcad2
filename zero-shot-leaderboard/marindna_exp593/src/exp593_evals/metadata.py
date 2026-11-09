"""Outcome-blind metadata selection using the actual training field vocabulary."""

from __future__ import annotations

from functools import cache

from exp593_dna.pipeline import load_registry

SPECIES = {
    "maize": ("Zea mays", "Zmays"),
    "sorghum": ("Sorghum bicolor", "Sbicolor"),
    "arabidopsis": ("Arabidopsis thaliana", "Athaliana"),
    "tomato": ("Solanum lycopersicum", None),
    "pharus": ("Pharus latifolius", None),
}


@cache
def _registry():
    return load_registry()


def metadata_for(
    spec: dict,
    row: dict,
    *,
    reverse_complement: bool = False,
    verified_mapping: dict | None = None,
) -> dict[str, str]:
    """Never select a field from a target, label, consequence, AF bin or split.

    Old sequence-based mappings included the target and are not accepted here.
    A caller supplying a recovered source location must provide a separate
    target-free-flank verification receipt. Unknown fields are omitted, not guessed.
    """
    if spec["kind"] == "af" or "maize" in spec["key"]:
        species = "maize"
    elif spec["kind"] == "sv":
        species = "arabidopsis"
    elif spec["key"] == "conservation_andropogoneae":
        species = "sorghum"
    elif spec["kind"] == "conservation":
        species = "pharus"
    elif "tomato" in spec["key"]:
        species = "tomato"
    else:
        raise ValueError("Unresolved benchmark source species")
    name, assembly = SPECIES[species]
    fields = {"species": name}
    if assembly:
        registry = _registry()[assembly]
        if registry["scientific_name"] != name:
            raise ValueError("Training registry and benchmark species disagree")
        fields.update(assembly=assembly, taxonomy=registry["taxonomy_path"])
    if spec["kind"] in {"af", "sv"}:
        key = "chrom" if spec["kind"] == "af" else "Chromosome"
        fields["chrom"] = str(row[key])
    elif species == "sorghum":
        # Source pipeline documentation fixes all 38,060 test rows to Chr10.
        # pipelines/evolutionary_constraint/0_data_prep/README.md, PR22 pin.
        fields["chrom"] = "Chr10"
    if verified_mapping is not None:
        if verified_mapping.get(
            "target_free_flanks_verified"
        ) is not True or not verified_mapping.get("verification_sha256"):
            raise ValueError("Recovered source metadata lacks target-free verification")
        for key in ("chrom", "source_strand"):
            if verified_mapping.get(key):
                fields[key] = str(verified_mapping[key])
    fields["orientation"] = (
        "reverse complement of the stored window"
        if reverse_complement
        else "same orientation as the stored window"
    )
    return fields
