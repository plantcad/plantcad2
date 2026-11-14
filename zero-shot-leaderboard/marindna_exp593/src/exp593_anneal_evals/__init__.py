"""Final-only lowercase evaluation of the four exp593 annealing recipes."""

from dataclasses import dataclass

OWNER = "s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/"
ARTIFACT_ROOT = OWNER + "evaluations/annealing-final-v1/"


@dataclass(frozen=True)
class Trial:
    key: str
    run_id: str
    source_tokens: int
    stage_tokens: int
    label: str


TRIALS = {
    t.key: t
    for t in (
        Trial(
            "181b",
            "dna-exp593-anneal-181b-guidedfim30-equalbase-seed0-v2",
            181818359808,
            216000000000,
            "182B + 216B FIM",
        ),
        Trial(
            "364b",
            "dna-exp593-anneal-364b-guidedfim30-equalbase-seed0-v2",
            363636719616,
            168000000000,
            "364B + 168B FIM",
        ),
        Trial(
            "545b",
            "dna-exp593-anneal-545b-guidedfim30-equalbase-seed0-v2",
            545455079424,
            120000000000,
            "545B + 120B FIM",
        ),
        Trial(
            "control",
            "dna-exp593-anneal-545b-originaldata-equalbase-seed0-v1",
            545455079424,
            119999561728,
            "545B + 120B AR control",
        ),
    )
}

VARIANTS = (
    "ar-lower-control",
    "fim-minimal-control",
    "ar-lower-exact",
    "fim-minimal-exact",
)
