"""Checkpoint evaluations for the augmented d768 MarinMoE experiment."""

STAGE_INPUT_TOKENS = 216_158_699_520
MAX_BATCH_TOKENS = 64 * 8192
RUNS = {
    "text": "dna-exp593-augmented-d768-pretrained-seed0-v1",
    "dna216b": "dna-exp593-augmented-d768-dna216b-lrm0p5-seed0-v1",
}
OWNER = "s3://marin-us-east-02a/MarinDNA/exp593_augmented_dna/"
ARTIFACT_ROOT = OWNER + "evaluations/plantcad2-v1/"
VARIANTS = (
    "ar-lower-control",
    "ar-lower-folded",
    "ar-upper-folded",
    "fim-minimal-folded",
    "fim-metadata-folded",
)
