"""Pinned inputs and reporting constants for the maize allele-frequency eval."""

DATASET_REPO = "plantcad/maize-allele-frequency"
DATASET_REVISION = "b2e5138cb7e8d952e6c674c488e89e35e86abe52"
DATASET_FILES = {
    "full/test.parquet": {"bytes": 28_160_879, "sha256": "1419f654431c10ff674e669ceb7f0e0c7d80712eabd5c27f1e75b89074387dc5"},
    "10k_all_consequences/test.parquet": {"bytes": 102_016, "sha256": "6ac316ced008c1a1ced447e52835dc4073fe16377ef95d2559658295003a7365"},
    "20k_all_consequences/test.parquet": {"bytes": 190_853, "sha256": "05692f85b7b8975ff9f62301b15340b82fb0ff2ee55aa3e3d8fb7c702335500d"},
}
CONSEQUENCES = (
    "intergenic_variant",
    "intron_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "3_prime_UTR_variant",
    "synonymous_variant",
    "missense_variant",
    "5_prime_UTR_variant",
    "grouped_splice_region",
    "non_coding_transcript_exon_variant",
    "grouped_start_stop",
)
SAMPLE_SIZES = (10_000, 20_000, 50_000, 100_000)
EXCLUDED_VARIANTS = (
    {
        "chrom": "2",
        "pos": 234_358_020,
        "ref": "A",
        "alt": "T",
        "reason": "The 8,192-bp window contains 771 non-ACGT reference bases; reverse complementation places them in the scored causal suffix.",
    },
)
EXPECTED_SAMPLE_ROWS = {10_000: 9_998, 20_000: 19_997, 50_000: 48_625, 100_000: 94_075}
EXPECTED_UNION_ROWS = 94_953
EXPECTED_GENERATED_SHA256 = {
    "sample-10000.parquet": "18af584707cbc59df8d2f36d274ede9c6ba6775c9fdbcc0d0628068099d0aceb",
    "sample-20000.parquet": "7e8a59c67a9cbb3f3b76ba6a0bd574f5961328c5628ecb27f3aca65bfed7a686",
    "sample-50000.parquet": "cb00dd802443e6cb760fc2372c5d99ec2ba53e9809d02912ce82ca2f833593ca",
    "sample-100000.parquet": "80858691bca0f5e0eee9f05019ebf6f2ab1a812fe21bca810fbbe61e01c73c8d",
    "union.parquet": "9dfc6b3dd38eb856b4d53b5b7c0e6ae84c38c66ef818553376dc5582a57a65e1",
}
SAMPLE_SEED = 42
WINDOW_SIZE = 8192
CONTEXT_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192)

GENOME_URL = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-62/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
GENOME_BYTES = 672_057_339
GENOME_S3_URI = "s3://marin-us-east-02a/MarinDNA/plantcad2-evals/references/zea_mays/release-62/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna_sm.toplevel.fa.gz"
GENOME_SHA256 = "6546dcabce42f8eed50dfbed6bf6a148ca002c6a59e7a623bac9fc9e83aa1196"

HISTORICAL_10K = {
    "PlantCaduceus_l32": {"window": 512, "pearson": 0.167, "spearman": 0.126},
    "PlantCAD2-Small": {"window": 8192, "pearson": 0.164, "spearman": 0.127},
    "PlantCAD2-Medium": {"window": 8192, "pearson": 0.186, "spearman": 0.147},
    "PlantCAD2-Large": {"window": 8192, "pearson": 0.202, "spearman": 0.158},
}
