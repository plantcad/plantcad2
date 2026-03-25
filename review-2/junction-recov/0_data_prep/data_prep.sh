# extract translation junctions

RESULTS_DIR="../results"
mkdir -p "${RESULTS_DIR}"

assembly=/workdir/jz963/genomes/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
annotation=/workdir/jz963/genomes/ncbi_dataset/data/GCF_000001405.26/genomic.gff
python extract_translation_junctions.py --gff3 "${annotation}" --genome "${assembly}" --upstream 4094 --downstream 4095 --prefix "${RESULTS_DIR}/GCF_000001405.26_GRCh38" --longest-transcript

assembly=/workdir/jz963/genomes/ncbi_dataset/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna
annotation=/workdir/jz963/genomes/ncbi_dataset/data/GCF_000001635.27/genomic.gff
python extract_translation_junctions.py --gff3 "${annotation}" --genome "${assembly}" --upstream 4094 --downstream 4095 --prefix "${RESULTS_DIR}/GCF_000001635.27_GRCm39" --longest-transcript



assembly=/workdir/jz963/genomes/ncbi_dataset/data/GCF_000001405.26/GCF_000001405.26_GRCh38_genomic.fna
annotation=/workdir/jz963/genomes/ncbi_dataset/data/GCF_000001405.26/genomic.gff
python extract_splice_junctions.py --gff3 "${annotation}" --genome "${assembly}" --upstream 4095 --downstream 4095 --prefix "${RESULTS_DIR}/GCF_000001405.26_GRCh38" --longest-transcript

assembly=/workdir/jz963/genomes/ncbi_dataset/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna
annotation=/workdir/jz963/genomes/ncbi_dataset/data/GCF_000001635.27/genomic.gff
python extract_splice_junctions.py --gff3 "${annotation}" --genome "${assembly}" --upstream 4095 --downstream 4095 --prefix "${RESULTS_DIR}/GCF_000001635.27_GRCm39" --longest-transcript

