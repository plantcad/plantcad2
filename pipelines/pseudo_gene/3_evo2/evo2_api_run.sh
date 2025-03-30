DATA_DIR="/workdir/jz963/utils/plantcad2/results/pseudo_gene/splice_junctions/"
OUTPUT="/workdir/jz963/utils/plantcad2/results/pseudo_gene/outputs/"
TAXA="B73"

# start sites
mkdir -p ${OUTPUT}/${TAXA}_start_sites_tmp
awk -v output="$OUTPUT" -v taxa="$TAXA" 'BEGIN {n_seq=0; n_file=1} /^>/ {n_seq++} n_seq>500 {n_file++; n_seq=1} {print > output"/"taxa"_start_sites_tmp/split_"n_file".fa"}' "${DATA_DIR}${TAXA}_start_sites_filtered.fasta"

python evo2_api.py --fasta ${DATA_DIR}${TAXA}_start_sites_filtered.fasta \
 --length 2046 \
 --output ${OUTPUT}/${TAXA}_start_sites_tmp \
 --n-tokens 3

# stop sites
mkdir -p ${OUTPUT}/${TAXA}_stop_sites_tmp
python evo2_api.py --fasta ${DATA_DIR}${TAXA}_stop_sites_filtered.fasta \
 --length 2046 \
 --output ${OUTPUT}/${TAXA}_stop_sites_tmp \
 --n-tokens 3

# donor sites
mkdir -p ${OUTPUT}/${TAXA}_donor_tmp
awk -v output="$OUTPUT" -v taxa="$TAXA" 'BEGIN {n_seq=0; n_file=1} /^>/ {n_seq++} n_seq>500 {n_file++; n_seq=1} {print > output"/"taxa"_donor_tmp/split_"n_file".fa"}' "${DATA_DIR}${TAXA}_donor_filtered.fasta"
find ${OUTPUT}/${TAXA}_donor_tmp -name "*.fa" | parallel -j 64 python evo2_api.py --fasta {} \
 --length 2047 \
 --output ${OUTPUT}/${TAXA}_donor_tmp/ \
 --n-tokens 2

# acceptor sites
mkdir -p ${OUTPUT}/${TAXA}_acceptor_tmp
awk -v output="$OUTPUT" -v taxa="$TAXA" 'BEGIN {n_seq=0; n_file=1} /^>/ {n_seq++} n_seq>500 {n_file++; n_seq=1} {print > output"/"taxa"_acceptor_tmp/split_"n_file".fa"}' "${DATA_DIR}${TAXA}_acceptor_filtered.fasta"
find ${OUTPUT}/${TAXA}_acceptor_tmp -name "*.fa" | parallel -j 64 python evo2_api.py --fasta {} \
 --length 2047 \
 --output ${OUTPUT}/${TAXA}_acceptor_tmp/ \
 --n-tokens 2