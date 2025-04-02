# extract translation junctions
ls /workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/Zea_mays | while read line
do
    assembly=/workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/Zea_mays/${line}/assembly.fa
    gff=$(find /workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/NAM_gffs -name "Zm-${line}*.gff3")
    python extract_translation_junctions.py --gff3 "${gff}" --genome "${assembly}" --upstream 4094 --downstream 4095 --prefix "${line}" &
done


# extract splice junctions
ls /workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/Zea_mays | while read line
do
    assembly=/workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/Zea_mays/${line}/assembly.fa
    gff=$(find /workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/NAM_gffs -name "Zm-${line}*.gff3")
    python extract_splice_junctions.py --gff3 "${gff}" --genome "${assembly}" --upstream 4095 --downstream 4095 --prefix "${line}" &
done



ls *fasta | parallel -j 8 'taxa=$(echo {} | sed "s/_.*//g"); \
prefix=$(basename {} .fasta); \
python generate_labels.py --input {} --taxa $taxa --output ${prefix}_filtered.fasta'

