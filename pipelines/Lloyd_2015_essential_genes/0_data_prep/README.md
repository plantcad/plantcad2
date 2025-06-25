## Task Overview

Raw data is from the `TPC2015-00051-LSBR3_Supplemental_Data_set_1.xls` from this [paper](https://academic.oup.com/plcell/article/27/8/2133/6096633)

## Genome download
- saccharomyces_cerevisiae: http://sgd-archive.yeastgenome.org/sequence/S288C_reference/genome_releases/S288C_reference_genome_R9-1-1_19990210.tgz
- arabidopsis： TAIR10
- Oryza sativa: Phytozome

## Pre-process
```bash
python preprocess.py \
    --gff3_file /workdir/jz963/genomes/Arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.58.gff3 \
    --genome_file /workdir/jz963/genomes/Arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
    --upstream_length 500 \
    --output_file ../../../results/Lloyd_2015_essential_genes/Ath_Essential_Gene.tsv \
    --sheet_name 'A. thaliana'
```

- For rice
```bash
python preprocess.py \
    --gff3_file /workdir/jz963/genomes/Osativa/annotation/Osativa_323_v7.0.gene_exons.gff3 \
    --genome_file /workdir/jz963/genomes/Osativa/assembly/Osativa_323_v7.0.fa \
    --upstream_length 500 \
    --output_file ../../../results/Lloyd_2015_essential_genes/Osa_Essential_Gene.tsv \
    --sheet_name 'O. sativa'
```

- For yeast
```bash
python preprocess.py \
    --gff3_file /workdir/jz963/genomes/S288C_reference_genome_R64-4-1_20230830/saccharomyces_cerevisiae_R64-4-1_20230830.gff \
    --genome_file /workdir/jz963/genomes/S288C_reference_genome_R64-4-1_20230830/saccharomyces_cerevisiae_R64-4-1_20230830.fasta \
    --upstream_length 500 \
    --output_file ../../../results/Lloyd_2015_essential_genes/Sce_Essential_Gene.tsv \
    --sheet_name 'S. cerevisiae'
```

## Extract translation and splicing junctions
```bash
# ath
OUTPUT_DIR="/workdir/jz963/utils/plantcad2/results/Lloyd_2015_essential_genes"
GFF="/workdir/jz963/genomes/Arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.58.gff3"
GENOME="/workdir/jz963/genomes/Arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa"
python ../../pseudo_gene/0_data_prep/extract_splice_junctions.py \
    --gff3 ${GFF} \
    --genome ${GENOME} \
    --upstream 4095 \
    --downstream 4095 \
    --prefix ${OUTPUT_DIR}/Ath

python ../../pseudo_gene/0_data_prep/extract_translation_junctions.py \
    --gff3 ${GFF} \
    --genome ${GENOME} \
    --upstream 4094 \
    --downstream 4095 \
    --prefix ${OUTPUT_DIR}/Ath

# osa
OUTPUT_DIR="/workdir/jz963/utils/plantcad2/results/Lloyd_2015_essential_genes"
GFF="/workdir/jz963/genomes/Osativa/annotation/Osativa_323_v7.0.gene_exons.gff3"
GENOME="/workdir/jz963/genomes/Osativa/assembly/Osativa_323_v7.0.fa"
python ../../pseudo_gene/0_data_prep/extract_splice_junctions.py \
    --gff3 ${GFF} \
    --genome ${GENOME} \
    --upstream 4095 \
    --downstream 4095 \
    --prefix ${OUTPUT_DIR}/Osa

python ../../pseudo_gene/0_data_prep/extract_translation_junctions.py \
    --gff3 ${GFF} \
    --genome ${GENOME} \
    --upstream 4094 \
    --downstream 4095 \
    --prefix ${OUTPUT_DIR}/Osa
```