## Task Overview

Raw data is from the `TPC2015-00051-LSBR3_Supplemental_Data_set_1.xls` from this [paper](https://academic.oup.com/plcell/article/27/8/2133/6096633)

## Pre-process
```bash
python preprocess.py \
    --gff3_file /workdir/jz963/genomes/Arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.58.gff3 \
    --genome_file /workdir/jz963/genomes/Arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa \
    --upstream_length 500 \
    --output_file ../../../results/Lloyd_2015_essential_genes/Ath_Essential_Gene.tsv
```
