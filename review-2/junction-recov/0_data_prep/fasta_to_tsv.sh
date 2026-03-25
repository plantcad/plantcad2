#!/bin/bash

for fasta in ../results/*.fasta; do
    prefix="${fasta%.fasta}"
    python fasta_to_tsv.py "${fasta}" -o "${prefix}.tsv"
    echo "Done: ${prefix}.tsv"
done
