import pandas as pd
from Bio import SeqIO
import sys


OUTPUT_DIR = '/workdir/jz963/utils/plantcad2/results/Lloyd_2015_essential_genes/'
prefix = sys.argv[1]
# Load the data
df = pd.read_csv(f'{OUTPUT_DIR}{prefix}_Essential_Gene.tsv', sep='\t')

# load translation start sites, stop sites, splice donor and acceptor fasta
def load_fasta(fasta_path):
    """
    Load a FASTA file and return names and sequences, respectively, remove sequences that are not 8192bp
    """
    records = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    names = []
    sequences = []
    for name, record in records.items():
        if len(record.seq) == 8192:
            names.append(name)
            sequences.append(str(record.seq))
    return names, sequences

def filter_and_label_sequences(names, seqs, gene_to_label, prefix):
    filtered_names = []
    filtered_seqs = []
    labels = []
    for idx, name in enumerate(names):
        if prefix == 'Ath':
            transcript = name.split('_')[0].replace('transcript:', '')
            gene_id = transcript[:9]
        elif prefix == 'Osa':
            gene_id = name[:14]
        elif prefix == 'Sce':
            gene_id = transcript
        else:
            raise ValueError(f"Unsupported prefix: {prefix}")
        if gene_id in gene_to_label:
            filtered_names.append(name)
            filtered_seqs.append(seqs[idx])
            labels.append(gene_to_label[gene_id])
    return filtered_names, filtered_seqs, labels

def write_fasta_file(names, seqs, output_path):
    with open(output_path, 'w') as fasta_file:
        for i in range(len(names)):
            fasta_file.write(f">{names[i]}\n")
            fasta_file.write(f"{seqs[i]}\n")

def write_label_file(names, labels, output_path):
    with open(output_path, 'w') as label_file:
        label_file.write("Name\tLabel\n")
        for name, label in zip(names, labels):
            label_file.write(f"{name}\t{label}\n")

tis_names, tis_seqs = load_fasta(f'{OUTPUT_DIR}{prefix}_start_sites.fasta')
tss_names, tss_seqs = load_fasta(f'{OUTPUT_DIR}{prefix}_stop_sites.fasta')
donor_names, donor_seqs = load_fasta(f'{OUTPUT_DIR}{prefix}_donor.fasta')
acceptor_names, acceptor_seqs = load_fasta(f'{OUTPUT_DIR}{prefix}_acceptor.fasta')

gene_to_label = dict(zip(df['Gene'], df['Label']))

site_types = {
    'tis': (tis_names, tis_seqs),
    'tss': (tss_names, tss_seqs),
    'donor': (donor_names, donor_seqs),
    'acceptor': (acceptor_names, acceptor_seqs)
}

for site, (names, seqs) in site_types.items():
    filtered_names, filtered_seqs, labels = filter_and_label_sequences(names, seqs, gene_to_label, prefix)
    write_fasta_file(filtered_names, filtered_seqs, f'{OUTPUT_DIR}{prefix}_filtered_{site}.fasta')
    write_label_file(filtered_names, labels, f'{OUTPUT_DIR}{prefix}_filtered_{site}_labels.tsv')
