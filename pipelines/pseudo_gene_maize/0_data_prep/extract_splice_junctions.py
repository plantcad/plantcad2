#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13:43 2024/11/27
@author: JINGJING ZHAI (jz963@cornell.edu; zhaijingjing603@gmail.com)
Description: Extract splice junction sequences from GFF3 and genome FASTA
"""

import gffutils
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pyfaidx
import argparse
import logging

def create_gff_db(gff_path, db_path):
    """Load or create a GFF database"""
    try:
        # Try to load existing database
        return gffutils.FeatureDB(db_path)
    except:
        # If loading fails, create new database
        return gffutils.create_db(gff_path, 
                                dbfn=db_path, 
                                force=True, 
                                keep_order=True, 
                                merge_strategy='create_unique',
                                verbose=True)


def extract_junction_sequences(junction_df, genome_fasta, upstream=255, downstream=255):
    """
    Extract sequences for junction sites from genome
    
    Parameters:
    -----------
    junction_df : pandas DataFrame
        DataFrame with columns: seqname, start, end, strand, tx_name
    genome_fasta : str
        Path to genome FASTA file
    upstream : int
        Number of bases to include upstream of the junction site
    downstream : int
        Number of bases to include downstream of the junction site
    """
    genome = pyfaidx.Fasta(genome_fasta)
    sequences = []
    
    for idx, row in tqdm(junction_df.iterrows(), total=len(junction_df), desc="Extracting sequences"):
        if row['strand'] == '+':
            padded_start = row['start'] - upstream
            padded_end = row['end'] + downstream
        else:
            padded_start = row['start'] - downstream
            padded_end = row['end'] + upstream
        
        try:
            seq = genome[row['seqname']][padded_start-1:padded_end].seq
            if row['strand'] == '-':
                seq = str(Seq(seq).reverse_complement())
            else:
                seq = str(seq)
            
            seq_id = f"{row['tx_name']}_{row['seqname']}_{padded_start}_{padded_end}_{row['strand']}"
            record = SeqRecord(Seq(seq), id=seq_id, description="")
            sequences.append(record)
            
        except Exception as e:
            logging.info(f"Error processing {row['seqname']}:{padded_start}-{padded_end}: {str(e)}")
            continue
    
    return sequences

def get_introns_from_exons(db):
    """Infer introns from exon coordinates within each transcript"""
    introns = db.create_introns()
    res = []
    for intron in tqdm(introns, desc="Inferring introns"):
        res.append({
            'seqname': intron.chrom,
            'start': intron.start,
            'end': intron.end,
            'strand': intron.strand,
            'tx_name': intron.attributes['Parent'][0]
        })
    return pd.DataFrame(res)

def extract_junctions(introns_df, padding=2):
    """
    Extract donor and acceptor junctions from introns DataFrame
    
    Parameters:
    -----------
    introns_df : pandas DataFrame
        DataFrame containing intron information with columns:
        seqname, start, end, strand, tx_name
    padding : int
        Number of bases to include in each junction site (default: 2)
    
    Returns:
    --------
    tuple of (donor_df, acceptor_df)
    """
    donor_df = introns_df.copy()
    acceptor_df = introns_df.copy()
    
    # Process positive strand
    pos_strand_mask = introns_df['strand'] == '+'
    donor_df.loc[pos_strand_mask, 'end'] = donor_df.loc[pos_strand_mask, 'start'] + padding - 1
    acceptor_df.loc[pos_strand_mask, 'start'] = acceptor_df.loc[pos_strand_mask, 'end'] - padding + 1
    
    # Process negative strand
    neg_strand_mask = introns_df['strand'] == '-'
    donor_df.loc[neg_strand_mask, 'start'] = donor_df.loc[neg_strand_mask, 'end'] - padding + 1
    acceptor_df.loc[neg_strand_mask, 'end'] = acceptor_df.loc[neg_strand_mask, 'start'] + padding - 1
    
    return donor_df, acceptor_df



def save_junction_sequences(introns_df, genome_fasta, donor_output, acceptor_output, 
                          junction_padding=2, upstream=255, downstream=255):
    """
    Extract and save donor and acceptor sequences to FASTA files
    """
    # Extract junction coordinates
    donor_sites, acceptor_sites = extract_junctions(introns_df, padding=junction_padding)
    
    # Extract and save sequences
    logging.info("Processing donor sequences...")
    donor_sequences = extract_junction_sequences(donor_sites, genome_fasta, upstream=upstream, downstream=downstream)
    SeqIO.write(donor_sequences, donor_output, "fasta")
    logging.info(f"Wrote {len(donor_sequences)} donor sequences to {donor_output}")
    
    logging.info("Processing acceptor sequences...")
    acceptor_sequences = extract_junction_sequences(acceptor_sites, genome_fasta, upstream=upstream, downstream=downstream)
    SeqIO.write(acceptor_sequences, acceptor_output, "fasta")
    logging.info(f"Wrote {len(acceptor_sequences)} acceptor sequences to {acceptor_output}")

def main():
    parser = argparse.ArgumentParser(description="Extract splice junction sequences from GFF3 and genome FASTA")
    parser.add_argument("--gff3", required=True, help="Input GFF3 file")
    parser.add_argument("--genome", required=True, help="Input genome FASTA file")
    parser.add_argument("--upstream", type=int, default=255, help="Number of bases to include upstream of junction site (default: 255)")
    parser.add_argument("--downstream", type=int, default=255, help="Number of bases to include downstream of junction site (default: 255)")
    parser.add_argument("--prefix", required=True, help="Output prefix for splice dataframe, donor and acceptor sequences")
    parser.add_argument("--junction-padding", type=int, default=2, help="Junction site size (default: 2)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    # Create database
    logging.info("Creating GFF database...")
    db = create_gff_db(args.gff3, args.prefix + "_gff3.db")
    
    # Get introns
    logging.info("Extracting introns...")
    introns = get_introns_from_exons(db)
    introns.to_csv(args.prefix + '_introns.tsv', index=False, sep='\t')
    logging.info(f"Found {len(introns)} introns")
    
    # Extract and save sequences
    save_junction_sequences(
        introns_df=introns,
        genome_fasta=args.genome,
        donor_output=args.prefix + '_donor.fasta',
        acceptor_output=args.prefix + '_acceptor.fasta',
        junction_padding=args.junction_padding,
        upstream=args.upstream,
        downstream=args.downstream
    )

if __name__ == "__main__":
    main()