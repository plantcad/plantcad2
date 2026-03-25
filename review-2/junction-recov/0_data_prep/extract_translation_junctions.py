#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14:26 2024/12/02
@author: JINGJING ZHAI (jz963@cornell.edu; zhaijingjing603@gmail.com)
Description: Extract translation start and stop site sequences from GFF3 and genome FASTA
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

def get_longest_transcripts(db):
    """Return set of transcript IDs: one longest mRNA per gene (by total exon length)."""
    longest = {}
    for gene in tqdm(db.features_of_type('gene'), desc="Finding longest transcripts"):
        best_tx, best_len = None, -1
        for tx in db.children(gene, featuretype='mRNA'):
            tx_len = sum(e.end - e.start for e in db.children(tx, featuretype='exon'))
            if tx_len > best_len:
                best_len = tx_len
                best_tx = tx.id
        if best_tx:
            longest[gene.id] = best_tx
    return set(longest.values())


def get_cds_range(db, longest_tx_ids=None):
    """Extract translation start and stop sites from CDS features"""
    res = []
    for transcript in tqdm(db.features_of_type('mRNA'), desc="Processing transcripts"):
        if longest_tx_ids is not None and transcript.id not in longest_tx_ids:
            continue
        cds_features = list(db.children(transcript, featuretype='CDS'))
        if not cds_features:
            continue
            
        # Sort CDS features by position
        cds_features.sort(key=lambda x: x.start)
        
        start_cds = cds_features[0]
        stop_cds = cds_features[-1]
        start_pos = start_cds.start
        stop_pos = stop_cds.end

            
        res.append({
            'seqname': transcript.chrom,
            'tx_name': transcript.id,
            'strand': transcript.strand,
            'start_site': start_pos,
            'stop_site': stop_pos
        })
    
    return pd.DataFrame(res)

def extract_translation_sites(sites_df, site_type='start', padding=3):
    """
    Extract translation start or stop site coordinates
    
    Parameters:
    -----------
    sites_df : pandas DataFrame
        DataFrame containing translation site information
    site_type : str
        Either 'start' or 'stop'
    padding : int
        Number of bases to include in each site (default: 3 for codon)
    """
    site_df = sites_df.copy()
    
    if site_type == 'start':
        # Process positive strand
        pos_strand_mask = site_df['strand'] == '+'
        site_df.loc[pos_strand_mask, 'start'] = site_df.loc[pos_strand_mask, 'start_site']
        site_df.loc[pos_strand_mask, 'end'] = site_df.loc[pos_strand_mask, 'start_site'] + padding - 1
    
        # Process negative strand
        neg_strand_mask = site_df['strand'] == '-'
        site_df.loc[neg_strand_mask, 'start'] = site_df.loc[neg_strand_mask, 'stop_site'] - padding + 1
        site_df.loc[neg_strand_mask, 'end'] = site_df.loc[neg_strand_mask, 'stop_site']
    else:
        # Process positive strand
        pos_strand_mask = site_df['strand'] == '+'
        site_df.loc[pos_strand_mask, 'start'] = site_df.loc[pos_strand_mask, 'stop_site'] - padding + 1
        site_df.loc[pos_strand_mask, 'end'] = site_df.loc[pos_strand_mask, 'stop_site']
    
        # Process negative strand
        neg_strand_mask = site_df['strand'] == '-'
        site_df.loc[neg_strand_mask, 'start'] = site_df.loc[neg_strand_mask, 'start_site']
        site_df.loc[neg_strand_mask, 'end'] = site_df.loc[neg_strand_mask, 'start_site'] + padding - 1
    
    result_df = site_df[['seqname', 'start', 'end', 'strand', 'tx_name']]
    
    # Convert numeric columns to integers
    result_df['start'] = result_df['start'].astype(int)
    result_df['end'] = result_df['end'].astype(int)
    
    
    return result_df

def save_translation_sites(sites_df, genome_fasta, start_output, stop_output,
                         site_padding=3, upstream=255, downstream=255):
    """
    Extract and save translation start and stop site sequences to FASTA files
    """
    # Extract site coordinates
    start_sites = extract_translation_sites(sites_df, site_type='start', padding=site_padding)
    stop_sites = extract_translation_sites(sites_df, site_type='stop', padding=site_padding)
    
    # Extract and save sequences
    logging.info("Processing translation start sites...")
    start_sequences = extract_junction_sequences(start_sites, genome_fasta, 
                                              upstream=upstream, downstream=downstream)
    SeqIO.write(start_sequences, start_output, "fasta")
    logging.info(f"Wrote {len(start_sequences)} start site sequences to {start_output}")
    
    logging.info("Processing translation stop sites...")
    stop_sequences = extract_junction_sequences(stop_sites, genome_fasta,
                                             upstream=upstream, downstream=downstream)
    SeqIO.write(stop_sequences, stop_output, "fasta")
    logging.info(f"Wrote {len(stop_sequences)} stop site sequences to {stop_output}")

def main():
    parser = argparse.ArgumentParser(description="Extract translation start and stop site sequences from GFF3 and genome FASTA")
    parser.add_argument("--gff3", required=True, help="Input GFF3 file")
    parser.add_argument("--genome", required=True, help="Input genome FASTA file")
    parser.add_argument("--upstream", type=int, default=255, help="Number of bases to include upstream of site (default: 255)")
    parser.add_argument("--downstream", type=int, default=255, help="Number of bases to include downstream of site (default: 255)")
    parser.add_argument("--prefix", required=True, help="Output prefix for files")
    parser.add_argument("--site-padding", type=int, default=3, help="Translation site size (default: 3)")
    parser.add_argument("--longest-transcript", action='store_true', help="Keep only the longest transcript per gene")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create database
    logging.info("Creating GFF database...")
    db = create_gff_db(args.gff3, args.prefix + "_gff3.db")

    longest_tx_ids = None
    if args.longest_transcript:
        logging.info("Filtering to longest transcript per gene...")
        longest_tx_ids = get_longest_transcripts(db)
        logging.info(f"Keeping {len(longest_tx_ids)} transcripts")

    # Get translation sites
    logging.info("Extracting translation sites...")
    sites = get_cds_range(db, longest_tx_ids=longest_tx_ids)
    sites.to_csv(args.prefix + '_translation_sites.tsv', index=False, sep='\t')
    logging.info(f"Found {len(sites)} translation site pairs")
    
    # Extract and save sequences
    save_translation_sites(
        sites_df=sites,
        genome_fasta=args.genome,
        start_output=args.prefix + '_start_sites.fasta',
        stop_output=args.prefix + '_stop_sites.fasta',
        site_padding=args.site_padding,
        upstream=args.upstream,
        downstream=args.downstream
    )

if __name__ == "__main__":
    main()