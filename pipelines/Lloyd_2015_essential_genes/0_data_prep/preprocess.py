import pandas as pd
from collections import defaultdict
import sys
from tqdm import tqdm
import argparse
import pysam
from Bio.Seq import Seq
def extract_cds_coordinates(gff3_file):
    """
    Extract CDS coordinates (TSS to stop codon) and strand for each transcript from a GFF3 file.
    
    Parameters:
        gff3_file (str): Path to the GFF3 file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Transcript ID', 'Chr', 'CDS Start', 'CDS End', 'Strand'].
    """
    transcript_data = []

    gff3_df = pd.read_csv(gff3_file, sep="\t", comment="#", header=None, 
                          names=["seqid", "source", "type", "start", "end", "score", 
                                 "strand", "phase", "attributes"])
    cds_df = gff3_df[gff3_df['type'] == 'CDS']
    
    cds_ranges = defaultdict(lambda: {"start": None, "end": None, "strand": None})

    # Extract CDS range and strand information
    for _, row in tqdm(cds_df.iterrows(), total=len(cds_df), desc="Extracting CDS info"):
        attributes = row['attributes']
        seqid = row['seqid']
        start = row['start']
        end = row['end']
        strand = row['strand']
        
        # Extract transcript ID (Parent attribute)
        transcript_id = None
        for attribute in attributes.split(";"):
            if attribute.startswith("Parent="):
                transcript_id = attribute.split("=")[1]
                break

        if transcript_id:
            # Update or initialize the CDS range for this transcript
            current_range = cds_ranges[transcript_id]
            current_range["seqid"] = seqid
            current_range["start"] = min(current_range["start"], start) if current_range["start"] is not None else start
            current_range["end"] = max(current_range["end"], end) if current_range["end"] is not None else end
            current_range["strand"] = strand

    # Convert the accumulated ranges to a list format
    for transcript_id, info in cds_ranges.items():
        transcript_data.append([transcript_id, info["seqid"], info["start"], info["end"], info["strand"]])
    
    return pd.DataFrame(transcript_data, columns=["Transcript ID", "Chr", "CDS Start", "CDS End", "Strand"])

def extract_gene_and_transcript_ids(gff3_file):
    """
    Extract transcript ID and Gene from a GFF3 file.
    
    Parameters:
        gff3_file (str): Path to the GFF3 file.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Gene', 'Transcript ID'].
    """
    # List to hold rows for the DataFrame
    data = []

    # Read the GFF3 file into a pandas DataFrame
    gff3_df = pd.read_csv(gff3_file, sep="\t", comment="#", header=None, 
                          names=["seqid", "source", "type", "start", "end", "score", 
                                 "strand", "phase", "attributes"])
    
    # Filter rows where the type is mRNA (for transcript IDs)
    mRNA_df = gff3_df[gff3_df['type'] == 'mRNA']
    
    # Extract gene and transcript IDs
    for _, row in tqdm(mRNA_df.iterrows(), total=len(mRNA_df), desc="Extracting Gene/Transcript info"):
        attributes = row['attributes']
        
        transcript_id = None
        gene_id = None
        
        for attribute in attributes.split(";"):
            if attribute.startswith("ID="):
                transcript_id = attribute.split("=")[1]  # Extract transcript ID
            elif attribute.startswith("Parent="):
                gene_id = attribute.split("=")[1]  # Extract Gene
        
        # Append to the data list if both transcript_id and gene_id are found
        if transcript_id and gene_id:
            data.append([gene_id, transcript_id])
    
    return pd.DataFrame(data, columns=["Gene", "Transcript ID"])

def extract_sequences(genome_file, cds_info, upstream_length):
    """
    Extract upstream from ATG and until 8192 bp of each transcript.
    
    Parameters:
        genome_file (str): Path to the genome file.
        cds_info (pd.DataFrame): DataFrame with columns ['Gene', 'Chr', 'CDS Start', 'CDS End', 'Strand'].
        upstream_length (int): Length of the upstream region to extract.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['Gene', 'Promoter Sequence', 'Terminator Sequence'].
    """
    fasta = pysam.FastaFile(genome_file)
    # cds_info = pd.read_csv(cds_info, sep="\t")

    results = []
    
    # Iterate over each transcript/gene and extract promoter/terminator sequences
    for _, row in tqdm(cds_info.iterrows(), total=len(cds_info), desc="Extracting sequences"):
        if "Transcript ID" in row:
            gene_id = row['Transcript ID']
        else:
            gene_id = row['Gene']
        
        chrom = str(row['Chr'])
        cds_start = row['CDS Start'] - 1  # Convert to 0-based index
        cds_end = row['CDS End']
        strand = row['Strand']
        
        # Get the length of the chromosome to avoid fetching sequences out of bounds
        chrom_length = fasta.get_reference_length(chrom)
        
        # Extract sequences based on strand orientation
        if strand == "+":
            seq_start = max(0, cds_start - upstream_length)
            seq_end = min(chrom_length, seq_start + 8192)
            seq = fasta.fetch(reference=chrom, start=seq_start, end=seq_end)
            
        else:
            seq_end = min(chrom_length, cds_end + upstream_length)
            seq_start = max(0, seq_end - 8192)
            seq = fasta.fetch(reference=chrom, start=seq_start, end=seq_end)
            seq = str(Seq(seq).reverse_complement())

        results.append({
            'Gene': gene_id,
            'Strand': strand,
            'Seq': seq
        })
    
    fasta.close()
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Extract promoter and terminator sequences from genome based on GFF3 file.")
    parser.add_argument("--gff3_file", type=str, required=True, help="Path to the GFF3 file.")
    parser.add_argument("--genome_file", type=str, required=True, help="Path to the genome file.")
    parser.add_argument("--upstream_length", type=int, default=500, help="Length of the upstream region to extract.")
    parser.add_argument("--output_file", type=str, default="promoter_terminator_sequences.csv", help="Output file name.")
    parser.add_argument("--sheet_name", type=str, default="A. thaliana", help="Sheet name for the Excel file.")
    
    args = parser.parse_args()
    
    cds_df = extract_cds_coordinates(args.gff3_file)
    if args.sheet_name == 'S. cerevisiae': # only for S. cerevisiae because its weird gff file
        temp_df = cds_df.copy()
        temp_df['Transcript ID'] = temp_df['Transcript ID'].str.split(',')
        cds_df = temp_df.explode('Transcript ID', ignore_index=True)

    gene_tx_df = extract_gene_and_transcript_ids(args.gff3_file)
    
    # Merge dataframes on Transcript ID
    merged_df = pd.merge(gene_tx_df, cds_df, on="Transcript ID")
    
    result = merged_df.groupby('Gene').agg({'Chr': 'first', 'CDS Start': 'min', 'CDS End': 'max', 'Strand': 'first'}).reset_index()
    
    # Extract sequences
    result_df = extract_sequences(args.genome_file, result, args.upstream_length)
    df = pd.read_excel("TPC2015-00051-LSBR3_Supplemental_Data_set_1.xls", sheet_name=args.sheet_name, skiprows=3)
    
    if args.sheet_name == 'A. thaliana':
        result_df['Gene'] = result_df['Gene'].str.replace("gene:", "")
        df.columns = ['Gene', 'Phenotype', 'Predicted']
        df = df[df['Predicted'] == '-']
    elif args.sheet_name == 'O. sativa':
        result_df['Gene'] = result_df['Gene'].str.replace(".MSUv7.0", "")
    elif args.sheet_name == 'S. cerevisiae':
        pass
    else:
        raise ValueError(f"Unsupported sheet name '{args.sheet_name}'. Supported sheet names are 'A. thaliana', 'O. sativa', and 'S. cerevisiae'.")


    df['Label'] = df['Phenotype'].apply(lambda x: 1 if x=='Lethal' else 0)
    res = pd.merge(df, result_df, on='Gene', how='left')
    res = res[['Gene', 'Seq', 'Label']]
    res = res[res['Seq'].str.len() == 8192]
    res.to_csv(args.output_file, index=False, sep="\t")

if __name__ == "__main__":
    main()
