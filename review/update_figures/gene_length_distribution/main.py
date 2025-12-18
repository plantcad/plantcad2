import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def main():
    data_dir = "/workdir/og_expression_results/1.raw/phytozome_65_genomes/annotation"
    gff_files = glob.glob(os.path.join(data_dir, "*.gff3"))

    gene_lengths = []

    print(f"Found {len(gff_files)} GFF3 files.")

    for gff_file in gff_files:
        # print(f"Processing {os.path.basename(gff_file)}...")
        with open(gff_file, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                feature_type = parts[2]
                if feature_type == "gene":
                    try:
                        start = int(parts[3])
                        end = int(parts[4])
                        length = end - start + 1
                        gene_lengths.append(length)
                    except ValueError:
                        continue

    print(f"Total genes found: {len(gene_lengths)}")
    if gene_lengths:
        print(f"Average length: {np.mean(gene_lengths):.2f}")
        print(f"Median length: {np.median(gene_lengths)}")
        print(f"Max length: {np.max(gene_lengths)}")
        print(f"Min length: {np.min(gene_lengths)}")

        # Plotting
        plt.style.use('ggplot')
        plt.figure(figsize=(12, 7))
        
        # Log10 transformation for x-axis
        log_gene_lengths = np.log10(gene_lengths)
        
        plt.hist(log_gene_lengths, bins=100, color='#4C72B0', edgecolor='white')
        
        plt.title("Gene Length Distribution across 65 Phytozome Genomes", fontsize=16)
        plt.xlabel("Log10(Gene Length (bp))", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        output_file = "gene_length_distribution.pdf"
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        print("No genes found.")

if __name__ == "__main__":
    main()
