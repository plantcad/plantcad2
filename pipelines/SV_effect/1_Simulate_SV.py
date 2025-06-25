from Bio import SeqIO
import random
import argparse

def random_dna(length):
    """Generate a random DNA sequence of given length."""
    return ''.join(random.choices("ACGT", k=length))

def simulate_sv(genome, num=100, sv_type="DEL", size_range=(10, 30)):
    vcf_records = []
    for i in range(num):
        chrom = random.choice(list(genome.keys()))
        seq = genome[chrom]
        max_pos = len(seq) - size_range[1] - 2  # avoid overflow
        pos = random.randint(1, max_pos)
        size = random.randint(*size_range)
        end = pos + size

        if sv_type == "DEL":
            ref = seq[pos - 1:end]  # full deleted sequence with anchor base
            alt = ref[0]            # anchor base only
            info = f"SVTYPE=DEL;END={end};SVLEN=-{size}"
        elif sv_type == "INS":
            ref = seq[pos - 1:pos]  # anchor base
            insert_seq = random_dna(size)
            alt = ref + insert_seq
            info = f"SVTYPE=INS;SVLEN={size}"
        else:
            raise ValueError("sv_type must be either 'DEL' or 'INS'")

        vcf_records.append({
            "CHROM": chrom,
            "POS": pos,
            "ID": f"{sv_type.lower()}{i+1}",
            "REF": ref,
            "ALT": alt,
            "INFO": info
        })
    return vcf_records

def write_vcf(filename, records):
    with open(filename, 'w') as f:
        # VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"Type of structural variant\">\n")
        f.write("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of the variant\">\n")
        f.write("##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"Length of the variant\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        # Entries
        for r in records:
            f.write(f"{r['CHROM']}\t{r['POS']}\t{r['ID']}\t{r['REF']}\t{r['ALT']}\t.\t.\t{r['INFO']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate structural variants in a genome.")
    parser.add_argument("--genome", required=True, help="Path to the genome file in FASTA format.")
    parser.add_argument("--num", type=int, default=100, help="Number of variants to simulate.")
    parser.add_argument("--sv_type", choices=["DEL", "INS", "BOTH"], default="BOTH", help="Type of structural variant.")
    parser.add_argument("--size_range", type=int, nargs=2, default=(10, 50), help="Size range for the variants.")
    parser.add_argument("--output", required=True, help="Output VCF file name.")
    args = parser.parse_args()

    genome = {record.id: str(record.seq) for record in SeqIO.parse(args.genome, "fasta")}

    records = []
    if args.sv_type == "BOTH":
        deletions = simulate_sv(genome, num=args.num//2, sv_type="DEL", size_range=args.size_range)
        insertions = simulate_sv(genome, num=args.num//2, sv_type="INS", size_range=args.size_range)
        records = deletions + insertions
    else:
        records = simulate_sv(genome, num=args.num, sv_type=args.sv_type, size_range=args.size_range)
    
    records.sort(key=lambda x: (x["CHROM"], x["POS"]))

    write_vcf(args.output, records)
