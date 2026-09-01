#!/usr/bin/env Rscript
#
# Assign a genomic class to every SNP in the Evolutionary_constraint benchmark,
# starting from nothing but chrom/pos.
#
# This complements stratify_noncoding_type.R, which refines rows already typed
# "Noncoding" by an upstream step. This script needs no pre-existing type column,
# assigns CDS itself, and adds an independent TE overlap flag, so it can annotate
# the raw train/valid tables directly.
#
# Coordinates: `pos` is 1-based on the v3.0.1 assembly and is the center base of
# the window, i.e. sequence index 4096 using 1-based indexing (window spans
# pos-4095 through pos+4096). GRanges is 1-based, so start = end = pos with no
# offset applied.

suppressPackageStartupMessages({
  library(GenomicFeatures)
  library(GenomicRanges)
  library(rtracklayer)
})

args <- commandArgs(trailingOnly = TRUE)

input_tsv <- if (length(args) >= 1) args[[1]] else "/local/workdir/jz963/datasets/Evolutionary_constraint/annotation/snps_sorghum.tsv"
output_tsv <- if (length(args) >= 2) args[[2]] else "/local/workdir/jz963/datasets/Evolutionary_constraint/annotation/snps_sorghum_annotated.tsv"
gff_name <- if (length(args) >= 3) args[[3]] else "/workdir/jz963/genomes/Sorghum_bicolor_v3.1.1/annotation/Sbicolor_454_v3.1.1.gene_exons.gff3"
te_gff_name <- if (length(args) >= 4) args[[4]] else "/workdir/jz963/genomes/Sorghum_bicolor_v3.1.1/annotation/Sbicolor_454_v3.1.1.repeatmasked_assembly_v3.0.1.gff3"
promoter_upstream <- if (length(args) >= 5) as.integer(args[[5]]) else 1000L

message("Reading input: ", input_tsv)
df <- data.table::fread(input_tsv, sep = "\t", header = TRUE, quote = "",
                        stringsAsFactors = FALSE)
df <- as.data.frame(df)

required_cols <- c("chrom", "pos")
missing_cols <- setdiff(required_cols, colnames(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}
message("SNPs read: ", nrow(df))

message("Loading annotation from: ", gff_name)
txdb <- makeTxDbFromGFF(file = gff_name)

message("Building genomic features")
utr5 <- unlist(fiveUTRsByTranscript(x = txdb, use.names = TRUE), use.names = FALSE)
utr3 <- unlist(threeUTRsByTranscript(x = txdb, use.names = TRUE), use.names = FALSE)
cds_gr <- unlist(cdsBy(x = txdb, by = "tx", use.names = TRUE), use.names = FALSE)
txs <- transcripts(txdb, use.names = FALSE)
promoters_gr <- trim(promoters(txs, upstream = promoter_upstream, downstream = 0))
introns_gr <- intronicParts(txdb = txdb)

# Seqnames in the Sorghum GFF are Chr01..Chr10 / super_*, matching the benchmark
# tables as-is. No "chr" prefix is added here, unlike the B73 v5 equivalents.
snps_gr <- GRanges(
  seqnames = df$chrom,
  ranges = IRanges(start = df$pos, end = df$pos)
)

hits_to_flag <- function(query_gr, subject_gr) {
  hits <- findOverlaps(query = query_gr, subject = subject_gr)
  flag <- rep(FALSE, length(query_gr))
  flag[unique(queryHits(hits))] <- TRUE
  flag
}

message("Assigning genomic classes")
is_intron <- hits_to_flag(snps_gr, introns_gr)
is_promoter <- hits_to_flag(snps_gr, promoters_gr)
is_utr5 <- hits_to_flag(snps_gr, utr5)
is_utr3 <- hits_to_flag(snps_gr, utr3)
is_cds <- hits_to_flag(snps_gr, cds_gr)

# Precedence, later overriding earlier. A site can fall in several categories
# across different transcripts; CDS wins so that coding sites are never counted
# as regulatory.
final_type <- rep("intergenic", nrow(df))
final_type[is_intron] <- "intron"
final_type[is_promoter] <- "promoter"
final_type[is_utr5] <- "utr5"
final_type[is_utr3] <- "utr3"
final_type[is_cds] <- "CDS"
df$finalType <- final_type

message("Loading repeat annotation from: ", te_gff_name)
te_gr <- rtracklayer::import(te_gff_name)
# Kept as its own column rather than folded into finalType, so the two are
# independent and either can be conditioned on.
df$is_TE <- hits_to_flag(snps_gr, te_gr)

message("Writing output: ", output_tsv)
write.table(df, file = output_tsv, sep = "\t", quote = FALSE,
            row.names = FALSE, col.names = TRUE)

message("Class counts:")
print(table(df$finalType))

message("Done")
