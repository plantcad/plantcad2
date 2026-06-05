#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(GenomicFeatures)
  library(GenomicRanges)
})

args <- commandArgs(trailingOnly = TRUE)

input_tsv <- if (length(args) >= 1) args[[1]] else "/workdir/jz963/utils/plantcad2/results/evolutionary_constraint/valid_with_type.tsv"
output_tsv <- if (length(args) >= 2) args[[2]] else "/workdir/jz963/utils/plantcad2/results/evolutionary_constraint/valid_with_type_stratified.tsv"
gff_name <- if (length(args) >= 3) args[[3]] else "/workdir/jz963/genomes/Sorghum_bicolor_v3.1.1/annotation/Sbicolor_454_v3.1.1.gene_exons.gff3"
promoter_upstream <- if (length(args) >= 4) as.integer(args[[4]]) else 1000L

message("Loading annotation from: ", gff_name)
txdb <- makeTxDbFromGFF(file = gff_name)

message("Building genomic features")
utr5 <- unlist(fiveUTRsByTranscript(x = txdb, use.names = TRUE), use.names = FALSE)
utr3 <- unlist(threeUTRsByTranscript(x = txdb, use.names = TRUE), use.names = FALSE)
txs <- transcripts(txdb, use.names = FALSE)
promoters_gr <- promoters(txs, upstream = promoter_upstream, downstream = 0)
introns_gr <- intronicParts(txdb = txdb)

message("Reading input: ", input_tsv)
df <- read.table(
  file = input_tsv,
  sep = "\t",
  header = TRUE,
  quote = "",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

required_cols <- c("chrom", "pos", "type")
missing_cols <- setdiff(required_cols, colnames(df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

df$type_broad <- df$type
noncoding_idx <- which(df$type == "Noncoding")

if (length(noncoding_idx) == 0) {
  message("No Noncoding rows found. Writing input back out with type_broad column.")
  write.table(df, file = output_tsv, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
  quit(save = "no")
}

message("Stratifying ", length(noncoding_idx), " noncoding rows")
noncoding_gr <- GRanges(
  seqnames = df$chrom[noncoding_idx],
  ranges = IRanges(start = df$pos[noncoding_idx], end = df$pos[noncoding_idx])
)

hits_to_flag <- function(query_gr, subject_gr) {
  hits <- findOverlaps(query = query_gr, subject = subject_gr)
  flag <- rep(FALSE, length(query_gr))
  flag[unique(queryHits(hits))] <- TRUE
  flag
}

is_intron <- hits_to_flag(noncoding_gr, introns_gr)
is_promoter <- hits_to_flag(noncoding_gr, promoters_gr)
is_utr5 <- hits_to_flag(noncoding_gr, utr5)
is_utr3 <- hits_to_flag(noncoding_gr, utr3)

final_type <- rep("intergenic", length(noncoding_idx))
final_type[is_intron] <- "intron"
final_type[is_promoter] <- "promoter"
final_type[is_utr5] <- "utr5"
final_type[is_utr3] <- "utr3"

df$type[noncoding_idx] <- final_type

message("Writing output: ", output_tsv)
write.table(
  df,
  file = output_tsv,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE
)

message("Done")
