library(GenomicRanges)
library(Biostrings)
library(rtracklayer)
library(dplyr)

setwd('/workdir/jz963/utils/plantcad2/results/review/3_potato_deleterious_mutations/raw')
neu_snps <- data.table::fread('Cell_neutral_snp_pos_maf.bed', sep = '\t', header = T,
                              quote = '', stringsAsFactors = F)
del_snps <- data.table::fread('Cell_Landrace_delesnp.bed', sep = '\t', header = F,
                              quote = '', stringsAsFactors = F)

del_snps <- as.data.frame(del_snps)
neu_snps <- as.data.frame(neu_snps)

del_snps_names <- paste0(del_snps$V1, '_', del_snps$V3)
neu_snps_names <- paste0(neu_snps$chr, '_', neu_snps$pos)

rownames(neu_snps) <- neu_snps_names
rownames(del_snps) <- del_snps_names

neu_snps$label <- 0
del_snps$label <- 1
all_snps <- data.frame(chr = c(del_snps$V1, neu_snps$chr),
                       pos = c(del_snps$V3, neu_snps$pos),
                       label = c(del_snps$label, neu_snps$label))


# get sequences
df <- data.frame(chr = all_snps$chr,
                 start = all_snps$pos - 4095,
                 end = all_snps$pos + 4096,
                 label = all_snps$label)
genomeInfo <- read.table(file = '/workdir/jz963/genomes/Stuberosum/v6.1/assembly/Stuberosum_686_v6.1.fa.fai',
                         sep = '\t', header = F, quote = '', stringsAsFactors = F)
df$chrLen <- genomeInfo$V2[match(df$chr, genomeInfo$V1)]

df <- df %>% filter(start >= 0 & end <= chrLen)
genomePath <- '/workdir/jz963/genomes/Stuberosum/v6.1/assembly/Stuberosum_686_v6.1.fa'

if(file.exists(paste0(genomePath, ".2bit"))){
  genome <- TwoBitFile(paste0(genomePath, ".2bit"))
}else{
  dna <- readDNAStringSet(genomePath)
  dna <- replaceAmbiguities(dna, new = "N")
  export(dna, paste0(genomePath, ".2bit"))
  rm(dna)
  genome <- TwoBitFile(paste0(genomePath, ".2bit"))
}

for(curchr in unique(df$chr)){
  curDF <- df %>% filter(chr == curchr)
  df_gr <- GenomicRanges::GRanges(curDF)
  Seq <- Biostrings::getSeq(genome, df_gr)
  curDF$sequence <- as.character(Seq)  
  write.table(curDF, file = paste0('../processed_v2/', curchr, '.tsv'), sep = '\t',
              quote = F, row.names = F, col.names = T)
}

