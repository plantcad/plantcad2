library(GenomicRanges)
library(Biostrings)
library(rtracklayer)
library(dplyr)

setwd('/workdir/jz963/utils/plantcad2/results/review/3_potato_deleterious_mutations/raw')
all_snps <- data.table::fread('Cell_Landrace_snp.bed', sep = '\t', header = F,
                              quote = '', stringsAsFactors = F)
del_snps <- data.table::fread('Cell_Landrace_delesnp.bed', sep = '\t', header = F,
                              quote = '', stringsAsFactors = F)

del_snps <- as.data.frame(del_snps)
all_snps <- as.data.frame(all_snps)

del_snps_names <- paste0(del_snps$V1, '_', del_snps$V3)
all_snps_names <- paste0(all_snps$V1, '_', all_snps$V3)

rownames(all_snps) <- all_snps_names
rownames(del_snps) <- del_snps_names

all_snps$label <- ifelse(all_snps_names %in% del_snps_names, 1, 0)

# get sequences
df <- data.frame(chr = all_snps$V1,
                 start = all_snps$V3 - 4095,
                 end = all_snps$V3 + 4096,
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

for(curchr in unique(df$chr)[2:12]){
  curDF <- df %>% filter(chr == curchr)
  df_gr <- GenomicRanges::GRanges(curDF)
  Seq <- Biostrings::getSeq(genome, df_gr)
  curDF$sequence <- as.character(Seq)  
  write.table(curDF, file = paste0('../processed/', curchr, '.tsv'), sep = '\t',
              quote = F, row.names = F, col.names = T)
}

