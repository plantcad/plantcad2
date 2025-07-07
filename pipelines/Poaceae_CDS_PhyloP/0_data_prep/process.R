setwd('/workdir/jz963/utils/plantcad2/results/Poaceae_CDS_PhyloP/')
library(seqinr)
CDS <- read.fasta(file = 'ASM1935983.cds5K.unique.fa', as.string = T)



fnames <- list.files(path = './Poaceae_phyloP_platRef_LRT/')

res <- NULL
mismatch <- NULL
for(i in 1:length(fnames)){
  fname <- fnames[i]
  curSeq <- toupper(as.character(CDS[[fname]]))
  curScore <- data.table::fread(input = paste0('Poaceae_phyloP_platRef_LRT/', fname), 
                                sep = '\t', header = T, quote = '', stringsAsFactors = F)
  colnames(curScore)[1] <- 'phyloP'
  
  if(nchar(curSeq)-10000 != nrow(curScore)){
    mismatch <- c(mismatch, fname)
  }else{
    # curBED <- data.frame(OG = fname, pos = 1:(nchar(curSeq)-2000), 
    #                      phyloP = curScore$phyloP)
    # res <- rbind(res, curBED)
  }
}

# write.table(x = res, file = 'Poaceae_phyloP.tsv', sep = '\t', quote = F,
#             row.names = F, col.names = T)


library(dplyr)
library(parallel)
options(scipen = 20)
res <- data.table::fread(input = 'Poaceae_phyloP.tsv', sep = '\t', quote = '', header = T) %>% as.data.frame()
res <- res %>% filter(!OG %in% mismatch)


res$start <- res$pos + 5000 - 2047
res$end <- res$pos + 5000 + 2048


bed <- data.frame(chr = res$OG, start = res$start-1, end = res$end)
write.table(x = bed, file = 'flank_4k.bed', sep = '\t', quote = F, row.names = F, col.names = F)
write.table(x = res, file = 'res_flank_4k.tsv',sep = '\t', quote = F, row.names = F, col.names = T)

# sequence <- apply(res, 1, FUN = function(x){
#   substr(x[6], x[4], x[5])
# })

# sequence <- toupper(sequence)
# table(nchar(sequence))
# 
# res$Seq <- sequence
# 
# # define conserved and neutral sites
# conserved <- res %>% filter(phyloP >= 5)
# neutral <- res %>% filter(phyloP <= 1.5)
# write.table(x = conserved, file = 'conserved_1024.tsv', sep = '\t', quote = F,
#             row.names = F, col.names = T)
# write.table(x = neutral, file = 'neutral_1024.tsv', sep = '\t', quote = F,
#             row.names = F, col.names = T)
