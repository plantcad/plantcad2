setwd('/workdir/jz963/utils/plantcad2/results/')
library(seqinr)
CDS <- read.fasta(file = 'ASM1935983.cds1K.fa', as.string = T)
seqLen <- lapply(CDS, nchar)
seqLen <- unlist(seqLen)
sum(seqLen)


fnames <- list.files(path = './Poaceae_phyloP_platRef_LRT/')

res <- NULL
mismatch <- NULL
for(i in 18817:length(fnames)){
  fname <- fnames[i]
  curSeq <- toupper(as.character(CDS[[fname]]))
  curScore <- read.table(file = paste0('Poaceae_phyloP_platRef_LRT/', fname), sep = '\t', header = T, quote = '', stringsAsFactors = F)
  colnames(curScore)[1] <- 'phyloP'
  
  if(nchar(curSeq)-2000 != nrow(curScore)){
    mismatch <- c(mismatch, fname)
  }else{
    curBED <- data.frame(OG = fname, pos = 1:(nchar(curSeq)-2000), 
                         phyloP = curScore$phyloP)
    res <- rbind(res, curBED)
  }
}

write.table(x = res, file = 'Poaceae_phyloP.tsv', sep = '\t', quote = F,
            row.names = F, col.names = T)


library(dplyr)
library(parallel)
res <- read.table(file = 'Poaceae_phyloP.tsv', sep = '\t', quote = '', header = T)
res$start <- res$pos + 1000 - 255
res$end <- res$pos + 1000 + 256


# extract sequences 
res$Seq <- unlist(CDS[res$OG])

sequence <- apply(res, 1, FUN = function(x){
  substr(x[6], x[4], x[5])
})

sequence <- toupper(sequence)
table(nchar(sequence))

res$Seq <- sequence

# define conserved and neutral sites
conserved <- res %>% filter(phyloP >= 5)
neutral <- res %>% filter(phyloP <= 1.5)
write.table(x = conserved, file = 'conserved.tsv', sep = '\t', quote = F,
            row.names = F, col.names = T)
write.table(x = neutral, file = 'neutral.tsv', sep = '\t', quote = F,
            row.names = F, col.names = T)
