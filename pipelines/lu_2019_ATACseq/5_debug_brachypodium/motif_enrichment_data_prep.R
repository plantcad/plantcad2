library(seqinr)
library(dplyr)

setwd('/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/accessible_angiosperm_c600/data')
dataDIR <- './test_oryza_sativa.tsv'
prefix <- 'oryza_sativa'
df <- read.table(file = dataDIR, sep = '\t', header = T, quote = '', stringsAsFactors = F)

pos <- df %>% filter(Label == 1)
neg <- df %>% filter(Label == 0)

posSeq <- as.list(pos$Seq)
names(posSeq) <- paste0(pos$Chr, '_', pos$Start, '_', pos$End)

negSeq <- as.list(neg$Seq)
names(negSeq) <- paste0(neg$Chr, '_', neg$Start, '_', neg$End)

write.fasta(sequences = posSeq, names = names(posSeq), file.out = paste0(prefix, '_pos.fasta'))
write.fasta(sequences = negSeq, names = names(negSeq), file.out = paste0(prefix, '_neg.fasta'))

