library(dplyr)
setwd('/workdir/jz963/Expression_modeling/Hai_PNAS_dataset/data/Raw_exp_data')
osa <- read.table(file = 'Osa_TPM.csv', sep = ',', header = T)
metadata <- read.delim(file = 'metadata_enriched.tsv', sep = '\t', 
                       header = T, quote = '', stringsAsFactors = F)
# filter for rice
metadata <- metadata %>% filter(Species == 'Oryza sativa')

# decide using PRJNA609211 with well-watered and select SRR11263077 and SRR11263078 with high pearson correlation
osa <- osa[,c('Geneid', 'SRR11263077', 'SRR11263078')]

# log10 + 1
osa$SRR11263077 <- log10(osa$SRR11263077 + 1)
osa$SRR11263078 <- log10(osa$SRR11263078 + 1)

df <- data.frame(geneID = gsub(pattern = '.MSUv7.0',
                               replacement = '',
                               x = osa$Geneid, ),
                 Label = (osa$SRR11263077 + osa$SRR11263078)/2)


seq <- read.table(file = '/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/max_exp_angiosperm/data/test_Osa.tsv',
                  sep = '\t', header = T, quote = '', stringsAsFactors = F)

df$Seq <- seq$Seq[match(df$geneID, seq$Gene)]
df <- na.omit(df)

# calculate N percentage
N_perc <- stringi::stri_count(df$Seq, fixed = "N")

# the max of N is 73 which is good!
OUTPUTDic <- '/workdir/jz963/utils/plantcad2/results/review/4_gene_exp_more_species/'
write.table(x = df, file = paste0(OUTPUTDic, 'osa_leaf_exp.tsv'),
            sep = '\t', quote = F, row.names = F, col.names = T)

df$Label <- ifelse(df$Label == 0, 0, 1)
write.table(x = df, file = paste0(OUTPUTDic, 'osa_leaf_bin.tsv'),
            sep = '\t', quote = F, row.names = F, col.names = T)
