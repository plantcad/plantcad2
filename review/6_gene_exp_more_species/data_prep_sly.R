library(dplyr)
setwd('/workdir/jz963/Expression_modeling/Hai_PNAS_dataset/data/Raw_exp_data')
sly <- read.table(file = 'Sly_TPM.csv', sep = ',', header = T)
metadata <- read.delim(file = 'metadata_enriched.tsv', sep = '\t', 
                       header = T, quote = '', stringsAsFactors = F)
# filter for rice
metadata <- metadata %>% filter(Species == 'Solanum lycopersicum')

idx <- grep('leaf|Leaf', metadata$Tissue)
interSamples <- intersect(metadata$Run[idx], colnames(sly))

pcc <- cor(sly[,interSamples])
pheatmap::pheatmap(pcc)

# decide using PRJNA505207 with well-watered and select SRR8236682 and SRR8236702 with high pearson correlation
sly <- sly[,c('Geneid', 'SRR8236682', 'SRR8236702')]

# log10 + 1
sly$SRR8236682 <- log10(sly$SRR8236682 + 1)
sly$SRR8236702 <- log10(sly$SRR8236702 + 1)

df <- data.frame(geneID = gsub(pattern = '.ITAG3.2',
                               replacement = '',
                               x = sly$Geneid, ),
                 Label = (sly$SRR8236682 + sly$SRR8236702)/2)


seq <- read.table(file = '/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/max_exp_angiosperm/data/test_Sly.tsv',
                  sep = '\t', header = T, quote = '', stringsAsFactors = F)

df$Seq <- seq$Seq[match(df$geneID, seq$Gene)]
df <- na.omit(df)

# calculate N percentage
N_perc <- stringi::stri_count(df$Seq, fixed = "N")

# the max of N is 82 which is good!
OUTPUTDic <- '/workdir/jz963/utils/plantcad2/results/review/4_gene_exp_more_species/'
write.table(x = df, file = paste0(OUTPUTDic, 'sly_leaf_exp.tsv'),
            sep = '\t', quote = F, row.names = F, col.names = T)

df$Label <- ifelse(df$Label == 0, 0, 1)
write.table(x = df, file = paste0(OUTPUTDic, 'sly_leaf_bin.tsv'),
            sep = '\t', quote = F, row.names = F, col.names = T)
