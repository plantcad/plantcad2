
library(dplyr)
setwd('/workdir/jz963/utils/plantcad2/results/translation_efficiency/arabidopsis/star_alignment')

fnames <- list.files()
fnames <- fnames[grep('SRR157', fnames)]

res <- read.table(file = 'SRR15700008/SRR15700008_gene_stringtie.gtf',
                   sep = '\t', header = T, quote = '', stringsAsFactors = F)

for (i in 2:length(fnames)) {
  path <- paste0(fnames[i], '/', fnames[i], '_gene_stringtie.gtf')
  curMat <- read.table(file = path,
                    sep = '\t', header = T, quote = '', stringsAsFactors = F)
  res[,fnames[i]] <- curMat$TPM[match(res$Gene.ID, curMat$Gene.ID)]
}

scc = cor(res[,9:14])

# get the mean
df <- data.frame(gene = res$Gene.ID,
                 chr = res$Reference,
                 strand = res$Strand,
                 TPM = apply(res[,9:14], 1, mean))

# add upstream sequences
seq <- read.table(file = '/workdir/jz963/genomes/Arabidopsis_thaliana/upstream_500.tsv',
                  sep = '\t', header = T, quote = '', stringsAsFactors = F)
seq$Gene.ID <- gsub(pattern = 'gene:', replacement = '', x = seq$Gene.ID)


interGene <- intersect(seq$Gene.ID, df$gene)
df <- df[df$gene %in% interGene, ]

df$seq <- seq$Promoter.Sequence[match(df$gene, seq$Gene.ID)]


# add gene family information
geneFamily_train <- read.table(file = '../../../PlantCAD2_tasks/max_exp_angiosperm/data/train.tsv',
                               header = T, quote = '', stringsAsFactors = F)
geneFamily_valid <- read.table(file = '../../../PlantCAD2_tasks/max_exp_angiosperm/data/valid.tsv',
                               header = T, quote = '', stringsAsFactors = F)

df$label <- log10(df$TPM + 1)
df <- df %>% filter(chr %in% 1:5)

df_train <- df[df$gene %in% geneFamily_train$Gene, ]
df_valid <- df[df$gene %in% geneFamily_valid$Gene, ]


par(mfrow = c(1,2))
hist(log10(df_train$TPM + 1), freq = FALSE)
hist(log10(df_valid$TPM + 1), freq = FALSE)



dir.create(path = '../../training_data')
write.table(x = df_train, file = '../../training_data/train.tsv',
            sep = '\t', quote = F, row.names = F, col.names = T)
write.table(x = df_valid, file = '../../training_data/valid.tsv',
            sep = '\t', quote = F, row.names = F, col.names = T)


df_train <- read.table(file = '../../training_data/train.tsv', sep = '\t', header = T,
                       quote = '', stringsAsFactors = F)
df_valid <- read.table(file = '../../training_data/valid.tsv', sep = '\t', header = T,
                       quote = '', stringsAsFactors = F)

df_valid$label <- ifelse(df_valid$TPM > 1, 1, 0)
df_train$label <- ifelse(df_train$TPM > 1, 1, 0)

write.table(x = df_train, file = '../../training_data/2_bin/train.tsv',
            sep = '\t', quote = F, row.names = F, col.names = T)
write.table(x = df_valid, file = '../../training_data/2_bin/valid.tsv',
            sep = '\t', quote = F, row.names = F, col.names = T)
