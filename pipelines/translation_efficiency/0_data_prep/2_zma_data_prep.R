library(dplyr)
setwd('/workdir/jz963/utils/plantcad2/results/translation_efficiency/maize/ribo_seq/')

metaData <- read.table(file = 'metadata.tsv', sep = '\t', header = F, quote = '',
                       stringsAsFactors = F)
acc <- metaData$V3[17:19]

res <- read.table(file = 'star_alignment/CRR533959/CRR533959_gene_stringtie.gtf',
                  sep = '\t', header = T, quote = '', stringsAsFactors = F)
for (i in 2:length(acc)) {
  path <- paste0('star_alignment/', acc[i], '/', acc[i], '_gene_stringtie.gtf')
  curMat <- read.table(file = path,
                       sep = '\t', header = T, quote = '', stringsAsFactors = F)
  res[,acc[i]] <- curMat$TPM[match(res$Gene.ID, curMat$Gene.ID)]
}
scc = cor(res[,9:11])

# drop CRR533961 because of the low SCC
df <- data.frame(gene = res$Gene.ID,
                 chr = res$Reference,
                 strand = res$Strand,
                 TPM = apply(res[,9:10], 1, mean))
df <- df %>% filter(chr %in% 1:10)
# add upstream sequences
seq <- read.table(file = '/workdir/jz963/genomes/B73_v5/upstream_500.tsv',
                  sep = '\t', header = T, quote = '', stringsAsFactors = F)
seq$Gene.ID <- gsub(pattern = 'gene:', replacement = '', x = seq$Gene.ID)

interGene <- intersect(seq$Gene.ID, df$gene)
df <- df[df$gene %in% interGene, ]

df$seq <- seq$Promoter.Sequence[match(df$gene, seq$Gene.ID)]

df$label <- log10(df$TPM + 1)
write.table(x = df, file = '../../training_data/1_absolute/test.tsv',
            sep = '\t', quote = F, row.names = F, col.names = T)


df$label <- ifelse(df$TPM > 1, 1, 0)
write.table(x = df, file = '../../training_data/2_bin/test.tsv',
            sep = '\t', quote = F, row.names = F, col.names = T)


