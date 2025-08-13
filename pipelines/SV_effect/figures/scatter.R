library(LSD)
setwd('/local/workdir/jz963/utils/plantcad2/pipelines/SV_effect/figures')
df <- read.table(file = '/workdir/jz963/utils/plantcad2/results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_scores.tsv',
                 sep = '\t', header = T, quote = '', stringsAsFactors = F)
df <- df[!is.na(df$meanPhyloP), ]
pdf(file = 'scatter.pdf', height = 5, width = 15)
par(mfrow = c(1,3))
scc <- cor(df$meanPhyloP, df$pcv2_large*-1, method = 'spearman')
LSD::heatscatter(df$meanPhyloP, df$pcv2_large*-1, 
                 xlab = 'Mean PhyloP', ylab = 'Mean delta lopP',
                 main = paste0('SCC: ', round(scc, 3)))

scc <- cor(df$meanPhyloP, df$pcv2_medium*-1, method = 'spearman')
LSD::heatscatter(df$meanPhyloP, df$pcv2_medium*-1, 
                 xlab = 'Mean PhyloP', ylab = 'Mean delta lopP',
                 main = paste0('SCC: ', round(scc, 3)))

scc <- cor(df$meanPhyloP, df$pcv2_small*-1, method = 'spearman')
LSD::heatscatter(df$meanPhyloP, df$pcv2_small*-1, 
                 xlab = 'Mean PhyloP', ylab = 'Mean delta lopP',
                 main = paste0('SCC: ', round(scc, 3)))

dev.off()

# plot mean phyloP distribution
pdf(file = 'mean_phyloP_distribution.pdf', height = 5, width = 5)
hist(df$meanPhyloP, col = 'sandybrown', xlim = c(-4, 4), xlab = '', main = '')
dev.off()
