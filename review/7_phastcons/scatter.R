library(LSD)
setwd('/workdir/jz963/utils/plantcad2/review/7_phastcons/')
df <- read.table(file = '/workdir/jz963/utils/plantcad2/results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_scores_with_phastcons.tsv',
                 sep = '\t', header = T, quote = '', stringsAsFactors = F)
df <- df[!is.na(df$mean_phastCons), ]

pdf(file = 'scatter.pdf', height = 5, width = 15)
par(mfrow = c(1,3))
scc <- cor(df$mean_phastCons, df$pcv2_large*-1, method = 'spearman')
LSD::heatscatter(df$mean_phastCons, df$pcv2_large*-1, 
                 xlab = 'Mean phastCons', ylab = 'Mean delta lopP',
                 main = paste0('SCC: ', round(scc, 3)))

scc <- cor(df$mean_phastCons, df$pcv2_medium*-1, method = 'spearman')
LSD::heatscatter(df$mean_phastCons, df$pcv2_medium*-1, 
                 xlab = 'Mean phastCons', ylab = 'Mean delta lopP',
                 main = paste0('SCC: ', round(scc, 3)))

scc <- cor(df$mean_phastCons, df$pcv2_small*-1, method = 'spearman')
LSD::heatscatter(df$mean_phastCons, df$pcv2_small*-1, 
                 xlab = 'Mean phastCons', ylab = 'Mean delta lopP',
                 main = paste0('SCC: ', round(scc, 3)))

dev.off()

# plot mean phastCons distribution
pdf(file = 'mean_phastCons_distribution.pdf', height = 5, width = 5)
hist(df$mean_phastCons, col = 'sandybrown', xlim = c(0, 1), xlab = '', main = '')
dev.off()