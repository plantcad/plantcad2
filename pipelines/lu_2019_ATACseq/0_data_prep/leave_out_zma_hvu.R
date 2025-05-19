library(dplyr)
setwd('/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/accessible_angiosperm_c600/')

fnames <- list.files(path = 'data/', pattern = 'tsv')
fnames <- fnames[-c(4,10)]


res <- lapply(fnames, function(x) data.table::fread(paste0('data/', x), 
                                                    sep = '\t', header = T) %>% as.data.frame())
res <- do.call(what = rbind, args = res)

shuffled_df <- res[sample(nrow(res)), ]

# remove sequences with more than 20% Ns
numN <- unlist(lapply(shuffled_df$Seq, FUN = function(x) {stringi::stri_count(str = x, regex = 'N')}))
idx <- which(numN == 0)
shuffled_df <- shuffled_df[idx, ]

# 10% as validation
valIdx <- sample(1:nrow(shuffled_df), nrow(shuffled_df)*0.1)
trainIdx <- setdiff(1:nrow(shuffled_df), valIdx)

train <- shuffled_df[trainIdx, ]
valid <- shuffled_df[valIdx, ]

write.table(x = train, file = 'data_leave_out_zma_hvu/train.tsv',
            sep = '\t', quote = F, row.names = F, col.names = T)
write.table(x = valid, file = 'data_leave_out_zma_hvu/valid.tsv',
            sep = '\t', quote = F, row.names = F, col.names = T)
