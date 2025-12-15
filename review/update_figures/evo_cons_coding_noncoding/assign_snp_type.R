library(parallel)
library(dplyr)
library(ggplot2)
library(rtracklayer)
library(GenomicFeatures)
library(ggridges)
library(tidyr)

setwd('/workdir/jz963/utils/plantcad2/results/review/3_potato_deleterious_mutations/processed_v2/')

gffName <- '/workdir/jz963/genomes/Stuberosum/v6.1/annotation/Stuberosum_686_v6.1.gene_exons.gff3.gz'
txdb <- makeTxDbFromGFF(file = gffName)

CDSbyTx <- cdsBy(x = txdb, by = c('tx', 'gene'), use.names = TRUE)
result <- mclapply(names(CDSbyTx), function(i) {
  range_i <- as.data.frame(range(CDSbyTx[i]))
  range_i$transcript <- i
  range_i
}, mc.cores = 32)
result <- do.call(what = rbind, args = result)

CDSbyTx <- GenomicRanges::GRanges(unique(result))
save(CDSbyTx, file = '/workdir/jz963/genomes/Stuberosum/v6.1/annotation/CDSbyTx.RData')
upstream <- promoters(x = CDSbyTx,
                      upstream = 1500,
                      downstream = 0)
downstream <- flank(x = CDSbyTx, start = FALSE, width = 1500)
introns <- intronicParts(txdb = txdb)
CDS <- cds(x = txdb, columns=c("cds_id", "tx_name"))

TE <- import('/workdir/jz963/genomes/Stuberosum/v6.1/annotation/Stuberosum_686_v6.1.repeatmasked_assembly_v6.1.gff3.gz')

getPercentage <- function(queryGR, subjectGR){
  hits <- findOverlaps(query = queryGR, subject = subjectGR)
  resDF <- rep(0, length(queryGR))
  resDF[unique(queryHits(hits))] <- 1
  resDF
}

df <- read.table(file = 'all_downsampled_with_types.tsv', sep = '\t', header = T, quote = '',
                 stringsAsFactors = F)
sites <- GenomicRanges::GRanges(data.frame(chr = df$chr,
                                           start = df$start + 4095,
                                           end = df$start + 4095))

getPercentage <- function(queryGR, subjectGR){
  hits <- findOverlaps(query = queryGR, subject = subjectGR)
  resDF <- rep(0, length(queryGR))
  resDF[unique(queryHits(hits))] <- 1
  resDF
}
cdsPerc <- getPercentage(queryGR = sites, subjectGR = CDS)
intronsPerc <- getPercentage(queryGR = sites, subjectGR = introns)
upstreamPerc <- getPercentage(queryGR = sites, subjectGR = upstream)
downstreamPerc <- getPercentage(queryGR = sites, subjectGR = downstream)
TEPerc <- getPercentage(queryGR = sites, subjectGR = TE)
resDF <- data.frame(CDS = cdsPerc,
                    Intron = intronsPerc,
                    Upstream = upstreamPerc,
                    Downstream = downstreamPerc,
                    TE = TEPerc)
weights <- list(TE = 5,
                CDS = 4,
                Upstream = 3,
                Intron = 2,
                Downstream = 1)

finalType <- mclapply(X = 1:nrow(resDF), FUN = function(x){
  curMat <- resDF[x, ]
  if(all(as.numeric(curMat) == 0) ){
    return('intergenic')
  }
  curType <- colnames(resDF)[which(curMat == 1)]
  curWeights <- weights[curType]
  curType <- names(curWeights)[which.max(curWeights)]
  return(curType)
}, mc.cores = 32)

finalTypeVec <- unlist(finalType)

df$finalType <- finalTypeVec
write.table(x = df, file = 'all_downsampled_with_types_detailed.tsv', 
            sep = '\t', quote = F, row.names = F, col.names = T)
