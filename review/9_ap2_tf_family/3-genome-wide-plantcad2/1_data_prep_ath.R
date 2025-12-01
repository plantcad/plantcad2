setwd('/workdir/jz963/utils/plantcad2/review/9_ap2_tf_family')
options(scipen = 20)
library(rtracklayer)
library(dplyr)

genomeInfo <- read.table(file = '/workdir/jz963/genomes/Arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.fai',
                         sep = '\t', header = F, quote = '', stringsAsFactors = F)

gff <- import('/workdir/jz963/genomes/Arabidopsis_thaliana/Arabidopsis_thaliana.TAIR10.58.gff3')
mRNA <- subset(gff, gff$type == 'mRNA')
# select longest transcript for each gene
mRNA <- as.data.frame(mRNA)
mRNA <- mRNA %>% group_by(Parent) %>% top_n(1, width)
mRNA <- mRNA[which(!duplicated(mRNA$Parent)), ]

canonical <- mRNA$transcript_id

CDS <- subset(gff, gff$type == 'CDS')
CDS <- CDS[which(as.character(CDS$protein_id) %in% canonical)]

cds_ranges <- range(split(CDS, CDS$protein_id))
cds_ranges <- unlist(cds_ranges)

TE <- import('/workdir/jz963/genomes/Arabidopsis_thaliana/TAIR10_Transposable_Elements.bed')

upstreams <- promoters(cds_ranges, upstream = 1500, downstream = 0)
upstreams$name <- names(upstreams)


upstreams <- GenomicRanges::setdiff(upstreams, TE)

res <- upstreams
res <- reduce(res)

res <- as.data.frame(res)

res$chrLen <- genomeInfo$V2[match(res$seqnames, genomeInfo$V1)]
res <- res %>% filter(seqnames %in% 1:5) %>% filter(start - 512 > 0 & end + 512 < chrLen)

pos <- apply(res, 1, FUN = function(x){
  tt <- seq(x[2], x[3], 1)
  df <- data.frame(chr = x[1], pos = tt, strand = x[5])
  df
})
pos <- do.call(what = rbind, args = pos)

res <- data.frame(chr = res$seqnames,
                  start = res$start - 1,
                  end = res$end,
                  four = '.',
                  five = '.',
                  strand = res$strand)

write.table(x = res, file = 'ath_upstream_1.5k.bed', sep = '\t', quote = F, row.names = F,
            col.names = F)

pos <- data.frame(chr = pos$chr,
                  start = pos$pos - 256,
                  end = pos$pos + 256,
                  four = '.',
                  five = '.',
                  strand = pos$strand)
write.table(x = pos[1:9484340, ] , 
            file = 'ath/ath_upstream_downstream_1.5k_singleBP_1.bed', sep = '\t', quote = F, row.names = F,
            col.names = F)
write.table(x = pos[9484341:nrow(pos), ] , 
            file = 'ath/ath_upstream_downstream_1.5k_singleBP_2.bed', sep = '\t', quote = F, row.names = F,
            col.names = F)
