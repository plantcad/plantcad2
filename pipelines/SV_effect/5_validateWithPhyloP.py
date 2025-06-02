import pandas as pd
import numpy as np
import pyranges as pr
from scipy.stats import spearmanr
import os

pcv2_large = pd.read_csv('../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_score.tsv', 
                         sep='\t')
pcv2_large['mean_pcv2_large'] = pcv2_large.iloc[:, 11:21].mean(axis=1)

res = pd.DataFrame({
    'chr': pcv2_large['Chromosome'],
    'start': pcv2_large['Start'],
    'end': pcv2_large['End'],
    'location': pcv2_large['Location'],
    'mean_pcv2_large': pcv2_large['mean_pcv2_large']
})

phyloP = pd.read_csv('../../results/SV_effect/Ath_PhyloP.bedGraph.gz', 
                         sep='\t', header=None, names=['chr', 'start', 'end', 'score'])
phyloP['chr'] = phyloP['chr'].str.replace('Chr', '', regex=False)

deletions_gr = pr.PyRanges(chromosomes=res['chr'], starts=res['start']-1, ends=res['end'])
phylop_gr = pr.PyRanges(chromosomes=phyloP['chr'], starts=phyloP['start'], ends=phyloP['end'])
phylop_gr.score = phyloP['score'].values

overlap = deletions_gr.join(phylop_gr)
overlap_df = overlap.df.copy()

overlap_df['location'] = (overlap_df['Chromosome'].astype(str) + ':' + 
                          (overlap_df['Start']+1).astype(str) + '-' + 
                          overlap_df['End'].astype(str))

mean_phylop = overlap_df.groupby('location')['score'].mean().reset_index()
res = res.merge(mean_phylop.rename(columns={'score': 'meanPhyloP'}), 
                on='location',
                how='left')
res.to_csv('../../results/SV_effect/outputs/Ath_Simulated_DEL_Len_1-50_withPhyloP.tsv', sep='\t', index=False)