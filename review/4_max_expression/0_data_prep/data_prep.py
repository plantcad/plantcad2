import sys, os
from tqdm import tqdm
import pandas as pd
import pysam
from Bio.Seq import Seq

def extract_sequences(genome_file, cds_info, upstream_length, downstream_length):
    """
    Extract promoter and terminator sequences from the genome.

    Parameters:
        genome_file (str): Path to the genome file.
        cds_info (pd.DataFrame): DataFrame with columns ['Gene ID', 'Chr', 'CDS Start', 'CDS End', 'Strand'].
        upstream_length (int): Length of the upstream region to extract.
        downstream_length (int): Length of the downstream region to extract.

    Returns:
        pd.DataFrame: DataFrame with columns ['Gene ID', 'Seq'].
    """
    fasta = pysam.FastaFile(genome_file)
    cds_info = pd.read_csv(cds_info, sep="\t")

    results = []

    # Iterate over each transcript/gene and extract promoter/terminator sequences
    for _, row in tqdm(cds_info.iterrows(), total=len(cds_info), desc="Extracting promoter and terminator sequences"):
        if "Transcript ID" in row:
            gene_id = row['Transcript ID']
        else:
            gene_id = row['Gene ID']
        chrom = row['Chr']
        cds_start = row['CDS Start'] - 1  # Convert to 0-based index
        cds_end = row['CDS End']
        strand = row['Strand']

        chrom_length = fasta.get_reference_length(chrom)

        if strand == "+":
            promoter_start = max(0, cds_start - upstream_length)
            promoter_seq = fasta.fetch(reference=chrom, start=promoter_start, end=cds_start)

            terminator_end = min(chrom_length, cds_end + downstream_length)
            terminator_seq = fasta.fetch(reference=chrom, start=cds_end, end=terminator_end)
        else:
            promoter_end = min(chrom_length, cds_end + upstream_length)
            promoter_seq = fasta.fetch(reference=chrom, start=cds_end, end=promoter_end)
            promoter_seq = str(Seq(promoter_seq).reverse_complement())

            terminator_start = max(0, cds_start - downstream_length)
            terminator_seq = fasta.fetch(reference=chrom, start=terminator_start, end=cds_start)
            terminator_seq = str(Seq(terminator_seq).reverse_complement())

        resSeq = promoter_seq + terminator_seq

        results.append({
            'id': gene_id,
            'Strand': strand,
            'Seq': resSeq,
        })

    # Close the fasta file to free resources
    fasta.close()

    return pd.DataFrame(results)

tasks = ['exp-max']
upstream = 1024
downstream = 1024
DATA_DIC = '/workdir/jz963/Expression_modeling/a2z/a2z_dataset/tasks/'
haploid = [
    'Andropogon_gerardi/CAM_1351',
    'Andropogon_virginicus/Kellogg_1287-8',
    'Bothriochloa_laguroides/Kellogg_1279-B',
    'Cymbopogon_refractus/AUB_69',
    'Elionurus_tripsacoides/Layton_Zhong_168',
    'Hemarthria_compressa/Kellogg_PI_404118',
    'Heteropogon_contortus/AUB_53-1',
    'Ischaemum_rugosum/Pasquet_1136',
    'Pogonatherum_paniceum/Clark',
    'Schizachyrium_scoparium/CAM_1384',
    'Sorghastrum_nutans/CAM_1369',
    'Thelepogon_elegans/Pasquet_1246',
    'Themeda_triandra/AUB_21-1',
    'Tripsacum_dactyloides/FL_9056069_6',
    'Tripsacum_dactyloides/McKain_334-5'
]

genomes = ['Ag-CAM1351-DRAFT-PanAnd-1.0_Ag00001aa.2.gff3.cdsInfo.tsv',
           'Av-Kellogg1287_8-REFERENCE-PanAnd-1.0_Av00001aa.2.gff3.cdsInfo.tsv',
           'Bl-K1279B-DRAFT-PanAnd-1.0_Bl00001aa.2.gff3.cdsInfo.tsv',
           'Cr-AUB069-DRAFT-PanAnd-1.0_Cr00001aa.2.gff3.cdsInfo.tsv',
           'Et-Layton_Zhong168-DRAFT-PanAnd-1.0_Et00001aa.2.gff3.cdsInfo.tsv',
           'Hp-KelloggPI404118-DRAFT-PanAnd-1.0_Hp00001aa.2.gff3.cdsInfo.tsv',
           'Hc-AUB53_1-DRAFT-PanAnd-1.0_Hc00001aa.2.gff3.cdsInfo.tsv',
           'Ir-Pasquet1136-DRAFT-PanAnd-1.0_Ir00001aa.2.gff3.cdsInfo.tsv',
           'Pi-Clark-DRAFT-PanAnd-1.0_Pi00001aa.2.gff3.cdsInfo.tsv',
           'Ss-CAM1384-DRAFT-PanAnd-1.0_Ss00002aa.2.gff3.cdsInfo.tsv',
           'Sn-CAM1369-DRAFT-PanAnd-1.0_Sn00001aa.2.gff3.cdsInfo.tsv',
           'Te-Pasquet1246-DRAFT-PanAnd-1.0_Te00001aa.2.gff3.cdsInfo.tsv',
           'Tt-AUB21_1-DRAFT-PanAnd-1.0_Tt00001aa.2.gff3.cdsInfo.tsv',
           'Td-FL_9056069_6-REFERENCE-PanAnd-2.0a_Td00001ba.2.gff3.cdsInfo.tsv',
           'Td-McKain334_5-DRAFT-PanAnd-1.0_Td00003aa.2.gff3.cdsInfo.tsv']

cdsInfoDIR = '/workdir/jz963/genomes/panand/PanAnd_v2/'
for task in tasks:
    DATA_DIR = f"{DATA_DIC}{task}/train.tsv"
    train = pd.read_csv(DATA_DIR, sep='\t')
    train = train[train['genome'].isin(haploid)]
    train = train.rename(columns={'targets': 'Label'})
    print('Data shape: ', train.shape)

    trainDF = []
    outputDIR = f'../../../results/PlantCAD2_tasks/{task}'
    os.makedirs(outputDIR, exist_ok=True)
    
    for hap, genome in zip(haploid, genomes):
        curDF = train[train['genome'] == hap]
        cdsInfo = f'{cdsInfoDIR}{genome}'
        genomeFA = f'/workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/{hap}/assembly.fa'
        res = extract_sequences(genomeFA, cdsInfo, upstream, downstream)
        curMat = curDF.merge(res, how = 'left', on = 'id')
        trainDF.append(curMat)
    
    trainDF = pd.concat(trainDF, axis=0, ignore_index=True)
    trainDF = trainDF[trainDF['Seq'].str.len() == (upstream + downstream)]
    trainDF = trainDF.drop(columns=['Strand'])
    trainDF.to_csv(f"{outputDIR}/train.tsv", sep='\t', index=False)

haploid = ['Tripsacum_zopilotense/DC_05_58_3A', 'Zea_diploperennis/Momo']
genomes = ['Tz-DC_05_58_3A-DRAFT-PanAnd-1.0_Tz00001aa.2.gff3.cdsInfo.tsv',
           'Zd-Momo-REFERENCE-PanAnd-1.0_Zd00003aa.2.gff3.cdsInfo.tsv']
cdsInfoDIR = '/workdir/jz963/genomes/panand/PanAnd_v2/'

for task in tasks:
    DATA_DIR = f"{DATA_DIC}{task}/valSpOG.tsv"
    valid = pd.read_csv(DATA_DIR, sep='\t')
    valid = valid.rename(columns={'targets': 'Label'})
    print('Data shape: ', valid.shape)

    validDF = []
    outputDIR = f'../../../results/PlantCAD2_tasks/{task}'

    for hap, genome in zip(haploid, genomes):
        curDF = valid[valid['genome'] == hap]
        cdsInfo = f'{cdsInfoDIR}{genome}'
        genomeFA = f'/workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/{hap}/assembly.fa'
        res = extract_sequences(genomeFA, cdsInfo, upstream, downstream)
        curMat = curDF.merge(res, how = 'left', on = 'id')
        validDF.append(curMat)

    validDF = pd.concat(validDF, axis=0, ignore_index=True)
    validDF = validDF[validDF['Seq'].str.len() == (upstream + downstream)]
    validDF = validDF.drop(columns=['Strand'])
    validDF.to_csv(f"{outputDIR}/valid.tsv", sep='\t', index=False)

cdsInfoDIR = "/workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/NAM_gffs/"
genomes = os.listdir(cdsInfoDIR)
genomes = [g for g in os.listdir(cdsInfoDIR) if g.endswith('.cdsInfo.tsv')]

for task in tasks:
    DATA_DIR = f"{DATA_DIC}{task}/test.tsv"
    test = pd.read_csv(DATA_DIR, sep='\t')
    test = test.rename(columns={'targets': 'Label'})
    print('Data shape: ', test.shape)
    haploid = list(test['genome'].unique())

    testDF = []
    outputDIR = f'../../../results/PlantCAD2_tasks/{task}'

    for hap in haploid:
        curDF = test[test['genome'] == hap]
        taxa = hap.split('/')[-1]
        genome = [g for g in genomes if taxa in g][0]
        cdsInfo = f'{cdsInfoDIR}{genome}'
        genomeFA = f'/workdir/jz963/Expression_modeling/a2z/a2z_dataset/genomes/{hap}/assembly.fa'
        res = extract_sequences(genomeFA, cdsInfo, upstream, downstream)
        curMat = curDF.merge(res, how = 'left', on = 'id')
        testDF.append(curMat)

    testDF = pd.concat(testDF, axis=0, ignore_index=True)
    testDF = testDF[testDF['Seq'].str.len() == (upstream + downstream)]
    testDF = testDF.drop(columns=['Strand'])
    testDF.to_csv(f"{outputDIR}/test.tsv", sep='\t', index=False)




