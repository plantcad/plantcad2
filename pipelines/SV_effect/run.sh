genome="/local/workdir/jz963/utils/plantcad2/results/SV_effect/Arabidopsis_thaliana.TAIR10.dna.toplevel.noMtPt.fa"
python 1_Simulate_SV.py --genome ${genome} --num 20000 --sv_type DEL --output ../../results/SV_effect/Ath_Simulated_DEL_20K.vcf
python 1_Simulate_SV.py --genome ${genome} --num 20000 --sv_type INS --output ../../results/SV_effect/Ath_Simulated_INS_20K.vcf

python 1_Simulate_SV.py --genome ${genome} --num 20000 --sv_type DEL --size_range 1 9 --output ../../results/SV_effect/Ath_Simulated_DEL_1-9_20K.vcf


# run VEP
singularity run -C --bind $PWD --pwd $PWD /programs/ensembl-vep-110.1/vep.sif vep -i Ath_Simulated_DEL_20K.vcf  -o Ath_Simulated_DEL_20K_VEP.vcf --fork 8 -gff myAnnot.gff.gz -fasta ${genome}
singularity run -C --bind $PWD --pwd $PWD /programs/ensembl-vep-110.1/vep.sif vep -i Ath_Simulated_INS_20K.vcf  -o Ath_Simulated_INS_20K_VEP.vcf --fork 8 -gff myAnnot.gff.gz -fasta Arabidopsis_thaliana.TAIR10.dna.toplevel.fa


singularity run -C --bind $PWD --pwd $PWD /programs/ensembl-vep-110.1/vep.sif vep -i Ath_Simulated_DEL_1-9_20K.vcf  -o Ath_Simulated_DEL_1-9_20K.vcf --fork 8 -gff Arabidopsis_thaliana.TAIR10.58.gff3.gz -fasta ${genome}