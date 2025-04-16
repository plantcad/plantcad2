DATA_DIR=/workdir/jz963/utils/plantcad2/results/PlantCAD2_tasks/accessible_angiosperm_c600/data

for sp in arabidopsis_thaliana brachypodium_distachyon sorghum_bicolor oryza_sativa
do
    ame --oc ${DATA_DIR}/${sp}_ame_output --control ${DATA_DIR}/${sp}_neg.fasta ${DATA_DIR}/${sp}_pos.fasta ${DATA_DIR}/JASPAR2024_CORE_plants_non-redundant_pfms_meme.txt &
done


