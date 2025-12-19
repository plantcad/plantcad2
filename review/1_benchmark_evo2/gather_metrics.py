import glob
import os
import re

species_list = [
    "brachypodium_distachyon",
    "glycine_max",
    "hordeum_vulgare",
    "oryza_sativa",
    "phaseolus_vulgaris",
    "sorghum_bicolor",
    "zea_mays"
]

models = [
    "plantcad2_small_mean",
    "plantcad2_small_max",
    "plantcad2_medium_mean",
    "plantcad2_medium_max",
    "plantcad2_large_mean",
    "plantcad2_large_max",
    "evo2_forward",
    "evo2_reverse",
    "evo2_average",
    "evo2_concatenate"
]

base_dir = "/workdir/jz963/utils/plantcad2/results/review/1_benchmark_evo2"

def get_auprc(filepath):
    if not os.path.exists(filepath):
        return "N/A"
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            match = re.search(r"AUPRC:\s+([0-9.]+)", content)
            if match:
                return match.group(1)
    except Exception:
        pass
    return "N/A"

print("| Species | Model | AUPRC (XGBoost) | AUPRC (Neural Network) |")
print("|---|---|---|---|")

for species in species_list:
    for model in models:
        # Construct filenames
        # PlantCAD2 files are in plantcad2/
        # Evo2 files are in evo2/
        
        if "plantcad2" in model:
            subdir = "plantcad2"
            # Format: xgb_plantcad2_small_mean_{species}_results.tsv
            xgb_file = os.path.join(base_dir, subdir, f"xgb_{model}_{species}_results.tsv")
            nn_file = os.path.join(base_dir, subdir, f"nn_{model}_{species}_results.tsv")
        else:
            subdir = "evo2"
            # Format: xgb_evo2_forward_{species}_results.tsv
            # The model string is like "evo2_forward", extracting purely the strategy part might be tricky if naming is inconsistent.
            # Let's verify naming.
            # Based on user output: nn_evo2_reverse_hordeum_vulgare_results.tsv
            # So naming is nn_{model}_{species}_results.tsv
            xgb_file = os.path.join(base_dir, subdir, f"xgb_{model}_{species}_results.tsv")
            nn_file = os.path.join(base_dir, subdir, f"nn_{model}_{species}_results.tsv")

        xgb_val = get_auprc(xgb_file)
        nn_val = get_auprc(nn_file)
        
        print(f"| {species} | {model} | {xgb_val} | {nn_val} |")
