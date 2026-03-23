
import pandas as pd
import io

data = """
| Species             | Model            | AUPRC (XGBoost) | AUPRC (Neural Network) |
|---------------------|------------------|-----------------|------------------------|
| arabidopsis         | plantcad2_small_mean  | 0.7086          | 0.6837                 |
| arabidopsis         | plantcad2_small_max   | 0.6235          | 0.5859                 |
| arabidopsis         | plantcad2_medium_mean | 0.6854          | 0.6574                 |
| arabidopsis         | plantcad2_medium_max  | 0.6269          | 0.5750                 |
| arabidopsis         | plantcad2_large_mean  | 0.7447          | 0.7339                 |
| arabidopsis         | plantcad2_large_max   | 0.6609          | 0.6339                 |
| arabidopsis         | evo2_forward     | 0.4840          | 0.5094                 |
| arabidopsis         | evo2_reverse     | 0.4813          | 0.5077                 |
| arabidopsis         | evo2_mean        | 0.5690          | 0.5923                 |
| arabidopsis         | evo2_concatenate | 0.5437          | 0.5347                 |
| setaria_viridis     | plantcad2_small_mean | 0.5338          | 0.5919                 |
| setaria_viridis     | plantcad2_small_max  | 0.5019          | 0.4712                 |
| setaria_viridis     | plantcad2_medium_mean | 0.3531          | 0.5365                 |
| setaria_viridis     | plantcad2_medium_max  | 0.4229          | 0.4576                 |
| setaria_viridis     | plantcad2_large_mean  | 0.5122          | 0.5729                 |
| setaria_viridis     | plantcad2_large_max   | 0.4952          | 0.4942                 |
| setaria_viridis     | evo2_forward     | 0.3353          | 0.3376                 |
| setaria_viridis     | evo2_reverse     | 0.3321          | 0.2851                 |
| setaria_viridis     | evo2_mean        | 0.4133          | 0.3678                 |
| setaria_viridis     | evo2_concatenate | 0.3987          | 0.3107                 |
| eutrema_salsugineum | plantcad2_small_mean | 0.4564          | 0.4472                 |
| eutrema_salsugineum | plantcad2_small_max  | 0.3940          | 0.3664                 |
| eutrema_salsugineum | plantcad2_medium_mean | 0.4458          | 0.4227                 |
| eutrema_salsugineum | plantcad2_medium_max  | 0.4018          | 0.3575                 |
| eutrema_salsugineum | plantcad2_large_mean  | 0.4975          | 0.4964                 |
| eutrema_salsugineum | plantcad2_large_max   | 0.4274          | 0.4040                 |
| eutrema_salsugineum | evo2_forward     | 0.2742          | 0.2998                 |
| eutrema_salsugineum | evo2_reverse     | 0.2806          | 0.2906                 |
| eutrema_salsugineum | evo2_mean        | 0.3443          | 0.3464                 |
| eutrema_salsugineum | evo2_concatenate | 0.3173          | 0.3086                 |
| populus_trichocarpa | plantcad2_small_mean | 0.4671          | 0.4625                 |
| populus_trichocarpa | plantcad2_small_max  | 0.4044          | 0.3675                 |
| populus_trichocarpa | plantcad2_medium_mean | 0.4513          | 0.5018                 |
| populus_trichocarpa | plantcad2_medium_max  | 0.4152          | 0.3887                 |
| populus_trichocarpa | plantcad2_large_mean  | 0.5093          | 0.4909                 |
| populus_trichocarpa | plantcad2_large_max   | 0.4359          | 0.3808                 |
| populus_trichocarpa | evo2_forward     | 0.2420          | 0.2646                 |
| populus_trichocarpa | evo2_reverse     | 0.2379          | 0.2397                 |
| populus_trichocarpa | evo2_mean        | 0.3104          | 0.3171                 |
| populus_trichocarpa | evo2_concatenate | 0.2780          | 0.2655                 |
| brachypodium_distachyon | plantcad2_small_mean | 0.3052 | 0.3891 |
| brachypodium_distachyon | plantcad2_small_max | 0.3125 | 0.3146 |
| brachypodium_distachyon | plantcad2_medium_mean | 0.2376 | 0.4179 |
| brachypodium_distachyon | plantcad2_medium_max | 0.2505 | 0.3110 |
| brachypodium_distachyon | plantcad2_large_mean | 0.2965 | 0.3337 |
| brachypodium_distachyon | plantcad2_large_max | 0.2937 | 0.2929 |
| brachypodium_distachyon | evo2_forward | 0.1777 | 0.1964 |
| brachypodium_distachyon | evo2_reverse | 0.1787 | 0.1474 |
| brachypodium_distachyon | evo2_average | 0.2263 | 0.1918 |
| brachypodium_distachyon | evo2_concatenate | 0.2155 | 0.1669 |
| glycine_max | plantcad2_small_mean | 0.4026 | 0.4002 |
| glycine_max | plantcad2_small_max | 0.3196 | 0.2799 |
| glycine_max | plantcad2_medium_mean | 0.3486 | 0.3947 |
| glycine_max | plantcad2_medium_max | 0.3063 | 0.2717 |
| glycine_max | plantcad2_large_mean | 0.4202 | 0.4427 |
| glycine_max | plantcad2_large_max | 0.3330 | 0.2738 |
| glycine_max | evo2_forward | 0.1696 | 0.1780 |
| glycine_max | evo2_reverse | 0.1641 | 0.1618 |
| glycine_max | evo2_average | 0.2296 | 0.2167 |
| glycine_max | evo2_concatenate | 0.2016 | 0.1689 |
| hordeum_vulgare | plantcad2_small_mean | 0.1633 | 0.1797 |
| hordeum_vulgare | plantcad2_small_max | 0.2185 | 0.1979 |
| hordeum_vulgare | plantcad2_medium_mean | 0.1316 | 0.3063 |
| hordeum_vulgare | plantcad2_medium_max | 0.1458 | 0.2241 |
| hordeum_vulgare | plantcad2_large_mean | 0.1346 | 0.1198 |
| hordeum_vulgare | plantcad2_large_max | 0.1818 | 0.2095 |
| hordeum_vulgare | evo2_forward | 0.0981 | 0.0572 |
| hordeum_vulgare | evo2_reverse | 0.1046 | 0.0395 |
| hordeum_vulgare | evo2_average | 0.1446 | 0.0470 |
| hordeum_vulgare | evo2_concatenate | 0.1395 | 0.0375 |
| oryza_sativa | plantcad2_small_mean | 0.4548 | 0.4721 |
| oryza_sativa | plantcad2_small_max | 0.3823 | 0.3292 |
| oryza_sativa | plantcad2_medium_mean | 0.2400 | 0.4551 |
| oryza_sativa | plantcad2_medium_max | 0.3104 | 0.3232 |
| oryza_sativa | plantcad2_large_mean | 0.4278 | 0.5456 |
| oryza_sativa | plantcad2_large_max | 0.3434 | 0.3863 |
| oryza_sativa | evo2_forward | 0.2291 | 0.2401 |
| oryza_sativa | evo2_reverse | 0.2278 | 0.2199 |
| oryza_sativa | evo2_average | 0.2911 | 0.3004 |
| oryza_sativa | evo2_concatenate | 0.2706 | 0.2467 |
| phaseolus_vulgaris | plantcad2_small_mean | 0.4196 | 0.4031 |
| phaseolus_vulgaris | plantcad2_small_max | 0.3536 | 0.3219 |
| phaseolus_vulgaris | plantcad2_medium_mean | 0.3713 | 0.4468 |
| phaseolus_vulgaris | plantcad2_medium_max | 0.3354 | 0.3085 |
| phaseolus_vulgaris | plantcad2_large_mean | 0.4382 | 0.4185 |
| phaseolus_vulgaris | plantcad2_large_max | 0.3634 | 0.3261 |
| phaseolus_vulgaris | evo2_forward | 0.2137 | 0.2091 |
| phaseolus_vulgaris | evo2_reverse | 0.2020 | 0.1766 |
| phaseolus_vulgaris | evo2_average | 0.2798 | 0.2277 |
| phaseolus_vulgaris | evo2_concatenate | 0.2502 | 0.1902 |
| sorghum_bicolor | plantcad2_small_mean | 0.3993 | 0.4498 |
| sorghum_bicolor | plantcad2_small_max | 0.4033 | 0.3878 |
| sorghum_bicolor | plantcad2_medium_mean | 0.2433 | 0.4676 |
| sorghum_bicolor | plantcad2_medium_max | 0.3181 | 0.3496 |
| sorghum_bicolor | plantcad2_large_mean | 0.3799 | 0.4395 |
| sorghum_bicolor | plantcad2_large_max | 0.3561 | 0.3674 |
| sorghum_bicolor | evo2_forward | 0.2190 | 0.2011 |
| sorghum_bicolor | evo2_reverse | 0.2252 | 0.1503 |
| sorghum_bicolor | evo2_average | 0.2866 | 0.1991 |
| sorghum_bicolor | evo2_concatenate | 0.2711 | 0.1624 |
| zea_mays | plantcad2_small_mean | 0.2735 | 0.2053 |
| zea_mays | plantcad2_small_max | 0.3692 | 0.3172 |
| zea_mays | plantcad2_medium_mean | 0.2580 | 0.3919 |
| zea_mays | plantcad2_medium_max | 0.3030 | 0.3500 |
| zea_mays | plantcad2_large_mean | 0.3341 | 0.2500 |
| zea_mays | plantcad2_large_max | 0.3290 | 0.3090 |
| zea_mays | evo2_forward | 0.2026 | 0.1443 |
| zea_mays | evo2_reverse | 0.2043 | 0.1087 |
| zea_mays | evo2_average | 0.2821 | 0.1219 |
| zea_mays | evo2_concatenate | 0.2632 | 0.1067 |
"""

# Read the data
lines = [line.strip() for line in data.strip().split('\n')]
lines = [l for l in lines if l and l.startswith('|')]

# Headers
headers = [h.strip() for h in lines[0].split('|') if h.strip()]
# data starts at index 2 (index 1 is separator)
rows = lines[2:]

species_data = {}

for row in rows:
    parts = [p.strip() for p in row.split('|') if p.strip()]
    if not parts: continue
    
    species = parts[0]
    model = parts[1]
    xgb = float(parts[2])
    nn = float(parts[3])
    
    if species == 'arabidopsis':
        continue
    
    if species not in species_data:
        species_data[species] = {}
    
    species_data[species][model] = {'xgb': xgb, 'nn': nn}

print(f"| Species | PlantCAD2 Small Mean (XGB) | PlantCAD2 Small Mean (NN) | PlantCAD2 Medium Mean (XGB) | PlantCAD2 Medium Mean (NN) | PlantCAD2 Large Mean (XGB) | PlantCAD2 Large Mean (NN) | EVO2 Max (XGB) | EVO2 Max (NN) |")
print(f"|---|---|---|---|---|---|---|---|---|")

for species, models in species_data.items():
    # PlantCAD2 means
    pc_small_mean_xgb = models.get('plantcad2_small_mean', {}).get('xgb', 'N/A')
    pc_small_mean_nn = models.get('plantcad2_small_mean', {}).get('nn', 'N/A')
    
    pc_medium_mean_xgb = models.get('plantcad2_medium_mean', {}).get('xgb', 'N/A')
    pc_medium_mean_nn = models.get('plantcad2_medium_mean', {}).get('nn', 'N/A')
    
    pc_large_mean_xgb = models.get('plantcad2_large_mean', {}).get('xgb', 'N/A')
    pc_large_mean_nn = models.get('plantcad2_large_mean', {}).get('nn', 'N/A')
    
    # Evo2 Max
    evo2_xgb_max = -1
    evo2_nn_max = -1
    
    found_evo2 = False
    for m, vals in models.items():
        if m.startswith('evo2_'):
            found_evo2 = True
            if vals['xgb'] > evo2_xgb_max:
                evo2_xgb_max = vals['xgb']
            if vals['nn'] > evo2_nn_max:
                evo2_nn_max = vals['nn']
    
    if not found_evo2:
        evo2_xgb_max = 'N/A'
        evo2_nn_max = 'N/A'
    
    print(f"| {species} | {pc_small_mean_xgb} | {pc_small_mean_nn} | {pc_medium_mean_xgb} | {pc_medium_mean_nn} | {pc_large_mean_xgb} | {pc_large_mean_nn} | {evo2_xgb_max} | {evo2_nn_max} |")

# Calculate averages
metrics = {
    'small_xgb': [], 'small_nn': [],
    'mid_xgb': [], 'mid_nn': [],
    'large_xgb': [], 'large_nn': []
}

for species, models in species_data.items():
    # Evo2 Max
    evo2_xgb_max = -1
    evo2_nn_max = -1
    found_evo2 = False
    for m, vals in models.items():
        if m.startswith('evo2_'):
            found_evo2 = True
            if vals['xgb'] > evo2_xgb_max:
                evo2_xgb_max = vals['xgb']
            if vals['nn'] > evo2_nn_max:
                evo2_nn_max = vals['nn']
    
    if not found_evo2: continue

    # PlantCAD2 means
    if 'plantcad2_small_mean' in models:
        metrics['small_xgb'].append(models['plantcad2_small_mean']['xgb'] - evo2_xgb_max)
        metrics['small_nn'].append(models['plantcad2_small_mean']['nn'] - evo2_nn_max)
    
    if 'plantcad2_medium_mean' in models:
        metrics['mid_xgb'].append(models['plantcad2_medium_mean']['xgb'] - evo2_xgb_max)
        metrics['mid_nn'].append(models['plantcad2_medium_mean']['nn'] - evo2_nn_max)
        
    if 'plantcad2_large_mean' in models:
        metrics['large_xgb'].append(models['plantcad2_large_mean']['xgb'] - evo2_xgb_max)
        metrics['large_nn'].append(models['plantcad2_large_mean']['nn'] - evo2_nn_max)

print("\n\n### Average Improvement over Evo2 (Max) [Excluding Arabidopsis]")
print("| Model (Mean Pooling) | Average Improvement (XGB) | Average Improvement (NN) |")
print("|---|---|---|")
print(f"| PlantCAD2 Small | {sum(metrics['small_xgb'])/len(metrics['small_xgb']):.4f} | {sum(metrics['small_nn'])/len(metrics['small_nn']):.4f} |")
print(f"| PlantCAD2 Medium | {sum(metrics['mid_xgb'])/len(metrics['mid_xgb']):.4f} | {sum(metrics['mid_nn'])/len(metrics['mid_nn']):.4f} |")
print(f"| PlantCAD2 Large | {sum(metrics['large_xgb'])/len(metrics['large_xgb']):.4f} | {sum(metrics['large_nn'])/len(metrics['large_nn']):.4f} |")
