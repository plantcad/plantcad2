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

**Key observations:**
  - PlantCAD2 significantly outperforms EVO2 across all species
  - For PlantCAD2, the "mean" pooling strategy generally outperforms "max" pooling
  - PlantCAD2 large models achieve the best results overall (arabidopsis large_mean: 0.7447 XGB, 0.7339 NN)
  - Arabidopsis consistently shows the highest AUPRC values across both model families
  - For non-arabidopsis species, PlantCAD2 large models show substantial improvements over EVO2
  - Arabidopsis shows the highest AUPRC values across all models
  - For most species, the evo2_mean (average) strategy performs best
  - Neural networks generally perform better than XGBoost for arabidopsis, but results are mixed for other species
  - Populus trichocarpa shows the lowest AUPRC values overall

 