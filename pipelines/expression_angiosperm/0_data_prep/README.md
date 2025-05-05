## Task Overview

| Task Name                                                                                     | Problem Type     | Description                                                                 | Context Window Size |
|-----------------------------------------------------------------------------------------------|------------------|-----------------------------------------------------------------------------|----------------------|
| [max_exp_angiosperm](https://huggingface.co/datasets/kuleshov-group/PlantCAD2_tasks/tree/main/max_exp_angiosperm)       | Regression       | Predict the maximum expression across different libraries                  | 8192bp               |
| [high_off_exp_angiosperm](https://huggingface.co/datasets/kuleshov-group/PlantCAD2_tasks/tree/main/high_off_exp_angiosperm) | Classification   | Predict highly expressed genes (labeled as 1) versus non-expressed genes (labeled as 0) | 8192bp               |
| [on_off_exp_angiosperm](https://huggingface.co/datasets/kuleshov-group/PlantCAD2_tasks/tree/main/on_off_exp_angiosperm) | Classification   | Predict expressed genes (labeled as 1) versus non-expressed genes (labeled as 0)       | 8192bp               |

## Note
This dataset is adapted from the following paper: [link](https://www.pnas.org/doi/epub/10.1073/pnas.2319811121). The original paper trains the model on multiple species and tests the performance on hold-out species. In this dataset, we train the model only on the model plant Arabidopsis and test its performance on 14 other Angiosperm species, excluding Ppa and Cre.