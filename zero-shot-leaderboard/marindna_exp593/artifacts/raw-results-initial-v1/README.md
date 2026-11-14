# Initial augmentation results: visual summaries

Approved replacement figures for [PR23 initial results](https://github.com/plantcad/plantcad2/pull/23#issuecomment-6741139231).

- `final-scores`: final native scores, all readouts, both strand conventions.
- `readout-changes`: paired final-score differences, with incompatible AF comparisons omitted.
- `zero-shot-training`: fixed AR/FIM readouts across four checkpoints.
- `probe-training`: concatenated forward/RC frozen probes; AF uses variant-site pooling.

`data.json` retains full-precision plotted values and hashes of the published source artifacts. `source-comment.json` is the pre-edit raw-results snapshot. `validation.json` records score-rounding checks and preservation of every original raw table row and source link. No training, inference, probe fits or task scores were changed.

All-uppercase inputs are not a test of native softmask preservation. The metadata prompt also changes training-supported instruction wording and labels. DNA216B initialization has 216B prior DNA tokens and half the adaptation LR. AF suffix and site readouts have different scoring definitions, as marked in the figures.
