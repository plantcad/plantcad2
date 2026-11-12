# MarinMoE 665B summary results

Figure assets and score data for the Summary Results comment on [PlantCAD2 PR #23](https://github.com/plantcad/plantcad2/pull/23).

- Primary: AR forward/RC conservation, whole-motif FIM forward/RC motif recovery and core/non-core, and historical forward-only AR SV.
- Alternative: teacher-forced FIM motif/core readouts, with other native scores unchanged.
- All #593 evaluations use `fast-fa3`. Throughput placements reuse [#586 measurements](https://github.com/Open-Athena/marin-dna/pull/590#issuecomment-6710081437); they are projections, not end-to-end evaluation timings.
- Ranks are recomputed jointly from raw scores using the PlantCAD2 leaderboard function: 11 baseline models plus the dense model, 216B MoE, and two 665B readouts. Weighted family weights: 15/40/40/5%; family-balanced ranks: 25% each.
- Matrix cells contain native raw scores and use a fixed 0–1 color scale. The smaller rank panel contains only the 665B model.
- Species membership was checked against all 44 original training metadata shards (2,638,656 windows), retaining the original 65 species. It describes MarinDNA's DNA corpus, not baseline training data or benchmark-locus overlap.

PNG and editable SVG versions are in `figures/`. The JSON/CSV files preserve plotted values, reference ranks and source checksums. Local `posted: false` fields in generation records refer to the unpublished generation step.
