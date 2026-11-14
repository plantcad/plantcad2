# Raw annealing results: annotated heatmap

The visualization mirrors the published task-family tables in [Raw Results](https://github.com/plantcad/plantcad2/pull/23#issuecomment-6810986230) after all four final checkpoint evaluations completed.

- All120 cells preserve the native displayed values, including16 undefined entries; no scores are recomputed.
- Cell shading is linear within each metric, independently for zero-shot and frozen-probe panels.
- Bold text and thin borders highlight the highest model score for the same readout and metric; displayed ties are retained. Different readouts are not pooled to select a winner.
- Base-by-base means teacher forcing; whole motif means exhaustive decoding. Exhaustive core/non-core uses observed joint motif probability AUROC.
- Historical AR SV remains forward-only; the raw FIM SV extension and zero-shot AF definitions retain their existing caveats. The Summary Results post combines selected readouts by family; this visualization preserves each raw readout row.

`heatmap-data.json` includes exact plotted values and the checksum of the source comment snapshot. The image generator remains in the analysis checkout's ignored `scratch/exp593-raw-post-cleanup/plot.py`.
