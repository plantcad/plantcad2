# exp586 Hugging Face releases and archive

Published a dedicated [216B-token language-pretrained model](https://huggingface.co/plantcad/marin-dna-exp586-1.5B-60M-216B) and added an exact [1T-token export copy](https://huggingface.co/marin-dna/marin-dna-exp586/tree/main/pretrained-lrm1-seed0-1t) to the research archive.
The original standalone [1T model](https://huggingface.co/plantcad/marin-dna-exp586-1.5B-60M) remains unchanged.
The model names in [PlantCAD2 PR #22's OP table](https://github.com/plantcad/plantcad2/pull/22) now link to the corresponding Hugging Face exports, including the existing scratch endpoint.
All W&B links, results links, model numbers and other OP text were preserved.

## Sources and verification

- The 216B source is `marin-dna/marin-dna-exp586` at `4b8537591261ad14204b28a9c830fa7c6e3f7fbd`, under `pretrained-lrm1-seed0/`.
- The 1T source is `plantcad/marin-dna-exp586-1.5B-60M` at `bd6c44fdc093610f20a892796624de9d96c7ca80`.
- Hugging Face Hub 1.33.0 copied the files between repositories, using server-side LFS copies.
- All ten 216B export files and twelve 1T source files match their pinned sources: LFS SHA-256 and size for large files, Git blob identity and size for regular files.
- Hugging Face appended the required destination-path rule for each copied `tokenizer.json` in `.gitattributes`; those exact additions were verified separately.
- The four existing archive exports and original manifest entries remain unchanged.
- Both new exports are publicly readable, with the expected checkpoint IDs and training-token counts.
- The updated OP was re-fetched and checked against the prepared replacement, including its rendered links.

The copy receipts record destination revisions and every source/destination file identity.
The model cards and updated archive manifest are saved alongside them.
`op-before.md`, `op-after.md` and `op-publication-receipt.json` preserve the exact GitHub change.

`reorganize_hf.py` is the one-off execution script, requiring `--execute` and pinned source revisions.
It refuses an advanced archive head rather than reapplying changes to different contents.
