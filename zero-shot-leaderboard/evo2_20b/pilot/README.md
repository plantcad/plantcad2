# Small causal-evaluation pilot

This pilot runs 64 deterministic examples by default from each of seven representative
PlantCAD2 tasks: two conservation tasks, two motif-recovery tasks, two core/non-core
tasks, and structural-variant effect prediction. Binary tasks are sampled evenly by
label. Conservation, motif, and core/non-core scores use both real strand orientations;
the structural-variant task follows the leaderboard's single-context protocol.

`convert_exp472.py` reconstructs the model from the exact exp472 recipe in a MarinDNA
checkout, converts a local Levanter checkpoint with Levanter's exporter, and can upload
the result to a subdirectory of a Hugging Face model repository.

On a Lambda Labs image with CUDA PyTorch already installed:

```bash
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python -r zero-shot-leaderboard/evo2_20b/pilot/requirements.txt

export PYTHON="$PWD/.venv/bin/python"
export MODEL=/path/to/exported/hf/checkpoint
export OUTPUT_DIR="$PWD/scratch/exp472-pilot-results"
zero-shot-leaderboard/evo2_20b/pilot/run_pilot.sh
```

Set `SAMPLES`, `BATCH_SIZE`, or `SV_BATCH_SIZE` to adjust the pilot. The TSV samples and
their manifest are retained separately from model outputs so repeated model evaluations
use identical examples.

`run_sampled_leaderboard.sh` covers all 20 leaderboard rows. It defaults to 2,800
unstratified random examples per row with the same seed 0, preserving class prevalence
while targeting about two hours on a single H100 for the exp472 Qwen3 model. Its
200,000-row streaming shuffle buffer covers every current split, avoiding source-order
bias before the sample is taken.
