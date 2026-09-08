#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export HF_XET_HIGH_PERFORMANCE=1

# All environment work happens in the remote task, never on the submitter.
python -m pip install --no-cache-dir uv
python -m uv venv --system-site-packages /tmp/plantcad2-cw-venv
PYTHON=/tmp/plantcad2-cw-venv/bin/python
python -m uv pip install --python "$PYTHON" -r "${SCRIPT_DIR}/requirements.txt"
FLASH_WHEEL="$($PYTHON -c 'import sys, torch; assert torch.__version__.split("+")[0] == "2.7.0"; assert torch.version.cuda == "12.8"; tag=f"cp{sys.version_info.major}{sys.version_info.minor}"; abi=str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(); print(f"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1%2Bcu12torch2.7cxx11abi{abi}-{tag}-{tag}-linux_x86_64.whl")')"
python -m uv pip install --python "$PYTHON" --no-deps "$FLASH_WHEEL"
"$PYTHON" -c 'import json, platform, torch, transformers, flash_attn; print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"transformers":transformers.__version__,"flash_attn":flash_attn.__version__,"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, indent=2)); assert torch.cuda.device_count()==8; assert all("H100" in torch.cuda.get_device_name(i) for i in range(8))'
exec "$PYTHON" "$@"
