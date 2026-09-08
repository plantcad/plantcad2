#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_XET_HIGH_PERFORMANCE=1
exec uv run --no-project --with pandas==2.2.3 --with polars==1.34.0 --with pyarrow==17.0.0 --with 'huggingface-hub>=1.5,<2' --with boto3 python "$SCRIPT_DIR/inspect_windows.py" "$@"
