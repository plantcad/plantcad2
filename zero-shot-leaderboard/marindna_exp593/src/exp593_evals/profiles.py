"""Explicit frozen deployment profiles; select environment before importing Torch."""

import os
import sys


def profile(name: str) -> dict:
    if name not in {"fast-fa3", "vllm-fa3", "torch-fa3"}:
        raise ValueError(f"Unknown comparison profile: {name}")
    fast = name == "fast-fa3"
    return {
        "name": name,
        "attention_provider": "flash_attn" if name == "torch-fa3" else "vllm",
        "flash_attention_version": 3,
        "moe_backend": "torch" if name == "torch-fa3" else "vllm",
        "fullgraph": True,
        "freezing": fast,
        "compile": {"fullgraph": True, "dynamic": False, "mode": "reduce-overhead"}
        if fast
        else {"fullgraph": True},
    }


def configure(name: str) -> dict:
    value = profile(name)
    current = os.environ.get("TORCHINDUCTOR_FREEZING") == "1"
    if "torch" in sys.modules and current != value["freezing"]:
        raise RuntimeError("Use a fresh process to change compiler freezing")
    if value["freezing"]:
        os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    else:
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)
    return value
