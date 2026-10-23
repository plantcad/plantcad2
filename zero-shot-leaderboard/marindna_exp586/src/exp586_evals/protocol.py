"""Load the existing PR20 evaluation modules, in place or from the built wheel."""

import importlib
import sys
from functools import cache
from pathlib import Path
from types import SimpleNamespace

REFERENCE_COMMIT = "475fba58f81b79efa234d0018eb0776d1a960bfd"


@cache
def original() -> SimpleNamespace:
    root = Path(__file__).parent / "exp472"
    if not root.is_dir():
        root = Path(__file__).resolve().parents[3] / "evo2_20b"
    for path in [root / "pilot", root / "pilot/sensitivity", root / "pilot/coreweave"]:
        sys.path.insert(0, str(path))
    return SimpleNamespace(
        **{
            key: importlib.import_module(module)
            for key, module in {
                "tasks": "run_condition",
                "inputs": "inputs",
                "partition": "partition",
                "aggregate": "aggregate",
                "compare": "compare",
                "af": "allele_frequency.scoring",
                "af_aggregate": "allele_frequency.aggregate",
                "probe": "allele_frequency_probe.probe",
            }.items()
        }
    )
