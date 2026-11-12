"""Exercise the actual remote commands before allocating their GPU workers."""

import runpy
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "worker",
        "full_bootstrap",
        "analysis_bootstrap",
        "analysis",
        "stage",
        "gate",
        "native",
    ],
)
def test_remote_command_imports_and_argument_parsing(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit) as result:
        runpy.run_module("exp593_anneal_evals." + module, run_name="__main__")
    assert result.value.code == 0
