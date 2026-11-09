"""Render fixed benchmark flanks through the unmodified training serialization.

The nested render closure is extracted from the pinned training source. Its
sampler is deliberately not called: that would choose a new crop/gap and change
the benchmark. Only supported closure arguments are supplied here.
"""

from __future__ import annotations

import ast
import hashlib
import json
from importlib.resources import files
from typing import Any

import numpy as np

from exp593_dna.pipeline import Span
from exp593_dna.templates import KEYS, phrase_catalog, render_metadata


def _render_closure(context: dict[str, Any]) -> Any:
    path = files("exp593_dna").joinpath("pipeline.py")
    source = path.read_bytes()
    manifest = json.loads(files("exp593_evals").joinpath("sources.json").read_text())
    expected = manifest["training"]["files"]["exp593_dna/pipeline.py"]
    if hashlib.sha256(source).hexdigest() != expected:
        raise ValueError("Training serialization source differs from its pin")
    functions = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "render"
    ]
    if len(functions) != 1:
        raise ValueError("Expected exactly one training render closure")
    namespace = {"Span": Span, **context}
    exec(  # noqa: S102 -- exact hash-verified training source
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["render"]


def fim_spans(
    left: str,
    right: str,
    answer: str,
    *,
    gap_length: int,
    metadata: dict[str, str] | None = None,
) -> tuple[Span, ...]:
    """Keep the benchmark geometry fixed and use one frozen supported template."""
    if gap_length < 1 or len(answer) not in {0, gap_length}:
        raise ValueError("Answer must be absent or have the caller-known gap length")
    if not left or not right:
        raise ValueError("Both target-free flanks are required")
    fields = {} if metadata is None else dict(metadata)
    if set(fields) - set(KEYS):
        raise ValueError("Metadata contains a field not rendered during training")
    if any(not isinstance(v, str) or not v or "\n" in v for v in fields.values()):
        raise ValueError("Metadata values must be nonempty single-line strings")
    catalog = phrase_catalog()
    if metadata is None:
        header, instruction = "", ""
        layout, labels, hint = (
            "compact",
            ("left", "right", "fill"),
            f"[gap:{gap_length}]\n",
        )
    else:
        # A fixed seed chooses the existing line-oriented metadata format. There is
        # no per-task prompt selection or test-score-dependent prompt tuning.
        seed = next(
            seed
            for seed in range(100)
            if render_metadata({}, np.random.default_rng(seed))[1] == "lines"
        )
        header, structure = render_metadata(fields, np.random.default_rng(seed))
        assert structure == "lines"
        case_description = "all nucleotides are represented in lowercase; repeat-mask information is not encoded by case."
        instruction = "Fill in the missing DNA using both flanks.\n"
        assert case_description in catalog["unmasked_lower"]
        assert instruction.strip() in catalog["fim_instruction"]
        header += case_description + "\n"
        layout, labels = "instruction_first", ("Left DNA", "Right flank", "Answer")
        hint = (
            f"Recover exactly {gap_length} {'base' if gap_length == 1 else 'bases'}.\n"
        )
        assert (
            labels[0] in catalog["left_label"]
            and labels[1] in catalog["right_label"]
            and labels[2] in catalog["answer_label"]
        )
    render = _render_closure(
        {
            "is_fim": True,
            "header": header,
            "instruction": instruction,
            "layout": layout,
            "left_label": labels[0],
            "right_label": labels[1],
            "answer_label": labels[2],
            "reverse_flanks": False,
            "separator": ": ",
            "gap_hint": hint,
        }
    )
    return render(left, right, answer, "")
