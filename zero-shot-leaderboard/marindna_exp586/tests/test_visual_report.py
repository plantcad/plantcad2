import json
from pathlib import Path

from exp586_evals.visual_report import create_report


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_visual_report_writes_reproducible_png_and_svg_assets(tmp_path: Path):
    root = Path(__file__).parents[1]
    historical = root / "historical"
    full = json.loads(
        (historical / "exp472-lr5e4-full-coreweave-20260831.json").read_text()
    )
    source_rows = full["results"]["022t"]["rows"]
    rows = [
        {
            "key": row["key"],
            "category": row["category"],
            "task": row["task"],
            "natural": row["selected_value"],
            "bidirectional_poe": (
                None
                if row["category"] == "Structural variant"
                else row["selected_value"]
            ),
        }
        for row in source_rows
    ]
    group_values = full["groups"]["022t"]
    groups = dict(
        zip(
            (
                "Conservation",
                "Masked motif",
                "Core/non-core",
                "Structural variant",
                "Composite",
            ),
            group_values,
            strict=True,
        )
    )
    af = {
        "scores": {
            key: {"overall": {"spearman": value}}
            for key, value in (
                ("llr_acgt_avg", 0.15),
                ("llr_full_vocab_avg", 0.151),
                ("llr_target_poe", 0.16),
            )
        }
    }
    result = {
        "profiles": {
            "context-8192": {
                "tasks": {"rows": rows, "groups": {"natural": groups}},
                "allele_frequency": af,
            },
            "context-2048": {
                "tasks": {
                    "rows": [
                        row for row in rows if row["category"] != "Structural variant"
                    ],
                    "groups": {"natural": groups},
                },
                "allele_frequency": af,
            },
        }
    }
    probe = {
        "metrics": {
            key: {"overall": {"probe": {"value": 0.18, "ci95": [0.17, 0.19]}}}
            for key in (
                "whole_window/strand_average",
                "whole_window/directional",
                "variant_token/strand_average",
                "variant_token/directional",
            )
        }
    }
    conditions = {}
    for name in ("pretrained", "random", "language_base"):
        directory = tmp_path / name
        write(directory / "result.json", result)
        for profile in ("context-8192", "context-2048"):
            write(directory / profile / "probes/result.json", probe)
        conditions[name] = directory
    method = {
        "intrinsic": {
            "context_sweep": {
                str(context): {
                    "left_test": {"nll": 1.1},
                    "right_test": {"nll": 1.08},
                    "poe_equal_test": {"nll": 1.0},
                }
                for context in (128, 512, 2048, 4095)
            },
            "prompt_methods_2048": {
                "rotated": {"nll": 1.09},
                "sandwich": {"nll": 1.2},
            },
        }
    }
    pretrained_method = tmp_path / "pretrained-method.json"
    write(pretrained_method, method)
    loss_history = {
        "schema": 1,
        "runs": {
            key: {
                "training": [
                    {"input_tokens": 6_000_000, "loss": start},
                    {"input_tokens": 1_000_000_000, "loss": 1.3},
                    {"input_tokens": 6_000_000_000, "loss": 1.15},
                    {"input_tokens": 216_000_000_000, "loss": final},
                ],
                "evaluation": [
                    {"input_tokens": 1_000_000_000, "loss": 1.3},
                    {"input_tokens": 6_000_000_000, "loss": 1.15},
                    {"input_tokens": 216_000_000_000, "loss": final},
                ],
            }
            for key, start, final in (
                ("d768-scratch-lrm1", 2.1, 0.95),
                ("d1536-pretrained", 4.7, 0.948),
            )
        },
    }
    write(tmp_path / "training-loss-history.json", loss_history)
    catalog = tmp_path / "catalog.json"
    write(
        catalog,
        {
            "schema": 1,
            "models": [
                {
                    "key": "d768-scratch-lrm1",
                    "label": "MarinMoE 1.5B-60M 0.22T (scratch, 1× LR)",
                    "result": "pretrained",
                    "method": "pretrained-method.json",
                    "color": "#0F766E",
                    "views": [
                        "overview",
                        "overview_detail",
                        "task_natural",
                        "task_two_sided",
                        "context",
                        "af",
                        "probes",
                        "methods",
                        "loss",
                    ],
                },
                {
                    "key": "d1536-pretrained",
                    "label": "MarinMoE 11.5B-481M 0.01T (pretrained, 1× LR)",
                    "result": "random",
                    "color": "#2563EB",
                    "views": [
                        "overview",
                        "overview_detail",
                        "task_natural",
                        "task_two_sided",
                        "af",
                        "probes",
                        "loss",
                    ],
                },
                {
                    "key": "d1536-language-base",
                    "label": "MarinMoE 11.5B-481M 0.00T (pretrained control)",
                    "result": "language_base",
                    "color": "#7C3AED",
                    "views": ["overview", "task_natural", "af", "probes"],
                },
            ],
        },
    )
    output = tmp_path / "assets"
    create_report(catalog, historical, output)
    names = (
        "exp586-model-family-overview",
        "exp586-training-loss",
        "exp586-task-heatmap",
        "exp586-task-context-deltas",
        "exp586-af-readouts",
        "exp586-probe-readouts",
        "exp586-intrinsic-methods",
        "exp586-prompt-methods",
    )
    for name in names:
        for extension in ("png", "svg"):
            assert (output / f"{name}.{extension}").stat().st_size > 1_000

    for directory in conditions.values():
        for profile in ("context-8192", "context-2048"):
            (directory / profile / "probes/result.json").unlink()
    without_probes = tmp_path / "assets-without-probes"
    create_report(catalog, historical, without_probes)
    assert not (without_probes / "exp586-probe-readouts.png").exists()
    assert (without_probes / "exp586-model-family-overview.png").exists()
