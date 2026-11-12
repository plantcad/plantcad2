"""Create the publication figures used in the exp586 PR results report."""

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from exp586_evals.historical import verify_exp582_results

COLORS = {
    "dense": "#64748B",
    "published": "#94A3B8",
    "poe": "#6D5BD0",
}


@dataclasses.dataclass(frozen=True)
class Model:
    key: str
    label: str
    directory: Path
    color: str
    views: frozenset[str]
    result: dict
    method: dict | None


def load_catalog(path: Path) -> list[Model]:
    payload = load(path)
    if payload.get("schema") != 1 or not payload.get("models"):
        raise ValueError("The model catalog must use schema 1 and contain models")
    models = []
    for item in payload["models"]:
        directory = (path.parent / item["result"]).resolve()
        method_path = (
            (path.parent / item["method"]).resolve() if item.get("method") else None
        )
        views = frozenset(item["views"])
        if not (directory / "result.json").is_file():
            raise ValueError(f"Missing result for {item['key']}: {directory}")
        if "methods" in views and (method_path is None or not method_path.is_file()):
            raise ValueError(f"Missing method report for {item['key']}")
        models.append(
            Model(
                key=item["key"],
                label=item["label"],
                directory=directory,
                color=item["color"],
                views=views,
                result=load(directory / "result.json"),
                method=load(method_path) if method_path is not None else None,
            )
        )
    if len({model.key for model in models}) != len(models):
        raise ValueError("Model catalog keys must be unique")
    if len({model.label for model in models}) != len(models):
        raise ValueError("Model catalog labels must be unique")
    return models


def selected(models: list[Model], view: str) -> list[Model]:
    values = [model for model in models if view in model.views]
    if not values:
        raise ValueError(f"Model catalog selects no models for {view}")
    return values


def compact_label(label: str) -> str:
    """Shorten label punctuation while preserving all model identity fields."""
    if label.startswith("MarinMoE "):
        return label.replace(", 1× LR)", " · 1×)").replace(", 2× LR)", " · 2×)")
    return label


def display_label(label: str) -> str:
    """Wrap MarinMoE details onto a second line without changing the label."""
    return compact_label(label).replace(" (", "\n(", 1)


def condition_label(model: Model) -> str:
    """Use a short condition label once the common model identity is established."""
    if "scratch" in model.key:
        return "Scratch · 1× LR"
    if "pretrained" in model.key:
        return "Language-pretrained · 1× LR"
    return compact_label(model.label)


CATEGORIES = (
    "Conservation",
    "Masked motif",
    "Core/non-core",
    "Structural variant",
    "Composite",
)
PUBLISHED = {
    "PlantCAD2-Small": ("PlantCAD2-S", "PlantCAD2-Small"),
    "PlantCAD2-Large": ("PlantCAD2-L", "PlantCAD2-Large"),
    "PlantCAD2.5-Large": ("PlantCAD2.5-L", "PlantCAD2.5-Large"),
    "Evo 2 20B": ("evo2_20b", "evo2_20b"),
}
DENSE_LABELS = {
    "022t": "Dense 0.22T",
    "039t": "Dense 0.39T",
    "056t": "Dense 0.56T",
}
PUBLISHED_TASK_IDS = {
    "conservation_andropogoneae": "cons_andropogoneae",
    "conservation_poaceae_non_tis": "cons_poaceae_nontis",
    "conservation_poaceae_tis": "cons_poaceae_tis",
    "motif_maize_tis": "motif_maize_start_sites",
    "motif_maize_tts": "motif_maize_stop_sites",
    "motif_maize_donor": "motif_maize_donor",
    "motif_maize_acceptor": "motif_maize_acceptor",
    "motif_tomato_tis": "motif_tomato_start_sites",
    "motif_tomato_tts": "motif_tomato_stop_sites",
    "motif_tomato_donor": "motif_tomato_donor",
    "motif_tomato_acceptor": "motif_tomato_acceptor",
    "core_maize_tis": "coreclass_maize_tis",
    "core_maize_tts": "coreclass_maize_tts",
    "core_maize_donor": "coreclass_maize_donor",
    "core_maize_acceptor": "coreclass_maize_acceptor",
    "core_tomato_tis": "coreclass_tomato_tis",
    "core_tomato_tts": "coreclass_tomato_tts",
    "core_tomato_donor": "coreclass_tomato_donor",
    "core_tomato_acceptor": "coreclass_tomato_acceptor",
    "sv_impact": "sv_impact_auprc",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        target = output / f"{name}.{extension}"
        figure.savefig(
            target,
            dpi=220 if extension == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        if extension == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text().splitlines())
                + "\n"
            )
    plt.close(figure)


def new_profile(result: dict, profile: str = "context-8192") -> dict:
    return result["profiles"][profile]


def overview_series(
    models: list[Model],
    historical: dict,
) -> dict[str, list[float]]:
    series = {
        historical["labels"][rung]: historical["groups"][rung]
        for rung in ("022t", "056t")
    }
    for label, (_, group_key) in PUBLISHED.items():
        series[label] = historical["published_groups"][group_key]
    for model in selected(models, "overview"):
        groups = new_profile(model.result)["tasks"]["groups"]["natural"]
        series[model.label] = [groups[name] for name in CATEGORIES]
    return series


def model_overview(
    models: list[Model],
    historical: dict,
    output: Path,
) -> None:
    series = overview_series(models, historical)
    labels = list(series)
    values = np.asarray([series[label] for label in labels])
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.4),
        gridspec_kw={"width_ratios": (1.05, 1.7)},
    )
    composite = values[:, -1]
    order = np.argsort(composite)
    new_colors = {model.label: model.color for model in models}
    new_short_labels = {model.label: condition_label(model) for model in models}
    colors = [
        new_colors.get(
            label,
            COLORS["dense"] if label.startswith("MarinDNA") else COLORS["published"],
        )
        for label in labels
    ]
    axes[0].barh(
        np.arange(len(labels)),
        composite[order],
        color=np.asarray(colors)[order],
        edgecolor="white",
    )
    axes[0].set_yticks(
        np.arange(len(labels)),
        np.asarray([display_label(label) for label in labels])[order],
    )
    axes[0].set_xlabel("20-task composite")
    axes[0].set_xlim(max(0, composite.min() - 0.04), min(1, composite.max() + 0.03))
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].set_axisbelow(True)
    for y, value in enumerate(composite[order]):
        axes[0].text(value + 0.002, y, f"{value:.3f}", va="center", fontsize=8)

    selected = [
        "MarinDNA 1B 0.22T",
        "MarinDNA 1B 0.56T",
        "PlantCAD2-Small",
        "PlantCAD2-Large",
        "PlantCAD2.5-Large",
        "Evo 2 20B",
    ] + [model.label for model in models if "overview_detail" in model.views]
    x = np.arange(4)
    selected_values = []
    for label in selected:
        row = series[label][:-1]
        selected_values.extend(row)
        emphasized = label in new_colors
        axes[1].plot(
            x,
            row,
            marker="o",
            linewidth=2.6 if emphasized else 1.15,
            alpha=1 if emphasized else 0.58,
            label=new_short_labels.get(label, display_label(label)),
            color=new_colors.get(label),
        )
    axes[1].set_xticks(x, CATEGORIES[:-1], rotation=18, ha="right")
    axes[1].set_ylabel("Mean task score")
    axes[1].set_ylim(max(0, min(selected_values) - 0.04), 0.9)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    figure.suptitle("Best-settings model comparison", fontsize=15, weight="bold")
    figure.text(
        0.5,
        0.01,
        "All points use the original full-split PlantCAD2 scoring definition with 8,192-base inputs.",
        ha="center",
        color="#475569",
        fontsize=9,
    )
    save(figure, output, "exp586-model-family-overview")


def historical_task_columns(
    historical: dict,
) -> tuple[list[str], dict[str, list[float]]]:
    dense_rows = historical["results"]["022t"]["rows"]
    task_labels = [f"{row['category']} · {row['task']}" for row in dense_rows]
    columns = {
        historical["labels"][rung]: [
            row["selected_value"] for row in historical["results"][rung]["rows"]
        ]
        for rung in ("022t", "039t", "056t")
    }
    for label, (row_key, _) in PUBLISHED.items():
        rows = [row for row in historical["published_rows"] if row["model"] == row_key]
        by_task = {row["task_id"]: float(row["value"]) for row in rows}
        expected = set(PUBLISHED_TASK_IDS.values())
        if len(rows) != len(task_labels) or set(by_task) != expected:
            raise ValueError(f"Published task inventory differs for {label}")
        columns[label] = [by_task[PUBLISHED_TASK_IDS[row["key"]]] for row in dense_rows]
    return task_labels, columns


def task_heatmap(
    models: list[Model],
    historical: dict,
    output: Path,
) -> None:
    labels, historical_columns = historical_task_columns(historical)
    columns = {
        "Dense 0.56T": historical_columns[historical["labels"]["056t"]],
        "PC2 Small": historical_columns["PlantCAD2-Small"],
        "PC2 Large": historical_columns["PlantCAD2-Large"],
        "PC2.5 Large": historical_columns["PlantCAD2.5-Large"],
        "Evo 2 20B": historical_columns["Evo 2 20B"],
    }
    for model in models:
        methods = []
        if "task_natural" in model.views:
            methods.append(("Natural", "natural"))
        if "task_two_sided" in model.views:
            methods.append(("two-sided", "bidirectional_poe"))
        if not methods:
            continue
        rows = new_profile(model.result)["tasks"]["rows"]
        if [row["key"] for row in rows] != [
            row["key"] for row in historical["results"]["022t"]["rows"]
        ]:
            raise ValueError(f"New task inventory differs for {model.label}")
        for label, field in methods:
            columns[f"{condition_label(model)}\n{label}"] = [row[field] for row in rows]
    names = list(columns)
    values = np.asarray(
        [
            [np.nan if value is None else value for value in columns[name]]
            for name in names
        ]
    ).T
    figure, axis = plt.subplots(figsize=(max(14.0, 1.75 * len(names)), 11.5))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="viridis",
        vmin=min(0.45, float(np.nanmin(values))),
        vmax=0.9,
    )
    axis.set_xticks(np.arange(len(names)), names, rotation=27, ha="right", fontsize=7.5)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value < 0.68 else "#111827",
                )
    for boundary in (2.5, 10.5, 18.5):
        axis.axhline(boundary, color="white", linewidth=1.8)
    axis.set_title("Full-split score by task at selected settings", weight="bold")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="Task score")
    figure.text(
        0.5,
        0.01,
        "MarinMoE uses the 8,192-base Natural readout; Dense uses its best 0.56T checkpoint. Published model values are reused from prior results.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    save(figure, output, "exp586-task-heatmap")


def binned_mean_loss(
    rows: list[dict], edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray([row["input_tokens"] for row in rows], dtype=np.float64) / 1e9
    losses = np.asarray([row["loss"] for row in rows], dtype=np.float64)
    assignments = np.digitize(tokens, edges) - 1
    centers, means = [], []
    for index in range(len(edges) - 1):
        values = losses[assignments == index]
        if not len(values):
            continue
        centers.append((edges[index] + edges[index + 1]) / 2)
        means.append(np.mean(values))
    return np.asarray(centers), np.asarray(means)


def training_loss(models: list[Model], history_path: Path, output: Path) -> None:
    models = selected(models, "loss")
    history = load(history_path)
    if history.get("schema") != 1:
        raise ValueError("Training-loss history must use schema 1")
    model_by_condition = {}
    run_by_condition = {}
    for model in models:
        run = history["runs"].get(model.key)
        if run is None:
            raise ValueError(f"Training-loss history omits {model.key}")
        condition = "scratch" if "scratch" in model.key else "pretrained"
        if condition in model_by_condition:
            raise ValueError(f"Loss view contains multiple {condition} models")
        if not run["training"] or not run["evaluation"]:
            raise ValueError(f"Training-loss history is empty for {model.key}")
        model_by_condition[condition] = model
        run_by_condition[condition] = run
    if set(model_by_condition) != {"scratch", "pretrained"}:
        raise ValueError("Loss view requires one scratch and one pretrained model")

    colors = [model_by_condition[name].color for name in ("scratch", "pretrained")]
    labels = [
        "Scratch · 8-token vocabulary",
        "Language-pretrained · 128k vocabulary",
    ]
    evaluations = []
    for condition in ("scratch", "pretrained"):
        evaluations.append(
            {
                row["input_tokens"]: row["loss"]
                for row in run_by_condition[condition]["evaluation"]
            }
        )
    common_tokens = np.asarray(
        sorted(evaluations[0].keys() & evaluations[1].keys()), dtype=np.int64
    )
    if not len(common_tokens):
        raise ValueError("Scratch and pretrained runs have no matched evaluations")
    eval_tokens = common_tokens / 1e9
    eval_losses = [
        np.asarray([values[int(token)] for token in common_tokens])
        for values in evaluations
    ]
    eval_gap = eval_losses[0] - eval_losses[1]

    first_ten_updates = 10 * 524_288
    early_edges = np.geomspace(first_ten_updates / 1e9, 5, 72)
    late_edges = np.arange(5, 225.001, 10)
    early_training = []
    late_training = []
    for condition in ("scratch", "pretrained"):
        rows = [
            row
            for row in run_by_condition[condition]["training"]
            if row["input_tokens"] > first_ten_updates
        ]
        early_training.append(binned_mean_loss(rows, early_edges))
        late_training.append(binned_mean_loss(rows, late_edges))
    if not np.allclose(late_training[0][0], late_training[1][0]):
        raise ValueError("Late training-loss intervals do not align")
    late_tokens = late_training[0][0]
    late_gap = late_training[0][1] - late_training[1][1]

    figure, axes = plt.subplots(2, 2, figsize=(14, 8.6))
    early_eval = eval_tokens <= 75
    for losses, color in zip(eval_losses, colors, strict=True):
        axes[0, 0].plot(
            eval_tokens[early_eval], losses[early_eval], color=color, linewidth=2
        )
    axes[0, 0].set_xlim(0, 75)
    axes[0, 0].set_ylim(1.045, 1.29)
    axes[0, 0].set_title("Evaluation: pretrained leads early", weight="bold")
    axes[0, 0].set_ylabel("Fixed-eval loss (lower is better)")
    axes[0, 0].annotate(
        f"{eval_gap[0]:.3f} lower at 1B",
        xy=(eval_tokens[0], eval_losses[1][0]),
        xytext=(11, 1.255),
        arrowprops={"arrowstyle": "->", "color": "#64748B"},
        color="#475569",
        fontsize=9,
    )
    threshold_index = next(
        index
        for index in range(len(eval_gap))
        if np.all(np.abs(eval_gap[index:]) < 0.005)
    )
    threshold = eval_tokens[threshold_index]
    axes[0, 0].annotate(
        f"gap stays below 0.005 after {threshold:.0f}B",
        xy=(threshold, eval_losses[1][threshold_index]),
        xytext=(31, 1.12),
        arrowprops={"arrowstyle": "->", "color": "#64748B"},
        color="#475569",
        fontsize=9,
    )

    axes[0, 1].plot(eval_tokens, eval_gap, color="#7C3AED", linewidth=2)
    axes[0, 1].fill_between(eval_tokens, 0, eval_gap, color="#7C3AED", alpha=0.12)
    axes[0, 1].axhline(0, color="#64748B", linewidth=1)
    axes[0, 1].set_xlim(0, 218)
    axes[0, 1].set_ylim(-0.001, 0.031)
    axes[0, 1].set_title("Evaluation: the advantage shrinks", weight="bold")
    axes[0, 1].set_ylabel("Scratch loss − pretrained loss")
    axes[0, 1].text(
        100,
        0.027,
        "Positive means pretrained has lower loss",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    axes[0, 1].annotate(
        f"{eval_gap[0]:.3f}",
        (eval_tokens[0], eval_gap[0]),
        xytext=(12, 0.028),
        arrowprops={"arrowstyle": "->", "color": "#64748B"},
        fontsize=9,
    )
    final_label = (
        f"Final: {eval_losses[0][-1]:.4f} scratch\n"
        f"{eval_losses[1][-1]:.4f} pretrained\n"
        f"Δ {eval_gap[-1]:.4f}"
    )
    axes[0, 1].annotate(
        final_label,
        (eval_tokens[-1], eval_gap[-1]),
        xytext=(158, 0.011),
        arrowprops={"arrowstyle": "->", "color": "#64748B"},
        fontsize=9,
    )

    for (tokens, losses), color in zip(early_training, colors, strict=True):
        axes[1, 0].plot(tokens, losses, color=color, linewidth=2)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlim(first_ten_updates / 1e9, 5)
    axes[1, 0].set_ylim(
        0.95, max(float(losses.max()) for _, losses in early_training) + 0.25
    )
    axes[1, 0].set_xticks(
        [0.01, 0.03, 0.1, 0.3, 1, 3, 5],
        [".01", ".03", ".1", ".3", "1", "3", "5"],
    )
    axes[1, 0].set_title("Training loss: rapid vocabulary adaptation", weight="bold")
    axes[1, 0].set_ylabel("Average training loss")
    axes[1, 0].annotate(
        "128k vocabulary starts costly",
        xy=(0.009, 4.55),
        xytext=(0.055, 4.75),
        arrowprops={"arrowstyle": "->", "color": "#64748B"},
        color="#475569",
        fontsize=9,
    )
    early_gap = early_training[0][1] - early_training[1][1]
    crossing_index = int(np.flatnonzero(early_gap >= 0)[0])
    crossing = early_training[0][0][crossing_index]
    axes[1, 0].annotate(
        f"pretrained catches up by ~{crossing * 1_000:.0f}M tokens",
        xy=(crossing, early_training[1][1][crossing_index]),
        xytext=(0.09, 2.65),
        arrowprops={"arrowstyle": "->", "color": "#64748B"},
        color="#475569",
        fontsize=9,
    )

    axes[1, 1].plot(
        late_tokens,
        late_gap,
        color="#7C3AED",
        linewidth=2,
        marker="o",
        markersize=3,
    )
    axes[1, 1].fill_between(
        late_tokens,
        0,
        late_gap,
        where=late_gap >= 0,
        color=model_by_condition["pretrained"].color,
        alpha=0.13,
        interpolate=True,
    )
    axes[1, 1].fill_between(
        late_tokens,
        0,
        late_gap,
        where=late_gap < 0,
        color=model_by_condition["scratch"].color,
        alpha=0.13,
        interpolate=True,
    )
    axes[1, 1].axhline(0, color="#64748B", linewidth=1)
    axes[1, 1].set_xlim(5, 218)
    axes[1, 1].set_ylim(
        min(-0.005, float(late_gap.min()) - 0.002), float(late_gap.max()) + 0.002
    )
    axes[1, 1].set_title("Late training loss: no persistent difference", weight="bold")
    axes[1, 1].set_ylabel("Scratch loss − pretrained loss")
    axes[1, 1].text(
        112,
        float(late_gap.max()) * 0.80,
        "Positive means pretrained has lower loss",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    axes[1, 1].annotate(
        "small differences fluctuate around zero",
        xy=(130, 0),
        xytext=(82, -0.0038),
        arrowprops={"arrowstyle": "->", "color": "#64748B"},
        color="#475569",
        fontsize=9,
    )

    for axis in axes.ravel():
        axis.grid(alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_xlabel("DNA tokens (billions)")
    figure.legend(
        handles=[
            Line2D([0], [0], color=color, linewidth=2, label=label)
            for color, label in zip(colors, labels, strict=True)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Language pretraining helps early evaluation; both models converge",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Evaluation lines use the same 512 examples at matched token counts. Training lines average nearby logged batches to remove batch noise; the isolated pretrained-only eval at 13M tokens is omitted.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.subplots_adjust(
        left=0.075, right=0.985, bottom=0.10, top=0.86, wspace=0.22, hspace=0.34
    )
    save(figure, output, "exp586-training-loss")


def task_context_deltas(
    models: list[Model],
    output: Path,
) -> None:
    models = selected(models, "context")
    columns = {}
    labels = None
    task_names = None
    for model in models:
        result = model.result
        long_rows = {
            row["key"]: row
            for row in result["profiles"]["context-8192"]["tasks"]["rows"]
        }
        short_rows = result["profiles"]["context-2048"]["tasks"]["rows"]
        current_labels = [f"{row['category']} · {row['task']}" for row in short_rows]
        if labels is None:
            labels = current_labels
            task_names = [row["task"] for row in short_rows]
        elif labels != current_labels:
            raise ValueError("2,048-token task inventories differ between conditions")
        for label, field in (
            ("natural", "natural"),
            ("two-sided PoE", "bidirectional_poe"),
        ):
            values = []
            for row in short_rows:
                baseline = long_rows[row["key"]]
                if row[field] is None or baseline[field] is None:
                    raise ValueError("Context delta requires paired task values")
                values.append(row[field] - baseline[field])
            columns[f"{condition_label(model)}\n{label}"] = values
    assert task_names is not None
    abbreviated = []
    for task in task_names:
        task = task.replace("Andropogoneae, genome-wide", "Androp.\ngenome")
        task = task.replace("Poaceae, non-TIS CDS", "Poaceae\nnon-TIS")
        task = task.replace("Poaceae, TIS CDS", "Poaceae\nTIS")
        task = task.replace(" (start)", "").replace(" (stop)", "")
        task = task.replace(" splice ", "\n")
        task = task.replace(" ", "\n", 1) if "\n" not in task else task
        abbreviated.append(task)
    values = np.asarray(list(columns.values()))
    limit = max(0.03, float(np.quantile(np.abs(values), 0.98)))
    figure, axis = plt.subplots(
        figsize=(max(12.0, 0.72 * len(task_names)), 3.4 + 0.55 * len(columns))
    )
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(np.arange(len(abbreviated)), abbreviated, fontsize=7)
    row_labels = list(columns)
    if len(models) == 1:
        row_labels = ["Natural", "Two-sided PoE"]
    axis.set_yticks(np.arange(len(columns)), row_labels, fontsize=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=(
                    "white" if abs(values[row, column]) > 0.55 * limit else "#111827"
                ),
            )
    for boundary in (2.5, 10.5):
        axis.axvline(boundary, color="white", linewidth=1.8)
    title = "Task-score change from 8,192 to 2,048 bases"
    if len(models) == 1:
        title += f"\n{compact_label(models[0].label)}"
    axis.set_title(title, weight="bold")
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.022,
        pad=0.02,
        label="Δ task score (2,048 − 8,192)",
    )
    figure.text(
        0.5,
        0.02,
        "Positive values favor the centered 2,048-base crop. Structural variants are omitted because their historical metric requires the 8,192-position indel boundary.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    left = 0.1 if len(models) == 1 else 0.22
    figure.subplots_adjust(bottom=0.25, left=left, right=0.93, top=0.8)
    save(figure, output, "exp586-task-context-deltas")


def af_readouts(
    models: list[Model],
    historical: dict,
    output: Path,
) -> None:
    models = selected(models, "af")
    methods = (
        ("Natural", "llr_acgt_avg"),
        ("Full vocab", "llr_full_vocab_avg"),
        ("Two-sided PoE", "llr_target_poe"),
    )
    contexts = (("context-8192", "8K"), ("context-2048", "2K"))
    display = {
        "Natural": "Natural",
        "Full vocab": "Full vocab",
        "Two-sided PoE": "Two-sided\nPoE",
    }
    columns = [
        f"{display[label]}\n{short}" for label, _ in methods for _, short in contexts
    ]
    values = []
    for model in models:
        row = []
        for _, key in methods:
            for context, _ in contexts:
                metrics = model.result["profiles"][context]["allele_frequency"][
                    "scores"
                ]
                row.append(metrics[key]["overall"]["spearman"])
        values.append(row)
    values = np.asarray(values)
    figure, axis = plt.subplots(figsize=(9.6, max(4.4, 1.05 * len(models) + 1.8)))
    spread = float(np.ptp(values))
    padding = max(0.003, 0.06 * spread)
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="viridis",
        vmin=float(values.min() - padding),
        vmax=float(values.max() + padding),
    )
    axis.set_xticks(np.arange(len(columns)), columns, fontsize=9)
    axis.set_yticks(
        np.arange(len(models)), [condition_label(model) for model in models]
    )
    best = np.argmax(values, axis=1)
    threshold = float(values.min() + 0.45 * spread)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                weight="bold" if column == best[row] else "normal",
                color="white" if values[row, column] < threshold else "#111827",
            )
    for boundary in (1.5, 3.5):
        axis.axvline(boundary, color="white", linewidth=1.8)
    axis.set_title("Maize allele-frequency Spearman correlation", weight="bold")
    figure.colorbar(
        image, ax=axis, fraction=0.025, pad=0.025, label="Spearman correlation"
    )
    dense = [
        historical[rung]["metrics"]["100000"]["llr_acgt_avg"]["spearman"]
        for rung in ("022t", "039t", "056t")
    ]
    legacy = historical["022t"]["historical_10k"]
    reference = (
        f"Prior 8K dense range: {min(dense):.3f}–{max(dense):.3f} · "
        f"PlantCAD2-Small: {legacy['PlantCAD2-Small']['spearman']:.3f} · "
        f"PlantCAD2-Large: {legacy['PlantCAD2-Large']['spearman']:.3f} "
        "(legacy 10k)"
    )
    figure.text(0.5, 0.02, reference, ha="center", fontsize=8.5, color="#475569")
    figure.subplots_adjust(bottom=0.18, left=0.29, right=0.93, top=0.86)
    save(figure, output, "exp586-af-readouts")


def probe_value(result: dict, profile: str, key: str) -> tuple[float, list[float]]:
    value = result[profile]["metrics"][key]["overall"]["probe"]
    return value["value"], value["ci95"]


def probe_readouts(
    models: list[Model],
    historical_probe: dict,
    output: Path,
) -> None:
    models = selected(models, "probes")
    probe_results = {
        model.key: {
            profile: load(model.directory / profile / "probes/result.json")
            for profile in ("context-8192", "context-2048")
        }
        for model in models
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), sharey=True)
    keys = (
        ("Whole\naveraged", "whole_window/strand_average"),
        ("Whole\ndirectional", "whole_window/directional"),
        ("Target\naveraged", "variant_token/strand_average"),
        ("Target\ndirectional", "variant_token/directional"),
    )
    x = np.arange(len(keys))
    for axis, profile in zip(axes, ("context-8192", "context-2048"), strict=True):
        offsets = np.linspace(-0.09, 0.09, len(models))
        for model, offset in zip(models, offsets, strict=True):
            values, lower, upper = [], [], []
            for _, key in keys:
                value, interval = probe_value(probe_results[model.key], profile, key)
                values.append(value)
                lower.append(value - interval[0])
                upper.append(interval[1] - value)
            axis.errorbar(
                x + offset,
                values,
                yerr=np.asarray([lower, upper]),
                marker="o",
                capsize=3,
                linewidth=1.8,
                color=model.color,
                label=condition_label(model),
            )
        if profile == "context-8192":
            for rung, style in zip(
                ("022t", "039t", "056t"), (":", "--", "-."), strict=True
            ):
                reference = historical_probe["validation"][rung]
                axis.plot(
                    [x[0], x[2]],
                    [reference["whole_window"], reference["variant_token"]],
                    color=COLORS["dense"],
                    linestyle=style,
                    marker=".",
                    alpha=0.65,
                    label=DENSE_LABELS[rung],
                )
        axis.set_xticks(x, [label for label, _ in keys])
        axis.set_title(profile.removeprefix("context-") + " bases", weight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Held-out Spearman correlation with maize AF")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
        fontsize=7,
        ncol=3,
    )
    figure.suptitle(
        "Frozen-embedding probes: strand averaging vs directional features",
        fontsize=13,
        weight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Points show the fixed held-out split; error bars are paired genomic-block bootstrap 95% intervals.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    figure.subplots_adjust(bottom=0.28, top=0.84, wspace=0.1)
    save(figure, output, "exp586-probe-readouts")


def intrinsic_methods(
    models: list[Model],
    output: Path,
) -> None:
    models = selected(models, "methods")
    figure, axes = plt.subplots(
        1, len(models), figsize=(7.0 * len(models), 4.5), sharey=False
    )
    axes = np.atleast_1d(axes)
    methods = (
        ("Forward", "left_test", "#0F766E"),
        ("Reverse complement", "right_test", "#EA580C"),
        ("Equal two-sided", "poe_equal_test", COLORS["poe"]),
    )
    for axis, model in zip(axes, models, strict=True):
        report = model.method
        assert report is not None
        sweep = report["intrinsic"]["context_sweep"]
        contexts = sorted(map(int, sweep))
        for label, key, color in methods:
            axis.plot(
                contexts,
                [sweep[str(context)][key]["nll"] for context in contexts],
                marker="o",
                linewidth=2,
                color=color,
                label=label,
            )
        axis.axhline(
            np.log(4),
            color="#64748B",
            linestyle=":",
            linewidth=1.3,
            label="Uniform-base chance",
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(contexts, [f"{value:,}" for value in contexts])
        axis.set_xlabel("One-sided context (bases)")
        if len(models) > 1:
            axis.set_title(condition_label(model), weight="bold", fontsize=10)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Held-out nucleotide NLL (lower is better)")
    axes[-1].legend(frameon=False, fontsize=8)
    title = "Task-independent bidirectional inference validation"
    if len(models) == 1:
        title += f"\n{compact_label(models[0].label)}"
    figure.suptitle(title, fontsize=13, weight="bold")
    figure.text(
        0.5,
        0.018,
        "The sweep is independent of PlantCAD2 task labels; both orientations from a source window stay in the same split.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    figure.subplots_adjust(top=0.76, bottom=0.19)
    save(figure, output, "exp586-intrinsic-methods")


def prompt_methods(
    models: list[Model],
    output: Path,
) -> None:
    models = selected(models, "methods")
    figure, axes = plt.subplots(
        1, len(models), figsize=(6.25 * len(models), 4.8), sharex=True
    )
    axes = np.atleast_1d(axes)
    labels = ("Forward", "Reverse comp.", "Two-sided PoE", "Rotated", "Sandwich")
    colors = ("#0F766E", "#EA580C", COLORS["poe"], "#0284C7", "#BE123C")
    for axis, model in zip(axes, models, strict=True):
        report = model.method
        assert report is not None
        intrinsic = report["intrinsic"]
        context = intrinsic["context_sweep"]["2048"]
        prompts = intrinsic["prompt_methods_2048"]
        values = (
            context["left_test"]["nll"],
            context["right_test"]["nll"],
            context["poe_equal_test"]["nll"],
            prompts["rotated"]["nll"],
            prompts["sandwich"]["nll"],
        )
        y = np.arange(len(labels))
        axis.barh(y, values, color=colors, alpha=0.9)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_title(condition_label(model), weight="bold")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        axis.axvline(np.log(4), color="#64748B", linestyle=":", linewidth=1.3, zorder=0)
        for row, value in enumerate(values):
            axis.text(value + 0.035, row, f"{value:.3f}", va="center", fontsize=8)
    axes[0].set_xlim(0.95, max(axis.get_xlim()[1] for axis in axes) + 0.25)
    for axis in axes:
        axis.set_xlabel("Held-out nucleotide NLL (lower is better)")
    figure.suptitle(
        "Inference-only bidirectional prompt alternatives at 2,048 bases",
        fontsize=15,
        weight="bold",
    )
    figure.subplots_adjust(bottom=0.2)
    figure.text(
        0.5,
        0.025,
        "The dotted line marks uniform-base chance. Equal two-sided scores use separate forward and reverse-complement passes; rotated and sandwich prompts use one causal pass.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    save(figure, output, "exp586-prompt-methods")


def create_report(
    catalog: Path,
    historical: Path,
    output: Path,
) -> None:
    verify_exp582_results(historical.parent)
    models = load_catalog(catalog)
    history = load(historical / "exp472-lr5e4-full-coreweave-20260831.json")
    af_history = {
        rung: load(historical / f"plantcad2-maize-af-{rung}-20260904-v1.json")
        for rung in ("022t", "039t", "056t")
    }
    model_overview(models, history, output)
    if any("loss" in model.views for model in models):
        training_loss(models, catalog.parent / "training-loss-history.json", output)
    task_heatmap(models, history, output)
    task_context_deltas(models, output)
    af_readouts(models, af_history, output)
    probe_models = selected(models, "probes")
    probe_paths = [
        model.directory / profile / "probes/result.json"
        for model in probe_models
        for profile in ("context-8192", "context-2048")
    ]
    present = [path.exists() for path in probe_paths]
    if any(present) and not all(present):
        raise ValueError("Probe results must be either complete or absent")
    if all(present):
        probe_history = load(
            historical / "exp472-lr5e4-maize-af-probe-validation-20260904.json"
        )
        probe_readouts(models, probe_history, output)
    intrinsic_methods(models, output)
    prompt_methods(models, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    create_report(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
