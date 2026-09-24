#!/usr/bin/env python3
"""Plot reviewer-facing response controls for the information-anchor manuscript.

The script consumes only audited CSV artifacts. It does not rerun a model or
change the registered MI estimator. The default entry point emits only the
native High/Low/Random and interpolation figures retained by the manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm


CODE_OR_REPO_ROOT = Path(__file__).resolve().parents[2]
if CODE_OR_REPO_ROOT.name == "code" and CODE_OR_REPO_ROOT.parent.name == "artifact":
    ROOT = CODE_OR_REPO_ROOT.parent
    OUT = ROOT / "generated_figures"
    TABLE_ROOT = ROOT / "tables"
else:
    ROOT = CODE_OR_REPO_ROOT
    OUT = ROOT / "results/information_anchor_iclr_figures"
    TABLE_ROOT = ROOT / "results"
AUDIT = TABLE_ROOT / "information_anchor_recency_target_audit"
TOPOLOGY = TABLE_ROOT / "information_anchor_v6_topology"
EVIDENCE = TABLE_ROOT / "information_anchor_v6_evidence"
PRECISION = TABLE_ROOT / "information_anchor_mi_control_baseline_evidence"

MODELS = ("Chronos2", "Moirai2", "Toto2", "TimesFM2.5", "ChronosBolt", "TTM")
MODEL_LABELS = {
    "Chronos2": "Chronos-2",
    "Moirai2": "Moirai-2",
    "Toto2": "Toto-2",
    "TimesFM2.5": "TimesFM 2.5",
    "ChronosBolt": "Chronos-Bolt",
    "TTM": "TTM-R2",
}
MODEL_COLORS = {
    "Chronos2": "#2F6690",
    "Moirai2": "#368F8B",
    "Toto2": "#B24A4A",
    "TimesFM2.5": "#7D5A9E",
    "ChronosBolt": "#B0782C",
    "TTM": "#555555",
}
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic")
DATASET_LABELS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic")
SEMANTIC_GROUPS = ("level", "trajectory", "local_dynamics", "frequency")
SEMANTIC_LABELS = ("Level", "Trajectory", "Local dynamics", "Frequency")

HIGH = "#B23A3A"
LOW = "#2F6690"
RECENT = "#D08A2E"
RANDOM = "#777777"
NEUTRAL = "#D5D5D5"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, subtitle: str, y: float = -0.26) -> None:
    ax.text(0.5, y, f"({label}) {subtitle}", transform=ax.transAxes, ha="center", va="top")


def percentile_rank(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = (np.arange(flat.size, dtype=np.float64) + 0.5) / flat.size
    return ranks.reshape(values.shape)


def audited_atlas() -> np.ndarray:
    topology = pd.read_csv(TOPOLOGY / "topology_descriptors.csv")
    row = topology[(topology["model"] == "Chronos2") & (topology["dataset"] == "ETTm1")].iloc[0]
    with np.load(Path(row["run_dir"]) / "mi_results.npz") as values:
        return np.asarray(values["mi_z"], dtype=np.float64)


def figure_1() -> None:
    atlas_z = audited_atlas()
    atlas = percentile_rank(atlas_z)
    top = np.unravel_index(np.nanargmax(atlas_z), atlas_z.shape)

    # Figure contract: one compact evidence chain from frozen forecast to a
    # discovery-locked candidate, held-out validation, and the supported claim.
    # The atlas is the hero object; pale sections organize the eye without the
    # heavy framed-card appearance of the previous version.
    fig, ax = plt.subplots(figsize=(7.2, 1.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ink = "#24364B"
    muted = "#687482"
    rules = "#CBD2D9"
    signal = "#C85C55"
    recency = "#D59A3A"
    validation = "#4E7D6A"

    sections = (
        (0.012, 0.075, 0.190, 0.855, "1", "FROZEN FORECAST", "#F5F8FA", "#66859B"),
        (0.240, 0.075, 0.247, 0.855, "2", "DISCOVERY", "#F8F6F9", "#8B7193"),
        (0.525, 0.075, 0.270, 0.855, "3", "HELD-OUT VALIDATION", "#FBF8F4", "#A97854"),
        (0.833, 0.075, 0.155, 0.855, "4", "SUPPORTED SCOPE", "#F3F8F5", validation),
    )
    for x, y, width, height, number, heading, face, accent in sections:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.004,rounding_size=0.010",
                facecolor=face, edgecolor="none",
            )
        )
        ax.plot([x + 0.010, x + width - 0.010], [y + height - 0.118] * 2,
                color=accent, linewidth=1.15, solid_capstyle="round")
        ax.text(x + 0.012, y + height - 0.062, number, color=accent,
                fontsize=7.2, fontweight="bold", va="center")
        ax.text(x + 0.035, y + height - 0.062, heading, color=ink,
                fontsize=6.1, fontweight="bold", va="center")

    for start, stop in ((0.205, 0.237), (0.490, 0.522), (0.798, 0.830)):
        ax.add_patch(FancyArrowPatch((start, 0.50), (stop, 0.50),
                                    arrowstyle="-|>", mutation_scale=7.5,
                                    linewidth=0.85, color="#88939D"))

    # Stage 1: a real multivariate forecasting object, drawn as trajectories
    # rather than matrix tiles so history and future are immediately legible.
    input_ax = ax.inset_axes((0.028, 0.265, 0.105, 0.405))
    x_hist = np.linspace(0.0, 1.0, 48)
    curves = (
        0.56 + 0.19 * np.sin(2.5 * np.pi * x_hist) + 0.11 * x_hist,
        0.42 + 0.15 * np.cos(2.0 * np.pi * x_hist + 0.7) - 0.06 * x_hist,
        0.30 + 0.10 * np.sin(4.2 * np.pi * x_hist + 0.8) + 0.21 * x_hist,
    )
    for values, color, alpha in zip(curves, ("#4F81A4", "#79A7B8", "#A6BCC6"), (1.0, 0.88, 0.85)):
        input_ax.plot(x_hist, values, color=color, linewidth=1.05, alpha=alpha)
    input_ax.axvline(1.0, color=recency, linewidth=1.0, linestyle=(0, (2, 1.5)))
    input_ax.set_xlim(0, 1.02); input_ax.set_ylim(0.08, 0.88)
    input_ax.set_xticks([]); input_ax.set_yticks([])
    for spine in input_ax.spines.values(): spine.set_visible(False)
    ax.text(0.081, 0.205, "multivariate history", ha="center", fontsize=5.7, color=muted)

    ax.add_patch(FancyBboxPatch((0.143, 0.42), 0.045, 0.155,
                               boxstyle="round,pad=0.003,rounding_size=0.006",
                               facecolor=ink, edgecolor="none"))
    ax.text(0.1655, 0.497, "frozen\nTSFM", ha="center", va="center",
            fontsize=5.6, color="white", fontweight="bold", linespacing=1.0)
    ax.text(0.136, 0.497, "→", ha="center", va="center", fontsize=8, color="#72808C")
    forecast_ax = ax.inset_axes((0.147, 0.270, 0.038, 0.095))
    xf = np.linspace(0, 1, 16)
    forecast_ax.plot(xf, 0.47 + 0.24 * np.sin(1.4 * np.pi * xf + 0.4), color=signal, linewidth=1.1)
    forecast_ax.set_axis_off()
    ax.text(0.166, 0.205, "forecast", ha="center", fontsize=5.7, color=signal)

    # Stage 2: the native layer-by-patch atlas is the hero object.
    atlas_ax = ax.inset_axes((0.257, 0.235, 0.126, 0.495))
    atlas_ax.imshow(atlas, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    atlas_ax.axvspan(atlas.shape[1] * 0.75 - 0.5, atlas.shape[1] - 0.5,
                    color="#FFD56B", alpha=0.13, linewidth=0)
    atlas_ax.scatter([top[1]], [top[0]], s=23, facecolors="none",
                     edgecolors="#FFF3A6", linewidths=1.15)
    atlas_ax.set_xticks([]); atlas_ax.set_yticks([])
    for spine in atlas_ax.spines.values(): spine.set_visible(False)
    ax.text(0.320, 0.188, "native patch  old  →  recent", ha="center", fontsize=5.5, color=muted)

    ax.text(0.397, 0.690, "candidate", fontsize=6.4, color="#6F5079", fontweight="bold")
    ax.text(0.397, 0.607, "locked before", fontsize=5.45, color=muted)
    ax.text(0.397, 0.548, "validation", fontsize=5.45, color=muted)
    for y, marker, label, color in (
        (0.390, "●", "temporal null", signal),
        (0.315, "●", "same-layer Low MI", "#6A88A3"),
        (0.240, "●", "same-layer Recent", recency),
    ):
        ax.text(0.399, y, marker, fontsize=5.3, color=color, va="center")
        ax.text(0.412, y, label, fontsize=5.3, color=ink, va="center")

    # Stage 3: three non-equivalent legs, plus sensitivity audits kept visually
    # subordinate to the primary validation rule.
    validation_rows = (
        (0.665, "1", "Dependence", r"shift-null $q<.05$", "#8B7193"),
        (0.505, "2", "Accessibility", r"held-out coarse $R^2$", "#5B7892"),
        (0.345, "3", "Contribution", r"paired ground-truth $\Delta$MSE", signal),
    )
    for y, number, heading, detail, color in validation_rows:
        ax.add_patch(FancyBboxPatch((0.544, y - 0.056), 0.232, 0.116,
                                   boxstyle="round,pad=0.003,rounding_size=0.008",
                                   facecolor="white", edgecolor=rules, linewidth=0.55))
        ax.scatter([0.562], [y], s=26, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(0.562, y, number, color="white", fontsize=5.0, fontweight="bold",
                ha="center", va="center", zorder=4)
        ax.text(0.581, y + 0.021, heading, fontsize=6.0, fontweight="bold", color=ink, va="center")
        ax.text(0.581, y - 0.030, detail, fontsize=5.35, color=muted, va="center")
    ax.plot([0.545, 0.775], [0.205, 0.205], color=rules, linewidth=0.55)
    ax.text(0.660, 0.145, "Sensitivity audits: forecast drift · position · donor dose",
            ha="center", fontsize=5.1, color=muted)

    # Stage 4: make the primary pass count prominent, but keep the claim boundary
    # equally visible so the figure cannot be read as a universal-policy result.
    ax.add_patch(FancyBboxPatch((0.849, 0.590), 0.123, 0.154,
                               boxstyle="round,pad=0.004,rounding_size=0.010",
                               facecolor="#E3F0E9", edgecolor="#A7C5B7", linewidth=0.65))
    ax.text(0.9105, 0.687, "33 / 42", ha="center", va="center",
            fontsize=9.0, color="#315E4C", fontweight="bold")
    ax.text(0.9105, 0.618, "validated anchors", ha="center", va="center",
            fontsize=5.35, color="#315E4C")
    ax.text(0.9105, 0.476, "recency-aligned", ha="center", fontsize=6.3,
            color=ink, fontweight="bold")
    ax.text(0.9105, 0.407, "temporal diagnostic", ha="center", fontsize=6.3,
            color=ink, fontweight="bold")
    ax.plot([0.858, 0.963], [0.340, 0.340], color="#B5C7BE", linewidth=0.65)
    ax.text(0.9105, 0.280, "Recent > MI on average", ha="center",
            fontsize=5.25, color=signal, fontweight="bold")
    ax.text(0.9105, 0.215, "27/42 removal curves", ha="center", fontsize=4.9, color=muted)
    ax.text(0.9105, 0.132, "fine layer identity is conditional", ha="center",
            fontsize=4.75, color="#7B858D")

    fig.subplots_adjust(left=0.003, right=0.997, top=0.99, bottom=0.015)
    save_figure(fig, "Fig1_evidence_overview")


def figure_3_probe_compact() -> None:
    data = pd.read_csv(EVIDENCE / "probe_semantics.csv")
    semantics = (
        "level_q1", "level_q2", "level_q3", "level_q4", "global_change", "linear_trend",
        "mean_absolute_change", "diff_std", "roughness", "low_frequency_energy",
        "mid_frequency_energy", "high_frequency_energy",
    )
    labels = ("L1", "L2", "L3", "L4", "Change", "Trend", "Mean |Δ|", "SD(Δ)", "Rough.", "Low E", "Mid E", "High E")

    def matrix(scope: str) -> np.ndarray:
        rows = data[data["scope"] == scope].copy()
        rows["semantic_clean"] = rows["semantic"].str.split(":").str[-1]
        output = np.full((len(MODELS), len(semantics)), np.nan, dtype=np.float64)
        for i, model in enumerate(MODELS):
            subset = rows[rows["model"] == model]
            for j, semantic in enumerate(semantics):
                values = subset.loc[subset["semantic_clean"] == semantic, "functional_top_minus_low_r2"]
                if len(values): output[i, j] = float(values.mean())
        return output

    global_matrix = matrix("global")
    target_matrix = matrix("target")
    finite = np.concatenate((global_matrix[np.isfinite(global_matrix)], target_matrix[np.isfinite(target_matrix)]))
    limit = float(max(abs(np.quantile(finite, 0.02)), abs(np.quantile(finite, 0.98)), 0.25))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.62), constrained_layout=True)
    image = None
    for index, (ax, values, scope) in enumerate(((axes[0], global_matrix, "Global semantics"), (axes[1], target_matrix, "Target-channel semantics"))):
        image = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[model] for model in MODELS] if index == 0 else [])
        ax.tick_params(length=0)
        for boundary in (3.5, 5.5, 8.5):
            ax.axvline(boundary, color="white", linewidth=1.2)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(0.5, -0.34, f"({chr(ord('a') + index)}) {scope}", transform=ax.transAxes, ha="center", va="top", fontsize=7)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.021, pad=0.018)
    colorbar.set_label(r"High-minus-low held-out $R^2$", labelpad=2)
    colorbar.ax.tick_params(labelsize=5.5)
    save_figure(fig, "Fig3_probe_semantics")


def figure_4_functional_controls() -> None:
    curves = pd.read_csv(AUDIT / "progressive_model_dataset_curves.csv")
    dose = pd.read_csv(AUDIT / "interpolation_conditions.csv")

    fig = plt.figure(figsize=(7.2, 2.55))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.10), wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    specs = (("remove", "top_minus_bottom", "Remove\nvs low", HIGH),
             ("remove", "top_minus_random_mean", "Remove\nvs random", "#8E4B4B"),
             ("keep", "top_minus_bottom", "Keep\nvs low", LOW),
             ("keep", "top_minus_random_mean", "Keep\nvs random", "#4B718E"))
    rng = np.random.default_rng(44)
    for index, (mode, contrast, label, color) in enumerate(specs):
        values = curves[(curves["mode"] == mode) & (curves["contrast"] == contrast)][
            "oriented_all_delta_mse_mean_across_fractions"
        ].to_numpy(dtype=np.float64)
        ax.scatter(index + rng.uniform(-0.10, 0.10, len(values)), values, s=10, color=color, alpha=0.45, linewidths=0)
        ax.plot([index - 0.20, index + 0.20], [np.mean(values), np.mean(values)], color="black", linewidth=1.25)
    ax.axhline(0, color="#777777", linewidth=0.65)
    ax.set_xticks(range(4))
    ax.set_xticklabels([s[2] for s in specs])
    ax.set_ylabel(r"Oriented ground-truth $\Delta$MSE")
    panel_label(ax, "a", "42 model-regime curves", -0.30)

    ax = fig.add_subplot(grid[0, 1])
    normalized = []
    for model in MODELS:
        rows = dose[dose["model"] == model]
        denominator = float(rows[(rows["rank"] == "high") & np.isclose(rows["alpha"], 1.0)]["all_forecast_change_mse_mean"].iloc[0])
        for rank in ("high", "low"):
            values = rows[rows["rank"] == rank].sort_values("alpha")
            y = values["all_forecast_change_mse_mean"].to_numpy(dtype=np.float64) / max(denominator, 1e-12)
            normalized.append(pd.DataFrame({"model": model, "rank": rank, "alpha": values["alpha"], "value": y}))
            ax.plot(values["alpha"], y, color=HIGH if rank == "high" else LOW, alpha=0.20, linewidth=0.8)
    normalized_frame = pd.concat(normalized, ignore_index=True)
    for rank, color, marker, label in (("high", HIGH, "o", "High MI"), ("low", LOW, "s", "Low MI")):
        summary = normalized_frame[normalized_frame["rank"] == rank].groupby("alpha")["value"].agg(
            mean="mean", lower="min", upper="max"
        )
        x = summary.index.to_numpy(dtype=np.float64)
        ax.plot(x, summary["mean"], color=color, marker=marker, markersize=3.4, linewidth=1.5, label=label)
        ax.fill_between(x, summary["lower"], summary["upper"], color=color, alpha=0.09, linewidth=0)
    ax.set_xticks((0.25, 0.5, 1.0))
    ax.set_xlabel(r"Donor interpolation $\alpha$")
    ax.set_ylabel(r"Change / high-MI effect at $\alpha=1$")
    ax.set_ylim(bottom=-0.01)
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "b", "Interpolation dose response", -0.30)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.96, bottom=0.25)
    save_figure(fig, "Fig4_functional_controls")


def figure_s_functional_compact() -> None:
    """Plot the complete 6-by-7 response matrix for Low-MI and Random controls."""
    curves = pd.read_csv(AUDIT / "progressive_model_dataset_curves.csv")
    dataset_sources = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic")
    specs = []
    for mode in ("remove", "keep"):
        for contrast, title in (
            ("top_minus_bottom", "High - Low"),
            ("top_minus_random_mean", "High - Random"),
        ):
            rows = curves[(curves["mode"] == mode) & (curves["contrast"] == contrast)]
            matrix = np.full((len(MODELS), len(dataset_sources)), np.nan)
            for row_index, model in enumerate(MODELS):
                for column_index, dataset in enumerate(dataset_sources):
                    values = rows[
                        (rows["model"] == model)
                        & (rows["dataset"].str.lower() == dataset.lower())
                    ]["oriented_all_delta_mse_mean_across_fractions"]
                    if len(values):
                        matrix[row_index, column_index] = float(values.mean())
            specs.append((matrix, f"{mode.capitalize()}: {title}"))

    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix, _ in specs])
    limit = float(max(abs(np.quantile(finite, 0.02)), abs(np.quantile(finite, 0.98))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    image = None
    for index, (ax, (matrix, title)) in enumerate(zip(axes.ravel(), specs)):
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set_xticks(range(len(DATASET_LABELS)), DATASET_LABELS, rotation=42, ha="right")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[model] for model in MODELS] if index % 2 == 0 else [])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=6.7, pad=2)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.016, pad=0.014)
    colorbar.set_label(r"Oriented curve-level ground-truth $\Delta$MSE")
    save_figure(fig, "FigS_functional_compact")


def figure_5_recency_sensitivity() -> None:
    recency = pd.read_csv(AUDIT / "topology_recency_audit.csv")
    probes = pd.read_csv(AUDIT / "probe_same_layer_recent.csv")
    sensitivity = pd.read_csv(AUDIT / "cached_mi_sensitivity.csv")

    fig = plt.figure(figsize=(7.2, 2.55))
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.05, 1.0), wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    rng = np.random.default_rng(5)
    for index, (column, label, color) in enumerate((
        ("patch_mi_position_spearman", "MI-position\nSpearman", HIGH),
        ("top_250_recent_jaccard", "Top-25% / recent\nJaccard", RECENT),
    )):
        values = recency[column].to_numpy(dtype=np.float64)
        ax.scatter(index + rng.uniform(-0.10, 0.10, len(values)), values, s=10, color=color, alpha=0.42, linewidths=0)
        ax.plot([index - 0.20, index + 0.20], [np.mean(values), np.mean(values)], color="black", linewidth=1.25)
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.set_xticks((0, 1))
    ax.set_xticklabels(("MI-position\nSpearman", "Top-25% / recent\nJaccard"))
    ax.set_ylabel("Across 42 atlases")
    panel_label(ax, "a", "Recency overlap", -0.32)

    ax = fig.add_subplot(grid[0, 1])
    x = np.arange(len(MODELS), dtype=np.float64)
    width = 0.34
    values = []
    for scope in ("global", "target"):
        rows = probes[probes["scope"] == scope].groupby("model").agg(
            top_recent=("top_minus_recent", "mean"), top_low=("top_minus_low", "mean")
        ).reindex(MODELS)
        values.append(rows)
    global_rows, target_rows = values
    ax.bar(x - width / 2, global_rows["top_low"], width, color=HIGH, alpha=0.82, label="High-low")
    ax.bar(x + width / 2, global_rows["top_recent"], width, color=RECENT, alpha=0.90, label="High-recent")
    ax.scatter(x - width / 2, target_rows["top_low"], marker="D", s=11, color="#5C1717", zorder=3)
    ax.scatter(x + width / 2, target_rows["top_recent"], marker="D", s=11, color="#7A4B12", zorder=3)
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=42, ha="right")
    ax.set_ylabel(r"Mean probe $R^2$ difference")
    ax.legend(frameon=False, loc="lower left")
    panel_label(ax, "b", "Same-layer probe controls", -0.39)

    ax = fig.add_subplot(grid[0, 2])
    categories = (
        ("full_seasonal", "patch_z_spearman_with_primary", "Seasonal null\npatch rank", HIGH),
        ("dynamics_legal", "patch_z_spearman_with_primary", "Dynamics target\npatch rank", LOW),
        ("full_seasonal", "layer_profile_spearman_with_primary", "Seasonal null\nlayer profile", "#B4772E"),
        ("dynamics_legal", "layer_profile_spearman_with_primary", "Dynamics target\nlayer profile", "#5A6B72"),
    )
    for index, (variant, column, _, color) in enumerate(categories):
        values = sensitivity[sensitivity["variant"] == variant][column].to_numpy(dtype=np.float64)
        ax.scatter(index + rng.uniform(-0.08, 0.08, len(values)), values, s=16, color=color, alpha=0.75, linewidths=0)
        ax.plot([index - 0.18, index + 0.18], [np.median(values), np.median(values)], color="black", linewidth=1.15)
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([item[2] for item in categories], rotation=34, ha="right")
    ax.set_ylabel("Spearman with primary atlas")
    ax.set_ylim(-0.55, 1.05)
    panel_label(ax, "c", "Target and null sensitivity", -0.39)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.96, bottom=0.31)
    save_figure(fig, "Fig5_recency_sensitivity")


def figure_6_precision() -> None:
    recovery = pd.read_csv(PRECISION / "recovery_cells.csv")
    recent = pd.read_csv(AUDIT / "precision_same_layer_recent_pairwise.csv")
    colors = {
        "high_mi": HIGH,
        "low_mi": LOW,
        "random": RANDOM,
        "null_mi": "#647649",
        "activation_energy": "#A6742D",
        "activation_variance": "#7C5B88",
    }
    labels = {
        "high_mi": "High MI",
        "low_mi": "Low MI",
        "random": "Random",
        "null_mi": "Null MI",
        "activation_energy": "Activation energy",
        "activation_variance": "Activation variance",
    }
    fig, axes = plt.subplots(1, 2, figsize=(5.1, 2.25), gridspec_kw={"width_ratios": (1.2, 1.0)}, constrained_layout=True)
    ax = axes[0]
    for strategy in colors:
        rows = recovery[recovery["strategy"] == strategy]
        summary = rows.groupby("requested_fraction")["fidelity_recovery_vs_all_low"].agg(
            mean="mean", lower=lambda x: np.quantile(x, 0.25), upper=lambda x: np.quantile(x, 0.75)
        ).sort_index()
        x = summary.index.to_numpy(dtype=np.float64) * 100
        ax.plot(x, summary["mean"], color=colors[strategy], marker="o", markersize=2.7, label=labels[strategy])
        ax.fill_between(x, summary["lower"], summary["upper"], color=colors[strategy], alpha=0.07, linewidth=0)
    ax.axhline(0, color="#888888", linewidth=0.6)
    ax.set_xlabel("Patches at native precision (%)")
    ax.set_ylabel("Forecast-fidelity recovery")
    ax.set_xticks((12.5, 25, 37.5, 50))
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.2, columnspacing=0.6)
    panel_label(ax, "a", "Six registered selectors", -0.31)

    ax = axes[1]
    for model in MODELS:
        rows = recent[recent["model"] == model].sort_values("fraction")
        ax.plot(
            rows["fraction"] * 100,
            rows["fidelity_recovery_advantage"],
            color=MODEL_COLORS[model],
            marker="o",
            markersize=2.8,
            label=MODEL_LABELS[model],
        )
    ax.axhline(0, color="#777777", linewidth=0.65)
    ax.set_xlabel("Patches at native precision (%)")
    ax.set_ylabel("High-MI minus Recent recovery")
    ax.set_xticks((12.5, 25, 37.5, 50))
    ax.set_ylim(-0.16, 0.16)
    panel_label(ax, "b", "Same-layer Recent, ETTh1", -0.31)
    save_figure(fig, "Fig6_precision_recovery")


def supplementary_controls() -> None:
    recency = pd.read_csv(AUDIT / "topology_recency_audit.csv")
    recency["dataset"] = recency["dataset"].replace({"electricity": "Electricity", "traffic": "Traffic", "weather": "Weather"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for ax, column, label, panel in (
        (axes[0], "patch_mi_position_spearman", "MI-position Spearman", "a"),
        (axes[1], "top_250_recent_jaccard", "Top-25% / recent Jaccard", "b"),
    ):
        matrix = recency.pivot(index="model", columns="dataset", values=column).reindex(index=MODELS, columns=DATASETS).to_numpy()
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=-0.1 if panel == "a" else 0, vmax=1)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASET_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_ylabel(label)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=5, color="white" if matrix[row, col] < 0.35 else "black")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        panel_label(ax, panel, label, -0.39)
    save_figure(fig, "FigS_recency_audit")

    strata = pd.read_csv(AUDIT / "position_stratified_contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    columns = ("all_forecast_change_mse_high_minus_low_mean", "target_forecast_change_mse_high_minus_low_mean")
    finite = np.concatenate([strata[column].to_numpy(dtype=np.float64) for column in columns])
    limit = max(np.max(np.abs(finite)), 1e-3)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for ax, column, label, panel in zip(axes, columns, ("All-variable forecast", "Diagnostic target"), ("a", "b")):
        matrix = strata.pivot(index="model", columns="position_bin", values=column).reindex(index=MODELS, columns=(0, 1, 2, 3)).to_numpy()
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(range(4))
        ax.set_xticklabels(("Oldest", "Early", "Late", "Boundary"))
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
        ax.set_ylabel("High-minus-low change MSE")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        panel_label(ax, panel, label, -0.29)
    save_figure(fig, "FigS_position_stratified")

    dose = pd.read_csv(AUDIT / "interpolation_conditions.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.1), constrained_layout=True)
    for index, (ax, model) in enumerate(zip(axes.flat, MODELS)):
        rows = dose[dose["model"] == model]
        for rank, color, marker in (("high", HIGH, "o"), ("low", LOW, "s")):
            values = rows[rows["rank"] == rank].sort_values("alpha")
            ax.plot(values["alpha"], values["all_forecast_change_mse_mean"], color=color, marker=marker, markersize=3, label=f"{rank.capitalize()} MI")
            ax.fill_between(values["alpha"], values["all_forecast_change_mse_ci95_lower"], values["all_forecast_change_mse_ci95_upper"], color=color, alpha=0.12, linewidth=0)
        ax.set_xticks((0.25, 0.5, 1.0))
        ax.set_xlabel(r"Interpolation $\alpha$")
        ax.set_ylabel("Forecast-change MSE")
        panel_label(ax, chr(ord("a") + index), MODEL_LABELS[model], -0.30)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    save_figure(fig, "FigS_interpolation_dose")

    sensitivity = pd.read_csv(AUDIT / "cached_mi_sensitivity.csv")
    metrics = ("cell_z_spearman_with_primary", "patch_z_spearman_with_primary", "layer_profile_spearman_with_primary", "top_quarter_cell_jaccard_with_primary")
    labels = ("Cell-rank Spearman", "Patch-rank Spearman", "Layer-profile Spearman", "Top-quarter Jaccard")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    for ax, variant, title, panel in ((axes[0], "full_seasonal", "Seasonality-matched null", "a"), (axes[1], "dynamics_legal", "Dynamics-only target", "b")):
        rows = sensitivity[sensitivity["variant"] == variant].set_index("model").reindex(MODELS)
        x = np.arange(len(metrics), dtype=np.float64)
        for model in MODELS:
            ax.plot(x, rows.loc[model, list(metrics)].to_numpy(dtype=np.float64), color=MODEL_COLORS[model], marker="o", markersize=2.7, alpha=0.82)
        ax.axhline(0, color="#888888", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=31, ha="right")
        ax.set_ylabel("Agreement with primary atlas")
        ax.set_ylim(-0.55, 1.05)
        panel_label(ax, panel, title, -0.41)
    save_figure(fig, "FigS_cached_sensitivity")

    recent_precision = pd.read_csv(AUDIT / "precision_same_layer_recent_pairwise.csv")
    matrix = recent_precision.pivot(index="model", columns="fraction", values="fidelity_recovery_advantage").reindex(index=MODELS).to_numpy()
    limit = max(np.max(np.abs(matrix)), 0.02)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    ax.set_xticks(range(4))
    ax.set_xticklabels(("12.5%", "25%", "37.5%", "50%"))
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_xlabel("Patches retained at native precision")
    ax.set_ylabel("High-MI minus Recent recovery")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=5.2)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    save_figure(fig, "FigS_precision_recent")


def main() -> None:
    required = (
        AUDIT / "progressive_model_dataset_curves.csv",
        AUDIT / "interpolation_conditions.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    figure_4_functional_controls()
    figure_s_functional_compact()
    print(f"Wrote retained prediction-response figures to {OUT}")


if __name__ == "__main__":
    main()
