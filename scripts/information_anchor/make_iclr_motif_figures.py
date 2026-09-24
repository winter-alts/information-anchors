"""Create Nature-style history-motif figures from the audited summaries.

The script reads only the saved motif audit tables.  It does not rerun a model
or alter any experimental result.  PDF is used by the manuscript; SVG and
TIFF are emitted for editable and high-resolution figure workflows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

# Nature-style typography/export baseline.  The first font is the preferred
# journal font; the remaining entries make the asset reproducible on Linux.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7.5,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.55,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# Muted Nature-style palette: blue denotes positive high-MI enrichment, red
# denotes depletion, and grey denotes a control/invalid cell.
BLUE = "#0F4D92"
BLUE_MID = "#3775BA"
BLUE_SOFT = "#DCE8F4"
RED = "#B64342"
RED_SOFT = "#F3DEDC"
TEAL = "#42949E"
NEUTRAL_LIGHT = "#E8E8E8"
NEUTRAL_MID = "#858585"
NEUTRAL_DARK = "#3F3F3F"
BLACK = "#272727"
DIV_CMAP = LinearSegmentedColormap.from_list(
    "motif_diverging",
    [RED, RED_SOFT, "#FFFFFF", BLUE_SOFT, BLUE],
    N=256,
)
DIV_CMAP.set_bad(NEUTRAL_LIGHT)


CODE_OR_REPO_ROOT = Path(__file__).resolve().parents[2]
if CODE_OR_REPO_ROOT.name == "code" and CODE_OR_REPO_ROOT.parent.name == "artifact":
    ROOT = CODE_OR_REPO_ROOT.parent
    OUT = ROOT / "generated_figures"
    MOTIF_ROOT = ROOT / "tables/information_anchor_history_motif_matrix"
else:
    ROOT = CODE_OR_REPO_ROOT
    OUT = ROOT / "results/information_anchor_iclr_figures"
    MOTIF_ROOT = ROOT / "results/information_anchor_history_motif_matrix"

MODELS = ("Chronos2", "ChronosBolt", "Moirai2", "TimesFM2.5", "Toto2", "TTM")
MODEL_LABELS = {
    "Chronos2": "Chronos-2",
    "ChronosBolt": "Chronos-Bolt",
    "Moirai2": "Moirai-2",
    "TimesFM2.5": "TimesFM 2.5",
    "Toto2": "Toto-2",
    "TTM": "TTM-R2",
}
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic")
DATASET_LABELS = {
    "etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2",
    "weather": "Weather", "electricity": "Electricity", "traffic": "Traffic",
}
MOTIF_LABELS = {
    "rising_trend": "Rising trend",
    "falling_trend": "Falling trend",
    "volatile_burst": "Volatile burst",
    "smooth_segment": "Smooth segment",
    "turning_up": "Turning up",
    "turning_down": "Turning down",
    "level_jump": "Level jump",
    "local_peak": "Local peak",
    "local_trough": "Local trough",
    "cross_channel_slope_sync": "Cross-channel slope sync",
    "cross_channel_volatility_burst": "Cross-channel volatility",
}
MOTIF_ORDER = (
    "turning_down",
    "turning_up",
    "local_peak",
    "local_trough",
    "rising_trend",
    "falling_trend",
    "level_jump",
    "volatile_burst",
    "smooth_segment",
    "cross_channel_volatility_burst",
    "cross_channel_slope_sync",
)
MOTIF_FAMILY = {
    "turning_down": "Turns and extrema",
    "turning_up": "Turns and extrema",
    "local_peak": "Turns and extrema",
    "local_trough": "Turns and extrema",
    "rising_trend": "Trend and level",
    "falling_trend": "Trend and level",
    "level_jump": "Trend and level",
    "volatile_burst": "Local dynamics",
    "smooth_segment": "Local dynamics",
    "cross_channel_volatility_burst": "Cross-channel",
    "cross_channel_slope_sync": "Cross-channel",
}
FAMILY_ORDER = ("Turns and extrema", "Trend and level", "Local dynamics", "Cross-channel")


def save_pub_figure(fig: plt.Figure, filename: str) -> None:
    """Write editable vector and journal-resolution raster assets."""
    path = Path(filename)
    fig.savefig(path.with_suffix(".svg"), format="svg")
    fig.savefig(path.with_suffix(".pdf"), format="pdf")
    fig.savefig(path.with_suffix(".png"), format="png", dpi=300)
    fig.savefig(
        path.with_suffix(".tiff"), format="tiff", dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.15, 1.04, label, transform=ax.transAxes,
        fontsize=8.5, fontweight="bold", color=BLACK,
        ha="left", va="bottom",
    )


def _valid_flag(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(("true", "1"))


def load_recent() -> pd.DataFrame:
    data = pd.read_csv(MOTIF_ROOT / "motif_recent_control_by_run.csv")
    return data[_valid_flag(data["critic_valid"])].copy()


def load_direction() -> pd.DataFrame:
    data = pd.read_csv(MOTIF_ROOT / "motif_direction_by_run.csv")
    return data[_valid_flag(data["critic_valid"])].copy()


def _bounded_vmax(*matrices: pd.DataFrame, minimum: float = 4.0) -> float:
    values = np.concatenate([np.abs(matrix.to_numpy(dtype=float)).ravel() for matrix in matrices])
    values = values[np.isfinite(values)]
    return max(float(np.max(values)) if values.size else 0.0, minimum)


def figure_motif_alignment() -> None:
    """Main figure: complete vocabulary, reference sensitivity, and family summary."""
    position_control = load_recent()
    direction = load_direction()
    summary = pd.read_csv(MOTIF_ROOT / "motif_recent_control_summary.csv").set_index("motif")
    high_summary = pd.read_csv(MOTIF_ROOT / "motif_direction_summary.csv").set_index("motif")

    forest = position_control[position_control.motif.isin(MOTIF_ORDER)].copy()
    reference_values = pd.DataFrame(
        {
            "All history": [high_summary.loc[m, "mean_high_delta_valid"] * 100 for m in MOTIF_ORDER],
            "Final quarter": [summary.loc[m, "mean_high_minus_recent"] * 100 for m in MOTIF_ORDER],
        }, index=MOTIF_ORDER,
    )
    support_values = pd.DataFrame(
        {
            "All history": [high_summary.loc[m, "positive_high_valid_runs"] / 36 for m in MOTIF_ORDER],
            "Final quarter": [summary.loc[m, "positive_valid_runs"] / 36 for m in MOTIF_ORDER],
        }, index=MOTIF_ORDER,
    )
    family_effect = pd.DataFrame(index=FAMILY_ORDER, columns=reference_values.columns, dtype=float)
    family_support = pd.DataFrame(index=FAMILY_ORDER, columns=reference_values.columns, dtype=float)
    for family in FAMILY_ORDER:
        motifs = [motif for motif in MOTIF_ORDER if MOTIF_FAMILY[motif] == family]
        family_effect.loc[family] = reference_values.loc[motifs].mean(axis=0)
        family_support.loc[family] = support_values.loc[motifs].mean(axis=0)
    vmax = _bounded_vmax(reference_values, minimum=4.0)

    fig = plt.figure(figsize=(7.2, 4.25))
    grid = fig.add_gridspec(1, 3, width_ratios=(2.75, 1.18, 1.55), wspace=0.72)

    # (a) Each point is one critic-valid model--dataset combination.
    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(MOTIF_ORDER))[::-1]
    rng = np.random.default_rng(7)
    for idx, motif in enumerate(MOTIF_ORDER):
        values = forest.loc[forest.motif == motif, "high_minus_recent"].to_numpy(float) * 100.0
        yy = np.full(values.size, y[idx], dtype=float) + rng.uniform(-0.10, 0.10, values.size)
        ax.scatter(
            values, yy, s=7, color=np.where(values >= 0, BLUE_MID, RED),
            alpha=0.28, linewidths=0, zorder=1,
        )
        lo, hi = np.quantile(values, [0.025, 0.975])
        mean = float(np.mean(values))
        ax.plot([lo, hi], [y[idx], y[idx]], color=NEUTRAL_DARK, lw=1.0, zorder=2)
        ax.scatter(
            [mean], [y[idx]], s=26, color=BLUE if mean >= 0 else RED,
            edgecolors="white", linewidths=0.7, zorder=3,
        )
        support = int(summary.loc[motif, "positive_valid_runs"])
        ax.text(10.3, y[idx], f"{support}/36", va="center", ha="left", fontsize=5.5, color=NEUTRAL_DARK)
    ax.axvline(0, color=NEUTRAL_MID, lw=0.7, ls=(0, (2.0, 2.0)), zorder=0)
    ax.set_yticks(y, [MOTIF_LABELS[m] for m in MOTIF_ORDER])
    ax.set_xlabel("High-MI − final quarter (percentage points)", labelpad=4)
    ax.set_xlim(-8.0, 12.0)
    ax.set_ylim(-0.65, len(MOTIF_ORDER) - 0.35)
    ax.grid(axis="x", color=NEUTRAL_LIGHT, lw=0.45)
    ax.set_axisbelow(True)
    ax.text(
        0.0, 1.045, "Central 95% run spread", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=6.1, color=NEUTRAL_MID,
    )
    ax.text(
        1.0, 1.045, "supporting runs", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.1, color=NEUTRAL_MID,
    )
    panel_label(ax, "a")

    # (b) Two history references expose reference-sensitive signs.
    ax = fig.add_subplot(grid[0, 1])
    im = ax.imshow(
        reference_values.to_numpy(dtype=float), aspect="auto", cmap=DIV_CMAP,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), interpolation="nearest",
    )
    ax.set_xticks((0, 1), ("All\nhistory", "Final\nquarter"))
    heatmap_labels = {
        "cross_channel_volatility_burst": "CC volatility",
        "cross_channel_slope_sync": "CC slope sync",
    }
    ax.set_yticks(
        range(len(MOTIF_ORDER)),
        [heatmap_labels.get(m, MOTIF_LABELS[m]) for m in MOTIF_ORDER],
    )
    ax.tick_params(length=0, labelsize=5.2)
    for row, motif in enumerate(MOTIF_ORDER):
        for col, reference in enumerate(reference_values.columns):
            value = float(reference_values.loc[motif, reference])
            support = int(round(float(support_values.loc[motif, reference]) * 36))
            color = "white" if abs(value) > vmax * 0.50 else BLACK
            ax.text(col, row, f"{value:+.1f}\n{support}/36", ha="center", va="center",
                    fontsize=4.5, color=color, linespacing=0.9)
    panel_label(ax, "b")
    ax.set_title("Reference audit", loc="left", pad=4, fontsize=7.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.10, pad=0.08, aspect=18)
    cbar.set_label("Enrichment (pp)", fontsize=5.8, labelpad=2)
    cbar.ax.tick_params(labelsize=5.2, length=2)
    cbar.outline.set_linewidth(0.45)

    # (c) Family means avoid treating eleven correlated labels as independent results.
    ax = fig.add_subplot(grid[0, 2])
    offsets = {"All history": -0.12, "Final quarter": 0.12}
    colors = {"All history": NEUTRAL_MID, "Final quarter": BLUE}
    y_family = np.arange(len(FAMILY_ORDER))[::-1]
    for reference in reference_values.columns:
        x_values = family_effect.loc[list(FAMILY_ORDER), reference].to_numpy(float)
        support = family_support.loc[list(FAMILY_ORDER), reference].to_numpy(float)
        ax.scatter(x_values, y_family + offsets[reference], s=25 + 75 * support,
                   color=colors[reference], edgecolors="white", linewidths=0.65,
                   label=reference, zorder=3)
    ax.axvline(0, color=NEUTRAL_MID, lw=0.7, ls=(0, (2.0, 2.0)), zorder=0)
    ax.set_yticks(y_family, FAMILY_ORDER)
    ax.set_xlabel("Family mean (pp)")
    ax.set_xlim(-2.7, 4.3)
    ax.set_ylim(-0.65, len(FAMILY_ORDER) - 0.35)
    ax.grid(axis="x", color=NEUTRAL_LIGHT, lw=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=5.2, handletextpad=0.4)
    ax.set_title("Structure families", loc="left", pad=4, fontsize=7.5, fontweight="bold")
    panel_label(ax, "c")

    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.16, top=0.91)
    save_pub_figure(fig, str(OUT / "Fig5_motif_alignment"))
    plt.close(fig)


def figure_motif_atlas() -> None:
    recent = load_recent()
    direction = load_direction()
    all_motifs = list(MOTIF_ORDER)
    high = (
        direction.groupby(["model", "motif"], observed=True)["high_delta"]
        .mean().unstack("motif").reindex(index=MODELS, columns=all_motifs)
    ) * 100.0
    control = (
        recent.groupby(["model", "motif"], observed=True)["high_minus_recent"]
        .mean().unstack("motif").reindex(index=MODELS, columns=all_motifs)
    ) * 100.0
    # A robust range keeps the dense atlas readable while preserving the full
    # values in the source table; the isolated extreme cells remain clipped at
    # the color scale boundary rather than flattening the complete atlas.
    atlas_abs = np.concatenate([
        np.abs(high.to_numpy(dtype=float)).ravel(),
        np.abs(control.to_numpy(dtype=float)).ravel(),
    ])
    atlas_abs = atlas_abs[np.isfinite(atlas_abs)]
    vmax = max(float(np.nanpercentile(atlas_abs, 97.5)) if atlas_abs.size else 0.0, 6.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.55), sharex=True)
    matrices = (high, control)
    titles = ("High-MI vs all history patches", "High-MI vs final-quarter patches")
    for idx, (ax, matrix, title) in enumerate(zip(axes, matrices, titles)):
        im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=DIV_CMAP,
                       norm=norm, interpolation="nearest")
        ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
        ax.tick_params(axis="y", labelsize=5.8, length=0)
        ax.tick_params(axis="x", length=0)
        panel_label(ax, "ab"[idx])
        ax.set_title(title, loc="left", pad=4, fontsize=7.5, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        if idx == 1:
            short_labels = {
                "cross_channel_slope_sync": "CC slope\nsync",
                "cross_channel_volatility_burst": "CC vol.",
            }
            labels = [short_labels.get(m, MOTIF_LABELS[m].replace(" ", "\n", 1)) for m in all_motifs]
            ax.set_xticks(range(len(all_motifs)), labels, fontsize=5.15)
            ax.tick_params(axis="x", pad=3)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, aspect=18)
    cbar.set_label("Enrichment (percentage points)", fontsize=6.2, labelpad=3)
    cbar.ax.tick_params(labelsize=5.2, length=2)
    cbar.outline.set_linewidth(0.45)
    fig.subplots_adjust(left=0.18, right=0.84, bottom=0.23, top=0.91, hspace=0.58)
    save_pub_figure(fig, str(OUT / "FigS_motif_atlas"))
    plt.close(fig)


def figure_motif_definitions() -> None:
    definitions = (
        ("Turning down", "rise then decline", BLUE_MID),
        ("Turning up", "decline then rise", TEAL),
        ("Local peak", "interior maximum", RED),
        ("Local trough", "interior minimum", BLUE),
        ("Rising trend", "positive robust slope", BLUE),
        ("Falling trend", "negative robust slope", RED),
        ("Volatile burst", "large local variation", TEAL),
        ("Smooth segment", "low slope and roughness", NEUTRAL_MID),
        ("Cross-channel volatility", "shared variability burst", BLUE_MID),
    )
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 3.6), sharex=True, sharey=True)
    x = np.linspace(0, 1, 64)
    curves = (
        0.18 + 0.75 * (1 - np.exp(-5 * x)) - 0.46 * np.maximum(x - 0.62, 0),
        0.82 - 0.68 * (1 - np.exp(-5 * x)) + 0.46 * np.maximum(x - 0.62, 0),
        0.12 + 0.7 * np.exp(-((x - 0.52) / 0.17) ** 2),
        0.88 - 0.7 * np.exp(-((x - 0.52) / 0.17) ** 2),
        0.15 + 0.75 * x,
        0.90 - 0.75 * x,
        0.38 + 0.08 * np.sin(7 * x) + 0.28 * np.sin(27 * x) * np.exp(-((x - 0.63) / 0.25) ** 2),
        0.48 + 0.05 * np.sin(2 * np.pi * x),
        0.40 + 0.08 * np.sin(7 * x) + 0.24 * np.sin(25 * x) * np.exp(-((x - 0.62) / 0.24) ** 2),
    )
    for ax, (name, description, color), curve in zip(axes.flat, definitions, curves):
        ax.plot(x, curve, color=color, lw=1.55)
        ax.fill_between(x, curve, 0.08, color=color, alpha=0.10)
        ax.set_title(name, fontsize=6.8, fontweight="bold", pad=3)
        ax.set_xticks((0, 1), ("start", "end"), fontsize=5.0)
        ax.tick_params(axis="y", length=0, labelleft=False)
        ax.tick_params(axis="x", length=2, pad=1.5)
        ax.text(0.5, -0.28, description, transform=ax.transAxes,
                ha="center", va="top", fontsize=5.0, color=NEUTRAL_MID)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", color=NEUTRAL_LIGHT, lw=0.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("normalized value", fontsize=6.0, labelpad=3)
    fig.text(0.52, 0.02, "Within-patch time", ha="center", va="bottom", fontsize=6.3)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.93, wspace=0.30, hspace=0.72)
    save_pub_figure(fig, str(OUT / "FigS_motif_definitions"))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_motif_alignment()
    figure_motif_atlas()
    figure_motif_definitions()


if __name__ == "__main__":
    main()
