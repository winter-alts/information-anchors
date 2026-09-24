#!/usr/bin/env python3
"""Create protocol-locked publication figures for the V6 information-anchor study."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import SymLogNorm, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/information_anchor_v6_figures"
TOPOLOGY = ROOT / "results/information_anchor_v6_topology"
EVIDENCE = ROOT / "results/information_anchor_v6_evidence"
ROBUSTNESS = ROOT / "results/information_anchor_v6_robustness"
AUDIT = ROOT / "results/information_anchor_v6_matrix_audit/matrix.json"
PRECISION = ROOT / "results/information_anchor_mi_control_baseline_evidence"

MODELS = ("Chronos2", "Moirai2", "Toto2", "TimesFM2.5", "ChronosBolt", "TTM")
MODEL_LABELS = {
    "Chronos2": "Chronos-2",
    "Moirai2": "Moirai-2",
    "Toto2": "Toto-2",
    "TimesFM2.5": "TimesFM 2.5",
    "ChronosBolt": "Chronos-Bolt",
    "TTM": "TTM-R2",
}
COLORS = {
    "Chronos2": "#0F4D92",
    "Moirai2": "#42949E",
    "Toto2": "#B64342",
    "TimesFM2.5": "#9A4D8E",
    "ChronosBolt": "#C28C2C",
    "TTM": "#4D4D4D",
}
SEMANTICS = (
    "level_q1",
    "level_q2",
    "level_q3",
    "level_q4",
    "global_change",
    "linear_trend",
    "mean_absolute_change",
    "diff_std",
    "roughness",
    "low_frequency_energy",
    "mid_frequency_energy",
    "high_frequency_energy",
)
SEMANTIC_LABELS = (
    "Level\nQ1",
    "Level\nQ2",
    "Level\nQ3",
    "Level\nQ4",
    "Global\nchange",
    "Linear\ntrend",
    "Mean |Δ|",
    "SD(Δ)",
    "Roughness",
    "Low-freq.\nenergy",
    "Mid-freq.\nenergy",
    "High-freq.\nenergy",
)
GROUPS = ("level", "trajectory", "local_dynamics", "frequency")
GROUP_LABELS = ("Level", "Trajectory", "Local dynamics", "Frequency")


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


def _save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(OUTPUT / f"{stem}.{extension}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _panel(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _clean_semantic(value: str) -> str:
    return value.split(":")[-1]


def _format_probe_colorbar(cbar: mpl.colorbar.Colorbar, vmin: float, vmax: float) -> None:
    """为非线性 probe 色标使用少量可直接阅读的 R2 刻度。"""
    ticks = [vmin]
    if vmin < -1.0:
        ticks.append(-1.0)
    ticks.extend([-0.1, 0.0, 0.1, vmax])
    unique_ticks = np.asarray(sorted(set(round(float(value), 6) for value in ticks)))
    cbar.set_ticks(unique_ticks)
    labels = [f"{value:.2f}".rstrip("0").rstrip(".") if abs(value) >= 1 else f"{value:.2g}"
              for value in unique_ticks]
    cbar.set_ticklabels(labels)
    cbar.ax.minorticks_off()


def _interp_profile(x: np.ndarray, y: np.ndarray, points: int = 64) -> np.ndarray:
    grid = (np.arange(points, dtype=np.float64) + 0.5) / points
    return np.interp(grid, np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))


def figure_1_framework() -> None:
    """绘制包含数据隔离、模型原生单元和三条证据链的详细流程图。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 顶部先固定研究对象，避免读者把 target probe 误解为单变量预测。
    ax.text(0.02, 0.965, "Registered multivariate forecast object", fontsize=9.2, fontweight="bold", va="top")
    ax.text(0.02, 0.918, r"$X_\tau\in\mathbb{R}^{L\times C}$  →  frozen $f_m$  →  $\widehat F_\tau\in\mathbb{R}^{P\times C}$",
            fontsize=8.2, va="top")
    ax.text(0.02, 0.875, "Seven registered multivariate regimes under one matched six-model protocol",
            fontsize=6.7, color="#555555", va="top")

    stage_x = (0.02, 0.265, 0.51, 0.755)
    stage_w = (0.205, 0.205, 0.205, 0.225)
    fills = ("#E8EEF6", "#E7F2F1", "#F4E9EE", "#F5EEDC")
    titles = ("1  Native units", "2  MI atlas", "3  Probe readout", "4  Donor intervention")
    body = (
        "Reachable-layer hooks\n$H_{\\tau,l,j,:}$ keeps native\npatch/channel structure",
        "PCA-8 hidden view\nvs. all-variable future\nKSG $k=5$; 199 nulls\nraw MI, $z$, $p$, BH $q$",
        "Select $l^*$ on discovery\nFit ridge $\\alpha$ on validation\nTest 12 semantics\n(global and target scopes)",
        "Top / bottom / random sets\n8 donors x 128 origins\n12.5--50% patches\nall-variable change MSE",
    )
    for index, (x, width, fill, title, text) in enumerate(zip(stage_x, stage_w, fills, titles, body)):
        rect = patches.FancyBboxPatch(
            (x, 0.445), width, 0.32,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=fill, edgecolor="#555555", linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(x + 0.015, 0.742, title, fontsize=7.6, fontweight="bold", va="top")
        ax.text(x + 0.015, 0.695, text, fontsize=5.9 if index == 2 else 6.2,
                va="top", linespacing=1.22)
        if index == 0:
            # 用三个稳定 patch 方块表示原生时序单元，避免曲线与定义文字混叠。
            for patch_index in range(4):
                ax.add_patch(patches.Rectangle((x + 0.024 + patch_index * 0.042, 0.475), 0.032, 0.032,
                                               facecolor=COLORS[MODELS[patch_index]], alpha=0.72,
                                               edgecolor="white", linewidth=0.5))
        elif index == 1:
            matrix = np.asarray([[0.15, 0.45, 0.75, 0.35], [0.28, 0.70, 0.42, 0.58], [0.62, 0.40, 0.32, 0.82]])
            ax.imshow(matrix, cmap="magma", extent=(x + 0.025, x + width - 0.025, 0.472, 0.535),
                      aspect="auto", origin="lower", vmin=0, vmax=1)
        elif index == 2:
            ax.text(x + 0.018, 0.492, "global", fontsize=5.8, color="#0F4D92", va="center")
            ax.text(x + width - 0.018, 0.492, "target", fontsize=5.8, color="#B64342", va="center", ha="right")
        else:
            for offset, color in ((0.0, "#B64342"), (0.052, "#0F4D92"), (0.104, "#8F8F8F")):
                ax.plot([x + 0.035 + offset, x + 0.065 + offset], [0.49, 0.49], color=color, lw=3.0)
    for left, right, width in zip(stage_x[:-1], stage_x[1:], stage_w[:-1]):
        ax.annotate("", xy=(right - 0.007, 0.605), xytext=(left + width + 0.006, 0.605),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555555"))

    # 底部的 split ledger 是图中最重要的防泄漏说明。
    ax.text(0.02, 0.365, "Data isolation and selection ledger", fontsize=8.5, fontweight="bold", va="top")
    ledger = [
        (0.02, 0.240, 0.285, 0.090, "Discovery", "PCA, MI, null atlas,\nMI-top cell and $l^*$", "#DCE8F5"),
        (0.325, 0.240, 0.285, 0.090, "Validation", "Ridge hyperparameter only;\nselection is already frozen", "#E2EFEA"),
        (0.630, 0.240, 0.35, 0.090, "Test", "Probe $R^2$, donor replacement,\nall-variable MSE/MAE", "#F4E3E1"),
    ]
    for x, y, w, h, title, text, fill in ledger:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.009,rounding_size=0.008",
                                      facecolor=fill, edgecolor="#777777", linewidth=0.65)
        ax.add_patch(rect)
        ax.text(x + 0.012, y + h - 0.014, title, fontsize=6.7, fontweight="bold", va="top")
        ax.text(x + 0.012, y + h - 0.036, text, fontsize=5.6, va="top", linespacing=1.15)
    ax.text(0.5, 0.185, "Three legs are complementary: MI = statistical dependence; probe = linear accessibility; intervention = forecast sensitivity.",
            ha="center", va="center", fontsize=7.0, fontweight="bold")
    ax.text(0.5, 0.135, "Cross-model comparison uses within-run normalized topology descriptors; raw MI magnitudes are never ranked across models.",
            ha="center", va="center", fontsize=6.5, color="#555555")
    ax.text(0.5, 0.070, "Information anchor claim: localization + semantic accessibility + functional involvement, with explicit scaling limits.",
            ha="center", va="center", fontsize=8.0, fontweight="bold")
    _save(fig, "Fig1_framework")


def figure_2_topology() -> None:
    temporal = pd.read_csv(TOPOLOGY / "temporal_profiles.csv")
    layers = pd.read_csv(TOPOLOGY / "layer_profiles.csv")
    grid = (np.arange(64) + 0.5) / 64
    temporal_profiles: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    layer_profiles: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    for (model, _dataset), group in temporal.groupby(["model", "dataset"], sort=False):
        values = _interp_profile(group.normalized_center.to_numpy(), group.temporal_mass.to_numpy())
        temporal_profiles[model].append(values / values.sum() * 64)
    for (model, _dataset), group in layers.groupby(["model", "dataset"], sort=False):
        values = _interp_profile(group.normalized_depth.to_numpy(), group.layer_mass.to_numpy())
        layer_profiles[model].append(values / values.sum() * 64)

    fig = plt.figure(figsize=(7.2, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    for model in MODELS:
        matrix = np.stack(temporal_profiles[model])
        mean = matrix.mean(axis=0)
        ax.plot(grid, mean, lw=1.6, color=COLORS[model], label=MODEL_LABELS[model])
        ax.fill_between(
            grid,
            np.percentile(matrix, 25, axis=0),
            np.percentile(matrix, 75, axis=0),
            color=COLORS[model],
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("Normalized history position  →  forecast boundary")
    ax.set_ylabel("Relative temporal information mass")
    ax.set_xlim(0, 1)
    ax.legend(ncol=2, fontsize=6.5, loc="upper left")
    _panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    matrix = np.stack([np.stack(layer_profiles[model]).mean(axis=0) for model in MODELS])
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.55, vmax=1.55)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODELS])
    ticks = np.asarray([0, 16, 32, 48, 63])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    ax.set_xlabel("Normalized layer depth")
    cbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Relative layer information mass", fontsize=7)
    _panel(ax, "b", x=-0.22)
    _save(fig, "Fig2_topology")

    descriptors = pd.read_csv(TOPOLOGY / "topology_descriptors.csv")
    fig, ax = plt.subplots(figsize=(4.0, 3.35))
    for model in MODELS:
        subset = descriptors[descriptors.model == model]
        ax.scatter(
            subset.mean_boundary_distance,
            subset.top_quarter_concentration,
            s=22,
            color=COLORS[model],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.35,
        )
        ax.scatter(
            subset.mean_boundary_distance.mean(),
            subset.top_quarter_concentration.mean(),
            s=55,
            color=COLORS[model],
            edgecolor="#222222",
            linewidth=0.6,
            label=MODEL_LABELS[model],
        )
    ax.set_xlabel("Mean distance from forecast boundary")
    ax.set_ylabel("Top-quarter concentration")
    ax.legend(fontsize=6.5, ncol=2)
    _save(fig, "Fig2_topology_descriptors")


def _semantic_matrices(scope: str) -> tuple[np.ndarray, np.ndarray]:
    semantic = pd.read_csv(EVIDENCE / "probe_semantics.csv")
    semantic = semantic[semantic.scope == scope].copy()
    semantic["base_semantic"] = semantic.semantic.map(_clean_semantic)
    matrix = np.empty((len(MODELS), len(SEMANTICS)), dtype=np.float64)
    for row_index, model in enumerate(MODELS):
        subset = semantic[semantic.model == model]
        for column_index, name in enumerate(SEMANTICS):
            matrix[row_index, column_index] = subset.loc[
                subset.base_semantic == name, "functional_top_r2"
            ].mean()
    groups = pd.read_csv(EVIDENCE / "probe_semantic_groups.csv")
    groups = groups[groups.scope == scope]
    difference = np.empty((len(MODELS), len(GROUPS)), dtype=np.float64)
    for row_index, model in enumerate(MODELS):
        for column_index, group in enumerate(GROUPS):
            difference[row_index, column_index] = groups.loc[
                (groups.model == model) & (groups.semantic_group == group),
                "functional_top_minus_low_r2",
            ].mean()
    return matrix, difference


def _semantic_figure(scope: str, stem: str, title: str) -> None:
    """绘制完整语义读出矩阵；非线性色标保留全部负 R2，而不使用截断符号。"""
    matrix, difference = _semantic_matrices(scope)
    vmin = min(-0.40, float(np.nanmin(matrix)))
    vmax = max(0.65, float(np.nanmax(matrix)))
    fig = plt.figure(figsize=(7.2, 3.75))
    gs = fig.add_gridspec(1, 2, width_ratios=(2.7, 1.0), wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    norm = SymLogNorm(linthresh=0.05, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODELS])
    ax.set_xticks(range(len(SEMANTICS)))
    ax.set_xticklabels(SEMANTIC_LABELS, rotation=38, ha="right", fontsize=6.5)
    ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.018)
    _format_probe_colorbar(cbar, vmin, vmax)
    cbar.set_label("Held-out test R²", fontsize=7)
    ax.set_title(title, fontsize=8.5, loc="left", pad=8)
    _panel(ax, "a", x=-0.08)

    ax = fig.add_subplot(gs[0, 1])
    limit = max(0.12, float(np.nanmax(np.abs(difference))) * 1.05)
    image = ax.imshow(
        difference,
        cmap="PuOr_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([])
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUP_LABELS, rotation=40, ha="right", fontsize=6.5)
    ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.07, pad=0.04)
    cbar.set_label("Top-MI − bottom-MI R²", fontsize=7)
    _panel(ax, "b", x=-0.12)
    _save(fig, stem)


def figure_3_probe_semantics_combined() -> None:
    """把 global/target 的全部十二项语义压缩成一张主文可读图。"""
    global_matrix, global_difference = _semantic_matrices("global")
    target_matrix, target_difference = _semantic_matrices("target")
    semantic_values = np.concatenate([global_matrix.ravel(), target_matrix.ravel()])
    vmin = min(-0.40, float(np.nanmin(semantic_values)))
    vmax = max(0.65, float(np.nanmax(semantic_values)))
    # 目标语义中的局部动力学 R2 可低至约 -4.7。对称对数色标保留真实
    # 负值，同时仍分辨零附近及正值差异，避免用截断符号替代数据。
    norm = SymLogNorm(linthresh=0.05, linscale=1.0, vmin=vmin, vmax=vmax, base=10)
    diff_limit = max(
        0.12,
        float(np.nanmax(np.abs(np.concatenate([global_difference.ravel(), target_difference.ravel()])))) * 1.05,
    )
    diff_norm = TwoSlopeNorm(vmin=-diff_limit, vcenter=0.0, vmax=diff_limit)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.65),
                             gridspec_kw={"height_ratios": (1.38, 0.95), "hspace": 0.72, "wspace": 0.34})
    for ax, matrix, title, panel in (
        (axes[0, 0], global_matrix, "Global future semantics", "a"),
        (axes[0, 1], target_matrix, "Target-channel future semantics", "b"),
    ):
        image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_title(title, fontsize=8.0, pad=4, loc="left")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=6.1)
        ax.set_xticks(range(len(SEMANTICS)))
        ax.set_xticklabels(SEMANTIC_LABELS, rotation=42, ha="right", fontsize=5.1)
        ax.tick_params(length=0)
        _panel(ax, panel, x=-0.12, y=1.04)
    top_image = axes[0, 0].images[0]
    top_cbar = fig.colorbar(top_image, ax=axes[0, :].tolist(), fraction=0.025, pad=0.018)
    _format_probe_colorbar(top_cbar, vmin, vmax)
    top_cbar.set_label("Held-out test $R^2$")
    for ax, difference, title, panel in (
        (axes[1, 0], global_difference, "Global: top-MI minus low-MI", "c"),
        (axes[1, 1], target_difference, "Target: top-MI minus low-MI", "d"),
    ):
        image = ax.imshow(difference, cmap="PuOr_r", norm=diff_norm, aspect="auto")
        ax.set_title(title, fontsize=7.4, pad=4, loc="left")
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=6.0)
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUP_LABELS, rotation=34, ha="right", fontsize=5.8)
        ax.tick_params(length=0)
        _panel(ax, panel, x=-0.12, y=1.04)
    bottom_image = axes[1, 0].images[0]
    fig.colorbar(bottom_image, ax=axes[1, :].tolist(), fraction=0.025, pad=0.018,
                 label="Top-MI minus low-MI $R^2$")
    fig.suptitle("Two probe scopes, all twelve future semantics", fontsize=9.0, y=0.995)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    _save(fig, "Fig3_probe_semantics")


def figure_4_target_depth() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    grid = (np.arange(64) + 0.5) / 64
    profiles: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
    for record in audit["records"]:
        run_dir = record.get("probe_target_dir")
        if not run_dir:
            continue
        arrays = np.load(Path(run_dir) / "probe_results.npz")
        r2 = np.asarray(arrays["test_r2"], dtype=np.float64)
        # Coarse, forecastable semantics: four levels plus change and trend.
        layer_values = np.nanmean(r2[:, :, :6], axis=(1, 2))
        centers = (np.arange(len(layer_values)) + 0.5) / len(layer_values)
        profiles[record["model"]].append(_interp_profile(centers, layer_values))

    fig, ax = plt.subplots(figsize=(5.0, 3.35))
    for model in MODELS:
        matrix = np.stack(profiles[model])
        ax.plot(grid, matrix.mean(axis=0), lw=1.6, color=COLORS[model], label=MODEL_LABELS[model])
        ax.fill_between(
            grid,
            np.percentile(matrix, 25, axis=0),
            np.percentile(matrix, 75, axis=0),
            color=COLORS[model],
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalized layer depth")
    ax.set_ylabel("Target level/trajectory test R²")
    ax.legend(ncol=2, fontsize=6.5)
    _save(fig, "Fig4_target_semantic_depth")


def _functional_figure(mode: str, stem: str, heading: str) -> None:
    conditions = pd.read_csv(EVIDENCE / "progressive_conditions.csv")
    subset = conditions[(conditions.dataset_key == "ettm1") & (conditions["mode"] == mode)]
    missing = [model for model in MODELS if model not in set(subset.model)]
    if missing:
        raise RuntimeError(f"Missing ETTm1 progressive results for {missing}.")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.65), sharex=True)
    strategies = ("top", "bottom", "random")
    strategy_labels = ("Top MI", "Bottom MI", "Random mean")
    strategy_colors = ("#B64342", "#0F4D92", "#8F8F8F")
    for ax, model in zip(axes.flat, MODELS):
        model_data = subset[subset.model == model]
        for strategy, label, color in zip(strategies, strategy_labels, strategy_colors):
            values = model_data[model_data.strategy == strategy]
            if strategy == "random":
                grouped = values.groupby("fraction").all_forecast_change_mse_mean
                x = np.asarray(sorted(grouped.groups), dtype=np.float64)
                y = grouped.mean().reindex(x).to_numpy()
                low = grouped.min().reindex(x).to_numpy()
                high = grouped.max().reindex(x).to_numpy()
                ax.fill_between(x * 100, low, high, color=color, alpha=0.12, linewidth=0)
            else:
                values = values.sort_values("fraction")
                x = values.fraction.to_numpy(dtype=np.float64)
                y = values.all_forecast_change_mse_mean.to_numpy(dtype=np.float64)
                low = values.all_forecast_change_mse_ci95_lower.to_numpy(dtype=np.float64)
                high = values.all_forecast_change_mse_ci95_upper.to_numpy(dtype=np.float64)
                ax.fill_between(x * 100, low, high, color=color, alpha=0.10, linewidth=0)
            ax.plot(x * 100, y, color=color, lw=1.4, marker="o", ms=3, label=label)
        ax.set_title(MODEL_LABELS[model], fontsize=8.2, pad=4)
        ax.set_xticks((12.5, 25, 37.5, 50))
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("Selected history patches (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Forecast-change MSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), fontsize=7)
    fig.suptitle(heading, fontsize=9, y=1.055)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    _save(fig, stem)


def figure_6_robustness() -> None:
    length = pd.read_csv(ROBUSTNESS / "length_robustness.csv")
    sensitivity = pd.read_csv(ROBUSTNESS / "estimator_sensitivity.csv")
    stress = pd.read_csv(ROBUSTNESS / "full_variable_stress.csv")
    fig = plt.figure(figsize=(7.2, 3.65))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.35, 1.05), wspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    summary = length.groupby("model")[["patch_profile_spearman", "layer_profile_spearman"]].mean()
    y = np.arange(len(summary))
    for index, (model, row) in enumerate(summary.iterrows()):
        ax.plot(
            [row.patch_profile_spearman, row.layer_profile_spearman],
            [index, index],
            color="#B8B8B8",
            lw=1.0,
        )
        ax.scatter(row.patch_profile_spearman, index, color="#0F4D92", s=24, label="Temporal" if index == 0 else None)
        ax.scatter(row.layer_profile_spearman, index, color="#B64342", s=24, label="Layer" if index == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(model, model) for model in summary.index])
    ax.set_xlim(-0.05, 1.02)
    ax.set_xlabel("Rank correlation\nL512/P96 vs L336/P192")
    ax.legend(fontsize=6.5, loc="lower right")
    _panel(ax, "a", x=-0.25)

    ax = fig.add_subplot(gs[0, 1])
    variants = ("k3", "k10", "pca4", "pca16", "null99", "null399")
    labels = ("k=3", "k=10", "PCA=4", "PCA=16", "Null=99", "Null=399")
    data = [sensitivity.loc[sensitivity.variant == variant, "atlas_spearman"].to_numpy() for variant in variants]
    box = ax.boxplot(
        data,
        positions=np.arange(len(data)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#272727", lw=1.0),
        whiskerprops=dict(color="#767676", lw=0.8),
        capprops=dict(color="#767676", lw=0.8),
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#B4C0E4")
        patch.set_edgecolor("#484878")
        patch.set_linewidth(0.8)
    for index, values in enumerate(data):
        offsets = np.linspace(-0.14, 0.14, len(values))
        ax.scatter(np.full(len(values), index) + offsets, values, color="#484878", s=9, alpha=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Atlas rank correlation")
    _panel(ax, "b", x=-0.18)

    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(stress))
    bar_colors = [COLORS[model] for model in stress.model]
    ax.bar(x, stress.atlas_spearman, color=bar_colors, width=0.72, edgecolor="white", linewidth=0.4)
    labels = [f"{MODEL_LABELS[model]}\n{dataset.capitalize()}" for model, dataset in zip(stress.model, stress.dataset)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=6.2)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_ylim(-0.1, 1.02)
    ax.set_ylabel("Registered vs full atlas\nrank correlation")
    _panel(ax, "c", x=-0.22)
    # 稳健性属于补充证据，避免与主文的架构综合图共享图号。
    _save(fig, "FigS1_robustness")


def figure_6_architecture_synthesis() -> None:
    topology = pd.read_csv(TOPOLOGY / "topology_descriptors.csv")
    probes = pd.read_csv(EVIDENCE / "probe_selected_units.csv")
    conditions = pd.read_csv(EVIDENCE / "progressive_conditions.csv")
    baselines = pd.read_csv(EVIDENCE / "forecast_baselines.csv")
    rows = []
    for model in MODELS:
        model_topology = topology[topology.model == model]
        model_probe = probes[probes.model == model]
        intervention = conditions[
            (conditions.model == model)
            & (conditions.dataset_key == "ettm1")
            & (conditions["mode"] == "remove")
            & (conditions.strategy == "top")
            & np.isclose(conditions.fraction, 0.25)
        ]
        baseline = baselines[(baselines.model == model) & (baselines.dataset_key == "ettm1")]
        rows.append(
            {
                "model": model,
                "Temporal concentration": model_topology.top_quarter_concentration.mean(),
                "Boundary locality": 1 - model_topology.mean_boundary_distance.mean(),
                "Layer depth": model_topology.expected_layer_depth.mean(),
                "MI–probe alignment": model_probe.mi_vs_probe_spearman.mean(),
                "Functional sensitivity": float(intervention.all_forecast_change_mse_mean.iloc[0] / baseline.baseline_all_mse.iloc[0]),
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    standardized = (table - table.mean(axis=0)) / table.std(axis=0, ddof=0)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    limit = float(np.max(np.abs(standardized.to_numpy())))
    image = ax.imshow(
        standardized.to_numpy(),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODELS], fontsize=6.2)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=42, ha="right", fontsize=5.8)
    ax.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=5.5, length=2)
    cbar.set_label("Within-descriptor z score", fontsize=5.8)
    ax.set_title("Descriptive architecture-conditioned evidence profile", loc="left", fontsize=7.2)
    fig.tight_layout(pad=0.35)
    _save(fig, "Fig6_architecture_synthesis")


def figure_5_precision_application() -> None:
    """Plot forecast-fidelity recovery under the registered precision budgets."""
    recovery = pd.read_csv(PRECISION / "recovery_cells.csv")
    labels = {
        "high_mi": "High-MI",
        "low_mi": "Low-MI",
        "random": "Random x5",
        "null_mi": "Null-MI x5",
        "activation_energy": "Activation energy",
        "activation_variance": "Activation variance",
    }
    styles = {
        "high_mi": ("#B64342", "o", "-", 1.8, 1.0),
        "low_mi": ("#0F4D92", "s", "-", 1.3, 1.0),
        "random": ("#666666", "D", "--", 1.2, 1.0),
        "null_mi": ("#C28C2C", "^", "-.", 1.2, 1.0),
        "activation_energy": ("#79A89D", "v", ":", 1.1, 0.95),
        "activation_variance": ("#90749A", "P", ":", 1.1, 0.95),
    }
    order = tuple(labels)
    fig, ax = plt.subplots(figsize=(3.45, 2.70))
    rng = np.random.default_rng(2021)
    for strategy in order:
        group = recovery[recovery.strategy == strategy]
        x_values, means, lower, upper = [], [], [], []
        for fraction, cells in group.groupby("requested_fraction", sort=True):
            values = cells.fidelity_recovery_vs_all_low.to_numpy(dtype=np.float64)
            samples = rng.integers(0, len(values), size=(10_000, len(values)))
            boot = values[samples].mean(axis=1)
            lo, hi = np.quantile(boot, (0.025, 0.975))
            x_values.append(100 * float(fraction))
            means.append(float(values.mean()))
            lower.append(float(values.mean() - lo))
            upper.append(float(hi - values.mean()))
        color, marker, linestyle, linewidth, alpha = styles[strategy]
        ax.errorbar(
            x_values,
            means,
            yerr=np.asarray([lower, upper]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=4.0 if strategy == "high_mi" else 3.4,
            markeredgewidth=0.5,
            capsize=1.8,
            elinewidth=0.7,
            alpha=alpha,
            label=labels[strategy],
            zorder=4 if strategy == "high_mi" else 2,
        )
    ax.axhline(0, color="#B8B8B8", linewidth=0.65, zorder=0)
    ax.set_xlim(10, 52.5)
    ax.set_ylim(-0.04, 0.94)
    ax.set_xticks((12.5, 25, 37.5, 50))
    ax.set_xticklabels(("12.5", "25", "37.5", "50"))
    ax.set_xlabel("History patches kept at native precision (%)")
    ax.set_ylabel("Forecast-fidelity recovery")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.55, zorder=0)
    ax.legend(
        ncol=2,
        fontsize=5.7,
        handlelength=2.2,
        columnspacing=0.8,
        borderaxespad=0,
        loc="upper left",
    )
    ax.text(
        0.98,
        0.03,
        "42 model-dataset cells per budget",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="#555555",
    )
    fig.tight_layout(pad=0.35)
    _save(fig, "Fig5_precision_application")


DATASET_ORDER = ("etth1", "etth2", "ettm1", "ettm2", "weather", "electricity16", "traffic16")
DATASET_LABELS = {
    "etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2",
    "weather": "Weather", "electricity16": "Electricity", "traffic16": "Traffic",
}
MODEL_SLUGS = {
    "Chronos2": "chronos2", "Moirai2": "moirai2", "Toto2": "toto2",
    "TimesFM2.5": "timesfm25", "ChronosBolt": "chronos_bolt", "TTM": "ttm",
}


def _reference_npz(model: str, dataset_key: str) -> Path:
    """从审计后的 topology 清单定位精确 atlas，兼容复用的 V5 ETTm1 运行。"""
    dataset_name = {
        "etth1": "ETTh1", "etth2": "ETTh2", "ettm1": "ETTm1", "ettm2": "ETTm2",
        "weather": "weather", "electricity16": "electricity", "traffic16": "traffic",
    }[dataset_key]
    topology = pd.read_csv(TOPOLOGY / "topology_descriptors.csv")
    rows = topology[(topology.model == model) & (topology.dataset == dataset_name)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one audited atlas for {model}/{dataset_key}, found {len(rows)}")
    path = Path(rows.iloc[0].run_dir) / "mi_results.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _dataset_matrix(data: pd.DataFrame, value: str, dataset_column: str = "dataset_key") -> np.ndarray:
    """把长表转成固定模型顺序和数据集顺序的热图矩阵。"""
    matrix = np.full((len(MODELS), len(DATASET_ORDER)), np.nan, dtype=np.float64)
    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASET_ORDER):
            rows = data[(data.model == model) & (data[dataset_column] == dataset)]
            if not rows.empty:
                matrix[i, j] = float(rows[value].mean())
    return matrix


def figure_s2_ettm1_atlas_grid() -> None:
    """绘制六模型 ETTm1 atlas，展示每个模型内部的相对 MI 形状。"""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.45), sharex=False, sharey=False,
                             constrained_layout=True)
    for ax, model in zip(axes.flat, MODELS):
        z = np.load(_reference_npz(model, "ettm1"))
        atlas = np.asarray(z["mi_z"], dtype=np.float64)
        ranks = np.argsort(np.argsort(atlas.ravel())).reshape(atlas.shape)
        ranks = (ranks + 0.5) / ranks.size
        image = ax.imshow(ranks, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        top_layer, top_patch = np.unravel_index(np.argmax(atlas), atlas.shape)
        ax.scatter([top_patch], [top_layer], s=22, facecolors="none", edgecolors="#F4D35E", linewidths=1.0)
        ax.set_title(MODEL_LABELS[model], fontsize=8.0, pad=3)
        ax.set_xlabel("History patch (old  →  recent)", fontsize=6.5)
        ax.set_ylabel("Native layer / stage", fontsize=6.5)
        ax.tick_params(labelsize=6, length=2)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02, shrink=0.82)
    cbar.set_label("Within-atlas MI percentile", fontsize=6.8)
    fig.suptitle("ETTm1: native layer--patch MI atlases (rank-normalized within each model)", fontsize=9)
    _save(fig, "FigS2_ettm1_atlas_grid")


def figure_s3_dataset_topology() -> None:
    """绘制所有注册数据集的四个 topology 描述符，补足主图的聚合视角。"""
    data = pd.read_csv(TOPOLOGY / "topology_descriptors.csv").copy()
    data["dataset_key"] = data["dataset"].map({
        "ETTh1": "etth1", "ETTh2": "etth2", "ETTm1": "ettm1", "ETTm2": "ettm2",
        "weather": "weather", "electricity": "electricity16", "traffic": "traffic16",
    })
    specs = (
        ("top_quarter_concentration", "Top-quarter concentration", "magma", 0.25, 0.50),
        ("anchor_ratio", "Anchor ratio", "viridis", 1.0, 2.1),
        ("mean_boundary_distance", "Distance from boundary", "Blues_r", 0.20, 0.60),
        ("expected_layer_depth", "Expected layer depth", "cividis", 0.30, 0.70),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7))
    for ax, (column, title, cmap, vmin, vmax) in zip(axes.flat, specs):
        matrix = _dataset_matrix(data, column)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8.0, pad=3)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=6.2)
        ax.set_xticks(range(len(DATASET_ORDER)))
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=6.0, rotation=30, ha="right")
        ax.tick_params(length=0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.4,
                            color="white" if matrix[i, j] > (vmin + vmax) / 2 else "#222222")
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Dataset-specific topology descriptors (within-run normalized quantities)", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "FigS3_dataset_topology")


def figure_s4_probe_alignment() -> None:
    """绘制 global/target probe 差异及 MI--probe 排名对齐。"""
    probes = pd.read_csv(EVIDENCE / "probe_selected_units.csv")
    global_diff = _dataset_matrix(probes[probes.scope == "global"], "functional_top_minus_low_r2")
    target_diff = _dataset_matrix(probes[probes.scope == "target"], "functional_top_minus_low_r2")
    alignment = _dataset_matrix(probes[probes.scope == "global"], "mi_vs_probe_spearman")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), sharey=True)
    matrices = (global_diff, target_diff, alignment)
    titles = ("Global probe: top − low $R^2$", "Target probe: top − low $R^2$", "MI–probe rank correlation")
    cmaps = ("RdBu_r", "RdBu_r", "PuOr")
    norms = (TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25),
             TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.25),
             TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5))
    for ax, matrix, title, cmap, norm in zip(axes, matrices, titles, cmaps, norms):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xticks(range(len(DATASET_ORDER)))
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=5.7, rotation=34, ha="right")
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=6.2)
    fig.suptitle("Semantic validation is scope-dependent and need not track MI monotonically", fontsize=8.8, y=1.03)
    fig.tight_layout()
    _save(fig, "FigS4_probe_alignment")


def figure_s5_functional_coverage() -> None:
    """绘制 25% top-minus-control 的跨数据集功能覆盖，而非只展示 ETTm1。"""
    contrasts = pd.read_csv(EVIDENCE / "progressive_contrasts.csv")
    subset = contrasts[(contrasts["mode"] == "remove") & (np.abs(contrasts.fraction - 0.25) < 1e-8)]
    matrices = []
    for contrast in ("top_minus_bottom", "top_minus_random_mean"):
        matrices.append(_dataset_matrix(subset[subset.contrast == contrast], "all_forecast_change_mse_mean"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True)
    titles = ("Top − bottom MI", "Top − random mean")
    for ax, matrix, title in zip(axes, matrices, titles):
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.10, vcenter=0, vmax=1.00))
        ax.set_title(title + " at 25%", fontsize=8, pad=3)
        ax.set_xticks(range(len(DATASET_ORDER)))
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=5.8, rotation=32, ha="right")
        ax.tick_params(length=0)
        ax.set_facecolor("#E7E7E7")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.4)
                else:
                    ax.text(j, i, "NA", ha="center", va="center", fontsize=5.1, color="#666666")
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=6.2)
    fig.suptitle("Complete functional coverage across registered regimes", fontsize=8.8, y=1.03)
    fig.tight_layout()
    _save(fig, "FigS5_functional_coverage")


def figure_s10_progressive_generalization() -> None:
    """汇总四个干预比例的跨模型、跨数据集显著性与条件一致性。"""
    contrasts = pd.read_csv(EVIDENCE / "progressive_contrasts.csv")
    subset = contrasts[contrasts["mode"] == "remove"].copy()
    subset["positive_significant"] = (
        (subset.all_forecast_change_mse_mean > 0)
        & (subset.all_forecast_change_mse_sign_flip_p_greater < 0.05)
    )
    fractions = np.asarray(sorted(subset.fraction.unique()), dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": (0.9, 1.55)})
    ax = axes[0]
    for contrast, label, color, marker in (
        ("top_minus_bottom", "Top > bottom MI", "#B64342", "o"),
        ("top_minus_random_mean", "Top > random mean", "#4D4D4D", "s"),
    ):
        values = subset[subset.contrast == contrast].groupby("fraction").positive_significant.agg(["sum", "count"])
        rates = (values["sum"] / values["count"]).reindex(fractions).to_numpy(dtype=np.float64)
        counts = values["sum"].reindex(fractions).to_numpy(dtype=int)
        ax.plot(fractions * 100, rates, color=color, marker=marker, lw=1.5, ms=4, label=label)
        for x, y, count in zip(fractions * 100, rates, counts):
            ax.text(x, y + 0.012, f"{count}/42", ha="center", va="bottom", fontsize=5.6, color=color)
    ax.set_xlim(10, 52.5)
    ax.set_ylim(0.72, 1.02)
    ax.set_xticks(fractions * 100)
    ax.set_xlabel("Selected history patches (%)")
    ax.set_ylabel("Positive and significant fraction")
    ax.legend(fontsize=6.2, loc="lower right")
    _panel(ax, "a", x=-0.18)

    pair_counts = np.zeros((len(MODELS), len(DATASET_ORDER)), dtype=np.int64)
    for i, model in enumerate(MODELS):
        for j, dataset in enumerate(DATASET_ORDER):
            pair = subset[(subset.model == model) & (subset.dataset_key == dataset)]
            count = 0
            for fraction in fractions:
                fraction_rows = pair[np.isclose(pair.fraction, fraction)]
                flags = fraction_rows.set_index("contrast").positive_significant
                if bool(flags.get("top_minus_bottom", False)) and bool(flags.get("top_minus_random_mean", False)):
                    count += 1
            pair_counts[i, j] = count
    ax = axes[1]
    image = ax.imshow(pair_counts, aspect="auto", cmap="Blues", vmin=0, vmax=4)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=6.2)
    ax.set_xticks(range(len(DATASET_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=5.8, rotation=30, ha="right")
    ax.tick_params(length=0)
    for i in range(pair_counts.shape[0]):
        for j in range(pair_counts.shape[1]):
            value = pair_counts[i, j]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=6,
                    color="white" if value >= 3 else "#222222")
    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.025, ticks=(0, 1, 2, 3, 4))
    cbar.set_label("Fractions passing both controls", fontsize=6.6)
    _panel(ax, "b", x=-0.13)
    fig.suptitle("Progressive functional enrichment across all registered model--regime pairs",
                 fontsize=8.8, y=1.03)
    fig.tight_layout()
    _save(fig, "FigS10_progressive_generalization")


def figure_s6_output_footprint() -> None:
    """绘制干预影响的输出通道覆盖和相对 forecast-change，保留无效 sentinel 为缺失。"""
    conditions = pd.read_csv(EVIDENCE / "progressive_conditions.csv")
    baselines = pd.read_csv(EVIDENCE / "forecast_baselines.csv")
    subset = conditions[(conditions["mode"] == "remove") & (conditions.strategy == "top") &
                        (np.abs(conditions.fraction - 0.25) < 1e-8)].copy()
    channel_counts = {"etth1": 7, "etth2": 7, "ettm1": 7, "ettm2": 7, "weather": 21,
                      "electricity16": 16, "traffic16": 16}
    subset["footprint_fraction"] = subset.apply(
        lambda row: row.output_footprint_effective_channels / channel_counts[row.dataset_key]
        if np.isfinite(row.output_footprint_effective_channels) and row.output_footprint_effective_channels < 1e6 else np.nan,
        axis=1,
    )
    joined = subset.merge(baselines[["model", "dataset_key", "baseline_all_mse"]], on=["model", "dataset_key"], how="left")
    joined["relative_change"] = joined.all_forecast_change_mse_mean / joined.baseline_all_mse.replace(0, np.nan)
    matrices = (_dataset_matrix(joined, "footprint_fraction"), _dataset_matrix(joined, "relative_change"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True)
    for ax, matrix, title, cmap, vmin, vmax in (
        (axes[0], matrices[0], "Affected output-channel fraction", "viridis", 0, 1),
        (axes[1], matrices[1], "Forecast-change MSE / baseline", "magma", 0, 1.5),
    ):
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title + " (25% top-MI)", fontsize=7.8, pad=3)
        ax.set_xticks(range(len(DATASET_ORDER)))
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER], fontsize=5.8, rotation=32, ha="right")
        ax.tick_params(length=0)
        ax.set_facecolor("#E7E7E7")
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    axes[0].set_yticks(range(len(MODELS)))
    axes[0].set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=6.2)
    fig.suptitle("Functional output footprint: which forecast channels respond to an anchor intervention?", fontsize=8.8, y=1.03)
    fig.tight_layout()
    _save(fig, "FigS6_output_footprint")


def figure_s7_panel_full_detail() -> None:
    """展开注册面板与全变量压力测试的 patch、layer 和 atlas 三种对齐。"""
    stress = pd.read_csv(ROBUSTNESS / "full_variable_stress.csv")
    labels = [f"{MODEL_LABELS[row.model]}\n{str(row.dataset).capitalize()}" for _, row in stress.iterrows()]
    x = np.arange(len(stress))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for offset, column, label, color in ((-0.20, "patch_profile_spearman", "Patch", "#0F4D92"),
                                         (0.0, "layer_profile_spearman", "Layer", "#B64342"),
                                         (0.20, "atlas_spearman", "Full atlas", "#4D4D4D")):
        ax.scatter(x + offset, stress[column], s=24, color=color, label=label, zorder=3)
        ax.plot(x + offset, stress[column], color=color, lw=0.7, alpha=0.45)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.set_ylim(-0.50, 1.02)
    ax.set_ylabel("Registered--full Spearman correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=6.0)
    ax.legend(ncol=3, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Scaling boundary: registered topology does not uniformly recover the full-variable atlas",
                 fontsize=8.6, loc="left", pad=20)
    _panel(ax, "a", x=-0.07)
    fig.tight_layout()
    _save(fig, "FigS7_panel_full_detail")


def figure_s8_null_calibration() -> None:
    """展示代表性 atlas 的 null calibration，说明 raw MI、z 和 q 的区别。"""
    model, dataset = "Chronos2", "ettm1"
    npz = np.load(_reference_npz(model, dataset))
    atlas = np.asarray(npz["mi_z"], dtype=np.float64)
    top = np.unravel_index(np.argmax(atlas), atlas.shape)
    null_values = np.asarray(npz["null_mi"][:, top[0], top[1]], dtype=np.float64)
    observed = float(npz["mi_raw"][top])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": (1.0, 1.05, 1.0)})
    raw = np.asarray(npz["mi_raw"], dtype=np.float64)
    image = axes[0].imshow(raw, origin="lower", aspect="auto", cmap="magma")
    axes[0].scatter([top[1]], [top[0]], s=22, facecolors="none", edgecolors="#F4D35E")
    axes[0].set_title("Raw KSG MI atlas", fontsize=7.8)
    axes[0].set_xlabel("History patch", fontsize=6.5); axes[0].set_ylabel("Layer", fontsize=6.5)
    axes[0].tick_params(labelsize=6, length=2)
    axes[1].hist(null_values, bins=18, color="#B7C8D8", edgecolor="white", linewidth=0.4)
    axes[1].axvline(observed, color="#B64342", lw=1.3, label="Observed top cell")
    axes[1].set_title("Circular-shift null at top cell", fontsize=7.8)
    axes[1].set_xlabel("KSG MI", fontsize=6.5); axes[1].set_ylabel("Null count", fontsize=6.5)
    axes[1].legend(fontsize=6.0)
    top_p = float(npz["p_values"][top])
    top_q = float(npz["q_values"][top])
    axes[1].text(0.97, 0.72, f"z={atlas[top]:.2f}\np={top_p:.3f}\nq={top_q:.3f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=6.2,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB"))
    image = axes[2].imshow(atlas, origin="lower", aspect="auto", cmap="viridis")
    axes[2].scatter([top[1]], [top[0]], s=22, facecolors="none", edgecolors="#F4D35E")
    axes[2].set_title("Null-calibrated atlas", fontsize=7.8)
    axes[2].set_xlabel("History patch", fontsize=6.5); axes[2].set_ylabel("Layer", fontsize=6.5)
    axes[2].tick_params(labelsize=6, length=2)
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.025, label=r"cell $z$ score")
    fig.suptitle("Null calibration prevents raw MI scale from being mistaken for cross-model evidence", fontsize=8.8, y=1.02)
    for ax, label in zip(axes, ("a", "b", "c")):
        _panel(ax, label, x=-0.14, y=1.02)
    fig.tight_layout()
    _save(fig, "FigS8_null_calibration")


def main() -> None:
    required = (
        TOPOLOGY / "temporal_profiles.csv",
        EVIDENCE / "probe_semantics.csv",
        EVIDENCE / "progressive_conditions.csv",
        ROBUSTNESS / "summary.json",
        PRECISION / "recovery_cells.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing V6 source tables: {missing}")
    figure_1_framework()
    figure_2_topology()
    figure_3_probe_semantics_combined()
    _semantic_figure(
        "global",
        "Fig3_global_semantics",
        "Global future semantics at the functional top-MI unit",
    )
    _semantic_figure(
        "target",
        "Fig4_target_semantics",
        "Target-channel future semantics at the functional top-MI unit",
    )
    figure_4_target_depth()
    _functional_figure(
        "remove",
        "Fig4_anchor_fraction",
        "Anchor fraction: replace selected patch sets with matched donor activations",
    )
    _functional_figure(
        "keep",
        "FigS9_complement_sufficiency",
        "Complement test: preserve the selected patch set and replace its complement",
    )
    figure_6_robustness()
    figure_5_precision_application()
    figure_6_architecture_synthesis()
    figure_s2_ettm1_atlas_grid()
    figure_s3_dataset_topology()
    figure_s4_probe_alignment()
    figure_s5_functional_coverage()
    figure_s10_progressive_generalization()
    figure_s6_output_footprint()
    figure_s7_panel_full_detail()
    figure_s8_null_calibration()
    print(json.dumps({"status": "complete", "output_dir": str(OUTPUT)}))


if __name__ == "__main__":
    main()
