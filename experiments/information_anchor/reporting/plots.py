from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_profile(
    values: np.ndarray,
    path: str | Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    q_values: np.ndarray | None = None,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    positions = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    ax.plot(positions, values, color="#2166ac", marker="o", linewidth=1.8, markersize=4)
    if q_values is not None:
        significant = np.asarray(q_values) < 0.05
        ax.scatter(positions[significant], values[significant], color="#b2182b", s=32, zorder=3, label="BH q < 0.05")
        if significant.any():
            ax.legend(frameon=False)
    ax.axhline(0.0, color="0.55", linewidth=0.8, linestyle="--")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_heatmap(
    matrix: np.ndarray,
    path: str | Path,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
) -> None:
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("History patch index (oldest to newest)")
    ax.set_ylabel("Layer index")
    ax.set_title(title, loc="left")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)
