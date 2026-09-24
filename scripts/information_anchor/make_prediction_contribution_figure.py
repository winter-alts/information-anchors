#!/usr/bin/env python3
"""Plot ETTm1 ground-truth loss changes from accepted progressive runs."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.information_anchor.interventions.common import hierarchical_bootstrap_ci

CONDITIONS = ROOT / "results/information_anchor_v6_evidence/progressive_conditions.csv"
OUT = ROOT / "results/information_anchor_iclr_figures"
AUDIT = ROOT / "results/information_anchor_recency_target_audit"

MODELS = ("Chronos2", "Moirai2", "Toto2", "TimesFM2.5", "ChronosBolt", "TTM")
MODEL_LABELS = {
    "Chronos2": "Chronos-2",
    "Moirai2": "Moirai-2",
    "Toto2": "Toto-2",
    "TimesFM2.5": "TimesFM 2.5",
    "ChronosBolt": "Chronos-Bolt",
    "TTM": "TTM-R2",
}
STRATEGIES = ("top", "bottom", "random")
STRATEGY_LABELS = {"top": "High MI", "bottom": "Low MI", "random": "Random"}
COLORS = {"top": "#B43C39", "bottom": "#356AA0", "random": "#777777"}
MARKERS = {"top": "o", "bottom": "s", "random": "^"}
FRACTIONS = (0.125, 0.25, 0.375, 0.5)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
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


def accepted_runs() -> dict[str, Path]:
    runs: dict[str, Path] = {}
    with CONDITIONS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset_key"] != "ettm1" or row["model"] not in MODELS:
                continue
            runs.setdefault(row["model"], Path(row["run_dir"]))
    missing = [model for model in MODELS if model not in runs]
    if missing:
        raise RuntimeError(f"Missing accepted ETTm1 runs: {missing}")
    return runs


def metric_key(strategy: str, fraction: float, repetition: int | None = None) -> str:
    code = f"{int(round(fraction * 100)):03d}"
    if strategy == "random":
        if repetition is None:
            raise ValueError("Random condition requires a repetition.")
        return f"remove__random_r{repetition:02d}_f{code}__all_delta_mse"
    return f"remove__{strategy}_f{code}__all_delta_mse"


def summarize(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    lower, upper = hierarchical_bootstrap_ci(values, repetitions=5_000, seed=seed)
    return float(values.mean()), lower, upper


def main() -> None:
    records: list[dict[str, object]] = []
    runs = accepted_runs()
    for model_index, model in enumerate(MODELS):
        path = runs[model] / "progressive_results.npz"
        with np.load(path) as arrays:
            for fraction_index, fraction in enumerate(FRACTIONS):
                for strategy_index, strategy in enumerate(STRATEGIES):
                    if strategy == "random":
                        repetitions = [
                            np.asarray(arrays[metric_key(strategy, fraction, repetition)], dtype=np.float64)
                            for repetition in range(5)
                        ]
                        values = np.mean(np.stack(repetitions, axis=0), axis=0)
                    else:
                        values = np.asarray(arrays[metric_key(strategy, fraction)], dtype=np.float64)
                    mean, lower, upper = summarize(
                        values,
                        seed=81_000 + model_index * 100 + fraction_index * 10 + strategy_index,
                    )
                    records.append(
                        {
                            "model": model,
                            "dataset": "ETTm1",
                            "mode": "remove",
                            "strategy": strategy,
                            "fraction": fraction,
                            "all_ground_truth_delta_mse_mean": mean,
                            "all_ground_truth_delta_mse_ci95_lower": lower,
                            "all_ground_truth_delta_mse_ci95_upper": upper,
                            "sample_count": int(values.shape[1]),
                            "donor_shift_count": int(values.shape[0]),
                            "run_dir": str(runs[model]),
                        }
                    )

    AUDIT.mkdir(parents=True, exist_ok=True)
    output_csv = AUDIT / "ettm1_prediction_contribution_conditions.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.18), constrained_layout=True)
    for model_index, (ax, model) in enumerate(zip(axes.flat, MODELS, strict=True)):
        rows = [row for row in records if row["model"] == model]
        for strategy in STRATEGIES:
            selected = sorted(
                [row for row in rows if row["strategy"] == strategy],
                key=lambda row: float(row["fraction"]),
            )
            x = np.asarray([float(row["fraction"]) * 100 for row in selected])
            mean = np.asarray([float(row["all_ground_truth_delta_mse_mean"]) for row in selected])
            lower = np.asarray([float(row["all_ground_truth_delta_mse_ci95_lower"]) for row in selected])
            upper = np.asarray([float(row["all_ground_truth_delta_mse_ci95_upper"]) for row in selected])
            ax.plot(
                x,
                mean,
                color=COLORS[strategy],
                marker=MARKERS[strategy],
                markersize=2.8,
                label=STRATEGY_LABELS[strategy],
            )
            ax.fill_between(x, lower, upper, color=COLORS[strategy], alpha=0.12, linewidth=0)
        ax.axhline(0, color="#888888", linewidth=0.65)
        ax.set_xticks(np.asarray(FRACTIONS) * 100)
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.45)
        if model_index >= 3:
            ax.set_xlabel("Replaced history patches (%)")
        if model_index % 3 == 0:
            ax.set_ylabel(r"Ground-truth loss change $\Delta$MSE")
        ax.text(
            0.5,
            -0.27,
            f"({chr(ord('a') + model_index)}) {MODEL_LABELS[model]}",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        handlelength=1.3,
        columnspacing=0.9,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 600}),
    ):
        fig.savefig(
            OUT / f"Fig4_ettm1_prediction_contribution.{suffix}",
            bbox_inches="tight",
            pad_inches=0.04,
            **options,
        )
    plt.close(fig)
    print(f"Wrote {output_csv}")
    print(f"Wrote ETTm1 prediction-contribution figure to {OUT}")


if __name__ == "__main__":
    main()
