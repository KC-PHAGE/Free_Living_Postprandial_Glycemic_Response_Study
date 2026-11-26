# -*- coding: utf-8 -*-
"""
Figure 5 Generation Script: Four-Metrics Waterfall Spectrum Panels (A–D and 2x2 Combined).
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_publication_style():
    """Configure publication-grade Matplotlib styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 15,
        "axes.titlesize": 18.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 16.5,
        "xtick.labelsize": 14.5,
        "ytick.labelsize": 14.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5.5,
        "ytick.major.size": 5.5,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def get_spectrum_configs(n_participants):
    """Return tier boundaries, labels, colors, and axis specs for all four metrics."""
    return [
        {
            "metric": "DeltaPeak",
            "panel": "A",
            "panel_title": "A. Postprandial Glucose Rise (ΔPeak)",
            "y_trend_label": "Participant Median ΔPeak (mg/dL)",
            "y_trend_short": "Median ΔPeak (mg/dL)",
            "bins": [-np.inf, 30.0, 50.0, 70.0, np.inf],
            "labels": ["< 30 mg/dL", "30 to < 50 mg/dL", "50 to < 70 mg/dL", "≥ 70 mg/dL"],
            "colors": ["#0284c7", "#10b981", "#f59e0b", "#ef4444"],
            "ylim": (10.0, 100.0),
            "yticks": [10, 25, 40, 55, 70, 85, 100],
            "svg_filename": "figure5a_spectrum_delta_peak.svg",
        },
        {
            "metric": "iAUC",
            "panel": "B",
            "panel_title": "B. Positive iAUC (0–180 min)",
            "y_trend_label": "Participant Median iAUC 0–180 (mg/dL·min)",
            "y_trend_short": "Median iAUC (mg/dL·min)",
            "bins": [-np.inf, 1300.0, 3100.0, 5400.0, np.inf],
            "labels": ["Q1: < 1,300", "Q2: 1,300 to < 3,100", "Q3: 3,100 to < 5,400", "Q4: ≥ 5,400 mg/dL·min"],
            "colors": ["#3b82f6", "#06b6d4", "#f97316", "#dc2626"],
            "ylim": (0.0, 9000.0),
            "yticks": [0, 1500, 3000, 4500, 6000, 7500, 9000],
            "svg_filename": "figure5b_spectrum_iauc.svg",
        },
        {
            "metric": "Peak",
            "panel": "C",
            "panel_title": "C. Postprandial Peak Glucose",
            "y_trend_label": "Participant Median Peak Glucose (mg/dL)",
            "y_trend_short": "Median Peak (mg/dL)",
            "bins": [-np.inf, 120.0, 140.0, 180.0, np.inf],
            "labels": ["< 120 mg/dL", "120 to < 140 mg/dL", "140 to < 180 mg/dL", "≥ 180 mg/dL"],
            "colors": ["#0d9488", "#84cc16", "#f59e0b", "#ef4444"],
            "ylim": (100.0, 225.0),
            "yticks": [100, 125, 150, 175, 200, 225],
            "svg_filename": "figure5c_spectrum_peak_glucose.svg",
        },
        {
            "metric": "Time_to_peak",
            "panel": "D",
            "panel_title": "D. Time to Peak Glucose",
            "y_trend_label": "Participant Median Time to Peak (min)",
            "y_trend_short": "Median Time to Peak (min)",
            "bins": [-np.inf, 45.0, 60.0, 90.0, np.inf],
            "labels": ["< 45 min", "45 to < 60 min", "60 to < 90 min", "≥ 90 min"],
            "colors": ["#0ea5e9", "#22c55e", "#f59e0b", "#a855f7"],
            "ylim": (30.0, 105.0),
            "yticks": [30, 45, 60, 75, 90, 105],
            "svg_filename": "figure5d_spectrum_time_to_peak.svg",
        },
    ]


def summarize_metric_spectrum(df_meals, cfg):
    """Compute per-participant tier proportions (%) and medians, sorted deterministically by upper-tier burden."""
    col = cfg["metric"]
    cats = pd.cut(df_meals[col], bins=cfg["bins"], labels=["c1", "c2", "c3", "c4"], right=False)
    df_tmp = df_meals[["ID", col]].copy()
    df_tmp["cat"] = cats

    ctab = pd.crosstab(df_tmp["ID"], df_tmp["cat"], normalize="index") * 100.0
    for c in ["c1", "c2", "c3", "c4"]:
        if c not in ctab.columns:
            ctab[c] = 0.0
    ctab = ctab[["c1", "c2", "c3", "c4"]]

    med = df_tmp.groupby("ID")[col].median().rename("trend")
    burden = ((df_tmp[col] >= cfg["bins"][2]).groupby(df_tmp["ID"]).mean() * 100.0).round(10).rename("burden")
    df_sum = ctab.join(med).join(burden).reset_index()
    df_sum = df_sum.sort_values(by=["burden", "trend", "ID"], ascending=True).reset_index(drop=True)
    return df_sum


def plot_spectrum_on_axis(ax1, df_sum, cfg, is_panel=False):
    """Plot 100% stacked waterfall spectrum and 61-participant rolling median trend."""
    total_n = len(df_sum)
    x = np.arange(total_n)

    ax1.stackplot(
        x,
        df_sum["c1"], df_sum["c2"], df_sum["c3"], df_sum["c4"],
        labels=cfg["labels"], colors=cfg["colors"], alpha=0.90,
    )
    ax1.set_xlim(0, total_n - 1)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Proportion of Meals (%)", fontsize=16.5)
    if not is_panel:
        ax1.set_xlabel(f"Participants Ranked by Upper-Category Burden (1 to {total_n:,})", fontsize=16.5)
    else:
        ax1.set_xlabel(f"Participants (n = {total_n:,})", fontsize=16.5)
    ax1.tick_params(axis="both", labelsize=14.5)
    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    roll_med = df_sum["trend"].rolling(window=61, center=True, min_periods=1).median()
    print(f"  [{cfg['panel']}] {cfg['metric']} rolling median range: {roll_med.min():.2f} to {roll_med.max():.2f}")
    ax2.plot(x, roll_med, color="#0f172a", linewidth=2.6, label="Median Trend (61-person rolling)")
    ax2.set_ylim(cfg["ylim"])
    ax2.set_yticks(cfg["yticks"])
    ax2.set_ylabel(cfg["y_trend_label"] if not is_panel else cfg["y_trend_short"], color="#0f172a", fontweight="bold", fontsize=16.5)
    ax2.tick_params(axis="y", labelsize=14.5)
    ax2.grid(False)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if is_panel:
        ax1.legend(
            handles1 + handles2, labels1 + labels2,
            loc="upper left", fontsize=13.5, frameon=True,
            facecolor="#ffffff", framealpha=0.92, edgecolor="#cbd5e1",
            handlelength=1.5, labelspacing=0.3, borderpad=0.35,
        )
        ax1.set_title(cfg["panel_title"], loc="left", pad=10, fontsize=18.5, fontweight="bold")
    else:
        ax1.legend(
            handles1 + handles2, labels1 + labels2,
            loc="upper center", bbox_to_anchor=(0.5, 1.15),
            ncol=3, frameon=False, fontsize=13.5,
        )


def generate_individual_spectrum(df_meals, cfg, out_dir):
    df_sum = summarize_metric_spectrum(df_meals, cfg)
    fig, ax = plt.subplots(figsize=(15.5, 5.8), constrained_layout=True)
    plot_spectrum_on_axis(ax, df_sum, cfg, is_panel=False)

    out_path = os.path.join(out_dir, cfg["svg_filename"])
    fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path} and .pdf")


def generate_combined_panel(df_meals, configs, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11.8), constrained_layout=True)
    for ax, cfg in zip(axes.flat, configs):
        df_sum = summarize_metric_spectrum(df_meals, cfg)
        plot_spectrum_on_axis(ax, df_sum, cfg, is_panel=True)

    out_path = os.path.join(out_dir, "figure5_four_metrics_spectrum_panel.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=300)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path} and .pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 5 waterfall spectrum panels.")
    parser.add_argument("--in-dir", default="./processed_data", help="Directory containing paper_meal_level.csv")
    parser.add_argument("--out-dir", default="./outputs", help="Directory for output figures")
    args = parser.parse_args()

    setup_publication_style()
    os.makedirs(args.out_dir, exist_ok=True)

    df_meals = pd.read_csv(os.path.join(args.in_dir, "paper_meal_level.csv"), dtype={"ID": str})
    counts = df_meals.groupby("ID").size()
    eligible_ids = counts[counts >= 10].index
    df_valid = df_meals[df_meals["ID"].isin(eligible_ids)].copy()

    n_p = len(eligible_ids)
    print(f"Eligible participants (>=10 meals): {n_p:,}")

    configs = get_spectrum_configs(n_p)
    for cfg in configs:
        generate_individual_spectrum(df_valid, cfg, args.out_dir)
    generate_combined_panel(df_valid, configs, args.out_dir)
    print("Completed generating Figure 5 spectrum panels.")


if __name__ == "__main__":
    main()
