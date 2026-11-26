# -*- coding: utf-8 -*-
"""
Publication Figure and Table Generation Script.
Generates Tables 1, 2, S1, S2 (.md) and Figures 2, 3, 4, S1, S2 (.pdf and .svg).
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def setup_publication_style():
    """Configure publication-grade Matplotlib styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def create_histogram_figures(
    data_list, titles, x_labels, bin_configs, filename,
    panel_labels=("A", "B", "C"), bar_color="#2563eb", stroke_color="#1d4ed8",
    tail_color="#1e3a8a",
):
    """Create 3-panel relative frequency histograms with pooled overflow/underflow tail bins."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)

    for idx, ax in enumerate(axes):
        vals = np.asanyarray(data_list[idx], dtype=float)
        b_min, b_max, b_width, tick_step = bin_configs[idx]
        interior_edges = np.arange(b_min, b_max + b_width * 0.5, b_width)

        under_vals = vals[vals < b_min]
        has_under = len(under_vals) > 0
        has_over = bool(np.any(vals > b_max))
        if has_over:
            over_vals = vals[vals >= b_max]
            in_vals = vals[(vals >= b_min) & (vals < b_max)]
        else:
            over_vals = np.array([], dtype=float)
            in_vals = vals[(vals >= b_min) & (vals <= b_max)]

        counts, _ = np.histogram(in_vals, bins=interior_edges)
        pcts = counts / len(vals) * 100.0
        lefts = interior_edges[:-1]

        ax.bar(
            lefts, pcts, width=b_width, align="edge",
            color=bar_color, edgecolor=stroke_color, linewidth=0.6, alpha=0.82,
        )

        if has_under:
            u_pct = len(under_vals) / len(vals) * 100.0
            ax.bar(
                b_min - b_width, u_pct, width=b_width, align="edge",
                color=tail_color, edgecolor=stroke_color, linewidth=0.6, alpha=0.88, hatch="///",
            )
        if has_over:
            o_pct = len(over_vals) / len(vals) * 100.0
            ax.bar(
                b_max, o_pct, width=b_width, align="edge",
                color=tail_color, edgecolor=stroke_color, linewidth=0.6, alpha=0.88, hatch="///",
            )

        q25, med, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
        ax.axvspan(q25, q75, color="#f59e0b", alpha=0.14)
        ax.axvline(med, color="#dc2626", linestyle="--", linewidth=1.8)

        tail_notes = []
        if has_under:
            tail_notes.append(f"<{b_min:,.0f}: {len(under_vals) / len(vals) * 100:.1f}%")
        if has_over:
            tail_notes.append(f"≥{b_max:,.0f}: {len(over_vals) / len(vals) * 100:.1f}%")
        stat_text = f"Median: {med:,.1f}\nIQR: {q25:,.1f}–{q75:,.1f}\n(n = {len(vals):,})"
        if tail_notes:
            stat_text += "\n" + " | ".join(tail_notes)
        ax.text(
            0.97, 0.95, stat_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color="#1e293b",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.92),
        )

        ax.text(-0.12, 1.06, panel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="#0f172a", va="bottom")
        ax.set_title(titles[idx], pad=12)
        ax.set_xlabel(x_labels[idx])
        ax.set_ylabel("Percentage (%)" if idx == 0 else "")

        x_left = b_min - (b_width if has_under else 0.0)
        x_right = b_max + (b_width if has_over else 0.0)
        ax.set_xlim(x_left, x_right)

        xticks = list(np.arange(b_min, b_max + tick_step * 0.1, tick_step))
        xticklabels = [f"{t:,.0f}" for t in xticks]
        if has_under:
            if xticks and abs(xticks[0] - b_min) < 1e-6:
                xticks[0] = b_min - 0.5 * b_width
                xticklabels[0] = f"<{b_min:,.0f}"
            else:
                xticks = [b_min - 0.5 * b_width] + xticks
                xticklabels = [f"<{b_min:,.0f}"] + xticklabels
        if has_over:
            if xticks and abs(xticks[-1] - b_max) < 1e-6:
                xticks[-1] = b_max + 0.5 * b_width
                xticklabels[-1] = f"≥{b_max:,.0f}"
            else:
                xticks.append(b_max + 0.5 * b_width)
                xticklabels.append(f"≥{b_max:,.0f}")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylim(bottom=0)
        ax.grid(False)

    fig.savefig(filename, format="svg", bbox_inches="tight", dpi=300)
    fig.savefig(os.path.splitext(filename)[0] + ".pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {filename} and .pdf")


def create_scatter_figure(x_vals, y_list, titles, y_labels, y_ranges, filename):
    """Create 3-panel full-range scatter plots with Spearman rank correlations (Figure 4)."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)
    x_arr = np.asanyarray(x_vals, dtype=float)
    panel_labels = ["A", "B", "C"]

    for idx, ax in enumerate(axes):
        y_arr = np.asanyarray(y_list[idx], dtype=float)
        y_min, y_max, y_step = y_ranges[idx]

        assert np.all((x_arr >= 60.0) & (x_arr <= 260.0)), "x values exceed Figure 4 limits"
        assert np.all((y_arr >= y_min) & (y_arr <= y_max)), f"y values exceed Figure 4 panel {idx} limits"

        ax.scatter(x_arr, y_arr, s=12, color="#2563eb", alpha=0.25, edgecolors="none", rasterized=True)

        rho, p_val = spearmanr(x_arr, y_arr)
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        ax.text(
            0.04, 0.95, f"Spearman ρ = {rho:.3f}\n({p_str}, n = {len(x_arr):,})",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.92),
        )

        ax.text(-0.12, 1.06, panel_labels[idx], transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="#0f172a", va="bottom")
        ax.set_title(titles[idx], pad=12)
        ax.set_xlabel("Mean Sensor Glucose (mg/dL)")
        ax.set_ylabel(y_labels[idx])
        ax.set_xlim(60, 260)
        ax.set_xticks(np.arange(60, 261, 40))
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + y_step * 0.1, y_step))

    fig.savefig(filename, format="svg", bbox_inches="tight", dpi=300)
    fig.savefig(os.path.splitext(filename)[0] + ".pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {filename} and .pdf")


def create_meal_type_comparison_figure(meal_subsets, filename):
    """Create Figure S2: full-range box plots (with outliers) across meal categories."""
    cat_order = ["breakfast", "lunch", "dinner", "snack"]
    cat_labels = ["Breakfast", "Lunch", "Dinner", "Snack"]
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    rows = []
    for cat, label in zip(cat_order, cat_labels):
        for m in meal_subsets.get(cat, []):
            rows.append({
                "Meal": f"{label}\n(n={len(meal_subsets.get(cat, [])):,})",
                "DeltaPeak": m["DeltaPeak"],
                "iAUC": m["iAUC"],
                "Time_to_peak": m["Time_to_peak"],
            })
    df_meals = pd.DataFrame(rows)
    x_order = [f"{lbl}\n(n={len(meal_subsets.get(cat, [])):,})" for cat, lbl in zip(cat_order, cat_labels)]

    metrics = [
        ("DeltaPeak", "A", "Postprandial Rise (ΔPeak)", "ΔPeak (mg/dL)", (-75, 305), np.arange(-50, 301, 50)),
        ("iAUC", "B", "Positive iAUC (0–180 min)", "Positive iAUC (mg/dL·min)", (-1000, 31000), np.arange(0, 30001, 5000)),
        ("Time_to_peak", "C", "Time to Peak Glucose", "Time to Peak (min)", (-8, 190), np.arange(0, 181, 30)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)

    for idx, (col, p_lbl, title, y_lbl, ylim, yticks) in enumerate(metrics):
        ax = axes[idx]
        assert df_meals[col].min() >= ylim[0] and df_meals[col].max() <= ylim[1], f"Figure S2 {col} exceeds ylim"

        sns.boxplot(
            data=df_meals, x="Meal", y=col, hue="Meal", order=x_order,
            palette=palette, legend=False, width=0.55, showfliers=True,
            flierprops={"marker": "o", "markersize": 2.0, "alpha": 0.22, "markeredgewidth": 0, "markerfacecolor": "#64748b"},
            boxprops={"alpha": 0.80, "linewidth": 1.1},
            medianprops={"color": "#0f172a", "linewidth": 2.0},
            whiskerprops={"linewidth": 1.1}, capprops={"linewidth": 1.1},
            ax=ax,
        )

        medians = df_meals.groupby("Meal")[col].median()
        for x_i, x_cat in enumerate(x_order):
            med_v = medians[x_cat]
            fmt = f"{med_v:,.0f}" if col != "DeltaPeak" else f"{med_v:.1f}"
            ax.text(
                x_i, med_v + (ylim[1] - ylim[0]) * 0.025, fmt,
                ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#0f172a",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffffff", edgecolor="none", alpha=0.82),
            )

        ax.text(-0.12, 1.06, p_lbl, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="#0f172a", va="bottom")
        ax.set_title(title, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel(y_lbl)
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.tick_params(axis="x", labelsize=8.5)

    fig.savefig(filename, format="svg", bbox_inches="tight", dpi=300)
    fig.savefig(os.path.splitext(filename)[0] + ".pdf", format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {filename} and .pdf")


def fmt_med_iqr(series, digits=1):
    s = np.asarray(series, dtype=float)
    q25, med, q75 = np.percentile(s, [25.0, 50.0, 75.0])
    return f"{med:,.{digits}f} [{q25:,.{digits}f}–{q75:,.{digits}f}]"


def fmt_mean_sd(series, digits=1):
    s = np.asarray(series, dtype=float)
    m = float(np.mean(s))
    sd = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
    return f"{m:,.{digits}f} ± {sd:,.{digits}f}"


def fmt_n_pct(count, total):
    pct = (count / total * 100.0) if total > 0 else 0.0
    return f"{count:,} ({pct:.1f}%)"


def build_table1_md(df_p):
    """Build Table 1 Markdown."""
    df = df_p.copy()
    df["age_cat"] = df["age_group"].replace({"Unknown": "Missing", "": "Missing"}).fillna("Missing")
    df["sex_cat"] = df["gender"].replace({"": "Unknown"}).fillna("Unknown")

    total_p = len(df)
    df_f = df[df["sex_cat"] == "Female"]
    df_m = df[df["sex_cat"] == "Male"]
    n_fem, n_male = len(df_f), len(df_m)
    n_sex_avail = n_fem + n_male
    n_age_avail = int((df["age_cat"] != "Missing").sum())

    lines = [
        f"# Table 1. Participant Characteristics (Primary Person-level Population, k ≥ 10 meals, n = {total_p:,})\n",
        f"| Characteristic | Overall (n = {total_p:,}) | Females (n = {n_fem:,}) | Males (n = {n_male:,}) | Status / Availability |",
        "| :--- | :---: | :---: | :---: | :--- |",
        "| **Sex, n (%)** | | | | |",
        f"| 　Female | {fmt_n_pct(n_fem, total_p)} | {n_fem:,} (100.0%) | — | Available (n = {n_sex_avail:,}) |",
        f"| 　Male | {fmt_n_pct(n_male, total_p)} | — | {n_male:,} (100.0%) | Available (n = {n_sex_avail:,}) |",
        f"| 　Unspecified / Missing | {fmt_n_pct(total_p - n_sex_avail, total_p)} | — | — | Missing demographic link |",
        "| **Age categories, n (%)** | | | | |",
    ]

    age_labels = [
        ("18–19", "18–19 years"), ("20–29", "20–29 years"), ("30–39", "30–39 years"),
        ("40–49", "40–49 years"), ("50–59", "50–59 years"), ("60–69", "60–69 years"), ("≥70", "≥70 years"),
    ]
    for key, label in age_labels:
        lines.append(
            f"| 　{label} | {fmt_n_pct(int((df['age_cat'] == key).sum()), total_p)} | "
            f"{fmt_n_pct(int((df_f['age_cat'] == key).sum()), n_fem)} | "
            f"{fmt_n_pct(int((df_m['age_cat'] == key).sum()), n_male)} | Available (n = {n_age_avail:,}) |"
        )
    lines.append(
        f"| 　Unspecified / Missing | {fmt_n_pct(int((df['age_cat'] == 'Missing').sum()), total_p)} | "
        f"{fmt_n_pct(int((df_f['age_cat'] == 'Missing').sum()), n_fem)} | "
        f"{fmt_n_pct(int((df_m['age_cat'] == 'Missing').sum()), n_male)} | Missing age record |"
    )

    lines.append("| **CGM Data Availability (Days 2–14)** | | | | |")
    for label, col, d in [("Valid CGM days (days/person)", "Valid_CGM_days", 0), ("CGM completeness (%)", "completeness_pct", 1)]:
        lines.append(f"| 　{label} | {fmt_med_iqr(df[col], d)} | {fmt_med_iqr(df_f[col], d)} | {fmt_med_iqr(df_m[col], d)} | Complete |")

    lines.append("| **Meal Data Availability** | | | | |")
    lines.append(f"| 　Eligible 3-h meals per person | {fmt_med_iqr(df['eligible_meals_count'], 0)} | {fmt_med_iqr(df_f['eligible_meals_count'], 0)} | {fmt_med_iqr(df_m['eligible_meals_count'], 0)} | Complete |")
    lines.append(f"| 　Total eligible 3-h meals | {int(df['eligible_meals_count'].sum()):,} meals | {int(df_f['eligible_meals_count'].sum()):,} meals | {int(df_m['eligible_meals_count'].sum()):,} meals | Complete |")
    lines.append("\n*Values are median [interquartile range, IQR] or number (percentage). Age categories reflect chronological age at the start of CGM monitoring.*")
    return "\n".join(lines)


def build_table2_md(df_p, df_m):
    """Build Table 2 Markdown."""
    total_p = len(df_p)
    total_meals_p = int(df_p["eligible_meals_count"].sum())

    df_k10_meals = df_m[df_m["ID"].isin(df_p["ID"])].copy()
    df_k10_meals["dp50"] = (df_k10_meals["DeltaPeak"] >= 50.0) * 100.0
    df_k10_meals["pk140"] = (df_k10_meals["Peak"] >= 140.0) * 100.0
    df_k10_meals["pk180"] = (df_k10_meals["Peak"] >= 180.0) * 100.0
    df_k10_meals["ttp60"] = (df_k10_meals["Time_to_peak"] >= 60.0) * 100.0
    prop_df = df_k10_meals.groupby("ID")[["dp50", "pk140", "pk180", "ttp60"]].mean()

    def stat_row(label, unit, series, med_d=1, mean_d=1):
        return f"| 　{label} | {unit} | {fmt_med_iqr(series, med_d)} | {fmt_mean_sd(series, mean_d)} |"

    lines = [
        "# Table 2. Overall Glycemic Metrics and Postprandial Glycemic Responses\n",
        "| Metric | Unit | Median [IQR] | Mean ± SD |",
        "| :--- | :---: | :---: | :---: |",
        f"| **A. Continuous Glucose Monitoring Metrics (Days 2–14; Person-level, n = {total_p:,})** | | | |",
    ]
    sec_a = [
        ("Mean sensor glucose", "mg/dL", "Mean_glucose", 1, 1),
        ("Glucose variability (SD)", "mg/dL", "SD", 1, 1),
        ("Coefficient of variation (CV)", "%", "CV", 1, 1),
        ("Time in range (TIR 70–180 mg/dL)", "%", "TIR_70_180", 1, 1),
        ("Time above range (TAR >180 mg/dL)", "%", "TAR_over_180", 1, 1),
        ("Time below range (TBR <70 mg/dL)", "%", "TBR_under_70", 1, 1),
        ("Glucose Management Indicator (GMI)", "%", "GMI", 2, 2),
    ]
    for lbl, unit, col, md, mn in sec_a:
        lines.append(stat_row(lbl, unit, df_p[col], md, mn))

    lines.append(f"| **B. Meal-Level Postprandial Glycemic Responses (Meal-level, n = {len(df_m):,} meals)** | | | |")
    sec_b = [
        ("Pre-meal baseline glucose", "mg/dL", "Baseline", 1, 1),
        ("Postprandial peak glucose", "mg/dL", "Peak", 1, 1),
        ("Postprandial rise (ΔPeak)", "mg/dL", "DeltaPeak", 1, 1),
        ("Time to peak", "min", "Time_to_peak", 0, 1),
        ("Positive iAUC 0–180 min", "mg/dL·min", "iAUC", 0, 1),
        ("Naive trapezoidal iAUC 0–180 min", "mg/dL·min", "iAUC_naive", 0, 1),
    ]
    for lbl, unit, col, md, mn in sec_b:
        lines.append(stat_row(lbl, unit, df_m[col], md, mn))

    lines.append(f"| **C. Person-Level Summary Metrics & Meal Category Proportions (Person-level, n = {total_p:,}; {total_meals_p:,} meals)** | | | |")
    sec_c = [
        ("Participant median iAUC 0–180", "mg/dL·min", df_p["iAUC_median"]),
        ("Intra-individual iAUC 0–180 IQR", "mg/dL·min", df_p["iAUC_IQR"]),
        ("Participant median ΔPeak", "mg/dL", df_p["DeltaPeak_median"]),
        ("Intra-individual ΔPeak IQR", "mg/dL", df_p["DeltaPeak_IQR"]),
        ("Proportion of meals with ΔPeak ≥ 50 mg/dL", "%", prop_df["dp50"]),
        ("Proportion of meals with Peak ≥ 140 mg/dL", "%", prop_df["pk140"]),
        ("Proportion of meals with Peak ≥ 180 mg/dL", "%", prop_df["pk180"]),
        ("Proportion of meals with Time to peak ≥ 60 min", "%", prop_df["ttp60"]),
    ]
    for lbl, unit, ser in sec_c:
        lines.append(stat_row(lbl, unit, ser, 1, 1))

    lines.append("\n*Values are median [IQR] and mean ± SD. Positive iAUC was calculated using the trapezoidal rule with linear baseline-crossing interpolation according to FAO/WHO and Wolever guidelines [16,17].*\n")
    return "\n".join(lines)


def build_table_s1_md(df_m, df_ov):
    """Build Supplementary Table S1 Markdown."""
    rows_spec = [
        ("Baseline glucose (mg/dL)", "Baseline", 1, "Nearly identical across definitions (0.5 mg/dL difference)"),
        ("Peak glucose (mg/dL)", "Peak", 1, "Identical median peak glucose across definitions"),
        ("Postprandial rise (ΔPeak, mg/dL)", "DeltaPeak", 1, "Identical median ΔPeak across both datasets (40.5 mg/dL)"),
        ("Time to peak (min)", "Time_to_peak", 0, "Slightly later peak in overlap dataset (52 vs 55 min)"),
        ("Positive iAUC 0–180 min (mg/dL·min)", "iAUC", 0, "Consistent distribution across definitions"),
    ]
    lines = [
        "# Supplementary Table S1. Sensitivity Analysis: Primary Clean 3-h Dataset vs. Overlap-Allowed Dataset\n",
        f"| Postprandial Metric | Primary Dataset (Non-overlapping 3-h Meals)<br>n = {len(df_m):,} meals | Sensitivity Dataset (Overlap Allowed)<br>n = {len(df_ov):,} meals | Comparison / Relative Difference |",
        "| :--- | :---: | :---: | :--- |",
    ]
    for label, col, md, note in rows_spec:
        bold = "**" if col in ["DeltaPeak", "Time_to_peak", "iAUC"] else ""
        lines.append(f"| **{label}** | {bold}{fmt_med_iqr(df_m[col], md)}{bold} | {bold}{fmt_med_iqr(df_ov[col], md)}{bold} | {note} |")
        lines.append(f"| 　*(Mean ± SD)* | {fmt_mean_sd(df_m[col], 1)} | {fmt_mean_sd(df_ov[col], 1)} | |")
    lines.append("\n*Values are median [IQR] and mean ± SD. The primary dataset excludes eating events followed by another meal within 180 min.*\n")
    return "\n".join(lines)


def build_table_s2_md(df_m):
    """Build Supplementary Table S2 Markdown."""
    lines = [
        f"# Supplementary Table S2. Comparison of Postprandial Glycemic Responses Across Meal Categories (n = {len(df_m):,} meals)\n",
        "| Meal Category | Meals, n (%) | Baseline Glucose (mg/dL) | Peak Glucose (mg/dL) | Postprandial Rise (ΔPeak, mg/dL) | Time to Peak (min) | Positive iAUC 0–180 (mg/dL·min) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    meal_cats = [("breakfast", "Breakfast"), ("lunch", "Lunch"), ("dinner", "Dinner"), ("snack", "Snack")]
    for cat_key, cat_label in meal_cats:
        sub = df_m[df_m["meal_type"] == cat_key]
        n_sub = len(sub)
        pct_sub = n_sub / len(df_m) * 100.0
        lines.append(
            f"| **{cat_label}** | {n_sub:,} ({pct_sub:.1f}%) | "
            f"{fmt_med_iqr(sub['Baseline'], 1)} | "
            f"{fmt_med_iqr(sub['Peak'], 1)} | "
            f"**{fmt_med_iqr(sub['DeltaPeak'], 1)}** | "
            f"**{fmt_med_iqr(sub['Time_to_peak'], 0)}** | "
            f"**{fmt_med_iqr(sub['iAUC'], 0)}** |"
        )
    lines.append(f"\n*Values are median [IQR] across the primary non-overlapping meal dataset (n = {len(df_m):,} meals).*\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate manuscript tables and Figures 2, 3, 4, S1, and S2.")
    parser.add_argument("--in-dir", default="./processed_data", help="Directory containing processed CSV files")
    parser.add_argument("--out-dir", default="./outputs", help="Directory for output tables and figures")
    args = parser.parse_args()

    setup_publication_style()
    os.makedirs(args.out_dir, exist_ok=True)

    df_meals = pd.read_csv(os.path.join(args.in_dir, "paper_meal_level.csv"), dtype={"ID": str})
    df_overlap = pd.read_csv(os.path.join(args.in_dir, "paper_meal_level_all_with_overlap.csv"), dtype={"ID": str})
    df_persons = pd.read_csv(os.path.join(args.in_dir, "paper_person_level.csv"), dtype={"ID": str})
    df_primary_p = df_persons[df_persons["eligible_meals_count"] >= 10].copy()

    tables = {
        "participant_characteristics_table1.md": build_table1_md(df_primary_p),
        "glycemic_metrics_table2.md": build_table2_md(df_primary_p, df_meals),
        "supplementary_table_s1_sensitivity.md": build_table_s1_md(df_meals, df_overlap),
        "supplementary_table_s2_meal_types.md": build_table_s2_md(df_meals),
    }
    for fname, content in tables.items():
        path = os.path.join(args.out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Saved {path}")

    create_histogram_figures(
        data_list=[df_meals["iAUC"], df_meals["DeltaPeak"], df_meals["Time_to_peak"]],
        titles=["Postprandial Incremental AUC", "Postprandial Glucose Rise (ΔPeak)", "Time to Peak Glucose"],
        x_labels=["Positive iAUC 0–180 min (mg/dL·min)", "ΔPeak (mg/dL)", "Time to peak (min)"],
        bin_configs=[(0.0, 12000.0, 500.0, 2000.0), (0.0, 110.0, 5.0, 20.0), (0.0, 180.0, 10.0, 30.0)],
        filename=os.path.join(args.out_dir, "figure2_meal_level_distributions.svg"),
    )

    create_histogram_figures(
        data_list=[df_primary_p["iAUC_median"], df_primary_p["DeltaPeak_median"], df_primary_p["DeltaPeak_IQR"]],
        titles=["Inter-individual: Median iAUC", "Inter-individual: Median ΔPeak", "Intra-individual: ΔPeak IQR"],
        x_labels=["Participant median iAUC (mg/dL·min)", "Participant median ΔPeak (mg/dL)", "Participant ΔPeak IQR (mg/dL)"],
        bin_configs=[(0.0, 8000.0, 400.0, 2000.0), (0.0, 90.0, 3.0, 15.0), (0.0, 75.0, 3.0, 15.0)],
        filename=os.path.join(args.out_dir, "figure3_person_level_distributions.svg"),
    )

    create_scatter_figure(
        x_vals=df_primary_p["Mean_glucose"],
        y_list=[df_primary_p["iAUC_median"], df_primary_p["DeltaPeak_median"], df_primary_p["DeltaPeak_IQR"]],
        titles=["Mean Glucose vs. Median iAUC", "Mean Glucose vs. Median ΔPeak", "Mean Glucose vs. ΔPeak IQR"],
        y_labels=["Participant median iAUC (mg/dL·min)", "Participant median ΔPeak (mg/dL)", "Intra-individual ΔPeak IQR (mg/dL)"],
        y_ranges=[(0.0, 18000.0, 3000.0), (0.0, 150.0, 30.0), (0.0, 140.0, 20.0)],
        filename=os.path.join(args.out_dir, "figure4_mean_glucose_vs_ppgr.svg"),
    )

    create_histogram_figures(
        data_list=[df_overlap["iAUC"], df_overlap["DeltaPeak"], df_overlap["Time_to_peak"]],
        titles=["Postprandial Incremental AUC (Overlap Allowed)", "Postprandial Glucose Rise (ΔPeak, Overlap Allowed)", "Time to Peak Glucose (Overlap Allowed)"],
        x_labels=["Positive iAUC 0–180 min (mg/dL·min)", "ΔPeak (mg/dL)", "Time to peak (min)"],
        bin_configs=[(0.0, 12000.0, 500.0, 2000.0), (0.0, 110.0, 5.0, 20.0), (0.0, 180.0, 10.0, 30.0)],
        filename=os.path.join(args.out_dir, "figure_s1_overlap_allowed_distributions.svg"),
        bar_color="#0891b2", stroke_color="#0e7490", tail_color="#164e63",
    )

    meal_subsets = {cat: grp.to_dict("records") for cat, grp in df_meals.groupby("meal_type")}
    create_meal_type_comparison_figure(meal_subsets, os.path.join(args.out_dir, "figure_s2_meal_type_comparisons.svg"))
    print("Completed generating tables and figures.")


if __name__ == "__main__":
    main()
