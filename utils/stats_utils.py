# -*- coding: utf-8 -*-
"""
Statistical summary, Spearman correlation, and FAO/WHO positive iAUC helpers.
"""

import math
import numpy as np


def mean(vals):
    arr = np.asanyarray(vals, dtype=float)
    return float(np.mean(arr)) if arr.size > 0 else 0.0


def std(vals, ddof=1):
    arr = np.asanyarray(vals, dtype=float)
    if arr.size <= ddof:
        return 0.0
    return float(np.std(arr, ddof=ddof))


def median(vals):
    arr = np.asanyarray(vals, dtype=float)
    return float(np.median(arr)) if arr.size > 0 else 0.0


def percentile(vals, q):
    """Linear-interpolated percentile (q in (0.0, 1.0) as fraction or 0..100 as percent)."""
    arr = np.asanyarray(vals, dtype=float)
    if arr.size == 0:
        return 0.0
    q_pct = q * 100.0 if (isinstance(q, float) and 0.0 < q < 1.0) else float(q)
    return float(np.percentile(arr, q_pct))


def iqr(vals):
    arr = np.asanyarray(vals, dtype=float)
    if arr.size == 0:
        return 0.0
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    return float(q75 - q25)


def summarize_distribution(vals):
    arr = np.asanyarray(vals, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "iqr": 0.0}
    p25, med, p75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(med),
        "p25": float(p25),
        "p75": float(p75),
        "iqr": float(p75 - p25),
    }


def calc_positive_iauc(curve_pts_or_times, glucoses=None, baseline=None):
    """Compute positive incremental area under the curve (Positive iAUC) per FAO/WHO & Wolever guidelines.

    Linearly interpolates baseline-crossing points so only positive area above pre-meal baseline is integrated.
    Accepts either [(t0, g0), (t1, g1), ...] or (times, glucoses).
    """
    if glucoses is None:
        pts = np.asanyarray(curve_pts_or_times, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return 0.0
        times, gls = pts[:, 0], pts[:, 1]
    else:
        times = np.asanyarray(curve_pts_or_times, dtype=float)
        gls = np.asanyarray(glucoses, dtype=float)
        if len(times) < 2 or len(gls) < 2:
            return 0.0

    if baseline is None:
        baseline = float(gls[0])

    sort_idx = np.argsort(times)
    t = times[sort_idx]
    y = gls[sort_idx] - baseline
    dt = np.diff(t)

    valid_dt = dt > 0
    if not np.any(valid_dt):
        return 0.0

    y1 = y[:-1][valid_dt]
    y2 = y[1:][valid_dt]
    dt = dt[valid_dt]
    iauc = 0.0

    both_pos = (y1 >= 0.0) & (y2 >= 0.0)
    if np.any(both_pos):
        iauc += np.sum(0.5 * (y1[both_pos] + y2[both_pos]) * dt[both_pos])

    cross_down = (y1 > 0.0) & (y2 < 0.0)
    if np.any(cross_down):
        t_cross = y1[cross_down] / (y1[cross_down] - y2[cross_down]) * dt[cross_down]
        iauc += np.sum(0.5 * y1[cross_down] * t_cross)

    cross_up = (y1 < 0.0) & (y2 > 0.0)
    if np.any(cross_up):
        t_pos = y2[cross_up] / (y2[cross_up] - y1[cross_up]) * dt[cross_up]
        iauc += np.sum(0.5 * y2[cross_up] * t_pos)

    return float(iauc)


def calc_spearman_corr(x, y):
    """Compute Spearman rank correlation coefficient (rho) and two-sided p-value."""
    try:
        from scipy.stats import spearmanr
        res = spearmanr(x, y)
        r = float(res.statistic if hasattr(res, "statistic") else res[0])
        p = float(res.pvalue if hasattr(res, "pvalue") else res[1])
        return r, p
    except ImportError:
        x_arr = np.asanyarray(x, dtype=float)
        y_arr = np.asanyarray(y, dtype=float)
        n = len(x_arr)
        if n < 3:
            return 0.0, 1.0

        def rank_array(a):
            sort_order = np.argsort(a)
            ranks = np.empty(n, dtype=float)
            ranks[sort_order] = np.arange(1, n + 1)
            sorted_a = a[sort_order]
            if np.any(np.diff(sorted_a) == 0):
                _, inv_idx, counts = np.unique(sorted_a, return_inverse=True, return_counts=True)
                for u_idx, count in enumerate(counts):
                    if count > 1:
                        matches = np.where(inv_idx == u_idx)[0]
                        ranks[sort_order[matches]] = float(np.mean(matches + 1))
            return ranks

        r = float(np.corrcoef(rank_array(x_arr), rank_array(y_arr))[0, 1])
        if abs(r) >= 1.0:
            p = 0.0
        else:
            t_stat = r * math.sqrt((n - 2) / (1.0 - r ** 2))
            p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
        return r, float(p)


def moving_median(arr, window=61):
    """Compute sliding-window median filter."""
    try:
        from scipy.ndimage import median_filter
        return median_filter(np.asanyarray(arr, dtype=float), size=window, mode="nearest").tolist()
    except ImportError:
        arr_np = np.asanyarray(arr, dtype=float)
        n = len(arr_np)
        half = window // 2
        return [float(np.median(arr_np[max(0, i - half): min(n, i + half + 1)])) for i in range(n)]
