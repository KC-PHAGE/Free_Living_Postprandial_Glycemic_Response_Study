# -*- coding: utf-8 -*-
"""
Core 24-h CGM consensus metric calculations (ATTD / ADA standards).
"""

import numpy as np


def calc_cgm_consensus_metrics(glucose_readings, target_range=(70.0, 180.0)):
    """Compute international consensus CGM metrics (Battelino et al., 2019; Bergenstal et al., 2018)."""
    gl = np.asanyarray(glucose_readings, dtype=float)
    gl = gl[~np.isnan(gl)]

    if gl.size == 0:
        return {
            "n_readings": 0, "mean_glucose": 0.0, "sd": 0.0, "cv": 0.0, "gmi": 0.0,
            "tir": 0.0, "tar_total": 0.0, "tar_level1": 0.0, "tar_level2": 0.0,
            "tbr_total": 0.0, "tbr_level1": 0.0, "tbr_level2": 0.0,
            "lbgi": 0.0, "hbgi": 0.0, "j_index": 0.0,
        }

    mean_gl = float(np.mean(gl))
    sd_gl = float(np.std(gl, ddof=1)) if gl.size > 1 else 0.0
    cv_gl = (sd_gl / mean_gl * 100.0) if mean_gl > 0 else 0.0
    gmi = float(3.31 + 0.02392 * mean_gl)

    low_bound, high_bound = target_range
    tir = float(np.mean((gl >= low_bound) & (gl <= high_bound)) * 100.0)
    tar_total = float(np.mean(gl > high_bound) * 100.0)
    tar_level1 = float(np.mean((gl > high_bound) & (gl <= 250.0)) * 100.0)
    tar_level2 = float(np.mean(gl > 250.0) * 100.0)

    tbr_total = float(np.mean(gl < low_bound) * 100.0)
    tbr_level1 = float(np.mean((gl >= 54.0) & (gl < low_bound)) * 100.0)
    tbr_level2 = float(np.mean(gl < 54.0) * 100.0)

    gl_safe = np.maximum(gl, 1.0)
    f_g = 1.509 * ((np.log(gl_safe) ** 1.084) - 5.381)
    r_g = 10.0 * (f_g ** 2)
    lbgi = float(np.mean(np.where(f_g < 0.0, r_g, 0.0)))
    hbgi = float(np.mean(np.where(f_g > 0.0, r_g, 0.0)))
    j_index = float(0.001 * ((mean_gl + sd_gl) ** 2))

    return {
        "n_readings": int(gl.size),
        "mean_glucose": mean_gl,
        "sd": sd_gl,
        "cv": cv_gl,
        "gmi": gmi,
        "tir": tir,
        "tar_total": tar_total,
        "tar_level1": tar_level1,
        "tar_level2": tar_level2,
        "tbr_total": tbr_total,
        "tbr_level1": tbr_level1,
        "tbr_level2": tbr_level2,
        "lbgi": lbgi,
        "hbgi": hbgi,
        "j_index": j_index,
    }


def calc_modd(daily_glucose_dict):
    """Compute Mean of Daily Differences (MODD) across consecutive monitoring days."""
    days = sorted(daily_glucose_dict.keys())
    if len(days) < 2:
        return 0.0

    diffs = []
    for i in range(len(days) - 1):
        if days[i + 1] - days[i] != 1:
            continue
        d1 = daily_glucose_dict[days[i]]
        d2 = daily_glucose_dict[days[i + 1]]
        min_len = min(len(d1), len(d2))
        if min_len > 0:
            diff_arr = np.abs(np.asarray(d1[:min_len], dtype=float) - np.asarray(d2[:min_len], dtype=float))
            valid = ~np.isnan(diff_arr)
            if np.any(valid):
                diffs.extend(diff_arr[valid])

    return float(np.mean(diffs)) if diffs else 0.0
