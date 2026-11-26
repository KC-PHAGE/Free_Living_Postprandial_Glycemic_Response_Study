# -*- coding: utf-8 -*-
"""
Data Preprocessing and Metric Computation Pipeline for the Free-Living PPGR Study.
Filters raw CGM and meal records, computes 24-h CGM and 3-h postprandial metrics,
and exports processed meal-level, person-level, and flowchart summary files.
"""

import argparse
import csv
import json
import os
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils.cgm_metrics import calc_cgm_consensus_metrics
from utils.stats_utils import calc_positive_iauc

JST = timezone(timedelta(hours=9))


def parse_iso(ts_str):
    """Parse ISO-8601 timestamp string into timezone-aware datetime."""
    ts_str = ts_str.strip()
    if not ts_str:
        return None
    try:
        return datetime(
            int(ts_str[0:4]), int(ts_str[5:7]), int(ts_str[8:10]),
            int(ts_str[11:13]), int(ts_str[14:16]), int(ts_str[17:19]),
            tzinfo=JST,
        )
    except Exception:
        return None


def parse_dob(val):
    """Parse date of birth from YYYY-MM-DD, YYYY/MM/DD, or Excel serial date."""
    val = (val or "").strip()
    if not val:
        return None
    try:
        n = float(val)
        if 10000 < n < 60000:
            return date(1899, 12, 30) + timedelta(days=int(n))
    except Exception:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(val[:10], fmt).date()
        except Exception:
            pass
    return None


def normalize_gender(g_en, g_loc=""):
    g_en_l = (g_en or "").strip().lower()
    g_loc_s = (g_loc or "").strip()
    if g_en_l == "female" or g_loc_s in ("F", "女性"):
        return "Female"
    if g_en_l == "male" or g_loc_s in ("M", "男性"):
        return "Male"
    return "Unknown"


def load_demographics(path, encoding="utf-8"):
    """Load participant demographics from a .csv or .xlsx (Sheet 2) file."""
    print(f"Loading participant demographics: {os.path.basename(path)}...")
    demographics = {}

    if path.lower().endswith(".xlsx"):
        with zipfile.ZipFile(path, "r") as z:
            ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            strings = ["".join(t.text for t in si.findall(".//main:t", ns) if t.text)
                       for si in ss_root.findall(".//main:si", ns)]
            s_root = ET.fromstring(z.read("xl/worksheets/sheet2.xml"))

            rows = []
            for r in s_root.findall(".//main:row", ns):
                row_vals = {}
                for c in r.findall(".//main:c", ns):
                    col_ref = "".join(ch for ch in c.attrib.get("r", "") if ch.isalpha())
                    t = c.attrib.get("t")
                    v = c.find(".//main:v", ns)
                    val = v.text if v is not None else ""
                    if t == "s" and val.isdigit():
                        val = strings[int(val)]
                    row_vals[col_ref] = val
                rows.append(row_vals)

        header = rows[0]
        col_map = {v: k for k, v in header.items()}
        id_col = col_map.get("glucose_flight_id", "A")
        gender_col = col_map.get("gender", "B")
        gender_loc_col = "C"
        dob_col = col_map.get("date_of_birth", "D")
        age_col = "E"

        for r in rows[1:]:
            gid = r.get(id_col, "").strip()
            if not gid:
                continue
            age_raw = r.get(age_col, "").strip()
            demographics[gid] = {
                "gender": normalize_gender(r.get(gender_col, ""), r.get(gender_loc_col, "")),
                "dob": parse_dob(r.get(dob_col, "")),
                "missing_age_flag": age_raw in ("", "不明", "Missing", "Unknown"),
            }
    else:
        with open(path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f)
            for r in reader:
                gid = (r.get("glucose_flight_id") or r.get("ID") or "").strip()
                if not gid:
                    continue
                age_raw = (r.get("age_group") or "").strip()
                demographics[gid] = {
                    "gender": normalize_gender(r.get("gender", "")),
                    "dob": parse_dob(r.get("date_of_birth", "")),
                    "missing_age_flag": age_raw in ("Missing", "Unknown", "不明"),
                }

    print(f"Loaded demographics for {len(demographics):,} registered participants.")
    return demographics


def compute_age_group_at_date(dob, ref_date, missing_flag=False):
    """Compute chronological age category at the start of CGM monitoring."""
    if missing_flag or dob is None or ref_date is None:
        return "Missing"
    age = ref_date.year - dob.year - ((ref_date.month, ref_date.day) < (dob.month, dob.day))
    if age < 18:
        return "Missing"
    if age < 20:
        return "18–19"
    if age < 30:
        return "20–29"
    if age < 40:
        return "30–39"
    if age < 50:
        return "40–49"
    if age < 60:
        return "50–59"
    if age < 70:
        return "60–69"
    return "≥70"


def run_pipeline(demographics_path, cgm_path, meals_path, out_dir, encoding="utf-8", end_date=None):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    cutoff_date = date.fromisoformat(end_date) if end_date else None

    print("=" * 80)
    print("Free-Living PPGR Data Preprocessing Pipeline")
    if cutoff_date is not None:
        print(f"Study cutoff date applied: <= {cutoff_date.isoformat()}")
    print("=" * 80)

    demographics = load_demographics(demographics_path, encoding=encoding)

    # 1. Load CGM records
    print(f"Loading CGM records: {os.path.basename(cgm_path)}...")
    cgm_raw = defaultdict(list)
    total_cgm_rows = 0
    with open(cgm_path, "r", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_idx = header.index("glucose_flight_id")
        time_idx = header.index("recorded_at")
        gl_idx = header.index("glucose")
        int_idx = header.index("interval_seconds") if "interval_seconds" in header else None
        for row in reader:
            total_cgm_rows += 1
            gid = row[id_idx].strip()
            dt = parse_iso(row[time_idx])
            if dt is None:
                continue
            if cutoff_date is not None and dt.date() > cutoff_date:
                continue
            try:
                gl = int(float(row[gl_idx]))
                interval = int(row[int_idx]) if (int_idx is not None and row[int_idx]) else 900
            except ValueError:
                continue
            cgm_raw[gid].append((dt, gl, interval))

    print(f"Total CGM rows: {total_cgm_rows:,}, participants with CGM data: {len(cgm_raw):,}")

    # 2. Segment monitoring episodes (>=24 h gap), deduplicate (<=2 min & identical value),
    #    exclude Day 1, require >=10 valid days (>=68 readings/day) across Days 2-14, select 1 episode/person.
    valid_cgm_analysis = {}
    cgm_metrics_dict = {}
    valid_participants_cgm = set()

    count_registered = len(demographics)
    count_cgm_users = len(cgm_raw)
    count_excluded_no_cgm = count_registered - count_cgm_users

    for gid, records in cgm_raw.items():
        records.sort(key=lambda x: x[0])
        segments = []
        cur_m = []
        for i, rec in enumerate(records):
            if i == 0 or (rec[0] - records[i - 1][0]).total_seconds() / 3600.0 < 24.0:
                cur_m.append(rec)
            else:
                segments.append(cur_m)
                cur_m = [rec]
        if cur_m:
            segments.append(cur_m)

        valid_candidates = []
        for m_records in segments:
            cleaned = []
            for j, rec in enumerate(m_records):
                if j == 0:
                    cleaned.append(rec)
                else:
                    gap_min = (rec[0] - m_records[j - 1][0]).total_seconds() / 60.0
                    same_gl = rec[1] == m_records[j - 1][1]
                    if not (gap_min <= 2.0 and same_gl):
                        cleaned.append(rec)
            if not cleaned:
                continue

            m_start_date = cleaned[0][0].date()
            day_buckets = defaultdict(list)
            for r in cleaned:
                day_offset = (r[0].date() - m_start_date).days
                if 1 <= day_offset < 14:
                    day_buckets[day_offset + 1].append(r)

            valid_day_buckets = {d: recs for d, recs in day_buckets.items() if len(recs) >= 68}
            n_valid_days = len(valid_day_buckets)
            if n_valid_days < 10:
                continue

            valid_recs = [r for d in sorted(valid_day_buckets.keys()) for r in valid_day_buckets[d]]
            valid_recs.sort(key=lambda x: x[0])
            valid_dates = set(r[0].date() for r in valid_recs)
            n_cgm = len(valid_recs)
            completeness = (n_cgm / (13.0 * 96.0)) * 100.0

            gl_arr = np.asarray([r[1] for r in valid_recs], dtype=float)
            cgm_res = calc_cgm_consensus_metrics(gl_arr)

            valid_candidates.append({
                "ID": gid,
                "n_CGM": n_cgm,
                "n_valid_days": n_valid_days,
                "Valid_CGM_days": n_valid_days,
                "Mean_glucose": cgm_res["mean_glucose"],
                "SD": cgm_res["sd"],
                "CV": cgm_res["cv"],
                "TIR_70_180": cgm_res["tir"],
                "TAR_over_180": cgm_res["tar_total"],
                "TBR_under_70": cgm_res["tbr_total"],
                "GMI": cgm_res["gmi"],
                "completeness_pct": completeness,
                "measurement_start": valid_recs[0][0],
                "measurement_end": valid_recs[-1][0],
                "start_date": m_start_date,
                "valid_dates": valid_dates,
                "d2_14": valid_recs,
            })

        if valid_candidates:
            best_m = max(valid_candidates, key=lambda x: (x["n_valid_days"], x["completeness_pct"], x["n_CGM"]))
            valid_cgm_analysis[gid] = best_m["d2_14"]
            cgm_metrics_dict[gid] = best_m
            valid_participants_cgm.add(gid)

    print(f"Eligible CGM participants (>=10 valid days, 1 period/person): {len(valid_participants_cgm):,}")

    # 3. Load meals and apply 5 postprandial quality criteria
    print(f"Loading meal logs: {os.path.basename(meals_path)}...")
    total_recorded_eating_events = 0
    count_invalid_time = 0
    count_duplicate_records = 0
    valid_meals_by_id = {}
    raw_meals_by_id = defaultdict(list)

    with open(meals_path, "r", encoding=encoding, errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        mid_idx = header.index("meal_id")
        gid_idx = header.index("glucose_flight_id")
        time_idx = header.index("started_at")
        cat_idx = header.index("category")

        for row in reader:
            total_recorded_eating_events += 1
            mid = row[mid_idx]
            m_time = parse_iso(row[time_idx])
            if m_time is None:
                count_invalid_time += 1
                continue
            if cutoff_date is not None and m_time.date() > cutoff_date:
                continue
            if mid in valid_meals_by_id:
                count_duplicate_records += 1
                continue

            gid = row[gid_idx].strip()
            meal_item = {
                "meal_id": mid,
                "ID": gid,
                "meal_time": m_time,
                "meal_type": row[cat_idx].strip(),
            }
            valid_meals_by_id[mid] = meal_item
            raw_meals_by_id[gid].append(meal_item)

    print(
        f"Total meal rows: {total_recorded_eating_events:,} "
        f"(unique meals: {len(valid_meals_by_id):,}, duplicates: {count_duplicate_records:,}, invalid time: {count_invalid_time:,})"
    )

    exc_no_valid_cgm = 0
    exc_outside_day2_14 = 0
    exc_baseline_insufficient = 0
    exc_post_cgm_insufficient = 0
    exc_cannot_interpolate_180 = 0
    exc_overlap_180min = 0

    eligible_meals = []
    eligible_meals_overlap_allowed = []

    for gid, m_list in raw_meals_by_id.items():
        if gid not in valid_cgm_analysis:
            exc_no_valid_cgm += len(m_list)
            continue

        cgm_records = valid_cgm_analysis[gid]
        cgm_info = cgm_metrics_dict[gid]
        m_start_date = cgm_info["start_date"]
        m_list.sort(key=lambda x: x["meal_time"])

        for i, m in enumerate(m_list):
            m_time = m["meal_time"]
            if not (cgm_info["measurement_start"] <= m_time <= cgm_info["measurement_end"]):
                exc_outside_day2_14 += 1
                continue
            if m_time.date() not in cgm_info["valid_dates"]:
                exc_outside_day2_14 += 1
                continue

            meal_day_no = (m_time.date() - m_start_date).days + 1
            if not (2 <= meal_day_no <= 14):
                exc_outside_day2_14 += 1
                continue

            rel_cgm = []
            for r in cgm_records:
                minute = (r[0] - m_time).total_seconds() / 60.0
                if -35.0 <= minute <= 200.0:
                    rel_cgm.append((minute, r[1]))

            # Criterion 2: Pre-meal baseline (-30 <= t <= 0 min, >= 2 readings)
            pre_pts = [pt for pt in rel_cgm if -30.0 <= pt[0] <= 0.0]
            pre_pts.sort(key=lambda x: x[0], reverse=True)
            if len(pre_pts) < 2:
                exc_baseline_insufficient += 1
                continue

            # Criterion 3: Postprandial completeness (0 <= t <= 180 min, 10..13 readings)
            post_pts = [pt for pt in rel_cgm if 0.0 <= pt[0] <= 180.0]
            if not (10 <= len(post_pts) <= 13):
                exc_post_cgm_insufficient += 1
                continue

            baseline = float(np.mean([pre_pts[0][1], pre_pts[1][1]]))

            # Criterion 4: 180-min boundary linear interpolation (bracket <= 30 min)
            pts_before_180 = [pt for pt in rel_cgm if pt[0] <= 180.0]
            pts_after_180 = [pt for pt in rel_cgm if pt[0] >= 180.0]
            if not pts_before_180 or not pts_after_180:
                exc_cannot_interpolate_180 += 1
                continue

            b_pt = max(pts_before_180, key=lambda x: x[0])
            a_pt = min(pts_after_180, key=lambda x: x[0])
            before_min, before_gl = b_pt
            after_min, after_gl = a_pt
            bracket_gap = after_min - before_min
            if not (before_min <= 180.0 <= after_min and bracket_gap <= 30.0):
                exc_cannot_interpolate_180 += 1
                continue

            if bracket_gap == 0.0:
                gl_180 = float(before_gl)
            else:
                gl_180 = float(before_gl + (after_gl - before_gl) * (180.0 - before_min) / bracket_gap)

            post_gls = np.asarray([pt[1] for pt in post_pts], dtype=float)
            post_times = np.asarray([pt[0] for pt in post_pts], dtype=float)
            peak = float(np.max(post_gls))
            peak_indices = np.where(post_gls == peak)[0]
            time_to_peak = float(post_times[peak_indices[0]])
            delta_peak = peak - baseline
            mean_post = float(np.mean(post_gls))
            sd_post = float(np.std(post_gls, ddof=1)) if len(post_gls) > 1 else 0.0
            cv_post = (sd_post / mean_post * 100.0) if mean_post > 0 else 0.0

            curve_times = [0.0]
            curve_gls = [baseline]
            for pt in rel_cgm:
                if 0.0 < pt[0] < 180.0:
                    curve_times.append(pt[0])
                    curve_gls.append(float(pt[1]))
            curve_times.append(180.0)
            curve_gls.append(gl_180)

            pts_matrix = np.column_stack([curve_times, curve_gls])
            iauc_positive = calc_positive_iauc(pts_matrix, baseline=baseline)

            t_arr = np.asarray(curve_times, dtype=float)
            y_inc = np.maximum(np.asarray(curve_gls, dtype=float) - baseline, 0.0)
            iauc_naive = float(np.sum(0.5 * (y_inc[:-1] + y_inc[1:]) * np.diff(t_arr)))

            meal_dict = {
                "meal_id": m["meal_id"],
                "ID": gid,
                "meal_day_no": meal_day_no,
                "meal_type": m["meal_type"],
                "meal_time": m_time.isoformat(),
                "Baseline": baseline,
                "Peak": peak,
                "DeltaPeak": delta_peak,
                "Time_to_peak": time_to_peak,
                "iAUC": iauc_positive,
                "iAUC_naive": iauc_naive,
                "Mean": mean_post,
                "CV": cv_post,
            }
            eligible_meals_overlap_allowed.append(meal_dict)

            # Criterion 5: Non-overlapping 3-h postprandial window
            has_overlap = False
            if i + 1 < len(m_list):
                gap_to_next = (m_list[i + 1]["meal_time"] - m_time).total_seconds() / 60.0
                if 0.0 <= gap_to_next < 180.0:
                    has_overlap = True

            if has_overlap:
                exc_overlap_180min += 1
            else:
                eligible_meals.append(meal_dict)

    primary_meal_count = len(eligible_meals)
    primary_meal_participants = len(set(m["ID"] for m in eligible_meals))
    overlap_allowed_meal_count = len(eligible_meals_overlap_allowed)
    overlap_allowed_participants = len(set(m["ID"] for m in eligible_meals_overlap_allowed))

    print(
        f"Sensitivity dataset: {overlap_allowed_meal_count:,} meals ({overlap_allowed_participants:,} participants); "
        f"Primary meal dataset: {primary_meal_count:,} meals ({primary_meal_participants:,} participants)"
    )

    # 4. Person-level aggregation
    meals_by_participant = defaultdict(list)
    for m in eligible_meals:
        meals_by_participant[m["ID"]].append(m)

    person_rows = []
    for gid, mls in meals_by_participant.items():
        cgm_info = cgm_metrics_dict[gid]
        demo = demographics.get(gid, {"gender": "Unknown", "dob": None, "missing_age_flag": True})
        age_group = (
            "Missing"
            if gid not in demographics
            else compute_age_group_at_date(demo["dob"], cgm_info["start_date"], demo["missing_age_flag"])
        )

        delta_peaks = np.asarray([m["DeltaPeak"] for m in mls], dtype=float)
        iaucs = np.asarray([m["iAUC"] for m in mls], dtype=float)
        time_to_peaks = np.asarray([m["Time_to_peak"] for m in mls], dtype=float)
        baselines = np.asarray([m["Baseline"] for m in mls], dtype=float)
        peaks = np.asarray([m["Peak"] for m in mls], dtype=float)

        dp_p25, dp_median, dp_p75 = np.percentile(delta_peaks, [25.0, 50.0, 75.0])
        iauc_p25, iauc_median, iauc_p75 = np.percentile(iaucs, [25.0, 50.0, 75.0])

        person_rows.append({
            "ID": gid,
            "gender": demo["gender"],
            "age_group": age_group,
            "Valid_CGM_days": cgm_info["Valid_CGM_days"],
            "completeness_pct": cgm_info["completeness_pct"],
            "Mean_glucose": cgm_info["Mean_glucose"],
            "SD": cgm_info["SD"],
            "CV": cgm_info["CV"],
            "TIR_70_180": cgm_info["TIR_70_180"],
            "TAR_over_180": cgm_info["TAR_over_180"],
            "TBR_under_70": cgm_info["TBR_under_70"],
            "GMI": cgm_info["GMI"],
            "eligible_meals_count": len(mls),
            "DeltaPeak_median": float(dp_median),
            "DeltaPeak_p25": float(dp_p25),
            "DeltaPeak_p75": float(dp_p75),
            "DeltaPeak_IQR": float(dp_p75 - dp_p25),
            "iAUC_median": float(iauc_median),
            "iAUC_p25": float(iauc_p25),
            "iAUC_p75": float(iauc_p75),
            "iAUC_IQR": float(iauc_p75 - iauc_p25),
            "TimeToPeak_median": float(np.median(time_to_peaks)),
            "Baseline_median": float(np.median(baselines)),
            "Peak_median": float(np.median(peaks)),
        })

    def write_csv(filename, rows):
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {path} ({len(rows):,} rows)")

    write_csv("paper_meal_level.csv", eligible_meals)
    write_csv("paper_meal_level_all_with_overlap.csv", eligible_meals_overlap_allowed)
    write_csv("paper_person_level.csv", person_rows)

    k_stats = {f"k>={k}": sum(1 for mls in meals_by_participant.values() if len(mls) >= k) for k in [1, 5, 10, 15, 20]}
    total_unique_meals = len(valid_meals_by_id)
    flowchart_counts = {
        "step1_registered": count_registered,
        "step1_excluded_no_cgm": count_excluded_no_cgm,
        "step1_cgm_users": count_cgm_users,
        "step1_valid_cgm70_participants": len(valid_participants_cgm),
        "step2_app_total_eating_events": total_recorded_eating_events,
        "step2_app_unique_meals": total_unique_meals,
        "step2_app_participants_with_meals": len(raw_meals_by_id),
        "step2_excluded_no_valid_cgm": exc_no_valid_cgm,
        "step2_meals_eligible_cohort": total_unique_meals - exc_no_valid_cgm,
        "step3_exclusions": {
            "outside_valid_cgm_period": exc_outside_day2_14,
            "insufficient_baseline_cgm": exc_baseline_insufficient,
            "insufficient_post_cgm": exc_post_cgm_insufficient,
            "cannot_interpolate_180min": exc_cannot_interpolate_180,
            "total_3h_quality_exclusions": exc_outside_day2_14 + exc_baseline_insufficient + exc_post_cgm_insufficient + exc_cannot_interpolate_180,
            "subtotal_valid_meals_with_overlap": overlap_allowed_meal_count,
            "subtotal_participants_with_overlap": overlap_allowed_participants,
            "subsequent_meal_within_180min": exc_overlap_180min,
        },
        "step4_primary_meal_level_cohort": {
            "meals": primary_meal_count,
            "participants": primary_meal_participants,
        },
        "step5_person_level_k_stats": k_stats,
        "step5_primary_person_level_k10": {"participants": k_stats["k>=10"]},
    }

    json_path = os.path.join(out_dir, "flowchart_counts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flowchart_counts, f, indent=2, ensure_ascii=False)
    print(f"  Saved {json_path}")
    print(f"Pipeline completed in {time.time() - t_start:.2f}s.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CGM & meal logs for PPGR analysis.")
    parser.add_argument("--demographics", required=True, help="Path to participant demographics (.csv or .xlsx)")
    parser.add_argument("--cgm", required=True, help="Path to CGM glucose records CSV")
    parser.add_argument("--meals", required=True, help="Path to meal logs CSV")
    parser.add_argument("--out-dir", default="./processed_data", help="Directory for processed outputs")
    parser.add_argument("--encoding", default="utf-8", help="CSV text encoding (e.g., utf-8 or cp932)")
    parser.add_argument("--end-date", default=None, help="Optional inclusive study end date cutoff (YYYY-MM-DD)")
    args = parser.parse_args()
    run_pipeline(args.demographics, args.cgm, args.meals, args.out_dir, encoding=args.encoding, end_date=args.end_date)


if __name__ == "__main__":
    main()
