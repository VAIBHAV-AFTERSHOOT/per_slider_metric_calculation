#!/usr/bin/env python3
"""
Per-Slider Delta-E Metrics Pipeline
=====================================

Given a list of PIDs, downloads their metrics directories from GCS,
computes MAE/R² per slider, builds single-slider-isolated JSONs,
and runs Delta-E metrics to measure each slider's individual visual impact.

Usage:
    python -m per_slider_delta_e.main \\
        --pids_csv /path/to/pids.csv \\
        --profiles_csv /path/to/PROFILES_CSV.csv \\
        --output_dir /path/to/output

    # Explicit sliders:
    python -m per_slider_delta_e.main \\
        --pids_csv pids.csv \\
        --profiles_csv PROFILES_CSV.csv \\
        --output_dir output \\
        --sliders WB Exposure Contrast

    # Ignore sliders with negative R²:
    python -m per_slider_delta_e.main \\
        --pids_csv pids.csv \\
        --profiles_csv PROFILES_CSV.csv \\
        --output_dir output \\
        --ignore_negative_r2
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Setup path for local imports
_MY_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_MY_DIR)
sys.path.insert(0, _MY_DIR)  # Ensure local directory is in path first

from download import download_metrics_dir
from slider_metrics import (
    discover_sliders,
    compute_mae_r2,
    build_multi_slider_json,
    filter_sliders,
    resolve_slider_group,
)
from delta_e_runner import run_delta_e_for_profile

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))

log = logging.getLogger("per_slider_delta_e")
log.setLevel(logging.INFO)

# Console handler setup
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
log.addHandler(_console_handler)


# ---------------------------------------------------------------------------
# PID CSV loader (flexible column name detection)
# ---------------------------------------------------------------------------

_PID_COLUMN_NAMES = [
    "pid", "pids", "PID", "PIDs", "profile_id", "Profile ID",
    "profile_key", "key", "from_profile_key",
]


def load_pids(csv_path: str) -> list[str]:
    """Load PIDs from a CSV with flexible column name detection."""
    df = pd.read_csv(csv_path)
    for col_name in _PID_COLUMN_NAMES:
        if col_name in df.columns:
            pids = df[col_name].dropna().astype(str).unique().tolist()
            log.info(f"Loaded {len(pids)} PIDs from column '{col_name}' in {csv_path}")
            return pids

    # Fallback: if single column, just use it
    if len(df.columns) == 1:
        pids = df.iloc[:, 0].dropna().astype(str).unique().tolist()
        log.info(f"Loaded {len(pids)} PIDs from single-column CSV {csv_path}")
        return pids

    log.error(f"Could not find PID column in {csv_path}. "
              f"Available columns: {df.columns.tolist()}")
    sys.exit(1)


def lookup_user_id(profiles_df: pd.DataFrame, pid: str) -> Optional[str]:
    """Look up user_id from the profiles CSV by profile key."""
    row = profiles_df[profiles_df["key"].astype(str) == pid]
    if row.empty:
        return None
    return str(row.iloc[0]["user_id"])


# ---------------------------------------------------------------------------
# Incremental summary
# ---------------------------------------------------------------------------

def _fmt(val, decimals=2):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def write_incremental_summary(
    summary_file: str,
    pid: str,
    slider_label: str,
    mae: float,
    r2: float,
    delta_e_result: Optional[dict],
    is_header: bool = False,
    missing_sliders: Optional[list[str]] = None,
    overall_csv: Optional[str] = None,
    pid_summary_txt: Optional[str] = None,
    pid_summary_csv: Optional[str] = None,
):
    """Append a row to the running summary file, overall CSV, and per-PID summary files."""
    delta_e = _fmt(delta_e_result.get("mean_delta_e_mean") if delta_e_result else None)
    delta_e_p75 = _fmt(delta_e_result.get("mean_delta_e_p75") if delta_e_result else None)
    psnr = _fmt(delta_e_result.get("psnr_mean") if delta_e_result else None)
    ssim = _fmt(delta_e_result.get("ssim_mean") if delta_e_result else None, 4)
    images = str(delta_e_result.get("metrics_images_count") if delta_e_result else "N/A")
    missing_str = ", ".join(missing_sliders) if missing_sliders else ""

    header = (
        f"{'PID':<38} | {'Group':<28} | {'MAE':<8} | {'R2':<8} | "
        f"{'Delta-E':<8} | {'dE-p75':<8} | {'PSNR':<8} | {'SSIM':<8} | "
        f"{'Images':<8} | {'Missing Sliders'}"
    )
    row = (
        f"{pid:<38} | {slider_label:<28} | {_fmt(mae, 4):<8} | {_fmt(r2, 4):<8} | "
        f"{delta_e:<8} | {delta_e_p75:<8} | {psnr:<8} | {ssim:<8} | "
        f"{images:<8} | {missing_str}"
    )

    if is_header:
        sep = "-" * len(header)
        with open(summary_file, "w") as f:
            f.write(header + "\n")
            f.write(sep + "\n")
        print("\n" + header)
        print(sep)

    with open(summary_file, "a") as f:
        f.write(row + "\n")
    print(row)

    # --- Incremental overall CSV ---
    csv_row = {
        "pid": pid,
        "group": slider_label,
        "mae": mae,
        "r2": r2,
        "delta_e": delta_e_result.get("mean_delta_e_mean") if delta_e_result else None,
        "delta_e_p75": delta_e_result.get("mean_delta_e_p75") if delta_e_result else None,
        "psnr": delta_e_result.get("psnr_mean") if delta_e_result else None,
        "ssim": delta_e_result.get("ssim_mean") if delta_e_result else None,
        "images": delta_e_result.get("metrics_images_count") if delta_e_result else None,
        "missing_sliders": missing_str,
        "success": delta_e_result.get("success", False) if delta_e_result else False,
    }
    row_df = pd.DataFrame([csv_row])

    if overall_csv:
        write_header = is_header or not os.path.exists(overall_csv)
        row_df.to_csv(overall_csv, mode="a", header=write_header, index=False)

    # --- Per-PID summary files ---
    if pid_summary_csv:
        pid_csv_header = not os.path.exists(pid_summary_csv)
        row_df.to_csv(pid_summary_csv, mode="a", header=pid_csv_header, index=False)

    if pid_summary_txt:
        pid_txt_exists = os.path.exists(pid_summary_txt)
        with open(pid_summary_txt, "a") as f:
            if not pid_txt_exists:
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
            f.write(row + "\n")


# ---------------------------------------------------------------------------
# Process a single PID
# ---------------------------------------------------------------------------

def process_pid(
    pid: str,
    user_id: str,
    output_dir: str,
    explicit_sliders: Optional[list[str]],
    ignore_negative_r2: bool,
    overall_deltaE: bool,
    no_cache: bool,
    summary_file: str,
    overall_csv: str,
    is_first_pid: bool,
    all_results: list,
):
    """Full pipeline for a single PID."""
    pid_output = os.path.join(output_dir, pid)
    os.makedirs(pid_output, exist_ok=True)

    # Per-PID summary paths
    pid_summary_csv = os.path.join(pid_output, "pid_summary.csv")
    pid_summary_txt = os.path.join(pid_output, "pid_summary.txt")
    # Remove stale per-PID summaries from prior runs
    for f in [pid_summary_csv, pid_summary_txt]:
        if os.path.exists(f):
            os.remove(f)
    
    # Setup profile-specific logger
    profile_log_path = os.path.join(pid_output, "profile_run.log")
    profile_handler = logging.FileHandler(profile_log_path)
    profile_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(profile_handler)
    
    try:
        log.info("=" * 70)
        log.info(f"Processing PID: {pid}")
        log.info("=" * 70)

        # Step 1: Download metrics dir
        log.info("Step 1: Downloading metrics directory...")
        predictions_csv, exif_csv, dngs_dir = download_metrics_dir(user_id, pid, output_dir)
        if predictions_csv is None:
            log.error(f"[{pid[:8]}] Failed to download metrics — skipping")
            return

        # Step 2: Load predictions and discover sliders
        log.info("Step 2: Discovering sliders and computing MAE/R²...")
        df = pd.read_csv(predictions_csv)
        all_sliders = discover_sliders(df)
        log.info(f"  Found {len(all_sliders)} slider pairs in predictions.csv")

        if not all_sliders:
            log.warning(f"[{pid[:8]}] No slider pairs found — skipping")
            return

        # OVERALL Delta-E (if requested)
        if overall_deltaE:
            log.info("\n--- [OVERALL] Running Delta-E for all original predictions ---")
            try:
                from A_B_utils import csv_to_json_dynamic
                overall_json_path = os.path.join(pid_output, "overall_metrics.json")
                csv_to_json_dynamic(predictions_csv, overall_json_path)
                
                overall_results = run_delta_e_for_profile(
                    pid=pid,
                    json_path=overall_json_path,
                    comparisons=[("Base", "Custom")],
                    dngs_dir=dngs_dir,
                    output_dir=pid_output,
                    no_cache=False,
                )
                overall_result = overall_results.get("OVERALL", {})
                write_incremental_summary(
                    summary_file, pid, "OVERALL", None, None, overall_result,
                    is_header=is_first_pid,
                    overall_csv=overall_csv,
                    pid_summary_txt=pid_summary_txt,
                    pid_summary_csv=pid_summary_csv,
                )
                all_results.append({
                    "pid": pid,
                    "group": "OVERALL",
                    "mae": None,
                    "r2": None,
                    "delta_e": overall_result.get("mean_delta_e_mean") if overall_result else None,
                    "delta_e_p75": overall_result.get("mean_delta_e_p75") if overall_result else None,
                    "psnr": overall_result.get("psnr_mean") if overall_result else None,
                    "ssim": overall_result.get("ssim_mean") if overall_result else None,
                    "images": overall_result.get("metrics_images_count") if overall_result else None,
                    "missing_sliders": "",
                    "success": overall_result.get("success", False) if overall_result else False,
                })
                is_first_pid = False
            except ImportError:
                log.error("A_B_utils not found locally — skipping OVERALL Delta-E")

        # Step 3: Compute MAE/R² for all sliders
        mae_r2_df = compute_mae_r2(df, all_sliders)
        log.info(f"\n{'='*60}")
        log.info(f"MAE / R² Summary for {pid[:8]}")
        log.info(f"{'='*60}")
        for _, row in mae_r2_df.iterrows():
            log.info(f"  {row['Slider']:<30}  MAE={row['MAE']:.4f}  R2={row['R2']:.4f}")

        # Save MAE/R² CSV
        mae_csv = os.path.join(pid_output, "mae_r2_summary.csv")
        mae_r2_df.to_csv(mae_csv, index=False)
        log.info(f"  Saved MAE/R² to {mae_csv}")

        # Step 4: Filter sliders — returns list of (group_name, missing_sliders)
        selected_with_missing = filter_sliders(all_sliders, mae_r2_df, explicit_sliders, ignore_negative_r2)
        selected = [grp for grp, _ in selected_with_missing]
        missing_map = {grp: missing for grp, missing in selected_with_missing}
        log.info(f"  Selected model groups for Delta-E: {selected}")

        if not selected:
            log.warning(f"[{pid[:8]}] No model groups remaining after filtering — skipping")
            return

        # Step 5: Build a single multi-slider JSON
        json_dir = os.path.join(pid_output, "slider_jsons")
        json_path = os.path.join(json_dir, "combined_metrics.json")
        
        log.info(f"\n--- Building Multi-Slider JSON for {len(selected)} model groups ---")
        result_path, identical_groups, n_images = build_multi_slider_json(
            df, selected, all_sliders, json_path
        )

        if result_path is None:
            log.error(f"[{pid[:8]}] Failed to build multi-slider JSON — skipping")
            return

        # Prepare active comparisons
        comparisons = []
        for grp in selected:
            if grp not in identical_groups:
                comparisons.append(("Base", f"Custom_{grp}"))

        # Run bulk Delta-E
        bulk_results = {}
        if comparisons:
            bulk_results = run_delta_e_for_profile(
                pid=pid,
                json_path=json_path,
                comparisons=comparisons,
                dngs_dir=dngs_dir,
                output_dir=pid_output,
                no_cache=no_cache,
            )

        # Log incrementally
        for i, group_entry in enumerate(selected):
            label = group_entry
            group_missing = missing_map.get(label, [])

            # Dynamically compute MAE and R² means across ACTIVE (present) sliders only
            active_sliders, _ = resolve_slider_group(label, all_sliders)
            target_lower = {s.lower() for s in active_sliders}
            
            group_rows = mae_r2_df[mae_r2_df["Slider"].str.lower().isin(target_lower)]
            slider_mae = group_rows["MAE"].mean() if not group_rows.empty else None
            slider_r2 = group_rows["R2"].mean() if not group_rows.empty else None

            if label in identical_groups:
                log.info(f"  [{label}] Base and Custom identical — bypassing Delta-E (0.00)")
                delta_e_result = {
                    "slider": label,
                    "success": True,
                    "psnr_mean": 100.0,
                    "ssim_mean": 1.0,
                    "hist_bhattacharyya_mean": 0.0,
                    "mean_delta_e_mean": 0.0,
                    "mean_delta_e_p75": 0.0,
                    "metrics_images_count": n_images
                }
            else:
                delta_e_result = bulk_results.get(label)

            write_incremental_summary(
                summary_file, pid, label, slider_mae, slider_r2, delta_e_result,
                is_header=(is_first_pid and i == 0),
                missing_sliders=group_missing,
                overall_csv=overall_csv,
                pid_summary_txt=pid_summary_txt,
                pid_summary_csv=pid_summary_csv,
            )

            missing_str = ", ".join(group_missing) if group_missing else ""
            all_results.append({
                "pid": pid,
                "group": label,
                "mae": slider_mae,
                "r2": slider_r2,
                "delta_e": delta_e_result.get("mean_delta_e_mean") if delta_e_result else None,
                "delta_e_p75": delta_e_result.get("mean_delta_e_p75") if delta_e_result else None,
                "psnr": delta_e_result.get("psnr_mean") if delta_e_result else None,
                "ssim": delta_e_result.get("ssim_mean") if delta_e_result else None,
                "images": delta_e_result.get("metrics_images_count") if delta_e_result else None,
                "missing_sliders": missing_str,
                "success": delta_e_result.get("success", False) if delta_e_result else False,
            })

        # Cleanup DNGs if no_cache
        if no_cache and dngs_dir and os.path.exists(dngs_dir):
            import shutil
            shutil.rmtree(dngs_dir)
            log.info(f"[{pid[:8]}] Cleaned up DNGs directory")

    finally:
        # Ensure we always remove the profile handler
        log.removeHandler(profile_handler)
        profile_handler.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-Slider Delta-E Metrics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pids_csv", type=str, required=True,
        help="CSV file with PID column (auto-detected: pid, PID, Profile ID, key, etc.)",
    )
    parser.add_argument(
        "--profiles_csv", type=str,
        default="/home/ubuntu/workspace/vaibhav-tmp/PROFILES_CSV.csv",
        help="Profiles CSV with 'key' and 'user_id' columns (default: PROFILES_CSV.csv)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for all results",
    )
    parser.add_argument(
        "--sliders", nargs="+", default=None,
        help="Explicit list of sliders (case-insensitive). Use 'WB' for Temperature+Tint. "
             "Default: all sliders.",
    )
    parser.add_argument(
        "--ignore_negative_r2", action="store_true", default=False,
        help="Skip sliders with negative R² (default: False)",
    )
    parser.add_argument(
        "--overall_deltaE", action="store_true", default=False,
        help="First calculate standard full Delta-E for all sliders before per-slider isolation (default: False)",
    )
    parser.add_argument(
        "--no_cache", action="store_true", default=False,
        help="Delete DNGs and cache directories after each PID",
    )

    args = parser.parse_args()

    # Load profiles CSV
    log.info(f"Loading profiles CSV: {args.profiles_csv}")
    profiles_df = pd.read_csv(args.profiles_csv)

    # Load PIDs
    pids = load_pids(args.pids_csv)

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup central logger
    central_log_path = os.path.join(args.output_dir, "central_pipeline.log")
    central_handler = logging.FileHandler(central_log_path)
    central_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(central_handler)
    
    log.info(f"Processing {len(pids)} PID(s)")

    # Summary files
    timestamp = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.txt")
    overall_csv = os.path.join(args.output_dir, f"overall_summary_{timestamp}.csv")

    all_results = []

    for idx, pid in enumerate(pids):
        # Lookup user_id
        user_id = lookup_user_id(profiles_df, pid)
        if user_id is None:
            log.error(f"PID {pid} not found in profiles CSV — skipping")
            continue

        process_pid(
            pid=pid,
            user_id=user_id,
            output_dir=args.output_dir,
            explicit_sliders=args.sliders,
            ignore_negative_r2=args.ignore_negative_r2,
            overall_deltaE=args.overall_deltaE,
            no_cache=args.no_cache,
            summary_file=summary_file,
            overall_csv=overall_csv,
            is_first_pid=(idx == 0),
            all_results=all_results,
        )

    # Final aggregated CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        agg_csv = os.path.join(args.output_dir, f"aggregated_results_{timestamp}.csv")
        results_df.to_csv(agg_csv, index=False)
        log.info(f"\nAggregated results saved to: {agg_csv}")
        log.info(f"Incremental overall CSV: {overall_csv}")

        # Print per-group aggregate summary (mean across all PIDs)
        print(f"\n{'='*70}")
        print("AGGREGATE SUMMARY — Mean across all PIDs")
        print(f"{'='*70}")
        for group in results_df["group"].unique():
            subset = results_df[results_df["group"] == group]
            successful = subset[subset["success"] == True]
            print(
                f"  {group:<28} | "
                f"MAE={_fmt(subset['mae'].mean(), 4):<8} | "
                f"R2={_fmt(subset['r2'].mean(), 4):<8} | "
                f"Delta-E={_fmt(successful['delta_e'].mean()) if not successful.empty else 'N/A':<8} | "
                f"dE-p75={_fmt(successful['delta_e_p75'].mean()) if not successful.empty else 'N/A':<8} | "
                f"PIDs={len(subset)}"
            )

        # Append aggregate to summary file
        with open(summary_file, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write("AGGREGATE SUMMARY — Mean across all PIDs\n")
            f.write(f"{'='*70}\n")
            for group in results_df["group"].unique():
                subset = results_df[results_df["group"] == group]
                successful = subset[subset["success"] == True]
                f.write(
                    f"  {group:<28} | "
                    f"MAE={_fmt(subset['mae'].mean(), 4):<8} | "
                    f"R2={_fmt(subset['r2'].mean(), 4):<8} | "
                    f"Delta-E={_fmt(successful['delta_e'].mean()) if not successful.empty else 'N/A':<8} | "
                    f"dE-p75={_fmt(successful['delta_e_p75'].mean()) if not successful.empty else 'N/A':<8} | "
                    f"PIDs={len(subset)}\n"
                )

    log.info(f"\nSummary file: {summary_file}")
    log.info(f"Overall CSV: {overall_csv}")
    log.info("Done!")


if __name__ == "__main__":
    main()
