"""
Slider-level metrics: MAE/R² computation and single-slider-isolated JSON generation.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import MODEL_GROUPS

log = logging.getLogger("per_slider_delta_e")

# ---------------------------------------------------------------------------
# Slider discovery
# ---------------------------------------------------------------------------

def discover_sliders(df: pd.DataFrame) -> list[str]:
    """
    Scan predictions.csv columns for all Base_{X} / Custom_{X} pairs.
    Returns slider names (original casing from the CSV).
    """
    sliders = []
    seen_lower = set()
    for col in df.columns:
        if not col.startswith("Custom_"):
            continue
        slider = col[len("Custom_"):]
        base_col = f"Base_{slider}"
        if base_col not in df.columns:
            continue
        if slider.lower() not in seen_lower:
            sliders.append(slider)
            seen_lower.add(slider.lower())
    return sliders


# ---------------------------------------------------------------------------
# MAE / R²
# ---------------------------------------------------------------------------

def _calculate_mae(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _calculate_r2(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_mae_r2(df: pd.DataFrame, sliders: list[str]) -> pd.DataFrame:
    """
    Compute MAE and R² for each slider (Custom_{S} vs Base_{S}).
    Returns DataFrame with columns [Slider, MAE, R2].
    """
    results = []
    for slider in sliders:
        custom_col = f"Custom_{slider}"
        base_col = f"Base_{slider}"
        if custom_col not in df.columns or base_col not in df.columns:
            continue

        pred = pd.to_numeric(df[custom_col], errors="coerce")
        gt = pd.to_numeric(df[base_col], errors="coerce")
        mask = pred.notna() & gt.notna()
        if mask.sum() < 2:
            continue

        mae = _calculate_mae(gt[mask], pred[mask])
        r2 = _calculate_r2(gt[mask], pred[mask])
        results.append({"Slider": slider, "MAE": round(mae, 4), "R2": round(r2, 4)})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Single-slider-isolated JSON builder
# ---------------------------------------------------------------------------

def _safe_value(val):
    """Convert a value to JSON-serializable form, skipping NaN."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val) if isinstance(val, (float, np.floating)) else int(val)
    try:
        s = str(val)
        if "." in s:
            return float(s)
        return int(s)
    except (ValueError, TypeError):
        return str(val)


def build_multi_slider_json(
    df: pd.DataFrame,
    slider_groups: list[str],
    all_sliders: list[str],
    output_path: str,
) -> tuple[Optional[str], set[str], int]:
    """
    Build a single metrics JSON where:
      - "Base" dict = all Base_* values (ground truth, untouched)
      - "Custom_{Group}" dicts = Base_* values EXCEPT for the specific slider group,
                                 which use Custom_* values.
    """
    id_col = None
    for c in ["img_path", "image_id", "id_global", "image_name"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        log.error(f"No image ID column found. Available: {df.columns.tolist()}")
        return None, set(), 0

    # Pre-resolve target groups to actual column sets (only active sliders)
    resolved_groups = {}
    for grp in slider_groups:
        active, _missing = resolve_slider_group(grp, all_sliders)
        resolved_groups[grp] = {s.lower() for s in active}

    # Discover all slider names (from Base_* columns)
    all_slider_cols = {}
    for col in df.columns:
        if col.startswith("Base_"):
            slider = col[len("Base_"):]
            custom_col = f"Custom_{slider}"
            if custom_col in df.columns:
                all_slider_cols[slider] = (col, custom_col)

    # Initialize edits framework
    edits = {"Base": {}}
    for grp in slider_groups:
        edits[f"Custom_{grp}"] = {}

    for _, row in df.iterrows():
        img_path = row.get(id_col)
        if pd.isna(img_path):
            continue
        img_path = str(img_path)

        base_edits = {}
        for slider, (base_col, _) in all_slider_cols.items():
            base_val = _safe_value(row[base_col])
            if base_val is not None:
                base_edits[slider] = base_val
        
        if not base_edits:
            continue

        edits["Base"][img_path] = base_edits

        for grp, target_lower in resolved_groups.items():
            custom_edits = {}
            for slider, (base_col, custom_col) in all_slider_cols.items():
                base_val = _safe_value(row[base_col])
                if base_val is None:
                    continue
                
                if slider.lower() in target_lower:
                    custom_val = _safe_value(row[custom_col])
                    custom_edits[slider] = custom_val if custom_val is not None else base_val
                else:
                    custom_edits[slider] = base_val
            
            edits[f"Custom_{grp}"][img_path] = custom_edits

    # Identify and drop exact matches
    identical_groups = set()
    for grp in list(slider_groups):
        key = f"Custom_{grp}"
        if edits["Base"] == edits[key]:
            identical_groups.add(grp)
            del edits[key]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(edits, f, indent=2, ensure_ascii=False)

    n_images = len(edits["Base"])
    skipped = len(identical_groups)
    active = len(slider_groups) - skipped
    
    log.info(f"  Built JSON with {n_images} images. Combinations: {active} active : {slider_groups}, {skipped} identical (skipped) : {identical_groups}.")
    return output_path, identical_groups, n_images


def filter_sliders(
    all_raw_sliders: list[str],
    mae_r2_df: pd.DataFrame,
    explicit_sliders: Optional[list[str]],
    ignore_negative_r2: bool,
) -> list[tuple[str, list[str]]]:
    """
    Returns a list of (group_name, missing_sliders) tuples for active MODEL_GROUPS.

    Previous logic: Required ALL sliders in a group to be present (strict subset).
                    If even one slider was absent, the entire group was silently skipped.

    New logic:      A group is included if AT LEAST ONE slider is present in predictions.csv
                    (intersection >= 1). Any missing sliders are tracked and returned so they
                    can be reported in logs and summaries.
    """
    target_groups = []

    # 1. Determine explicitly requested groups (or all if None)
    if explicit_sliders:
        req_lower_map = {s.lower(): s for s in explicit_sliders}
        for grp_name in MODEL_GROUPS.keys():
            if grp_name.lower() in req_lower_map:
                target_groups.append(grp_name)
    else:
        target_groups = list(MODEL_GROUPS.keys())

    # Lowercase lookup for available raw sliders
    available_raw_lower = {s.lower() for s in all_raw_sliders}

    filtered_groups: list[tuple[str, list[str]]] = []
    for grp in target_groups:
        req_sliders = MODEL_GROUPS[grp]
        req_lower = {s.lower() for s in req_sliders}

        # 2. Include group if at least 1 slider is present (was: strict subset)
        present_lower = req_lower & available_raw_lower
        if not present_lower:
            log.info(f"  Skipping group '{grp}' — none of its sliders found in CSV")
            continue

        missing = [s for s in req_sliders if s.lower() not in available_raw_lower]

        # 3. Check negative Mean R² (only across present sliders)
        if ignore_negative_r2 and not mae_r2_df.empty:
            group_rows = mae_r2_df[mae_r2_df["Slider"].str.lower().isin(present_lower)]
            mean_r2 = group_rows["R2"].mean() if not group_rows.empty else -1.0
            if mean_r2 < 0.0:
                log.info(f"  Skipping group '{grp}' (Mean R²={mean_r2:.4f} < 0)")
                continue

        if missing:
            log.info(f"  Group '{grp}': {len(present_lower)}/{len(req_sliders)} sliders present. "
                     f"Missing: {missing}")
        else:
            log.info(f"  Group '{grp}': all {len(req_sliders)} sliders present")

        filtered_groups.append((grp, missing))

    return filtered_groups


def resolve_slider_group(group_name: str, all_raw_sliders: list[str]) -> tuple[list[str], list[str]]:
    """
    Given a MODEL_GROUP string, returns (active_sliders, missing_sliders).
    active_sliders are those actually found in the CSV; missing are defined but absent.
    """
    available_lower = {s.lower() for s in all_raw_sliders}

    if group_name in MODEL_GROUPS:
        defined = MODEL_GROUPS[group_name]
        active = [s for s in defined if s.lower() in available_lower]
        missing = [s for s in defined if s.lower() not in available_lower]
        return active, missing

    # Fallback to literal match
    for raw in all_raw_sliders:
        if raw.lower() == group_name.lower():
            return [raw], []
    return [group_name], []
