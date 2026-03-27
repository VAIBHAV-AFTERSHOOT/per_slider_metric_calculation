"""
Slider-level metrics: MAE/R² computation and single-slider-isolated JSON generation.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("per_slider_delta_e")

KNOWN_NON_REGRESSION_SLIDERS = {
    "parametrichighlightsplit",
    "parametricmidtonesplit",
    "parametricshadowsplit",
}

# Temperature and Tint are always treated together as "WB"
WB_SLIDERS = {"temperature", "tint"}


# ---------------------------------------------------------------------------
# Slider discovery
# ---------------------------------------------------------------------------

def discover_sliders(df: pd.DataFrame) -> list[str]:
    """
    Scan predictions.csv columns for all Base_{X} / Custom_{X} pairs.
    Returns slider names (original casing from the CSV).
    Excludes known non-regression sliders.
    """
    sliders = []
    seen_lower = set()
    for col in df.columns:
        if not col.startswith("Custom_"):
            continue
        slider = col[len("Custom_"):]
        if slider.lower() in KNOWN_NON_REGRESSION_SLIDERS:
            continue
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
    
    If a Custom_{Group} dictionary ends up being perfectly identical to Base
    across all images, it is bypassed and returned in the identical_groups set.

    Args:
        df: predictions DataFrame
        slider_groups: list of grouped slider names to isolate (e.g. from filter_sliders)
        all_sliders: original list of all discovered sliders in the CSV
        output_path: where to write the JSON

    Returns:
        (Path to written JSON, set of identical groups, image count)
    """
    id_col = None
    for c in ["img_path", "image_id", "id_global", "image_name"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        log.error(f"No image ID column found. Available: {df.columns.tolist()}")
        return None, set(), 0

    # Pre-resolve target groups to actual column sets
    resolved_groups = {}
    for grp in slider_groups:
        resolved_groups[grp] = {s.lower() for s in resolve_slider_group(grp, all_sliders)}

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
    
    log.info(f"  Built JSON with {n_images} images. Combinations: {active} active, {skipped} identical (skipped).")
    return output_path, identical_groups, n_images


# Need os for makedirs
import os


def filter_sliders(
    all_sliders: list[str],
    mae_r2_df: pd.DataFrame,
    explicit_sliders: Optional[list[str]] = None,
    ignore_negative_r2: bool = False,
) -> list[str]:
    """
    Filter the slider list based on user args.

    - Case-insensitive matching for explicit_sliders
    - Optionally skip sliders with negative R²
    - Always groups Temperature + Tint together as "WB"

    Returns a list of slider names (or ["WB"] pseudo-entry for the temp+tint group).
    """
    # Build a case-insensitive index of available sliders
    lower_to_original = {s.lower(): s for s in all_sliders}
    available_lower = set(lower_to_original.keys())

    # Start with all sliders if no explicit list
    if explicit_sliders:
        selected_lower = set()
        for s in explicit_sliders:
            sl = s.lower()
            if sl in available_lower:
                selected_lower.add(sl)
            elif sl == "wb":
                # WB = Temperature + Tint
                selected_lower.update(WB_SLIDERS & available_lower)
            else:
                log.warning(f"Slider '{s}' not found in predictions.csv (case-insensitive)")
    else:
        selected_lower = available_lower.copy()

    # Optionally filter negative R²
    if ignore_negative_r2 and not mae_r2_df.empty:
        negative_sliders = set(
            mae_r2_df[mae_r2_df["R2"] < 0]["Slider"].str.lower().tolist()
        )
        before = len(selected_lower)
        selected_lower -= negative_sliders
        dropped = before - len(selected_lower)
        if dropped:
            log.info(f"Dropped {dropped} slider(s) with negative R²")

    # Group Temperature + Tint into "WB"
    has_temp = "temperature" in selected_lower
    has_tint = "tint" in selected_lower

    result = []
    wb_added = False

    for sl in sorted(selected_lower):
        if sl in WB_SLIDERS:
            if not wb_added and has_temp and has_tint:
                result.append("WB")
                wb_added = True
            elif not wb_added:
                # Only one of temp/tint available, add individually
                result.append(lower_to_original.get(sl, sl))
        else:
            result.append(lower_to_original.get(sl, sl))

    return result


def resolve_slider_group(slider_entry: str, all_sliders: list[str]) -> list[str]:
    """
    Resolve a slider entry to actual column names.
    "WB" -> ["Temperature", "Tint"] (using actual casing from CSV)
    Otherwise -> [slider_entry]
    """
    if slider_entry == "WB":
        lower_to_original = {s.lower(): s for s in all_sliders}
        group = []
        for wb in sorted(WB_SLIDERS):
            if wb in lower_to_original:
                group.append(lower_to_original[wb])
        return group
    return [slider_entry]
