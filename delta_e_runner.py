"""
Delta-E / EditsMetricsPipeline runner for a single slider JSON.
"""
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from config import EDITOR_PATH,METRICS_WORKERS
log = logging.getLogger("per_slider_delta_e")

# Attempt to import local editing-ml pipeline wrapper
try:
    from get_metrics import calculate_metrics_for_profile, delete_xmp_files
    EDITING_ML_AVAILABLE = True
except ImportError as e:
    EDITING_ML_AVAILABLE = False
    log.warning(f"editing-ml (editlib) or local dependencies not available: {e} — Delta-E metrics will be skipped.")


def run_delta_e_for_profile(
    pid: str,
    json_path: str,
    comparisons: list[tuple[str, str]],
    dngs_dir: str,
    output_dir: str,
    no_cache: bool = False,
) -> dict[str, dict]:
    """
    Run the EditsMetricsPipeline on a multi-slider JSON concurrently.

    Args:
        pid: Profile ID
        json_path: Path to the combined multi-slider JSON
        comparisons: List of (source1, source2) tuples, e.g., [("Base", "Custom_Exposure"), ...]
        dngs_dir: Path to the extracted DNGs directory
        output_dir: Base output directory for this PID
        no_cache: Whether to delete cache after completion

    Returns:
        Dictionary mapping slider_label to its metrics result dict.
    """
    results = {}
    if not comparisons:
        return results

    if not EDITING_ML_AVAILABLE:
        log.warning(f"  [{pid[:8]}] Skipping Delta-E — editlib not available")
        return results

    import json as json_mod
    from editlib.metrics.pipeline import EditsMetricsPipeline, Editor

    # EDITOR_PATH = "/home/ubuntu/workspace/DNG_Converter/Adobe DNG Converter.exe"

    pipeline_out_dir = os.path.join(output_dir, "delta_e_comparisons")
    cache_dir = os.path.join(output_dir, "cache_bulk_metrics")

    try:
        # Clean up previous runs
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        if os.path.exists(pipeline_out_dir):
            shutil.rmtree(pipeline_out_dir)

        # Load JSON
        with open(json_path, "r") as f:
            json_dict = json_mod.load(f)

        imgs_dir = Path(dngs_dir)
        if not imgs_dir.exists():
            log.error(f"  [{pid[:8]}] DNGs directory not found: {imgs_dir}")
            return results

        # Clean XMP files
        delete_xmp_files(str(imgs_dir), dry_run=False)

        # Run pipeline
        pipeline = EditsMetricsPipeline(cache_dir=Path(cache_dir))

        # Clean XMP again
        delete_xmp_files(str(imgs_dir), dry_run=False)

        log.info(f"  [{pid[:8]}] Running bulk EditsMetricsPipeline on {len(comparisons)} comparisons...")
        pipeline.run(
            imgs_dir=imgs_dir,
            edits=json_dict,
            out_dir=Path(pipeline_out_dir),
            editor=Editor.AdobeDNGConverter,
            editor_path=EDITOR_PATH,
            workers=METRICS_WORKERS,
            keep_cache=True,
            comparisons=comparisons,
        )

        for source1, source2 in comparisons:
            if source2 == "Custom":
                slider_label = "OVERALL"
            elif source2.startswith("Custom_"):
                slider_label = source2[len("Custom_"):]
            else:
                continue
            
            stats_csv = os.path.join(pipeline_out_dir, f"{source1}_{source2}_stats.csv")
            
            sim_result = {
                "slider": slider_label,
                "success": False,
                "psnr_mean": None,
                "ssim_mean": None,
                "hist_bhattacharyya_mean": None,
                "mean_delta_e_mean": None,
                "mean_delta_e_p75": None,
                "metrics_images_count": None,
            }

            if os.path.exists(stats_csv):
                stats_df = pd.read_csv(stats_csv, index_col=0)
                if "mean" in stats_df.index:
                    mean_row = stats_df.loc["mean"]
                    sim_result["psnr_mean"] = mean_row.get("psnr", None)
                    sim_result["ssim_mean"] = mean_row.get("ssim", None)
                    sim_result["hist_bhattacharyya_mean"] = mean_row.get("hist_bhattacharyya", None)
                    sim_result["mean_delta_e_mean"] = mean_row.get("mean_delta_e", None)
                    
                    if "75%" in stats_df.index:
                        sim_result["mean_delta_e_p75"] = stats_df.loc["75%"].get("mean_delta_e", None)

                    if "count" in stats_df.index:
                        sim_result["metrics_images_count"] = int(stats_df.loc["count"].get("psnr", 0))

                    sim_result["success"] = True
                    log.info(
                        f"  [{slider_label}] Bulk Delta-E={sim_result['mean_delta_e_mean']:.2f} (P75={sim_result['mean_delta_e_p75']:.2f}), "
                        f"PSNR={sim_result['psnr_mean']:.2f}, "
                        f"SSIM={sim_result['ssim_mean']:.4f}, "
                        f"Images={sim_result['metrics_images_count']}"
                    )
            else:
                log.warning(f"  [{slider_label}] Stats CSV not found: {stats_csv}")
            
            results[slider_label] = sim_result

        # Cleanup cache if requested
        if no_cache and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            log.info(f"  [{pid[:8]}] Cleaned up bulk cache")

        return results

    except Exception as e:
        log.error(f"  [{pid[:8]}] Bulk Delta-E failed: {e}")
        return results
