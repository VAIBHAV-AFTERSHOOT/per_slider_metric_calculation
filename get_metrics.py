"""
Metrics Calculation Functions

This module provides functions for calculating metrics using EditsMetricsPipeline.
"""

from editlib.metrics.pipeline import EditsMetricsPipeline, Editor
from pathlib import Path
import shutil
import json
import os

from A_B_logger import get_logger
from config import EDITOR_PATH, METRICS_WORKERS, DEFAULT_COMPARISONS

logger = get_logger(__name__)


def delete_xmp_files(directory_path: str, dry_run: bool = False):
    """
    Deletes .xmp and .xmp.bak files from the specified directory.

    Args:
        directory_path (str): The path to the directory to clean.
        dry_run (bool): If True, only prints the files that would be deleted.
    """
    target_dir = Path(directory_path)

    if not target_dir.is_dir():
        logger.warning(f"Directory not found: {directory_path}")
        return 0

    deleted_count = 0
    patterns = ["*.xmp", "*.xmp.bak"]

    for pattern in patterns:
        for file_path in target_dir.glob(pattern):
            if file_path.is_file():
                if dry_run:
                    logger.debug(f"[DRY RUN] Would delete: {file_path}")
                else:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except OSError as e:
                        logger.error(f"Failed to delete {file_path}: {e}")

    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} XMP files from {directory_path}")
    
    return deleted_count


def calculate_metrics_for_profile(
    pid: str,
    profile_path: str,
    json_path: str,
    workers: int = None,
    comparisons: list = None
) -> dict:
    """
    Calculate metrics for a single profile using EditsMetricsPipeline.

    Args:
        pid: Profile ID
        profile_path: Base path to the profile directory
        json_path: Path to the metrics_sliders.json file
        workers: Number of parallel workers for the pipeline (uses config default)
        comparisons: List of comparison tuples (uses config default)

    Returns:
        dict: Result with success status, output_path, and stats means
              e.g. {"success": True, "output_path": "...", "psnr_mean": 37.25, ...}
    """
    import pandas as pd
    
    # Use config defaults if not provided
    if workers is None:
        workers = METRICS_WORKERS
    if comparisons is None:
        comparisons = DEFAULT_COMPARISONS

    result = {
        "success": False,
        "output_path": "",
        "psnr_mean": None,
        "ssim_mean": None,
        "hist_bhattacharyya_mean": None,
        "hist_bhattacharyya_mean": None,
        "mean_delta_e_mean": None,
        "metrics_images_count": None
    }

    try:
        logger.info(f"Calculating metrics for profile: {pid}")
        
        # Load the JSON edits
        with open(json_path, "r") as f:
            json_dict = json.load(f)
        
        # Setup paths
        profile_dir = Path(profile_path)
        imgs_dir = profile_dir / "metrics_new" / "DNGs"
        cache_dir = profile_dir / "m_cache_ALL"
        out_dir = profile_dir / "output_edited_ALL"
        
        result["output_path"] = str(out_dir)
        
        # Check if DNGs directory exists
        if not imgs_dir.exists():
            logger.error(f"DNGs directory not found: {imgs_dir}")
            return result
        
        # Clean XMP files before processing
        delete_xmp_files(str(imgs_dir), dry_run=False)
        
        # Remove existing cache if present
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info(f"Removed existing cache directory: {cache_dir}")
        
        # Create pipeline
        pipeline = EditsMetricsPipeline(cache_dir=cache_dir)
        
        # Delete XMP files again to be safe
        delete_xmp_files(str(imgs_dir), dry_run=False)
        
        logger.info(f"Running metrics pipeline for: {pid}")
        logger.info(f"  Images dir: {imgs_dir}")
        logger.info(f"  Output dir: {out_dir}")
        logger.info(f"  Comparisons: {comparisons}")
        
        # Run the pipeline
        stats_files = pipeline.run(
            imgs_dir=imgs_dir,
            edits=json_dict,
            out_dir=out_dir,
            editor=Editor.AdobeDNGConverter,
            editor_path=EDITOR_PATH,
            workers=workers,
            keep_cache=True,
            comparisons=comparisons
        )
        
        logger.info(f"Metrics calculation completed for: {pid}")
        logger.info(f"Stats files: {stats_files}")
        
        # Read the stats CSV and extract mean values
        stats_csv_path = out_dir / "Base_Custom_stats.csv"
        if stats_csv_path.exists():
            stats_df = pd.read_csv(stats_csv_path, index_col=0)
            if "mean" in stats_df.index:
                mean_row = stats_df.loc["mean"]
                result["psnr_mean"] = mean_row.get("psnr", None)
                result["ssim_mean"] = mean_row.get("ssim", None)
                result["hist_bhattacharyya_mean"] = mean_row.get("hist_bhattacharyya", None)
                result["mean_delta_e_mean"] = mean_row.get("mean_delta_e", None)
                
                if "count" in stats_df.index:
                    count_row = stats_df.loc["count"]
                    # Take count from any column, psnr is safe
                    result["metrics_images_count"] = int(count_row.get("psnr", 0))
                
                logger.info(f"Stats means - PSNR: {result['psnr_mean']:.2f}, SSIM: {result['ssim_mean']:.4f}, Delta-E: {result['mean_delta_e_mean']:.2f}, Count: {result['metrics_images_count']}")
        else:
            logger.warning(f"Stats CSV not found: {stats_csv_path}")
        
        result["success"] = True
        return result
        
    except Exception as e:
        logger.error(f"Error calculating metrics for {pid}: {e}")
        return result