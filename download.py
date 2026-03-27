"""
Download metrics directory (predictions.csv, metrics_exif.csv, DNG zips)
from GCS for a given profile.
"""
import os
import zipfile
import logging
from pathlib import Path
from typing import Optional

from gcs_utils import gsutil_ls, gsutil_cp

log = logging.getLogger("per_slider_delta_e")


def download_metrics_dir(
    user_id: str,
    pid: str,
    output_dir: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Download predictions.csv, metrics_exif.csv, and DNG zips from
    gs://editing_userdata/{user_id}/{pid}/training_data/metrics/.

    Extracts DNGs from zips into <output_dir>/<pid>/metrics_new/DNGs/.
    Skips profile if any required file is missing.

    Returns:
        (predictions_csv_path, metrics_exif_csv_path, dngs_dir) or (None, None, None)
    """
    metrics_gcs_base = f"gs://editing_userdata/{user_id}/{pid}/training_data/metrics"
    log.info(f"[{pid[:8]}] Fetching from {metrics_gcs_base}")

    pid_dir = os.path.join(output_dir, pid)
    metrics_local = os.path.join(pid_dir, "metrics_new")
    os.makedirs(metrics_local, exist_ok=True)
    dngs_dir = os.path.join(metrics_local, "DNGs")
    os.makedirs(dngs_dir, exist_ok=True)

    # List remote files
    all_files = gsutil_ls(metrics_gcs_base + "/")
    if not all_files:
        log.error(f"[{pid[:8]}] No files found in {metrics_gcs_base}")
        return None, None, None

    # --- predictions.csv (required) ---
    pred_gcs = f"{metrics_gcs_base}/predictions.csv"
    if pred_gcs not in all_files:
        log.error(f"[{pid[:8]}] Missing predictions.csv")
        return None, None, None

    predictions_csv = os.path.join(metrics_local, "predictions.csv")
    log.info(f"[{pid[:8]}] Downloading predictions.csv")
    if not gsutil_cp(pred_gcs, predictions_csv):
        log.error(f"[{pid[:8]}] Failed to download predictions.csv")
        return None, None, None

    # --- metrics_exif.csv (optional but logged) ---
    exif_gcs = f"{metrics_gcs_base}/metrics_exif.csv"
    exif_csv = None
    if exif_gcs in all_files:
        exif_csv = os.path.join(metrics_local, "metrics_exif.csv")
        log.info(f"[{pid[:8]}] Downloading metrics_exif.csv")
        if not gsutil_cp(exif_gcs, exif_csv):
            log.warning(f"[{pid[:8]}] Failed to download metrics_exif.csv")
            exif_csv = None
    else:
        log.info(f"[{pid[:8]}] metrics_exif.csv not found (not required for this pipeline)")

    # --- DNG zips (required) ---
    zip_files = [f for f in all_files if f.lower().endswith(".zip")]
    if not zip_files:
        log.error(f"[{pid[:8]}] No DNG .zip files found")
        return None, None, None

    for gcs_zip in zip_files:
        zip_name = Path(gcs_zip).name
        local_zip = os.path.join(metrics_local, zip_name)

        log.info(f"[{pid[:8]}] Downloading {zip_name}")
        if not gsutil_cp(gcs_zip, local_zip):
            log.warning(f"[{pid[:8]}] Failed to download {zip_name}, skipping")
            continue

        try:
            with zipfile.ZipFile(local_zip, "r") as z:
                n_dng = 0
                for m in z.namelist():
                    if Path(m).suffix.upper() == ".DNG":
                        data = z.read(m)
                        out_path = os.path.join(dngs_dir, Path(m).name)
                        with open(out_path, "wb") as f:
                            f.write(data)
                        n_dng += 1
                if n_dng:
                    log.info(f"[{pid[:8]}] Extracted {n_dng} DNG(s) from {zip_name}")
        except zipfile.BadZipFile:
            log.warning(f"[{pid[:8]}] Bad zip: {zip_name}")
        finally:
            try:
                os.remove(local_zip)
            except Exception:
                pass

    dng_count = len([f for f in os.listdir(dngs_dir) if Path(f).suffix.upper() == ".DNG"])
    log.info(f"[{pid[:8]}] Total DNGs extracted: {dng_count}")

    if dng_count == 0:
        log.error(f"[{pid[:8]}] No DNGs were extracted.")
        return None, None, None

    return predictions_csv, exif_csv, dngs_dir
