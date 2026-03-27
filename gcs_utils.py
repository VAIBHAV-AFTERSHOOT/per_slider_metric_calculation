"""
GCS utility helpers for gsutil operations.
"""
import subprocess
from pathlib import Path
import logging

log = logging.getLogger("per_slider_delta_e")


def gsutil_ls(gcs_path: str) -> list[str]:
    """List objects at a GCS path."""
    result = subprocess.run(
        ["gsutil", "ls", gcs_path],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return []
    return [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]


def gsutil_cp(gcs_src: str, local_dst: str) -> bool:
    """Copy a single object from GCS to local. Returns True on success."""
    result = subprocess.run(
        ["gsutil", "cp", gcs_src, local_dst],
        capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0


def list_timestamps(gcs_trained_models_base: str) -> list[str]:
    """Return timestamp folder names sorted descending (newest first)."""
    entries = gsutil_ls(gcs_trained_models_base + "/")
    names = [Path(e.rstrip("/")).name for e in entries]
    return sorted([n for n in names if n.isdigit()], key=int, reverse=True)
