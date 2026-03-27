import os
import sys
import subprocess
import shutil
import glob
import pandas as pd
import cv2
import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from A_B_logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

from config import SLACK_BOT_TOKEN, SLACK_CHANNEL_ID





def extract_and_verify_dngs(
    pid: str, base_output_dir: str = r"V:\git-fs\metrics-automation\output_AB_test"
) -> dict:
    """
    Extract DNG files from metrics cache and verify against predictions.csv.

    Args:
        pid: Profile ID (e.g., 'e671ef3f-5e16-4a41-96b3-d4e10e39259b')s
        base_output_dir: Base directory containing the output folders

    Returns:
        dict with verification results
    """
    try:
        logger.info(f"Starting DNG extraction for PID: {pid}")
        pid_dir = os.path.join(base_output_dir, pid)

        # Find the metrics folder using glob pattern: trained_models\**\metrics\metrics_cache\edited_images\Base
        source_pattern = os.path.join(
            pid_dir,
            "trained_models",
            "*",
            "metrics",
            "metrics_cache",
            "edited_images",
            "Base",
        )
        source_matches = glob.glob(source_pattern)

        if not source_matches:
            logger.error(
                f"No source directory found matching pattern: {source_pattern}"
            )
            raise FileNotFoundError(
                f"No source directory found matching pattern: {source_pattern}"
            )

        source_dir = source_matches[0]  # Take the first match
        logger.info(f"Source directory: {source_dir}")

        # Find predictions.csv: trained_models\**\metrics\predictions.csv
        predictions_pattern = os.path.join(
            pid_dir, "trained_models", "*", "metrics", "predictions.csv"
        )
        predictions_matches = glob.glob(predictions_pattern)

        if not predictions_matches:
            logger.error(
                f"No predictions.csv found matching pattern: {predictions_pattern}"
            )
            raise FileNotFoundError(
                f"No predictions.csv found matching pattern: {predictions_pattern}"
            )

        predictions_csv = predictions_matches[0]
        logger.info(f"Predictions CSV: {predictions_csv}")

        # Destination directory: {pid}\metrics\test_metrics_dngs
        dest_dir = os.path.join(pid_dir, "metrics_new", "test_metrics_dngs")
        os.makedirs(dest_dir, exist_ok=True)
        predictions_csv = shutil.copy2(predictions_csv, os.path.dirname(dest_dir))
        logger.info(f"Destination directory: {dest_dir}")

        # Get all DNG files from source
        dng_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".dng")]
        dng_names = {os.path.splitext(f)[0] for f in dng_files}

        logger.info(f"Found {len(dng_files)} DNG files in source directory")

        # Copy DNG files to destination
        logger.info(f"Copying DNG files...")
        for dng in dng_files:
            src_path = os.path.join(source_dir, dng)
            dst_path = os.path.join(dest_dir, dng)
            shutil.copy2(src_path, dst_path)
        logger.info(f"Copied {len(dng_files)} DNG files to destination")

        # Read predictions.csv and get image_path column
        logger.info(f"Reading predictions.csv...")
        df = pd.read_csv(predictions_csv)
        csv_image_names = set(df["img_path"].astype(str).tolist())

        logger.info(f"Found {len(csv_image_names)} image paths in predictions.csv")

        # Verification
        logger.info("=" * 50 + " VERIFICATION RESULTS " + "=" * 50)

        dngs_not_in_csv = dng_names - csv_image_names
        csv_not_in_dngs = csv_image_names - dng_names

        if dngs_not_in_csv:
            logger.warning(f"DNGs NOT in predictions.csv: {len(dngs_not_in_csv)}")
            for name in sorted(dngs_not_in_csv)[:10]:
                logger.debug(f"   - {name}")
            if len(dngs_not_in_csv) > 10:
                logger.debug(f"   ... and {len(dngs_not_in_csv) - 10} more")
        else:
            logger.info("All DNGs are present in predictions.csv")

        if csv_not_in_dngs:
            logger.warning(f"CSV entries NOT in DNG folder: {len(csv_not_in_dngs)}")
            for name in sorted(csv_not_in_dngs)[:10]:
                logger.debug(f"   - {name}")
            if len(csv_not_in_dngs) > 10:
                logger.debug(f"   ... and {len(csv_not_in_dngs) - 10} more")
        else:
            logger.info("All CSV entries have corresponding DNG files")

        verification_passed = not dngs_not_in_csv and not csv_not_in_dngs

        if verification_passed:
            logger.info(
                "VERIFICATION PASSED: All DNGs match the predictions.csv entries!"
            )
        else:
            logger.warning(
                f"VERIFICATION FAILED: DNGs without CSV entry: {len(dngs_not_in_csv)}, CSV entries without DNG: {len(csv_not_in_dngs)}"
            )

        return {
            "verification_passed": verification_passed,
            "dng_count": len(dng_files),
            "csv_count": len(csv_image_names),
            "dngs_not_in_csv": dngs_not_in_csv,
            "csv_not_in_dngs": csv_not_in_dngs,
            "dest_dir": dest_dir,
        }

    except Exception as e:
        logger.error(f"Failed to extract and verify DNGs for PID {pid}: {e}")
        raise


# # Example usage
# if __name__ == "__main__":
#     pid = "e671ef3f-5e16-4a41-96b3-d4e10e39259b"
#     result = extract_and_verify_dngs(pid)


def convert_to_tif(dng_path, output_folder_name="tifs_converted", validate=True):
    """
    Detects OS, locates the correct binary, and converts a single DNG to TIF.
    """
    try:
        # 1. OS Detection for Binary Path
        is_windows = sys.platform == "win32"
        bin_name = (
            "binaries_dng_validate_dng_validate.exe"
            if is_windows
            else "binaries_dng_validate_dng_validate.linux"
        )
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dng_sdk_path = os.path.join(base_dir, "binaries", bin_name)

        # 2. Setup Paths
        dng_file = Path(dng_path)
        dng_name = dng_file.name

        # Create output directory relative to the DNG location
        dest_dir = dng_file.parent.parent / output_folder_name
        os.makedirs(dest_dir, exist_ok=True)

        # Define destination TIF path
        dest_path = str(dest_dir / f"{dng_file.stem}.tif")

        # 3. Prepare Arguments for subprocess
        # Note: dng_validate uses -tif to specify output path and the last arg as input
        tif_args = [dng_sdk_path, "-proxy", "1024", "-tif", dest_path, str(dng_file)]

        p = subprocess.Popen(tif_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            logger.error(f"Failed to convert {dng_name}: {stderr.decode()}")
            return False
        else:
            # Keep tqdm progress bar print for conversion progress
            print(f"✅ Converted: {dng_name} -> {dest_path}")
            return dest_path, str(dng_file), dng_name

    except Exception as e:
        logger.error(f"Error during conversion of {dng_path}: {e}")
        return False


# convert_to_tif(args=[r"dng_to_tiffs/RpmnDfm7xBw3JQWtRzUSUaEJ3nwzWf.exe",r"get_currTemp_Tint_bins/AMA_8051.tif",r"S_DNGs/S1/S1/AMA_8051.dng", r"test_AMA_8051.dng"])

# ------------------------------- RUN THE PROCESS --------------------------------------

# process_lrcat_folder(path,FLAG="ONCL")


def find_decrypted_file_path(base_path, pid, model=True, file_name="heregoes.onnx"):
    """
    Robustly finds the path to the decrypted onnx model.
    target_path: The root directory for the specific PID (profile_id)
    """
    try:
        # Look for the goes.onnx file anywhere inside the trained_models structure
        if model:
            search_pattern = (
                # f"{base_path}/{pid}/trained_models/**/models_dec.as/{file_name}"
                f"{base_path}/{pid}/exposure_nonubp/{file_name}"
            )
        else:
            search_pattern = f"{base_path}/{pid}/trained_models/**/{file_name}"
        found_files = glob.glob(search_pattern, recursive=True)

        if not found_files:
            logger.error(f"Could not find decrypted model in {search_pattern}")
            return None

        # Sort by creation time to ensure we get the absolute latest if multiple exist
        found_files.sort(key=os.path.getmtime, reverse=True)
        logger.debug(f"Found decrypted file: {found_files[0]}")
        return found_files[0]

    except Exception as e:
        logger.error(f"Error finding decrypted file: {e}")
        return None


def load_exposure_bounds(model_dir: Path) -> Optional[Tuple[float, float]]:
    """
    Load exposure bounds from info.json in the same directory as the model
    Returns: (min_val, max_val) or None if not found
    """
    info_json_path = model_dir / "info.json"

    if not info_json_path.exists():
        logger.warning(
            f"info.json not found at {info_json_path}, skipping bounds check"
        )
        return None

    try:
        with open(info_json_path, "r") as f:
            data = json.load(f)

        if "exposure" in data and "info_range" in data["exposure"]:
            min_val, max_val = data["exposure"]["info_range"]
            logger.info(f"Loaded exposure bounds: [{min_val}, {max_val}]")
            return (min_val, max_val)
        else:
            logger.warning(f"'exposure' key not found in info.json")
            return None

    except Exception as e:
        logger.error(f"Error loading bounds from info.json: {e}")
        return None


def cap_exposure_to_bounds(
    value: float, bounds: Tuple[float, float]
) -> Tuple[float, bool]:
    """
    Cap exposure value to bounds and return (capped_value, exceeded_flag)
    """
    min_val, max_val = bounds
    exceeded = False
    original_value = value

    if value < min_val:
        exceeded = True
        value = min_val
    elif value > max_val:
        exceeded = True
        value = max_val

    return value, exceeded


def run_inference_and_save_csv(
    base_path,
    pid,
    existing_csv_name="metrics_exif.csv",
    TIFFs_path: str = None,
    inp_csv_path: str = None,
    flag=None,
):
    try:
        logger.info(f"Starting inference for PID: {pid}")

        # 1. ROBUST PATH DISCOVERY
        model_path = find_decrypted_file_path(
            base_path=base_path, pid=pid, model=True, file_name="exposure_nonubp.onnx"
        )
        model_path = Path(model_path)

        if not model_path:
            logger.error("Model path not found, skipping inference.")
            return

        # Get model directory for loading bounds
        model_dir = Path(model_path).parent

        # Load exposure bounds from info.json
        exposure_bounds = load_exposure_bounds(model_dir)

        if not exposure_bounds:
            logger.warning("No exposure bounds loaded - predictions will not be capped")

        pid_dir = Path(base_path) / pid
        if inp_csv_path is None:
            csv_path = pid_dir / "metrics_new" / existing_csv_name
        else:
            csv_path = Path(inp_csv_path) / existing_csv_name

        if TIFFs_path is None:
            tifs_dir = pid_dir / "metrics_new" / "TIFFs"
        else:
            tifs_dir = Path(TIFFs_path)
        logger.info(f"TIFs directory: {tifs_dir}")

        if inp_csv_path is None:
            output_csv = pid_dir / f"combined_sliders_{flag}_final.csv"
        else:
            output_csv = Path(inp_csv_path) / f"combined_sliders_{flag}_final.csv"

        if not csv_path.exists():
            logger.error(f"CSV file not found at {csv_path}, skipping inference.")
            return
        else:
            if not tifs_dir.exists():
                logger.error(
                    f"TIFs directory not found at {tifs_dir}, skipping inference."
                )
                return

        # 3. Load Session
        session = ort.InferenceSession(str(model_path))
        df = pd.read_csv(str(csv_path))
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        inference_results = []

        # Track bounds exceeded stats
        total_exceeded = 0
        total_predictions = 0

        # Track missing TIFFs
        missing_tifs = []

        for _, row in df.iterrows():
            image_id = str(row["image_id"])
            ev_value = float(row["ev"])

            tif_file = tifs_dir / f"{image_id}.tif"
            if not tif_file.exists():
                missing_tifs.append(image_id)
                continue

            # Load and Resize to 256x256
            img = cv2.imread(str(tif_file))
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ✅ Convert to RGB

            # RESIZE to 256x256 as expected by your model
            img = cv2.resize(img, (256, 256))

            # Convert to float32
            img = img.astype(np.float32)

            # Add batch dimension: [1, 256, 256, 3]
            img = np.expand_dims(img, axis=0)

            # Prepare Inputs
            inputs = {
                session.get_inputs()[0].name: img,
                session.get_inputs()[1].name: np.array([[ev_value]], dtype=np.float32),
            }

            try:
                outputs = session.run(None, inputs)
                pred = outputs[0].flatten()

                result_entry = {"image_id": image_id}

                # Process exposure prediction (assuming single output)
                exposure_pred = float(pred[0])

                # Apply bounds if available
                if exposure_bounds:
                    capped_exposure, exceeded = cap_exposure_to_bounds(
                        exposure_pred, exposure_bounds
                    )
                    result_entry["predicted_exposure"] = round(capped_exposure, 2)
                    result_entry["bounds_exceeded"] = exceeded

                    if exceeded:
                        total_exceeded += 1
                else:
                    # No bounds available, use original value
                    result_entry["predicted_exposure"] = round(exposure_pred, 2)
                    result_entry["bounds_exceeded"] = False

                total_predictions += 1
                inference_results.append(result_entry)

            except Exception as e:
                logger.error(f"Inference error on {image_id}: {e}")
                continue

        # 5. Print inference statistics
        logger.info(
            f"Inference Statistics: Total rows in CSV: {len(df)}, Images inferenced: {len(inference_results)}, Missing TIFFs: {len(missing_tifs)}"
        )

        # If 0 images inferenced, print diagnostic info
        if len(inference_results) == 0:
            logger.warning(f"No images were inferenced!")
            if len(missing_tifs) > 0:
                logger.warning(
                    f"The following TIFFs from CSV are missing in {tifs_dir}:"
                )
                for name in missing_tifs[:20]:  # Show first 20
                    logger.debug(f"   - {name}.tif")
                if len(missing_tifs) > 20:
                    logger.debug(f"   ... and {len(missing_tifs) - 20} more")
            else:
                logger.warning(f"TIFFs directory may be empty or images failed to load")
            return

        # 6. Save Results
        if inference_results:
            res_df = pd.DataFrame(inference_results)
            final_df = pd.merge(df, res_df, on="image_id", how="left")

            # 6.1 Update Custom_Exposure in predictions.csv with new predictions
            predictions_csv_path = pid_dir / "metrics_new" / "predictions.csv"
            if predictions_csv_path.exists():
                logger.info(f"Updating Custom_Exposure in {predictions_csv_path}")
                predictions_df = pd.read_csv(str(predictions_csv_path))

                if "img_path" in predictions_df.columns:
                    # Create mapping from image_id to predicted_exposure from inference results
                    exposure_map = dict(
                        zip(res_df["image_id"], res_df["predicted_exposure"])
                    )

                    # Update Custom_Exposure using the mapping (img_path = image_id directly)
                    updated_count = 0
                    inferenced_indices = []
                    for idx, row in predictions_df.iterrows():
                        img_id = str(row["img_path"])
                        if img_id in exposure_map:
                            predictions_df.at[idx, "Custom_Exposure"] = exposure_map[
                                img_id
                            ]
                            updated_count += 1
                            inferenced_indices.append(idx)

                    # Save predictions_only.csv with only inferenced rows
                    predictions_only_path = (
                        predictions_csv_path.parent / "predictions_only.csv"
                    )
                    predictions_only_df = predictions_df.loc[inferenced_indices]
                    predictions_only_df.to_csv(str(predictions_only_path), index=False)
                    logger.info(
                        f"Saved {len(predictions_only_df)} inferenced rows to predictions_only.csv"
                    )

                    # Generate metrics_sliders.json from predictions_only.csv
                    metrics_sliders_path = (
                        predictions_csv_path.parent / "metrics_sliders.json"
                    )
                    csv_to_json_dynamic(
                        csv_path=str(predictions_only_path),
                        output_path=str(metrics_sliders_path),
                    )
                    logger.info(
                        f"Generated metrics_sliders.json at {metrics_sliders_path}"
                    )

                    # Save full predictions.csv with updated Custom_Exposure
                    predictions_df.to_csv(str(predictions_csv_path), index=False)
                    logger.info(
                        f"Updated {updated_count} Custom_Exposure values in predictions.csv"
                    )
                else:
                    logger.warning(f"predictions.csv missing 'img_path' column")
            else:
                logger.warning(f"predictions.csv not found at {predictions_csv_path}")

            final_df.to_csv(str(output_csv), index=False)

            # Print summary
            bounds_exceeded_count = (
                final_df["bounds_exceeded"].sum()
                if "bounds_exceeded" in final_df.columns
                else 0
            )
            logger.info(f"Successfully saved predictions to: {output_csv}")
            logger.info(
                f"Total images processed: {len(inference_results)}, Bounds exceeded: {bounds_exceeded_count}/{total_predictions}"
            )
            if total_predictions > 0:
                logger.info(
                    f"Percentage exceeded: {100*bounds_exceeded_count/total_predictions:.1f}%"
                )

    except Exception as e:
        logger.error(f"Failed to run inference for PID {pid}: {e}")
        raise


# SLACK INTEGRATION



# Initialize Slack Client


def send_slack_message(message, channel=None, slack_token=None):
    try:
        if slack_token is None:
            slack_token = SLACK_BOT_TOKEN
        if channel is None:
            channel = SLACK_CHANNEL_ID
            
        if not slack_token:
            logger.warning("SLACK_BOT_TOKEN not set, skipping Slack message.")
            return False

        client = WebClient(token=slack_token)

        response = client.chat_postMessage(
            channel=channel,
            text=message
        )
        return True
    except SlackApiError as e:
        logger.error(f"Error sending Slack message: {e.response['error']}")
        return False

# # Integration Example
# if __name__ == "__main__":
#     channel_id = "C0ADV494MCP"
    
#     # Use it during your cleanup process
#     send_slack_message(channel_id, "🚀 Model cleanup complete. Kept .json and .onnx files!")

# Example usage:
# pid = "2248b3eb-2f82-42dd-afde-bc9d861dc871"
# base_path = "output_UBP_test"
# print(f"--- COMPLETED DNG TO TIF CONVERSION FOR PROFILE: {pid} ---\n")
# print("Running ONNX inference and saving CSV...")

# run_inference_and_save_csv(
#     base_path=base_path,
#     pid=pid,
#     flag="NON_UBP"
# )

# ------------------------------- DYNAMIC CSV TO JSON --------------------------------------

def csv_to_json_dynamic(csv_path: str, output_path: str = None) -> dict:
    """
    Convert CSV to JSON using dynamic column discovery logic (based on EditsMetricsPipeline).
    Handles 'Custom_Custom' prefix hack and standard 'Base_'/'Custom_' prefixes.
    Does NOT use strict mapping or dot notation keys.
    """
    try:
        csv_path = Path(csv_path)
        if output_path is None:
            output_path = csv_path.parent / "metrics_sliders_dynamic.json"
            
        logger.info(f"Loading CSV for dynamic JSON conversion: {csv_path}")
        df = pd.read_csv(csv_path)
        
        edits = {}
        
        # Convert to records to match the provided script's logic
        records = df.to_dict(orient='records')
        
        for row in records:
            # Hack: Handle double prefixes if present (e.g. Custom_Custom_ -> Custom_)
            new_row = {}
            for key in row:
                key_str = str(key)
                if 'Custom_Custom' in key_str or 'Custom_Base' in key_str:
                    # Remove the first 'Custom_' prefix
                    new_key = key_str.replace('Custom_', '', 1)
                    new_row[new_key] = row[key]
                else:
                    new_row[key] = row[key]
            
            row = new_row
            img_path = row.get('img_path')
            
            if not img_path:
                continue
                
            img_path = str(img_path)

            for prefix in ['Base', 'Custom']:
                if prefix not in edits:
                    edits[prefix] = {}
                
                if img_path not in edits[prefix]:
                    edits[prefix][img_path] = {}
                
                # Iterate keys to find matching prefix
                for key in row:
                    key_str = str(key)
                    if key_str.startswith(f'{prefix}_'):
                        # Remove prefix to get slider name
                        slider_name = key_str.replace(f'{prefix}_', '', 1)
                        value = row[key]
                        
                        # Handle NaN
                        if pd.isna(value):
                            continue
                            
                        # Convert types
                        if isinstance(value, (int, float)):
                            pass # keep as is
                        else:
                            try:
                                if "." in str(value):
                                    value = float(value)
                                else:
                                    value = int(value)
                            except (ValueError, TypeError):
                                value = str(value)
                                
                        edits[prefix][img_path][slider_name] = value

        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(edits, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved dynamic metrics JSON with {len(edits.get('Base', {}))} images to {output_path}")
        return edits

    except Exception as e:
        logger.error(f"Error converting CSV to JSON (dynamic): {e}")
        return None
