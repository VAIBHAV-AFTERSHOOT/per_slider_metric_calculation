# Per-Model Delta-E Metrics Pipeline

A fully self-contained, portable pipeline to compute isolated Delta-E, MAE/R², PSNR, and SSIM metrics grouped by **ML model output heads**.

## Architecture Overview

### Model Groups (defined in `config.py → MODEL_GROUPS`)

Instead of evaluating sliders individually, the pipeline bundles sliders by the ML model that predicts them. Each group is rendered as a single unit and compared against the ground-truth `Base` edits.

| Group Name              | Sliders                                                      |
|-------------------------|--------------------------------------------------------------|
| `Exposure`              | Exposure                                                     |
| `CHS`                   | Contrast, Highlights, Shadows                                |
| `WB`                    | Temperature, Tint                                            |
| `Whites_Blacks`         | Whites, Blacks                                               |
| `Presence_CTD`          | Clarity, Texture, Dehaze                                     |
| `Presence_SV`           | Saturation, Vibrance                                         |
| `Detail_Sharpness`      | Sharpness, SharpenRadius, SharpenEdgeMasking, SharpenDetail  |
| `Detail_NoiseReduction` | LuminanceNoiseReduction*, LuminanceSmoothing, ColorNoiseReduction* |
| `HSL_Hue`               | HueAdjustment{Aqua,Green,Blue,Red,Magenta,Purple,Yellow,Orange} |
| `HSL_Saturation`        | SaturationAdjustment{...}                                    |
| `HSL_Luminance`         | LuminanceAdjustment{...}                                     |
| `ToneCurve_Parametric`  | ParametricHighlightSplit, ParametricMidtoneSplit, ...         |
| `ToneCurve_Gen`         | ToneCurve2012                                          |
| `ToneCurve_RGB`         | ToneCurve2012Red, ToneCurve2012Green, ToneCurve2012Blue                                          |
| `GrayMixer_Bundle`      | GrayMixer{Aqua,Green,Blue,Red,Magenta,Purple,Yellow,Orange}  |

### Partial Group Support

A model group is included if **at least one** of its defined sliders exists in `predictions.csv`. Any missing sliders are:
- Reported in console logs during discovery
- Listed in the `Missing Sliders` column of both the text summary and CSV outputs

**Example log output:**
```
  Group 'Detail_Sharpness': 3/4 sliders present. Missing: ['SharpenDetail']
```

### Bulk Pipeline Optimization

All model groups are evaluated in a **single `EditsMetricsPipeline.run()` call**:

1. A single `combined_metrics.json` is built with one `"Base"` dictionary and one `"Custom_{GroupName}"` dictionary per active group.
2. The pipeline is invoked once with `comparisons=[("Base", "Custom_Exposure"), ("Base", "Custom_CHS"), ...]`.
3. This means the Adobe DNG Converter renders the `Base` images **exactly once**, saving ~50% of compute time vs. iterating individually.

### Identical Edit Skip

If a `Custom_{Group}` dictionary is structurally identical to `Base` across all images (meaning the model predicted exactly the ground truth), the group is automatically skipped — no DNG conversion or metrics calculation is performed. The summary shows `Delta-E=0.00` immediately.

---

## Pipeline Flow

```
1. Load PIDs from CSV → Lookup user_id from profiles CSV
2. For each PID:
   a. Download metrics dir from GCS (predictions.csv + DNGs)
   b. Discover all Base_*/Custom_* slider pairs
   c. [Optional] Run OVERALL Delta-E (full Base vs Custom)
   d. Compute MAE/R² for all raw sliders
   e. Filter to active MODEL_GROUPS (partial match, R² filter)
   f. Build single combined_metrics.json (1 Base + N Custom_{Group} keys)
   g. Skip identical groups (Custom == Base)
   h. Run EditsMetricsPipeline.run() ONCE for all comparisons
   i. Parse stats CSVs → extract Delta-E, P75, PSNR, SSIM
   j. Write incremental summary (text + CSVs)
3. Aggregate results across all PIDs
```

---

## Output Structure

```
<output_dir>/
├── central_pipeline.log              # Global pipeline log
├── summary_<timestamp>.txt           # Running text summary table
├── overall_summary_<timestamp>.csv   # Incremental CSV (row per group per PID)
├── aggregated_results_<timestamp>.csv # Final complete CSV
└── <pid>/
    ├── profile_run.log               # Per-PID log
    ├── pid_summary.csv               # Per-PID CSV summary
    ├── pid_summary.txt               # Per-PID text summary
    ├── mae_r2_summary.csv            # Raw slider MAE/R²
    ├── slider_jsons/
    │   └── combined_metrics.json     # Multi-group edits JSON
    └── delta_e_comparisons/
        ├── Base_Custom_Exposure.csv
        ├── Base_Custom_Exposure_stats.csv
        ├── Base_Custom_CHS.csv
        ├── Base_Custom_CHS_stats.csv
        └── ...
```

---

## Requirements

Ensure the following local dependencies are present in the same directory:
- `get_metrics.py` (provides `calculate_metrics_for_profile`, `delete_xmp_files`)
- `A_B_utils.py`  
- `config.py` (contains `MODEL_GROUPS`, `EDITOR_PATH`, `METRICS_WORKERS`)
- `A_B_logger.py`

Run from the `editing-ml` virtual environment (`editlib` is required).

The directory is **fully portable** — copy it to any environment with `editlib` installed and run directly.

---

## Usage


python main.py \
    --pids_csv "/home/ubuntu/workspace/vaibhav-tmp/custom_slider_metrics/OG_WB_all_pids.csv" \
    --profiles_csv /home/ubuntu/workspace/vaibhav-tmp/PROFILES_CSV.csv \
    --output_dir /home/ubuntu/workspace/vaibhav-tmp/prod_inference_pipeline/per_slider_delta_etest_1 \
    --overall_deltaE

    
```bash
# Evaluate ALL model groups (auto-discovered from predictions.csv):
python main.py \
    --pids_csv "/path/to/pids.csv" \
    --profiles_csv /path/to/PROFILES_CSV.csv \
    --output_dir /path/to/output

# Evaluate specific model groups only:
python main.py \
    --pids_csv pids.csv \
    --output_dir output \
    --sliders WB Exposure CHS

# Run with overall Delta-E verification first:
python main.py \
    --pids_csv pids.csv \
    --output_dir output \
    --overall_deltaE

# Skip groups with negative mean R²:
python main.py \
    --pids_csv pids.csv \
    --output_dir output \
    --ignore_negative_r2

# Clean up DNGs after each PID to save disk space:
python main.py \
    --pids_csv pids.csv \
    --output_dir output \
    --no_cache
```

## CLI Arguments

| Argument              | Required | Default    | Description                                              |
|-----------------------|----------|------------|----------------------------------------------------------|
| `--pids_csv`          | Yes      | —          | CSV with PID column (auto-detected: pid, PID, key, etc.) |
| `--profiles_csv`      | No       | PROFILES_CSV.csv | CSV with `key` and `user_id` columns             |
| `--output_dir`        | Yes      | —          | Root output directory                                    |
| `--sliders`           | No       | All groups | Space-separated list of MODEL_GROUP names                |
| `--ignore_negative_r2`| No       | False      | Skip groups with negative mean R²                        |
| `--overall_deltaE`    | No       | False      | Run full Base→Custom Delta-E first                       |
| `--no_cache`          | No       | False      | Delete DNGs/cache after each PID                         |
