# Per-Slider Delta-E Metrics Pipeline

A fully modular pipeline to compute isolated Delta-E and MAE/R² metrics for individual rendering sliders.

## Features
- **Slider Isolation**: Generates a metrics JSON where the `Custom` rendering exactly matches the `Base` ground-truth, *except* for the single slider being tested. This perfectly isolates the visual impact (Delta-E) of predicting that specific slider.
- **Case-Insensitive Sliders**: Automatically matches user CLI args with whatever casing is in `predictions.csv`.
- **White Balance Grouping**: Pass `--sliders WB` and the script automatically bundles `Temperature` and `Tint` together into a single isolated render.
- **Robust Multi-Layer Logging**:
  - Console standard out
  - Global central pipeline log (`<output_dir>/central_pipeline.log`)
  - File-specific isolated log (`<output_dir>/<pid>/profile_run.log`)
- **Incremental Results**: Writes rows to the summary CSV immediately after measuring each slider so you don't have to wait for the entire process to finish before seeing data.

## Requirements
Ensure you have copied the following dependencies into this same root directory. The pipeline will import them locally:
- `get_metrics.py` (which includes `calculate_metrics_for_profile` and `delete_xmp_files`)
- `A_B_utils.py`
- `config.py`
- `A_B_logger.py`

Run the script from inside the `editing-ml` virtual environment, as `EditsMetricsPipeline` relies on `editlib`.

## Usage
Run the module using `python -m`:

```bash
# Evaluate ALL sliders:
python main.py \
    --pids_csv "/home/ubuntu/workspace/vaibhav-tmp/custom_slider_metrics/OG_WB_all_pids.csv" \
    --profiles_csv /home/ubuntu/workspace/vaibhav-tmp/PROFILES_CSV.csv \
    --output_dir /home/ubuntu/workspace/vaibhav-tmp/prod_inference_pipeline/per_slider_delta_etest_1 \
    --overall_deltaE


# Evaluate only specific sliders:
python main.py \
    --pids_csv pids.csv \
    --output_dir output \
    --sliders WB Exposure Contrast

# Automatically skip any sliders with negative R²:
python main.py \
    --pids_csv pids.csv \
    --output_dir output \
    --ignore_negative_r2

# Clean up DNGs as you go to save massive storage space:
python main.py \
    --pids_csv pids.csv \
    --output_dir output \
    --no_cache
```

python main.py \
    --pids_csv pids.csv \
    --output_dir output \
    --overall_deltaE
