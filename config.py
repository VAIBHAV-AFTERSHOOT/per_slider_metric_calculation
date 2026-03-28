import os
from dotenv import load_dotenv
from datetime import timezone, timedelta

# Load environment variables from .env file
load_dotenv()

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))
# Profile Configuration

# profiles_list is an alias for pid_dict for backward compatibility
profiles_list = pid_dict = {}

# =============================================================================
# Common Configuration Variables
# =============================================================================

# Base output directory for all A/B test results
BASE_PATH = "test_results"

# Metrics queue CSV file name
METRICS_QUEUE_CSV = "metrics_queue.csv"

# Profiles CSV file path
PROFILES_CSV_PATH = "profiles.csv"

# Editor path for DNG conversion
EDITOR_PATH = "/home/ubuntu/workspace/DNG_Converter/Adobe DNG Converter.exe"

# Number of workers for metrics pipeline
METRICS_WORKERS = 6

# Number of parallel workers for profile downloading/processing
MAX_DOWNLOAD_WORKERS = os.cpu_count()-2

# Default comparisons for metrics calculation
# Default comparisons for metrics calculation
DEFAULT_COMPARISONS = [("Base", "Custom")]

# Slack Configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
SLACK_CHANNEL_ID_2 = os.getenv("SLACK_CHANNEL_ID_2")
SEND_SLACK_NOTIFICATIONS = False # Set to False to disable Slack notifications

# =============================================================================
# Pipeline Semantic Model Groupings
# =============================================================================
# Used by the Per-Slider Delta-E pipeline to dynamically bundle model prediction groups together.
MODEL_GROUPS = {
    "Exposure": ["Exposure"],
    "CHS": ["Contrast", "Highlights", "Shadows"],
    "WB": ["Temperature", "Tint"],
    "Whites_Blacks": ["Whites", "Blacks"],
    "Presence_CTD": ["Clarity", "Texture", "Dehaze"],
    "Presence_SV": ["Saturation", "Vibrance"],
    "Detail_Sharpness": ["Sharpness", "SharpenRadius", "SharpenEdgeMasking", "SharpenDetail"],
    "Detail_NoiseReduction": [
        "LuminanceNoiseReductionContrast", "LuminanceNoiseReductionDetail", 
        "LuminanceSmoothing", "ColorNoiseReduction", "ColorNoiseReductionDetail", 
        "ColorNoiseReductionSmoothness"
    ],
    "HSL_Hue": [
        "HueAdjustmentAqua", "HueAdjustmentGreen", "HueAdjustmentBlue",
        "HueAdjustmentRed", "HueAdjustmentMagenta", "HueAdjustmentPurple",
        "HueAdjustmentYellow", "HueAdjustmentOrange"
    ],
    "HSL_Saturation": [
        "SaturationAdjustmentAqua", "SaturationAdjustmentGreen", "SaturationAdjustmentBlue",
        "SaturationAdjustmentRed", "SaturationAdjustmentMagenta", "SaturationAdjustmentPurple",
        "SaturationAdjustmentYellow", "SaturationAdjustmentOrange"
    ],
    "HSL_Luminance": [
        "LuminanceAdjustmentAqua", "LuminanceAdjustmentGreen", "LuminanceAdjustmentBlue",
        "LuminanceAdjustmentRed", "LuminanceAdjustmentMagenta", "LuminanceAdjustmentPurple",
        "LuminanceAdjustmentYellow", "LuminanceAdjustmentOrange"
    ],
    "ToneCurve_Parametric": [
        "ParametricHighlightSplit", "ParametricMidtoneSplit", "ParametricShadowSplit",
        "ParametricDarks", "ParametricHighlights", "ParametricLights", "ParametricShadows"
    ],
    "ToneCurve_Gen": ["ToneCurvePV2012_Gen"],
    "ToneCurve_RGB": ["ToneCurvePV2012_RGB"],
    "GrayMixer_Bundle": [
        "GrayMixerAqua", "GrayMixerGreen", "GrayMixerBlue", "GrayMixerRed",
        "GrayMixerMagenta", "GrayMixerPurple", "GrayMixerYellow", "GrayMixerOrange"
    ]
}
