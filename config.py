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
