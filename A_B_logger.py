"""
Centralized Logger Configuration for A_B Pipeline

This module provides a unified logging system for all A_B* files.
Log format: [DATE TIME IST] | [LEVEL] | [FILENAME:FUNCTION] | message
All timestamps are in IST (UTC+5:30).
"""

import logging
import os
from datetime import datetime, timezone, timedelta

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Default log file location (can be overridden per session)
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "A_B_pipeline.log")


class ISTFormatter(logging.Formatter):
    """Custom formatter that uses IST timezone and includes filename + function name."""

    def formatTime(self, record, datefmt=None):
        """Override to use IST instead of server local time."""
        dt = datetime.fromtimestamp(record.created, tz=IST)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        # Get just the filename without path and extension
        record.filename_short = os.path.basename(record.pathname).replace(".py", "")
        return super().format(record)


def set_log_file(path: str):
    """
    Change the global LOG_FILE path. Call this BEFORE any get_logger() calls
    to redirect the main pipeline log to a session-specific file.
    """
    global LOG_FILE
    LOG_FILE = path


def reconfigure_logger(name: str) -> logging.Logger:
    """
    Re-configure an existing logger to use the current LOG_FILE.
    Call this after set_log_file() for loggers that were already created.
    Returns the reconfigured logger.
    """
    logger = logging.getLogger(name)
    
    # Remove all existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # Re-add with current LOG_FILE
    log_format = "%(asctime)s | %(levelname)-5s | %(filename_short)s:%(funcName)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = ISTFormatter(log_format, datefmt=date_format)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the specified name.
    All timestamps are in IST.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create IST formatter
        log_format = "%(asctime)s | %(levelname)-5s | %(filename_short)s:%(funcName)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = ISTFormatter(log_format, datefmt=date_format)

        # File handler - logs everything
        file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console handler - logs INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


def get_profile_logger(pid: str, profile_dir: str) -> logging.Logger:
    """
    Get or create a logger for a specific profile that logs to the profile folder.
    Also logs to the main pipeline log file. All timestamps are in IST.

    Args:
        pid: Profile ID
        profile_dir: Path to the profile directory

    Returns:
        Configured logger instance for the profile
    """
    logger_name = f"profile_{pid}"
    logger = logging.getLogger(logger_name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create IST formatter
        log_format = "%(asctime)s | %(levelname)-5s | %(filename_short)s:%(funcName)s | %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = ISTFormatter(log_format, datefmt=date_format)

        # Ensure the profile directory exists
        os.makedirs(profile_dir, exist_ok=True)

        # Profile-specific file handler
        profile_log_file = os.path.join(profile_dir, "pipeline.log")
        file_handler = logging.FileHandler(profile_log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Also add to main log file
        main_file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        main_file_handler.setLevel(logging.DEBUG)
        main_file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(main_file_handler)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

    return logger


# Convenience function to clear log file
def clear_log():
    """Clear the log file."""
    if os.path.exists(LOG_FILE):
        open(LOG_FILE, "w").close()


# Log separator for new runs
def log_new_run(logger: logging.Logger):
    """Log a separator for a new pipeline run with IST timestamp."""
    logger.info("=" * 60)
    logger.info(f"NEW PIPELINE RUN - {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")
    logger.info("=" * 60)
