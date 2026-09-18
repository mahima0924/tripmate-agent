"""
Central logging configuration for TripMate.

Sets up a logger that writes structured log lines to BOTH:
  - the console (so tool calls / reasoning are visible live, e.g. during
    a Loom recording or interactive demo), and
  - a log file (logs/tripmate.log), so example trace logs can be
    captured and included as a deliverable.
"""

import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "tripmate.log")


def setup_logging(level=logging.INFO):
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger("tripmate")
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging() is called more than once.
    if root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # capture everything to file

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return root_logger