"""Shared application settings loaded from the project .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def _get_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": _get_int("DB_PORT", 3306),
    "user": os.getenv("DB_USER", "rulecop"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "university_violations"),
}

FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = _get_int("FLASK_PORT", 5000)
FLASK_DEBUG = _get_bool("FLASK_DEBUG", False)
CAMERA_INDEX = _get_int("CAMERA_INDEX", 0)

YOLO_CONFIDENCE = _get_float("YOLO_CONFIDENCE", 0.3)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
USE_ML_MODEL = _get_bool("USE_ML_MODEL", False)
MODEL_PATH = os.getenv("MODEL_PATH", "models/uniform_model.pkl")
UNIFORM_RATIO_THRESHOLD = _get_float("UNIFORM_RATIO_THRESHOLD", 0.25)
MIN_CONFIDENCE = _get_float("MIN_CONFIDENCE", 0.6)
LOITERING_ENABLED = _get_bool("LOITERING_ENABLED", True)
LOITERING_TIME = _get_int("LOITERING_TIME", 30)
LOITERING_MOVEMENT_THRESHOLD = _get_int("LOITERING_MOVEMENT_THRESHOLD", 25)
FRAME_SKIP = _get_int("FRAME_SKIP", 5)
DISPLAY_WIDTH = _get_int("DISPLAY_WIDTH", 1280)
SAVE_SNAPSHOTS = _get_bool("SAVE_SNAPSHOTS", True)
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "violation_snapshots")
