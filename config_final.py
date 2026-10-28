from settings import (
    DB_CONFIG,
    DISPLAY_WIDTH,
    FRAME_SKIP,
    LOITERING_ENABLED,
    LOITERING_MOVEMENT_THRESHOLD,
    LOITERING_TIME,
    MIN_CONFIDENCE,
    MODEL_PATH,
    SAVE_SNAPSHOTS,
    SNAPSHOT_DIR,
    UNIFORM_RATIO_THRESHOLD,
    USE_ML_MODEL,
    YOLO_CONFIDENCE,
    YOLO_MODEL,
)


class Config:
    # ============= DATABASE =============
    DB_HOST = DB_CONFIG['host']
    DB_USER = DB_CONFIG['user']
    DB_PASSWORD = DB_CONFIG['password']
    DB_NAME = DB_CONFIG['database']
    SAVE_TO_DATABASE = True  # Set False to skip database
    
    # ============= DETECTION =============
    # YOLO person detection
    YOLO_CONFIDENCE = YOLO_CONFIDENCE
    YOLO_MODEL = YOLO_MODEL
    
    # Uniform detection method
    USE_ML_MODEL = USE_ML_MODEL
    MODEL_PATH = MODEL_PATH
    
    # ============= HSV COLOR DETECTION =============
    # Use calibrate_colors.py to find these values!
    # Example values for light blue uniform:
    UNIFORM_HSV_LOWER = (90, 50, 80)    # (Hue, Saturation, Value)
    UNIFORM_HSV_UPPER = (130, 255, 255)
    
    # How much of torso must be uniform color (0.0 to 1.0)
    UNIFORM_RATIO_THRESHOLD = UNIFORM_RATIO_THRESHOLD
    
    # Confidence threshold for reporting violations (0.0 to 1.0)
    MIN_CONFIDENCE = MIN_CONFIDENCE
    
    # ============= LOITERING =============
    LOITERING_ENABLED = LOITERING_ENABLED
    LOITERING_TIME = LOITERING_TIME
    LOITERING_MOVEMENT_THRESHOLD = LOITERING_MOVEMENT_THRESHOLD
    
    # ============= VIDEO/PERFORMANCE =============
    FRAME_SKIP = FRAME_SKIP
    DISPLAY_WIDTH = DISPLAY_WIDTH
    SAVE_SNAPSHOTS = SAVE_SNAPSHOTS
    SNAPSHOT_DIR = SNAPSHOT_DIR