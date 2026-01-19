class Config:
    # ============= DATABASE =============
    DB_HOST = 'localhost'
    DB_USER = 'root'
    DB_PASSWORD = 'root@123'
    DB_NAME = 'university_violations'
    SAVE_TO_DATABASE = True  # Set False to skip database
    
    # ============= DETECTION =============
    # YOLO person detection
    YOLO_CONFIDENCE = 0.3  # Lower = detect more people (0.2-0.5)
    YOLO_MODEL = 'yolov8n.pt'
    
    # Uniform detection method
    USE_ML_MODEL = False  # True = use trained model, False = use HSV color
    MODEL_PATH = 'models/uniform_model.pkl'
    
    # ============= HSV COLOR DETECTION =============
    # Use calibrate_colors.py to find these values!
    # Example values for light blue uniform:
    UNIFORM_HSV_LOWER = (90, 50, 80)    # (Hue, Saturation, Value)
    UNIFORM_HSV_UPPER = (130, 255, 255)
    
    # How much of torso must be uniform color (0.0 to 1.0)
    UNIFORM_RATIO_THRESHOLD = 0.25  # 25% of torso must be uniform color
    
    # Confidence threshold for reporting violations (0.0 to 1.0)
    MIN_CONFIDENCE = 0.6  # Only report violations above this confidence
    
    # ============= LOITERING =============
    LOITERING_ENABLED = True
    LOITERING_TIME = 30  # seconds
    LOITERING_MOVEMENT_THRESHOLD = 25  # pixels
    
    # ============= VIDEO/PERFORMANCE =============
    FRAME_SKIP = 2  # Process every Nth frame (1=all frames, 2=every 2nd)
    DISPLAY_WIDTH = 1280  # Display window width
    SAVE_SNAPSHOTS = True
    SNAPSHOT_DIR = 'violation_snapshots'

