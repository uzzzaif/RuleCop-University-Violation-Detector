# RuleCop - University Violation Detection System

A YOLOv8-based computer vision system that detects uniform violations and loitering in university campuses using real-time video analysis.

## Features

- **Real-time Detection**: YOLOv8 person detection with GPU acceleration
- **Uniform Violations**: HSV color-based or ML model detection for uniform compliance
- **Loitering Detection**: Identifies people staying in restricted areas beyond threshold
- **Web Dashboard**: Flask-based web interface for viewing violations
- **Database Integration**: MySQL for storing violation records and snapshots
- **Configurable Settings**: Easy-to-use configuration file for tuning detection parameters

## Project Structure

```
university_project/
├── dashboard_app.py          # Flask web dashboard
├── final_detector_db.py      # Main YOLOv8 detector with DB integration
├── db_setup.py               # Database initialization script
├── config_final.py           # Configuration settings
├── calibrate_colors.py       # Color calibration utility
├── templates/
│   └── dashboard.html        # Web interface
├── static/
│   └── style.css             # Dashboard styling
├── yolov8n.pt               # Pre-trained YOLOv8 model
└── violation_snapshots/      # Output violation images
```

## Installation

### Prerequisites
- Python 3.8+
- MySQL Server
- Git

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd university_project
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
source .venv/bin/activate      # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure database**
Edit `config_final.py` with your MySQL credentials:
```python
DB_HOST = 'localhost'
DB_USER = 'your_mysql_user'
DB_PASSWORD = 'your_mysql_password'
DB_NAME = 'university_violations'
```

5. **Initialize database**
```bash
python db_setup.py
```

## Usage

### Run Detection Only (No Dashboard)
```bash
python final_detector_db.py
```

### Run Web Dashboard
```bash
python dashboard_app.py
```
Then open http://localhost:5000 in your browser.

### Run Both
Open two terminals:
```bash
# Terminal 1
python final_detector_db.py

# Terminal 2
python dashboard_app.py
```

### Calibrate Uniform Colors
```bash
python calibrate_colors.py
```
This helps find the correct HSV color range for your uniform.

## Configuration

Edit `config_final.py` to customize:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `YOLO_CONFIDENCE` | float | 0.3 | Person detection confidence (0.0-1.0) |
| `UNIFORM_HSV_LOWER` | tuple | (90, 50, 80) | Lower HSV bounds for uniform color |
| `UNIFORM_HSV_UPPER` | tuple | (130, 255, 255) | Upper HSV bounds for uniform color |
| `UNIFORM_RATIO_THRESHOLD` | float | 0.25 | Percentage of torso that must match uniform color |
| `MIN_CONFIDENCE` | float | 0.6 | Violation confidence threshold |
| `LOITERING_TIME` | int | 30 | Seconds before flagging loitering |
| `FRAME_SKIP` | int | 2 | Process every Nth frame for performance |
| `SAVE_SNAPSHOTS` | bool | True | Save violation snapshots to disk |

## Database Schema

The system uses MySQL with the following main tables:
- `violations` - Recorded violation events
- `sessions` - Detection session logs
- `snapshots` - Associated violation images

Run `db_setup.py` to create all required tables.

## API Endpoints

### Dashboard
- `GET /` - Main dashboard page
- `GET /api/stats` - Violation statistics
- `GET /api/violations` - List all violations
- `GET /api/snapshots/<id>` - Retrieve violation snapshot

## Performance Tips

1. **Adjust FRAME_SKIP**: Increase to process fewer frames (faster but less accurate)
2. **Adjust YOLO_CONFIDENCE**: Lower detects more people (slower), higher detects fewer (faster)
3. **Reduce DISPLAY_WIDTH**: Smaller display improves rendering speed
4. **Use GPU**: Ensure CUDA is installed for GPU acceleration with YOLOv8

## Troubleshooting

### Database Connection Error
```
Solution: Ensure MySQL is running and credentials in config_final.py are correct
```

### "Address already in use" on port 5000
```
Solution: Change port in dashboard_app.py or close other Flask applications
```

### Model Download Failed
```
Solution: YOLOv8 models auto-download on first run. Ensure internet connection is available.
```

## License

[Add your license here]

## Contact

[Add contact information here]

## Contributors

[Add contributors here]
