# RuleCop - University Violation Detection System

RuleCop uses YOLOv8 and OpenCV to detect uniform violations and loitering from a webcam. It stores violation records in MySQL and provides a Flask dashboard for reviewing them.

## Features

- Real-time YOLOv8 person detection
- HSV-based uniform compliance detection
- Loitering detection
- MySQL violation and session records
- Flask dashboard
- Configurable settings through `.env`
- Optional GPU acceleration with CUDA

## Requirements

- Windows 10/11
- Python 3.8 or newer
- MySQL Server 8.x
- A webcam
- Git (optional)

## Install on a new Windows laptop

### 1. Install MySQL Server

Download **MySQL Community Server** from:

<https://dev.mysql.com/downloads/mysql/>

During the MySQL Installer setup:

1. Choose **Developer Default** or **Server only**.
2. Keep the default TCP port `3306`.
3. Choose **Use Strong Password Encryption**.
4. Set and save the MySQL root password.
5. Install MySQL as a Windows service and enable it to start automatically.

MySQL Workbench is optional but useful for running SQL and inspecting tables:

<https://dev.mysql.com/downloads/workbench/>

To check the MySQL service in PowerShell:

```powershell
Get-Service MySQL*
```

If it is stopped, start it from an elevated PowerShell window:

```powershell
Start-Service MySQL80
```

The service may have a different name on your computer. Use the name shown by `Get-Service MySQL*`.

### 2. Get the project

```powershell
git clone <repository-url>
cd "RULE COP"
```

If you received the project as a folder, open PowerShell in that folder instead.

### 3. Create and activate a Python environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Create a dedicated MySQL application user

Using MySQL Workbench, open a SQL tab and run the following. Replace the password with a strong password of your choice:

```sql
CREATE DATABASE IF NOT EXISTS university_violations;

CREATE USER IF NOT EXISTS 'rulecop'@'localhost'
IDENTIFIED BY 'replace_with_a_strong_password';

GRANT ALL PRIVILEGES ON university_violations.* TO 'rulecop'@'localhost';
FLUSH PRIVILEGES;
```

You can run the same SQL from MySQL Shell after connecting as root:

```powershell
mysql -u root -p
```

Do not use the root account in the application. The dedicated account limits database access to RuleCop's database.

### 5. Configure `.env`

Copy the safe template:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `DB_PASSWORD` to the password used when creating the `rulecop` user:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=rulecop
DB_PASSWORD=replace_with_a_strong_password
DB_NAME=university_violations
```

Never commit `.env`. It is ignored by `.gitignore`. Keep `.env.example` in the project as the portable configuration template.

### 6. Create the application tables

With MySQL running and the virtual environment activated:

```powershell
python db_setup.py
```

This selects the configured database and creates the `violations` and `detection_sessions` tables. The command can be run again safely.

### 7. Start the dashboard and detector

Open two PowerShell windows. Activate the environment in both.

Dashboard terminal:

```powershell
cd "E:\Projects\RULE COP"
.\.venv\Scripts\Activate.ps1
python dashboard_app.py
```

Open <http://localhost:5000>.

Detector terminal:

```powershell
cd "E:\Projects\RULE COP"
.\.venv\Scripts\Activate.ps1
python final_detector_db.py
```

Press `Q` in the detector window to stop detection.

## Optional: calibrate uniform colors

Run:

```powershell
python calibrate_colors.py
```

Position a person wearing the correct uniform in front of the webcam. Adjust the sliders until the uniform is highlighted, press `Q`, and copy the displayed HSV values into `config_final.py`.

## Configuration

The main settings are in `.env`:

| Variable | Default | Description |
| --- | --- | --- |
| `DB_HOST` | `localhost` | MySQL host |
| `DB_PORT` | `3306` | MySQL port |
| `DB_USER` | `rulecop` | MySQL application user |
| `DB_PASSWORD` | — | MySQL application password |
| `DB_NAME` | `university_violations` | Database name |
| `CAMERA_INDEX` | `0` | Webcam index; try `1` for another camera |
| `FLASK_HOST` | `127.0.0.1` | Dashboard bind address |
| `FLASK_PORT` | `5000` | Dashboard port |
| `FLASK_DEBUG` | `false` | Flask debug mode |
| `YOLO_CONFIDENCE` | `0.3` | Person detection confidence |
| `LOITERING_TIME` | `30` | Seconds before loitering is reported |
| `FRAME_SKIP` | `5` | Process every Nth frame |
| `SAVE_SNAPSHOTS` | `true` | Save violation images |

Detection HSV values remain in `config_final.py` because they are normally tuned interactively. The database, dashboard, camera, model, and performance settings are loaded centrally from `.env`.

## Project structure

```text
RULE COP/
├── dashboard_app.py       # Flask web dashboard
├── final_detector_db.py   # YOLOv8 detector and database integration
├── db_setup.py            # Database and table initialization
├── settings.py            # Shared .env loader and typed settings
├── config_final.py        # Detector configuration
├── calibrate_colors.py    # HSV calibration utility
├── .env.example           # Safe configuration template
├── requirements.txt
├── templates/
├── static/
└── yolov8n.pt             # YOLOv8 model
```

## API endpoints

- `GET /` - Dashboard
- `GET /api/stats` - Violation statistics
- `GET /api/violations` - Violation list
- `GET /api/timeline` - Violation timeline
- `POST /api/violations/<id>/update` - Update a violation
- `GET /image/<filename>` - Retrieve a snapshot

## Troubleshooting

### `Access denied for user 'rulecop'`

Verify `DB_USER` and `DB_PASSWORD` in `.env`. Then confirm the user exists and has privileges:

```sql
SHOW GRANTS FOR 'rulecop'@'localhost';
```

### Cannot connect to MySQL

Check that the MySQL service is running, that `DB_HOST` and `DB_PORT` are correct, and that Windows Firewall is not blocking the local service.

### `python-dotenv` or another module is missing

Activate the virtual environment and reinstall dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Webcam cannot be opened

Close other applications using the camera. Then set `CAMERA_INDEX=1` or another available index in `.env`.

### Port 5000 is already in use

Set a different value in `.env`:

```env
FLASK_PORT=5001
```

Then open <http://localhost:5001>.

### Model download fails

Ensure the laptop has internet access. The bundled `yolov8n.pt` file is used by default; set `YOLO_MODEL` in `.env` only if you want to use another model.

## License

No license has been specified yet.
