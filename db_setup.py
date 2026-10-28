"""Initialize the RuleCop MySQL database and tables."""

import mysql.connector
from mysql.connector import Error

from settings import DB_CONFIG


def create_database():
    """Create the configured database and required tables."""
    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            port=DB_CONFIG["port"],
        )
        cursor = connection.cursor()

        print("=" * 70)
        print("SETTING UP DATABASE")
        print("=" * 70)

        print(f"\n1. Selecting database '{DB_CONFIG['database']}'...")
        cursor.execute(f"USE `{DB_CONFIG['database']}`")
        print("   Database selected")

        print("\n2. Creating violations table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                violation_type ENUM('uniform', 'loitering') NOT NULL,
                timestamp DATETIME NOT NULL,
                confidence FLOAT,
                image_path VARCHAR(500),
                person_id VARCHAR(100),
                dwell_time INT,
                status ENUM('pending', 'reviewed', 'resolved') DEFAULT 'pending',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_timestamp (timestamp),
                INDEX idx_type (violation_type),
                INDEX idx_status (status)
            )
        """)
        print("   Violations table created")

        print("\n3. Creating detection sessions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                total_violations INT DEFAULT 0,
                total_frames INT DEFAULT 0,
                status ENUM('running', 'stopped') DEFAULT 'running',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("   Sessions table created")

        connection.commit()
        cursor.close()
        connection.close()

        print("\n" + "=" * 70)
        print("DATABASE SETUP COMPLETE")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run detection: python final_detector_db.py")
        print("  2. View dashboard: python dashboard_app.py")
        print("  3. Open browser: http://localhost:5000")
        return True

    except Error as error:
        print(f"\nDatabase setup error: {error}")
        print("Check that MySQL is running and .env credentials are correct.")
        return False


if __name__ == "__main__":
    create_database()
