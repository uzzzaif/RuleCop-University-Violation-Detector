"""
Database Setup Script
Run this ONCE before using the dashboard: python db_setup.py
"""

import mysql.connector
from mysql.connector import Error

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root@123',  # Change this to your MySQL password
    'database': 'university_violations'
}

def create_database():
    """Create database and tables"""
    
    try:
        # Connect without database to create it
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        
        cursor = connection.cursor()
        
        print("=" * 70)
        print("SETTING UP DATABASE")
        print("=" * 70)
        
        # Create database
        print("\n1. Creating database...")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        print(f"   ✅ Database '{DB_CONFIG['database']}' created")
        
        # Use database
        cursor.execute(f"USE {DB_CONFIG['database']}")
        
        # Create violations table
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
        print("   ✅ Violations table created")
        
        # Create sessions table
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
        print("   ✅ Sessions table created")
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print("\n" + "=" * 70)
        print("✅ DATABASE SETUP COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Run detection: python final_detector_db.py")
        print("  2. View dashboard: python dashboard_app.py")
        print("  3. Open browser: http://localhost:5000")
        print("=" * 70)
        
        return True
        
    except Error as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. MySQL is running (XAMPP/MySQL Workbench)")
        print("  2. Username and password are correct in db_setup.py")
        print("  3. You have permission to create databases")
        return False

if __name__ == '__main__':
    create_database()
