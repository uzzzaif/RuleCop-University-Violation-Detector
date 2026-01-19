"""
Flask Web Dashboard
Run this to view violations in browser: python dashboard_app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, send_file, request
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root@123',  # Change this
    'database': 'university_violations'
}

def get_db_connection():
    """Get database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get statistics for dashboard"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Total violations
        cursor.execute("SELECT COUNT(*) as total FROM violations")
        total = cursor.fetchone()['total']
        
        # Violations by type
        cursor.execute("""
            SELECT violation_type, COUNT(*) as count 
            FROM violations 
            GROUP BY violation_type
        """)
        by_type = cursor.fetchall()
        
        # Last 24 hours
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM violations 
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
        """)
        last_24h = cursor.fetchone()['count']
        
        # Pending violations
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM violations 
            WHERE status = 'pending'
        """)
        pending = cursor.fetchone()['count']
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'total': total,
            'by_type': by_type,
            'last_24h': last_24h,
            'pending': pending
        })
    
    except Error as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/violations')
def get_violations():
    """Get list of violations"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        # Get parameters
        limit = request.args.get('limit', 50, type=int)
        violation_type = request.args.get('type', None)
        status = request.args.get('status', None)
        
        # Build query
        query = "SELECT * FROM violations WHERE 1=1"
        params = []
        
        if violation_type:
            query += " AND violation_type = %s"
            params.append(violation_type)
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(query, params)
        violations = cursor.fetchall()
        
        # Convert datetime to string
        for v in violations:
            if v.get('timestamp'):
                v['timestamp'] = v['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if v.get('created_at'):
                v['created_at'] = v['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.close()
        conn.close()
        
        return jsonify(violations)
    
    except Error as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/timeline')
def get_timeline():
    """Get violations timeline for chart"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)
        
        days = request.args.get('days', 7, type=int)
        
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                violation_type,
                COUNT(*) as count
            FROM violations
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(timestamp), violation_type
            ORDER BY date ASC
        """, (days,))
        
        results = cursor.fetchall()
        
        # Convert date to string
        for r in results:
            if r.get('date'):
                r['date'] = r['date'].strftime('%Y-%m-%d')
        
        cursor.close()
        conn.close()
        
        return jsonify(results)
    
    except Error as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/violations/<int:violation_id>/update', methods=['POST'])
def update_violation(violation_id):
    """Update violation status"""
    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        data = request.get_json()
        status = data.get('status')
        notes = data.get('notes', '')
        
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE violations 
            SET status = %s, notes = %s 
            WHERE id = %s
        """, (status, notes, violation_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True})
    
    except Error as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image/<path:filename>')
def serve_image(filename):
    """Serve violation images"""
    try:
        return send_file(filename, mimetype='image/jpeg')
    except:
        return "Image not found", 404

if __name__ == '__main__':
    print("=" * 70)
    print("STARTING FLASK DASHBOARD")
    print("=" * 70)
    print("\nDashboard will be available at:")
    print("👉 http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)