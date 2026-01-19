"""
Updated detector with database integration
This replaces your final_detector.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict
import os
import pickle
import mysql.connector
from mysql.connector import Error

# Import config
from config_final import Config

class Database:
    """Database manager for storing violations"""
    
    def __init__(self):
        self.config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'root@123',  # Change to your password
            'database': 'university_violations'
        }
        self.connection = None
        self.session_id = None
        self.connect()
    
    def connect(self):
        """Connect to database"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                print("✅ Connected to database")
                self.start_session()
                return True
        except Error as e:
            print(f"⚠️  Database connection failed: {e}")
            print("   Detection will continue without database logging")
            self.connection = None
            return False
    
    def start_session(self):
        """Start a new detection session"""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO detection_sessions (start_time, status)
                VALUES (%s, 'running')
            """, (datetime.now(),))
            self.connection.commit()
            self.session_id = cursor.lastrowid
            cursor.close()
            print(f"✅ Started detection session #{self.session_id}")
        except Error as e:
            print(f"⚠️  Could not start session: {e}")
    
    def save_violation(self, violation_type, confidence, image_path, 
                       person_id, dwell_time=None):
        """Save violation to database"""
        if not self.connection:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO violations 
                (violation_type, timestamp, confidence, image_path, 
                 person_id, dwell_time, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'pending')
            """, (
                violation_type,
                datetime.now(),
                confidence,
                image_path,
                person_id,
                dwell_time
            ))
            self.connection.commit()
            violation_id = cursor.lastrowid
            cursor.close()
            
            # Update session count
            self.update_session_count()
            
            return violation_id
        except Error as e:
            print(f"⚠️  Could not save violation: {e}")
            return None
    
    def update_session_count(self):
        """Update violation count in session"""
        if not self.connection or not self.session_id:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE detection_sessions 
                SET total_violations = (
                    SELECT COUNT(*) FROM violations
                )
                WHERE id = %s
            """, (self.session_id,))
            self.connection.commit()
            cursor.close()
        except Error as e:
            pass
    
    def end_session(self, total_frames):
        """End detection session"""
        if not self.connection or not self.session_id:
            return
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE detection_sessions 
                SET end_time = %s, total_frames = %s, status = 'stopped'
                WHERE id = %s
            """, (datetime.now(), total_frames, self.session_id))
            self.connection.commit()
            cursor.close()
            print(f"✅ Ended detection session #{self.session_id}")
        except Error as e:
            print(f"⚠️  Could not end session: {e}")
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("✅ Database connection closed")


class UniformDetector:
    """Uniform violation detector"""
    
    def __init__(self):
        self.use_ml = Config.USE_ML_MODEL
        self.model = None
        self.scaler = None
        
        if self.use_ml and os.path.exists(Config.MODEL_PATH):
            with open(Config.MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(Config.MODEL_PATH.replace('model', 'scaler'), 'rb') as f:
                self.scaler = pickle.load(f)
    
    def extract_torso(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        person = frame[y1:y2, x1:x2]
        if person.size == 0:
            return None
        
        ph = person.shape[0]
        pw = person.shape[1]
        torso = person[int(ph*0.15):int(ph*0.65), int(pw*0.1):int(pw*0.9)]
        
        return torso if torso.size > 0 else None
    
    def detect_hsv(self, torso):
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lower = np.array(Config.UNIFORM_HSV_LOWER)
        upper = np.array(Config.UNIFORM_HSV_UPPER)
        mask = cv2.inRange(hsv, lower, upper)
        ratio = cv2.countNonZero(mask) / (torso.shape[0] * torso.shape[1])
        is_violation = ratio < Config.UNIFORM_RATIO_THRESHOLD
        confidence = 1.0 - ratio if is_violation else ratio
        return is_violation, confidence
    
    def detect_ml(self, torso):
        torso_resized = cv2.resize(torso, (100, 100))
        hsv = cv2.cvtColor(torso_resized, cv2.COLOR_BGR2HSV)
        
        features = []
        for i in range(3):
            channel = hsv[:,:,i].flatten()
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
                np.max(channel) - np.min(channel)
            ])
        
        hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        
        features.extend(hist_h / (hist_h.sum() + 1e-7))
        features.extend(hist_s / (hist_s.sum() + 1e-7))
        features.extend(hist_v / (hist_v.sum() + 1e-7))
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        is_violation = (pred == 0)
        confidence = proba[0] if is_violation else proba[1]
        
        return is_violation, confidence
    
    def detect(self, frame, bbox):
        torso = self.extract_torso(frame, bbox)
        if torso is None:
            return False, 0.0
        
        if self.use_ml and self.model:
            return self.detect_ml(torso)
        else:
            return self.detect_hsv(torso)


class LoiteringDetector:
    """Loitering detector"""
    
    def __init__(self):
        self.tracks = defaultdict(lambda: {
            'first_seen': None,
            'positions': [],
            'last_update': None
        })
    
    def update(self, person_id, bbox):
        now = datetime.now()
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        track = self.tracks[person_id]
        
        if track['first_seen'] is None:
            track['first_seen'] = now
        
        track['last_update'] = now
        track['positions'].append(center)
        
        if len(track['positions']) > 30:
            track['positions'] = track['positions'][-30:]
        
        dwell_time = (now - track['first_seen']).total_seconds()
        
        if len(track['positions']) > 10:
            positions = np.array(track['positions'][-10:])
            movement = np.std(positions, axis=0).mean()
            
            is_loitering = (dwell_time > Config.LOITERING_TIME and 
                           movement < Config.LOITERING_MOVEMENT_THRESHOLD)
            
            return is_loitering, dwell_time
        
        return False, dwell_time
    
    def cleanup(self):
        now = datetime.now()
        to_remove = []
        for pid, track in self.tracks.items():
            if track['last_update']:
                if (now - track['last_update']).total_seconds() > 5:
                    to_remove.append(pid)
        
        for pid in to_remove:
            del self.tracks[pid]


class ViolationDetectionSystem:
    """Main detection system with database"""
    
    def __init__(self):
        print("=" * 70)
        print("INITIALIZING DETECTION SYSTEM WITH DATABASE")
        print("=" * 70)
        
        # Initialize database
        self.db = Database()
        
        # Load YOLO
        print("Loading YOLO model...")
        self.yolo = YOLO(Config.YOLO_MODEL)
        print("✅ YOLO loaded")
        
        # Initialize detectors
        self.uniform_detector = UniformDetector()
        self.loitering_detector = LoiteringDetector() if Config.LOITERING_ENABLED else None
        
        # Setup
        self.next_id = 0
        if Config.SAVE_SNAPSHOTS:
            os.makedirs(Config.SNAPSHOT_DIR, exist_ok=True)
        
        print("✅ System ready!")
        print("=" * 70)
    
    def save_snapshot(self, frame, bbox, violation_type, confidence):
        if not Config.SAVE_SNAPSHOTS:
            return None
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        x1 = max(0, x1 - 20)
        y1 = max(0, y1 - 20)
        x2 = min(w, x2 + 20)
        y2 = min(h, y2 + 20)
        
        snapshot = frame[y1:y2, x1:x2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{Config.SNAPSHOT_DIR}/{violation_type}_{confidence:.2f}_{timestamp}.jpg"
        
        cv2.imwrite(filename, snapshot)
        return filename
    
    def process_frame(self, frame):
        results = self.yolo(frame, conf=Config.YOLO_CONFIDENCE, verbose=False)
        
        violations = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) != 0:
                    continue
                
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                person_id = f"P{self.next_id}"
                self.next_id += 1
                
                # Check uniform
                is_uniform_violation, uniform_conf = self.uniform_detector.detect(frame, bbox)
                
                if is_uniform_violation and uniform_conf >= Config.MIN_CONFIDENCE:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"UNIFORM VIOLATION: {uniform_conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    snapshot_path = self.save_snapshot(frame, bbox, 'uniform', uniform_conf)
                    
                    # Save to database
                    violation_id = self.db.save_violation(
                        'uniform', uniform_conf, snapshot_path, person_id
                    )
                    
                    violations.append({
                        'type': 'uniform',
                        'confidence': uniform_conf,
                        'person_id': person_id,
                        'snapshot': snapshot_path,
                        'db_id': violation_id
                    })
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "OK", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check loitering
                if self.loitering_detector:
                    is_loitering, dwell_time = self.loitering_detector.update(person_id, bbox)
                    
                    if is_loitering:
                        cv2.putText(frame, f"LOITERING: {dwell_time:.0f}s", 
                                   (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (255, 0, 0), 2)
                        
                        # Save to database
                        violation_id = self.db.save_violation(
                            'loitering', 1.0, None, person_id, int(dwell_time)
                        )
                        
                        violations.append({
                            'type': 'loitering',
                            'dwell_time': dwell_time,
                            'person_id': person_id,
                            'db_id': violation_id
                        })
        
        if self.loitering_detector:
            self.loitering_detector.cleanup()
        
        return frame, violations
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        print("\n🎥 Starting detection with database logging...")
        print("Press 'Q' to quit\n")
        
        frame_count = 0
        violation_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                if frame_count % Config.FRAME_SKIP == 0:
                    h, w = frame.shape[:2]
                    if w > Config.DISPLAY_WIDTH:
                        scale = Config.DISPLAY_WIDTH / w
                        frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    
                    processed, violations = self.process_frame(frame)
                    
                    violation_count += len(violations)
                    
                    for v in violations:
                        if v['type'] == 'uniform':
                            print(f"⚠️  UNIFORM VIOLATION (ID: {v.get('db_id')}) Confidence: {v['confidence']:.2%}")
                            if v.get('snapshot'):
                                print(f"   Saved: {v['snapshot']}")
                        elif v['type'] == 'loitering':
                            print(f"⚠️  LOITERING (ID: {v.get('db_id')}) Time: {v['dwell_time']:.0f}s")
                    
                    stats = f"Frames: {frame_count} | Violations: {violation_count}"
                    cv2.putText(processed, stats, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Violation Detection - Press Q to Quit', processed)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.db.end_session(frame_count)
            self.db.close()
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n{'='*70}")
            print(f"Detection completed!")
            print(f"Total frames: {frame_count}")
            print(f"Total violations: {violation_count}")
            print(f"Violations saved to database ✅")
            print(f"{'='*70}")


if __name__ == '__main__':
    try:
        system = ViolationDetectionSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
