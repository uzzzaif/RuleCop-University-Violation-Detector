"""
Quick color calibration tool - finds HSV values for your uniform
Run this FIRST to find your uniform color values
"""

import cv2
import numpy as np

def calibrate_uniform_color():
    """Interactive tool to find uniform HSV values"""
    
    print("=" * 70)
    print("UNIFORM COLOR CALIBRATION")
    print("=" * 70)
    print("\nInstructions:")
    print("1. Position someone in CORRECT uniform in front of camera")
    print("2. Adjust sliders until ONLY the uniform is highlighted")
    print("3. Note the HSV values shown")
    print("4. Press 'q' to quit")
    print("5. Update config_final.py with these values")
    print("=" * 70 + "\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    # Create window and sliders
    cv2.namedWindow('Calibration')
    cv2.createTrackbar('H Low', 'Calibration', 90, 180, lambda x: None)
    cv2.createTrackbar('H High', 'Calibration', 130, 180, lambda x: None)
    cv2.createTrackbar('S Low', 'Calibration', 50, 255, lambda x: None)
    cv2.createTrackbar('S High', 'Calibration', 255, 255, lambda x: None)
    cv2.createTrackbar('V Low', 'Calibration', 80, 255, lambda x: None)
    cv2.createTrackbar('V High', 'Calibration', 255, 255, lambda x: None)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get trackbar values
        h_low = cv2.getTrackbarPos('H Low', 'Calibration')
        h_high = cv2.getTrackbarPos('H High', 'Calibration')
        s_low = cv2.getTrackbarPos('S Low', 'Calibration')
        s_high = cv2.getTrackbarPos('S High', 'Calibration')
        v_low = cv2.getTrackbarPos('V Low', 'Calibration')
        v_high = cv2.getTrackbarPos('V High', 'Calibration')
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Calculate coverage
        coverage = (cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])) * 100
        
        # Apply mask to frame
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Add text
        text = f"HSV_LOWER = ({h_low}, {s_low}, {v_low})"
        text2 = f"HSV_UPPER = ({h_high}, {s_high}, {v_high})"
        text3 = f"Coverage: {coverage:.1f}% (aim for 20-40% for torso)"
        
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show side by side
        combined = np.hstack([frame, result])
        cv2.imshow('Calibration', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n" + "=" * 70)
            print("COPY THESE VALUES TO config_final.py:")
            print("=" * 70)
            print(f"UNIFORM_HSV_LOWER = ({h_low}, {s_low}, {v_low})")
            print(f"UNIFORM_HSV_UPPER = ({h_high}, {s_high}, {v_high})")
            print("=" * 70)
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    calibrate_uniform_color()
    