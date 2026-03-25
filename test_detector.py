"""
Test script to verify A-Z detection and word building
Run this to test detection without the web app
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


class SimpleASLDetector:
    """Simple ASL A-Z detector for testing"""
    
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_mcps = [2, 5, 9, 13, 17]
        
    def get_extended_fingers(self, landmarks):
        """Check which fingers are extended"""
        extended = []
        wrist = landmarks[0]
        
        # Thumb
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) + 0.03
        extended.append(thumb_extended)
        
        # Other fingers
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            mcp = landmarks[self.finger_mcps[i]]
            finger_extended = tip.y < mcp.y - 0.04
            extended.append(finger_extended)
        
        return extended
    
    def detect(self, landmarks):
        """Detect ASL letter (simplified for testing)"""
        if not landmarks or len(landmarks) != 21:
            return "", 0.0
        
        extended = self.get_extended_fingers(landmarks)
        
        # Common letters for testing
        # A - Fist with thumb
        if not any(extended[1:5]) and extended[0]:
            return "A", 0.90
        
        # B - Four fingers up
        if all(extended[1:5]) and not extended[0]:
            return "B", 0.90
        
        # V - Peace sign
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            if abs(index_tip.x - middle_tip.x) > 0.05:
                return "V", 0.92
        
        # L - Thumb and index perpendicular
        if extended[0] and extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            thumb_vec = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
            index_vec = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
            
            cos_angle = np.dot(thumb_vec, index_vec) / (
                np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-6
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            if 70 < angle < 110:
                return "L", 0.90
        
        # I - Pinky only
        if extended[4] and not any(extended[0:4]):
            return "I", 0.92
        
        # Y - Thumb and pinky (shaka)
        if extended[0] and extended[4] and not any(extended[1:4]):
            return "Y", 0.90
        
        # O - Circle shape
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        if dist < 0.07:
            return "O", 0.88
        
        # U - Index and middle together
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            if abs(index_tip.x - middle_tip.x) < 0.03:
                return "U", 0.90
        
        return "", 0.0


def main():
    """Test detection with live camera"""
    print("=" * 70)
    print("🎯 ASL A-Z WORD DETECTION TEST")
    print("=" * 70)
    print("\nTesting ASL letter detection and word building...")
    print("\nSupported letters (for testing):")
    print("  A - Fist with thumb alongside")
    print("  B - Four fingers up, thumb down")
    print("  V - Peace sign (separated)")
    print("  L - Thumb and index perpendicular")
    print("  I - Pinky up")
    print("  Y - Shaka (thumb and pinky)")
    print("  O - Circle shape")
    print("  U - Index and middle together")
    print("\nControls:")
    print("  Hold letter for 1.5s to add to word")
    print("  Pause 3s (no hand) to complete word")
    print("  Press 'c' to clear current word")
    print("  Press 'q' to quit")
    print("=" * 70)
    
    # Initialize
    detector = SimpleASLDetector()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Word building
    current_word = ""
    completed_words = []
    last_letter = ""
    last_letter_time = 0
    last_detection_time = 0
    letter_hold_time = 1.5
    space_delay = 3.0
    
    # Smoothing
    predictions = deque(maxlen=10)
    
    print("✅ Camera opened successfully!")
    print("🎥 Starting detection...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Failed to read frame")
            continue
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        label, conf = "", 0.0
        current_time = time.time()
        
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )
            
            landmarks = results.multi_hand_landmarks[0].landmark
            label, conf = detector.detect(landmarks)
            
            if label and conf > 0.70:
                predictions.append(label)
                last_detection_time = current_time
                
                # Get stable prediction
                if len(predictions) >= 5:
                    counter = Counter(predictions)
                    most_common = counter.most_common(1)[0]
                    if most_common[1] >= 6:  # At least 6 times
                        stable_label = most_common[0]
                        
                        # Add to word if held
                        if stable_label == last_letter:
                            if current_time - last_letter_time >= letter_hold_time:
                                # Already added
                                pass
                        else:
                            current_word += stable_label
                            last_letter = stable_label
                            last_letter_time = current_time
                            print(f"✅ Added letter: {stable_label} -> Word: '{current_word}'")
        
        # Check for space (no detection for a while)
        if current_word and current_time - last_detection_time > space_delay:
            completed_words.append(current_word)
            print(f"📝 Word completed: '{current_word}'")
            print(f"   All words: {' | '.join(completed_words)}")
            current_word = ""
            last_letter = ""
            predictions.clear()
        
        # Display info
        h, w = frame.shape[:2]
        
        # Top - Current letter
        if label and conf > 0.70:
            cv2.rectangle(frame, (10, 10), (400, 80), (0, 100, 0), -1)
            cv2.putText(frame, f"Letter: {label}", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Conf: {conf:.0%}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (10, 10), (400, 50), (50, 50, 50), -1)
            cv2.putText(frame, "Show ASL letter...", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Middle - Current word
        cv2.rectangle(frame, (10, 100), (630, 160), (60, 60, 60), -1)
        cv2.putText(frame, f"Word: {current_word if current_word else '(empty)'}", (20, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        
        # Bottom - Completed words
        if completed_words:
            cv2.rectangle(frame, (10, h - 80), (630, h - 10), (40, 40, 40), -1)
            recent = ' | '.join(completed_words[-3:])
            cv2.putText(frame, f"Words: {recent}", (20, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
            cv2.putText(frame, f"Total: {len(completed_words)}", (20, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Show frame
        cv2.imshow('ASL Word Detection Test', frame)
        
        # Check for quit or clear
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_word = ""
            last_letter = ""
            print("🗑️ Current word cleared")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print(f"✅ Test complete!")
    print(f"   Total words created: {len(completed_words)}")
    if completed_words:
        print(f"   Words: {' | '.join(completed_words)}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()