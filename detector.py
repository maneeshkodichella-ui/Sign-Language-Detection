import threading
from typing import Tuple, Optional, List, Any
import cv2
import numpy as np
import logging
from collections import deque, Counter
import time
import gc
import base64

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    raise ImportError("MediaPipe not installed") from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_state_lock = threading.Lock()
_latest_label: Tuple[str, float] = ("", 0.0)
_detection_count: int = 0


def get_latest_label() -> Tuple[str, float]:
    with _state_lock:
        return _latest_label


def get_detection_count() -> int:
    with _state_lock:
        return _detection_count


class ASLRecognizer:
    """Complete ASL alphabet recognizer (A-Z) with two-hand support"""
    
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [3, 6, 10, 14, 18]
        self.finger_mcps = [2, 5, 9, 13, 17]
        
    def recognize(self, landmarks: Any, all_landmarks: Optional[List[Any]] = None, 
                  handedness_list: Optional[List[Any]] = None) -> Tuple[str, float]:
        """Recognize ASL letter from hand landmarks
        
        Args:
            landmarks: Primary hand landmarks (backwards compatible)
            all_landmarks: List of all detected hands for two-hand gestures
            handedness_list: List of handedness info (Left/Right) for each hand
        """
        if landmarks is None or len(landmarks) != 21:
            return "", 0.0
        
        # Check for two-hand gestures first if multiple hands detected
        if all_landmarks and len(all_landmarks) >= 2:
            two_hand_result = self._check_two_hand_letters(
                all_landmarks, 
                handedness_list
            )
            if two_hand_result[0]:
                return two_hand_result
        
        # Fall back to single-hand detection
        extended = self._get_extended_fingers(landmarks)
        
        checks = [
            ("Y", self._is_letter_y, landmarks, extended),
            ("I", self._is_letter_i, extended, landmarks),
            ("L", self._is_letter_l, landmarks, extended),
            ("V", self._is_letter_v, landmarks, extended),
            ("W", self._is_letter_w, landmarks, extended),
            ("B", self._is_letter_b, extended, landmarks),
            ("C", self._is_letter_c, landmarks),
            ("O", self._is_letter_o, landmarks),
            ("F", self._is_letter_f, landmarks, extended),
            ("D", self._is_letter_d, landmarks, extended),
            ("R", self._is_letter_r, landmarks, extended),
            ("U", self._is_letter_u, landmarks, extended),
            ("K", self._is_letter_k, landmarks, extended),
            ("H", self._is_letter_h, landmarks, extended),
            ("G", self._is_letter_g, landmarks, extended),
            ("A", self._is_letter_a, extended, landmarks),
            ("S", self._is_letter_s, landmarks, extended),
            ("E", self._is_letter_e, landmarks),
            ("M", self._is_letter_m, landmarks, extended),
            ("N", self._is_letter_n, landmarks, extended),
            ("T", self._is_letter_t, landmarks, extended),
            ("P", self._is_letter_p, landmarks, extended),
            ("Q", self._is_letter_q, landmarks, extended),
            ("X", self._is_letter_x, landmarks, extended),
            ("J", self._is_letter_j, landmarks, extended),
            ("Z", self._is_letter_z, landmarks, extended),
        ]
        
        for letter, check_func, *args in checks:
            confidence = check_func(*args)
            if confidence > 0.0:
                return letter, float(confidence)
        
        return "", 0.0
    
    def _check_two_hand_letters(self, all_landmarks: List[Any], 
                                handedness_list: Optional[List[Any]] = None) -> Tuple[str, float]:
        """Check for letters that require two hands
        
        Some ASL letters can benefit from two-hand variations or confirmations.
        This is a framework for adding specific two-hand patterns.
        """
        if len(all_landmarks) < 2:
            return "", 0.0
        
        # Determine left and right hands
        hand1 = all_landmarks[0]
        hand2 = all_landmarks[1]
        
        # Use handedness info if available, otherwise use x-coordinate
        if handedness_list and len(handedness_list) >= 2:
            if handedness_list[0].classification[0].label == "Left":
                left_hand = hand1
                right_hand = hand2
            else:
                left_hand = hand2
                right_hand = hand1
        else:
            # Fallback: determine by x-coordinate (left hand appears on right side of screen)
            if hand1[0].x < hand2[0].x:
                left_hand = hand1
                right_hand = hand2
            else:
                left_hand = hand2
                right_hand = hand1
        
        # Get extended fingers for both hands
        left_extended = self._get_extended_fingers(left_hand)
        right_extended = self._get_extended_fingers(right_hand)
        
        # Example: Two-hand confirmation for high-confidence detection
        # You can add specific two-hand letters here
        
        # For now, return empty to allow single-hand detection
        # Future additions: specific two-hand ASL patterns
        
        return "", 0.0
    
    def _get_extended_fingers(self, landmarks: Any) -> List[bool]:
        extended = []
        wrist = landmarks[0]
        
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) + 0.02
        extended.append(thumb_extended)
        
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            mcp = landmarks[self.finger_mcps[i]]
            finger_extended = tip.y < mcp.y - 0.03
            extended.append(finger_extended)
        
        return extended
    
    def _distance(self, p1: Any, p2: Any) -> float:
        return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2))
    
    def _is_letter_a(self, extended: List[bool], landmarks: Any) -> float:
        if not any(extended[1:5]) and extended[0]:
            thumb_tip = landmarks[4]
            index_mcp = landmarks[5]
            if abs(thumb_tip.y - index_mcp.y) < 0.10:
                return 0.92
            return 0.85
        return 0.0
    
    def _is_letter_b(self, extended: List[bool], landmarks: Any) -> float:
        if all(extended[1:5]) and not extended[0]:
            tips_aligned = all(
                abs(landmarks[self.finger_tips[i]].y - landmarks[self.finger_tips[i+1]].y) < 0.05
                for i in range(1, 4)
            )
            return 0.94 if tips_aligned else 0.86
        return 0.0
    
    def _is_letter_c(self, landmarks: Any) -> float:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        gap = self._distance(thumb_tip, index_tip)
        
        if 0.08 < gap < 0.28:
            all_curved = all(
                landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y + 0.02
                for i in range(1, 5)
            )
            if all_curved:
                return 0.90
            return 0.82
        return 0.0
    
    def _is_letter_d(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            
            if self._distance(thumb_tip, middle_tip) < 0.09 or self._distance(thumb_tip, ring_tip) < 0.09:
                return 0.92
            return 0.83
        return 0.0
    
    def _is_letter_e(self, landmarks: Any) -> float:
        all_curled = all(
            landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.02
            for i in range(1, 5)
        )
        
        if all_curled:
            thumb_tip = landmarks[4]
            index_pip = landmarks[6]
            if self._distance(thumb_tip, index_pip) < 0.10:
                return 0.89
            return 0.81
        return 0.0
    
    def _is_letter_f(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[2] and extended[3] and extended[4]:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            if self._distance(thumb_tip, index_tip) < 0.07:
                return 0.92
            return 0.83
        return 0.0
    
    def _is_letter_g(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[0] and extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            if abs(thumb_tip.y - index_tip.y) < 0.08:
                if abs(thumb_tip.y - wrist.y) < 0.18:
                    return 0.90
                return 0.82
        return 0.0
    
    def _is_letter_h(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            wrist = landmarks[0]
            
            if (abs(index_tip.x - middle_tip.x) < 0.04 and
                abs(index_tip.y - middle_tip.y) < 0.05 and
                abs(index_tip.y - wrist.y) < 0.15):
                return 0.90
        return 0.0
    
    def _is_letter_i(self, extended: List[bool], landmarks: Any) -> float:
        if extended[4] and not any(extended[0:4]):
            fist_formed = all(
                landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.05
                for i in range(1, 4)
            )
            if fist_formed:
                return 0.94
            return 0.85
        return 0.0
    
    def _is_letter_j(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[4] and not any(extended[0:4]):
            return 0.86
        return 0.0
    
    def _is_letter_k(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            thumb_tip = landmarks[4]
            index_base = landmarks[5]
            middle_base = landmarks[9]
            
            if (thumb_tip.y < index_base.y and 
                abs(thumb_tip.x - (index_base.x + middle_base.x) / 2) < 0.05):
                return 0.90
            return 0.82
        return 0.0
    
    def _is_letter_l(self, landmarks: Any, extended: List[bool]) -> float:
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
            
            if 65 < angle < 115:
                return 0.93
            return 0.82
        return 0.0
    
    def _is_letter_m(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[3:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            
            if (self._distance(thumb_tip, index_tip) < 0.10 and
                self._distance(thumb_tip, middle_tip) < 0.12 and
                self._distance(thumb_tip, ring_tip) < 0.14):
                return 0.87
        return 0.0
    
    def _is_letter_n(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[2:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            if (self._distance(thumb_tip, index_tip) < 0.10 and
                self._distance(thumb_tip, middle_tip) < 0.12):
                return 0.87
        return 0.0
    
    def _is_letter_o(self, landmarks: Any) -> float:
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        if self._distance(thumb_tip, index_tip) < 0.08:
            all_curved = all(
                landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y + 0.04
                for i in range(1, 5)
            )
            if all_curved:
                return 0.92
            return 0.83
        return 0.0
    
    def _is_letter_p(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            if index_tip.y > wrist.y + 0.05:
                return 0.87
        return 0.0
    
    def _is_letter_q(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[0] and extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            wrist = landmarks[0]
            
            if thumb_tip.y > wrist.y + 0.05 and index_tip.y > wrist.y + 0.05:
                return 0.87
        return 0.0
    
    def _is_letter_r(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            index_pip = landmarks[6]
            middle_pip = landmarks[10]
            
            if self._distance(index_tip, middle_tip) < 0.04:
                pip_distance = self._distance(index_pip, middle_pip)
                if pip_distance > 0.02:
                    return 0.91
                return 0.83
        return 0.0
    
    def _is_letter_s(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[1:5]):
            thumb_tip = landmarks[4]
            index_mcp = landmarks[5]
            middle_mcp = landmarks[9]
            
            if (self._distance(thumb_tip, index_mcp) < 0.07 or
                self._distance(thumb_tip, middle_mcp) < 0.08):
                return 0.90
        return 0.0
    
    def _is_letter_t(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[1:5]):
            thumb_tip = landmarks[4]
            index_pip = landmarks[6]
            middle_pip = landmarks[10]
            
            if (self._distance(thumb_tip, index_pip) < 0.08 and
                self._distance(thumb_tip, middle_pip) < 0.10):
                return 0.89
        return 0.0
    
    def _is_letter_u(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            if abs(index_tip.x - middle_tip.x) < 0.04:
                if abs(index_tip.y - middle_tip.y) < 0.05:
                    return 0.93
                return 0.85
        return 0.0
    
    def _is_letter_v(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            separation = abs(index_tip.x - middle_tip.x)
            
            if separation > 0.06:
                if abs(index_tip.y - middle_tip.y) < 0.06:
                    return 0.95
                return 0.86
        return 0.0
    
    def _is_letter_w(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and extended[2] and extended[3] and not extended[4]:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            
            sep1 = abs(index_tip.x - middle_tip.x)
            sep2 = abs(middle_tip.x - ring_tip.x)
            
            if sep1 > 0.04 and sep2 > 0.04:
                if (abs(index_tip.y - middle_tip.y) < 0.06 and
                    abs(middle_tip.y - ring_tip.y) < 0.06):
                    return 0.93
                return 0.84
        return 0.0
    
    def _is_letter_x(self, landmarks: Any, extended: List[bool]) -> float:
        if not any(extended[2:5]) and not extended[0]:
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            wrist = landmarks[0]
            
            if (index_tip.y < wrist.y and 
                index_tip.y > index_mcp.y - 0.10):
                return 0.87
        return 0.0
    
    def _is_letter_y(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[0] and extended[4] and not any(extended[1:4]):
            thumb_tip = landmarks[4]
            pinky_tip = landmarks[20]
            
            if self._distance(thumb_tip, pinky_tip) > 0.16:
                return 0.94
            return 0.84
        return 0.0
    
    def _is_letter_z(self, landmarks: Any, extended: List[bool]) -> float:
        if extended[1] and not any(extended[2:5]):
            return 0.84
        return 0.0


class PredictionSmoother:
    """Advanced smoothing for stable letter detection"""
    
    def __init__(self, window_size: int = 10, min_confidence: float = 0.75, consensus_threshold: float = 0.60):
        self.predictions: deque = deque(maxlen=window_size)
        self.confidences: deque = deque(maxlen=window_size)
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.consensus_threshold = consensus_threshold
        
    def add_prediction(self, label: str, confidence: float):
        self.predictions.append(label)
        self.confidences.append(confidence)
        
    def get_stable_prediction(self) -> Tuple[str, float]:
        if len(self.predictions) < 5:
            return "", 0.0
        
        valid = [
            (p, c) for p, c in zip(self.predictions, self.confidences)
            if c >= self.min_confidence and p != ""
        ]
        
        if len(valid) < 3:
            return "", 0.0
        
        labels = [p for p, _ in valid]
        counter = Counter(labels)
        
        if not counter:
            return "", 0.0
        
        most_common_label, count = counter.most_common(1)[0]
        
        consensus = count / len(self.predictions)
        if consensus < self.consensus_threshold:
            return "", 0.0
        
        label_confs = [c for p, c in valid if p == most_common_label]
        avg_conf = float(np.mean(label_confs))
        final_conf = min(avg_conf + 0.03, 0.98)
        
        return most_common_label, final_conf
    
    def clear(self):
        self.predictions.clear()
        self.confidences.clear()


class VideoCamera:
    """Enhanced video camera with ASL detection - 1 or 2 Hand Support"""
    
    def __init__(self, socketio=None, room=None):
        logger.info("🎥 Initializing VideoCamera for 1-2 hand letter detection...")
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.hands: Optional[Any] = None
        self.is_running = False
        self.frame_count = 0
        self.process_every_n_frames = 2
        self.last_gc_time = time.time()
        self.last_successful_frame_time = time.time()
        
        # Socket.IO integration
        self.socketio = socketio
        self.room = room
        
        self._initialize_camera()
        self._initialize_mediapipe()
        
        self.recognizer = ASLRecognizer()
        self.smoother = PredictionSmoother(
            window_size=10, 
            min_confidence=0.75,
            consensus_threshold=0.60
        )
        
        self.last_label = ""
        self.last_conf = 0.0
        self.detection_count = 0
        
        # Thread for processing
        self.processing_thread = None
        
        logger.info("✅ VideoCamera initialized for 1-2 hand letter detection")

    def _initialize_camera(self):
        for idx in [0, 1, -1]:
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    time.sleep(0.2)
                    
                    for _ in range(10):
                        cap.read()
                    
                    success, _ = cap.read()
                    if success:
                        self.cap = cap
                        logger.info(f"✅ Camera opened on index {idx}")
                        return
                    cap.release()
            except Exception as e:
                logger.warning(f"Camera {idx} failed: {e}")
        
        raise RuntimeError("❌ No camera available")

    def _initialize_mediapipe(self):
        try:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands=2  # Support 2 hands
            )
            logger.info("✅ MediaPipe initialized with 2-hand support")
        except Exception as e:
            logger.error(f"❌ MediaPipe initialization failed: {e}")
            raise

    def start_detection(self):
        """Start detection in background thread"""
        if self.is_running:
            logger.warning("⚠️ Detection already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("▶️ Detection thread started")

    def _detection_loop(self):
        """Main detection loop that emits frames via Socket.IO"""
        global _latest_label, _detection_count
        
        consecutive_failures = 0
        max_consecutive_failures = 30
        last_detected_letter = ""
        
        logger.info("▶️ Starting detection loop")
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.error("❌ Camera closed unexpectedly")
                    break
                
                success, frame = self.cap.read()
                
                if not success:
                    consecutive_failures += 1
                    logger.warning(f"⚠️ Failed to read frame (consecutive failures: {consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("❌ Too many consecutive failures - stopping")
                        break
                    
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                self.frame_count += 1
                self.last_successful_frame_time = time.time()
                
                if self.frame_count % 100 == 0:
                    logger.info(f"✅ Frame {self.frame_count} processed successfully")
                
                frame = cv2.flip(frame, 1)
                
                current_time = time.time()
                if current_time - self.last_gc_time > 15.0:
                    gc.collect()
                    self.last_gc_time = current_time
                
                should_process = (self.frame_count % self.process_every_n_frames == 0)
                
                if should_process:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.hands.process(rgb)
                        
                        if results is not None and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                            # Draw landmarks for all detected hands
                            num_hands = len(results.multi_hand_landmarks)
                            for idx, handLms in enumerate(results.multi_hand_landmarks):
                                # Use different colors for left/right hands
                                if num_hands > 1:
                                    color = (0, 255, 0) if idx == 0 else (255, 165, 0)  # Green for first, orange for second
                                else:
                                    color = (0, 255, 0)
                                
                                mp_drawing.draw_landmarks(
                                    frame, handLms, 
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                                )
                            
                            # Get all hand landmarks and handedness for recognition
                            all_landmarks = [hand.landmark for hand in results.multi_hand_landmarks]
                            primary_landmarks = results.multi_hand_landmarks[0].landmark
                            
                            # Get handedness info if available
                            handedness_list = results.multi_handedness if hasattr(results, 'multi_handedness') else None
                            
                            # Pass all information to recognizer
                            current_label, current_conf = self.recognizer.recognize(
                                primary_landmarks, 
                                all_landmarks if len(all_landmarks) > 1 else None,
                                handedness_list
                            )
                            
                            if current_label and current_conf > 0.70:
                                self.smoother.add_prediction(current_label, current_conf)
                                self.last_label = current_label
                                self.last_conf = current_conf
                        else:
                            self.smoother.add_prediction("", 0.0)
                            self.last_label = ""
                            self.last_conf = 0.0
                        
                        label, conf = self.smoother.get_stable_prediction()
                        
                        # Count unique letter detections
                        if label and label != last_detected_letter and conf > 0.80:
                            self.detection_count += 1
                            last_detected_letter = label
                            logger.info(f"✅ Detected letter '{label}' with confidence {conf:.2%}")
                            
                            with _state_lock:
                                _detection_count = self.detection_count
                        
                        with _state_lock:
                            _latest_label = (label, conf)
                    except Exception as e:
                        logger.error(f"❌ MediaPipe processing error: {e}")
                        label, conf = "", 0.0
                else:
                    label, conf = self.last_label, self.last_conf
                
                # Get number of hands detected for UI
                num_hands = len(results.multi_hand_landmarks) if (results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks) else 0
                
                self._draw_ui(frame, label, conf, self.detection_count, num_hands)
                
                # Encode frame to base64 and emit via Socket.IO
                if self.socketio and self.room:
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Emit frame and detection state together
                        self.socketio.emit('video_frame', {
                            'image': frame_base64,
                            'label': label,
                            'confidence': float(conf),
                            'detection_count': self.detection_count,
                            'num_hands': num_hands
                        }, room=self.room)
                    except Exception as e:
                        logger.error(f"❌ Socket.IO emit error: {e}")
                
                time.sleep(0.033)  # ~30 FPS
                       
            except Exception as e:
                logger.error(f"❌ Detection loop error: {e}", exc_info=True)
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("❌ Too many errors - stopping")
                    break
                time.sleep(0.1)
        
        self.is_running = False
        logger.info("⏹️ Detection loop stopped")

    def _draw_ui(self, frame: np.ndarray, label: str, conf: float, detection_count: int, num_hands: int = 0):
        h, w = frame.shape[:2]
        
        # Top banner - Current detection
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        
        if label and conf > 0.70:
            cv2.putText(frame, f"Detected: {label}", (15, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            # Confidence bar
            bar_width = int(350 * conf)
            cv2.rectangle(frame, (15, 75), (365, 100), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, 75), (15 + bar_width, 100), (0, 255, 0), -1)
            cv2.putText(frame, f"{conf:.0%}", (375, 93), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Hand count indicator
            hand_color = (0, 255, 0) if num_hands > 0 else (100, 100, 100)
            cv2.putText(frame, f"Hands: {num_hands}", (15, 125), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        else:
            cv2.putText(frame, "Show ASL letter...", (15, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            
            # Hand count indicator
            if num_hands > 0:
                hand_color = (0, 255, 0) if num_hands <= 2 else (255, 165, 0)
                cv2.putText(frame, f"Hands detected: {num_hands}", (15, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
        
        # Bottom banner - Stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        cv2.putText(frame, f"Total Detections: {detection_count}", (15, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        
        # Instructions
        instructions = [
            "1 or 2 hands supported | Hold steady",
            "Good lighting improves accuracy"
        ]
        
        y_offset = 20
        for instruction in instructions:
            cv2.putText(frame, instruction, (w - 420, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 22

    def reset_count(self):
        """Reset detection counter"""
        self.detection_count = 0
        global _detection_count
        with _state_lock:
            _detection_count = 0
        logger.info("🔄 Detection count reset")

    def stop(self):
        logger.info("🛑 Stopping camera...")
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.hands is not None:
            try:
                self.hands.close()
                self.hands = None
                logger.info("✅ MediaPipe hands closed")
            except Exception as e:
                logger.error(f"❌ Error closing MediaPipe: {e}")
        
        if self.cap is not None:
            try:
                self.cap.release()
                self.cap = None
                logger.info("✅ Camera released")
            except Exception as e:
                logger.error(f"❌ Error releasing camera: {e}")
        
        gc.collect()
        logger.info("✅ Stop complete")

    def __del__(self):
        self.is_running = False
        
        if self.hands:
            try:
                self.hands.close()
            except:
                pass
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
        
        gc.collect()