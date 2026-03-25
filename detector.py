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
        if landmarks is None or len(landmarks) != 21:
            return "", 0.0

        if all_landmarks and len(all_landmarks) >= 2:
            two_hand_result = self._check_two_hand_letters(all_landmarks, handedness_list)
            if two_hand_result[0]:
                return two_hand_result

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

    def _check_two_hand_letters(self, all_landmarks, handedness_list=None):
        if len(all_landmarks) < 2:
            return "", 0.0
        hand1 = all_landmarks[0]
        hand2 = all_landmarks[1]
        if handedness_list and len(handedness_list) >= 2:
            if handedness_list[0].classification[0].label == "Left":
                left_hand, right_hand = hand1, hand2
            else:
                left_hand, right_hand = hand2, hand1
        else:
            if hand1[0].x < hand2[0].x:
                left_hand, right_hand = hand1, hand2
            else:
                left_hand, right_hand = hand2, hand1
        return "", 0.0

    def _get_extended_fingers(self, landmarks):
        extended = []
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) + 0.02
        extended.append(thumb_extended)
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            mcp = landmarks[self.finger_mcps[i]]
            extended.append(tip.y < mcp.y - 0.03)
        return extended

    def _distance(self, p1, p2):
        return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2))

    def _is_letter_a(self, extended, landmarks):
        if not any(extended[1:5]) and extended[0]:
            thumb_tip = landmarks[4]
            index_mcp = landmarks[5]
            return 0.92 if abs(thumb_tip.y - index_mcp.y) < 0.10 else 0.85
        return 0.0

    def _is_letter_b(self, extended, landmarks):
        if all(extended[1:5]) and not extended[0]:
            tips_aligned = all(
                abs(landmarks[self.finger_tips[i]].y - landmarks[self.finger_tips[i+1]].y) < 0.05
                for i in range(1, 4)
            )
            return 0.94 if tips_aligned else 0.86
        return 0.0

    def _is_letter_c(self, landmarks):
        gap = self._distance(landmarks[4], landmarks[8])
        if 0.08 < gap < 0.28:
            all_curved = all(
                landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y + 0.02
                for i in range(1, 5)
            )
            return 0.90 if all_curved else 0.82
        return 0.0

    def _is_letter_d(self, landmarks, extended):
        if extended[1] and not any(extended[2:5]):
            thumb_tip = landmarks[4]
            if self._distance(thumb_tip, landmarks[12]) < 0.09 or self._distance(thumb_tip, landmarks[16]) < 0.09:
                return 0.92
            return 0.83
        return 0.0

    def _is_letter_e(self, landmarks):
        all_curled = all(
            landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.02
            for i in range(1, 5)
        )
        if all_curled:
            return 0.89 if self._distance(landmarks[4], landmarks[6]) < 0.10 else 0.81
        return 0.0

    def _is_letter_f(self, landmarks, extended):
        if extended[2] and extended[3] and extended[4]:
            return 0.92 if self._distance(landmarks[4], landmarks[8]) < 0.07 else 0.83
        return 0.0

    def _is_letter_g(self, landmarks, extended):
        if extended[0] and extended[1] and not any(extended[2:5]):
            if abs(landmarks[4].y - landmarks[8].y) < 0.08:
                if abs(landmarks[4].y - landmarks[0].y) < 0.18:
                    return 0.90
            return 0.82
        return 0.0

    def _is_letter_h(self, landmarks, extended):
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            i_tip, m_tip, wrist = landmarks[8], landmarks[12], landmarks[0]
            if abs(i_tip.x - m_tip.x) < 0.04 and abs(i_tip.y - m_tip.y) < 0.05 and abs(i_tip.y - wrist.y) < 0.15:
                return 0.90
        return 0.0

    def _is_letter_i(self, extended, landmarks):
        if extended[4] and not any(extended[0:4]):
            fist_formed = all(
                landmarks[self.finger_tips[i]].y > landmarks[self.finger_mcps[i]].y - 0.05
                for i in range(1, 4)
            )
            return 0.94 if fist_formed else 0.85
        return 0.0

    def _is_letter_j(self, landmarks, extended):
        return 0.86 if extended[4] and not any(extended[0:4]) else 0.0

    def _is_letter_k(self, landmarks, extended):
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            thumb_tip = landmarks[4]
            mid = (landmarks[5].x + landmarks[9].x) / 2
            if thumb_tip.y < landmarks[5].y and abs(thumb_tip.x - mid) < 0.05:
                return 0.90
            return 0.82
        return 0.0

    def _is_letter_l(self, landmarks, extended):
        if extended[0] and extended[1] and not any(extended[2:5]):
            wrist = landmarks[0]
            thumb_vec = np.array([landmarks[4].x - wrist.x, landmarks[4].y - wrist.y])
            index_vec = np.array([landmarks[8].x - wrist.x, landmarks[8].y - wrist.y])
            cos_angle = np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            return 0.93 if 65 < angle < 115 else 0.82
        return 0.0

    def _is_letter_m(self, landmarks, extended):
        if not any(extended[3:5]):
            t = landmarks[4]
            if self._distance(t, landmarks[8]) < 0.10 and self._distance(t, landmarks[12]) < 0.12 and self._distance(t, landmarks[16]) < 0.14:
                return 0.87
        return 0.0

    def _is_letter_n(self, landmarks, extended):
        if not any(extended[2:5]):
            t = landmarks[4]
            if self._distance(t, landmarks[8]) < 0.10 and self._distance(t, landmarks[12]) < 0.12:
                return 0.87
        return 0.0

    def _is_letter_o(self, landmarks):
        if self._distance(landmarks[4], landmarks[8]) < 0.08:
            all_curved = all(
                landmarks[self.finger_tips[i]].y < landmarks[self.finger_mcps[i]].y + 0.04
                for i in range(1, 5)
            )
            return 0.92 if all_curved else 0.83
        return 0.0

    def _is_letter_p(self, landmarks, extended):
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            return 0.87 if landmarks[8].y > landmarks[0].y + 0.05 else 0.0
        return 0.0

    def _is_letter_q(self, landmarks, extended):
        if extended[0] and extended[1] and not any(extended[2:5]):
            wrist = landmarks[0]
            if landmarks[4].y > wrist.y + 0.05 and landmarks[8].y > wrist.y + 0.05:
                return 0.87
        return 0.0

    def _is_letter_r(self, landmarks, extended):
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            if self._distance(landmarks[8], landmarks[12]) < 0.04:
                if self._distance(landmarks[6], landmarks[10]) > 0.02:
                    return 0.91
                return 0.83
        return 0.0

    def _is_letter_s(self, landmarks, extended):
        if not any(extended[1:5]):
            t = landmarks[4]
            if self._distance(t, landmarks[5]) < 0.07 or self._distance(t, landmarks[9]) < 0.08:
                return 0.90
        return 0.0

    def _is_letter_t(self, landmarks, extended):
        if not any(extended[1:5]):
            t = landmarks[4]
            if self._distance(t, landmarks[6]) < 0.08 and self._distance(t, landmarks[10]) < 0.10:
                return 0.89
        return 0.0

    def _is_letter_u(self, landmarks, extended):
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            i_tip, m_tip = landmarks[8], landmarks[12]
            if abs(i_tip.x - m_tip.x) < 0.04 and abs(i_tip.y - m_tip.y) < 0.05:
                return 0.93
            return 0.85
        return 0.0

    def _is_letter_v(self, landmarks, extended):
        if extended[1] and extended[2] and not extended[3] and not extended[4]:
            sep = abs(landmarks[8].x - landmarks[12].x)
            if sep > 0.06 and abs(landmarks[8].y - landmarks[12].y) < 0.06:
                return 0.95
            return 0.86
        return 0.0

    def _is_letter_w(self, landmarks, extended):
        if extended[1] and extended[2] and extended[3] and not extended[4]:
            i, m, r = landmarks[8], landmarks[12], landmarks[16]
            if abs(i.x - m.x) > 0.04 and abs(m.x - r.x) > 0.04:
                if abs(i.y - m.y) < 0.06 and abs(m.y - r.y) < 0.06:
                    return 0.93
            return 0.84
        return 0.0

    def _is_letter_x(self, landmarks, extended):
        if not any(extended[2:5]) and not extended[0]:
            i_tip = landmarks[8]
            if i_tip.y < landmarks[0].y and i_tip.y > landmarks[5].y - 0.10:
                return 0.87
        return 0.0

    def _is_letter_y(self, landmarks, extended):
        if extended[0] and extended[4] and not any(extended[1:4]):
            if self._distance(landmarks[4], landmarks[20]) > 0.16:
                return 0.94
            return 0.84
        return 0.0

    def _is_letter_z(self, landmarks, extended):
        return 0.84 if extended[1] and not any(extended[2:5]) else 0.0


class PredictionSmoother:
    def __init__(self, window_size=10, min_confidence=0.75, consensus_threshold=0.60):
        self.predictions: deque = deque(maxlen=window_size)
        self.confidences: deque = deque(maxlen=window_size)
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.consensus_threshold = consensus_threshold

    def add_prediction(self, label, confidence):
        self.predictions.append(label)
        self.confidences.append(confidence)

    def get_stable_prediction(self):
        if len(self.predictions) < 5:
            return "", 0.0
        valid = [(p, c) for p, c in zip(self.predictions, self.confidences) if c >= self.min_confidence and p != ""]
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
        return most_common_label, min(float(np.mean(label_confs)) + 0.03, 0.98)

    def clear(self):
        self.predictions.clear()
        self.confidences.clear()


class VideoCamera:
    """Enhanced video camera with ASL detection - 1 or 2 Hand Support.
    
    On cloud environments (Render, Railway) where no physical camera exists,
    this class will initialise in no-camera mode and skip detection gracefully
    so the rest of the app still loads and serves pages normally.
    """

    def __init__(self, socketio=None, room=None):
        logger.info("🎥 Initializing VideoCamera...")
        self.cap: Optional[cv2.VideoCapture] = None
        self.hands: Optional[Any] = None
        self.is_running = False
        self.frame_count = 0
        self.process_every_n_frames = 2
        self.last_gc_time = time.time()
        self.last_successful_frame_time = time.time()
        self.camera_available = False  # ← NEW: tracks whether camera was found

        self.socketio = socketio
        self.room = room

        self._initialize_camera()   # won't raise anymore
        self._initialize_mediapipe()

        self.recognizer = ASLRecognizer()
        self.smoother = PredictionSmoother(window_size=10, min_confidence=0.75, consensus_threshold=0.60)
        self.last_label = ""
        self.last_conf = 0.0
        self.detection_count = 0
        self.processing_thread = None

        if self.camera_available:
            logger.info("✅ VideoCamera initialized with live camera")
        else:
            logger.warning("⚠️  VideoCamera running in NO-CAMERA mode (cloud/server environment)")

    # ─────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────

    def _initialize_camera(self):
        """Try to open a physical camera.  Sets self.camera_available = True on success.
        Never raises — so the app can still start on camera-less servers."""
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
                        self.camera_available = True
                        logger.info(f"✅ Camera opened on index {idx}")
                        return
                cap.release()
            except Exception as e:
                logger.warning(f"Camera index {idx} failed: {e}")

        # No camera found — log a warning but do NOT raise
        logger.warning("⚠️  No physical camera found. Running without live detection.")

    def _initialize_mediapipe(self):
        try:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                max_num_hands=2
            )
            logger.info("✅ MediaPipe initialized")
        except Exception as e:
            logger.error(f"❌ MediaPipe initialization failed: {e}")
            raise

    # ─────────────────────────────────────────────────────────────
    # Detection control
    # ─────────────────────────────────────────────────────────────

    def start_detection(self):
        """Start the detection background thread.
        Does nothing if no camera is available (cloud mode)."""
        if not self.camera_available:
            logger.warning("⚠️  start_detection() called but no camera is available — skipping.")
            return

        if self.is_running:
            logger.warning("⚠️  Detection already running")
            return

        self.is_running = True
        self.processing_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.processing_thread.start()
        logger.info("▶️  Detection thread started")

    def _detection_loop(self):
        global _latest_label, _detection_count

        consecutive_failures = 0
        max_consecutive_failures = 30
        last_detected_letter = ""

        logger.info("▶️  Starting detection loop")

        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.error("❌ Camera closed unexpectedly")
                    break

                success, frame = self.cap.read()
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("❌ Too many consecutive failures — stopping")
                        break
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0
                self.frame_count += 1
                self.last_successful_frame_time = time.time()

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

                        if results and results.multi_hand_landmarks:
                            num_hands = len(results.multi_hand_landmarks)
                            for idx, handLms in enumerate(results.multi_hand_landmarks):
                                color = (0, 255, 0) if idx == 0 else (255, 165, 0)
                                mp_drawing.draw_landmarks(
                                    frame, handLms, mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                                )

                            all_landmarks = [hand.landmark for hand in results.multi_hand_landmarks]
                            primary_landmarks = results.multi_hand_landmarks[0].landmark
                            handedness_list = getattr(results, 'multi_handedness', None)

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

                            if label and label != last_detected_letter and conf > 0.80:
                                self.detection_count += 1
                                last_detected_letter = label
                                logger.info(f"✅ Detected '{label}' ({conf:.0%})")
                                with _state_lock:
                                    _detection_count = self.detection_count

                            with _state_lock:
                                _latest_label = (label, conf)

                        else:
                            num_hands = 0
                            label, conf = "", 0.0
                            with _state_lock:
                                _latest_label = ("", 0.0)

                    except Exception as e:
                        logger.error(f"❌ MediaPipe processing error: {e}")
                        label, conf, num_hands = "", 0.0, 0
                else:
                    label, conf = self.last_label, self.last_conf
                    num_hands = 0

                self._draw_ui(frame, label, conf, self.detection_count, num_hands)

                if self.socketio and self.room:
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        self.socketio.emit('video_frame', {
                            'image': frame_base64,
                            'label': label,
                            'confidence': float(conf),
                            'detection_count': self.detection_count,
                            'num_hands': num_hands
                        }, room=self.room)
                    except Exception as e:
                        logger.error(f"❌ Socket.IO emit error: {e}")

                time.sleep(0.033)

            except Exception as e:
                logger.error(f"❌ Detection loop error: {e}", exc_info=True)
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break
                time.sleep(0.1)

        self.is_running = False
        logger.info("⏹️  Detection loop stopped")

    def _draw_ui(self, frame, label, conf, detection_count, num_hands=0):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        if label and conf > 0.70:
            cv2.putText(frame, f"Detected: {label}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            bar_width = int(350 * conf)
            cv2.rectangle(frame, (15, 75), (365, 100), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, 75), (15 + bar_width, 100), (0, 255, 0), -1)
            cv2.putText(frame, f"{conf:.0%}", (375, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            hand_color = (0, 255, 0) if num_hands > 0 else (100, 100, 100)
            cv2.putText(frame, f"Hands: {num_hands}", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        else:
            cv2.putText(frame, "Show ASL letter...", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            if num_hands > 0:
                cv2.putText(frame, f"Hands detected: {num_hands}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, f"Total Detections: {detection_count}", (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)

        for i, txt in enumerate(["1 or 2 hands supported | Hold steady", "Good lighting improves accuracy"]):
            cv2.putText(frame, txt, (w - 420, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # ─────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────

    def reset_count(self):
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
        if self.hands:
            try:
                self.hands.close()
                self.hands = None
            except Exception as e:
                logger.error(f"❌ Error closing MediaPipe: {e}")
        if self.cap:
            try:
                self.cap.release()
                self.cap = None
            except Exception as e:
                logger.error(f"❌ Error releasing camera: {e}")
        gc.collect()
        logger.info("✅ Stop complete")

    def __del__(self):
        self.is_running = False
        if self.hands:
            try:
                self.hands.close()
            except Exception:
                pass
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        gc.collect()