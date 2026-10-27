import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import mediapipe as mp
from subject_lock import ExerciseSubjectLock
from pushup import PushupTracker


class DynamicBoundingBox:
    """
    Computes an adaptive, padded bounding box around detected landmarks
    and applies exponential smoothing (EMA) to follow movement smoothly.
    """
    def __init__(self, padding_ratio=0.12, smoothing_factor=0.25):
        self.padding_ratio = padding_ratio
        self.smoothing_factor = smoothing_factor
        self.smooth_box = None  # [xmin, ymin, xmax, ymax] normalized coords

    def calculate_raw_box(self, landmarks):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        raw_xmin, raw_xmax = min(xs), max(xs)
        raw_ymin, raw_ymax = min(ys), max(ys)

        width = raw_xmax - raw_xmin
        height = raw_ymax - raw_ymin

        pad_x = max(width * self.padding_ratio, 0.05)
        pad_y = max(height * self.padding_ratio, 0.05)

        xmin = max(0.0, raw_xmin - pad_x)
        ymin = max(0.0, raw_ymin - pad_y)
        xmax = min(1.0, raw_xmax + pad_x)
        ymax = min(1.0, raw_ymax + pad_y)

        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

    def update(self, landmarks):
        current_box = self.calculate_raw_box(landmarks)
        if self.smooth_box is None:
            self.smooth_box = current_box
        else:
            alpha = self.smoothing_factor
            self.smooth_box = (1.0 - alpha) * self.smooth_box + alpha * current_box
        return self.smooth_box

    def reset(self):
        self.smooth_box = None


def main():
    tracker = PushupTracker()
    
    # Windows camera access
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = 'Backend Engine Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("🎥 Starting webcam... Press 'q' to quit, 'r' to reset lock, 'f' to toggle fullscreen.")

    # MediaPipe pose & drawing utils
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles

    # Lock and Bounding Box Tracker
    lock = ExerciseSubjectLock(max_jump_distance=0.18, max_lost_frames=45)
    bbox_tracker = DynamicBoundingBox(padding_ratio=0.12, smoothing_factor=0.25)

    last_telemetry = {
        "state": tracker.state,
        "total_reps": tracker.good_reps + tracker.bad_reps + tracker.half_reps,
        "good_reps": tracker.good_reps,
        "half_reps": tracker.half_reps,
        "bad_reps": tracker.bad_reps,
        "form_score": 0,
        "feedback": tracker.current_feedback
    }

    with mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            results = pose.process(rgb_image)
            frame.flags.writeable = True

            if results.pose_landmarks:
                valid_person, landmarks = lock.validate_and_update(results.pose_landmarks.landmark)

                if valid_person:
                    # 1. Update and render dynamic moving bounding box
                    smooth_box = bbox_tracker.update(landmarks)
                    bx1, by1, bx2, by2 = smooth_box
                    p1 = (int(bx1 * w), int(by1 * h))
                    p2 = (int(bx2 * w), int(by2 * h))

                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame, "TARGET LOCKED", (p1[0], max(p1[1] - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    # 2. Draw Skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # 3. Draw Hip centroid anchor
                    hip_px = int(lock.locked_hip[0] * w)
                    hip_py = int(lock.locked_hip[1] * h)
                    cv2.circle(frame, (hip_px, hip_py), 8, (0, 255, 0), -1)

                    # 4. Feed clean landmarks to exercise tracker
                    telemetry = tracker.process_frame(landmarks)
                    last_telemetry["state"] = telemetry.get('state', last_telemetry['state'])
                    last_telemetry["total_reps"] = telemetry.get('total_reps', last_telemetry['total_reps'])
                    last_telemetry["good_reps"] = telemetry.get('good_reps', last_telemetry['good_reps'])
                    last_telemetry["half_reps"] = telemetry.get('half_reps', last_telemetry['half_reps'])
                    last_telemetry["bad_reps"] = telemetry.get('bad_reps', last_telemetry['bad_reps'])
                    last_telemetry["form_score"] = telemetry.get('form_score', last_telemetry['form_score'])
                    last_telemetry["feedback"] = telemetry.get('feedback', last_telemetry['feedback'])
                else:
                    # Passerby or occluded frame - freeze bounding box at last known location in amber
                    if bbox_tracker.smooth_box is not None:
                        bx1, by1, bx2, by2 = bbox_tracker.smooth_box
                        p1 = (int(bx1 * w), int(by1 * h))
                        p2 = (int(bx2 * w), int(by2 * h))
                        cv2.rectangle(frame, p1, p2, (0, 165, 255), 2)

                    last_telemetry["state"] = "OCCLUDED / PASSERBY"
                    last_telemetry["feedback"] = "Ignoring passerby - reacquiring subject..."
            else:
                lock.validate_and_update(None)
                if not lock.is_locked:
                    bbox_tracker.reset()

                last_telemetry["state"] = "NO PERSON"
                last_telemetry["feedback"] = "Step into frame to begin"

            # ---------------------------------------------------------
            # UI OVERLAY (Persistent Telemetry)
            # ---------------------------------------------------------
            feedback_text = last_telemetry["feedback"]

            if "No rep" in feedback_text or "Danger" in feedback_text:
                feedback_color = (0, 0, 255)
            elif "Half rep" in feedback_text or "Half Rep" in feedback_text:
                feedback_color = (0, 140, 255)
            elif any(w in feedback_text for w in ["Perfect", "Great", "Good"]):
                feedback_color = (0, 255, 0)
            elif last_telemetry["state"] in ["NO PERSON", "OCCLUDED / PASSERBY"]:
                feedback_color = (0, 165, 255)
            else:
                feedback_color = (0, 255, 255)

            cv2.putText(frame, f"State: {last_telemetry['state']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            cv2.putText(frame, f"Total Reps: {last_telemetry['total_reps']}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Good: {last_telemetry['good_reps']} | Half: {last_telemetry['half_reps']} | Bad: {last_telemetry['bad_reps']}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Score: {last_telemetry['form_score']}%", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"Feedback: {feedback_text}", (10, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, feedback_color, 2)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                lock.reset()
                bbox_tracker.reset()
                print("Target lock manually reset.")
            elif key == ord('f'):
                is_full = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                new_prop = cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_prop)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()