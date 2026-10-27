import sys
from pathlib import Path

# Add project root directory (formcheck) to sys.path so 'core' and 'exercises' resolve cleanly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import mediapipe as mp
from subject_lock import ExerciseSubjectLock

# Change this import to whichever exercise you want to test:
# from exercises.pushup import PushupTracker
# from exercises.squat import SquatTracker
from exercises.dumbbell_biceps_curls import DumbbellBicepsCurlsTracker


def main(tracker_instance=None):
    # Use provided tracker or fallback to Dumbbell Biceps Curls
    tracker = tracker_instance if tracker_instance is not None else DumbbellBicepsCurlsTracker()
    
    # Camera setup with DirectShow for reliable Windows performance
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = 'Backend Engine Test - Centroid Lock'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("🎥 Starting webcam... Press 'q' to quit, 'r' to reset lock, 'f' to toggle fullscreen.")

    # MediaPipe setup
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Centroid Subject Lock instance
    lock = ExerciseSubjectLock(max_jump_distance=0.18, max_lost_frames=45)

    # Persistent telemetry state
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

            # Mirror horizontally for natural workout display
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # MediaPipe processing in RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            frame.flags.writeable = True

            if results.pose_landmarks:
                valid_person, landmarks = lock.validate_and_update(results.pose_landmarks.landmark)

                if valid_person:
                    # 1. Draw Skeleton
                    mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                    # 2. Draw hip centroid anchor circle
                    hip_px = int(lock.locked_hip[0] * w)
                    hip_py = int(lock.locked_hip[1] * h)
                    cv2.circle(frame, (hip_px, hip_py), 8, (0, 255, 0), -1)

                    # 3. Process validated landmarks through exercise tracker
                    telemetry = tracker.process_frame(landmarks)
                    last_telemetry["state"] = telemetry.get('state', last_telemetry['state'])
                    last_telemetry["total_reps"] = telemetry.get('total_reps', last_telemetry['total_reps'])
                    last_telemetry["good_reps"] = telemetry.get('good_reps', last_telemetry['good_reps'])
                    last_telemetry["half_reps"] = telemetry.get('half_reps', last_telemetry['half_reps'])
                    last_telemetry["bad_reps"] = telemetry.get('bad_reps', last_telemetry['bad_reps'])
                    last_telemetry["form_score"] = telemetry.get('form_score', last_telemetry['form_score'])
                    last_telemetry["feedback"] = telemetry.get('feedback', last_telemetry['feedback'])
                else:
                    # Passerby or background detection ignored
                    last_telemetry["state"] = "OCCLUDED / PASSERBY"
                    last_telemetry["feedback"] = "Ignoring passerby - holding lock..."
            else:
                lock.validate_and_update(None)
                last_telemetry["state"] = "NO PERSON"
                last_telemetry["feedback"] = "Step into frame to begin"

            # ---------------------------------------------------------
            # UI OVERLAY (Persistent Telemetry)
            # ---------------------------------------------------------
            feedback_text = last_telemetry["feedback"]

            if "No rep" in feedback_text or "Danger" in feedback_text:
                feedback_color = (0, 0, 255)    # Red
            elif "Half rep" in feedback_text or "Half Rep" in feedback_text:
                feedback_color = (0, 140, 255)  # Orange
            elif any(w in feedback_text for w in ["Perfect", "Great", "Good"]):
                feedback_color = (0, 255, 0)    # Green
            elif last_telemetry["state"] in ["NO PERSON", "OCCLUDED / PASSERBY"]:
                feedback_color = (0, 165, 255)  # Amber
            else:
                feedback_color = (0, 255, 255)  # Yellow

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

            # Display frame
            cv2.imshow(window_name, frame)

            # Key controls
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                lock.reset()
                print("Target lock manually reset.")
            elif key == ord('f'):
                is_full = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                new_prop = cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_prop)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test with default tracker:
    main()

    # Or pass any converted PyTorch tracker directly:
    # from exercises.pushup import PushupTracker
    # main(tracker_instance=PushupTracker())