import cv2
import mediapipe as mp
from person_lock import ExerciseSubjectLock

# Plug in your exercise tracker here (DeadliftTracker, LateralRaiseTracker, etc.)
# from deadlift import DeadliftTracker

def run_camera(tracker=None, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose = mp_pose.Pose(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    )

    lock = ExerciseSubjectLock(max_jump_distance=0.18, max_lost_frames=45)

    print("Camera running. Press 'q' to exit. Press 'r' to manually reset the lock.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Run MediaPipe directly on full image (no cropping overhead)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            valid_person, landmarks = lock.validate_and_update(results.pose_landmarks.landmark)

            if valid_person:
                # Draw skeleton for the verified target
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                # Visual anchor on tracked hip center
                hip_px = int(lock.locked_hip[0] * w)
                hip_py = int(lock.locked_hip[1] * h)
                cv2.circle(frame, (hip_px, hip_py), 8, (0, 255, 0), -1)

                cv2.putText(frame, "TARGET: LOCKED", (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Forward clean landmarks to exercise tracker
                if tracker is not None:
                    data = tracker.process_frame(landmarks)
                    reps_text = f"Good: {data.get('good_reps', 0)} | Half: {data.get('half_reps', 0)} | Bad: {data.get('bad_reps', 0)}"
                    feedback_text = data.get("feedback", "")
                    
                    cv2.putText(frame, reps_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
                    cv2.putText(frame, feedback_text, (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
            else:
                # A secondary person or passerby was detected and ignored
                cv2.putText(frame, "TARGET OCCLUDED / IGNORING PASSERBY", (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            lock.validate_and_update(None)
            cv2.putText(frame, "SEARCHING FOR SUBJECT...", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

        cv2.imshow("Centroid Lock Fitness Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            lock.reset()
            print("Lock reset manually.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test standalone with skeleton visualization
    run_camera(tracker=None)
    
    # Or pass an active tracker instance:
    # tracker = DeadliftTracker()
    # run_camera(tracker=tracker)