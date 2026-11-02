import time
import cv2
import mediapipe as mp
from exercises.barbell_row import BarbellRowTracker

# Initialize MediaPipe drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    tracker = BarbellRowTracker()
    
    # Using CAP_DSHOW for reliable Windows camera access
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Create a resizable window
    window_name = 'Backend Engine Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("🎥 Starting webcam... Press 'q' to quit.")
    
    # 5-second countdown timer (starts only when person is visible)
    countdown_duration = 5.0
    countdown_start = None
    is_counting_down = True

    # Skeleton flash / fault state
    skeleton_flash_color = None
    flash_start_time = 0.0
    flash_duration = 1.0  # Duration in seconds for completion flash

    prev_good_reps = 0
    prev_bad_reps = 0

    # Persistent telemetry state
    last_telemetry = {
        "state": "WAITING FOR PERSON",
        "total_reps": tracker.good_reps + tracker.bad_reps,
        "good_reps": tracker.good_reps,
        "bad_reps": tracker.bad_reps,
        "form_score": 0,
        "feedback": "Step into camera frame to begin"
    }
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert color for MediaPipe processing (BGR -> RGB)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            results = pose.process(rgb_image)
            
            # Keep original BGR frame writeable for natural colors on screen
            frame.flags.writeable = True
            now = time.time()

            if results.pose_landmarks:
                # 1. Start/progress the countdown only when a person is visible
                if is_counting_down:
                    if countdown_start is None:
                        countdown_start = now

                    time_elapsed = now - countdown_start
                    remaining_time = max(0.0, countdown_duration - time_elapsed)

                    if remaining_time <= 0:
                        is_counting_down = False
                        if hasattr(tracker, "frame_buffer"):
                            tracker.frame_buffer.clear()
                        prev_good_reps = tracker.good_reps
                        prev_bad_reps = tracker.bad_reps
                    else:
                        last_telemetry["state"] = "PREPARING"
                        last_telemetry["feedback"] = f"Starting in {int(remaining_time) + 1}s..."

                # 2. Run tracker logic only after countdown finishes
                if not is_counting_down:
                    telemetry = tracker.process_frame(results.pose_landmarks.landmark)
                    last_telemetry["state"] = telemetry.get('state', last_telemetry['state'])
                    last_telemetry["good_reps"] = telemetry.get('good_reps', last_telemetry['good_reps'])
                    last_telemetry["bad_reps"] = telemetry.get('bad_reps', last_telemetry['bad_reps'])
                    last_telemetry["total_reps"] = last_telemetry["good_reps"] + last_telemetry["bad_reps"]
                    last_telemetry["form_score"] = telemetry.get('form_score', last_telemetry['form_score'])
                    last_telemetry["feedback"] = telemetry.get('feedback', last_telemetry['feedback'])

                    # Trigger flash color on rep increment
                    if last_telemetry["good_reps"] > prev_good_reps:
                        skeleton_flash_color = (0, 255, 0)      # Green flash on good rep
                        flash_start_time = now
                        prev_good_reps = last_telemetry["good_reps"]
                    elif last_telemetry["bad_reps"] > prev_bad_reps:
                        skeleton_flash_color = (0, 0, 255)      # Red flash on bad rep
                        flash_start_time = now
                        prev_bad_reps = last_telemetry["bad_reps"]

                # 3. Determine active skeleton color
                # Stays solid RED whenever frozen in LOCKED_FAULT
                if last_telemetry["state"] == "LOCKED_FAULT":
                    current_line_color = (0, 0, 255)            # Red lines while stuck in fault
                    current_point_color = (0, 0, 255)           # Red joints while stuck in fault
                elif skeleton_flash_color and (now - flash_start_time < flash_duration):
                    current_line_color = skeleton_flash_color
                    current_point_color = skeleton_flash_color
                else:
                    skeleton_flash_color = None
                    current_line_color = (255, 255, 255)        # Default white lines
                    current_point_color = (0, 215, 255)         # Default gold joints

                line_spec = mp_drawing.DrawingSpec(color=current_line_color, thickness=3, circle_radius=2)
                point_spec = mp_drawing.DrawingSpec(color=current_point_color, thickness=2, circle_radius=4)

                # 4. Draw the skeleton with dynamic colors
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=point_spec,
                    connection_drawing_spec=line_spec
                )
            else:
                # If no person is visible, reset countdown start
                if is_counting_down:
                    countdown_start = None
                    last_telemetry["state"] = "WAITING FOR PERSON"
                    last_telemetry["feedback"] = "Step into camera frame to begin"
                else:
                    last_telemetry["state"] = "NO PERSON"
                    last_telemetry["feedback"] = "Step into frame to begin"

            # ---------------------------------------------------------
            # UI OVERLAY
            # ---------------------------------------------------------
            feedback_text = last_telemetry["feedback"]
            
            # Dynamic color coding for feedback
            if is_counting_down and countdown_start is not None:
                feedback_color = (0, 215, 255)
            elif "Bad rep" in feedback_text or "No rep" in feedback_text or "Danger" in feedback_text or last_telemetry["state"] == "LOCKED_FAULT":
                feedback_color = (0, 0, 255)    # Red for violation/locked fault
            elif "Perfect" in feedback_text or "Great" in feedback_text or "Good" in feedback_text or "Excellent" in feedback_text:
                feedback_color = (0, 255, 0)    # Green for good rep
            elif "NO PERSON" in last_telemetry["state"] or "WAITING" in last_telemetry["state"]:
                feedback_color = (0, 165, 255)  # Amber warning
            else:
                feedback_color = (0, 255, 255)  # Yellow for guidance

            # Render persistent telemetry values
            cv2.putText(frame, f"State: {last_telemetry['state']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            
            cv2.putText(frame, f"Total Reps: {last_telemetry['total_reps']}", (10, 70),  
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Good: {last_telemetry['good_reps']} | Bad: {last_telemetry['bad_reps']}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
            cv2.putText(frame, f"Score: {last_telemetry['form_score']}%", (10, 150),  
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        
            cv2.putText(frame, f"Feedback: {feedback_text}", (10, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
            
            # Show the original BGR frame inside the resizable window
            cv2.imshow(window_name, frame)
            
            # Press 'f' to toggle fullscreen, 'q' to quit
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                is_full = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                new_prop = cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_prop)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()