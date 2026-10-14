import time
import cv2
import mediapipe as mp
from exercises.standing_shoulder_scaption   import StandingShoulderScaptionTracker

# Initialize MediaPipe drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    tracker = StandingShoulderScaptionTracker()
    
    # Using CAP_DSHOW for reliable Windows camera access
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Create a resizable window
    window_name = 'Backend Engine Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("🎥 Starting webcam... Press 'q' to quit.")
    
    # Persistent telemetry state
    last_telemetry = {
        "state": tracker.state,
        "total_reps": tracker.good_reps + tracker.bad_reps,
        "good_reps": tracker.good_reps,
        "bad_reps": tracker.bad_reps,
        "feedback": tracker.current_feedback
    }
    
    # 5-second countdown controls
    countdown_duration = 5.0
    countdown_start_time = None
    session_active = False
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert color for MediaPipe processing (BGR -> RGB)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            results = pose.process(rgb_image)
            
            # Keep original BGR frame writeable for rendering
            frame.flags.writeable = True
            
            if results.pose_landmarks:
                # 1. Draw the skeleton
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 2. 5-Second Countdown Logic before session starts
                if not session_active:
                    if countdown_start_time is None:
                        countdown_start_time = time.time()
                    
                    elapsed = time.time() - countdown_start_time
                    remaining = int(countdown_duration - elapsed) + 1
                    
                    if remaining > 0:
                        last_telemetry["state"] = "GET READY"
                        last_telemetry["feedback"] = f"Get into starting position! Starting in {remaining}s..."
                        
                        # Big prominent countdown on screen
                        h, w, _ = frame.shape
                        cv2.putText(frame, str(remaining), (w // 2 - 30, h // 2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 215, 255), 5)
                    else:
                        session_active = True
                        last_telemetry["state"] = "ACTIVE"
                        last_telemetry["feedback"] = "Session started! Begin movement."
                else:
                    # 3. Session Active: Process frame telemetry
                    telemetry = tracker.process_frame(results.pose_landmarks.landmark)
                    last_telemetry["state"] = telemetry.get('state', last_telemetry['state'])
                    last_telemetry["total_reps"] = tracker.good_reps + tracker.bad_reps
                    last_telemetry["good_reps"] = tracker.good_reps
                    last_telemetry["bad_reps"] = tracker.bad_reps
                    last_telemetry["feedback"] = telemetry.get('feedback', last_telemetry['feedback'])
            else:
                # Reset countdown if user steps completely out of frame before it starts
                if not session_active:
                    countdown_start_time = None
                last_telemetry["state"] = "NO PERSON"
                last_telemetry["feedback"] = "Step into frame to begin"

            # ---------------------------------------------------------
            # UI OVERLAY
            # ---------------------------------------------------------
            feedback_text = last_telemetry["feedback"]
            
            # Dynamic color coding for feedback
            if "No rep" in feedback_text or "Danger" in feedback_text:
                feedback_color = (0, 0, 255)    # Red for violation
            elif "Perfect" in feedback_text or "Great" in feedback_text or "Good" in feedback_text:
                feedback_color = (0, 255, 0)    # Green for good rep
            elif last_telemetry["state"] in ["NO PERSON", "GET READY"]:
                feedback_color = (0, 165, 255)  # Amber warning
            else:
                feedback_color = (0, 255, 255)  # Yellow for guidance

            # Render UI
            cv2.putText(frame, f"State: {last_telemetry['state']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Total Reps: {last_telemetry['total_reps']}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Good: {last_telemetry['good_reps']} | Bad: {last_telemetry['bad_reps']}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
            cv2.putText(frame, f"Feedback: {feedback_text}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, feedback_color, 2)
            
            cv2.imshow(window_name, frame)
            
            # Key bindings
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