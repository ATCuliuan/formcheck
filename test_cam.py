import cv2
import mediapipe as mp
from exercises.dumbbell_biceps_curls import DumbbellBicepsCurlsTracker

# Initialize MediaPipe drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    tracker = DumbbellBicepsCurlsTracker()
    
    # Using CAP_DSHOW for reliable Windows camera access
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Create a resizable window
    window_name = 'Backend Engine Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("🎥 Starting webcam... Press 'q' to quit.")
    
    # Persistent telemetry state so information remains visible when no one is in frame
    last_telemetry = {
        "state": tracker.state,
        "total_reps": tracker.good_reps + tracker.bad_reps + tracker.half_reps,
        "good_reps": tracker.good_reps,
        "half_reps": tracker.half_reps,
        "bad_reps": tracker.bad_reps,
        "form_score": 0,
        "feedback": tracker.current_feedback
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
            
            if results.pose_landmarks:
                # 1. Draw the skeleton
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 2. Run tracker logic and update the persistent telemetry
                telemetry = tracker.process_frame(results.pose_landmarks.landmark)
                last_telemetry["state"] = telemetry.get('state', last_telemetry['state'])
                last_telemetry["total_reps"] = telemetry.get('total_reps', last_telemetry['total_reps'])
                last_telemetry["good_reps"] = telemetry.get('good_reps', last_telemetry['good_reps'])
                last_telemetry["half_reps"] = telemetry.get('half_reps', last_telemetry['half_reps'])
                last_telemetry["bad_reps"] = telemetry.get('bad_reps', last_telemetry['bad_reps'])
                last_telemetry["form_score"] = telemetry.get('form_score', last_telemetry['form_score'])
                last_telemetry["feedback"] = telemetry.get('feedback', last_telemetry['feedback'])
            else:
                # Keep rep counters intact, but show a prompt if no person is detected
                last_telemetry["state"] = "NO PERSON"
                last_telemetry["feedback"] = "Step into frame to begin"

            # ---------------------------------------------------------
            # UI OVERLAY (Renders on EVERY frame, even if no person is seen)
            # ---------------------------------------------------------
            half_reps = last_telemetry["half_reps"]
            feedback_text = last_telemetry["feedback"]
            
            # Dynamic color coding for feedback
            if "No rep" in feedback_text or "Danger" in feedback_text:
                feedback_color = (0, 0, 255)    # Red for violation
            elif "Half rep" in feedback_text or "Half Rep" in feedback_text:
                feedback_color = (0, 140, 255)  # Orange for half rep
            elif "Perfect" in feedback_text or "Great" in feedback_text or "Good" in feedback_text:
                feedback_color = (0, 255, 0)    # Green for good rep
            elif last_telemetry["state"] == "NO PERSON":
                feedback_color = (0, 165, 255)  # Amber warning
            else:
                feedback_color = (0, 255, 255)  # Yellow for guidance

            # Render persistent telemetry values
            cv2.putText(frame, f"State: {last_telemetry['state']}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            
            cv2.putText(frame, f"Total Reps: {last_telemetry['total_reps']}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Good: {last_telemetry['good_reps']} | Half: {half_reps} | Bad: {last_telemetry['bad_reps']}", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
            cv2.putText(frame, f"Score: {last_telemetry['form_score']}%", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        
            cv2.putText(frame, f"Feedback: {feedback_text}", (10, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, feedback_color, 2)
            
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