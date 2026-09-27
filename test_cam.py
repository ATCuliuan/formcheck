import cv2
import mediapipe as mp
from exercises.standing_ab_twists import StandingAbTwistTracker

# Initialize MediaPipe drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    tracker = StandingAbTwistTracker()
    
    # Using CAP_DSHOW for reliable Windows camera access
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # 1. Create a resizable window
    window_name = 'Backend Engine Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Optional: Uncomment the line below if you want it to launch in borderless fullscreen immediately:
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("🎥 Starting webcam... Press 'q' to quit.")
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert color for MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            # Convert back to BGR for OpenCV display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # 1. Draw the skeleton
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 2. Run tracker logic
                telemetry = tracker.process_frame(results.pose_landmarks.landmark)
                
                # 3. Display metrics
                cv2.putText(image, f"State: {telemetry['state']}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.putText(image, f"Total Reps: {telemetry['total_reps']}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(image, f"Good: {telemetry['good_reps']} | Bad: {telemetry['bad_reps']}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                cv2.putText(image, f"Score: {telemetry['form_score']}%", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            
                cv2.putText(image, f"Feedback: {telemetry['feedback']}", (10, 190), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show the frame inside the resizable window
            cv2.imshow(window_name, image)
            
            # Press 'f' to toggle fullscreen, 'q' to quit
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                # Toggle true full screen mode with keyboard
                is_full = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                new_prop = cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, new_prop)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()