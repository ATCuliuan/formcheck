import cv2
import mediapipe as mp
from exercises.pushup import PushupTracker

# Initialize MediaPipe drawing tools
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main():
    # Load your backend tracking logic
    tracker = PushupTracker()
    
    # Open the laptop's default webcam (0)
    cap = cv2.VideoCapture(0)
    
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
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # 1. Draw the skeleton on your body
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 2. Run your SquatTracker logic
                telemetry = tracker.process_frame(results.pose_landmarks.landmark)
                
                # 3. Display the backend data on the screen
                cv2.putText(image, f"State: {telemetry['state']}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(image, f"Reps: {telemetry['rep_count']}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(image, f"Score: {telemetry['form_score']}%", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(image, f"Feedback: {telemetry['feedback']}", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show the live window
            cv2.imshow('Backend Engine Test', image)
            
            # Press 'q' to close the window safely
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()