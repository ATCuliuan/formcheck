import cv2
import numpy as np
import base64
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from exercises.squat import SquatTracker
from exercises.pushup import PushupTracker
from exercises.lunge import LungeTracker
from exercises.plank import PlankTracker

app = FastAPI(title="FORMCHECK Backend API")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,          # Complexity 1 is fastest for real-time mobile streams
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Active exercise trackers map
trackers = {
    "squat": SquatTracker(),
    "pushup": PushupTracker(),
    "lunge": LungeTracker(),
    "plank": PlankTracker()
}

@app.websocket("/ws/track/{exercise_name}")
async def websocket_endpoint(websocket: WebSocket, exercise_name: str):
    await websocket.accept()
    tracker = trackers.get(exercise_name.lower(), SquatTracker())
    
    print(f"Client connected for tracking: {exercise_name}")

    try:
        while True:
            # 1. Receive base64 frame from React Native
            data = await websocket.receive_text()
            
            # Decode base64 image
            img_bytes = base64.b64decode(data.split(",")[-1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 2. Process landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if not results.pose_landmarks or len(results.pose_landmarks.landmark) != 33:
                await websocket.send_json({
                    "status": "warning",
                    "feedback": "Move so your full body is visible"
                })
                continue

            # 3. Feed to kinematic state tracker
            telemetry = tracker.process_frame(results.pose_landmarks.landmark)
            telemetry["status"] = "tracking"

            # 4. Send telemetry back to React Native instantly
            await websocket.send_json(telemetry)

    except WebSocketDisconnect:
        print("Client disconnected.")