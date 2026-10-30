import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# ==========================================
# 1. CONFIGURATION
# ==========================================
EXERCISE = "good_morning" # Change this to test different models
MODEL_PATH = f"models/tflite/st_gcn_{EXERCISE}.tflite"
SEQUENCE_LENGTH = 30

# ==========================================
# 2. TFLITE SETUP
# ==========================================
print(f"Loading FormCheck TFLite Model: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==========================================
# 3. AI PRE-PROCESSING 
# ==========================================
def apply_joint_mask(sequence, exercise):
    masked_seq = np.copy(sequence)
    upper_body_exercises = [
        "standing_overhead_extension", "dumbbell_kickback", "floor_skull_crusher",
        "reverse_curl", "concentration_curl", "dumbbell_floor_press",
        "dumbbell_floor_flye", "pullup", "tricep_dips"
    ]
    plank_core_exercises = [
        "renegade_row", "wall_pushup", "decline_pushup", "plank_triceps_extension", 
        "good_morning" 
    ]
    
    if exercise in upper_body_exercises:
        joints_to_mask = [0, 12, 13, 14, 15, 16, 17, 18, 19]
        for j in joints_to_mask:
            if j < masked_seq.shape[1]: masked_seq[:, j, :] = 0.0
                
    elif exercise in plank_core_exercises:
        joints_to_mask = [13, 14, 15, 17, 18, 19]
        for j in joints_to_mask:
            if j < masked_seq.shape[1]: masked_seq[:, j, :] = 0.0
                
    elif exercise == "seated_knee_tucks":
        joints_to_mask = [14, 15, 18, 19]
        for j in joints_to_mask:
            if j < masked_seq.shape[1]: masked_seq[:, j, :] = 0.0
            
    return masked_seq

def normalize_skeleton(sequence):
    norm_seq = np.copy(sequence)
    for i in range(len(norm_seq)):
        frame = norm_seq[i]
        center = np.copy(frame[2]) 
        shoulder_dist = np.linalg.norm(frame[4] - frame[8]) 
        if shoulder_dist < 1e-5: shoulder_dist = 1.0 
            
        for j in range(25):
            if np.all(frame[j] == 0.0):
                norm_seq[i, j] = 0.0
            else:
                norm_seq[i, j] = (frame[j] - center) / shoulder_dist
    return norm_seq

def mediapipe_to_ntu25(landmarks):
    ntu = np.zeros((25, 3))
    def get_xyz(mp_idx):
        lm = landmarks.landmark[mp_idx]
        return [lm.x, lm.y, lm.z]
    
    ntu[3] = get_xyz(0)
    ntu[4] = get_xyz(11)
    ntu[8] = get_xyz(12)
    ntu[5] = get_xyz(13)
    ntu[9] = get_xyz(14)
    ntu[6] = get_xyz(15)
    ntu[10] = get_xyz(16)
    ntu[12] = get_xyz(23)
    ntu[16] = get_xyz(24)
    ntu[13] = get_xyz(25)
    ntu[17] = get_xyz(26)
    ntu[14] = get_xyz(27)
    ntu[18] = get_xyz(28)
    ntu[15] = get_xyz(31)
    ntu[19] = get_xyz(32)
    
    ntu[0] = (ntu[12] + ntu[16]) / 2.0
    ntu[2] = (ntu[4] + ntu[8]) / 2.0
    ntu[1] = (ntu[0] + ntu[2]) / 2.0
    return ntu

# THE MOTION GATEKEEPER
def is_moving(sequence, threshold=0.015):
    # Calculates the standard deviation of all joints across the 30 frames.
    # If the skeleton is completely still, the deviation is near 0.0.
    movement_variance = np.mean(np.std(sequence, axis=0))
    return movement_variance > threshold

# ==========================================
# 4. WEBCAM INFERENCE LOOP
# ==========================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
current_prediction = "Waiting for 30 frames..."
prediction_color = (0, 255, 255) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        ntu_frame = mediapipe_to_ntu25(results.pose_landmarks)
        frame_buffer.append(ntu_frame)
        
        if len(frame_buffer) == SEQUENCE_LENGTH:
            raw_sequence = np.array(frame_buffer)
            masked_sequence = apply_joint_mask(raw_sequence, EXERCISE)
            normalized_sequence = normalize_skeleton(masked_sequence)
            
            # Check if the user is actually moving before running the AI
            if not is_moving(normalized_sequence):
                current_prediction = "STANDING STILL - Waiting for rep..."
                prediction_color = (150, 150, 150) # Gray
            else:
                input_data = np.expand_dims(normalized_sequence, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                confidence = output_data[0]
                predicted_class = np.argmax(confidence)
                
                if predicted_class == 1:
                    current_prediction = f"GOOD FORM ({confidence[1]*100:.1f}%)"
                    prediction_color = (0, 255, 0) # Green
                else:
                    current_prediction = f"BAD FORM ({confidence[0]*100:.1f}%)"
                    prediction_color = (0, 0, 255) # Red

    cv2.putText(frame, f"Exercise: {EXERCISE.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, current_prediction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, prediction_color, 3)
    cv2.imshow('FormCheck AI Live Test', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()