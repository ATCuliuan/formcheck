import cv2
import mediapipe as mp


def open_max_quality_camera(cam_index=0):
    """
    Opens the webcam using DirectShow on Windows and requests MJPG + Full HD (1920x1080)
    so the feed runs at the camera's true native quality and frame rate.
    """
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Camera initialized at native quality: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")
    return cap


def main():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Thicker, high-contrast joint keypoints and skeleton lines
    joint_spec = mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=4, circle_radius=7)
    bone_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=5, circle_radius=4)

    cap = open_max_quality_camera(0)
    if not cap.isOpened():
        print("❌ Error: Could not access webcam.")
        return

    window_name = "Keypoints Only - Native Camera Quality (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=joint_spec,
                    connection_drawing_spec=bone_spec,
                )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()