"""
Standalone camera test tool.

Use this to figure out your best camera setup (index, resolution, zoom)
BEFORE plugging it into test_cam.py. Doesn't touch MediaPipe or any
exercise tracker -- just raw OpenCV, so it's fast to iterate on.

Controls while running:
  q       - quit
  +  / -  - zoom in / out (if your camera driver supports it)
  [ / ]   - decrease / increase resolution (cycles through presets)
  0-9     - switch camera index (tries CAMERA_INDEX = pressed number)
"""

import cv2

RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
]

CAMERA_INDEX = 0


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"❌ Could not open camera index {index}")
        return None
    print(f"✅ Opened camera index {index}")
    return cap


def print_camera_info(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zoom = cap.get(cv2.CAP_PROP_ZOOM)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"   Resolution: {w}x{h} | FPS: {fps:.1f} | Zoom: {zoom}")


def main():
    global CAMERA_INDEX
    cap = open_camera(CAMERA_INDEX)
    if cap is None:
        return

    res_idx = 0
    w, h = RESOLUTIONS[res_idx]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    zoom_level = cap.get(cv2.CAP_PROP_ZOOM)
    if zoom_level is None or zoom_level < 0:
        zoom_level = 0

    print_camera_info(cap)
    print("\nControls: q=quit | +/- zoom | [ ] resolution | 0-9 switch camera index\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed. Camera may be disconnected or busy.")
            break

        h_frame, w_frame = frame.shape[:2]

        # Overlay a center crosshair + grid so you can judge framing/coverage
        cv2.line(frame, (w_frame // 2, 0), (w_frame // 2, h_frame), (0, 255, 0), 1)
        cv2.line(frame, (0, h_frame // 2), (w_frame, h_frame // 2), (0, 255, 0), 1)

        info_text = f"Cam {CAMERA_INDEX} | {w_frame}x{h_frame} | Zoom {cap.get(cv2.CAP_PROP_ZOOM):.0f}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow('Camera Test', frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('+') or key == ord('='):
            zoom_level += 10
            cap.set(cv2.CAP_PROP_ZOOM, zoom_level)
            print(f"Zoom -> {cap.get(cv2.CAP_PROP_ZOOM)}")

        elif key == ord('-') or key == ord('_'):
            zoom_level = max(0, zoom_level - 10)
            cap.set(cv2.CAP_PROP_ZOOM, zoom_level)
            print(f"Zoom -> {cap.get(cv2.CAP_PROP_ZOOM)}")

        elif key == ord('['):
            res_idx = (res_idx - 1) % len(RESOLUTIONS)
            w, h = RESOLUTIONS[res_idx]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            print_camera_info(cap)

        elif key == ord(']'):
            res_idx = (res_idx + 1) % len(RESOLUTIONS)
            w, h = RESOLUTIONS[res_idx]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            print_camera_info(cap)

        elif chr(key).isdigit():
            new_index = int(chr(key))
            if new_index != CAMERA_INDEX:
                cap.release()
                new_cap = open_camera(new_index)
                if new_cap is not None:
                    cap = new_cap
                    CAMERA_INDEX = new_index
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    print_camera_info(cap)
                else:
                    print(f"   Staying on camera {CAMERA_INDEX}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()