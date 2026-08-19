import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=10, prefix="img"):
    """
    Extracts frames from a video and saves them to a designated folder.
    
    :param video_path: Path to the input video file.
    :param output_folder: Where to save the extracted images.
    :param frame_skip: Saves every Nth frame (e.g., 10 means save 1 frame every 10 frames).
    :param prefix: Text to put at the start of every image file name.
    """
    # Make sure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print(f"🎬 Starting extraction for: {video_path}")
    print(f"📁 Saving to: {output_folder}")

    while True:
        success, frame = cap.read()
        
        # If we reach the end of the video, break the loop
        if not success:
            break
            
        # Only save every Nth frame based on frame_skip
        if frame_count % frame_skip == 0:
            # Generate a filename like: my_squat_frame_001.jpg
            filename = f"{prefix}_frame_{saved_count:03d}.jpg"
            save_path = os.path.join(output_folder, filename)
            
            # Save the image
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"✅ Done! Extracted {saved_count} images.")

if __name__ == "__main__":
    # --- CONFIGURE YOUR EXTRACTION HERE ---
    
    # 1. Path to your video file
    VIDEO_FILE = "video_exercises/Squat/positive_squat_8.mp4" 
    
    # 2. Where to put the images (Cleaned up path)
    OUTPUT_DIR = "dataset/test/squat/positive"
    
    # 3. Save every 15th frame 
    SKIP = 15 
    
    # 4. A prefix so files don't overwrite if you do this for multiple videos
    FILE_PREFIX = "positive_squat_8" 

    extract_frames(VIDEO_FILE, OUTPUT_DIR, frame_skip=SKIP, prefix=FILE_PREFIX)