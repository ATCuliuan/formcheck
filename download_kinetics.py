import pandas as pd
import os
import sys
import subprocess
import imageio_ffmpeg

CSV_FILE = "train.csv"
TARGET_VIDEOS = 20
EXERCISE_LABEL = "situp"

def download_kinetics_class():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Could not find {CSV_FILE}. Please download it to this folder.")
        return

    # Automatically locates the portable ffmpeg binary
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    print("Loading Kinetics-400 database...")
    df = pd.read_csv(CSV_FILE)
    
    # Filter the database for our specific exercise
    exercise_df = df[df['label'] == EXERCISE_LABEL]
    print(f"Found {len(exercise_df)} total videos for {EXERCISE_LABEL}.")
    
    os.makedirs("raw_videos", exist_ok=True)
    success_count = 0

    for index, row in exercise_df.iterrows():
        if success_count >= TARGET_VIDEOS:
            break
            
        vid_id = row['youtube_id']
        start = int(row['time_start'])
        end = int(row['time_end'])
        
        out_file = f"raw_videos/sit_up_{vid_id}.mp4"
        
        if os.path.exists(out_file):
            print(f"Skipping {out_file} (Already exists)")
            success_count += 1
            continue

        url = f"https://www.youtube.com/watch?v={vid_id}"
        print(f"\n[{success_count + 1}/{TARGET_VIDEOS}] Attempting to download {url} (Seconds {start} to {end})...")
        
        # FIXED: yt_dlp with an underscore, and a forgiving video-first format string
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-f", "bv*[ext=mp4]/b[ext=mp4]/best",
            "--download-sections", f"*{start}-{end}",
            "--ffmpeg-location", ffmpeg_path,
            "-o", out_file,
            url
        ]
        
        # Run as a list to prevent Windows command line string errors
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if result.returncode == 0:
            print(f"✅ Successfully extracted clip!")
            success_count += 1
        else:
            print(f"⚠️ Failed to download (could be deleted, private, or age-restricted). Moving to next...")

    print(f"\n🎉 Finished! {success_count} {EXERCISE_LABEL} clips downloaded to raw_videos/")

if __name__ == "__main__":
    download_kinetics_class()