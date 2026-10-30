import os
import sys
import subprocess
import imageio_ffmpeg

BAD_FORM_DATASET = {
    # --- FIT3D CORE EXERCISES ONLY ---
    "pushup": [
        {"url": "https://youtube.com/shorts/yQEx9OC2C3E?si=ocO5ugK5xJkiN1mt", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/1AugTfDd3Yk?si=TSjqV2NSNcKa-B3t", "start": 0, "end": 2},
        {"url": "https://youtu.be/E7Pd5wrC4_c?si=xRuH-No4bbpoVjI2", "start": 41, "end": 46},
        {"url": "https://youtu.be/E7Pd5wrC4_c?si=xRuH-No4bbpoVjI2", "start": 47, "end": 51},
        {"url": "https://youtu.be/E7Pd5wrC4_c?si=xRuH-No4bbpoVjI2", "start": 54, "end": 59},
        {"url": "https://youtube.com/shorts/Bei1RbZdFgI?si=xQGuQ45c4O0GtMPR", "start": 0, "end": 7},
        {"url": "https://youtube.com/shorts/UIcct-7b6oE?si=48r8LFTP3Mk-VtUk", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/udQBmkvYOmo?si=AXzP6-StlfD74XeA", "start": 0, "end": 2}
    ],
    "burpees": [
        {"url": "https://youtube.com/shorts/S6lclRRTjyc?si=V4R7RayWhqlpHgeX", "start": 1, "end": 7},
        {"url": "https://youtube.com/shorts/kS7xbNw4IBU?si=zDmMGUcJxidGo2v8", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/24AwdIlc7WA?si=rKRTr041mneRewMz", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/zlYA1SENYG4?si=FiX08i_KlEK_PWcz", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/oUOeCtL0lg4?si=a4Vso_66fdPETjQK", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/oUOeCtL0lg4?si=a4Vso_66fdPETjQK", "start": 5, "end": 10},
        {"url": "https://youtube.com/shorts/52e6CxY7siw?si=FLPYEl3_j-L0j9mC", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/aiWQ96gU6Vg?si=7P1LEpEjNqG5ABsE", "start": 15, "end": 20},
        {"url": "https://youtube.com/shorts/pHp_Z9jLsUM?si=aZJuK4F_jY6uRMde", "start": 0, "end": 3},
        {"url": "https://youtu.be/W8x9Zt5A2-0?si=8lYR3_YSqXDTbarf", "start": 33, "end": 64}
    ],
    "mule_kick": [
        {"url": "https://youtu.be/Y7947iQH4cc?si=zgv8NdyKj7lu7CCs", "start": 0, "end": 5},
        {"url": "https://youtu.be/H8T3PHYUUSo?si=qkO3wI_5WDZZsvtt", "start": 0, "end": 19},
        {"url": "https://youtube.com/shorts/mvAVPx28mn4?si=19E0Jz6rT7A-0JqM", "start": 3, "end": 21},
        {"url": "https://youtu.be/k9nOum4LwtA?si=RgBlh2Yajh51m6Jr", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/_83ens2hJCA?si=mqPj3NWTbyOsnE1A", "start": 4, "end": 9}
    ],
    "dumbbell_reverse_lunge": [
        {"url": "https://youtube.com/shorts/-kCI9EuTpZs?si=TOzbBYm2Hrht1aFX", "start": 1, "end": 23},
        {"url": "https://youtube.com/shorts/7g9jihaQT7M?si=9a9ygjKiK-dooK86", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/cPCLYMxjqwA?si=v61HtsS0BwVTqQl_", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/YmQf4vkjyKw?si=vXQZhOzu2gb2K2R4", "start": 4, "end": 10},
        {"url": "https://youtube.com/shorts/YmQf4vkjyKw?si=vXQZhOzu2gb2K2R4", "start": 16, "end": 18},
        {"url": "https://youtube.com/shorts/YmQf4vkjyKw?si=vXQZhOzu2gb2K2R4", "start": 23, "end": 26}
    ],
    "dumbbell_overhead_shoulder_press": [
        {"url": "https://youtube.com/shorts/s6hDp6prx4Y?si=dbmBiPAqJbOri_e3", "start": 0, "end": 10},
        {"url": "https://youtube.com/shorts/daoVqvBm7I0?si=TfoSwJnl6SWKtyxr", "start": 0, "end": 4}
    ],
    "dumbbell_hammer_curls": [
        {"url": "https://youtu.be/rbOhYRsd9Ak?si=hqAKiiQDBMJbw-kA", "start": 109, "end": 122},
        {"url": "https://youtu.be/rbOhYRsd9Ak?si=hqAKiiQDBMJbw-kA", "start": 2, "end": 7}
    ],
    "band_pull_apart": [
        {"url": "https://youtube.com/shorts/SLFDrgKarUs?si=xibBPlejx6h74nrV", "start": 4, "end": 16},
        {"url": "https://youtube.com/shorts/QHy6QzOocbM?si=QMQ8xXywhRnWRAiJ", "start": 0, "end": 3},
        {"url": "https://youtu.be/mt73tb2npJM?si=p6QLUNmhA4eWsZD-", "start": 43, "end": 50},
        {"url": "https://youtube.com/shorts/f3_6vLXKL1I?si=oNbpJWjVsJQK31j3", "start": 19, "end": 26},
        {"url": "https://youtube.com/shorts/Nwm80m3TjbE?si=8Ls42kqYNPpRHFMU", "start": 7, "end": 9},
        {"url": "https://youtube.com/shorts/6eictYvPulw?si=8w5w5htTkS7XixMR", "start": 0, "end": 13},
        {"url": "https://youtube.com/shorts/KG_SamTu_UY?si=4SA9q8xt8R6FOJ9I", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/9PEryeipUu4?si=nUfc5MP2fpgLCCCu", "start": 0, "end": 13},
        {"url": "https://youtube.com/shorts/KMOLpSSqCL8?si=s7fLfoS8A14mD_QS", "start": 0, "end": 4}
    ],
    "barbell_row": [
        {"url": "https://youtu.be/qXrTDQG1oUQ?si=wBpOYSwtmFk155db", "start": 7, "end": 18},
        {"url": "https://youtu.be/4xq3BsKTzYo?si=PowzQRkc9UJHBijH", "start": 43, "end": 47}
    ],
    "deadlift": [
        #all manual screen record clips
    ],
    "dumbbell_biceps_curls": [
        {"url": "https://youtube.com/shorts/2jpteC44QKg?si=SgofrLdWC9lla_Wm", "start": 0, "end": 8},
        {"url": "https://youtube.com/shorts/803JIAWBj_c?si=7K1JcqsVHM0_PJwv", "start": 3, "end": 14},
        {"url": "https://youtube.com/shorts/MCC0Wj9RErI?si=3KREoLtzXVXWeFWS", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/2dfDQGXvSYo?si=556wgAS7vyuLfC4W", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/iui51E31sX8?si=_6IVDuKkef9We8PF", "start": 7, "end": 9}
    ],
    "overhead_trap_raises": [
        {"url": "https://youtube.com/shorts/JadOGd042P0?si=L_IXT2YibTLVoV-O", "start": 0, "end": 26},
        {"url": "https://youtube.com/shorts/vrR2nIKOH2s?si=84wEbkNbUkdr3h5Y", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/KbK6ByIg7bw?si=eDj0tQeke0bT7_yd", "start": 2, "end": 7}
    ],
    "side_lateral_raise": [
        {"url": "https://youtube.com/shorts/8EROITjumZM?si=LrdVrIG9aGNfzHI8", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/G-piLwLu0d4?si=X8N4weZpvrqSZc92", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/R-bxVwJEhFE?si=LR9FcYpE4XLzJWLm", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/rXz1mOrbQm8?si=CuwvGHzBCioRY9-L", "start": 1, "end": 9}
    ]
}

def download_bad_videos():
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Save to a completely separate folder to isolate bad form data
    output_dir = "raw_bad_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    for exercise_name, videos in BAD_FORM_DATASET.items():
        if not videos:
            continue
            
        print(f"\nDownloading {len(videos)} BAD FORM clips for {exercise_name.upper()}...")
        
        for i, video in enumerate(videos):
            url = video.get("url")
            start = video.get("start")
            end = video.get("end")
            
            if not url or start is None or end is None:
                print(f"⚠️ Skipping invalid entry in {exercise_name}: {video}")
                continue
            
            # Explicitly label the output file as bad form
            out_file = os.path.join(output_dir, f"{exercise_name}_bad_{i+1}.mp4")
            
            if os.path.exists(out_file):
                print(f"Skipping {out_file} (Already exists)")
                continue
                
            print(f"[{i+1}/{len(videos)}] Downloading {url} (Sec {start} to {end})...")
            
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "-f", "bv*[ext=mp4]/b[ext=mp4]/best",
                "--download-sections", f"*{start}-{end}",
                "--ffmpeg-location", ffmpeg_path,
                "-o", out_file,
                url
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                print(f"✅ Extracted!")
            else:
                print(f"⚠️ Failed to download clip. Video might be private or link is dead.")
    
if __name__ == "__main__":
    download_bad_videos()