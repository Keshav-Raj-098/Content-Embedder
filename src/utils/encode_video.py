import subprocess
from pathlib import Path
import shutil


def encode_video_ffmpeg(frames_dir: Path, output_path: Path, fps: float):
    
    # print("FFmpeg found at:", shutil.which("ffmpeg"))
    
    
    if fps is None or fps <= 1:
        fps = 30.0
    FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",      # 🔑 browser requirement
        "-profile:v", "baseline",
        "-movflags", "+faststart",  # 🔑 streaming friendly
        str(output_path),
    ]

    subprocess.run(cmd, check=True)
