from pathlib import Path
import shutil

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


frames_dir = Path("media/frames")



# Prepare frames directory: delete if exists, then create
def prepare_frames_dir() -> Path:
    frames_dir = Path("media/frames")

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir


# Prepare output frames directory: delete if exists, then create
def prepare_output_frames_dir() -> Path:
    frames_dir = Path("media/output")

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir



# Prepare output paths: create necessary directories
def prepare_output_paths():
    frames_dir = Path("media/tmp_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path("media/output_videos")
    video_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = video_dir / "output.mp4"

    return frames_dir, output_video_path
