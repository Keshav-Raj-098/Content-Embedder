import shutil
from pathlib import Path
from typing import Optional,List



import cv2
import numpy as np
from PIL import Image
import os

from src.state import app_state as state,APPState
from src.config.log_config import get_logger
from src.controllers.exceptions import VideoLoadError,AdLoadError



from helpers.editor_helper import normalize_quad 
from helpers.sam2_helper import SAM3TrackerService

logger = get_logger("EDITOR")

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


frames_dir = Path("media/frames")

# Global service instance (lazy loaded)
_sam3_service: Optional[SAM3TrackerService] = None

def prepare_frames_dir() -> Path:
    frames_dir = Path("media/frames")

    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir






def load_video(video_path: str) -> dict:
    """
    Load a video from disk, extract frames, and update application state.

    Returns:
        dict with metadata about the loaded video

    Raises:
        VideoLoadError: if the video cannot be loaded
    """

    if not video_path:
        raise VideoLoadError("No video provided")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoLoadError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    #frames_dir = prepare_frames_dir()   ## disable saving frames to disk for now
    frame_idx = 0
    frames_rgb = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = Image.fromarray(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        frames_rgb.append(frame_rgb)
        # frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg" ## disable saving frames to disk for now
        # frame_rgb.save(frame_path, format="JPEG", quality=95)  ## disable saving frames to disk for now
        
        frame_idx += 1

    cap.release()

    if not frames_rgb:
        raise VideoLoadError("No frames read from video")

    # ---- STATE MUTATION (explicit & intentional) ----
    state = APPState()
    state.reset()
    state.video.frame_rgb = frames_rgb
    state.video.fps = fps
    state.video.current_frame_idx = 0
    state.video.video_path = video_path
    # -------------------------------------------------

    first_frame = frames_rgb[0]
    height, width = first_frame.size[1], first_frame.size[0]
    
    state.video.height = height
    state.video.width = width

    return {
        "total_frames": len(frames_rgb),
        "frame_url": f"/media/frames/frame_000000.jpg",
        "fps": fps,
        "resolution": f"{width}x{height}",
        "duration_sec": len(frames_rgb) / fps,
    }


def load_ad_file(ad_path: str) -> dict:
    """
    Load an ad image from disk and update application state.

    Returns:
        dict with metadata about the loaded ad
    """

    if not ad_path:
        raise AdLoadError("No ad image path provided")

    if not os.path.exists(ad_path):
        raise AdLoadError("Ad image file does not exist")

    ext = os.path.splitext(ad_path)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise AdLoadError(f"Unsupported ad image type: {ext}")

    try:
        image = Image.open(ad_path)
        image = image.convert("RGBA")  # important for compositing
    except Exception as e:
        raise AdLoadError("Failed to load ad image") from e

    width, height = image.size

    # ---- update state ----
    

    # optional: clear previous ad
    # if hasattr(state, "ad") and state.ad is not None:
    #     try:
    #         state.ad.delete_file()
    #     except Exception:
    #         pass

    state.ad = {
        "image": image,
        "path": ad_path,
        "width": width,
        "height": height,
    }

    return {
        "width": width,
        "height": height,
        "mode": image.mode,
    }


def set_placement(quad):
    """
    Set the placement quadrilateral for the ad in the application state. 
    """
    logger.info(f"Received placement quad: {quad}")
    quad = np.asarray(quad, dtype=np.float32)

    if quad.shape != (4, 2):
        raise ValueError("quad must be shape (4, 2)")

    quad = normalize_quad(quad)
    
    logger.info(f"Setting placement quad: {quad.tolist()}")

    state.placement.quad = quad
    
    
def get_placement():
    """
    Get the placement quadrilateral for the ad in the application state. 
    """

    quad = state.placement.quad 
    return quad
    
    
def get_frame_url(frame_idx: int) -> str:
    state.video.current_frame_idx = frame_idx
    return f"/media/frames/frame_{frame_idx:06d}.jpg"



def get_sam3_service() -> SAM3TrackerService:
    """Get or create SAM3 service singleton."""
    global _sam3_service
    if _sam3_service is None:
        logger.info("Initializing SAM3 tracking service...")
        _sam3_service = SAM3TrackerService()
    return _sam3_service



def track_placement_quad(output_dir: str = "media/tracking") -> dict:
    """
    Track the placement quad across all video frames.
    
    Uses the quad from state.placement.quad and tracks it through
    all frames in state.video.frame_rgb.
    
    Args:
        output_dir: Directory to save tracking results
        
    Returns:
        dict with tracking results
        
    Raises:
        ValueError: if video or placement quad not set
    """
    # Validate state
    if not state.video.frame_rgb or len(state.video.frame_rgb) == 0:
        raise VideoLoadError("No video frames loaded. Load a video first.")
    
    if state.placement.quad is None:
        raise ValueError("No placement quad set. Set placement first.")
    
    logger.info(f"Starting quad tracking for {len(state.video.frame_rgb)} frames")
    
    # Get SAM3 service
    sam3 = get_sam3_service()
    
    # Track quad
    results = sam3.track_quad_in_video(
        frames=state.video.frame_rgb,
        initial_quad=state.placement.quad,
        output_dir=Path(output_dir),
        obj_id=0
    )
    
    logger.info(f"Tracking complete: {results['total_frames']} frames processed")
    
    return results


def track_custom_box(
    box: List[float],  # [x_min, y_min, x_max, y_max]
    output_dir: str = "media/tracking"
) -> dict:
    """
    Track a custom bounding box across video frames.
    
    Args:
        box: Bounding box [x_min, y_min, x_max, y_max]
        output_dir: Directory to save results
        
    Returns:
        dict with tracking results
    """
    if not state.video.frame_rgb or len(state.video.frame_rgb) == 0:
        raise VideoLoadError("No video frames loaded")
    
    logger.info(f"Tracking custom box: {box}")
    
    sam3 = get_sam3_service()
    
    results = sam3.track_box_in_video(
        frames=state.video.frame_rgb,
        initial_box=box,
        output_dir=Path(output_dir),
        obj_id=0
    )
    
    return results


def load_tracking_mask(frame_idx: int, masks_dir: str) -> np.ndarray:
    """
    Load a specific tracking mask from disk.
    
    Args:
        frame_idx: Frame index
        masks_dir: Directory containing masks
        
    Returns:
        Mask as numpy array
    """
    mask_path = Path(masks_dir) / f"mask_{frame_idx:06d}.npy"
    
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    return SAM3TrackerService.load_mask(str(mask_path))


def visualize_tracking_result(
    frame_idx: int,
    masks_dir: str,
    color: tuple = (0, 255, 0),
    alpha: float = 0.5
) -> Image.Image:
    """
    Visualize tracking result on a specific frame.
    
    Args:
        frame_idx: Frame to visualize
        masks_dir: Directory with tracking masks
        color: RGB color for mask overlay
        alpha: Transparency
        
    Returns:
        PIL Image with mask overlay
    """
    if frame_idx < 0 or frame_idx >= len(state.video.frame_rgb):
        raise IndexError(f"Frame index {frame_idx} out of range")
    
    # Get frame
    frame = state.video.frame_rgb[frame_idx]
    
    # Load mask
    mask = load_tracking_mask(frame_idx, masks_dir)
    
    # Visualize
    return SAM3TrackerService.visualize_mask_on_frame(
        frame=frame,
        mask=mask,
        color=color,
        alpha=alpha
    )


def get_all_tracking_masks(masks_dir: str) -> List[np.ndarray]:
    """
    Load all tracking masks from directory.
    
    Args:
        masks_dir: Directory containing masks
        
    Returns:
        List of mask arrays
    """
    masks_path = Path(masks_dir)
    
    if not masks_path.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
    
    mask_files = sorted(masks_path.glob("mask_*.npy"))
    
    logger.info(f"Loading {len(mask_files)} masks from {masks_dir}")
    
    masks = []
    for mask_file in mask_files:
        mask = np.load(mask_file)
        masks.append(mask)
    
    return masks



