#  currently this is acting as a database substitute
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import os
import shutil


@dataclass
class VideoState:
    frames: List[np.ndarray] = field(default_factory=list)
    frame_rgb: List[Image.Image] = field(default_factory=list)
    masks_by_frame: List[np.ndarray] = field(default_factory=list)
    fps: float = 30.0
    video_path: str | None = field(default=None)
    frames_dir: str | None = field(default=None)
    height: int = 0
    width: int = 0
    current_frame_idx: int = 0
    processed_frames: list[Image.Image] = field(default_factory=list)
    
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def get_rgb_frame(self, idx: int) -> Image.Image:
        if 0 <= idx < len(self.frame_rgb):
            self.current_frame_idx = idx
            return self.frame_rgb[idx]
        else:
            raise IndexError("Frame index out of range")
    
    
    def delete_video_file(self):
        if self.video_path and os.path.exists(self.video_path):
            os.remove(self.video_path)
        
        # Delete frames directory if it exists
        if self.frames_dir and os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
    
    
@dataclass
class AdState:
    image: Optional[np.ndarray] = None
    height: int = 0
    width: int = 0
    corners: Optional[np.ndarray] = None  # [[0,0],[w,0],[w,h],[0,h]]
    ad_path: str | None = None
    
    
    def delete_ad_file(self):
        if self.ad_path and os.path.exists(self.ad_path):
            os.remove(self.ad_path)




@dataclass
class PlacementState:
    corner_offsets: np.ndarray | None = None
    quad: np.ndarray | None = None


@dataclass
class TrackingState:
    prev_gray: Optional[np.ndarray] = None
    prev_pts: Optional[np.ndarray] = None   # Nx1x2
    active: bool = False




@dataclass
class APPState:
        
    def __init__(self):
        self.video = VideoState()
        self.placement = PlacementState()
        self.ad = AdState()
        self.tracking = TrackingState()

        self.reset()
        
        
    video: VideoState = field(default_factory=VideoState)
    ad: AdState = field(default_factory=AdState)
    placement: PlacementState = field(default_factory=PlacementState)
    tracking: TrackingState = field(default_factory=TrackingState)
    

    
    def reset(self):

        self.video.delete_video_file()
        self.ad.delete_ad_file()
        self.video = VideoState()
        self.placement = PlacementState()
        self.tracking = TrackingState()




    
app_state = APPState()