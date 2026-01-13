#  currently this is acting as a database substitute
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class VideoState:
    frames: List[np.ndarray] = field(default_factory=list)
    frame_rgb: List[Image.Image] = field(default_factory=list) 
    fps: float = 30.0
    video_path: str | None = field(default=None)
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
    
    
@dataclass
class AdState:
    image: Optional[np.ndarray] = None
    height: int = 0
    width: int = 0
    corners: Optional[np.ndarray] = None  # [[0,0],[w,0],[w,h],[0,h]]


@dataclass
class PlacementState:
    x: int = 0
    y: int = 0
    scale: float = 1.0

    w: int = 0
    h: int = 0

    corner_offsets: np.ndarray | None = None
    quad: np.ndarray | None = None





@dataclass
class APPState:
    
        
    def __init__(self):
        self.video = VideoState()
        self.placement = PlacementState()
        self.ad = AdState()

        self.reset()
        
        
    video: VideoState = field(default_factory=VideoState)
    ad: AdState = field(default_factory=AdState)
    placement: PlacementState = field(default_factory=PlacementState)
    # tracking: TrackingState = field(default_factory=TrackingState)
    

    
    def reset(self):

        self.video.delete_video_file()
        self.video = VideoState()
        self.placement = PlacementState()
        # self.tracking = TrackingState()




    
app_state = APPState()