#  currently this is acting as a database substitute
from PIL import Image
import numpy as np

class AppState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.name = ""
        self.video_frames: list[Image.Image] = []
        self.video_frames_without_obj: list[Image.Image] = []
        self.inference_session = None
        self.video_fps: float | None = None
        self.video_path: str | None = None  # Store original video path for audio extraction
        self.masks_by_frame: dict[int, dict[int, np.ndarray]] = {}
        self.color_by_obj: dict[int, tuple[int, int, int]] = {}
        self.color_by_prompt: dict[str, tuple[int, int, int]] = {}
        self.clicks_by_frame_obj: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
        self.boxes_by_frame_obj: dict[int, dict[int, list[tuple[int, int, int, int]]]] = {}
        self.text_prompts_by_frame_obj: dict[int, dict[int, str]] = {}
        self.composited_frames: dict[int, Image.Image] = {}
        self.current_frame_idx: int = 0
        self.current_obj_id: int = 1
        self.current_label: str = "positive"
        self.current_clear_old: bool = True
        self.current_prompt_type: str = "Points"
        self.pending_box_start: tuple[int, int] | None = None
        self.pending_box_start_frame_idx: int | None = None
        self.pending_box_start_obj_id: int | None = None
        self.active_tab: str = "point_box"
        self.generated_sprites_by_obj: dict[int, Image.Image] = {}  # Store AI-generated sprites per object ID

    def __repr__(self):
        return f"AppState(video_frames={len(self.video_frames)}, video_fps={self.video_fps}, masks_by_frame={len(self.masks_by_frame)}, color_by_obj={len(self.color_by_obj)})"

    @property
    def num_frames(self) -> int:
        return len(self.video_frames)
    
    
app_state = AppState()