import torch
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from typing import List, Tuple, Optional
import os

from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
from src.config.log_config import get_logger

logger = get_logger("SAM3_TRACKER")


class SAM3TrackerService:
    """
    SAM3 Video Tracker Service for quad tracking and mask generation.
    Stateless service that works with external state management.
    """
    
    def __init__(
        self, 
        model_repo_id: str = "facebook/sam3",
        device: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        """
        Initialize SAM3 tracker model.
        
        Args:
            model_repo_id: HuggingFace model repository ID
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            hf_token: HuggingFace token for model access
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        logger.info(f"Initializing SAM3 on device: {self.device}")
        
        # Load model and processor
        self.model = Sam3TrackerVideoModel.from_pretrained(
            model_repo_id, 
            torch_dtype=self.dtype, 
            device_map=self.device,
            token=self.hf_token
        ).eval()
        
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(
            model_repo_id,
            token=self.hf_token
        )
        
        logger.info("SAM3 model loaded successfully")
    
    
    def track_quad_in_video(
        self,
        frames: List[Image.Image],
        initial_quad: np.ndarray,
        output_dir: Path,
        obj_id: int = 0
    ) -> dict:
        """
        Track a quadrilateral region across video frames and save masks.
        
        Args:
            frames: List of PIL Image frames (RGB)
            initial_quad: Initial quad corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            output_dir: Directory to save mask files
            obj_id: Object ID for tracking (default: 0)
            
        Returns:
            dict with tracking results and metadata
        """
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        if initial_quad.shape != (4, 2):
            raise ValueError("initial_quad must have shape (4, 2)")
        
        # Prepare output directory
        masks_dir = self._prepare_masks_dir(output_dir)
        
        logger.info(f"Tracking quad across {len(frames)} frames")
        
        # Convert frames to numpy arrays for SAM3
        raw_video = [np.array(frame) for frame in frames]
        
        # Initialize video session
        inference_session = self.processor.init_video_session(
            video=raw_video,
            inference_device=self.device,
            video_storage_device="cpu",  # Store frames on CPU to save GPU memory
            processing_device="cpu",
            inference_state_device=self.device,
            dtype=self.dtype,
        )
        
        # Convert quad to bounding box for initial prompt
        # SAM3 needs box format: [x_min, y_min, x_max, y_max]
        x_coords = initial_quad[:, 0]
        y_coords = initial_quad[:, 1]
        initial_box = [
            float(x_coords.min()),
            float(y_coords.min()),
            float(x_coords.max()),
            float(y_coords.max())
        ]
        
        logger.info(f"Adding box prompt: {initial_box} for object {obj_id}")
        
        # Add box prompt on first frame
        inference_session = self.processor.add_box_prompt(
            inference_session=inference_session,
            frame_idx=0,
            obj_id=obj_id,
            box=initial_box
        )
        
        # Track through video
        mask_paths = []
        tracked_quads = []
        
        with torch.no_grad():
            for sam_output in self.model.propagate_in_video_iterator(
                inference_session=inference_session
            ):
                frame_idx = sam_output.frame_idx
                
                # Post-process masks
                video_res_masks = self.processor.post_process_masks(
                    [sam_output.pred_masks],
                    original_sizes=[[
                        inference_session.video_height,
                        inference_session.video_width
                    ]]
                )[0]
                
                # Get mask for our object
                obj_idx = inference_session.obj_ids.index(obj_id)
                mask_2d = video_res_masks[obj_idx].cpu().numpy()
                
                # Save mask
                mask_path = masks_dir / f"mask_{frame_idx:06d}.npy"
                np.save(mask_path, mask_2d)
                mask_paths.append(str(mask_path))
                
                # Extract quad from mask (optional, for visualization)
                quad = self._extract_quad_from_mask(mask_2d)
                tracked_quads.append(quad)
                
                if frame_idx % 30 == 0:
                    logger.info(f"Processed frame {frame_idx}/{len(frames)}")
        
        logger.info(f"Tracking complete. Saved {len(mask_paths)} masks to {masks_dir}")
        
        return {
            "mask_paths": mask_paths,
            "tracked_quads": tracked_quads,
            "masks_dir": str(masks_dir),
            "total_frames": len(mask_paths),
            "obj_id": obj_id
        }
    
    
    def track_box_in_video(
        self,
        frames: List[Image.Image],
        initial_box: List[float],  # [x_min, y_min, x_max, y_max]
        output_dir: Path,
        obj_id: int = 0
    ) -> dict:
        """
        Track a bounding box region across video frames.
        
        Args:
            frames: List of PIL Image frames (RGB)
            initial_box: Initial box [x_min, y_min, x_max, y_max]
            output_dir: Directory to save mask files
            obj_id: Object ID for tracking
            
        Returns:
            dict with tracking results
        """
        if len(frames) == 0:
            raise ValueError("No frames provided")
        
        masks_dir = self._prepare_masks_dir(output_dir)
        logger.info(f"Tracking box across {len(frames)} frames")
        
        raw_video = [np.array(frame) for frame in frames]
        
        # Initialize session
        inference_session = self.processor.init_video_session(
            video=raw_video,
            inference_device=self.device,
            video_storage_device="cpu",
            processing_device="cpu",
            inference_state_device=self.device,
            dtype=self.dtype,
        )
        
        # Add box prompt
        inference_session = self.processor.add_box_prompt(
            inference_session=inference_session,
            frame_idx=0,
            obj_id=obj_id,
            box=initial_box
        )
        
        # Track and save
        mask_paths = []
        
        with torch.no_grad():
            for sam_output in self.model.propagate_in_video_iterator(
                inference_session=inference_session
            ):
                frame_idx = sam_output.frame_idx
                
                video_res_masks = self.processor.post_process_masks(
                    [sam_output.pred_masks],
                    original_sizes=[[
                        inference_session.video_height,
                        inference_session.video_width
                    ]]
                )[0]
                
                obj_idx = inference_session.obj_ids.index(obj_id)
                mask_2d = video_res_masks[obj_idx].cpu().numpy()
                
                mask_path = masks_dir / f"mask_{frame_idx:06d}.npy"
                np.save(mask_path, mask_2d)
                mask_paths.append(str(mask_path))
                
                if frame_idx % 30 == 0:
                    logger.info(f"Processed frame {frame_idx}/{len(frames)}")
        
        logger.info(f"Box tracking complete. Saved {len(mask_paths)} masks")
        
        return {
            "mask_paths": mask_paths,
            "masks_dir": str(masks_dir),
            "total_frames": len(mask_paths),
            "obj_id": obj_id
        }
    
    
    def _prepare_masks_dir(self, output_dir: Path) -> Path:
        """Prepare clean directory for mask storage."""
        masks_dir = Path(output_dir) / "masks"
        
        if masks_dir.exists():
            shutil.rmtree(masks_dir)
        
        masks_dir.mkdir(parents=True, exist_ok=True)
        return masks_dir
    
    
    def _extract_quad_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract approximate quad corners from binary mask.
        Returns corners in order: top-left, top-right, bottom-right, bottom-left.
        """
        # Find mask boundaries
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Empty mask, return zero quad
            return np.zeros((4, 2), dtype=np.float32)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Create quad from bounding box
        quad = np.array([
            [x_min, y_min],  # top-left
            [x_max, y_min],  # top-right
            [x_max, y_max],  # bottom-right
            [x_min, y_max],  # bottom-left
        ], dtype=np.float32)
        
        return quad
    
    
    @staticmethod
    def load_mask(mask_path: str) -> np.ndarray:
        """Load a saved mask from disk."""
        return np.load(mask_path)
    
    
    @staticmethod
    def visualize_mask_on_frame(
        frame: Image.Image,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Overlay mask on frame for visualization.
        
        Args:
            frame: PIL Image
            mask: Binary mask (H, W)
            color: RGB color tuple
            alpha: Transparency
            
        Returns:
            PIL Image with mask overlay
        """
        frame_np = np.array(frame).astype(np.float32)
        
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        # Create colored overlay
        mask_3d = mask[:, :, np.newaxis]
        color_np = np.array(color, dtype=np.float32)
        
        overlay = frame_np * (1 - alpha * mask_3d) + color_np * alpha * mask_3d
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return Image.fromarray(overlay)