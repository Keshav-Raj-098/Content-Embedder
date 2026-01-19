import cv2
import numpy as np
from PIL import Image

import os
import tempfile

from src.state import app_state as state,APPState


class PlanarTracker:
    def __init__(self, min_points: int = 20):
        self.prev_gray = None
        self.prev_pts = None
        self.active = False
        self.min_points = min_points
        
        
    def initialize(self, app_state: APPState):
        frame = app_state.video.frames[0]

        if not isinstance(frame, np.ndarray):
            raise RuntimeError(
                f"[CRITICAL] Expected numpy frame in initialize(), got {type(frame)}"
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        quad = app_state.placement.quad.astype(np.int32)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad, 255)

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=7,
            mask=mask
        )

        if pts is None or len(pts) < self.min_points:
            raise RuntimeError("Not enough features to track")

        self.prev_gray = gray
        self.prev_pts = pts
        self.active = True
        
        
        
        
    def update(self, app_state: APPState, frame_idx: int) -> bool:
        """
        Update tracking for the given frame index. 
        Returns True if tracking is successful, False otherwise.
        """
        if not self.active:
            return False

        frame = app_state.video.frames[frame_idx]
        

        if not isinstance(frame, np.ndarray):
            raise RuntimeError(
                f"[CRITICAL] Expected numpy frame, got {type(frame)} at index {frame_idx}"
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None
        )

        if curr_pts is None:
            self.active = False
            return False

        good_prev = self.prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_curr) < self.min_points:
            self.active = False
            return False

        H, inliers = cv2.findHomography(
            good_prev, good_curr, cv2.RANSAC, 3.0
        )

        if H is None or inliers.sum() < self.min_points:
            self.active = False
            return False

        quad = app_state.placement.quad.reshape(-1, 1, 2).astype(np.float32)
        new_quad = cv2.perspectiveTransform(quad, H).reshape(4, 2)

        h, w = gray.shape
        if np.all((new_quad[:, 0] < 0) |
                (new_quad[:, 0] > w) |
                (new_quad[:, 1] < 0) |
                (new_quad[:, 1] > h)):
            self.active = False
            return False

        app_state.placement.quad = new_quad
        self.prev_gray = gray
        self.prev_pts = good_curr.reshape(-1, 1, 2)

        return True


def render_ad_on_frame(app_state: APPState, frame_idx: int) -> np.ndarray:
    """
    Warp the ad image into the tracked quad and composite it onto the frame
    with feathered edges (no sharp boundaries).
    """

    # Base frame (BGR)
    frame = app_state.video.frames[frame_idx].copy()
    h, w = frame.shape[:2]

    # Source (ad) and destination (tracked quad)
    src_pts = app_state.ad.corners.astype(np.float32)
    dst_pts = app_state.placement.quad.astype(np.float32)

    # Homography: ad -> wall
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp ad to frame size
    warped_ad = cv2.warpPerspective(
        app_state.ad.image,
        H,
        (w, h)
    )

    # ----------------------------
    # 1️⃣ Build HARD mask from quad
    # ----------------------------
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)

    # ----------------------------
    # 2️⃣ Feather the mask
    # ----------------------------
    FEATHER_RADIUS = 31  # try 15 / 31 / 51
    mask = cv2.GaussianBlur(mask, (FEATHER_RADIUS, FEATHER_RADIUS), 0)

    # ----------------------------
    # 3️⃣ Alpha blending
    # ----------------------------
    alpha = mask.astype(np.float32) / 255.0
    alpha = alpha[..., None]  # (H, W, 1)

    out = (
        warped_ad.astype(np.float32) * alpha +
        frame.astype(np.float32) * (1.0 - alpha)
    )

    return out.astype(np.uint8)





# def apply_color_transfer(
#     fg: np.ndarray,
#     bg: np.ndarray,
#     mask: np.ndarray,
#     method: str = "luma_only"
# ) -> np.ndarray:
#     """
#     Alternative color matching using different strategies.
    
#     Args:
#         method: "luma_only", "subtle", "none"
#     """
    
#     if method == "none" or cv2.countNonZero(mask) < 50:
#         return fg
    
#     binary_mask = (mask > 30).astype(np.uint8) * 255
    
#     if method == "luma_only":
#         # Only adjust brightness, keep colors intact
#         fg_hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV).astype(np.float32)
#         bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float32)
        
#         # Get V (value/brightness) statistics
#         fg_v_mean = cv2.mean(fg_hsv[:, :, 2], mask=binary_mask)[0]
#         bg_v_mean = cv2.mean(bg_hsv[:, :, 2], mask=binary_mask)[0]
        
#         # Calculate adjustment
#         v_diff = bg_v_mean - fg_v_mean
        
#         # Apply only to masked region with conservative adjustment
#         mask_bool = mask > 0
#         fg_hsv[mask_bool, 2] = np.clip(
#             fg_hsv[mask_bool, 2] + v_diff * 0.4,  # 40% of the difference
#             0, 255
#         )
        
#         return cv2.cvtColor(fg_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
#     elif method == "subtle":
#         # Very subtle adjustment in LAB space
#         fg_lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB).astype(np.float32)
#         bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
        
#         mask_bool = mask > 0
        
#         for channel in range(3):
#             fg_ch_mean = cv2.mean(fg_lab[:, :, channel], mask=binary_mask)[0]
#             bg_ch_mean = cv2.mean(bg_lab[:, :, channel], mask=binary_mask)[0]
            
#             # Only apply 30% of the adjustment
#             adjustment = (bg_ch_mean - fg_ch_mean) * 0.3
#             fg_lab[mask_bool, channel] = np.clip(
#                 fg_lab[mask_bool, channel] + adjustment,
#                 0, 255
#             )
        
#         return cv2.cvtColor(fg_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
#     return fg


# def render_ad_on_frame(app_state: APPState, frame_idx: int) -> np.ndarray:
#     """
#     Warp the ad image into the tracked quad and composite it onto the frame
#     with feathered edges + brightness matching.
#     """
    
#     # Base frame (BGR)
#     frame = app_state.video.frames[frame_idx].copy()
#     h, w = frame.shape[:2]
    
#     # Source (ad) and destination (tracked quad)
#     src_pts = app_state.ad.corners.astype(np.float32)
#     dst_pts = app_state.placement.quad.astype(np.float32)
    
#     # Homography: ad -> wall
#     H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
#     # Warp ad to frame size with high-quality interpolation
#     warped_ad = cv2.warpPerspective(
#         app_state.ad.image,
#         H,
#         (w, h),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=(0, 0, 0)
#     )
    
#     # ----------------------------
#     # 1️⃣ Build mask from quad
#     # ----------------------------
#     mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    
#     # ----------------------------
#     # 2️⃣ Feather the mask
#     # ----------------------------
#     FEATHER_RADIUS = 31
#     mask = cv2.GaussianBlur(mask, (FEATHER_RADIUS, FEATHER_RADIUS), 0)
    
#     # ----------------------------
#     # ⭐ 3️⃣ CONSERVATIVE brightness matching
#     # ----------------------------
#     # Try these options in order of preference:
    
#     # OPTION 1: Luma only (recommended - keeps colors vibrant)
#     # warped_ad = apply_color_transfer(warped_ad, frame, mask, method="luma_only")
    
#     # OPTION 2: Conservative LAB matching (uncomment to try) = works well
#     warped_ad = match_brightness_and_contrast(
#         fg=warped_ad,
#         bg=frame,
#         mask=mask,
#         strength=0.4,  # Very conservative
#         preserve_saturation=True
#     )
    
#     # OPTION 3: No color correction (uncomment to try)
#     # warped_ad = apply_color_transfer(warped_ad, frame, mask, method="none")
    
#     # ----------------------------
#     # 4️⃣ Alpha blending
#     # ----------------------------
#     alpha = mask.astype(np.float32) / 255.0
#     alpha = alpha[..., None]
    
#     out = (
#         warped_ad.astype(np.float32) * alpha +
#         frame.astype(np.float32) * (1.0 - alpha)
#     )
    
#     return np.clip(out, 0, 255).astype(np.uint8)