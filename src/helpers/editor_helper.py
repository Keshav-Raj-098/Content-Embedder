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



def match_brightness_and_contrast(
    fg: np.ndarray,
    bg: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.6,
    preserve_saturation: bool = True,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Match brightness & contrast of fg to bg under the given mask.
    
    Args:
        fg: uint8 BGR foreground image
        bg: uint8 BGR background image  
        mask: uint8 (0–255), feathered mask
        strength: How much to apply correction (0.0-1.0). Lower = more conservative
        preserve_saturation: Keep original color saturation
        eps: small value to prevent division by zero
    
    Returns:
        Color-matched foreground image
    """
    
    # If mask is too small, skip
    if cv2.countNonZero(mask) < 50:
        return fg

    fg_f = fg.astype(np.float32)
    bg_f = bg.astype(np.float32)
    
    # Create binary mask for stable statistics
    binary_mask = (mask > 30).astype(np.uint8) * 255
    
    if preserve_saturation:
        # Work in LAB color space to preserve color information
        fg_lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Only adjust L (luminance) channel
        fg_l_mean = cv2.mean(fg_lab[:, :, 0], mask=binary_mask)[0]
        bg_l_mean = cv2.mean(bg_lab[:, :, 0], mask=binary_mask)[0]
        
        fg_l_std = np.std(fg_lab[binary_mask > 0, 0]) if np.any(binary_mask > 0) else 1.0
        bg_l_std = np.std(bg_lab[binary_mask > 0, 0]) if np.any(binary_mask > 0) else 1.0
        
        matched_lab = fg_lab.copy()
        mask_bool = mask > 0
        
        # Apply correction only where mask exists
        l_channel = matched_lab[:, :, 0]
        l_adjusted = (l_channel - fg_l_mean) * (bg_l_std / (fg_l_std + eps)) + bg_l_mean
        
        # Blend between original and adjusted based on strength
        l_channel[mask_bool] = (
            strength * l_adjusted[mask_bool] + 
            (1 - strength) * l_channel[mask_bool]
        )
        
        matched_lab[:, :, 0] = np.clip(l_channel, 0, 255)
        
        matched = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        
    else:
        # Original method but more conservative
        fg_mean, fg_std = cv2.meanStdDev(fg_f, mask=binary_mask)
        bg_mean, bg_std = cv2.meanStdDev(bg_f, mask=binary_mask)
        
        fg_mean = fg_mean.reshape(1, 1, 3)
        fg_std = fg_std.reshape(1, 1, 3)
        bg_mean = bg_mean.reshape(1, 1, 3)
        bg_std = bg_std.reshape(1, 1, 3)
        
        matched = fg_f.copy()
        mask_bool = mask > 0
        
        # Apply transformation
        adjusted = (fg_f - fg_mean) * (bg_std / (fg_std + eps)) + bg_mean
        
        # Blend between original and adjusted
        matched[mask_bool] = (
            strength * adjusted[mask_bool] + 
            (1 - strength) * fg_f[mask_bool]
        )
    
    return np.clip(matched, 0, 255).astype(np.uint8)




def match_texture(fg: np.ndarray, bg: np.ndarray, mask: np.ndarray, strength=0.25):
    """
    Transfers high-frequency texture from bg to fg inside mask.
    """

    fg = fg.astype(np.float32)
    bg = bg.astype(np.float32)

    # High-frequency extraction
    fg_blur = cv2.GaussianBlur(fg, (7, 7), 0)
    bg_blur = cv2.GaussianBlur(bg, (7, 7), 0)

    fg_high = fg - fg_blur
    bg_high = bg - bg_blur

    mask_f = (mask > 0).astype(np.float32)[..., None]

    fg += strength * (bg_high - fg_high) * mask_f

    return np.clip(fg, 0, 255)



def add_film_grain(img, strength=0.015):
    h, w, c = img.shape
    noise = np.random.normal(0, 1, (h, w, 1)).astype(np.float32)

    # Blur noise slightly to mimic sensor grain
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    
    # 🔒 HARD SAFETY: ensure 3 channels
    if noise.ndim == 2:
        noise = noise[..., None]

    img = img.astype(np.float32)
    img += img * noise * strength

    return np.clip(img, 0, 255)



def render_ad_on_frame(app_state: APPState, frame_idx: int) -> np.ndarray:
    """
    Physically-plausible ad compositing with:
    - Spatial lighting transfer
    - Contact shadows
    - Noise & gamma matching
    """

    # ---------------------------------------------------
    # Base frame
    # ---------------------------------------------------
    frame = app_state.video.frames[frame_idx].copy()
    h, w = frame.shape[:2]

    src_pts = app_state.ad.corners.astype(np.float32)
    dst_pts = app_state.placement.quad.astype(np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_ad = cv2.warpPerspective(
        app_state.ad.image,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # ---------------------------------------------------
    # Mask
    # ---------------------------------------------------
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = mask_f[..., None]

    # ---------------------------------------------------
    # 1️⃣ Spatial illumination extraction (MOST IMPORTANT)
    # ---------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    illumination = cv2.GaussianBlur(gray, (101, 101), 0)
    illumination = illumination.astype(np.float32) / 255.0
    illumination = illumination[..., None]

    warped_ad = warped_ad.astype(np.float32)
    warped_ad *= illumination

    # ---------------------------------------------------
    # 2️⃣ Conservative brightness normalization
    # ---------------------------------------------------
    warped_ad = match_brightness_and_contrast(
        fg=warped_ad.astype(np.uint8),
        bg=frame,
        mask=mask,
        strength=0.25,
        preserve_saturation=True
    ).astype(np.float32)

    # ---------------------------------------------------
    # 3️⃣ Contact shadow (edge darkening)
    # ---------------------------------------------------
    edge = cv2.Canny(mask, 50, 150)
    edge = cv2.GaussianBlur(edge, (31, 31), 0)
    edge = edge.astype(np.float32) / 255.0
    warped_ad *= (1.0 - 0.18 * edge[..., None])

    # ---------------------------------------------------
    # 4️⃣ Noise & texture matching (SAFE)
    # ---------------------------------------------------
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    noise = frame.astype(np.float32) - frame_blur.astype(np.float32)

    warped_ad += 0.35 * noise
    warped_ad = np.clip(warped_ad, 0, 255)
    
    # ---------------------------------------------------
    # 4.5️⃣ Texture matching (NEW)
    # ---------------------------------------------------
    warped_ad = match_texture(
        fg=warped_ad,
        bg=frame,
        mask=mask,
        strength=0.2
    )
    
    
    # ---------------------------------------------------
    # 4.7️⃣ Film grain (NEW)
    # ---------------------------------------------------
    warped_ad = add_film_grain(warped_ad, strength=0.020)



    # ---------------------------------------------------
    # 5️⃣ Gamma-correct blending (SAFE)
    # ---------------------------------------------------
    def to_linear(img):
        img = np.clip(img, 0.0, 255.0)
        img = img / 255.0
        return np.power(img, 2.2)

    def to_srgb(img):
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, 1.0 / 2.2)
        return img * 255.0

    out = (
        to_linear(warped_ad) * mask_f +
        to_linear(frame.astype(np.float32)) * (1.0 - mask_f)
    )

    out = to_srgb(out)


    return np.clip(out, 0, 255).astype(np.uint8)
