import cv2
import numpy as np
from PIL import Image

import os
import tempfile

from src.state import app_state as state

import numpy as np

def normalize_quad(quad: np.ndarray) -> np.ndarray:
    """
    Returns quad ordered as: TL, TR, BR, BL
    """
    quad = np.asarray(quad, dtype=np.float32)

    center = quad.mean(axis=0)

    angles = np.arctan2(quad[:,1] - center[1],
                        quad[:,0] - center[0])
    quad = quad[np.argsort(angles)]

    s = quad.sum(axis=1)
    diff = quad[:,0] - quad[:,1]

    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmax(diff)]
    bl = quad[np.argmin(diff)]

    return np.float32([tl, tr, br, bl])



def compute_quad(x, y, scale, offsets):
    ah, aw = state.ad.image.shape[:2]
    w = int(aw * scale)
    h = int(ah * scale)

    base = np.float32([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h],
    ])

    return base + offsets, w, h


def overlay_ad_with_base_and_offsets(
    x, y, scale,
    tl_dx, tl_dy,
    tr_dx, tr_dy,
    br_dx, br_dy,
    bl_dx, bl_dy
):
    if not state.video.frame_rgb or state.ad.image is None:
        return None

    if state.video.current_idx != 0:
        return state.video.frame_rgb[state.video.current_idx]

    frame = np.array(state.video.frame_rgb[0])
    fh, fw = frame.shape[:2]

    offsets = np.float32([
        [tl_dx, tl_dy],
        [tr_dx, tr_dy],
        [br_dx, br_dy],
        [bl_dx, bl_dy],
    ])

    quad, w, h = compute_quad(x, y, scale, offsets)

    state.placement.quad = quad
    state.placement.x = int(x)
    state.placement.y = int(y)
    state.placement.scale = scale
    state.placement.w = w
    state.placement.h = h
    state.placement.corner_offsets = offsets

    ad = state.ad.image
    ah, aw = ad.shape[:2]

    H = cv2.getPerspectiveTransform(
        np.float32([[0,0],[aw,0],[aw,ah],[0,ah]]),
        quad
    )

    warped = cv2.warpPerspective(ad, H, (fw, fh))

    mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
    bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(warped, warped, mask=mask)

    return Image.fromarray(cv2.add(bg, fg))


def propagate_ad():
    if state.placement.quad is None:
        return "No placement found", None, None

    frames = state.video.frames
    ad = state.ad.image

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts = state.placement.quad.reshape(-1, 1, 2).astype(np.float32)

    output_frames = []

    # Correct frame-0
    output_frames.append(
        overlay_ad_with_base_and_offsets(
            state.placement.x,
            state.placement.y,
            state.placement.scale,
            *state.placement.corner_offsets.flatten()
        )
    )

    for i in range(1, len(frames)):
        frame = frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if status.sum() < 4:
            break

        quad = next_pts.reshape(4, 2)
        fh, fw = frame.shape[:2]
        ah, aw = ad.shape[:2]

        H = cv2.getPerspectiveTransform(
            np.float32([[0,0],[aw,0],[aw,ah],[0,ah]]),
            quad
        )

        warped = cv2.warpPerspective(ad, H, (fw, fh))
        mask = (warped.sum(axis=2) > 0).astype(np.uint8) * 255
        bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        fg = cv2.bitwise_and(warped, warped, mask=mask)

        output_frames.append(
            Image.fromarray(cv2.cvtColor(cv2.add(bg, fg), cv2.COLOR_BGR2RGB))
        )

        prev_gray = gray
        prev_pts = next_pts

    state.video.processed_frames = output_frames

    return "Propagation complete", None, output_frames
