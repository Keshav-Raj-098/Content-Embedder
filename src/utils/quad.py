import numpy as np
from src.state import app_state as state




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