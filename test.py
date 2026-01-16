import cv2
import numpy as np

img_path = "media/frames/frame_000010.jpg"
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"Could not read image: {img_path}")

quad = np.array([
    [1293.8565673828125, 108.8963851928711],
    [1517.2935791015625, 103.7108383178711],
    [1517.2935791015625, 362.9879455566406],
    [1309.737548828125, 326.07342529296875],
], dtype=np.float32)

# 🔑 OpenCV drawing needs int32
quad_int = quad.astype(np.int32)

cv2.polylines(img, [quad_int], isClosed=True, color=(0, 255, 0), thickness=2)

cv2.imwrite("output_quad.png", img)
