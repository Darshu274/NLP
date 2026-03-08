"""Hybrid evaluation pipeline: function made with the help of ChatGPT"""

import cv2
import numpy as np

def looks_like_quantum_circuit(image_path: str) -> bool:
    """
    Heuristically determine whether an image visually resembles a quantum circuit.

    The function applies classical image processing techniques to detect
    structural patterns commonly found in quantum circuit diagrams, such as
    multiple long horizontal lines representing qubit wires. It uses edge
    detection and probabilistic Hough transform to identify line segments and
    counts horizontally aligned lines above a minimum length threshold.

    Parameters
    ----------
    image_path : str
        Path to the input image file.

    Returns
    -------
    bool
        True if the image is likely to represent a quantum circuit diagram,
        otherwise False.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # Resize for stable thresholds
    h, w = img.shape[:2]
    scale = 1200 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Binarize
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 7)

    # Find line segments
    edges = cv2.Canny(bw, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=60, maxLineGap=8)

    if lines is None:
        return False

    # Count mostly-horizontal long lines (wires)
    horiz = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        length = (dx*dx + dy*dy) ** 0.5
        if length > 80 and dy <= 6 and dx > 4*dy:
            horiz += 1

    # Circuits usually have multiple wires => several horizontal lines
    return horiz >= 4


#     # ---------------------------
#     # 6. Gate rectangle detection
#     # ---------------------------
#     contours, _ = cv2.findContours(
#         bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )

#     gate_boxes = 0
#     for c in contours:
#         x, y, w, h = cv2.boundingRect(c)
#         area = w * h
#         aspect = w / float(h + 1e-6)

#         # Typical quantum gate size and shape
#         if 400 < area < 20000 and 0.6 < aspect < 1.6:
#             gate_boxes += 1

#     # ---------------------------
#     # 7. Final decision rule
#     # ---------------------------
#     return (
#         horiz >= 4 and       # multiple qubit wires
#         gate_boxes >= 2 and  # at least two gates
#         vertical >= 1        # at least one control connection
#     )