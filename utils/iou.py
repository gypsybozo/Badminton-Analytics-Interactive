# utils/iou.py
import numpy as np

def calculate_iou(boxA, boxB):
    """ Calculates Intersection over Union for two bounding boxes [x1, y1, x2, y2] """
    if boxA is None or boxB is None: return 0.0
    if not (isinstance(boxA, (list, np.ndarray)) and len(boxA) == 4): return 0.0
    if not (isinstance(boxB, (list, np.ndarray)) and len(boxB) == 4): return 0.0
    try:
        boxA = [int(c) for c in boxA]
        boxB = [int(c) for c in boxB]
        if boxA[0] >= boxA[2] or boxA[1] >= boxA[3] or boxB[0] >= boxB[2] or boxB[1] >= boxB[3]:
            return 0.0
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        # Add a small epsilon to avoid division by zero
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
    except Exception as e:
        # print(f"Error calculating IoU for {boxA}, {boxB}: {e}") # Optional debug
        return 0.0