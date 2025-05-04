# utils/geom.py
import math
import numpy as np
import pandas as pd # For isnan check

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    if point1 is None or point2 is None: return float('inf')
    # Check if points are valid tuples/lists of numbers
    if not (isinstance(point1, (tuple, list)) and len(point1) == 2 and all(isinstance(c, (int, float)) for c in point1)): return float('inf')
    if not (isinstance(point2, (tuple, list)) and len(point2) == 2 and all(isinstance(c, (int, float)) for c in point2)): return float('inf')
    try:
        # Ensure no NaN/inf values
        if any(not math.isfinite(c) for c in point1) or any(not math.isfinite(c) for c in point2):
            return float('inf')
        return math.dist(point1, point2)
    except Exception: # Catch potential errors during calculation
        return float('inf')

def get_bbox_center(box):
    """Calculate the center point of a bounding box [x1, y1, x2, y2]"""
    if box is None or len(box) != 4: return None
    try:
        # Ensure all coordinates are valid numbers before casting
        if any(c is None or not isinstance(c, (int, float, np.number)) or not math.isfinite(c) for c in box):
            return None
        x1, y1, x2, y2 = map(int, box)
        # Add check for valid box dimensions
        if x1 >= x2 or y1 >= y2: return None
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    except (ValueError, TypeError):
        return None

def format_coords(coords):
    """ Formats real-world coordinates safely """
    # --- MODIFIED CONDITION ---
    # 1. Check specifically for None first.
    # 2. Then check type (tuple or list) and length.
    # 3. Then check element types and finiteness.
    if coords is not None and \
       isinstance(coords, (tuple, list)) and \
       len(coords) == 2 and \
       all(isinstance(c, (float, int, np.number)) for c in coords) and \
       all(math.isfinite(c) for c in coords):
         # Format only if all checks pass
         return f"({coords[0]:.4f}, {coords[1]:.4f})"
    # --- END MODIFIED CONDITION ---

    # Return None if any check fails
    return None

def crop_image_safely(frame, box):
    """ Safely crops an image using bounding box coordinates """
    if frame is None or box is None: return None
    try:
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if y1 >= y2 or x1 >= x2: return None # Check for valid crop dimensions
        return frame[y1:y2, x1:x2]
    except Exception as e:
        print(f"    [DEBUG CROP UTIL] Error cropping image: {e}", flush=True)
        return None