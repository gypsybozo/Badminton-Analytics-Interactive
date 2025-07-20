# utils/geom.py
import math
import numpy as np
import pandas as pd # For isnan check
from trackers.court import CourtDetector

def get_shuttle_ground_projection(shuttle_box_image, homography_matrix):
        if shuttle_box_image is None or homography_matrix is None:
            return None
        # Option 1: Use centroid (simpler if bottom isn't clear)
        shuttle_center_image = get_bbox_center(shuttle_box_image)
        if not shuttle_center_image: return None
        # Option 2: Try to estimate bottom-center of shuttle if possible (more accurate for ground)
        # This is tricky for a small, fast object. Centroid might be more robust.
        # For now, let's use centroid:
        point_to_project = shuttle_center_image

        try:
            # Homography expects (x, y)
            real_world_coords = CourtDetector.translate_to_real_world( # Assuming CourtDetector has this static/class method
                point_to_project, homography_matrix
            )
            if real_world_coords is not None and not np.isnan(real_world_coords).any():
                return tuple(real_world_coords[:2]) # Return (X_ground, Y_ground)
        except Exception as e:
            print(f"    [DEBUG SHUTTLE PROJ] Error projecting shuttle point {point_to_project}: {e}", flush=True)
        return None
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