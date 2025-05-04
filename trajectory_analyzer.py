# trajectory_analyzer.py
import pandas as pd
import numpy as np
import time
import traceback
import math # Make sure math is imported
from utils.geom import get_bbox_center
# Import constants correctly
from utils.constants import (TRAJ_WINDOW_SIZE, TRAJ_MIN_FRAMES_BETWEEN_CHANGES,
                             TRAJ_ANGLE_CHANGE_THRESHOLD, TRAJ_VELOCITY_THRESHOLD)

class TrajectoryAnalyzer:
    def __init__(self):
        print("  [DEBUG] Initializing TrajectoryAnalyzer...", flush=True)
        print("  [DEBUG] TrajectoryAnalyzer initialized.", flush=True)

    def interpolate_shuttle_positions(self, shuttle_detections_boxes):
        """ Interpolates missing shuttle positions (list of boxes or None) """
        print(f"    [DEBUG TA Interp] Interpolating {len(shuttle_detections_boxes)} potential boxes...", flush=True)
        if not shuttle_detections_boxes: return []

        processed_boxes = []
        for box in shuttle_detections_boxes:
            if box is None:
                processed_boxes.append([np.nan] * 4) # Use np.nan for numeric interpolation
            elif isinstance(box, (list, np.ndarray)) and len(box) == 4:
                safe_box = [(float(c) if isinstance(c, (int, float, np.number)) and np.isfinite(c) else np.nan) for c in box]
                processed_boxes.append(safe_box)
            else:
                 print(f"    [DEBUG TA Interp] Warning: Unexpected data type {type(box)}. Replacing with NaNs.")
                 processed_boxes.append([np.nan] * 4)

        try:
            df_positions = pd.DataFrame(processed_boxes, columns=['x1', 'y1', 'x2', 'y2'])

            # Interpolate using pandas (handles NaNs)
            df_positions = df_positions.interpolate(method='linear', limit_direction='both', axis=0)
            # Fill any remaining NaNs at ends if interpolation didn't cover them
            df_positions = df_positions.bfill().ffill()

            # Convert back to list of lists, making NaNs None and numbers integers
            interpolated_boxes = df_positions.values.tolist()
            result = []
            for box in interpolated_boxes:
                result.append([int(c) if pd.notna(c) else None for c in box])

            print(f"    [DEBUG TA Interp] Interpolation done, returning {len(result)} positions.", flush=True)
            return result

        except Exception as e:
             print(f"ERROR during DataFrame creation or interpolation: {e}\n{traceback.format_exc()}", flush=True)
             return [[None]*4] * len(processed_boxes)


    def detect_direction_changes(self, interpolated_positions):
        """ Detects potential shot candidates based on trajectory changes. """
        print(f"    [DEBUG TA Detect] Detecting direction changes on {len(interpolated_positions)} positions...", flush=True)
        # Use the constant for window size check
        required_len = TRAJ_WINDOW_SIZE * 2 + 1 # Need enough points for centered rolling window
        if not interpolated_positions or len(interpolated_positions) < required_len:
            print(f"    [DEBUG TA Detect] Not enough positions ({len(interpolated_positions)} < {required_len}) for direction change detection.", flush=True)
            return []

        # --- Calculate Smoothed Centers, Velocity, Angle ---
        centers = [get_bbox_center(box) for box in interpolated_positions]
        df = pd.DataFrame(centers, columns=['center_x', 'center_y'], index=pd.RangeIndex(len(centers)))

        # Drop rows where center calculation failed (returned None)
        valid_centers_mask = df.notna().all(axis=1)
        df_valid = df[valid_centers_mask].copy()

        if len(df_valid) < required_len: # Check again after dropping NaNs
            print(f"    [DEBUG TA Detect] Not enough valid center points ({len(df_valid)} < {required_len}) after dropna.", flush=True)
            return []

        # Apply smoothing (adjust window size or method if needed)
        smoothing_window = TRAJ_WINDOW_SIZE # Example: Use constant directly, maybe adjust formula if needed
        # Ensure window is odd for center=True
        if smoothing_window % 2 == 0: smoothing_window += 1
        print(f"    [DEBUG TA Detect] Using smoothing window: {smoothing_window}", flush=True)

        df_valid['x_smooth'] = df_valid['center_x'].rolling(window=smoothing_window, min_periods=2, center=True).mean()
        df_valid['y_smooth'] = df_valid['center_y'].rolling(window=smoothing_window, min_periods=2, center=True).mean()
        # Fill NaNs created by rolling window at edges - Assign back
        df_valid['x_smooth'] = df_valid['x_smooth'].fillna(df_valid['center_x'])
        df_valid['y_smooth'] = df_valid['y_smooth'].fillna(df_valid['center_y'])

        # Calculate differences on smoothed data
        df_valid['dx'] = df_valid['x_smooth'].diff()
        df_valid['dy'] = df_valid['y_smooth'].diff()

        # Calculate speed and angle (handle potential NaNs from diff)
        df_valid['speed'] = np.sqrt(df_valid['dx']**2 + df_valid['dy']**2)
        df_valid['angle'] = np.degrees(np.arctan2(df_valid['dy'], df_valid['dx']))

        # Calculate angle change robustly
        # Shift angle to compare previous frame to current
        angle_prev = df_valid['angle'].shift(1)
        df_valid['angle_change'] = np.abs(df_valid['angle'] - angle_prev)
        # Handle angle wrapping
        df_valid.loc[df_valid['angle_change'] > 180, 'angle_change'] = 360 - df_valid['angle_change']

        # --- Identify Potential Hit Indices ---
        # Ensure columns exist and are numeric before filtering
        required_cols = ['angle_change', 'speed']
        if not all(col in df_valid.columns for col in required_cols):
             print("    [DEBUG TA Detect] Error: Required columns for filtering not found.", flush=True)
             return []
        # Drop rows where angle_change or speed might be NaN (e.g., first row after diff)
        df_filtered = df_valid.dropna(subset=required_cols)

        potential_hit_indices = df_filtered[
            (df_filtered['angle_change'] > TRAJ_ANGLE_CHANGE_THRESHOLD) &
            (df_filtered['speed'] > TRAJ_VELOCITY_THRESHOLD)
        ].index.tolist()

        print(f"    [DEBUG TA Detect] Angle Change Threshold: {TRAJ_ANGLE_CHANGE_THRESHOLD}, Speed Threshold: {TRAJ_VELOCITY_THRESHOLD}")
        # print(f"    [DEBUG TA Detect] Sample data used for filtering:\n{df_filtered[['speed', 'angle_change']].head().to_string()}", flush=True) # Debug data

        if not potential_hit_indices:
            print("    [DEBUG TA Detect] No potential hits found based on angle/velocity thresholds.", flush=True)
            return []

        # --- Filter Hits Based on Min Frame Separation ---
        filtered_hits = []
        if potential_hit_indices:
            last_hit_frame_index = -float('inf')
            potential_hit_indices.sort() # Ensure indices are sorted
            for hit_idx in potential_hit_indices:
                # hit_idx is the original index because we used df_valid.index
                if hit_idx - last_hit_frame_index >= TRAJ_MIN_FRAMES_BETWEEN_CHANGES:
                    filtered_hits.append(hit_idx)
                    last_hit_frame_index = hit_idx

        print(f"    [DEBUG TA Detect] Found {len(potential_hit_indices)} potential candidates before time filter, filtered to {len(filtered_hits)}.", flush=True)
        return filtered_hits