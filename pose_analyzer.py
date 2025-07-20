# pose_analyzer.py
import mediapipe as mp
import numpy as np
import math
# Import constants including wrist indices
from utils.constants import (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
                             LEFT_WRIST, RIGHT_WRIST, POSE_VISIBILITY_THRESHOLD, RIGHT_HANDED, LEFT_HANDED)
from utils.geom import calculate_distance # Optional

class PoseAnalyzer:
    def __init__(self):
        print("  [DEBUG PA] Initializing PoseAnalyzer...", flush=True)
        print("  [DEBUG PA] PoseAnalyzer initialized.", flush=True)

    def get_landmark_coords(self, landmarks, landmark_index, img_width, img_height):
        """Safely gets visible landmark coordinates."""
        # (Keep the existing get_landmark_coords function)
        if landmarks is None or landmark_index >= len(landmarks.landmark): return None
        try:
            landmark = landmarks.landmark[landmark_index]
            if landmark.visibility > POSE_VISIBILITY_THRESHOLD:
                x = min(img_width - 1, max(0, int(landmark.x * img_width)))
                y = min(img_height - 1, max(0, int(landmark.y * img_height)))
                return (x, y)
            else: return None
        except Exception as e: print(f"    [DEBUG PA] Error getting landmark {landmark_index}: {e}", flush=True); return None

    def analyze_stroke_type(self, player_id, handedness, landmarks, hitting_wrist_coord, hitting_wrist_index, img_width, img_height):
        """
        Analyzes pose landmarks to determine stroke type based on handedness.

        Args:
            player_id (int): 1 or 2.
            handedness (str): 'right' or 'left'.
            landmarks (...): Pose landmarks.
            hitting_wrist_coord (tuple): (x, y) coordinates of hitting wrist.
            hitting_wrist_index (int): Landmark index (LEFT_WRIST or RIGHT_WRIST) that hit.
            img_width (int): Width of the image/frame.
            img_height (int): Height of the image/frame.

        Returns:
            str: "Forehand High", "Forehand Low", "Backhand High", "Backhand Low", or "Unknown".
        """
        stroke = "Unknown"
        if landmarks is None or hitting_wrist_coord is None or handedness not in [RIGHT_HANDED, LEFT_HANDED]:
            print(f"    [DEBUG PA Stroke P{player_id}] Skipping: Missing landmarks, wrist coord, or invalid handedness ('{handedness}').", flush=True)
            return stroke

        # Get relevant landmarks
        left_shoulder = self.get_landmark_coords(landmarks, LEFT_SHOULDER, img_width, img_height)
        right_shoulder = self.get_landmark_coords(landmarks, RIGHT_SHOULDER, img_width, img_height)

        # Determine dominant/reference shoulder based on handedness
        dominant_shoulder_coord = None
        # Define non_dominant_wrist_index needed for backhand check refinement
        non_dominant_wrist_index = None

        if handedness == RIGHT_HANDED:
            dominant_shoulder_coord = right_shoulder
            non_dominant_wrist_index = LEFT_WRIST
        else: # LEFT_HANDED
            dominant_shoulder_coord = left_shoulder
            non_dominant_wrist_index = RIGHT_WRIST

        if hitting_wrist_coord is None or dominant_shoulder_coord is None:
            print(f"    [DEBUG PA Stroke P{player_id} {handedness}] Skipping: Missing hitting wrist or dominant shoulder.", flush=True)
            return stroke

        # --- Define Body Centerline ---
        torso_center_x = None
        if left_shoulder and right_shoulder:
            torso_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        elif dominant_shoulder_coord: # Fallback using dominant shoulder
            torso_center_x = dominant_shoulder_coord[0]
        else: # Fallback using the other shoulder if dominant was missing (shouldn't happen often now)
             other_shoulder = left_shoulder if handedness == RIGHT_HANDED else right_shoulder
             if other_shoulder: torso_center_x = other_shoulder[0]
             else: print(f"    [DEBUG PA Stroke P{player_id} {handedness}] Skipping: No visible shoulders for torso center.", flush=True); return stroke

        wrist_x, wrist_y = hitting_wrist_coord
        dominant_shoulder_x, dominant_shoulder_y = dominant_shoulder_coord

        # --- Height Check (Relative to Dominant Shoulder) ---
        height_tolerance = 50 # Pixels
        is_high_shot = wrist_y >= (dominant_shoulder_y + height_tolerance)

        # --- Side Check (Relative to Torso Center X) ---
        # These definitions are absolute (screen left/right)
        is_right_side_of_torso = wrist_x > torso_center_x
        is_left_side_of_torso = wrist_x < torso_center_x

        print(f"    [DEBUG PA Stroke P{player_id} {handedness}] Wrist:({wrist_x},{wrist_y}), TorsoX:{torso_center_x:.0f}, DomShoulderY:{dominant_shoulder_y}, High:{is_high_shot}, RightOfTorso:{is_right_side_of_torso}", flush=True)

        # --- Decision Logic based on Handedness ---
        if handedness == RIGHT_HANDED:
            if is_high_shot:
                if is_right_side_of_torso: stroke = "Forehand High" # Overhead Forehand
                else: stroke = "Backhand High" # Overhead Backhand
            else: # Low Shot
                if is_right_side_of_torso: stroke = "Forehand Low"
                else: stroke = "Backhand Low"
            # Refinement: If it looks like a backhand based on side, but the *right* wrist hit, it's awkward/forced
            if "Backhand" in stroke and hitting_wrist_index == RIGHT_WRIST:
                 print(f"    [DEBUG PA Stroke P{player_id} {handedness}] Info: Classified as Backhand but RIGHT wrist hit.", flush=True)
                 # Keep as Backhand for now, or add a new category like "Forced Backhand"?

        else: # LEFT_HANDED
            if is_high_shot:
                if is_left_side_of_torso: stroke = "Forehand High" # Overhead Forehand
                else: stroke = "Backhand High" # Overhead Backhand
            else: # Low Shot
                if is_left_side_of_torso: stroke = "Forehand Low"
                else: stroke = "Backhand Low"
            # Refinement: If it looks like a backhand based on side, but the *left* wrist hit, it's awkward/forced
            if "Backhand" in stroke and hitting_wrist_index == LEFT_WRIST:
                 print(f"    [DEBUG PA Stroke P{player_id} {handedness}] Info: Classified as Backhand but LEFT wrist hit.", flush=True)

        print(f"    [DEBUG PA Stroke P{player_id} {handedness}] Determined Stroke: {stroke}", flush=True)
        return stroke