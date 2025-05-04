# shot_confirmer.py
import math
import time
import numpy as np
import traceback
from utils.constants import (RACKET_CLASS_ID, RACKET_SHUTTLE_IOU_THRESHOLD,
                             WRIST_SHUTTLE_PROXIMITY_THRESHOLD, CONFIRMATION_WINDOW)
from utils.iou import calculate_iou
from utils.geom import calculate_distance, get_bbox_center

class ShotConfirmer:
    def __init__(self, frame_processor): # Removed court_detector dependency for now
        print("  [DEBUG] Initializing ShotConfirmer...", flush=True)
        self.frame_processor = frame_processor # Needed for get_wrist_keypoints
        print("  [DEBUG] ShotConfirmer initialized.", flush=True)

    def confirm_shot(self, shot_candidate_idx, likely_player, # Pass likely_player in
                     all_frame_detections_processed, all_player_poses_processed,
                     interpolated_positions, get_player_positions_func): # Pass player pos func
        """
        Applies the 3-stage confirmation logic for a candidate shot.
        Returns: tuple (confirmed, player_who_hit, confirmation_method, confirmation_frame_idx)
        """
        print(f"\n[DEBUG SC Check {shot_candidate_idx}] Likely Player={likely_player}", flush=True)
        confirmed = False
        player_who_hit = 0
        confirmation_method = "None"
        confirmation_frame_idx = -1 # Processed index where confirmation occurred
        best_iou_in_window = 0.0
        min_wrist_dist_in_window = float('inf')

        # Define window, ensuring bounds are valid
        start_check_idx = max(0, shot_candidate_idx - CONFIRMATION_WINDOW)
        end_check_idx = min(len(interpolated_positions) - 1, shot_candidate_idx + CONFIRMATION_WINDOW)
        print(f"  [DEBUG SC Check {shot_candidate_idx}] Interaction window: [{start_check_idx} - {end_check_idx}]", flush=True)

        # --- Check Interaction Window Frames ---
        interaction_check_start = time.time()
        for check_idx in range(start_check_idx, end_check_idx + 1):
            if confirmed: break

            print(f"    [DEBUG SC Window] Checking Index: {check_idx}", flush=True) # Verbose
            # --- Get Data for Current Check Index ---
            if not (0 <= check_idx < len(all_frame_detections_processed) and \
                    0 <= check_idx < len(interpolated_positions) and \
                    0 <= check_idx < len(all_player_poses_processed)):
                 print(f"    [DEBUG SC Window] Skipping index {check_idx} - out of bounds.", flush=True) # Verbose
                 continue

            current_detections = all_frame_detections_processed[check_idx]
            shuttle_box_check = interpolated_positions[check_idx]
            poses_check = all_player_poses_processed[check_idx] # (p1_landmarks, p2_landmarks)

            if shuttle_box_check is None or not all(c is not None for c in shuttle_box_check):
                 print(f"    [DEBUG SC Window] Skipping index {check_idx}: No valid shuttle box.", flush=True) # Verbose
                 continue

            # --- Stage 2: Racket-Shuttle IoU ---
            if not confirmed and likely_player != 0:
                rackets = [d for d in current_detections if d.get('class_id') == RACKET_CLASS_ID]
                player_rackets_boxes = []
                try:
                    _, _, p_box_check_tuple = get_player_positions_func(current_detections) # Call passed function
                    target_player_box = p_box_check_tuple[0] if likely_player == 1 else p_box_check_tuple[1]

                    if target_player_box and rackets:
                        target_center = get_bbox_center(target_player_box)
                        max_dist = max(target_player_box[2]-target_player_box[0], target_player_box[3]-target_player_box[1]) * 0.75 if target_player_box[2]>target_player_box[0] and target_player_box[3]>target_player_box[1] else 200 # Default distance

                        if target_center:
                             for r in rackets:
                                 r_box = r.get('box')
                                 r_center = get_bbox_center(r_box)
                                 if r_center and calculate_distance(target_center, r_center) < max_dist:
                                     player_rackets_boxes.append(r_box)
                except Exception as e: print(f"      [DEBUG SC Stage 2] Error finding rackets: {e}", flush=True)

                for r_box in player_rackets_boxes:
                     try:
                         iou = calculate_iou(shuttle_box_check, r_box)
                         if iou > best_iou_in_window: best_iou_in_window = iou
                         if iou > RACKET_SHUTTLE_IOU_THRESHOLD:
                              confirmed = True; player_who_hit = likely_player
                              confirmation_method = f"IoU ({iou:.3f})"; confirmation_frame_idx = check_idx
                              print(f"      [DEBUG SC Stage 2] CONFIRMED via IoU at index {check_idx} by Player {player_who_hit}", flush=True)
                              break
                     except Exception as e: print(f"      [DEBUG SC Stage 2] Error calculating IoU: {e}", flush=True)
                if confirmed: continue # Skip stage 3 if confirmed


            # --- Stage 3: Wrist-Shuttle Proximity ---
            if not confirmed and likely_player != 0:
                p1_landmarks, p2_landmarks = poses_check
                player_landmarks = p1_landmarks if likely_player == 1 else p2_landmarks
                if player_landmarks:
                    target_player_box = None
                    try:
                        _, _, p_box_check_tuple = get_player_positions_func(current_detections)
                        target_player_box = p_box_check_tuple[0] if likely_player == 1 else p_box_check_tuple[1]
                    except Exception as e: print(f"      [DEBUG SC Stage 3] Error getting player box: {e}", flush=True)

                    if target_player_box:
                         try:
                             p_img_x1, p_img_y1, p_img_x2, p_img_y2 = map(int, target_player_box)
                             p_img_w, p_img_h = p_img_x2 - p_img_x1, p_img_y2 - p_img_y1
                             if p_img_w > 0 and p_img_h > 0: # Ensure valid dimensions
                                 lw_coords, rw_coords = self.frame_processor.get_wrist_keypoints(player_landmarks, p_img_w, p_img_h)
                                 # Adjust coords
                                 if lw_coords: lw_coords = (lw_coords[0] + p_img_x1, lw_coords[1] + p_img_y1)
                                 if rw_coords: rw_coords = (rw_coords[0] + p_img_x1, rw_coords[1] + p_img_y1)

                                 shuttle_center_check = get_bbox_center(shuttle_box_check)
                                 if shuttle_center_check:
                                     for wrist_coords in [lw_coords, rw_coords]:
                                         if wrist_coords:
                                             dist = calculate_distance(wrist_coords, shuttle_center_check)
                                             if dist < min_wrist_dist_in_window: min_wrist_dist_in_window = dist
                                             if dist < WRIST_SHUTTLE_PROXIMITY_THRESHOLD:
                                                 confirmed = True; player_who_hit = likely_player
                                                 confirmation_method = f"Wrist ({dist:.1f}px)"; confirmation_frame_idx = check_idx
                                                 print(f"      [DEBUG SC Stage 3] CONFIRMED via Wrist at index {check_idx} by Player {player_who_hit}", flush=True)
                                                 break # wrist loop
                                 if confirmed: break # check_idx loop
                         except Exception as e: print(f"      [DEBUG SC Stage 3] Error during wrist check logic: {e}", flush=True)

        interaction_check_dur = time.time() - interaction_check_start
        print(f"  [DEBUG SC Check {shot_candidate_idx}] Interaction check duration: {interaction_check_dur:.4f}s", flush=True) # Verbose

        if confirmed and player_who_hit != 0:
             print(f"  [DEBUG SC Check {shot_candidate_idx}] CONFIRMED. Method: {confirmation_method} at index {confirmation_frame_idx}.", flush=True) # Verbose
             return True, player_who_hit, confirmation_method, confirmation_frame_idx
        else:
             print(f"  [DEBUG SC Check {shot_candidate_idx}] REJECTED. (Max IoU: {best_iou_in_window:.3f}, Min Wrist: {min_wrist_dist_in_window:.1f}px)", flush=True) # Verbose
             return False, 0, "None", -1