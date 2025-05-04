# shot_confirmer.py
import math
import time
import numpy as np
import traceback
from utils.constants import (RACKET_CLASS_ID, RACKET_SHUTTLE_IOU_THRESHOLD,
                             WRIST_SHUTTLE_PROXIMITY_THRESHOLD, CONFIRMATION_WINDOW,
                             PLAYER_CLASS_ID)
from utils.iou import calculate_iou
from utils.geom import calculate_distance, get_bbox_center

class ShotConfirmer:
    def __init__(self, frame_processor): # Removed court_detector dependency for now
        print("  [DEBUG] Initializing ShotConfirmer...", flush=True)
        self.frame_processor = frame_processor # Needed for get_wrist_keypoints
        print("  [DEBUG] ShotConfirmer initialized.", flush=True)
        
    def _get_player_id_for_racket(self, racket_center, p1_center, p2_center, max_dist_factor=1.5):
        """ Determines player ID based on racket proximity to player centers. """
        if racket_center is None: return 0
        dist1 = calculate_distance(racket_center, p1_center)
        dist2 = calculate_distance(racket_center, p2_center)

        # Simple closest player assignment (can be refined)
        if dist1 < dist2 and dist1 < max_dist_factor * 100: # Add a max distance check (e.g., 150-200px)
             # print(f"          Racket assigned to P1 (Dist: {dist1:.1f})") # Debug
             return 1
        elif dist2 < dist1 and dist2 < max_dist_factor * 100:
             # print(f"          Racket assigned to P2 (Dist: {dist2:.1f})") # Debug
             return 2
        else:
             # print(f"          Racket unassigned (D1:{dist1:.1f}, D2:{dist2:.1f})") # Debug
             return 0 # Unassigned or too far


    def confirm_shot(self, shot_candidate_idx,
                     all_frame_detections_processed, all_player_poses_processed,
                     interpolated_positions, get_player_positions_func): # Pass player pos func
        """
        Applies the 3-stage confirmation logic for a candidate shot.
        Returns: tuple (confirmed, player_who_hit, confirmation_method, confirmation_frame_idx)
        """
        print(f"\n[DEBUG SC Check {shot_candidate_idx}] Confirming Shot...", flush=True) 
        confirmed = False
        player_who_hit = 0 # Reset for each candidate
        confirmation_method = "None"
        confirmation_frame_idx = -1
        best_iou = 0.0 # Track best IoU found for any racket
        min_wrist_dist = float('inf') # Track overall min wrist dist
        confirming_iou_details = {} # Store details if IoU confirms
        confirming_wrist_details = {} # Store details if Wrist confirms

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
             
            # --- Get Player Centers for association ---
            p1_center, p2_center = None, None
            p1_box, p2_box = None, None
            try:
                 _, p_img_ctr_tuple, p_box_tuple = get_player_positions_func(current_detections)
                 p1_center, p2_center = p_img_ctr_tuple
                 p1_box, p2_box = p_box_tuple # Get boxes for wrist calculations later
            except Exception as e: print(f"      [DEBUG SC] Error getting player pos data @{check_idx}: {e}", flush=True)

            # --- Stage 2: Racket-Shuttle IoU ---
            rackets = [d for d in current_detections if d.get('class_id') == RACKET_CLASS_ID]
            # print(f"      [DEBUG Stage 2 @{check_idx}] Rackets found: {len(rackets)}") # Verbose
            for r in rackets:
                r_box = r.get('box')
                if not r_box: continue
                try:
                    iou = calculate_iou(shuttle_box_check, r_box)
                    if iou > best_iou: best_iou = iou # Track max iou found globally

                    if iou > RACKET_SHUTTLE_IOU_THRESHOLD:
                        r_center = get_bbox_center(r_box)
                        # Determine player for THIS racket
                        p_id = self._get_player_id_for_racket(r_center, p1_center, p2_center)
                        if p_id != 0: # Only confirm if racket can be assigned
                            confirmed = True
                            player_who_hit = p_id
                            confirmation_method = f"IoU ({iou:.3f})"
                            confirmation_frame_idx = check_idx
                            # Store details about the confirmation
                            confirming_iou_details = {'iou': iou, 'racket_box': r_box, 'player': p_id, 'frame': check_idx}
                            print(f"      [DEBUG Stage 2] !!!!! Tentative IoU CONFIRMATION @{check_idx} by Player {p_id} (IoU: {iou:.3f}) !!!!!", flush=True)
                            break # Found a confirming IoU, stop checking rackets for this frame_idx
                except Exception as e: print(f"      [DEBUG SC Stage 2] Error calculating IoU: {e}", flush=True)
            if confirmed: break # Stop checking other frames if confirmed by IoU

            # --- Stage 3: Check BOTH Players' Wrists for Proximity (Fallback) ---
            if not confirmed: # Only proceed if IoU didn't confirm
                p1_landmarks, p2_landmarks = poses_check
                shuttle_center_check = get_bbox_center(shuttle_box_check) # Recalculate just in case
                if not shuttle_center_check: continue

                # Check Player 1 Wrist
                if p1_landmarks and p1_box:
                     try:
                         p_img_x1, p_img_y1, p_img_x2, p_img_y2 = map(int, p1_box)
                         p_img_w, p_img_h = p_img_x2 - p_img_x1, p_img_y2 - p_img_y1
                         if p_img_w > 0 and p_img_h > 0:
                             lw_rel, rw_rel = self.frame_processor.get_wrist_keypoints(p1_landmarks, p_img_w, p_img_h)
                             lw_abs = (lw_rel[0] + p_img_x1, lw_rel[1] + p_img_y1) if lw_rel else None
                             rw_abs = (rw_rel[0] + p_img_x1, rw_rel[1] + p_img_y1) if rw_rel else None
                             for wrist_coords_abs in [lw_abs, rw_abs]:
                                 if wrist_coords_abs:
                                     dist = calculate_distance(wrist_coords_abs, shuttle_center_check)
                                     if dist < min_wrist_dist: min_wrist_dist = dist # Track overall min
                                     if dist < WRIST_SHUTTLE_PROXIMITY_THRESHOLD:
                                         # Tentatively confirm based on P1 wrist
                                         if not confirmed: # Only set if not already confirmed by P2 wrist in same frame
                                             confirmed = True; player_who_hit = 1
                                             confirmation_method = f"Wrist ({dist:.1f}px)"; confirmation_frame_idx = check_idx
                                             confirming_wrist_details = {'dist': dist, 'wrist_coord': wrist_coords_abs, 'player': 1, 'frame': check_idx}
                                             print(f"      [DEBUG Stage 3] !!!!! Tentative Wrist CONFIRMATION @{check_idx} by Player 1 (Dist: {dist:.1f}) !!!!!", flush=True)
                                             break # Found P1 wrist confirmation
                     except Exception as e: print(f"      [DEBUG SC Stage 3] Error during P1 wrist check: {e}", flush=True)
                if confirmed: break # Stop checking P2 if P1 confirmed

                # Check Player 2 Wrist
                if not confirmed and p2_landmarks and p2_box: # Only check if P1 didn't confirm
                     try:
                         p_img_x1, p_img_y1, p_img_x2, p_img_y2 = map(int, p2_box)
                         p_img_w, p_img_h = p_img_x2 - p_img_x1, p_img_y2 - p_img_y1
                         if p_img_w > 0 and p_img_h > 0:
                             lw_rel, rw_rel = self.frame_processor.get_wrist_keypoints(p2_landmarks, p_img_w, p_img_h)
                             lw_abs = (lw_rel[0] + p_img_x1, lw_rel[1] + p_img_y1) if lw_rel else None
                             rw_abs = (rw_rel[0] + p_img_x1, rw_rel[1] + p_img_y1) if rw_rel else None
                             for wrist_coords_abs in [lw_abs, rw_abs]:
                                 if wrist_coords_abs:
                                     dist = calculate_distance(wrist_coords_abs, shuttle_center_check)
                                     if dist < min_wrist_dist: min_wrist_dist = dist
                                     if dist < WRIST_SHUTTLE_PROXIMITY_THRESHOLD:
                                         # Tentatively confirm based on P2 wrist
                                         confirmed = True; player_who_hit = 2
                                         confirmation_method = f"Wrist ({dist:.1f}px)"; confirmation_frame_idx = check_idx
                                         confirming_wrist_details = {'dist': dist, 'wrist_coord': wrist_coords_abs, 'player': 2, 'frame': check_idx}
                                         print(f"      [DEBUG Stage 3] !!!!! Tentative Wrist CONFIRMATION @{check_idx} by Player 2 (Dist: {dist:.1f}) !!!!!", flush=True)
                                         break # Found P2 wrist confirmation
                     except Exception as e: print(f"      [DEBUG SC Stage 3] Error during P2 wrist check: {e}", flush=True)
                if confirmed: break # Stop checking other frames if confirmed by wrist

        interaction_check_dur = time.time() - interaction_check_start
        # --- Final Decision after Checking Window ---
        if confirmed: # Check if any confirmation occurred in the window
             # Prioritize IoU confirmation if both happened (potentially in different frames of window)
             if confirming_iou_details and confirming_wrist_details:
                 # Simple rule: If IoU confirmation happened, prefer it.
                 # More complex: check if wrist confirmation is closer in time to candidate_idx?
                 print(f"  [DEBUG SC Check {shot_candidate_idx}] Confirmed by both IoU and Wrist in window. Prioritizing IoU.")
                 player_who_hit = confirming_iou_details['player']
                 confirmation_method = f"IoU ({confirming_iou_details['iou']:.3f})"
                 confirmation_frame_idx = confirming_iou_details['frame']
             elif confirming_iou_details:
                 # Already set correctly
                 pass
             elif confirming_wrist_details:
                 # Already set correctly
                 pass
             # Else block should not happen if confirmed is True

             print(f"  [DEBUG SC Check {shot_candidate_idx}] FINAL RESULT: CONFIRMED. Player: {player_who_hit}, Method: {confirmation_method} at index {confirmation_frame_idx}.", flush=True)
             return True, player_who_hit, confirmation_method, confirmation_frame_idx
        else:
             print(f"  [DEBUG SC Check {shot_candidate_idx}] FINAL RESULT: REJECTED. (Max IoU: {best_iou:.3f}, Min Wrist: {min_wrist_dist:.1f}px)", flush=True)
             return False, 0, "None", -1