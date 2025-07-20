# shot_confirmer.py
import math
import time
import numpy as np
import traceback
from utils.constants import (RACKET_CLASS_ID, RACKET_SHUTTLE_IOU_THRESHOLD,
                             WRIST_SHUTTLE_PROXIMITY_THRESHOLD, CONFIRMATION_WINDOW,
                             PLAYER_CLASS_ID, # Keep if needed for _get_player_id_for_racket
                             LEFT_WRIST, RIGHT_WRIST) # Import wrist indices
from utils.iou import calculate_iou
from utils.geom import calculate_distance, get_bbox_center

class ShotConfirmer:
    def __init__(self, frame_processor):
        """Initializes the ShotConfirmer."""
        print("  [DEBUG] Initializing ShotConfirmer...", flush=True)
        self.frame_processor = frame_processor # Needed for get_wrist_keypoints
        print("  [DEBUG] ShotConfirmer initialized.", flush=True)

    def _get_player_id_for_racket(self, racket_center, p1_center, p2_center, max_dist_factor=1.5):
        """ Determines player ID based on racket proximity to player centers. """
        if racket_center is None: return 0
        # Calculate distances only if player centers are valid points
        dist1 = calculate_distance(racket_center, p1_center)
        dist2 = calculate_distance(racket_center, p2_center)

        # Define a reasonable max distance threshold (e.g., 150-200 pixels)
        # Might need tuning based on video resolution and typical player separation
        max_allowable_dist = max_dist_factor * 125 # Example: 187.5px

        # Assign to closest player only if within the max distance threshold
        if dist1 < dist2 and dist1 < max_allowable_dist:
             # print(f"          Racket assigned to P1 (Dist: {dist1:.1f})") # Debug
             return 1
        elif dist2 < dist1 and dist2 < max_allowable_dist:
             # print(f"          Racket assigned to P2 (Dist: {dist2:.1f})") # Debug
             return 2
        else:
             # print(f"          Racket unassigned (D1:{dist1:.1f}, D2:{dist2:.1f}, Max:{max_allowable_dist:.1f})") # Debug
             return 0 # Unassigned or too far

    def confirm_shot(self, shot_candidate_idx,
                     all_frame_detections_processed, all_player_poses_processed,
                     interpolated_positions, get_player_positions_func,player1_dominant_wrist_idx, # Passed from BadmintonAnalyzer
                     player2_dominant_wrist_idx):
        """
        Applies the 3-stage confirmation logic for a candidate shot by checking
        an interaction window and prioritizing the best evidence found.

        Returns:
            tuple: (confirmed, player_who_hit, confirmation_method,
                    confirmation_frame_idx, hitting_wrist_info)
                   hitting_wrist_info is dict {'index': ?, 'coord': ?} or None.
        """
        print(f"\n[DEBUG SC Check {shot_candidate_idx}] Confirming Shot...", flush=True)

        # --- Initialize variables to track best evidence across the window ---
        best_iou_hit = None # Stores {'iou': val, 'frame': idx, 'r_box': box, 'p1c': p1c, 'p2c': p2c}
        best_wrist_hit_data = None
        best_wrist_hit = None # Stores {'dist': val, 'frame': idx, 'player': pid, 'index': widx, 'coord': wcoord}

        min_overall_wrist_dist = float('inf') # For logging rejected cases
        max_overall_iou = 0.0 # For logging rejected cases

        # --- Define Interaction Window ---
        start_check_idx = max(0, shot_candidate_idx - CONFIRMATION_WINDOW)
        end_check_idx = min(len(interpolated_positions) - 1, shot_candidate_idx + CONFIRMATION_WINDOW)
        print(f"  [DEBUG SC Check {shot_candidate_idx}] Interaction window: [{start_check_idx} - {end_check_idx}]", flush=True)

        # --- Iterate through the time window around the candidate ---
        interaction_check_start = time.time()
        for check_idx in range(start_check_idx, end_check_idx + 1):
            # print(f"    [DEBUG SC Window] Checking Index: {check_idx}", flush=True) # Verbose

            # --- Get Data for Current Check Index ---
            if not (0 <= check_idx < len(all_frame_detections_processed) and \
                    0 <= check_idx < len(interpolated_positions) and \
                    0 <= check_idx < len(all_player_poses_processed)):
                 continue # Skip if index out of bounds for any required list

            current_detections = all_frame_detections_processed[check_idx]
            shuttle_box_check = interpolated_positions[check_idx]
            poses_check = all_player_poses_processed[check_idx] # (p1_landmarks, p2_landmarks)

            if shuttle_box_check is None or not all(c is not None for c in shuttle_box_check):
                 continue # Skip if no valid shuttle box

            shuttle_center_check = get_bbox_center(shuttle_box_check)
            if not shuttle_center_check: continue # Skip if no valid shuttle center

            # --- Get Player Centers/Boxes for association/calculation ---
            p1_center, p2_center = None, None
            p1_box, p2_box = None, None
            try:
                 _, p_img_ctr_tuple, p_box_tuple = get_player_positions_func(current_detections)
                 p1_center, p2_center = p_img_ctr_tuple
                 p1_box, p2_box = p_box_tuple
            except Exception as e: print(f"      [DEBUG SC] Error getting player pos data @{check_idx}: {e}", flush=True)


            # --- Stage 2: Check ALL Rackets for IoU ---
            rackets = [d for d in current_detections if d.get('class_id') == RACKET_CLASS_ID]
            for r in rackets:
                r_box = r.get('box')
                if not r_box: continue
                try:
                    iou = calculate_iou(shuttle_box_check, r_box)
                    if iou > max_overall_iou: max_overall_iou = iou # Track max for logging

                    if iou > RACKET_SHUTTLE_IOU_THRESHOLD:
                        # Found a potential IoU hit, store its details
                        potential_hit = {'iou': iou, 'frame': check_idx, 'r_box': r_box, 'p1c': p1_center, 'p2c': p2_center}
                        print(f"      [DEBUG Stage 2 @{check_idx}] Potential IoU Hit Found: {iou:.3f}", flush=True)
                        # Update best_iou_hit if this one is better
                        if best_iou_hit is None or iou > best_iou_hit['iou']:
                            best_iou_hit = potential_hit
                except Exception as e: print(f"      [DEBUG SC Stage 2] Error calculating IoU: {e}", flush=True)


            # --- Stage 3: Check ALL Wrists for Proximity ---
            p1_landmarks, p2_landmarks = poses_check
            # all_wrist_candidates_this_frame = [] # Track potential hits *in this frame*

            # Player 1 Dominant Wrist
            if p1_landmarks and p1_box:
                try:
                    p1_x1,p1_y1,p1_x2,p1_y2 = map(int,p1_box); p1_w,p1_h = p1_x2-p1_x1,p1_y2-p1_y1
                    if p1_w > 0 and p1_h > 0:
                        p1_lw_rel, p1_rw_rel = self.frame_processor.get_wrist_keypoints(p1_landmarks, p1_w, p1_h)
                        p1_dominant_wrist_rel = p1_rw_rel if player1_dominant_wrist_idx == RIGHT_WRIST else p1_lw_rel

                        if p1_dominant_wrist_rel:
                            p1_dom_wrist_abs = (p1_dominant_wrist_rel[0] + p1_x1, p1_dominant_wrist_rel[1] + p1_y1)
                            dist = calculate_distance(p1_dom_wrist_abs, shuttle_center_check)
                            if dist < min_overall_wrist_dist: min_overall_wrist_dist = dist # Track for logging
                            print(f"      [DEBUG Stage 3 P1 DomWrist @{check_idx}] Idx:{player1_dominant_wrist_idx}, Coord:{p1_dom_wrist_abs}, Dist:{dist:.1f}", flush=True)
                            if dist < WRIST_SHUTTLE_PROXIMITY_THRESHOLD:
                                # This is a potential dominant wrist hit for P1
                                current_wrist_hit = {'dist': dist, 'frame': check_idx, 'player': 1,
                                                     'index': player1_dominant_wrist_idx, 'coord': p1_dom_wrist_abs}
                                # Update best_wrist_hit if this one is better
                                if best_wrist_hit_data is None or dist < best_wrist_hit_data['dist']:
                                    best_wrist_hit_data = current_wrist_hit
                                print(f"        Potential P1 Dominant Wrist Hit!", flush=True)
                except Exception as e: print(f"      [DEBUG SC Stage 3] Error P1 DomWrist: {e}", flush=True)

            # Player 2 Dominant Wrist
            if p2_landmarks and p2_box:
                try:
                    p2_x1,p2_y1,p2_x2,p2_y2 = map(int,p2_box); p2_w,p2_h = p2_x2-p2_x1,p2_y2-p2_y1
                    if p2_w > 0 and p2_h > 0:
                        p2_lw_rel, p2_rw_rel = self.frame_processor.get_wrist_keypoints(p2_landmarks, p2_w, p2_h)
                        p2_dominant_wrist_rel = p2_rw_rel if player2_dominant_wrist_idx == RIGHT_WRIST else p2_lw_rel

                        if p2_dominant_wrist_rel:
                            p2_dom_wrist_abs = (p2_dominant_wrist_rel[0] + p2_x1, p2_dominant_wrist_rel[1] + p2_y1)
                            dist = calculate_distance(p2_dom_wrist_abs, shuttle_center_check)
                            if dist < min_overall_wrist_dist: min_overall_wrist_dist = dist
                            print(f"      [DEBUG Stage 3 P2 DomWrist @{check_idx}] Idx:{player2_dominant_wrist_idx}, Coord:{p2_dom_wrist_abs}, Dist:{dist:.1f}", flush=True)
                            if dist < WRIST_SHUTTLE_PROXIMITY_THRESHOLD:
                                current_wrist_hit = {'dist': dist, 'frame': check_idx, 'player': 2,
                                                     'index': player2_dominant_wrist_idx, 'coord': p2_dom_wrist_abs}
                                if best_wrist_hit_data is None or dist < best_wrist_hit_data['dist']:
                                    best_wrist_hit_data = current_wrist_hit
                                print(f"        Potential P2 Dominant Wrist Hit!", flush=True)
                except Exception as e: print(f"      [DEBUG SC Stage 3] Error P2 DomWrist: {e}", flush=True)

        # --- End Window Check Loop ---
        interaction_check_dur = time.time() - interaction_check_start

        # --- Make Final Decision AFTER Checking Window ---
        confirmed = False
        player_who_hit = 0
        confirmation_method = "None"
        confirmation_frame_idx = -1
        hitting_wrist_info = None

        # Prioritize Best IoU hit from the entire window
        if best_iou_hit is not None:
            # Try to assign player to the best IoU hit
            iou, check_idx, r_box, p1c, p2c = best_iou_hit['iou'], best_iou_hit['frame'], best_iou_hit['r_box'], best_iou_hit['p1c'], best_iou_hit['p2c']
            r_center = get_bbox_center(r_box)
            p_id = self._get_player_id_for_racket(r_center, p1c, p2c)
            if p_id != 0:
                confirmed = True
                player_who_hit = p_id
                confirmation_method = f"IoU ({iou:.3f})"
                confirmation_frame_idx = check_idx
                hitting_wrist_info = None # Clear wrist info
                print(f"  [DEBUG SC Decision] Prioritizing IoU Confirmation.", flush=True)
            else:
                 print(f"  [DEBUG SC Decision] Best IoU hit ({iou:.3f} @{check_idx}) couldn't be assigned to a player. Falling back to wrist check.", flush=True)
                 # Fall through to check wrist hits

        # If no valid IoU confirmation, check best wrist hit from the entire window
        if not confirmed and best_wrist_hit_data is not None: # Only consider wrist if IoU didn't confirm
            dist, frame, p_id, wrist_idx, wrist_coord = best_wrist_hit_data['dist'], best_wrist_hit_data['frame'], best_wrist_hit_data['player'], best_wrist_hit_data['index'], best_wrist_hit_data['coord']
            confirmed = True; player_who_hit = p_id
            confirmation_method = f"Wrist ({dist:.1f}px)"; confirmation_frame_idx = frame
            hitting_wrist_info = {'index': wrist_idx, 'coord': wrist_coord} # This is the dominant wrist info


        # --- Log and Return ---
        if confirmed:
            print(f"  [DEBUG SC Check {shot_candidate_idx}] FINAL RESULT: CONFIRMED. Player: {player_who_hit}, Method: {confirmation_method} at index {confirmation_frame_idx}.", flush=True)
            return True, player_who_hit, confirmation_method, confirmation_frame_idx, hitting_wrist_info
        else:
            print(f"  [DEBUG SC Check {shot_candidate_idx}] FINAL RESULT: REJECTED. (Max IoU Found: {max_overall_iou:.3f}, Min Dom. Wrist Dist Found: {min_overall_wrist_dist:.1f}px)", flush=True)
            return False, 0, "None", -1, None