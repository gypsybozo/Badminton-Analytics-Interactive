# badminton_analyzer.py
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime
import time
import traceback
import os
import math
import mediapipe as mp

# --- Local Imports ---
from trackers.court import CourtDetector
from trackers.shuttle_detector import ShuttleDetector
from trajectory_analyzer import TrajectoryAnalyzer
from frame_processor import FrameProcessor
from shot_confirmer import ShotConfirmer
from video_output import VideoOutputWriter
from data_manager import DataManager
from utils.constants import (PLAYER_CLASS_ID, RACKET_CLASS_ID, SHUTTLE_CLASS_ID,
                             COURT_LENGTH_METERS, COURT_HALF_LENGTH_METERS,
                             MIN_FRAMES_BETWEEN_CONFIRMED_SHOTS, FRAMES_BETWEEN_RALLIES)
from utils.geom import format_coords, get_bbox_center, calculate_distance

class BadmintonAnalyzer:
    def __init__(self, shuttle_model_path, court_model_path, video_path, conf_threshold=0.3, frame_skip=1):
        """ Initializes the BadmintonAnalyzer orchestrator and its components. """
        print("Initializing BadmintonAnalyzer Orchestrator...", flush=True)
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.frame_skip = max(1, frame_skip)
        self.homography_matrix = None
        self.court_corners = None
        self.class_labels = None

        # --- Load Models ---
        try:
            print(f"  Loading shuttle model from: {shuttle_model_path}", flush=True)
            # Important: Ensure the YOLO model is loaded correctly
            # Check if the path exists before loading
            if not os.path.exists(shuttle_model_path):
                 raise FileNotFoundError(f"YOLO model not found at {shuttle_model_path}")
            self.yolo_model = YOLO(shuttle_model_path)

            if hasattr(self.yolo_model, 'names'):
                self.class_labels = self.yolo_model.names
                print(f"  Using class labels from model: {self.class_labels}")
                 # Map class IDs using constants
                player_search = 'person'
                self.player_class_id = next((k for k, v in self.class_labels.items() if str(v).lower() == player_search.lower()), None)
                self.racket_class_id = next((k for k, v in self.class_labels.items() if str(v).lower() == 'racket'.lower()), None)
                self.shuttle_class_id = next((k for k, v in self.class_labels.items() if str(v).lower() == 'shuttle'.lower()), None)
                print(f"  Mapped Class IDs: Player={self.player_class_id}, Racket={self.racket_class_id}, Shuttle={self.shuttle_class_id}")
                if None in [self.player_class_id, self.racket_class_id, self.shuttle_class_id]:
                     raise ValueError("Could not map all required classes (person, racket, shuttle) in model.")
            else:
                 raise AttributeError("YOLO model missing 'names' attribute.")
            print("  Shuttle model loaded.", flush=True)

        except FileNotFoundError as e: print(f"\nERROR initializing models: {e}", flush=True); raise
        except AttributeError as e: print(f"\nERROR initializing models: {e}", flush=True); raise
        except ValueError as e: print(f"\nERROR initializing models: {e}", flush=True); raise
        except Exception as e: print(f"\nERROR initializing models: {e}\n{traceback.format_exc()}", flush=True); raise

        # --- Initialize Components ---
        try:
            print("  Initializing components...", flush=True)
            self.court_detector = CourtDetector(conf_threshold=conf_threshold, model_path=court_model_path)
            self.shuttle_detector = ShuttleDetector() # Uses default thresholds
            self.trajectory_analyzer = TrajectoryAnalyzer()
            # Pass only necessary parts to FrameProcessor
            self.frame_processor = FrameProcessor(self.yolo_model, self.class_labels)
            self.shot_confirmer = ShotConfirmer(self.frame_processor) # Pass frame_processor
            self.data_manager = DataManager()
            # Store output dirs for main.py reporting
            self.shots_output_dir = self.data_manager.shots_output_dir
            self.player_shots_dir = self.data_manager.player_shots_dir
            print("  Components initialized.", flush=True)
        except Exception as e: print(f"\nERROR initializing components: {e}\n{traceback.format_exc()}", flush=True); raise

        # --- Rally State ---
        self.current_rally_id = 0
        self.current_shot_num_in_rally = 0
        self.last_confirmed_shot_frame_index = -float('inf')

        print("BadmintonAnalyzer Orchestrator initialized successfully.", flush=True)

    def get_player_positions(self, frame_detections):
        """ Gets player positions using current detections and stored homography. """
        # (Keep the existing get_player_positions function code here - Uses self.homography_matrix)
        players_data = []
        if self.homography_matrix is None: return [(None, None), (None, None)], [(None, None), (None, None)], [None, None]
        for detection in frame_detections:
            if detection.get('class_id') == PLAYER_CLASS_ID:
                box = detection.get('box') 
                if not box: continue
                img_ctr = get_bbox_center(box);
                if not img_ctr: continue # Use util
                # Get bottom center for homography
                img_bottom_ctr = (img_ctr[0], box[3])
                real_pos = None
                try: real_pos = self.court_detector.translate_to_real_world(img_bottom_ctr, self.homography_matrix)
                except Exception as e: print(f"DEBUG: Error translating point {img_bottom_ctr}: {e}")
                if real_pos is not None and not np.isnan(real_pos).any() and not np.isinf(real_pos).any():
                    players_data.append({'real_pos': tuple(real_pos), 'image_center': img_ctr, 'box': box })
        # Sort players
        if len(players_data) == 2:
             try:
                valid = [p for p in players_data if p['real_pos'] and isinstance(p['real_pos'][1], (int, float))]
                if len(valid) != 2: return [(None, None), (None, None)], [(None, None), (None, None)], [None, None]
                valid.sort(key=lambda p: p['real_pos'][1])
                p1, p2 = valid[0], valid[1]
                return (p1['real_pos'], p2['real_pos']), (p1['image_center'], p2['image_center']), (p1['box'], p2['box'])
             except Exception as e: print(f"ERROR sorting players: {e}"); return [(None, None),(None, None)], [(None, None),(None, None)], [None, None]
        elif len(players_data) == 1:
             p = players_data[0]
             try:
                 if p['real_pos'] and p['real_pos'][1] < COURT_HALF_LENGTH_METERS: return (p['real_pos'], None), (p['image_center'], None), (p['box'], None) # P1
                 elif p['real_pos']: return (None, p['real_pos']), (None, p['image_center']), (None, p['box']) # P2
                 else: return [(None, None),(None, None)], [(None, None),(None, None)], [None, None]
             except: return [(None, None),(None, None)], [(None, None),(None, None)], [None, None]
        else: return [(None, None),(None, None)], [(None, None),(None, None)], [None, None]

    def process_video_with_shot_detection(self, save_output=True, output_path=None, draw_pose=True):
        """ Orchestrates the video processing pipeline using refactored components. """
        print("\nStarting Refactored Video Processing Pipeline...", flush=True)
        overall_start_time = time.time()

        # --- Video Capture and Output Setup ---
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): print(f"FATAL ERROR: Could not open video file {self.video_path}", flush=True); return None
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        effective_fps = (original_fps / self.frame_skip) if original_fps and self.frame_skip else 30.0
        video_writer = None
        if save_output:
            if output_path is None: output_path = f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            video_writer = VideoOutputWriter(output_path, effective_fps, frame_width, frame_height)
            if not video_writer.is_opened(): save_output = False

        # --- Data Collection Lists ---
        all_frames_processed = []
        all_detections_processed = []
        all_shuttle_candidates_processed = []
        all_poses_processed = []
        all_player_boxes_processed = []
        processed_frame_idx_map = {}
        
        # --- Initialize State Variables ---
        frame_counter = 0
        processed_frame_count = 0
        self.shuttle_detector.reset()
        # Reset homography state for this run
        self.homography_matrix = None
        self.court_corners = None
        court_detection_done = False # <-- INITIALIZE HERE
        last_p1_box, last_p2_box = None, None

        # --- Phase 1: Frame Collection ---
        print("\n--- Phase 1: Collecting frames, detections, and poses ---", flush=True)
        phase1_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break
            current_original_frame = frame_counter
            if (current_original_frame % self.frame_skip) != 0: frame_counter += 1; continue

            print(f"--- [DEBUG BA] Processing Frame {frame_counter} (Orig: {current_original_frame}, Proc Idx: {processed_frame_count}) ---", flush=True)
            processed_frame_idx_map[processed_frame_count] = current_original_frame
            frame_copy = frame.copy()

            # --- Homography (Once) ---
            if not court_detection_done and processed_frame_count <= 10:
             print("  [DEBUG BA] Checking Court/Homography...", flush=True)
             homography_start_time = time.time()
             try:
                 # --- ### START: FILL IN HOMOGRAPHY LOGIC ### ---
                 temp_court_coords = self.court_detector.detect_court_boundary(frame)
                 if temp_court_coords and len(temp_court_coords) == 4: # Or adjust condition based on your detector
                    print("    [DEBUG BA] Court boundary detected (potential corners).", flush=True)
                    # Assuming sort_court_coords returns sorted boxes/coords needed for corner derivation
                    self.court_coords = self.court_detector.sort_court_coords(temp_court_coords)
                    # Assuming draw_court_lines returns the 4 corner points needed for homography
                    corners = self.court_detector.draw_court_lines(frame_copy, self.court_coords) # Pass a copy if it draws

                    if corners and len(corners) == 4:
                        print(f"    [DEBUG BA] Derived corners: {corners}", flush=True)
                        self.homography_matrix = self.court_detector.compute_homography(corners)
                        if self.homography_matrix is not None:
                            print(f"INFO: Homography computed successfully at frame {current_original_frame}!", flush=True)
                            self.court_corners = corners # Store the corners used
                            court_detection_done = True
                            print(f"    [DEBUG BA] Homography duration: {time.time() - homography_start_time:.4f}s", flush=True)
                        else:
                            print(f"WARNING: Homography computation failed at frame {current_original_frame}.", flush=True)
                            self.court_corners = None
                    else:
                        print(f"    [DEBUG BA] Failed to derive valid corners at frame {current_original_frame}.", flush=True)
                 # --- ### END: FILL IN HOMOGRAPHY LOGIC ### ---
             except Exception as e:
                  print(f"    [DEBUG BA] ERROR during court detection/homography attempt: {e}\n{traceback.format_exc()}", flush=True)
             # Check if still not done after trying
             if not court_detection_done and processed_frame_count == 10:
                  print("INFO: Court/Homography not detected in initial frames. Proceeding without it.", flush=True)
                  court_detection_done = True # Stop trying

            # --- Process Frame (using FrameProcessor) ---
            detections, shuttle_candidates, poses, pose_dur = self.frame_processor.process_frame(
                frame, current_original_frame, self.conf_threshold, (last_p1_box, last_p2_box)
            )
            if detections is None: continue # Skip frame if processing failed

            # --- Get Current Player Boxes (Needed for next frame's pose fallback and drawing) ---
            _, _, current_player_boxes_tuple = self.get_player_positions(detections)
            current_p1_box, current_p2_box = current_player_boxes_tuple

            # --- Store Results ---
            all_frames_processed.append(frame_copy)
            all_detections_processed.append(detections)
            all_shuttle_candidates_processed.append(shuttle_candidates)
            all_poses_processed.append(poses)
            all_player_boxes_processed.append((current_p1_box, current_p2_box))

            # --- Update Last Known Boxes ---
            if current_p1_box: last_p1_box = current_p1_box
            if current_p2_box: last_p2_box = current_p2_box

            processed_frame_count += 1
            frame_counter += 1
        # --- End Phase 1 Loop ---
        cap.release()
        phase1_duration = time.time() - phase1_start_time
        print(f"\n--- Phase 1 Summary ---", flush=True)
        print(f"Duration: {phase1_duration:.2f}s. Processed {processed_frame_count} frames.", flush=True)
        # (Add list length checks/trimming if needed)

        # --- Phase 2: Interpolation & Best Shuttle Selection ---
        print("\n--- Phase 2: Interpolating shuttle positions ---", flush=True)
        phase2_start_time = time.time()
        best_shuttles = [self.shuttle_detector.select_best_shuttle_detection(cands) for cands in all_shuttle_candidates_processed]
        shuttle_boxes_for_interp = [d.get('box') if isinstance(d, dict) else None for d in best_shuttles]
        interpolated_positions = self.trajectory_analyzer.interpolate_shuttle_positions(shuttle_boxes_for_interp)
        phase2_duration = time.time() - phase2_start_time
        print(f"--- Phase 2 Duration: {phase2_duration:.2f}s ---", flush=True)

        # --- Phase 3: Candidate Detection ---
        print("\n--- Phase 3: Detecting potential shot triggers ---", flush=True)
        phase3_start_time = time.time()
        candidate_shot_indices = self.trajectory_analyzer.detect_direction_changes(interpolated_positions)
        phase3_duration = time.time() - phase3_start_time
        print(f"--- Phase 3 Duration: {phase3_duration:.2f}s. Candidates found: {len(candidate_shot_indices)} ---", flush=True)

        # --- Phase 4: Shot Confirmation ---
        print("\n--- Phase 4: Confirming shots ---", flush=True)
        phase4_start_time = time.time()
        shot_dataset = []
        confirmed_shot_indices = []
        self.current_rally_id = 0
        self.current_shot_num_in_rally = 0
        self.last_confirmed_shot_frame_index = -float('inf')

        if candidate_shot_indices:
            for candidate_idx, shot_candidate_idx in enumerate(candidate_shot_indices):
                if shot_candidate_idx - self.last_confirmed_shot_frame_index < MIN_FRAMES_BETWEEN_CONFIRMED_SHOTS:
                    print(f"  [DEBUG BA Conf] Skipping candidate {shot_candidate_idx}, too close to last confirmed {self.last_confirmed_shot_frame_index}", flush=True)
                    continue # Skip if too close to last confirmed

                  # --- Call ShotConfirmer (No likely_player needed) ---
                print(f"  [DEBUG BA] Calling confirm_shot for candidate index {shot_candidate_idx}", flush=True)
                confirmed, player_who_hit, method, conf_frame_idx = self.shot_confirmer.confirm_shot(
                    shot_candidate_idx, # Pass candidate index
                    all_detections_processed, # Pass all collected data
                    all_poses_processed,
                    interpolated_positions,
                    self.get_player_positions # Pass the method
                  )
                print(f"  [DEBUG BA] confirm_shot returned: confirmed={confirmed}, player={player_who_hit}, method='{method}', conf_idx={conf_frame_idx}", flush=True)

                if confirmed and player_who_hit != 0: # Check player_who_hit is valid
                    # (Rally Logic - same as before)
                    if shot_candidate_idx - self.last_confirmed_shot_frame_index > FRAMES_BETWEEN_RALLIES:
                        self.current_rally_id += 1
                        self.current_shot_num_in_rally = 0
                        print(f"\n--- [DEBUG BA] New Rally (# {self.current_rally_id}) ---", flush=True)
                    self.current_shot_num_in_rally += 1
                    self.last_confirmed_shot_frame_index = shot_candidate_idx # Update based on *trigger* time
                    confirmed_shot_indices.append(shot_candidate_idx)

                    # (Gather Data & Create Entry - same as before, using player_who_hit)
                    original_frame_num = processed_frame_idx_map.get(shot_candidate_idx, -1)
                    frame_at_conf = all_frames_processed[conf_frame_idx] if 0 <= conf_frame_idx < len(all_frames_processed) else None
                    detections_at_conf = all_detections_processed[conf_frame_idx] if 0 <= conf_frame_idx < len(all_detections_processed) else []
                    # Get positions at confirmation frame
                    p1_real_conf, p2_real_conf, p1_box_conf, p2_box_conf = None, None, None, None
                    try:
                       p_real_conf_tuple, _, p_box_conf_tuple = self.get_player_positions(detections_at_conf)
                       p1_real_conf, p2_real_conf = p_real_conf_tuple
                       p1_box_conf, p2_box_conf = p_box_conf_tuple
                    except: pass
                    shuttle_box_conf = interpolated_positions[conf_frame_idx] if 0 <= conf_frame_idx < len(interpolated_positions) else None
                    shuttle_center_img_conf = get_bbox_center(shuttle_box_conf)
                    shuttle_real_world_conf = None
                    if shuttle_center_img_conf and self.homography_matrix is not None:
                        try: shuttle_real_world_conf = self.court_detector.translate_to_real_world(shuttle_center_img_conf, self.homography_matrix)
                        except: shuttle_real_world_conf = None
                    shot_entry = {
                         'rally_id': self.current_rally_id, 'shot_num': self.current_shot_num_in_rally,
                         'player_who_hit': player_who_hit,
                         'player1_coords': format_coords(p1_real_conf), 'player2_coords': format_coords(p2_real_conf),
                         'shuttle_coords_impact': format_coords(shuttle_real_world_conf),
                         'shot_played': "Unknown", 'stroke_hand': "Unknown", 'hitting_posture': "Normal",
                         'confirmation_method': method,
                         'frame_number': original_frame_num,
                         'confirmation_frame': processed_frame_idx_map.get(conf_frame_idx, -1)
                    }
                    shot_dataset.append(shot_entry)
                    # (Save Image using self.data_manager as before)
                    racket_box_conf = next((d.get('box') for d in detections_at_conf if d.get('class_id') == RACKET_CLASS_ID), None)
                    shot_info = {'shuttle_box': shuttle_box_conf, 'player1_box': p1_box_conf, 'player2_box': p2_box_conf, 'racket_box': racket_box_conf}
                    self.data_manager.save_shot_image(frame_at_conf, shot_info, shot_entry)

        phase4_duration = time.time() - phase4_start_time
        print(f"--- Phase 4 Duration: {phase4_duration:.2f}s. Confirmed {len(shot_dataset)} shots. ---", flush=True)

        # --- Phase 5: Output Video ---
        print("\n--- Phase 5: Creating output video ---", flush=True)
        phase5_start_time = time.time()
        if save_output and video_writer and video_writer.is_opened():
             print(f"  [DEBUG BA] Writing video frames...", flush=True)
             for frame_idx, orig_frame_num in processed_frame_idx_map.items():
                  # Boundary checks for all necessary lists
                  if not (0 <= frame_idx < len(all_frames_processed) and \
                          0 <= frame_idx < len(interpolated_positions) and \
                          0 <= frame_idx < len(best_shuttles) and \
                          0 <= frame_idx < len(all_poses_processed) and \
                          0 <= frame_idx < len(all_player_boxes_processed)):
                          continue # Skip if any list index is invalid

                  frame_to_write = all_frames_processed[frame_idx]
                  shuttle_box = interpolated_positions[frame_idx]
                  shuttle_detected_originally = isinstance(best_shuttles[frame_idx], dict)
                  p1_landmarks, p2_landmarks = all_poses_processed[frame_idx]
                  p1_box, p2_box = all_player_boxes_processed[frame_idx]

                  video_writer.write_frame(
                       frame=frame_to_write, frame_index=frame_idx, original_frame_num=orig_frame_num,
                       court_corners=self.court_corners, interpolated_shuttle_box=shuttle_box,
                       shuttle_detection_status=shuttle_detected_originally,
                       p1_landmarks=p1_landmarks, p2_landmarks=p2_landmarks, p1_box=p1_box, p2_box=p2_box,
                       confirmed_shot_indices=confirmed_shot_indices, shot_dataset=shot_dataset,
                       draw_pose=draw_pose # Pass the flag
                  )
             video_writer.release()
        # (Handle cases where saving was disabled or writer failed)
        phase5_duration = time.time() - phase5_start_time
        print(f"--- Phase 5 Duration: {phase5_duration:.2f}s ---", flush=True)

        # --- Final Summary & CSV Saving ---
        print("\n--- Final Summary & Saving ---", flush=True)
        csv_path, save_success = self.data_manager.save_dataset_to_csv(shot_dataset)

        total_duration = time.time() - overall_start_time
        print(f"\n--- Total Processing Time: {total_duration:.2f} seconds ---", flush=True)
        avg_fps = processed_frame_count / total_duration if total_duration > 0 else 0
        print(f"--- Average Processing FPS (effective): {avg_fps:.2f} ---", flush=True)

        # --- Prepare Return Dictionary ---
        return {
            'shots_confirmed_indices': confirmed_shot_indices,
            'shot_original_frame_numbers': [processed_frame_idx_map.get(i, -1) for i in confirmed_shot_indices],
            'total_processed_frames': processed_frame_count,
            'dataset': shot_dataset,
            'dataset_path': csv_path,
            'video_output_path': output_path if save_output and video_writer else None
        }