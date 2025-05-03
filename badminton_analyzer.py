# badminton_analyzer.py
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime
import os
import math
# Ensure trackers.court maps to the correct file path if it's in a subdirectory
from trackers.court import CourtDetector
from shot_detector import ShotDetector
from shuttle_detector import ShuttleDetector
from shuttle_trajectory import ShuttleTrajectoryAnalyzer
import traceback # For detailed error printing

# --- Constants ---
PLAYER_CLASS_ID = 0
RACKET_CLASS_ID = 1
SHUTTLE_CLASS_ID = 2
COURT_LENGTH_METERS = 13.4

class BadmintonAnalyzer:
    def __init__(self, shuttle_model_path, court_model_path, video_path, conf_threshold=0.3, frame_skip=1):
        """
        Initialize the BadmintonAnalyzer with model and video paths

        Args:
            shuttle_model_path (str): Path to trained YOLOv8 model weights for shuttle detection
            court_model_path (str): Path to trained YOLOv8 model weights for court detection
            video_path (str): Path to input video file
            conf_threshold (float): Confidence threshold for detections
            frame_skip (int): Process every Nth frame
        """
        print("Initializing BadmintonAnalyzer...")
        try:
            print(f"  Loading shuttle model from: {shuttle_model_path}")
            self.model = YOLO(shuttle_model_path)
            print("  Shuttle model loaded.")
            self.video_path = video_path
            self.conf_threshold = conf_threshold
            self.frame_skip = max(1, frame_skip) # Ensure frame_skip is at least 1
            self.trajectory_data = []

            print("  Initializing CourtDetector...")
            self.court_detector = CourtDetector(
                conf_threshold=conf_threshold,
                model_path=court_model_path
            )
            print("  CourtDetector initialized.")
            self.court_coords = None
            self.court_corners = None
            self.homography_matrix = None

            print("  Initializing ShuttleDetector...")
            self.shuttle_detector = ShuttleDetector(max_shuttle_movement=400, min_shuttle_movement=10)
            print("  ShuttleDetector initialized.")
            print("  Initializing ShuttleTrajectoryAnalyzer...")
            self.trajectory_analyzer = ShuttleTrajectoryAnalyzer()
            print("  ShuttleTrajectoryAnalyzer initialized.")

            # --- Class Label Handling ---
            if hasattr(self.model, 'names'):
                 self.class_labels = self.model.names
                 print(f"  Using class labels from model: {self.class_labels}")
            else:
                 raise AttributeError("Loaded YOLO model does not have a 'names' attribute for class labels.")

            # *** ADJUSTED THIS LINE ***
            player_search_name = 'person'    # Changed from 'player' based on model output
            # **************************
            racket_search_name = 'racket'    # This one seems correct
            shuttle_search_name = 'shuttle' # This one seems correct

            # Find the numeric IDs using a case-insensitive search
            self.player_class_id = next((k for k, v in self.class_labels.items() if str(v).lower() == player_search_name.lower()), None)
            self.racket_class_id = next((k for k, v in self.class_labels.items() if str(v).lower() == racket_search_name.lower()), None)
            self.shuttle_class_id = next((k for k, v in self.class_labels.items() if str(v).lower() == shuttle_search_name.lower()), None)

            print(f"  Mapped Class IDs: Player={self.player_class_id}, Racket={self.racket_class_id}, Shuttle={self.shuttle_class_id}")

            # Check if essential classes were found
            missing_classes = []
            if self.player_class_id is None: missing_classes.append(f"'{player_search_name}'")
            # Only check shuttle if needed, but it was found before
            if self.shuttle_class_id is None: missing_classes.append(f"'{shuttle_search_name}'")

            if missing_classes:
                 # Updated error message slightly
                 error_message = f"Could not find required class(es) {', '.join(missing_classes)} in the loaded model's names: {self.class_labels}. Please adjust search names in the __init__ method."
                 raise ValueError(error_message)
            # --- End Class Label Handling ---


            print("  Initializing ShotDetector...")
            self.shot_detector = ShotDetector()
            print("  ShotDetector initialized.")

            self.current_rally_id = 0
            self.current_shot_num = 0
            self.frames_between_rallies = 150
            self.shot_data = []
            self.shots_output_dir = "shot_images"
            self.player_shots_dir = "player_shot_images"

            print(f"  Creating output directories ('{self.shots_output_dir}', '{self.player_shots_dir}')...")
            os.makedirs(self.shots_output_dir, exist_ok=True)
            os.makedirs(self.player_shots_dir, exist_ok=True)
            print("BadmintonAnalyzer initialized successfully.")

        except FileNotFoundError as e:
            print(f"\nERROR initializing BadmintonAnalyzer: Model file not found.")
            print(e)
            raise
        except Exception as e:
            print(f"\nERROR initializing BadmintonAnalyzer: {e}")
            print(traceback.format_exc())
            raise


    def crop_player_image(self, frame, player_box):
        """ Safely crops the player image from the frame """
        # (Same as previous version - no changes needed here)
        if player_box is None: return None
        try:
            x1, y1, x2, y2 = map(int, player_box)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if y1 >= y2 or x1 >= x2: return None
            return frame[y1:y2, x1:x2]
        except Exception as e:
            print(f"Error cropping player image with box {player_box}: {e}")
            return None

    def get_player_positions(self, frame_detections):
        """
        Get player positions (real-world and image), sorted by real-world Y.
        Returns real_pos, image_center, and box for each player.
        """
        players_data = [] # Stores {'real_pos':(x,y), 'image_center':(px,py), 'box':[x1,y1,x2,y2]}
        if self.homography_matrix is None:
             # Return structure indicating failure but matching format
             return [(None, None), (None, None)], [(None, None), (None, None)], [None, None]

        for detection in frame_detections:
            if detection.get('class_id') == self.player_class_id:
                box = detection.get('box')
                if not box: continue

                # Image center calculation (bottom-center)
                image_center_x = (box[0] + box[2]) // 2
                image_center_y = box[3]
                image_center = (image_center_x, image_center_y)

                # Real-world position calculation
                real_world_pos = self.court_detector.translate_to_real_world(
                    image_center, self.homography_matrix
                )

                # Store if real-world position is valid (needed for sorting)
                if real_world_pos is not None:
                    if isinstance(real_world_pos, (np.ndarray, list)) and len(real_world_pos) == 2:
                       if not np.isnan(real_world_pos).any() and not np.isinf(real_world_pos).any():
                           players_data.append({
                               'real_pos': tuple(real_world_pos),
                               'image_center': image_center, # Store image center
                               'box': box
                           })
                       # else: print(...) # DEBUG NaN/Inf
                    # else: print(...) # DEBUG Invalid format
        # --- Sort Players (based on real_pos Y) ---
        if len(players_data) == 2:
            try:
                # Filter ensure real_pos exists and Y is valid before sorting
                valid_players = [p for p in players_data if isinstance(p.get('real_pos'), tuple) and len(p['real_pos'])==2 and isinstance(p['real_pos'][1], (int, float))]
                if len(valid_players) != 2:
                     print("Warning: Could not get 2 valid real positions for sorting.")
                     # Return default empty structure
                     return [(None, None), (None, None)], [(None, None), (None, None)], [None, None]

                sorted_players_data = sorted(valid_players, key=lambda p: p['real_pos'][1])

                p1_data = sorted_players_data[0]
                p2_data = sorted_players_data[1]
                # Return all relevant data per player
                return (p1_data['real_pos'], p2_data['real_pos']), \
                       (p1_data['image_center'], p2_data['image_center']), \
                       (p1_data['box'], p2_data['box'])
            except Exception as e:
                 print(f"ERROR sorting player positions: {e}");
                 return [(None, None), (None, None)], [(None, None), (None, None)], [None, None]
        elif len(players_data) == 1:
             # Handle single player (return data in correct slot based on Y)
             p_data = players_data[0]
             p_real = p_data['real_pos']
             p_img_ctr = p_data['image_center']
             p_box = p_data['box']
             try:
                 half_court_y = COURT_LENGTH_METERS / 2
                 if isinstance(p_real[1], (int, float)) and p_real[1] < half_court_y:
                     return (p_real, None), (p_img_ctr, None), (p_box, None) # P1
                 elif isinstance(p_real[1], (int, float)):
                     return (None, p_real), (None, p_img_ctr), (None, p_box) # P2
                 else: return [(None, None), (None, None)], [(None, None), (None, None)], [None, None]
             except Exception as e:
                 print(f"Error determining side for single player: {e}");
                 return [(None, None), (None, None)], [(None, None), (None, None)], [None, None]
        else: # 0 or >2 players
            return [(None, None), (None, None)], [(None, None), (None, None)], [None, None]


    def save_shot_image(self, frame, shot_info, shot_data):
        """ Saves an image of the shot with relevant information overlaid """
        # (Same as previous version - no changes needed here)
        try:
            shot_frame = frame.copy()
            racket_center, shuttle_center = None, None
            if shot_info.get('racket_box'):
                rb = shot_info['racket_box']
                if rb and len(rb) == 4: racket_center = ((rb[0]+rb[2])//2, (rb[1]+rb[3])//2)
            if shot_info.get('shuttle_box'):
                 sb = shot_info['shuttle_box']
                 if sb and len(sb) == 4 and all(isinstance(c,(int,float)) for c in sb): shuttle_center = ((int(sb[0])+int(sb[2]))//2, (int(sb[1])+int(sb[3]))//2)
            if racket_center and shuttle_center: cv2.line(shot_frame, racket_center, shuttle_center, (0,255,255), 2)

            player_hit_val = shot_data.get('player_who_hit', 0)
            player_hit_text = f"Player Hit: {player_hit_val}" if player_hit_val != 0 else "Player Hit: Unknown"
            info_text = [f"Rally: {shot_data.get('rally_id','X')}", f"Shot: {shot_data.get('shot_num','Y')}", player_hit_text, f"Frame: {shot_data.get('frame_number','N/A')}"]
            y_offset = 30
            for line in info_text: cv2.putText(shot_frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,200,255), 2); y_offset += 25

            filename = f"shot_r{shot_data.get('rally_id','X')}_s{shot_data.get('shot_num','Y')}.jpg"
            filepath = os.path.join(self.shots_output_dir, filename)
            if not cv2.imwrite(filepath, shot_frame): print(f"Warning: Failed to save shot image: {filepath}")

            player_box_to_crop = shot_info.get('player1_box') if player_hit_val == 1 else shot_info.get('player2_box')
            if player_box_to_crop:
                player_img = self.crop_player_image(frame, player_box_to_crop) # Crop from original frame
                if player_img is not None and player_img.size > 0:
                    p_filename = f"player_r{shot_data.get('rally_id','X')}_s{shot_data.get('shot_num','Y')}_p{player_hit_val}.jpg"
                    p_filepath = os.path.join(self.player_shots_dir, p_filename)
                    if not cv2.imwrite(p_filepath, player_img): print(f"Warning: Failed to save player crop: {p_filepath}")
        except Exception as e:
             print(f"ERROR saving shot image for r{shot_data.get('rally_id','X')}_s{shot_data.get('shot_num','Y')}: {e}\n{traceback.format_exc()}")

    def find_closest_interaction(self, frame_idx_window, shuttle_center_img, detections_list, player_centers_list, interaction_thresh, player_racket_thresh):
        """
        Helper to find the best shuttle-racket-player interaction within a window.
        (Same as provided two steps ago)
        """
        best_interaction_player = 0
        min_overall_dist = float('inf')
        best_frame_idx_interaction = -1

        if not shuttle_center_img: return 0, -1, float('inf')

        for relative_idx, frame_idx in enumerate(frame_idx_window):
            if not (0 <= frame_idx < len(detections_list)): continue

            current_detections = detections_list[frame_idx]
            # Ensure player_centers_list has data for this index
            if relative_idx >= len(player_centers_list): continue
            p1_ctr, p2_ctr = player_centers_list[relative_idx]

            rackets = [d for d in current_detections if d.get('class_id') == self.racket_class_id]
            if not rackets: continue

            closest_racket_in_frame = None
            min_dist_sh_ra_frame = float('inf')

            for r in rackets:
                # Check if 'box' exists and has 4 elements
                if not isinstance(r.get('box'), list) or len(r['box']) != 4: continue
                try:
                    r_center = ((r['box'][0] + r['box'][2]) // 2, (r['box'][1] + r['box'][3]) // 2)
                    dist_sh_ra = math.dist(shuttle_center_img, r_center)
                    if dist_sh_ra < interaction_thresh and dist_sh_ra < min_dist_sh_ra_frame:
                        min_dist_sh_ra_frame = dist_sh_ra
                        closest_racket_in_frame = {'center': r_center, 'box': r['box']}
                except (TypeError, ValueError, IndexError) as e:
                    print(f"  Warn: Error processing racket {r.get('box')} in interaction: {e}")
                    continue # Skip this racket

            if closest_racket_in_frame:
                r_center = closest_racket_in_frame['center']
                dist_ra_p1 = math.dist(r_center, p1_ctr) if p1_ctr else float('inf')
                dist_ra_p2 = math.dist(r_center, p2_ctr) if p2_ctr else float('inf')

                assigned_player_this_frame = 0
                player_dist = float('inf')
                if dist_ra_p1 < dist_ra_p2 and dist_ra_p1 < player_racket_thresh:
                    assigned_player_this_frame = 1; player_dist = dist_ra_p1
                elif dist_ra_p2 < dist_ra_p1 and dist_ra_p2 < player_racket_thresh:
                    assigned_player_this_frame = 2; player_dist = dist_ra_p2

                if assigned_player_this_frame != 0:
                    if min_dist_sh_ra_frame < min_overall_dist:
                         min_overall_dist = min_dist_sh_ra_frame
                         best_interaction_player = assigned_player_this_frame
                         best_frame_idx_interaction = frame_idx # Record frame where best interaction happened

        return best_interaction_player, best_frame_idx_interaction, min_overall_dist
    def process_video_with_shot_detection(self, save_output=True, output_path=None):
        """
        Process video, detect shots, assign players based on proximity.
        Includes fixes for print statements and adds distance debugging.
        """
        print("\nStarting video processing pipeline...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened(): print(f"FATAL ERROR: Could not open video file {self.video_path}"); return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_width == 0 or frame_height == 0: print(f"FATAL ERROR: Invalid frame dimensions: {frame_width}x{frame_height}"); cap.release(); return None
        if original_fps is None or original_fps <= 0: print(f"Warning: Invalid FPS ({original_fps}), defaulting to 30."); original_fps = 30.0
        else: original_fps = float(original_fps)
        effective_fps = original_fps / self.frame_skip
        print(f"Video Info: {frame_width}x{frame_height} @ {original_fps:.2f} FPS (Processing every {self.frame_skip} frame(s) -> Effective Rate: {effective_fps:.2f} FPS)")

        out = None
        if save_output:
            if output_path is None: output_path = f'shot_detection_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            try:
                out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
                if not out.isOpened(): print(f"ERROR: Failed to open VideoWriter for: {output_path}"); save_output = False; out = None
                else: print(f"Output video will be saved to: {output_path}")
            except Exception as e: print(f"ERROR initializing VideoWriter: {e}"); save_output = False; out = None

        frame_counter = 0
        processed_frame_count = 0
        processed_frame_idx_map = {}
        all_frames_processed = []
        all_shuttle_detections_processed = []
        all_frame_detections_processed = []

        print("\n--- Phase 1: Collecting frames and detections ---")
        self.shuttle_detector.reset()
        self.homography_matrix = None
        court_detection_done = False # Flag to do it only once

        while True:
            try: ret, frame = cap.read()
            except Exception as e: print(f"ERROR reading frame {frame_counter}: {e}"); break
            if not ret: print(f"\nFrame {frame_counter}: Video ended or read failed."); break
            if frame is None: print(f"Frame {frame_counter}: Read successful but frame is None."); break

            current_original_frame = frame_counter
            frame_counter += 1
            if (current_original_frame % self.frame_skip) != 0: continue

            processed_frame_index = processed_frame_count
            processed_frame_count += 1
            processed_frame_idx_map[processed_frame_index] = current_original_frame
            frame_copy_for_storage = frame.copy()
            all_frames_processed.append(frame_copy_for_storage)

            # --- Court Detection & Homography (Once) ---
            if not court_detection_done:
                 # Try to detect court on the first few frames until successful
                 if self.homography_matrix is None and processed_frame_count <= 10: # Try for first 10 processed frames
                     temp_court_coords = self.court_detector.detect_court_boundary(frame)
                     if temp_court_coords and len(temp_court_coords) == 4:
                        self.court_coords = self.court_detector.sort_court_coords(temp_court_coords)
                        corners = self.court_detector.draw_court_lines(frame, self.court_coords) # Use helper to get corners without drawing
                        if corners:
                            # *** Print corners ONLY ONCE here ***
                            print("\n" + "="*15 + " Court Corner Detection " + "="*15)
                            print(f"Derived Image Corners for Homography (Frame {current_original_frame}):")
                            print(f"  Top Left: {corners[0]}")
                            print(f"  Top Right: {corners[1]}")
                            print(f"  Bottom Right: {corners[2]}")
                            print(f"  Bottom Left: {corners[3]}")
                            print("="*52)

                            self.homography_matrix = self.court_detector.compute_homography(corners)
                            if self.homography_matrix is not None:
                                print(f"INFO: Homography computed successfully at original frame {current_original_frame}!")
                                self.court_corners = corners # Store the corners used
                                court_detection_done = True # Stop trying
                            else:
                                print(f"WARNING: Homography computation failed at frame {current_original_frame}.")
                                self.court_corners = None
                        # else: print(f"WARNING: Failed to derive corners from sorted coords at frame {current_original_frame}.") # DEBUG
                 # If it's past the initial frames and still no homography, stop trying
                 elif processed_frame_count > 10 and self.homography_matrix is None:
                      print("INFO: Court/Homography not detected in initial frames. Proceeding without it.")
                      court_detection_done = True # Stop trying

            # --- Object Detection ---
            try:
                results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
            except Exception as e:
                print(f"ERROR during YOLO inference frame {current_original_frame}: {e}\n{traceback.format_exc()}")
                all_shuttle_detections_processed.append(None); all_frame_detections_processed.append([]); continue

            current_frame_detections = []
            current_shuttle_candidates = []
            if results.boxes is not None and results.boxes.data is not None:
                for detection in results.boxes.data:
                    try:
                        x1, y1, x2, y2, confidence, class_id_tensor = detection.cpu().split([1,1,1,1,1,1], dim=0) # Split tensor
                        class_id = int(class_id_tensor.item()) # Get Python int from tensor
                        label = self.class_labels.get(class_id, "Unknown")
                        if class_id not in [self.player_class_id, self.racket_class_id, self.shuttle_class_id]: continue # Optional: Filter unwanted classes early

                        detection_info = {
                            'box': [int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())],
                            'confidence': float(confidence.item()),
                            'class_id': class_id, 'label': label
                        }
                        current_frame_detections.append(detection_info)
                        if class_id == self.shuttle_class_id: current_shuttle_candidates.append(detection_info)
                    except Exception as e: print(f"Error processing detection data {detection}: {e}")

            best_shuttle = self.shuttle_detector.select_best_shuttle_detection(current_shuttle_candidates)
            all_shuttle_detections_processed.append(best_shuttle)
            all_frame_detections_processed.append(current_frame_detections)

        # --- End Frame Reading Loop ---
        print("\nExited video processing loop.")
        cap.release(); print("Video capture released.")
        if not all_frames_processed: print("FATAL ERROR: No frames processed."); return None
        print(f"\nCollected {len(all_frames_processed)} frames total after skipping.")

        # --- Phase 2: Interpolation ---
        print("\n--- Phase 2: Interpolating shuttle positions ---")
        shuttle_boxes_for_interp = [d.get('box') if isinstance(d, dict) else None for d in all_shuttle_detections_processed]
        try:
            interpolated_positions = self.trajectory_analyzer.interpolate_shuttle_positions(shuttle_boxes_for_interp)
            print(f"Interpolation complete. Generated {len(interpolated_positions)} positions.")
            if len(interpolated_positions) != len(all_frames_processed): print(f"WARNING: Interpolated positions count ({len(interpolated_positions)}) != processed frames count ({len(all_frames_processed)}).")
        except Exception as e: print(f"ERROR during interpolation: {e}\n{traceback.format_exc()}"); return None

        # --- Phase 3: Shot Detection ---
        print("\n--- Phase 3: Detecting direction changes (shots) ---")
        try:
            shots_indices = self.trajectory_analyzer.detect_direction_changes(
                interpolated_positions, window_size=5, min_frames_between_changes=5,
                angle_change_threshold=40.0, velocity_threshold=3.0, max_frames_between_hits=120 )
            print(f"Shot detection complete. Found {len(shots_indices)} potential shot events.")
        except Exception as e: print(f"ERROR during shot detection: {e}\n{traceback.format_exc()}"); shots_indices = []

        # --- Phase 4: Creating dataset and assigning players (Interaction Priority) ---
        print("\n--- Phase 4: Creating dataset and assigning players (Interaction Priority) ---")
        print("-" * 70)
        shot_dataset = []
        self.current_rally_id = 0
        current_shot_num_in_rally = 0
        last_shot_processed_frame_index = -self.frames_between_rallies
        # Tunable parameters for interaction detection
        interaction_window_radius = 1 # Look at frame i-1, i, i+1
        interaction_threshold_pixels = 75 # Max distance shuttle-racket
        player_racket_threshold_pixels = 200 # Max distance racket-player
        player_shuttle_threshold_pixels = 300 # Max distance player-shuttle for fallback (Increased slightly)

        for shot_processed_idx in shots_indices:
             try:
                # --- Basic Setup & Rally Logic ---
                if not (0 <= shot_processed_idx < len(all_frames_processed)): continue
                original_frame_num = processed_frame_idx_map.get(shot_processed_idx, -1)
                if original_frame_num == -1: continue
                frame_at_shot = all_frames_processed[shot_processed_idx]

                # Get data for the window around the shot index
                window_indices = list(range(max(0, shot_processed_idx - interaction_window_radius),
                                            min(len(all_frames_processed), shot_processed_idx + interaction_window_radius + 1)))
                # Ensure window_indices are valid before proceeding
                if not window_indices: continue

                window_detections = [all_frame_detections_processed[i] for i in window_indices if 0 <= i < len(all_frame_detections_processed)]
                window_player_img_centers = []
                window_player_real_pos = []
                window_player_boxes = []
                for i, idx in enumerate(window_indices):
                    # Check if index is valid for detections list
                    if idx < len(all_frame_detections_processed):
                        dets = all_frame_detections_processed[idx]
                        p_real, p_img_ctr, p_box = self.get_player_positions(dets)
                        window_player_img_centers.append(p_img_ctr)
                        window_player_real_pos.append(p_real)
                        window_player_boxes.append(p_box)
                    else: # Append placeholders if index is out of bounds (shouldn't happen with check above, but safer)
                        window_player_img_centers.append(((None, None), (None, None)))
                        window_player_real_pos.append(((None, None), (None, None)))
                        window_player_boxes.append(((None, None), (None, None)))

                # Get primary data (center of window) safely
                center_window_index = window_indices.index(shot_processed_idx) if shot_processed_idx in window_indices else -1
                if center_window_index == -1: continue # Cannot proceed if primary index isn't in window

                primary_p1_real, primary_p2_real = window_player_real_pos[center_window_index]
                primary_p1_img_ctr, primary_p2_img_ctr = window_player_img_centers[center_window_index]
                primary_p1_box, primary_p2_box = window_player_boxes[center_window_index]

                # Rally Logic
                if shot_processed_idx - last_shot_processed_frame_index > self.frames_between_rallies:
                    self.current_rally_id += 1; current_shot_num_in_rally = 0
                    print(f"\n--- New Rally (# {self.current_rally_id}) near frame {original_frame_num} ---")
                current_shot_num_in_rally += 1
                last_shot_processed_frame_index = shot_processed_idx

                print(f"\n[Shot Analysis] Rally: {self.current_rally_id}, Shot: {current_shot_num_in_rally}, Orig Frame: {original_frame_num} (Checking window: {window_indices})")

                # Shuttle Position at primary index
                shuttle_box_at_hit = interpolated_positions[shot_processed_idx]
                shuttle_center_img = None
                if shuttle_box_at_hit and all(c is not None for c in shuttle_box_at_hit):
                    x1,y1,x2,y2 = map(int, shuttle_box_at_hit)
                    shuttle_center_img = ((x1+x2)//2, (y1+y2)//2)
                print(f"  Shuttle Pos (Image @ {shot_processed_idx}): {shuttle_center_img if shuttle_center_img else 'N/A'}")

                # --- Assignment Logic ---
                player_who_hit = 0
                assignment_method = "Unknown"

                # 1. Attempt Interaction-Based Assignment
                interaction_player, interaction_frame, interaction_dist = self.find_closest_interaction(
                    window_indices, shuttle_center_img, all_frame_detections_processed, window_player_img_centers,
                    interaction_threshold_pixels, player_racket_threshold_pixels
                )

                if interaction_player != 0:
                    player_who_hit = interaction_player
                    assignment_method = f"Interaction @ frame {processed_frame_idx_map.get(interaction_frame, interaction_frame)} (Dist: {interaction_dist:.1f}px)"
                    print(f"  Interaction Found: Player {player_who_hit} assigned.")
                else:
                    # 2. Fallback: Player Proximity to Shuttle at original shot frame
                    print(f"  No clear interaction found. Falling back to Player-Shuttle proximity @ frame {shot_processed_idx}.")
                    assignment_method = "Fallback Proximity"
                    if shuttle_center_img:
                        dist1_sh_p_img, dist2_sh_p_img = float('inf'), float('inf')
                        valid_p1_img = primary_p1_img_ctr and isinstance(primary_p1_img_ctr, tuple) and len(primary_p1_img_ctr) == 2
                        valid_p2_img = primary_p2_img_ctr and isinstance(primary_p2_img_ctr, tuple) and len(primary_p2_img_ctr) == 2
                        dist1_str, dist2_str = "N/A", "N/A"
                        try:
                            if valid_p1_img: dist1_sh_p_img = math.dist(shuttle_center_img, primary_p1_img_ctr); dist1_str = f"{dist1_sh_p_img:.1f}px"
                            if valid_p2_img: dist2_sh_p_img = math.dist(shuttle_center_img, primary_p2_img_ctr); dist2_str = f"{dist2_sh_p_img:.1f}px"
                            print(f"  Player-Shuttle Dist Check: P1={dist1_str}, P2={dist2_str}")

                            p1_is_closer = dist1_sh_p_img < dist2_sh_p_img
                            p2_is_closer = dist2_sh_p_img < dist1_sh_p_img

                            if p1_is_closer and dist1_sh_p_img < player_shuttle_threshold_pixels:
                                player_who_hit = 1
                                assignment_method += f" (P1 Close {dist1_str})"
                            elif p2_is_closer and dist2_sh_p_img < player_shuttle_threshold_pixels:
                                player_who_hit = 2
                                assignment_method += f" (P2 Close {dist2_str})"
                            else:
                                print(f"  Fallback failed: No player close enough to shuttle (Threshold: {player_shuttle_threshold_pixels}px).")
                                assignment_method += " (No Player Close)"

                        except (TypeError, ValueError) as e: print(f"    ERROR calculating fallback distance: {e}")
                    else:
                        print("  Fallback failed: Missing shuttle image coords.")
                        assignment_method += " (No Shuttle)"

                print(f"  >>> Assigned Player: {player_who_hit if player_who_hit != 0 else 'Unknown'} (Method: {assignment_method})")
                primary_frame_detections = all_frame_detections_processed[shot_processed_idx]
                # --- Prepare & Store Data (Same as before, ensuring safe coord formatting) ---
                racket_box = next((d.get('box') for d in primary_frame_detections if d.get('class_id') == self.racket_class_id), None)                
                shot_played = "Unknown"
                shot_info_for_img = {'shuttle_box': shuttle_box_at_hit, 'player1_box': primary_p1_box, 'player2_box': primary_p2_box, 'racket_box': racket_box}

                shuttle_real_world = None
                if shuttle_center_img and self.homography_matrix is not None:
                    temp_shuttle_real = self.court_detector.translate_to_real_world(shuttle_center_img, self.homography_matrix)
                    if temp_shuttle_real is not None and isinstance(temp_shuttle_real, (np.ndarray, list)) and len(temp_shuttle_real) == 2 and not np.isnan(temp_shuttle_real).any():
                        shuttle_real_world = tuple(temp_shuttle_real)

                player_hit_coords = primary_p1_real if player_who_hit == 1 else primary_p2_real

                def format_coords(coords):
                    if coords and isinstance(coords, tuple) and len(coords)==2 and all(isinstance(c, (int, float)) for c in coords): return f"({coords[0]:.4f}, {coords[1]:.4f})"
                    return None

                shot_entry = {
                    'rally_id': self.current_rally_id, 'shot_num': current_shot_num_in_rally,
                    'player_who_hit': player_who_hit,
                    'player1_coords': format_coords(primary_p1_real),
                    'player2_coords': format_coords(primary_p2_real),
                    'shot_played': shot_played,
                    'description': self.shot_detector.get_shot_description(shot_played, player_hit_coords, shuttle_real_world),
                    'frame_number': original_frame_num }

                self.save_shot_image(frame_at_shot, shot_info_for_img, shot_entry)
                shot_dataset.append(shot_entry)

             except Exception as e:
                 print(f"ERROR processing shot index {shot_processed_idx} (Frame ~{original_frame_num}): {e}\n{traceback.format_exc()}")

        print("\n" + "-" * 50) # End separator for debug prints
        print("Dataset creation loop finished.")

        # --- Phase 5: Output Video ---
        print("\n--- Phase 5: Creating output video ---")
        if save_output and out is not None:
            print(f"Writing {len(all_frames_processed)} processed frames to video...")
            processed_frame_write_counter = 0
            for frame_index, original_frame_num in processed_frame_idx_map.items():
                try:
                    frame_to_write = all_frames_processed[frame_index].copy()
                    interpolated_box = interpolated_positions[frame_index]

                    # Draw court lines using stored corners if available
                    if self.court_corners:
                        cv2.polylines(frame_to_write, [np.array(self.court_corners, dtype=np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)

                    # Draw Interpolated Shuttle Box
                    if interpolated_box and all(c is not None for c in interpolated_box):
                        x1, y1, x2, y2 = map(int, interpolated_box)
                        color = (0, 255, 0); label = "Shuttle"
                        if not isinstance(all_shuttle_detections_processed[frame_index], dict): label += " (Interp.)"
                        cv2.rectangle(frame_to_write, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame_to_write, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Annotate Shot Frames
                    if frame_index in shots_indices:
                        shot_data_list = [sd for sd in shot_dataset if sd['frame_number'] == original_frame_num]
                        if shot_data_list:
                            sd = shot_data_list[0]
                            text = f"SHOT! R{sd['rally_id']} S{sd['shot_num']}"
                            ts, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                            tx = (frame_width - ts[0]) // 2
                            cv2.putText(frame_to_write, text, (tx, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
                            ph = sd['player_who_hit']
                            pht = f"Player Hit: {ph}" if ph != 0 else "Player Hit: ?"
                            cv2.putText(frame_to_write, pht, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,200,255), 2)

                    # Frame Number
                    cv2.putText(frame_to_write, f'Frame: {original_frame_num}', (frame_width-150, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    out.write(frame_to_write)
                    processed_frame_write_counter += 1
                except IndexError: print(f"ERROR: Index out of bounds ({frame_index}) writing video. Skipping."); continue
                except Exception as e: print(f"ERROR writing frame {original_frame_num} to video: {e}\n{traceback.format_exc()}"); continue
            print(f"Finished writing {processed_frame_write_counter} frames.")
            out.release(); print("Output video writer released.")
        elif save_output: print("Video saving enabled but writer failed.")
        else: print("Video saving disabled.")

        # --- Final Summary & Return ---
        print("\n--- Final Summary ---")
        csv_path = 'badminton_shot_dataset.csv'
        save_success = False
        if not shot_dataset:
            print("WARNING: No shots recorded.")
            df = pd.DataFrame(columns=['rally_id', 'shot_num', 'player_who_hit', 'player1_coords', 'player2_coords', 'shot_played', 'description', 'frame_number'])
        else:
            df = pd.DataFrame(shot_dataset)
            print(f"Total shots recorded: {len(df)}")
            print(f"Rallies identified: {df['rally_id'].nunique() if not df.empty else 0}")
        try: df.to_csv(csv_path, index=False); print(f"Dataset saved to {csv_path}"); save_success = True
        except Exception as e: print(f"ERROR saving dataset CSV ({csv_path}): {e}")

        return {
            'shots_processed_indices': shots_indices,
            'shot_original_frame_numbers': [processed_frame_idx_map.get(i, -1) for i in shots_indices],
            'total_processed_frames': len(all_frames_processed),
            'dataset': shot_dataset,
            'dataset_path': csv_path if save_success else None,
            'video_output_path': output_path if save_output and out is not None else None }

    # Alias methods
    def process_video(self, save_output=True, output_path=None): return self.process_video_with_shot_detection(save_output, output_path)
    def process_video_with_direction_detection(self, save_output=True, output_path=None): return self.process_video_with_shot_detection(save_output, output_path)

# Need to update court.py as well to fix the print statement location