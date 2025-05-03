import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime
import os
from trackers.court import CourtDetector


class BadmintonAnalyzer:
    def __init__(self, shuttle_model_path, court_model_path, video_path, conf_threshold=0.3, frame_skip=1):
        """
        Initialize the BadmintonAnalyzer with model and video paths
        
        Args:
            shuttle_model_path (str): Path to trained YOLOv8 model weights for shuttle detection
            court_model_path (str): Path to trained YOLOv8 model weights for court detection
            video_path (str): Path to input video file
            conf_threshold (float): Confidence threshold for detections
            frame_skip (int): Number of frames to skip for faster processing
        """
        self.model = YOLO(shuttle_model_path)
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        
        # Initialize court detector
        self.court_detector = CourtDetector(
            conf_threshold=conf_threshold,
            model_path=court_model_path
        )
        self.court_coords = None
        self.court_corners = None
        self.homography_matrix = None
        
        self.class_labels = {0: "Player", 1: "Racket", 2: "Shuttle"}
        
        # Shuttle tracking parameters
        self.prev_shuttle_pos = None
        self.max_shuttle_movement = 400  # Maximum pixel distance shuttle can move between frames
        self.min_shuttle_movement = 10   # Minimum movement to consider shuttle as "moving"
        
        # Rally tracking parameters
        self.current_rally_id = 1
        self.current_shot_num = 0
        self.last_shot_frame = 0
        self.frames_between_rallies = 150
        
        self.direction_change_shots = []
        self.rally_timeout_frames = 150  # Frames without shots to consider a new rally
        
        # Output directories
        self.shots_output_dir = "shot_images"
        os.makedirs(self.shots_output_dir, exist_ok=True)
        
        # Shot data collection
        self.shot_data = []
        
    def get_bbox_center(self, box):
        """Calculate the center point of a bounding box"""
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def select_best_shuttle_detection(self, shuttle_detections):
        """
        Select the most likely shuttle detection based on movement from previous frame
        
        Args:
            shuttle_detections: List of dictionaries containing shuttle detections
                              Each detection should have 'box' and 'confidence' keys
        
        Returns:
            Best shuttle detection or None if no valid detection found
        """
        if not shuttle_detections:
            return None
            
        # If first detection, choose highest confidence detection
        if self.prev_shuttle_pos is None:
            best_detection = max(shuttle_detections, key=lambda x: x['confidence'])
            self.prev_shuttle_pos = self.get_bbox_center(best_detection['box'])
            return best_detection
            
        moving_shuttles = []
        for detection in shuttle_detections:
            current_center = self.get_bbox_center(detection['box'])
            distance = self.calculate_distance(current_center, self.prev_shuttle_pos)
            
            # Skip if movement is impossibly large
            if distance > self.max_shuttle_movement:
                continue
                
            # Only consider shuttles that have moved more than minimum threshold
            if distance > self.min_shuttle_movement:
                moving_shuttles.append({
                    'detection': detection,
                    'distance': distance,
                    'center': current_center
                })
        
        # If we found moving shuttles, pick the one with highest confidence
        if moving_shuttles:
            best_candidate = max(moving_shuttles, 
                               key=lambda x: x['detection']['confidence'])
            self.prev_shuttle_pos = best_candidate['center']
            return best_candidate['detection']
        
        # If no moving shuttles found, fallback to highest confidence detection
        best_detection = max(shuttle_detections, key=lambda x: x['confidence'])
        self.prev_shuttle_pos = self.get_bbox_center(best_detection['box'])
        return best_detection

    def interpolate_shuttle_positions(self, shuttle_detections):
        """
        Interpolate missing shuttle positions
        
        Args:
            shuttle_detections: List of dictionaries with shuttle detections or None if no detection
        
        Returns:
            List of interpolated shuttle positions (each being [x1, y1, x2, y2] or None)
        """
        # Extract only the box coordinates
        box_positions = []
        for detection in shuttle_detections:
            if detection and 'box' in detection:
                # If detection exists, add its box coordinates
                box_positions.append(detection['box'])
            else:
                # If no detection, add None values for each coordinate
                box_positions.append([None, None, None, None])
        
        # Convert to DataFrame
        df_positions = pd.DataFrame(box_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_positions = df_positions.interpolate(method='linear')
        
        # Backfill to handle edge case at the beginning
        df_positions = df_positions.bfill()
        
        # Convert back to list of box coordinates
        interpolated_positions = df_positions.values.tolist()
        
        return interpolated_positions

    def detect_direction_changes(self, shuttle_positions, window_size=3, min_frames_between_changes=2, y_change_threshold=4.0):
        """
        Detect changes in shuttle's vertical (y) direction only, with adjustable sensitivity
        
        Args:
            shuttle_positions: List of interpolated shuttle positions [x1, y1, x2, y2]
            window_size: Window size for rolling mean to smooth trajectory (smaller = more sensitive)
            min_frames_between_changes: Minimum frames to wait before detecting another direction change
            y_change_threshold: Minimum absolute change in y direction to consider significant (in pixels)
        
        Returns:
            List of frame indices where direction changes occur
        """
        # Convert to DataFrame
        df_positions = pd.DataFrame(shuttle_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Calculate center of shuttle
        df_positions['center_y'] = (df_positions['y1'] + df_positions['y2']) / 2
        
        # Calculate rolling mean with a smaller window to smooth the trajectory but remain sensitive
        df_positions['y_rolling'] = df_positions['center_y'].rolling(window=window_size, min_periods=1, center=True).mean()
        
        # Calculate delta (change in vertical position)
        df_positions['delta_y'] = df_positions['y_rolling'].diff()
        
        # Mark potential direction changes
        df_positions['dir_change'] = 0
        
        # Look for vertical direction changes that exceed the threshold
        for i in range(1, len(df_positions) - 1):
            # Only consider changes that are significant enough
            if abs(df_positions['delta_y'].iloc[i]) >= y_change_threshold:
                # Check for sign changes in vertical movement
                if (df_positions['delta_y'].iloc[i-1] > 0 and df_positions['delta_y'].iloc[i] < 0) or \
                (df_positions['delta_y'].iloc[i-1] < 0 and df_positions['delta_y'].iloc[i] > 0):
                    # Mark as direction change
                    df_positions.loc[i, 'dir_change'] = 1
        
        # Find frames with direction changes
        potential_changes = df_positions[df_positions['dir_change'] == 1].index.tolist()
        
        # Filter out changes that are too close together
        direction_changes = []
        last_change = -min_frames_between_changes  # Initialize to allow first change
        
        for change in potential_changes:
            if change - last_change >= min_frames_between_changes:
                direction_changes.append(change)
                last_change = change
        
        return direction_changes

    def determine_player(self, player_positions):
        """
        Determine which player is at which side of the court
        Returns player1_pos (back court), player2_pos (front court)
        """
        if not player_positions or len(player_positions) < 2:
            return None, None
            
        # Sort by y-coordinate (court depth)
        sorted_players = sorted(player_positions, key=lambda pos: pos[1], reverse=True)
        return sorted_players[0], sorted_players[1]  # Return (player1_pos, player2_pos)

    def get_player_positions(self, frame_detections):
        """Get player positions in real-world coordinates"""
        if not self.homography_matrix is not None:
            return None, None
            
        player_positions = []
        
        for detection in frame_detections:
            if detection['label'] == 'Player':
                # Get center of player bounding box
                box = detection['box']
                center = (
                    (box[0] + box[2]) // 2,
                    box[3]  # Use bottom of bounding box for feet position
                )
                
                # Convert to real-world coordinates
                try:
                    real_world_pos = self.court_detector.translate_to_real_world(
                        center, 
                        self.homography_matrix
                    )
                    player_positions.append(real_world_pos)
                except:
                    continue
        
        return self.determine_player(player_positions)

    def get_racket_info(self, frame_detections):
        """
        Extract racket information including height and position
        Returns a list of racket information dictionaries
        """
        if not self.homography_matrix:
            return []
            
        racket_info = []
        
        for detection in frame_detections:
            if detection['label'] == 'Racket':
                box = detection['box']
                
                # Calculate racket height in pixels
                racket_height_px = box[3] - box[1]
                
                # Get top and bottom of racket in pixels
                racket_top = (int((box[0] + box[2]) / 2), box[1])
                racket_bottom = (int((box[0] + box[2]) / 2), box[3])
                
                # Convert to real-world coordinates
                try:
                    top_real = self.court_detector.translate_to_real_world(racket_top, self.homography_matrix)
                    bottom_real = self.court_detector.translate_to_real_world(racket_bottom, self.homography_matrix)
                    
                    # Calculate real-world height
                    racket_height_real = np.sqrt((top_real[0] - bottom_real[0])**2 + 
                                                (top_real[1] - bottom_real[1])**2)
                    
                    # Get racket center
                    center = (
                        (box[0] + box[2]) // 2,
                        (box[1] + box[3]) // 2
                    )
                    center_real = self.court_detector.translate_to_real_world(center, self.homography_matrix)
                    
                    racket_info.append({
                        'box': box,
                        'height_px': racket_height_px,
                        'height_real': racket_height_real,
                        'position_real': center_real,
                        'confidence': detection['confidence']
                    })
                except:
                    continue
                    
        return racket_info

    def save_shot_image(self, frame, frame_idx, shot_data, shuttle_box=None, racket_boxes=None):
        """Save an image of the shot with relevant information overlaid"""
        # Create a copy of the frame to draw on
        shot_frame = frame.copy()
        
        # Draw shot connection lines if we have both shuttle and rackets
        if shuttle_box and racket_boxes:
            shuttle_center = (
                (shuttle_box[0] + shuttle_box[2]) // 2,
                (shuttle_box[1] + shuttle_box[3]) // 2
            )
            
            # Draw lines from each racket to shuttle
            for racket_box in racket_boxes:
                racket_center = (
                    (racket_box[0] + racket_box[2]) // 2,
                    (racket_box[1] + racket_box[3]) // 2
                )
                cv2.line(shot_frame, racket_center, shuttle_center, (0, 255, 255), 2)
        
        # Add shot information overlay
        info_text = [
            f"Rally: {shot_data['rally_id']}",
            f"Shot: {shot_data['shot_num']}",
            f"Frame: {frame_idx}",
            f"Player: {shot_data['player_who_hit'] if 'player_who_hit' in shot_data else 'Unknown'}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(shot_frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 25
            
        # Draw the court lines if detected
        if self.court_coords is not None:
            self.court_detector.draw_court_lines(shot_frame, self.court_coords)
            
        # Save the image
        filename = f"shot_r{shot_data['rally_id']}_s{shot_data['shot_num']}.jpg"
        cv2.imwrite(os.path.join(self.shots_output_dir, filename), shot_frame)

    def analyze_video(self, save_output=True, output_path=None):
        """
        Process video with shot detection based on shuttle direction changes
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        effective_fps = original_fps // self.frame_skip
        
        # Adjust frames_between_rallies based on video FPS
        self.frames_between_rallies = int(5 * effective_fps)
        
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'shot_detection_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
        
        # Phase 1: Collect all frames and detections
        print("Phase 1: Collecting frames and detections...")
        all_frames = []
        all_frame_detections = []  # All detections by frame
        all_shuttle_detections = []  # Only shuttle detections
        all_racket_info = []  # Racket information by frame
        processed_frame_indices = []
        
        frame_count = 0
        court_detected = False
        
        # First pass: collect all frames and detections
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
            
            processed_frame_indices.append(frame_count)
            all_frames.append(frame.copy())
            
            # Detect court for the first time
            if not court_detected:
                self.court_coords = self.court_detector.detect_court_boundary(frame)
                if self.court_coords is not None:
                    self.court_coords = self.court_detector.sort_court_coords(self.court_coords)
                    self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
                    self.homography_matrix = self.court_detector.compute_homography(self.court_corners)
                    court_detected = True
            
            # Run object detection
            results = self.model(frame, conf=self.conf_threshold)[0]
            
            # Gather all detections for this frame
            frame_detections = []
            shuttle_detections = []
            
            for detection in results.boxes.data:
                x1, y1, x2, y2, confidence, class_id = detection
                box = [int(x1), int(y1), int(x2), int(y2)]
                class_id = int(class_id)
                confidence = float(confidence)
                
                detection_dict = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / effective_fps,
                    'box': box,
                    'confidence': confidence,
                    'label': self.class_labels.get(class_id, "Unknown")
                }
                
                frame_detections.append(detection_dict)
                
                # Collect shuttle detections separately
                if class_id == 2:  # Shuttle
                    shuttle_detections.append(detection_dict)
            
            # Get best shuttle detection
            best_shuttle = self.select_best_shuttle_detection(shuttle_detections)
            
            # Store detections for this frame
            all_frame_detections.append(frame_detections)
            all_shuttle_detections.append(best_shuttle)
            
            # Collect racket information
            racket_info = self.get_racket_info(frame_detections)
            all_racket_info.append(racket_info)
            
            frame_count += 1
            
            # Display progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        print(f"Collected {len(all_frames)} frames and detections")
        
        # Phase 2: Interpolate shuttle positions
        print("Phase 2: Interpolating shuttle positions...")
        interpolated_shuttle_positions = self.interpolate_shuttle_positions(all_shuttle_detections)
        
        # Phase 3: Detect direction changes (shots)
        print("Phase 3: Detecting direction changes (shots)...")
        shot_frames = self.detect_direction_changes(
            interpolated_shuttle_positions, 
            window_size=3,
            min_frames_between_changes=2,
            y_change_threshold=4.0
        )
        
        print(f"Detected {len(shot_frames)} shots through direction changes")
        
        # Phase 4: Process shot data and create dataset
        print("Phase 4: Processing shot data and creating dataset...")
        
        last_shot_frame_idx = -1
        for i, shot_frame_idx in enumerate(shot_frames):
            real_frame_idx = processed_frame_indices[shot_frame_idx]
            
            # Check if this is a new rally
            frames_since_last_shot = real_frame_idx - last_shot_frame_idx if last_shot_frame_idx != -1 else 0
            
            if frames_since_last_shot > self.frames_between_rallies or last_shot_frame_idx == -1:
                # Start new rally
                self.current_rally_id += 1
                self.current_shot_num = 0
                print(f"\nNew Rally {self.current_rally_id} started at frame {real_frame_idx}")
            
            # Increment shot number
            self.current_shot_num += 1
            
            # Get player positions
            player1_pos, player2_pos = self.get_player_positions(all_frame_detections[shot_frame_idx])
            
            # Determine which player hit
            player_who_hit = None
            if player1_pos is not None and player2_pos is not None:
                # Get racket positions for this frame
                racket_info = all_racket_info[shot_frame_idx]
                
                if racket_info:
                    # Find closest racket to each player
                    min_dist_player1 = float('inf')
                    min_dist_player2 = float('inf')
                    
                    for racket in racket_info:
                        racket_pos = racket['position_real']
                        
                        # Calculate distances
                        dist_to_player1 = np.sqrt(
                            (racket_pos[0] - player1_pos[0])**2 + 
                            (racket_pos[1] - player1_pos[1])**2
                        )
                        dist_to_player2 = np.sqrt(
                            (racket_pos[0] - player2_pos[0])**2 + 
                            (racket_pos[1] - player2_pos[1])**2
                        )
                        
                        min_dist_player1 = min(min_dist_player1, dist_to_player1)
                        min_dist_player2 = min(min_dist_player2, dist_to_player2)
                    
                    # Apply slight bias for back court player
                    BACK_COURT_BIAS = 0.8
                    min_dist_player1 *= BACK_COURT_BIAS
                    
                    # Determine who hit the shot
                    player_who_hit = 1 if min_dist_player1 < min_dist_player2 else 2
            
            # Get shuttle position
            shuttle_box = interpolated_shuttle_positions[shot_frame_idx]
            
            # Extract racket boxes for this frame
            racket_boxes = [racket['box'] for racket in all_racket_info[shot_frame_idx]] if all_racket_info[shot_frame_idx] else []
            
            # Calculate racket heights (if available)
            racket_heights_px = [racket['height_px'] for racket in all_racket_info[shot_frame_idx]] if all_racket_info[shot_frame_idx] else []
            racket_heights_real = [racket['height_real'] for racket in all_racket_info[shot_frame_idx]] if all_racket_info[shot_frame_idx] else []
            avg_racket_height_px = np.mean(racket_heights_px) if racket_heights_px else None
            avg_racket_height_real = np.mean(racket_heights_real) if racket_heights_real else None
            
            # Create shot data entry
            shot_data_entry = {
                'rally_id': self.current_rally_id,
                'shot_num': self.current_shot_num,
                'frame_number': real_frame_idx,
                'timestamp': real_frame_idx / effective_fps,
                'frames_since_last_shot': frames_since_last_shot,
                'player_who_hit': player_who_hit,
                'player1_position': player1_pos.tolist() if player1_pos is not None else None,
                'player2_position': player2_pos.tolist() if player2_pos is not None else None,
                'avg_racket_height_px': avg_racket_height_px,
                'avg_racket_height_real': avg_racket_height_real,
                'shuttle_box': shuttle_box,
                'shuttle_position': self.get_bbox_center(shuttle_box) if shuttle_box else None
            }
            
            # Save the shot image
            self.save_shot_image(all_frames[shot_frame_idx], real_frame_idx, shot_data_entry, shuttle_box, racket_boxes)
            
            # Add to shot data list
            self.shot_data.append(shot_data_entry)
            
            # Update last shot frame
            last_shot_frame_idx = real_frame_idx
        
        # Phase 5: Annotate and save output video
        print("Phase 5: Annotating and saving output video...")
        
        # Reopen video for the second pass
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        
        # Process frames and save video
        while cap.isOpened() and save_output:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
            
            # Get the index in our processed frames
            try:
                frame_idx = processed_frame_indices.index(frame_count)
            except ValueError:
                frame_count += 1
                continue
            
            # Draw court lines if detected
            if court_detected:
                self.court_detector.draw_court_lines(frame, self.court_coords)
            
            # Check if this frame is a shot (direction change)
            is_shot = frame_idx in shot_frames
            
            # Draw shuttle box
            if frame_idx < len(interpolated_shuttle_positions):
                box = interpolated_shuttle_positions[frame_idx]
                if box and None not in box:
                    x1, y1, x2, y2 = map(int, box)
                    # Draw shuttle in white
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Add text for shots
            if is_shot:
                shot_idx = shot_frames.index(frame_idx)
                rally_id = self.shot_data[shot_idx]['rally_id']
                shot_num = self.shot_data[shot_idx]['shot_num']
                player = self.shot_data[shot_idx].get('player_who_hit', 'Unknown')
                
                # Draw "SHOT" text
                cv2.putText(frame, 'DIRECTION CHANGE (SHOT)', (10, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add rally and shot info
                cv2.putText(frame, f'Rally: {rally_id} Shot: {shot_num} Player: {player}', 
                          (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add frame number
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save the frame to video
            out.write(frame)
            
            # Show the frame
            cv2.imshow('Badminton Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
        # Clean up
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        # Convert shot data to DataFrame and save to CSV
        shot_df = pd.DataFrame(self.shot_data)
        csv_filename = f"shot_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shot_df.to_csv(csv_filename, index=False)
        
        print(f"Shot data saved to {csv_filename}")
        
        return shot_df
    def create_shot_dataset(self, save_output=True, output_path=None):
        """
        Process video with direction change detection and create a comprehensive shot dataset
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        effective_fps = original_fps // self.frame_skip
        
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'shot_detection_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
        
        # Directory for saving shot images
        shot_dir = "shot_images"
        os.makedirs(shot_dir, exist_ok=True)
        
        frame_count = 0
        court_detected = False
        all_frames = []
        all_detections = []  # Store all detections for each frame
        all_shuttle_detections = []
        processed_frame_indices = []
        
        print("Phase 1: Collecting frames and detections...")
        
        # First pass: collect all frames and detections
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
            
            all_frames.append(frame.copy())
            processed_frame_indices.append(frame_count)
            
            if not court_detected:
                self.court_coords = self.court_detector.detect_court_boundary(frame)
                if self.court_coords is not None:
                    self.court_coords = self.court_detector.sort_court_coords(self.court_coords)
                    self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
                    self.homography_matrix = self.court_detector.compute_homography(self.court_corners)
                    court_detected = True
            
            # Detect all objects (players, rackets, shuttle)
            results = self.model(frame, conf=self.conf_threshold)[0]
            frame_detections = []
            shuttle_detections = []
            
            for detection in results.boxes.data:
                x1, y1, x2, y2, confidence, class_id = detection
                class_id = int(class_id)
                box = [int(x1), int(y1), int(x2), int(y2)]
                
                detection_dict = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / effective_fps,
                    'box': box,
                    'confidence': float(confidence),
                    'label': self.class_labels.get(class_id, "Unknown")
                }
                
                # Add to appropriate collection
                if class_id == 2:  # Shuttle
                    shuttle_detections.append(detection_dict)
                else:
                    frame_detections.append(detection_dict)
            
            # Get best shuttle detection
            best_shuttle = self.select_best_shuttle_detection(shuttle_detections)
            if best_shuttle:
                frame_detections.append(best_shuttle)
            
            all_shuttle_detections.append(best_shuttle)
            all_detections.append(frame_detections)
            
            frame_count += 1
        
        cap.release()
        
        print(f"Collected {len(all_frames)} frames and detections")
        print("Phase 2: Interpolating shuttle positions...")
        
        # Interpolate shuttle positions
        interpolated_positions = self.interpolate_shuttle_positions(all_shuttle_detections)
        
        print("Phase 3: Detecting direction changes...")
        
        # Detect direction changes
        direction_changes = self.detect_direction_changes(
            interpolated_positions, 
            window_size=3,
            min_frames_between_changes=5  # Adjust sensitivity as needed
        )
        
        print(f"Detected {len(direction_changes)} direction changes")
        print("Phase 4: Creating shot dataset...")
        
        # Initialize variables for rally tracking
        current_rally_id = 1
        last_shot_frame_idx = -self.rally_timeout_frames  # Initialize to allow first shot
        
        # Process each direction change as a shot
        for shot_idx, frame_idx in enumerate(direction_changes):
            actual_frame_number = processed_frame_indices[frame_idx]
            
            # Check if we need to start a new rally
            frames_since_last_shot = frame_idx - last_shot_frame_idx
            if frames_since_last_shot > self.rally_timeout_frames:
                current_rally_id += 1
                print(f"New rally {current_rally_id} starting at frame {actual_frame_number}")
            
            # Get the frame and its detections
            frame = all_frames[frame_idx]
            frame_detections = all_detections[frame_idx]
            
            # Get player positions (using homography to translate to court coordinates)
            player1_pos, player2_pos = self.get_player_positions(frame_detections)
            
            # Get shuttle position
            shuttle_box = interpolated_positions[frame_idx]
            shuttle_center = None
            if shuttle_box and None not in shuttle_box:
                shuttle_center = (
                    (shuttle_box[0] + shuttle_box[2]) // 2,
                    (shuttle_box[1] + shuttle_box[3]) // 2
                )
            
            # Find closest racket to determine which player hit the shot
            player_who_hit = None
            closest_racket = None
            min_distance = float('inf')
            
            # Check for rackets in nearby frames (small window to detect the hit)
            window_size = 3  # Look 3 frames before and after
            window_start = max(0, frame_idx - window_size)
            window_end = min(len(all_detections), frame_idx + window_size + 1)
            
            for i in range(window_start, window_end):
                for detection in all_detections[i]:
                    if detection['label'] == 'Racket' and shuttle_center:
                        racket_box = detection['box']
                        racket_center = (
                            (racket_box[0] + racket_box[2]) // 2,
                            (racket_box[1] + racket_box[3]) // 2
                        )
                        
                        # Calculate distance from racket to shuttle
                        distance = self.calculate_distance(racket_center, shuttle_center)
                        
                        # If this is the closest racket so far, store it
                        if distance < min_distance:
                            min_distance = distance
                            closest_racket = racket_box
            
            # Based on closest racket, determine which player hit the shot
            if closest_racket and player1_pos is not None and player2_pos is not None:
                # Convert racket to real-world coordinates
                racket_center = (
                    (closest_racket[0] + closest_racket[2]) // 2,
                    (closest_racket[1] + closest_racket[3]) // 2
                )
                racket_real_world = self.court_detector.translate_to_real_world(
                    racket_center, 
                    self.homography_matrix
                )
                
                # Calculate distances from racket to both players
                dist_to_player1 = np.sqrt(
                    (racket_real_world[0] - player1_pos[0]) ** 2 + 
                    (racket_real_world[1] - player1_pos[1]) ** 2
                )
                dist_to_player2 = np.sqrt(
                    (racket_real_world[0] - player2_pos[0]) ** 2 + 
                    (racket_real_world[1] - player2_pos[1]) ** 2
                )
                
                # Determine which player is closest to the racket
                player_who_hit = 1 if dist_to_player1 < dist_to_player2 else 2
            
            # Determine shot type based on shuttle trajectory and position
            shot_type = self.determine_shot_type(
                interpolated_positions, 
                frame_idx, 
                window_size=5
            )
            
            # Create shot data entry
            shot_data = {
                'rally_id': current_rally_id,
                'shot_id': shot_idx + 1,
                'frame_number': actual_frame_number,
                'timestamp': actual_frame_number / effective_fps,
                'player_who_hit': player_who_hit,
                'shot_type': shot_type,
                'player1_position': player1_pos.tolist() if player1_pos is not None else None,
                'player2_position': player2_pos.tolist() if player2_pos is not None else None,
                'shuttle_position': shuttle_box,
                'racket_position': closest_racket,
                'frames_since_last_shot': frames_since_last_shot
            }
            
            # Add to shot dataset
            self.direction_change_shots.append(shot_data)
            
            # Update last shot frame
            last_shot_frame_idx = frame_idx
            
            # Visualize and save the shot frame
            annotated_frame = self.visualize_shot(
                frame.copy(), 
                shot_data,
                shuttle_box,
                closest_racket
            )
            
            # Save the shot image
            shot_filename = f"shot_r{current_rally_id}_s{shot_idx+1}.jpg"
            cv2.imwrite(os.path.join(shot_dir, shot_filename), annotated_frame)
            
            # Add to output video
            if save_output:
                out.write(annotated_frame)
        
        print("Phase 5: Creating output video with all frames...")
        
        # Create full video with all frames and annotations
        if save_output:
            # Reset video writer to start from beginning
            out.release()
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
            
            # Process all frames with annotations
            for i, frame in enumerate(all_frames):
                frame_copy = frame.copy()
                
                # Draw court lines
                if court_detected:
                    self.court_corners = self.court_detector.draw_court_lines(frame_copy, self.court_coords)
                
                # Draw shuttle position
                box = interpolated_positions[i]
                if box and None not in box:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Check if this frame is a shot
                is_shot = i in direction_changes
                
                if is_shot:
                    shot_idx = direction_changes.index(i)
                    shot_data = self.direction_change_shots[shot_idx]
                    
                    # Get shot number and rally
                    rally_id = shot_data['rally_id']
                    shot_id = shot_data['shot_id']
                    
                    # Add shot indicator
                    cv2.putText(frame_copy, f"SHOT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame_copy, f"Rally: {rally_id}, Shot: {shot_id}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Add player who hit
                    if shot_data['player_who_hit']:
                        cv2.putText(frame_copy, f"Player: {shot_data['player_who_hit']}", (10, 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Add shot type
                    if shot_data['shot_type']:
                        cv2.putText(frame_copy, f"Type: {shot_data['shot_type']}", (10, 120),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Add frame number
                cv2.putText(frame_copy, f'Frame: {processed_frame_indices[i]}', (frame_width - 200, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add to output video
                out.write(frame_copy)
        
        if save_output:
            out.release()
        
        # Save shot dataset to CSV
        self.save_shot_dataset()
        
        return self.direction_change_shots
    
    def determine_shot_type(self, shuttle_positions, frame_idx, window_size=5):
        """
        Determine shot type based on shuttle trajectory and position
        
        Args:
            shuttle_positions: List of shuttle box coordinates
            frame_idx: Current frame index
            window_size: Number of frames to analyze before and after
            
        Returns:
            Shot type (string)
        """
        # Get window of positions around current frame
        start_idx = max(0, frame_idx - window_size)
        end_idx = min(len(shuttle_positions), frame_idx + window_size)
        
        # Get positions before and after the shot
        positions_before = shuttle_positions[start_idx:frame_idx]
        positions_after = shuttle_positions[frame_idx+1:end_idx]
        
        # Filter out None values
        positions_before = [p for p in positions_before if p and None not in p]
        positions_after = [p for p in positions_after if p and None not in p]
        
        # If not enough data, return generic shot
        if not positions_before or not positions_after:
            return "Unknown"
        
        # Calculate center points
        centers_before = [(p[0] + p[2]) / 2 for p in positions_before]
        centers_after = [(p[0] + p[2]) / 2 for p in positions_after]
        
        # Calculate average horizontal position before and after
        avg_x_before = sum(centers_before) / len(centers_before)
        avg_x_after = sum(centers_after) / len(centers_after)
        
        # Calculate height differences
        heights_before = [(p[1] + p[3]) / 2 for p in positions_before]
        heights_after = [(p[1] + p[3]) / 2 for p in positions_after]
        
        avg_height_before = sum(heights_before) / len(heights_before)
        avg_height_after = sum(heights_after) / len(heights_after)
        
        # Calculate direction change
        x_direction_change = avg_x_after - avg_x_before
        y_direction_change = avg_height_after - avg_height_before
        
        # Determine shot type based on direction changes
        # These are simplified heuristics - you may want to refine them
        if y_direction_change < -20:  # Shuttle goes higher after shot
            if abs(x_direction_change) < 20:
                return "Clear"
            elif x_direction_change < -20:
                return "Cross Court Clear"
            else:
                return "Down the Line Clear"
        elif y_direction_change > 20:  # Shuttle goes lower after shot
            if abs(x_direction_change) < 20:
                return "Drop"
            elif x_direction_change < -20:
                return "Cross Court Drop"
            else:
                return "Down the Line Drop"
        elif abs(x_direction_change) > 40:  # Significant horizontal movement
            if x_direction_change < 0:
                return "Cross Court Drive"
            else:
                return "Down the Line Drive"
        else:
            return "Drive"
    
    def visualize_shot(self, frame, shot_data, shuttle_box, racket_box):
        """
        Create a visualization of the shot with annotations
        
        Args:
            frame: Original video frame
            shot_data: Dictionary containing shot information
            shuttle_box: Bounding box of shuttle
            racket_box: Bounding box of racket
            
        Returns:
            Annotated frame
        """
        # Draw shuttle box
        if shuttle_box and None not in shuttle_box:
            x1, y1, x2, y2 = map(int, shuttle_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw shuttle center
            shuttle_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, shuttle_center, 5, (0, 255, 0), -1)
        
        # Draw racket box
        if racket_box:
            x1, y1, x2, y2 = map(int, racket_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw racket center
            racket_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.circle(frame, racket_center, 5, (255, 0, 0), -1)
            
            # Draw line connecting racket and shuttle
            if shuttle_box and None not in shuttle_box:
                shuttle_center = ((shuttle_box[0] + shuttle_box[2]) // 2, 
                                 (shuttle_box[1] + shuttle_box[3]) // 2)
                cv2.line(frame, racket_center, shuttle_center, (0, 255, 255), 2)
        
        # Add shot information
        cv2.putText(frame, f"SHOT!", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Rally: {shot_data['rally_id']}, Shot: {shot_data['shot_id']}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add player who hit
        if shot_data['player_who_hit']:
            cv2.putText(frame, f"Player: {shot_data['player_who_hit']}", (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add shot type
        if shot_data['shot_type']:
            cv2.putText(frame, f"Type: {shot_data['shot_type']}", (10, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def save_shot_dataset(self):
        """
        Save the shot dataset to CSV and JSON files
        """
        # Convert to DataFrame for easy CSV export
        if self.direction_change_shots:
            df = pd.DataFrame(self.direction_change_shots)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"shot_dataset_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            # Save to JSON
            json_path = f"shot_dataset_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.direction_change_shots, f, indent=4)
            
            print(f"Shot dataset saved to {csv_path} and {json_path}")
            
            # Generate summary statistics
            self.generate_shot_statistics(df)
        else:
            print("No shots detected in the video")
    
    def generate_shot_statistics(self, shot_df):
        """
        Generate and print summary statistics from the shot dataset
        
        Args:
            shot_df: DataFrame containing shot data
        """
        print("\n=== SHOT STATISTICS ===")
        
        # Total shots
        total_shots = len(shot_df)
        print(f"Total Shots: {total_shots}")
        
        # Shots per rally
        rally_counts = shot_df.groupby('rally_id').size()
        avg_shots_per_rally = rally_counts.mean()
        print(f"Number of Rallies: {len(rally_counts)}")
        print(f"Average Shots per Rally: {avg_shots_per_rally:.2f}")
        print(f"Longest Rally: {rally_counts.max()} shots (Rally #{rally_counts.idxmax()})")
        
        # Breakdown by player
        if 'player_who_hit' in shot_df.columns:
            player_counts = shot_df['player_who_hit'].value_counts()
            print("\nShots by Player:")
            for player, count in player_counts.items():
                print(f"Player {player}: {count} shots ({count/total_shots*100:.1f}%)")
        
        # Breakdown by shot type
        if 'shot_type' in shot_df.columns:
            shot_type_counts = shot_df['shot_type'].value_counts()
            print("\nShots by Type:")
            for shot_type, count in shot_type_counts.items():
                print(f"{shot_type}: {count} shots ({count/total_shots*100:.1f}%)")
            
            # Shot types by player
            if 'player_who_hit' in shot_df.columns:
                print("\nShot Types by Player:")
                player_shot_types = shot_df.groupby(['player_who_hit', 'shot_type']).size().unstack(fill_value=0)
                print(player_shot_types)

if __name__ == "__main__":
    shuttle_model_path = "models/shuttle_player_racket/45epochs/best.pt"
    court_model_path = "models/court_detection/best.pt"
    video_path = "input/video.mov"
    
    analyzer = BadmintonAnalyzer(shuttle_model_path, court_model_path, video_path)
    shot_data = analyzer.analyze_video(save_output=True)
    
    print(f"\nDetected {len(shot_data)} shots in the video")