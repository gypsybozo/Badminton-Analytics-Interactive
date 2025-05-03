# shuttle tracker.py
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
from datetime import datetime
import json
from trackers.court import CourtDetector 
from utils.shot_detector import ShotDetector
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch


import os

def preprocess_image(image):
    crop_y_start = 100
    crop_y_end = image.shape[0] - 100
    cropped_image = image[crop_y_start:crop_y_end, :]

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)

    edges = cv2.Canny(enhanced_image, threshold1=50, threshold2=150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    return edges_bgr

class BadmintonAnalyzer:
    def __init__(self, shuttle_model_path, court_model_path, video_path, conf_threshold=0.3, frame_skip=1):
        """
        Initialize the BadmintonAnalyzer with model and video paths
        
        Args:
            shuttle_model_path (str): Path to trained YOLOv8 model weights for shuttle detection
            court_model_path (str): Path to trained YOLOv8 model weights for court detection
            video_path (str): Path to input video file
            conf_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(shuttle_model_path)
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.trajectory_data = []
        # Initialize court detector
        self.court_detector = CourtDetector(
            conf_threshold=conf_threshold,
            model_path=court_model_path
        )
        self.court_coords = None
        self.court_corners = None
        
        self.class_labels = {0: "Player", 1: "Racket", 2: "Shuttle"}
        self.prev_shuttle_pos = None
        self.max_shuttle_movement = 400  # Maximum pixel distance shuttle can move between frames
        
        self.shot_detector = ShotDetector()
        
        self.current_rally_id = 1
        self.current_shot_num = 0
        self.last_shot_frame = 0
        self.frames_between_rallies = 150
        self.shot_data = []
        self.shots_output_dir = "shot_images"
        self.prev_shuttle_pos = None
        self.max_shuttle_movement = 400  # Maximum pixel distance shuttle can move between frames
        self.min_shuttle_movement = 10   # Minimum movement to consider shuttle as "moving"
        
        # Create directory for shot images if it doesn't exist
        os.makedirs(self.shots_output_dir, exist_ok=True)
        
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
      
    def determine_player(self, player_position):
        """Determine if player is Player 1 (back) or 2 (front) based on y-coordinate"""
        if player_position[1] > 6.7:  # Middle of court is 13.4/2
            return 1
        return 2

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
    

    def process_video_with_direction_detection(self, save_output=True, output_path=None):
        """
        Process video with shuttle direction change detection
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
                output_path = f'direction_detection_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
        
        # Directory for saving direction change images
        direction_change_dir = "direction_change_images"
        os.makedirs(direction_change_dir, exist_ok=True)
        
        frame_count = 0
        court_detected = False
        all_frames = []
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
            
            # Detect shuttle
            results = self.model(frame, conf=self.conf_threshold)[0]
            shuttle_detections = []
            
            for detection in results.boxes.data:
                x1, y1, x2, y2, confidence, class_id = detection
                if int(class_id) == 2:  # Shuttle
                    shuttle_detections.append({
                        'box': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence)
                    })
            
            # Get best shuttle detection
            best_shuttle = self.select_best_shuttle_detection(shuttle_detections)
            all_shuttle_detections.append(best_shuttle)
            
            frame_count += 1
        
        cap.release()
        
        print(f"Collected {len(all_frames)} frames and shuttle detections")
        print("Phase 2: Interpolating shuttle positions...")
        
        # Interpolate shuttle positions
        interpolated_positions = self.interpolate_shuttle_positions(all_shuttle_detections)
        
        print("Phase 3: Detecting direction changes...")
        
        # Detect direction changes
        direction_changes = self.detect_direction_changes(
            interpolated_positions, 
            window_size=3,  # Smaller window = more sensitive
            min_frames_between_changes=2  # Allow changes to be detected closer together
        )
        
        print(f"Detected {len(direction_changes)} direction changes")
        print("Phase 4: Annotating and saving video...")
        
        # Reopen video for second pass
        cap = cv2.VideoCapture(self.video_path)
        
        # Second pass: annotate frames and save
        for i, (frame, box, frame_idx) in enumerate(zip(all_frames, interpolated_positions, processed_frame_indices)):
            # Draw court lines if detected
            if court_detected:
                self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
            
            # Draw interpolated shuttle box if available
            if box and None not in box:
                x1, y1, x2, y2 = map(int, box)
                # Draw box in green to indicate interpolated position
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label to indicate if this is a real detection or interpolated
                is_interpolated = all_shuttle_detections[i] is None
                label = "Shuttle (Interpolated)" if is_interpolated else "Shuttle"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if this frame is a direction change
            is_direction_change = i in direction_changes
            
            if is_direction_change:
                # Add text in top right corner
                text = "DIRECTION CHANGE"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = frame_width - text_size[0] - 10
                cv2.putText(frame, text, (text_x, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Save this frame as an image
                filename = f"direction_change_{frame_idx}.jpg"
                cv2.imwrite(os.path.join(direction_change_dir, filename), frame)
                
                # Annotate with rally and shot number
                shot_num = direction_changes.index(i) + 1
                
                cv2.putText(frame, f"Shot: {shot_num}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add frame number
            cv2.putText(frame, f'Frame: {frame_idx}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if save_output:
                out.write(frame)
                
            # Display the frame
            cv2.imshow('Badminton Direction Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        # Return info about direction changes
        return {
            'direction_changes': direction_changes,
            'frame_indices': [processed_frame_indices[i] for i in direction_changes],
            'total_frames': len(all_frames)
        }
    def get_player_positions(self, frame_detections):
        """Get player positions in real-world coordinates"""
        players = []
        
        for detection in frame_detections:
            if detection['label'] == 'Player':
                # Get center of player bounding box
                box = detection['box']
                center = (
                    (box[0] + box[2]) // 2,
                    (box[3])  # Use bottom of bounding box for feet position
                )
                
                # Convert to real-world coordinates
                real_world_pos = self.court_detector.translate_to_real_world(
                    center, 
                    self.homography_matrix
                )
                players.append(real_world_pos)
        
        # If we found 2 players, sort them by y-coordinate
        # The player with larger y-coordinate is at the back (Player 1)
        if len(players) == 2:
            players.sort(key=lambda pos: pos[1], reverse=True)  # Sort by y-coordinate in descending order
            return players[0], players[1]  # Return (player1_pos, player2_pos)
        
        return None, None

    
    def save_shot_image(self, frame, shot_info, shot_data):
        """Save an image of the shot with relevant information overlaid"""
        # Create a copy of the frame to draw on
        shot_frame = frame.copy()
        
        # Draw shot connection line
        racket_center = (
            (shot_info['racket_box'][0] + shot_info['racket_box'][2]) // 2,
            (shot_info['racket_box'][1] + shot_info['racket_box'][3]) // 2
        )
        shuttle_center = (
            (shot_info['shuttle_box'][0] + shot_info['shuttle_box'][2]) // 2,
            (shot_info['shuttle_box'][1] + shot_info['shuttle_box'][3]) // 2
        )
        cv2.line(shot_frame, racket_center, shuttle_center, (0, 255, 255), 2)
        
        # Add shot information overlay
        info_text = [
            f"Rally: {shot_data['rally_id']}",
            f"Shot: {shot_data['shot_num']}",
            f"Player: {shot_data['player_who_hit']}",
            f"Type: {shot_info['type']}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(shot_frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 25
            
        # Save the image
        filename = f"shot_r{shot_data['rally_id']}_s{shot_data['shot_num']}.jpg"
        cv2.imwrite(os.path.join(self.shots_output_dir, filename), shot_frame)
        
    
    
    def process_video(self, save_output=True, output_path=None):
        cap = cv2.VideoCapture(self.video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        effective_fps = original_fps // self.frame_skip
        
        # Calculate frames threshold based on video FPS
        # Default to 1.5 seconds without shots to consider it a new rally
        self.frames_between_rallies = int(5 * effective_fps)
        self.shot_data = []
        
        if save_output:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f'output_video_{timestamp}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, effective_fps, (frame_width, frame_height))
        
        frame_count = 0
        court_detected = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
                
            if not court_detected:
                self.court_coords = self.court_detector.detect_court_boundary(frame)
                if self.court_coords is not None:
                    self.court_coords = self.court_detector.sort_court_coords(self.court_coords)
                    self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
                    
                    # # Store the cropping region
                    # x, _, w, _ = cv2.boundingRect(np.array(self.court_corners, dtype=np.int32))
                    # self.crop_x, self.crop_w = x, w  # Store these values to avoid recalculating
                    
                    self.homography_matrix = self.court_detector.compute_homography(self.court_corners)
                    court_detected = True

            # Apply stored crop directly, avoiding recomputation
            if court_detected:
                self.court_corners = self.court_detector.draw_court_lines(frame, self.court_coords)
                # frame = frame[:, self.crop_x:self.crop_x + self.crop_w]  # Fast slicing
                
            # frame = preprocess_image(frame)
            # frame = cropped_frame
            results = self.model(frame, conf=self.conf_threshold)[0]
            # print(results)
            frame_detections = []
            shuttle_detections = []

            for detection in results.boxes.data:
                x1, y1, x2, y2, confidence, class_id = detection
                box = [int(x1), int(y1), int(x2), int(y2)]
                confidence = float(confidence)
                class_id = int(class_id)
                
                detection_dict = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / effective_fps,
                    'box': box,
                    'confidence': confidence,
                    'label': self.class_labels.get(class_id, "Unknown")
                }
                
                if class_id == 2:  # Shuttle
                    shuttle_detections.append(detection_dict)
                else:
                    frame_detections.append(detection_dict)
                    
                    # Draw non-shuttle detections
                    color = (0, 0, 0) if class_id == 0 else (255, 0, 0)  # Black for players, blue for rackets
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                    cv2.putText(frame, f'{self.class_labels.get(class_id)}: {confidence:.2f}', 
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Process shuttle detections
            best_shuttle = self.select_best_shuttle_detection(shuttle_detections)
            if best_shuttle:
                frame_detections.append(best_shuttle)
                box = best_shuttle['box']
                # Draw shuttle detection in white
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 3)
                cv2.putText(frame, f'Shuttle: {best_shuttle["confidence"]:.2f}', 
                        (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
              
            
            # Shot detection
            shot_detected, shot_info = self.shot_detector.detect_shot(frame_detections, frame_count)
            if shot_detected and shot_info:
                # Initialize variables
                player_who_hit = None
                player1_pos = None
                player2_pos = None
                distance_to_player1 = None
                distance_to_player2 = None
                frames_since_last_shot = frame_count - self.last_shot_frame
                
                # Check if this is a new rally based on frame gap
                frames_since_last_shot = frame_count - self.last_shot_frame
                
                if frames_since_last_shot > self.frames_between_rallies:
                    # Start new rally
                    self.current_rally_id += 1
                    self.current_shot_num = 0
                    print(f"\nNew Rally {self.current_rally_id} started at frame {frame_count}")
                
                # Get player positions
                player1_pos, player2_pos = self.get_player_positions(frame_detections)
                
                if player1_pos is not None and player2_pos is not None:
                    # Get racket position in real world coordinates
                    racket_center = (
                        (shot_info['racket_box'][0] + shot_info['racket_box'][2]) // 2,
                        (shot_info['racket_box'][1] + shot_info['racket_box'][3]) // 2
                    )
                    racket_real_world = self.court_detector.translate_to_real_world(
                        racket_center, 
                        self.homography_matrix
                    )
                    
                    # Calculate distances from racket to both players
                    distance_to_player1 = np.sqrt(
                        (racket_real_world[0] - player1_pos[0]) ** 2 + 
                        (racket_real_world[1] - player1_pos[1]) ** 2
                    )
                    distance_to_player2 = np.sqrt(
                        (racket_real_world[0] - player2_pos[0]) ** 2 + 
                        (racket_real_world[1] - player2_pos[1]) ** 2
                    )
                    
                    # Add distance threshold to favor back court player slightly
                    # This helps prevent misattribution when players are close together
                    BACK_COURT_THRESHOLD = 0.5  # Adjust this value based on your needs
                    if player1_pos[1] > player2_pos[1]:  # If player1 is in back court
                        distance_to_player1 *= (1 - BACK_COURT_THRESHOLD)  # Reduce distance by threshold
                    else:
                        distance_to_player2 *= (1 - BACK_COURT_THRESHOLD)
                    
                    # Determine which player is closest to the racket
                    player_who_hit = 1 if distance_to_player1 < distance_to_player2 else 2
                    
                    # For debugging
                    print(f"Frame {frame_count}:")
                    print(f"Racket position: {racket_real_world}")
                    print(f"Player 1 position: {player1_pos}, distance: {distance_to_player1}")
                    print(f"Player 2 position: {player2_pos}, distance: {distance_to_player2}")
                    print(f"Player who hit: {player_who_hit}\n")
                        
                # Increment shot number
                self.current_shot_num += 1
                # print(f"shotnum:{self.current_shot_num} player1dist: {distance_to_player1} player2dist: {distance_to_player2} playerwhohit: {player_who_hit}")
                
                # Create shot data entry
                shot_data_entry = {
                    'rally_id': self.current_rally_id,
                    'shot_num': self.current_shot_num,
                    'frame_number': frame_count,
                    'timestamp': frame_count / effective_fps,
                    'frames_since_last_shot': frames_since_last_shot,
                    'player_who_hit': player_who_hit,
                    'player1_position': player1_pos.tolist() if player1_pos is not None else None,
                    'player2_position': player2_pos.tolist() if player2_pos is not None else None,
                    'shot_type': shot_info['type'],
                    'confidence': shot_info['confidence']
                }
                
                # Add to shot data list
                self.shot_data.append(shot_data_entry)
                
                # Update last shot frame
                self.last_shot_frame = frame_count
                # Print rally information
                print(f"Rally {self.current_rally_id}, Shot {self.current_shot_num}")
                print(f"Frames since last shot: {frames_since_last_shot}")
                
                # Save shot image
                self.save_shot_image(frame, shot_info, shot_data_entry)
                
                # Draw shot detection visualization
                if shot_info:
                    # Draw connection line between racket and shuttle
                    racket_center = (
                        (shot_info['racket_box'][0] + shot_info['racket_box'][2]) // 2,
                        (shot_info['racket_box'][1] + shot_info['racket_box'][3]) // 2
                    )
                    shuttle_center = (
                        (shot_info['shuttle_box'][0] + shot_info['shuttle_box'][2]) // 2,
                        (shot_info['shuttle_box'][1] + shot_info['shuttle_box'][3]) // 2
                    )
                    # Draw yellow line connecting racket and shuttle
                    cv2.line(frame, racket_center, shuttle_center, (0, 255, 255), 2)
                    # Add "SHOT" text
                    cv2.putText(frame, 'SHOT!', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # Add rally and shot information to frame
                    cv2.putText(frame, f'Rally: {self.current_rally_id} Shot: {self.current_shot_num}', 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    
                # Check for new rally 
                if shot_detected and self.prev_shuttle_pos is None:
                    self.current_rally_id += 1
                    self.current_shot_num = 0
                
            # At the end of processing, convert shot data to DataFrame for easy analysis
            shot_df = pd.DataFrame(self.shot_data)
            
            
            self.trajectory_data.extend(frame_detections)

            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if save_output:
                out.write(frame)
            
            cv2.imshow('Badminton Analysis', frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_output:
            out.release()
        cv2.destroyAllWindows()
        
        # Save shot data to CSV
        csv_filename = f"shot_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shot_df.to_csv(csv_filename, index=False)
    
        return shot_df
        
        
if __name__ == "__main__":
    shuttle_model_path = "models/shuttle_player_racket/45epochs/best.pt"
    court_model_path = "models/court_detection/best.pt"
    video_path = "/Users/kriti.bharadwaj03/Badminton_Analysis/input/video.mov"
    
    analyzer = BadmintonAnalyzer(shuttle_model_path, court_model_path, video_path)
    # shot_data = analyzer.process_video(save_output=True)
    analyzer = BadmintonAnalyzer(shuttle_model_path, court_model_path, video_path)
    result = analyzer.process_video_with_direction_detection(save_output=True)
    
    print(f"Detected {len(result['direction_changes'])} direction changes")
    print(f"Direction changes at frames: {result['frame_indices']}")
    
    # # Print shot statistics
    # print(f"\nDetected {len(shot_data)} shots in the video")
    # print(shot_data)
    # for shot in shot_data:
        
        # print(f"Shot at frame {shot['frame_number']} ({shot['timestamp']:.2f}s) - Type: {shot['shot_info']['type']}")
    
